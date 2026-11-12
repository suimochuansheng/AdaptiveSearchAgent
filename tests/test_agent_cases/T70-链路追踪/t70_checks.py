#!/usr/bin/env python
"""T70 链路追踪（T7-0）—— 测试窗口 G6 独立验收脚本。

用途：
    只验收 T7-0 的 TraceSpan v2 链路追踪：A 类 7 条验收（A1~A7）、B 类 4 条不变量
    （B1~B4）、C 类 5 条边界 / 反例（C1~C5）。**不读开发窗口自测结论**，每条都从
    当前代码、临时目录里的真实 JSONL 与真命令输出独立取证；不改 `agent/`、`src/`、
    `api_main.py`、`config.py`。契约口径 = `docs/contracts/trace-span.md` v2。

用法（仓库根目录，任意 cwd 均可）：
    poetry run python "tests/test_agent_cases/T70-链路追踪/t70_checks.py"

判定与退出码：
    每项一行 `[PASS]/[FAIL] T70-xxx 名称 — 证据`；末行 `RESULT: n passed, m failed`
    与 `EXIT=<0|1>`。`0` = A/B 全过（C 类只记 open_issue，不判失败）；`1` = 至少一项
    A/B 失败。C 类发现额外打印 `OPEN_ISSUE:` 与 `C-FINDINGS: n open issue`。

为什么不是 `test_*.py`：
    会被 pytest 收集，`--collect-only` 从 227/231 变成 227+1，`docs/baseline.md`
    §2 的 nodeids 哈希立刻失效（见 tests/test_agent_cases/README.md）。

只读承诺：
    本脚本不写仓库内任何文件（JSONL 只写 `tempfile.TemporaryDirectory()`）；
    pytest 子进程带 `-p no:cacheprovider`；台账 `台账用例.csv` 由测试窗口在脚本全绿后
    另行追加（GB18030 / CRLF）。
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import agent / src / config
    sys.path.insert(0, str(REPO))

#: 卡级基线（pytest 通过数 / 收集数 / nodeids）唯一来源：`docs/baseline.md` §2。
CASE_ROOT = REPO / "tests" / "test_agent_cases"
if str(CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CASE_ROOT))

import _baseline  # noqa: E402

LEDGER = REPO / "tests" / "test_agent_cases" / "台账用例.csv"
#: 交接单记录的开工基线（T7-0 之前）
BASE_COMMIT = "355cabd"
#: T7-0 功能提交（开发窗口）
FEATURE_COMMIT = "0d880d2"
#: T7-0 契约 v2 追加提交（开发窗口）
CONTRACT_COMMIT = "d20f463"
#: 本卡验收提交（G6）：“零改动”只约束到本卡收口，后续卡的合法改动不算越界
ACCEPT_COMMIT = "d9fbe48"
SCHEMA_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
EXPECTED_IMPORTS = "3 kept, 0 broken"

TRACE_FIELDS = [
    "trace_id",
    "span_id",
    "parent_span_id",
    "session_id",
    "turn_index",
    "name",
    "kind",
    "start",
    "end",
    "attrs",
]
SPAN_KINDS = {"turn", "llm", "tool", "compress", "offload", "approval"}
SPAN_ATTRS = {"tool", "tokens_in", "tokens_out", "cost", "error_kind", "approval_id"}
TURN_ONLY_ATTRS = {"tools", "span_count"}
ERROR_KINDS = {"transient", "llm_recoverable", "user_fixable", "fatal"}

#: flag 关时 `/chat/conversation` 必须逐帧透传的假 SSE 帧
ENDPOINT_FRAMES = [
    'data: {"type":"token","content":"hello"}\n\n',
    'data: {"type":"ping"}\n\n',
    'data: {"type":"final","content":"done"}\n\n',
]

ISSUES: list[str] = []


# ── 基础设施 ─────────────────────────────────────────────────────────


class Checker:
    """每项一行 `[PASS]/[FAIL]`；异常信息即失败证据。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def run(self, case_id: str, title: str, fn: Any) -> None:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - 验收脚本要汇总所有失败项
            self.failed += 1
            print(f"[FAIL] {case_id} {title} — {type(exc).__name__}: {exc}")
        else:
            self.passed += 1
            print(f"[PASS] {case_id} {title} — {detail}")


def issue(message: str) -> None:
    """C 类发现：记 open_issue，不影响 A/B 判定。"""
    ISSUES.append(message)
    print(f"OPEN_ISSUE: {message}")


@contextmanager
def _settings(**changes: Any) -> Any:
    """临时改 settings 字段，退出时恢复（pydantic Settings 允许属性赋值）。"""
    from config import settings

    old = {key: getattr(settings, key) for key in changes}
    for key, value in changes.items():
        setattr(settings, key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(settings, key, value)


def _run(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(  # noqa: S603 - 参数固定、shell=False
        args,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        check=False,
    )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args])


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(  # noqa: S603 - 参数固定、shell=False
        ["git", *args],
        cwd=str(REPO),
        capture_output=True,
        check=True,
    ).stdout


def _last_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


# ── 假图 / 假工具（本地，不联网） ─────────────────────────────────────


from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402, TC002
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402


class ProbeModel(BaseChatModel):
    """假模型：不联网，回一条带 token 用量的 AIMessage。"""

    @property
    def _llm_type(self) -> str:
        return "t70-probe-model"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(
            content="probe-answer",
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
async def probe_tool(value: int = 1, explode: bool = False) -> str:
    """T70 假工具：explode=True 时抛 RuntimeError，用于验证错误 span。"""
    if explode:
        raise RuntimeError("probe-tool-boom")
    return f"probe:{value}"


class ProbeState(MessagesState):
    """最小图状态：消息历史 + 是否让假工具抛异常。"""

    explode: bool


async def _node_model(state: ProbeState, config: RunnableConfig) -> dict[str, Any]:
    response = await ProbeModel().ainvoke(state["messages"], config=config)
    return {"messages": [response]}


async def _node_tool(state: ProbeState, config: RunnableConfig) -> dict[str, Any]:
    try:
        result = await probe_tool.ainvoke(
            {"value": 1, "explode": state.get("explode", False)}, config=config
        )
    except Exception as exc:  # noqa: BLE001 - 模拟 Agent 把工具异常回灌给模型
        result = f"tool-error:{type(exc).__name__}"
    return {"messages": [AIMessage(content=str(result))]}


def build_graph(checkpointer: Any = None) -> Any:
    """两节点最小图：model -> tool。"""
    graph = StateGraph(ProbeState)
    graph.add_node("model", _node_model)
    graph.add_node("tool", _node_tool)
    graph.add_edge(START, "model")
    graph.add_edge("model", "tool")
    graph.add_edge("tool", END)
    return graph.compile(checkpointer=checkpointer)


async def _run_turn(
    graph: Any,
    tracer: Any,
    callbacks: list[Any],
    session_id: str,
    turn_index: int,
    *,
    explode: bool = False,
) -> None:
    """镜像 `/chat/conversation`：turn_span 包住整轮，callbacks 只挂图 config。"""
    from agent.observability.turn import turn_span

    async with turn_span(tracer, session_id, turn_index):
        await graph.ainvoke(
            {"messages": [HumanMessage(content="hi")], "explode": explode},
            config={
                "configurable": {"thread_id": session_id},
                "callbacks": callbacks,
            },
        )


def _read_records(directory: Path) -> list[dict[str, Any]]:
    """读目录下所有 JSONL 行（逐行 json.loads）。"""
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _parse_sse(text: str) -> list[dict[str, Any]]:
    """按空行切帧、逐帧 json.loads（与 tdm_checks 的 SSE 口径一致）。"""
    frames: list[dict[str, Any]] = []
    for chunk in text.split("\n\n"):
        line = chunk.strip()
        if not line:
            continue
        assert line.startswith("data: "), f"帧不是 data: 前缀: {line[:40]!r}"
        frames.append(json.loads(line[len("data: ") :]))
    return frames


class _FakeLock:
    async def acquire(self, blocking: bool = False) -> bool:
        return True

    async def release(self) -> None:
        return None


class _FakeRedis:
    def lock(self, name: str, timeout: int = 600) -> _FakeLock:
        return _FakeLock()

    async def exists(self, key: str) -> bool:
        return True

    async def delete(self, key: str) -> int:
        return 0


def _conversation_endpoint() -> tuple[str, dict[str, Any], int]:
    """走真实 `/chat/conversation` 端点，打桩 Redis / 图 / 流；返回 (body, config, code)。"""
    from fastapi.testclient import TestClient

    import api_main
    import src.conversation_stream as conversation_stream

    captured: dict[str, Any] = {}

    async def _get_state(thread_id: str) -> str:
        return "idle"

    async def _set_state(thread_id: str, state: str, interrupt_time: Any = None) -> None:
        return None

    async def _graph() -> Any:
        return object()

    async def _stream(
        graph: Any,
        state: dict[str, Any],
        config: dict[str, Any],
        is_disconnected: Any = None,
    ) -> Any:
        captured["config"] = dict(config)
        for frame in ENDPOINT_FRAMES:
            yield frame

    originals = {
        name: getattr(api_main, name)
        for name in ("redis_client", "get_task_state", "set_task_state", "get_conversation_graph")
    }
    original_stream = conversation_stream.stream_conversation
    api_main.redis_client = _FakeRedis()  # type: ignore[assignment]
    api_main.get_task_state = _get_state
    api_main.set_task_state = _set_state
    api_main.get_conversation_graph = _graph
    conversation_stream.stream_conversation = _stream  # type: ignore[assignment]
    try:
        client = TestClient(api_main.app)
        response = client.post(
            "/chat/conversation", json={"thread_id": "t70-endpoint", "message": "hi"}
        )
    finally:
        for name, value in originals.items():
            setattr(api_main, name, value)
        conversation_stream.stream_conversation = original_stream
    return response.text, captured.get("config", {}), response.status_code


# ── A 类：7 条验收 ──────────────────────────────────────────────────


def check_001() -> str:
    """A1：flag 关 -> TRACE_DIR 无文件，`/chat/conversation` config 无 callbacks。"""
    with tempfile.TemporaryDirectory() as td, _settings(ENABLE_TRACE=False, TRACE_DIR=td):
        body, config, status = _conversation_endpoint()
        leftovers = [str(path) for path in Path(td).rglob("*")]
    assert status == 200, f"端点状态码 {status}"
    assert "callbacks" not in config, f"flag 关却设置了 callbacks: {config.keys()}"
    assert not leftovers, f"flag 关仍产生文件: {leftovers}"
    assert body, "SSE body 为空"
    return "flag=False：SSE 200、config=calls 0、TRACE_DIR 0 文件"


def check_002() -> str:
    """A2：flag 开 -> 有 turn span，tool/llm 的 parent 指向该 turn 的 span_id。"""
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    with tempfile.TemporaryDirectory() as td:
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        graph = build_graph()
        asyncio.run(_run_turn(graph, tracer, callbacks, "t70-a2", 1))
        records = _read_records(Path(td))

    turns = [r for r in records if r["kind"] == "turn"]
    tools = [r for r in records if r["kind"] == "tool"]
    llms = [r for r in records if r["kind"] == "llm"]
    assert len(turns) == 1, f"turn span 数量 {len(turns)}"
    assert tools and llms, f"缺 tool/llm span: kinds={[r['kind'] for r in records]}"
    turn_id = turns[0]["span_id"]
    assert turns[0]["parent_span_id"] is None, "turn.parent_span_id 不是 null"
    bad = [r["name"] for r in tools + llms if r["parent_span_id"] != turn_id]
    assert not bad, f"父 id 未指向 turn: {bad}"
    return f"turn={turn_id[:8]}；tool/llm 各 {len(tools)}/{len(llms)} 条 parent 全部指向 turn"


def check_003() -> str:
    """A3：v2 字段合规（十字段 / 六 kind / attrs 按 kind 分支 / error_kind 值域）。"""
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    with tempfile.TemporaryDirectory() as td:
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        asyncio.run(_run_turn(build_graph(), tracer, callbacks, "t70-a3", 1))
        asyncio.run(_run_turn(build_graph(), tracer, callbacks, "t70-a3", 2, explode=True))
        records = _read_records(Path(td))

    span_ids = {r["span_id"] for r in records}
    for rec in records:
        assert list(rec) == TRACE_FIELDS, f"字段不是十字段/顺序不对: {list(rec)}"
        assert rec["kind"] in SPAN_KINDS, f"非法 kind: {rec['kind']}"
        assert "status" not in rec, "落盘出现 status 字段"
        assert "status" not in rec["attrs"], "attrs 出现 status 键"
        allowed = SPAN_ATTRS | TURN_ONLY_ATTRS if rec["kind"] == "turn" else SPAN_ATTRS
        extra = set(rec["attrs"]) - allowed
        assert not extra, f"{rec['kind']} span attrs 越界: {extra}"
        if rec["kind"] == "turn":
            tools = rec["attrs"]["tools"]
            count = rec["attrs"]["span_count"]
            assert isinstance(tools, list) and all(isinstance(item, str) for item in tools), (
                f"turn.tools 不是 string[]: {tools!r}"
            )
            assert isinstance(count, int) and not isinstance(count, bool) and count >= 0, (
                f"turn.span_count 不是非负 int: {count!r}"
            )
            assert rec["parent_span_id"] is None, "turn.parent_span_id 不是 null"
        else:
            assert rec["parent_span_id"] in span_ids, (
                f"{rec['kind']} span 父 id 找不到: {rec['parent_span_id']}"
            )
        error_kind = rec["attrs"].get("error_kind")
        assert error_kind is None or error_kind in ERROR_KINDS, f"error_kind 越界: {error_kind!r}"
        start = datetime.fromisoformat(rec["start"])
        end = datetime.fromisoformat(rec["end"])
        assert start <= end, f"start > end: {rec['start']} / {rec['end']}"
    turn_spans = [r for r in records if r["kind"] == "turn"]
    return (
        f"{len(records)} 行：十字段/六 kind/attrs 分支 0 越界；"
        f"turn {len(turn_spans)} 条含 tools/span_count；error_kind 值域 0 越界"
    )


def check_004() -> str:
    """A4：一轮汇总 —— turn.attrs 含 tools/tokens_in/tokens_out/cost/span_count。"""
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    with tempfile.TemporaryDirectory() as td:
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        asyncio.run(_run_turn(build_graph(), tracer, callbacks, "t70-a4", 1))
        records = _read_records(Path(td))

    turn = next(r for r in records if r["kind"] == "turn")
    attrs = turn["attrs"]
    required = {"tools", "tokens_in", "tokens_out", "cost", "span_count"}
    missing = required - set(attrs)
    assert not missing, f"turn 汇总缺键: {missing}"
    children = [r for r in records if r["parent_span_id"] == turn["span_id"]]
    assert attrs["tools"] == [r["name"] for r in children if r["kind"] == "tool"], (
        f"tools 与子 span 不一致: {attrs['tools']}"
    )
    assert attrs["tokens_in"] == sum(int(r["attrs"].get("tokens_in", 0)) for r in children), (
        f"tokens_in 不等于子 span 求和: {attrs['tokens_in']}"
    )
    assert attrs["tokens_out"] == sum(int(r["attrs"].get("tokens_out", 0)) for r in children), (
        f"tokens_out 不等于子 span 求和: {attrs['tokens_out']}"
    )
    assert isinstance(attrs["cost"], int | float), f"cost 不是数: {attrs['cost']!r}"
    assert attrs["span_count"] == len(records), (
        f"span_count={attrs['span_count']} 与落盘行数 {len(records)} 不一致"
    )
    return (
        f"turn.attrs={sorted(attrs)}；tools={attrs['tools']}；tokens={attrs['tokens_in']}/"
        f"{attrs['tokens_out']}；cost={attrs['cost']}；span_count={attrs['span_count']}"
    )


def check_005() -> str:
    """A5：错误路径 —— 假工具抛异常 -> tool span 存在且 error_kind 非空。"""
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    with tempfile.TemporaryDirectory() as td:
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        asyncio.run(_run_turn(build_graph(), tracer, callbacks, "t70-a5", 1, explode=True))
        records = _read_records(Path(td))

    tools = [r for r in records if r["kind"] == "tool"]
    assert tools, "错误路径没有 tool span"
    error_kind = tools[0]["attrs"].get("error_kind")
    assert error_kind, f"tool span error_kind 为空: attrs={tools[0]['attrs']}"
    assert error_kind in ERROR_KINDS, f"error_kind 不在四类: {error_kind!r}"
    assert any(r["kind"] == "turn" for r in records), "错误路径 turn span 未落盘"
    return f"tool span error_kind={error_kind}；turn span 正常落盘，会话未中断"


def check_006() -> str:
    """A6：JSONL 按天分文件；20 条并发写后逐行 json.loads 全部成功。"""
    from agent.observability.tracer import JsonlTracer, bind_turn

    with tempfile.TemporaryDirectory() as td:
        tracer = JsonlTracer(td)
        with bind_turn(tracer, session_id="t70-a6", turn_index=1):
            root = tracer.start_span("turn", "turn")
            handles = [
                tracer.start_span(f"worker-{index}", "tool", attrs={"tool": f"worker-{index}"})
                for index in range(20)
            ]
            barrier = threading.Barrier(len(handles))

            def write(handle: Any) -> None:
                barrier.wait(timeout=15)
                tracer.end_span(handle)

            with ThreadPoolExecutor(max_workers=len(handles)) as pool:
                list(pool.map(write, handles))
            tracer.end_span(root)
        files = sorted(Path(td).glob("*"))
        names = [path.name for path in files]
        lines = files[0].read_text(encoding="utf-8").splitlines() if files else []

    expected = f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    assert names == [expected], f"不是按天单文件: {names}"
    parsed = [json.loads(line) for line in lines if line.strip()]
    ids = {record["span_id"] for record in parsed}
    assert len(lines) == 21, f"应有 21 行（turn+20），实际 {len(lines)}"
    assert len(ids) == 21, f"span_id 重复/缺失: {len(ids)} 个唯一值"
    return f"文件 {expected}；21 行逐行 json.loads 通过，21 个唯一 span_id（20 并发）"


def check_007() -> str:
    """A7：turn_index 首轮 = 1，第二轮 = 2。"""
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer
    from agent.observability.turn import next_turn_index

    with tempfile.TemporaryDirectory() as td:
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        graph = build_graph(MemorySaver())
        config = {"configurable": {"thread_id": "t70-a7"}}

        async def two_turns() -> list[int]:
            indexes: list[int] = []
            for _ in range(2):
                index = await next_turn_index(graph, config)
                indexes.append(index)
                await _run_turn(graph, tracer, callbacks, "t70-a7", index)
            return indexes

        indexes = asyncio.run(two_turns())
        records = _read_records(Path(td))

    turn_indexes = [r["turn_index"] for r in records if r["kind"] == "turn"]
    assert indexes == [1, 2], f"next_turn_index 两轮 = {indexes}"
    assert turn_indexes == [1, 2], f"turn span 的 turn_index = {turn_indexes}"
    return "next_turn_index=1,2；turn span turn_index=1,2"


# ── B 类：4 条不变量 ────────────────────────────────────────────────


def check_008() -> str:
    """B1：`src/conversation_stream.py`、`src/conversation_graph.py` 零改动。"""
    paths = ["src/conversation_stream.py", "src/conversation_graph.py"]
    diff = _git("diff", "--quiet", BASE_COMMIT, ACCEPT_COMMIT, "--", *paths)
    assert diff.returncode == 0, f"{BASE_COMMIT}..{ACCEPT_COMMIT} 有改动: {diff.stdout.strip()}"
    status = _git("status", "--porcelain", "--", *paths).stdout.strip()
    assert not status, f"工作区有未提交改动: {status}"
    return f"{BASE_COMMIT}..{ACCEPT_COMMIT}：两文件 0 行差异；git status 空"


def check_009() -> str:
    """B2：flag 关时 SSE 帧序列与原始流逐帧一致（端到端透传）。"""
    with _settings(ENABLE_TRACE=False):
        body, _, status = _conversation_endpoint()
    expected = "".join(ENDPOINT_FRAMES)
    assert status == 200, f"端点状态码 {status}"
    assert body == expected, f"SSE 原文不一致: {body!r}"
    got_frames = _parse_sse(body)
    want_frames = _parse_sse(expected)
    assert got_frames == want_frames, f"逐帧不一致: got={got_frames} want={want_frames}"
    assert len(got_frames) == len(ENDPOINT_FRAMES) == 3, f"帧数 {len(got_frames)}"
    return "3 帧原文逐字节一致；解析后 token/ping/final 帧序一致"


def check_010() -> str:
    """B3：pytest / collect 与 `docs/baseline.md` §2 一致 / lint-imports 3 kept / schema 哈希不变。"""
    exp = _baseline.expected()
    pytest_out = _run(["poetry", "run", "pytest", "-q", "-p", "no:cacheprovider"])
    assert exp["pytest"] in pytest_out.stdout, (
        f"pytest: {_last_line(pytest_out.stdout + pytest_out.stderr)}"
    )
    collect_out = _run(
        ["poetry", "run", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
    )
    assert f"{exp['collect']} tests collected" in collect_out.stdout, (
        f"collect: {_last_line(collect_out.stdout + collect_out.stderr)}"
    )
    lint_out = _run(["poetry", "run", "lint-imports"])
    assert EXPECTED_IMPORTS in lint_out.stdout, (
        f"lint-imports: {_last_line(lint_out.stdout + lint_out.stderr)}"
    )
    schema_out = _run(["poetry", "run", "python", "scripts/smoke_tools_schema.py"])
    assert SCHEMA_SHA256 in schema_out.stdout, "schema sha256 与基线不一致"
    assert "2558" in schema_out.stdout, "schema bytes 不是 2558"
    return (
        f"pytest {exp['pytest']}；collect {exp['collect']}；lint-imports 3 kept；"
        "schema c48c9d48...2a99 / 2558B"
    )


def check_011() -> str:
    """B4：台账 T70 行只由测试窗口写；开发提交（0d880d2/d20f463）0 条。"""
    rows = list(csv.reader(io.StringIO(LEDGER.read_bytes().decode("gb18030"))))
    t70 = [row for row in rows[1:] if row[0].startswith("T70-")]
    bad = [row[0] for row in t70 if row[12] != "test-G6" or row[11] != "2026/11/11"]
    assert not bad, f"T70 行 run_by/last_run 不是测试窗口: {bad}"
    for commit in (FEATURE_COMMIT, CONTRACT_COMMIT):
        baseline = _git_bytes("show", f"{commit}:tests/test_agent_cases/台账用例.csv")
        assert "T70-" not in baseline.decode("gb18030"), f"开发提交 {commit} 已写 T70 台账行"
    return f"当前 T70 {len(t70)} 行均 run_by=test-G6 / 2026/11/11；开发两提交 0 条"


# ── C 类：5 条边界 / 反例（只记 open_issue） ─────────────────────────


def check_012() -> str:
    """C1：flag 关 + 工具抛异常 -> 无 callbacks 时异常原样冒泡。"""
    with _settings(ENABLE_TRACE=False):
        try:
            asyncio.run(probe_tool.ainvoke({"value": 1, "explode": True}, config={}))
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("flag 关时异常被吞掉了")
    assert message == "probe-tool-boom", f"异常内容被改写: {message!r}"
    return "flag 关无 callbacks：RuntimeError('probe-tool-boom') 原样冒泡"


def check_013() -> str:
    """C2：`aget_state` 抛异常 -> turn_index 回落 0 且会话不中断。"""
    from agent.observability.tracer import JsonlTracer
    from agent.observability.turn import next_turn_index, turn_span

    class _BadGraph:
        async def aget_state(self, config: Any) -> Any:
            raise RuntimeError("state unavailable")

    with tempfile.TemporaryDirectory() as td:
        tracer = JsonlTracer(td)

        async def run() -> int:
            index = await next_turn_index(_BadGraph(), {"configurable": {"thread_id": "t70-c2"}})
            async with turn_span(tracer, "t70-c2", index):
                pass
            return index

        index = asyncio.run(run())
        records = _read_records(Path(td))

    assert index == 0, f"turn_index 未回落 0: {index}"
    turns = [r for r in records if r["kind"] == "turn"]
    assert len(turns) == 1 and turns[0]["turn_index"] == 0, f"兜底 turn span 异常: {turns}"
    return "aget_state 抛错 -> next_turn_index=0；turn span 落盘 turn_index=0，会话未中断"


def check_014() -> str:
    """C3+C4：TRACE_DIR 不可写 / Langfuse 缺密钥 -> 均只 warning 一次并降级。"""

    def capture(logger_name: str) -> tuple[list[str], logging.Handler]:
        messages: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                messages.append(record.getMessage())

        handler = _Capture()
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        return messages, handler

    import agent.observability.langchain_handler as lch
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer, bind_turn
    from config import settings

    trace_messages, trace_handler = capture("agent.observability.tracer")
    lch_messages, lch_handler = capture("agent.observability.langchain_handler")

    try:
        with tempfile.TemporaryDirectory() as td:
            blocker = Path(td) / "blocked"
            blocker.write_text("x", encoding="utf-8")
            tracer = JsonlTracer(blocker / "trace")
            with bind_turn(tracer, session_id="t70-c3", turn_index=1):
                root = tracer.start_span("turn", "turn")
                tracer.end_span(root)
                child = tracer.start_span("llm", "llm")
                tracer.end_span(child)
            assert tracer._broken, "写盘失败后没有降级为 Noop"
            write_warnings = [m for m in trace_messages if "链路追踪写入失败" in m]
            assert len(write_warnings) == 1, f"写盘失败 warning 次数 {len(write_warnings)}"

        old_enabled = settings.LANGFUSE_ENABLED
        old_warned = lch._warned_no_langfuse
        try:
            settings.LANGFUSE_ENABLED = True
            lch._warned_no_langfuse = False
            lch_messages.clear()
            first = build_callbacks(tracer, None)
            second = build_callbacks(tracer, None)
        finally:
            settings.LANGFUSE_ENABLED = old_enabled
            lch._warned_no_langfuse = old_warned
        assert len(first) == 1 and len(second) == 1, (
            f"handlers 未降级为 1: {len(first)}/{len(second)}"
        )
        langfuse_warnings = [m for m in lch_messages if "Langfuse" in m and "降级" in m]
        assert len(langfuse_warnings) == 1, f"Langfuse 缺密钥 warning 次数 {len(langfuse_warnings)}"
    finally:
        logging.getLogger("agent.observability.tracer").removeHandler(trace_handler)
        logging.getLogger("agent.observability.langchain_handler").removeHandler(lch_handler)
    return "C3 写盘失败 warning=1 且 tracer 降级；C4 handlers=1/1 且 warning=1"


def check_015() -> str:
    """C5：统计每个 turn span 的子 span 数；0 子 span 的 turn 记 open_issue。"""
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    with tempfile.TemporaryDirectory() as td:
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        graph = build_graph(MemorySaver())

        async def two_turns() -> None:
            for index in (1, 2):
                await _run_turn(graph, tracer, callbacks, "t70-c5", index)

        asyncio.run(two_turns())
        records = _read_records(Path(td))

    turns = [r for r in records if r["kind"] == "turn"]
    counts = {
        turn["turn_index"]: sum(1 for r in records if r["parent_span_id"] == turn["span_id"])
        for turn in turns
    }
    empty = [index for index, count in counts.items() if count == 0]
    if empty:
        issue(f"C5 存在 0 子 span 的 turn：轮次 {empty}（turn span 数 {len(turns)}）")
    return f"turn 子 span 数 {counts}；0 子 span 的 turn {len(empty)} 个"


# ── 入口 ─────────────────────────────────────────────────────────────


def main() -> int:
    checker = Checker()
    checker.run("T70-001", "A1 flag 关：无 trace 文件且 config 无 callbacks", check_001)
    checker.run("T70-002", "A2 flag 开：tool/llm 父 id 指向 turn", check_002)
    checker.run("T70-003", "A3 字段合规（v2：非 turn 六键 / turn 可多两键）", check_003)
    checker.run("T70-004", "A4 一轮汇总 tools/tokens/cost/span_count", check_004)
    checker.run("T70-005", "A5 工具异常 -> tool span error_kind 非空", check_005)
    checker.run("T70-006", "A6 JSONL 按天分文件 + 20 并发可解析", check_006)
    checker.run("T70-007", "A7 turn_index 首轮 1、次轮 2", check_007)
    checker.run("T70-008", "B1 conversation_stream/graph 零改动", check_008)
    checker.run("T70-009", "B2 flag 关 SSE 逐帧一致", check_009)
    checker.run("T70-010", "B3 pytest/collect/lint-imports/schema 基线", check_010)
    checker.run("T70-011", "B4 台账 T70 行均出自 test-G6", check_011)
    checker.run("T70-012", "C1 flag 关 + 工具异常原样冒泡", check_012)
    checker.run("T70-013", "C2 aget_state 异常 -> turn_index=0 不中断", check_013)
    checker.run("T70-014", "C3+C4 写盘失败/Langfuse 缺密钥降级一次", check_014)
    checker.run("T70-015", "C5 统计 turn 子 span 数（0 子记 open_issue）", check_015)

    if ISSUES:
        print(f"C-FINDINGS: {len(ISSUES)} open issue")
    print(f"RESULT: {checker.passed} passed, {checker.failed} failed")
    code = 0 if checker.failed == 0 else 1
    print(f"EXIT={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
