#!/usr/bin/env python
"""INT 跨卡集成（I2 治理闭环）—— 注册表 / 权限预算 / 追踪三条链叠加验收。

用途：
    验证 T6-0 工具注册表、T6-3 权限与预算、T7-0 链路追踪三条链**叠在一起**仍成立，
    而不是各自的单卡验收。覆盖四组开关矩阵（M1~M4）、三条对账（L1~L3）、两处已知
    缺口取证（G1~G2）。

用法（仓库根目录，任意 cwd 均可）：
    poetry run python "tests/test_agent_cases/INT-跨卡集成/int_checks.py"

判定与退出码：
    每项一行 `[PASS]/[FAIL] INT2-xxx 名称 — 证据`；末行 `RESULT: n passed, m failed`
    与 `EXIT=<0|1>`。`0` = M/L 全过（G 类只取证，不判失败）；`1` = 至少一项 M/L 失败。

为什么不是 `test_*.py`：
    会被 pytest 收集，`--collect-only` 从 227/231 变成 227+1，`docs/baseline.md`
    §2 的 nodeids 哈希立刻失效（见 tests/test_agent_cases/README.md）。

只读承诺：
    本脚本不写仓库内任何文件（JSONL 只写 `tempfile.TemporaryDirectory()`）；
    假图与假工具全在进程内，不联网、不碰 Redis（账本用内存实现注入）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import agent / src / config
    sys.path.insert(0, str(REPO))

#: T6 基线：5 个内置工具的 schema 哈希（flag 关闭时必须逐字节不变）
BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
#: 基线工具顺序（与 registry.TOOL_SPECS / conversation_graph.TOOLS 一致）
EXPECTED_ORDER = [
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
]
#: 假工具每次调用的计价（元）：用于验证账本 calls / cost 与 turn span cost 对账
PROBE_UNIT_COST = 0.5

#: 假工具被真正执行的计数（deny 时必须保持 0）
CALLS: dict[str, Any] = {"n": 0, "values": []}
#: 假图每轮请求的工具调用次数（G1 口调用）
REQUESTS: dict[str, int] = {"n": 0}

ISSUES: list[str] = []


# ── 基础设施 ─────────────────────────────────────────────────────────


class Checker:
    """每项一行 `[PASS]/[FAIL]`；异常信息即失败证据。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def run(self, case_id: str, title: str, fn: Any, *, g_class: bool = False) -> None:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - 验收脚本要汇总所有失败项
            self.failed += 1
            print(f"[FAIL] {case_id} {title} — {type(exc).__name__}: {exc}")
        else:
            self.passed += 1
            print(f"[PASS] {case_id} {title} — {detail}")


def _last_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


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


def _sha_tools(tools: list[Any]) -> str:
    """按 T63 口径算工具 schema 哈希（sort_keys + UTF-8）。"""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schemas = [convert_to_openai_tool(tool) for tool in tools]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def _read_records(directory: Path) -> list[dict[str, Any]]:
    """读目录下所有 JSONL 行（逐行 json.loads）。"""
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


# ── 假图 / 假工具（本地，不联网、不碰 Redis） ────────────────────────

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402, TC002
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402


async def probe_tool(value: int = 1) -> str:
    """I2 集成假工具：记录真实执行次数，返回 probe:<value>。"""
    CALLS["n"] += 1
    CALLS["values"].append(value)
    return f"probe:{value}"


class ProbeState(MessagesState):
    """最小图状态：消息历史（与 src.conversation_graph 的循环形状一致）。"""


async def _node_model(state: ProbeState, config: RunnableConfig) -> dict[str, Any]:
    """假模型：有 ToolMessage 就收尾，否则发起一次工具调用。"""
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content="probe-done")]}
    REQUESTS["n"] += 1
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "probe_tool", "args": {"value": 1}, "id": "call-1"}],
            )
        ]
    }


def _probe_spec() -> Any:
    from agent.domain.tool import CostModel, ToolSpec

    return ToolSpec(
        name="probe_tool",
        description="I2 集成假工具。",
        input_schema={
            "properties": {"value": {"default": 1, "type": "integer"}},
            "type": "object",
        },
        permission="read",
        cost_model=CostModel(unit="call", unit_cost=PROBE_UNIT_COST),
        timeout_seconds=5.0,
        max_output_tokens=100,
        requires_approval=False,
        feature_flag=None,
    )


def _build_graph(tool_fn: Any) -> Any:
    """model -> tools -> model 的最小图；tools 节点用 tool_scope 绑定会话作用域。"""
    from agent.adapters.tools import executor

    tool_node = ToolNode([tool_fn])

    async def _node_tools(state: ProbeState, config: RunnableConfig) -> dict[str, Any]:
        thread_id = str((config.get("configurable") or {}).get("thread_id", "default"))
        with executor.tool_scope(thread_id):
            result = await tool_node.ainvoke(state)
        return {"messages": result["messages"]}

    graph = StateGraph(ProbeState)
    graph.add_node("model", _node_model)
    graph.add_node("tools", _node_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model", lambda state: "tools" if state["messages"][-1].tool_calls else END
    )
    graph.add_edge("tools", "model")
    return graph.compile()


async def _run_graph_only(graph: Any, session_id: str) -> None:
    """不开追踪的一轮：镜像 ENABLE_TRACE=False 时的调用路径（config 无 callbacks）。"""
    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": session_id}},
    )


async def _run_traced_turn(
    graph: Any, tracer: Any, callbacks: list[Any], session_id: str, turn_index: int = 1
) -> None:
    """镜像 `/chat/conversation`：turn_span 包住整轮，callbacks 只挂图 config。"""
    from agent.observability.turn import turn_span

    async with turn_span(tracer, session_id, turn_index):
        await graph.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={
                "configurable": {"thread_id": session_id},
                "callbacks": callbacks,
            },
        )


# ── M1：两个开关都关（基线） ─────────────────────────────────────────


def m1_001() -> str:
    """M1-1：两个开关都关，工具 schema 哈希仍是基线。"""
    from agent.adapters.tools import registry

    with _settings(ENABLE_TOOL_POLICY=False, ENABLE_TRACE=False):
        sha = _sha_tools(registry.to_lc_tools())
    assert sha == BASELINE_SHA256, f"schema 哈希漂移：{sha}"
    return f"sha256={sha[:8]}..{sha[-4:]}（与基线一致）"


def m1_002() -> str:
    """M1-2：注册表返回的 5 个工具与 conversation_graph.TOOLS 逐项 is 同一对象。"""
    from agent.adapters.tools import registry

    with _settings(ENABLE_TOOL_POLICY=False, ENABLE_TRACE=False):
        from src import conversation_graph

        tools = registry.to_lc_tools()
        graph_tools = list(conversation_graph.TOOLS)

    assert len(tools) == len(graph_tools) == 5, f"数量不一致：{len(tools)} / {len(graph_tools)}"
    names = [tool.__name__ for tool in tools]
    assert names == EXPECTED_ORDER, f"顺序漂移：{names}"
    bad = [
        name
        for name, tool, graph_tool in zip(names, tools, graph_tools, strict=True)
        if tool is not graph_tool
    ]
    assert not bad, f"不是同一对象：{bad}"
    return f"5/5 identity True；顺序={names}"


def m1_003() -> str:
    """M1-3/M1-4：追踪目录零文件；会话端点 config 里没有 callbacks。"""
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

    async def _stream(graph: Any, state: dict[str, Any], config: dict[str, Any], **_: Any) -> Any:
        captured["config"] = dict(config)
        yield 'data: {"type": "final", "message": "i2"}\n\n'

    originals = {
        name: getattr(api_main, name)
        for name in ("redis_client", "get_task_state", "set_task_state", "get_conversation_graph")
    }
    original_stream = conversation_stream.stream_conversation
    api_main.redis_client = _FakeRedis()  # type: ignore[assignment]
    api_main.get_task_state = _get_state  # type: ignore[assignment]
    api_main.set_task_state = _set_state  # type: ignore[assignment]
    api_main.get_conversation_graph = _graph  # type: ignore[assignment]
    conversation_stream.stream_conversation = _stream  # type: ignore[assignment]
    with (
        tempfile.TemporaryDirectory() as td,
        _settings(ENABLE_TOOL_POLICY=False, ENABLE_TRACE=False, TRACE_DIR=td),
    ):
        try:
            client = TestClient(api_main.app)
            response = client.post(
                "/chat/conversation", json={"thread_id": "i2-m1", "message": "hi"}
            )
        finally:
            for name, value in originals.items():
                setattr(api_main, name, value)
            conversation_stream.stream_conversation = original_stream
        leftovers = [str(path) for path in Path(td).rglob("*")]
    assert response.status_code == 200, f"端点状态码 {response.status_code}"
    assert "callbacks" not in captured.get("config", {}), (
        f"flag 关却挂了 callbacks：{captured.get('config', {}).keys()}"
    )
    assert not leftovers, f"flag 关仍产生追踪文件：{leftovers}"
    return "端点 200；config 无 callbacks；TRACE_DIR 0 文件"


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


# ── M2：只开权限预算 ────────────────────────────────────────────────


def m2_001() -> str:
    """M2-1：被拒绝时工具函数本身调用计数为 0，账本也不记。"""
    from agent.adapters.tools.budget_memory import MemoryBudgetLedger
    from agent.adapters.tools.executor import set_default_ledger, tool_scope, wrap_tool

    ledger = MemoryBudgetLedger()
    CALLS["n"], CALLS["values"] = 0, []
    with _settings(ENABLE_TOOL_POLICY=True, ENABLE_TRACE=False, TOOL_MAX_CALLS_PER_TURN=0):
        wrapped = wrap_tool(probe_tool, _probe_spec(), ledger)

        async def scenario() -> str:
            with tool_scope("i2-m2-deny"):
                return await wrapped(value=1)

        result = asyncio.run(scenario())
    set_default_ledger(None)
    snapshot = ledger.snapshot("i2-m2-deny")
    assert CALLS["n"] == 0, f"被拒绝仍执行了工具：{CALLS['n']} 次"
    assert snapshot.calls == 0 and snapshot.cost == 0.0, f"deny 却记账：{snapshot}"
    assert "拒绝" in result, f"没有可读拒绝理由：{result!r}"
    return f"拒绝文案={result!r}；工具执行 0 次；账本 calls=0 cost=0"


def m2_002() -> str:
    """M2-2/M2-3：正常调用后账本 calls/cost 正确；追踪开关关时追踪目录零文件。"""
    from agent.adapters.tools import executor
    from agent.adapters.tools.budget_memory import MemoryBudgetLedger

    ledger = MemoryBudgetLedger()
    executor.set_default_ledger(ledger)
    CALLS["n"], CALLS["values"], REQUESTS["n"] = 0, [], 0
    with (
        tempfile.TemporaryDirectory() as td,
        _settings(
            ENABLE_TOOL_POLICY=True,
            ENABLE_TRACE=False,
            TRACE_DIR=td,
            TOOL_MAX_CALLS_PER_TURN=8,
            TOOL_SESSION_COST_LIMIT=2.0,
        ),
    ):
        wrapped = executor.wrap_tool(probe_tool, _probe_spec(), ledger)
        graph = _build_graph(wrapped)
        asyncio.run(_run_graph_only(graph, "i2-m2-ok"))
        leftovers = [str(path) for path in Path(td).rglob("*")]
    executor.set_default_ledger(None)
    snapshot = ledger.snapshot("i2-m2-ok")
    assert CALLS["n"] == 1, f"工具执行次数不是 1：{CALLS['n']}"
    assert snapshot.calls == 1, f"账本 calls 不是 1：{snapshot.calls}"
    assert abs(snapshot.cost - PROBE_UNIT_COST) < 1e-9, (
        f"账本 cost 不是 {PROBE_UNIT_COST}：{snapshot.cost}"
    )
    assert snapshot.tokens > 0, f"账本 tokens 未累计：{snapshot.tokens}"
    assert not leftovers, f"追踪开关关却产生文件：{leftovers}"
    return (
        f"执行 1 次；账本 calls=1 cost={snapshot.cost} tokens={snapshot.tokens}；TRACE_DIR 0 文件"
    )


# ── M3：只开追踪 ────────────────────────────────────────────────────


def m3_001() -> str:
    """M3-1/M3-2：turn 与 tool 两类 span 父子正确；权限预算关时账本零写入。"""
    from agent.adapters.tools import executor
    from agent.adapters.tools.budget_memory import MemoryBudgetLedger
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    ledger = MemoryBudgetLedger()
    executor.set_default_ledger(ledger)
    CALLS["n"], CALLS["values"], REQUESTS["n"] = 0, [], 0
    with (
        tempfile.TemporaryDirectory() as td,
        _settings(ENABLE_TOOL_POLICY=False, ENABLE_TRACE=True, TRACE_DIR=td),
    ):
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        graph = _build_graph(probe_tool)  # flag 关：不包装
        asyncio.run(_run_traced_turn(graph, tracer, callbacks, "i2-m3"))
        records = _read_records(Path(td))
    executor.set_default_ledger(None)

    turns = [r for r in records if r["kind"] == "turn"]
    tools = [r for r in records if r["kind"] == "tool"]
    assert len(turns) == 1, f"turn span 数量 {len(turns)}"
    assert len(tools) == 1, f"tool span 数量 {len(tools)}"
    assert turns[0]["parent_span_id"] is None, "turn.parent_span_id 不是 null"
    assert tools[0]["parent_span_id"] == turns[0]["span_id"], "tool 父 id 未指向 turn"
    snapshot = ledger.snapshot("i2-m3")
    assert snapshot.calls == 0 and snapshot.cost == 0.0, f"权限预算关却记账：{snapshot}"
    assert turns[0]["attrs"].get("cost") == 0.0, (
        f"turn cost 不是 0：{turns[0]['attrs'].get('cost')}"
    )
    return f"turn=1 tool=1 父子正确；账本 calls=0 cost=0；turn.cost=0；共 {len(records)} 条 span"


# ── M4：两个开关都开（核心组合） ────────────────────────────────────


def m4_001() -> str:
    """M4-1/M4-2/M4-3：span 树 + 账本同时成立；turn.cost == 账本累计；工具名一致。"""
    return _m4_scenario("i2-m4")


def _m4_scenario(session_id: str) -> str:
    from agent.adapters.tools import executor
    from agent.adapters.tools.budget_memory import MemoryBudgetLedger
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    ledger = MemoryBudgetLedger()
    executor.set_default_ledger(ledger)
    CALLS["n"], CALLS["values"], REQUESTS["n"] = 0, [], 0
    with (
        tempfile.TemporaryDirectory() as td,
        _settings(
            ENABLE_TOOL_POLICY=True,
            ENABLE_TRACE=True,
            TRACE_DIR=td,
            TOOL_MAX_CALLS_PER_TURN=8,
            TOOL_SESSION_COST_LIMIT=2.0,
        ),
    ):
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        wrapped = executor.wrap_tool(probe_tool, _probe_spec(), ledger)
        graph = _build_graph(wrapped)
        asyncio.run(_run_traced_turn(graph, tracer, callbacks, session_id))
        records = _read_records(Path(td))
        snapshot = ledger.snapshot(session_id)
    executor.set_default_ledger(None)

    turns = [r for r in records if r["kind"] == "turn"]
    tools = [r for r in records if r["kind"] == "tool"]
    assert len(turns) == 1, f"turn span 数量 {len(turns)}"
    assert tools, f"缺 tool span：{[r['kind'] for r in records]}"
    assert snapshot.calls == 1, f"账本 calls 不是 1：{snapshot.calls}"
    turn_cost = turns[0]["attrs"].get("cost")
    assert abs(float(turn_cost) - snapshot.cost) < 1e-9, (
        f"turn.cost={turn_cost} 与账本累计 cost={snapshot.cost} 不一致"
    )
    turn_tools = turns[0]["attrs"].get("tools")
    span_tools = [r["name"] for r in tools]
    assert turn_tools == span_tools, f"turn.tools={turn_tools} 与 tool span={span_tools} 不一致"
    return (
        f"span {len(records)} 条（turn=1 tool={len(tools)}）；账本 calls=1 cost={snapshot.cost}；"
        f"turn.cost={turn_cost} 与账本一致；tools={turn_tools}"
    )


# ── L1：账本次数与工具 span 条数对账 ───────────────────────────────


def l1_001() -> str:
    """L1：正常场景下账本 calls == tool 类 span 条数。"""
    from agent.adapters.tools import executor
    from agent.adapters.tools.budget_memory import MemoryBudgetLedger
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    ledger = MemoryBudgetLedger()
    executor.set_default_ledger(ledger)
    CALLS["n"], CALLS["values"], REQUESTS["n"] = 0, [], 0
    with (
        tempfile.TemporaryDirectory() as td,
        _settings(ENABLE_TOOL_POLICY=True, ENABLE_TRACE=True, TRACE_DIR=td),
    ):
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        wrapped = executor.wrap_tool(probe_tool, _probe_spec(), ledger)
        graph = _build_graph(wrapped)
        asyncio.run(_run_traced_turn(graph, tracer, callbacks, "i2-l1"))
        records = _read_records(Path(td))
        snapshot = ledger.snapshot("i2-l1")
    executor.set_default_ledger(None)
    tool_spans = [r for r in records if r["kind"] == "tool"]
    assert snapshot.calls == len(tool_spans) == 1, (
        f"账本 calls={snapshot.calls} 与 tool span={len(tool_spans)} 不一致"
    )
    return f"账本 calls={snapshot.calls} == tool span {len(tool_spans)}（拒绝场景见 G1）"


# ── L2：同轮 trace_id 一致，session_id == thread_id ─────────────────


def l2_001() -> str:
    """L2：同轮所有 span 的 trace_id 相同；session_id 恒等于 thread_id。"""
    from agent.adapters.tools import executor
    from agent.adapters.tools.budget_memory import MemoryBudgetLedger
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    ledger = MemoryBudgetLedger()
    executor.set_default_ledger(ledger)
    CALLS["n"], CALLS["values"], REQUESTS["n"] = 0, [], 0
    with (
        tempfile.TemporaryDirectory() as td,
        _settings(ENABLE_TOOL_POLICY=True, ENABLE_TRACE=True, TRACE_DIR=td),
    ):
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        wrapped = executor.wrap_tool(probe_tool, _probe_spec(), ledger)
        graph = _build_graph(wrapped)
        asyncio.run(_run_traced_turn(graph, tracer, callbacks, "i2-l2"))
        records = _read_records(Path(td))
    executor.set_default_ledger(None)
    trace_ids = {r["trace_id"] for r in records}
    sessions = {r["session_id"] for r in records}
    assert len(trace_ids) == 1, f"同轮出现多个 trace_id：{trace_ids}"
    assert sessions == {"i2-l2"}, f"session_id 与 thread_id 不一致：{sessions}"
    assert all(r["turn_index"] == 1 for r in records), "同轮 turn_index 不一致"
    return f"{len(records)} 条 span；trace_id={next(iter(trace_ids))[:8]}.. 唯一；session_id=i2-l2"


# ── L3：两个会话并发，作用域不串 ────────────────────────────────────


def l3_001() -> str:
    """L3：两个会话并发跑一轮，span 的 session_id 与账本作用域都不串。"""
    from agent.adapters.tools import executor
    from agent.adapters.tools.budget_memory import MemoryBudgetLedger
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    ledger = MemoryBudgetLedger()
    executor.set_default_ledger(ledger)
    CALLS["n"], CALLS["values"], REQUESTS["n"] = 0, [], 0
    with (
        tempfile.TemporaryDirectory() as td,
        _settings(ENABLE_TOOL_POLICY=True, ENABLE_TRACE=True, TRACE_DIR=td),
    ):
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        wrapped = executor.wrap_tool(probe_tool, _probe_spec(), ledger)
        graph = _build_graph(wrapped)

        async def scenario() -> None:
            await asyncio.gather(
                _run_traced_turn(graph, tracer, callbacks, "i2-l3-a"),
                _run_traced_turn(graph, tracer, callbacks, "i2-l3-b"),
            )

        asyncio.run(scenario())
        records = _read_records(Path(td))
        snap_a = ledger.snapshot("i2-l3-a")
        snap_b = ledger.snapshot("i2-l3-b")
    executor.set_default_ledger(None)

    turns = [r for r in records if r["kind"] == "turn"]
    assert len(turns) == 2, f"turn span 数量 {len(turns)}"
    trace_of: dict[str, str] = {}
    for record in records:
        session = record["session_id"]
        assert session in {"i2-l3-a", "i2-l3-b"}, f"串会话：{session}"
        trace_of.setdefault(session, record["trace_id"])
        assert trace_of[session] == record["trace_id"], f"{session} 出现多个 trace_id"
    assert trace_of["i2-l3-a"] != trace_of["i2-l3-b"], "两个会话共用了一个 trace_id"
    assert snap_a.calls == 1 and snap_b.calls == 1, f"账本作用域串了：{snap_a} / {snap_b}"
    assert abs(snap_a.cost - PROBE_UNIT_COST) < 1e-9, f"A 会话 cost 异常：{snap_a.cost}"
    assert abs(snap_b.cost - PROBE_UNIT_COST) < 1e-9, f"B 会话 cost 异常：{snap_b.cost}"
    return (
        f"{len(records)} 条 span 分属 2 个 session，trace_id 各异；"
        f"账本 a={snap_a.calls}/{snap_a.cost} b={snap_b.calls}/{snap_b.cost}"
    )


# ── G1：拒绝场景的追踪缺口（只取证，不判失败） ──────────────────────


def g1_001() -> str:
    """G1：构造一次拒绝，统计 tool 请求数 / tool span 数 / 实际执行数的差值。"""
    from agent.adapters.tools import executor
    from agent.adapters.tools.budget_memory import MemoryBudgetLedger
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import JsonlTracer

    ledger = MemoryBudgetLedger()
    executor.set_default_ledger(ledger)
    CALLS["n"], CALLS["values"], REQUESTS["n"] = 0, [], 0
    with (
        tempfile.TemporaryDirectory() as td,
        _settings(
            ENABLE_TOOL_POLICY=True,
            ENABLE_TRACE=True,
            TRACE_DIR=td,
            TOOL_MAX_CALLS_PER_TURN=0,
        ),
    ):
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        wrapped = executor.wrap_tool(probe_tool, _probe_spec(), ledger)
        graph = _build_graph(wrapped)
        asyncio.run(_run_traced_turn(graph, tracer, callbacks, "i2-g1"))
        records = _read_records(Path(td))
        snapshot = ledger.snapshot("i2-g1")
    executor.set_default_ledger(None)

    requests = REQUESTS["n"]
    tool_spans = len([r for r in records if r["kind"] == "tool"])
    executions = CALLS["n"]
    denied_spans = [r for r in records if r["kind"] == "tool" and not r["attrs"].get("error_kind")]
    delta_exec = tool_spans - executions
    delta_span = requests - tool_spans
    ISSUES.append(
        f"OI-INT-002 拒绝场景追踪缺口：tool 请求 {requests} 次，tool span {tool_spans} 条，"
        f"实际执行 {executions} 次，账本 calls={snapshot.calls}；"
        f"span-执行差 delta={delta_exec}，请求-span 差 delta={delta_span}；"
        f"该 span 无 error_kind / 拒绝标记，无法与正常执行区分"
    )
    print(f"OPEN_ISSUE: {ISSUES[-1]}")
    return (
        f"tool 请求={requests}；tool span={tool_spans}；实际执行={executions}；"
        f"账本 calls={snapshot.calls}；span-执行差={delta_exec}；请求-span 差={delta_span}；"
        f"缺拒绝标记 span={len(denied_spans)}"
    )


# ── G2：依赖检查的忽略项告警（只取证） ─────────────────────────────


def g2_001() -> str:
    """G2：lint-imports 的 adapters-only-ports 忽略项无匹配告警是否仍在。"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run(  # noqa: S603 - 参数固定、shell=False
        ["poetry", "run", "lint-imports"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
        check=False,
    )
    output = proc.stdout + proc.stderr
    last = _last_line(output)
    assert "3 kept, 0 broken" in output, f"契约不是 3 kept, 0 broken：{last}"
    warning = [line for line in output.splitlines() if "No matches for ignored import" in line]
    if warning:
        ISSUES.append(
            "OI-INT-003 adapters-only-ports 忽略项无匹配告警仍在：" + warning[0].strip()[:160]
        )
        print(f"OPEN_ISSUE: {ISSUES[-1]}")
    return f"Contracts: 3 kept, 0 broken；忽略项告警 {'仍在' if warning else '已消失'}"


# ── 入口 ────────────────────────────────────────────────────────────


def main() -> int:
    checker = Checker()
    print("== M1 两个开关都关（基线） ==")
    checker.run("INT2-M1-1", "工具结构哈希等于基线", m1_001)
    checker.run("INT2-M1-2", "注册表与 conversation_graph.TOOLS 同一批对象", m1_002)
    checker.run("INT2-M1-3", "追踪零文件 + 端点 config 无 callbacks", m1_003)

    print("== M2 只开权限预算 ==")
    checker.run("INT2-M2-1", "被拒绝时工具执行 0 次且不记账", m2_001)
    checker.run("INT2-M2-2", "正常调用账本 calls/cost 正确且追踪零文件", m2_002)

    print("== M3 只开追踪 ==")
    checker.run("INT2-M3-1", "turn/tool span 父子正确且账本零写入", m3_001)

    print("== M4 两个开关都开 ==")
    checker.run("INT2-M4-1", "span 树与账本同时成立且费用/工具名对账", m4_001)

    print("== L1/L2/L3 三条对账 ==")
    checker.run("INT2-L1", "账本 calls == tool span 条数", l1_001)
    checker.run("INT2-L2", "同轮 trace_id 一致且 session_id == thread_id", l2_001)
    checker.run("INT2-L3", "两会话并发作用域不串", l3_001)

    print("== G1/G2 已知缺口（只记录） ==")
    checker.run("INT2-G1", "拒绝场景 span 与调用次数差值", g1_001, g_class=True)
    checker.run("INT2-G2", "adapters-only-ports 忽略项告警", g2_001, g_class=True)

    total = checker.passed + checker.failed
    print(f"RESULT: {checker.passed} passed, {checker.failed} failed（共 {total} 项）")
    if ISSUES:
        print(f"C-FINDINGS: {len(ISSUES)} open issue")
    return 0 if checker.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
