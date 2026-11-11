#!/usr/bin/env python
"""T7-0 链路追踪冒烟（不连模型 / 不连 DB / 不连 Redis）。

用一张最小假图（两个节点 + 一个假工具）跑两轮：

- 第 1 轮：假模型 + 假工具正常返回 -> `turn -> llm` / `turn -> tool`；
- 第 2 轮：假工具抛异常 -> 仍产出一条 `status=error` 的 tool span（`error_kind` 非空）。

断言：

1. 开关关闭：`TRACE_DIR` 不产生任何文件，路由 config 里没有 `callbacks`；
2. 开关打开：span 树形、十字段、`kind` 六枚举、`attrs` 白名单、父子关系；
3. 一轮汇总：turn attrs 含 `tools / tokens_in / tokens_out / cost / span_count`；
4. 错误路径：tool span 的 `status=error`（内存句柄）且 `attrs.error_kind` 非空；
5. Langfuse 同时启用时 handlers 长度 2；无 handler 时降级为 1 并 warning。

用法（仓库根目录）：
  poetry run python scripts/smoke_trace.py

退出码：0 = 全过；1 = 有失败。
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig  # noqa: TC002 - LangGraph 运行期按签名注入
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

import agent.observability.langchain_handler as lc_handler
from agent.observability.langchain_handler import SpanCollectorHandler, build_callbacks
from agent.observability.tracer import (
    SPAN_ATTRS,
    SPAN_KINDS,
    TRACE_FIELDS,
    TURN_ONLY_ATTRS,
    SpanHandle,
    Tracer,
    get_tracer,
)
from agent.observability.turn import turn_span
from config import settings

_PASS = 0
_FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """打印一条断言结果并累计计数。"""
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        _FAIL += 1
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


class FakeChatModel(BaseChatModel):
    """本地假模型：不联网，只回一条带 usage_metadata 的 AIMessage。"""

    @property
    def _llm_type(self) -> str:
        return "fake-trace-model"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(
            content="mock-answer",
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
async def fake_tool(value: int = 1, explode: bool = False) -> str:
    """假工具：正常返回；explode=True 时抛异常，用于验证错误 span。"""
    if explode:
        raise RuntimeError("fake tool boom")
    return f"ok:{value}"


class SmokeState(MessagesState):
    """最小图状态：消息历史 + 是否让假工具爆炸。"""

    explode: bool


async def node_model(state: SmokeState, config: RunnableConfig) -> dict[str, Any]:
    """节点 1：走假模型（产生 llm span）。"""
    response = await FakeChatModel().ainvoke(state["messages"], config=config)
    return {"messages": [response]}


async def node_tool(state: SmokeState, config: RunnableConfig) -> dict[str, Any]:
    """节点 2：走假工具（产生 tool span；异常在节点内收口，模拟 Agent 的容错）。"""
    try:
        result = await fake_tool.ainvoke(
            {"value": 1, "explode": state.get("explode", False)}, config=config
        )
    except Exception as exc:  # noqa: BLE001 - 模拟 Agent 把工具异常回灌给模型
        result = f"tool-error:{type(exc).__name__}"
    return {"messages": [AIMessage(content=str(result))]}


def build_graph() -> Any:
    """两个节点 + 一个假工具的最小图。"""
    graph = StateGraph(SmokeState)
    graph.add_node("model", node_model)
    graph.add_node("tool", node_tool)
    graph.add_edge(START, "model")
    graph.add_edge("model", "tool")
    graph.add_edge("tool", END)
    return graph.compile()


class RecordingTracer:
    """记录 end_span 的 status（JSONL 里没有 status 字段，只能从内存句柄核对）。"""

    def __init__(self, inner: Tracer) -> None:
        self._inner = inner
        self.ended: list[tuple[SpanHandle, str]] = []

    def start_span(
        self,
        name: str,
        kind: str,
        *,
        parent: SpanHandle | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> SpanHandle:
        return self._inner.start_span(name, kind, parent=parent, attrs=attrs)

    def end_span(
        self,
        h: SpanHandle,
        *,
        status: str = "ok",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.ended.append((h, status))
        self._inner.end_span(h, status=status, attrs=attrs)


def route_config(thread_id: str, tracer: Tracer | None = None) -> dict[str, Any]:
    """镜像 api_main 里 /chat/conversation 的 config 组装（flag 关 -> 不碰 callbacks）。"""
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if settings.ENABLE_TRACE:
        config["callbacks"] = build_callbacks(tracer or get_tracer())
    return config


async def run_turn(
    graph: Any,
    tracer: Tracer,
    callbacks: list[Any],
    *,
    session_id: str,
    turn_index: int,
    explode: bool = False,
) -> None:
    """跑一轮：turn_span 包住整轮，callbacks 只挂在图 config 上。"""
    async with turn_span(tracer, session_id, turn_index):
        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hi")],
                "explode": explode,
            },
            config={
                "configurable": {"thread_id": session_id},
                "callbacks": callbacks,
            },
        )


def read_records(directory: Path) -> list[dict[str, Any]]:
    """读目录下所有 JSONL 行。"""
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def assert_fields(records: list[dict[str, Any]]) -> None:
    """十字段 / 六 kind / attrs 白名单 / 时间格式。"""
    bad_fields = [rec for rec in records if list(rec) != list(TRACE_FIELDS)]
    check("十字段齐全且顺序一致", not bad_fields, f"{len(records)} 条 span")

    bad_kinds = [rec["kind"] for rec in records if rec["kind"] not in SPAN_KINDS]
    check("kind ∈ 六枚举", not bad_kinds, f"kinds={sorted({rec['kind'] for rec in records})}")

    # v2：turn 根 span 额外承载本轮汇总 tools/span_count；其余五 kind 严格六键
    violations: list[tuple[str, list[str]]] = []
    for rec in records:
        allowed = SPAN_ATTRS | (TURN_ONLY_ATTRS if rec["kind"] == "turn" else frozenset())
        extra = sorted(set(rec["attrs"]) - allowed)
        if extra:
            violations.append((rec["kind"], extra))
    check(
        "attrs 键合规（turn 两汇总键为 v2 允许，其余严格六键）", not violations, str(violations[:3])
    )

    bad_turn_types: list[str] = []
    for rec in records:
        if rec["kind"] != "turn":
            continue
        tools = rec["attrs"].get("tools")
        count = rec["attrs"].get("span_count")
        if not (isinstance(tools, list) and all(isinstance(item, str) for item in tools)):
            bad_turn_types.append(f"tools={tools!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            bad_turn_types.append(f"span_count={count!r}")
    check(
        "turn 汇总键类型 tools: string[] / span_count: int", not bad_turn_types, str(bad_turn_types)
    )

    allowed_error_kinds = {"transient", "llm_recoverable", "user_fixable", "fatal"}
    bad_error_kinds = [
        rec["attrs"]["error_kind"]
        for rec in records
        if rec["attrs"].get("error_kind") not in (None, *allowed_error_kinds)
    ]
    check("error_kind ∈ 四类（空 = 成功）", not bad_error_kinds, str(bad_error_kinds))

    bad_time = []
    for rec in records:
        try:
            start = datetime.fromisoformat(rec["start"])
            end = datetime.fromisoformat(rec["end"])
        except ValueError:
            bad_time.append(rec["span_id"])
            continue
        if (
            start.tzinfo is None
            or end.tzinfo is None
            or end < start
            or start.microsecond % 1000
            or end.microsecond % 1000
        ):
            bad_time.append(rec["span_id"])
    check("start/end 为带偏移的毫秒 ISO 8601 且 end ≥ start", not bad_time, f"{len(records)} 条")


def assert_tree(records: list[dict[str, Any]]) -> None:
    """turn 为根、其余 span 的父 id 能在同一文件里找到。"""
    span_ids = {rec["span_id"] for rec in records}
    turns = [rec for rec in records if rec["kind"] == "turn"]
    check("turn span 的 parent_span_id 为 null", all(t["parent_span_id"] is None for t in turns))
    orphan = [
        rec["span_id"]
        for rec in records
        if rec["kind"] != "turn" and rec["parent_span_id"] not in span_ids
    ]
    check("非 turn span 的父节点都存在", not orphan, f"orphan={orphan[:3]}")


def assert_turn_summary(records: list[dict[str, Any]], turn_index: int, *, explode: bool) -> None:
    """一轮汇总：tools / tokens / cost / span_count 与父子关系。"""
    turn = next(
        (rec for rec in records if rec["kind"] == "turn" and rec["turn_index"] == turn_index),
        None,
    )
    check(f"第 {turn_index} 轮有 turn 根 span", turn is not None)
    if turn is None:
        return
    children = [rec for rec in records if rec["parent_span_id"] == turn["span_id"]]
    kinds = {rec["kind"] for rec in children}
    check(
        f"第 {turn_index} 轮 turn -> llm/tool 父子关系",
        {"llm", "tool"} <= kinds,
        str(sorted(kinds)),
    )

    attrs = turn["attrs"]
    check(
        f"第 {turn_index} 轮汇总键齐全",
        {"tools", "tokens_in", "tokens_out", "cost", "span_count"} <= set(attrs),
        str(sorted(attrs)),
    )
    check(
        f"第 {turn_index} 轮 tools 列表",
        attrs.get("tools") == ["fake_tool"],
        str(attrs.get("tools")),
    )
    check(
        f"第 {turn_index} 轮 token 汇总",
        attrs.get("tokens_in") == 7 and attrs.get("tokens_out") == 3,
        f"in={attrs.get('tokens_in')} out={attrs.get('tokens_out')}",
    )
    check(
        f"第 {turn_index} 轮 cost（G5 关 -> 0）", attrs.get("cost") == 0.0, str(attrs.get("cost"))
    )
    check(
        f"第 {turn_index} 轮 span_count = turn+llm+tool",
        attrs.get("span_count") == 3,
        str(attrs.get("span_count")),
    )
    if explode:
        tool_span = next((rec for rec in children if rec["kind"] == "tool"), None)
        check(
            f"第 {turn_index} 轮错误工具 span 的 error_kind 非空",
            tool_span is not None and bool(tool_span["attrs"].get("error_kind")),
            str(tool_span and tool_span["attrs"]),
        )


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="trace-smoke-"))
    print(f"临时目录：{tmp}")

    graph = build_graph()
    session_id = "tid-smoke"
    original_langfuse = settings.LANGFUSE_ENABLED
    # 先把 Langfuse 关掉，保证 collector 单独可测；共存场景在最后一节单独打开
    settings.LANGFUSE_ENABLED = False

    # ── 1) flag 关：零文件、零 callbacks ──────────────────────────
    settings.ENABLE_TRACE = False
    settings.TRACE_DIR = str(tmp / "off")
    off_config = route_config(session_id)
    check("flag 关：config 没有 callbacks", "callbacks" not in off_config, str(sorted(off_config)))
    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")], "explode": False},
        config={"configurable": {"thread_id": session_id}},
    )
    check("flag 关：TRACE_DIR 无任何文件", not (tmp / "off").exists())

    # ── 2) flag 开：两轮（正常 + 工具异常）────────────────────────
    settings.ENABLE_TRACE = True
    settings.TRACE_DIR = str(tmp / "on")
    base_tracer = get_tracer()
    recording = RecordingTracer(base_tracer)
    handler = SpanCollectorHandler(recording)
    on_config = route_config(session_id, tracer=recording)
    check("flag 开：config 挂上 callbacks", len(on_config.get("callbacks", [])) == 1)

    await run_turn(graph, recording, [handler], session_id=session_id, turn_index=1, explode=False)
    await run_turn(graph, recording, [handler], session_id=session_id, turn_index=2, explode=True)

    records = read_records(tmp / "on")
    check("flag 开：span 已落盘", len(records) == 6, f"{len(records)} 条")

    assert_fields(records)
    assert_tree(records)
    assert_turn_summary(records, 1, explode=False)
    assert_turn_summary(records, 2, explode=True)

    error_statuses = [(h.name, status) for h, status in recording.ended if status == "error"]
    check(
        "错误路径：tool span 内存 status=error",
        any(name == "fake_tool" for name, _ in error_statuses),
        str(error_statuses),
    )

    print("\nspan 树（JSONL 重建）：")
    for rec in records:
        indent = "" if rec["kind"] == "turn" else "  "
        print(
            f"{indent}turn_index={rec['turn_index']} {rec['kind']:<5} {rec['name']}"
            f" span={rec['span_id'][:8]} parent={str(rec['parent_span_id'])[:8]}"
        )

    sample = records[0]
    print("\n样例一行：")
    print(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))

    # ── 3) Langfuse 共存 / 降级 ───────────────────────────────────
    fake_langfuse = object()
    check(
        "Langfuse 同时启用：handlers 长度 2", len(build_callbacks(base_tracer, fake_langfuse)) == 2
    )

    capture = _Capture()
    handler_logger = logging.getLogger("agent.observability.langchain_handler")
    handler_logger.addHandler(capture)
    settings.LANGFUSE_ENABLED = True
    lc_handler._warned_no_langfuse = False  # 冒烟脚本显式重置“只警告一次”的进程级标记
    try:
        degraded = build_callbacks(base_tracer, None)
    finally:
        settings.LANGFUSE_ENABLED = original_langfuse
        handler_logger.removeHandler(capture)
    check("无 Langfuse 密钥：handlers 降级为 1", len(degraded) == 1)
    check(
        "无 Langfuse 密钥：只 warning 一次",
        sum("降级" in message for message in capture.messages) == 1,
        str(capture.messages),
    )

    print(f"\nRESULT: {_PASS} passed, {_FAIL} failed")
    if _FAIL == 0:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
