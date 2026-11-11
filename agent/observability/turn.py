"""一轮的根 span 与汇总（T7-0）。

`turn_span(tracer, session_id, turn_index)` 是异步上下文管理器：

- 进入时绑定本轮上下文（trace_id / session / turn_index）并开 `kind="turn"` 根 span；
- 子 span（llm / tool）由 `SpanCollectorHandler` 自动挂到根 span 下；
- 退出时把子 span 聚合进 `attrs`：`tools / tokens_in / tokens_out / cost / span_count`；
- **异常也关闭**：`status="error"`，`attrs.error_kind` 非空（契约没有 status 字段，
  错误只通过 `error_kind` 表达）。

`cost` 读 G5 账本快照；`ENABLE_TOOL_POLICY` 关闭时恒为 0（账本本来就不记）。
口径是**会话累计**（账本 `snapshot()` 的返回值，不是本轮增量），与 T9-5 费用归因一致。
"""

from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from agent.observability.tracer import SpanHandle, Tracer, bind_turn
from config import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


async def _ledger_cost(session_id: str) -> float:
    """读 G5 账本快照里的会话累计费用；开关关/读失败都记 0。"""
    if not settings.ENABLE_TOOL_POLICY:
        return 0.0
    try:
        from agent.adapters.tools.executor import get_default_ledger

        snapshot = get_default_ledger().snapshot(session_id)
        if inspect.isawaitable(snapshot):
            snapshot = await snapshot
        return round(float(snapshot.cost), 6)
    except BaseException as exc:  # noqa: BLE001 - 含 CancelledError：收尾阶段不能因读账本卡住落盘
        logger.warning("读取 G5 账本失败，turn cost 记 0：%s", exc)
        return 0.0


async def _aggregate(
    spans: list[SpanHandle],
    *,
    root: SpanHandle,
    error_kind: str | None,
) -> dict[str, Any]:
    """把本轮子 span 汇总成 turn 根 span 的 attrs。"""
    children = [h for h in spans if h is not root and not h.skipped]
    attrs: dict[str, Any] = {
        "tools": [h.name for h in children if h.kind == "tool"],
        "tokens_in": sum(int(h.attrs.get("tokens_in", 0) or 0) for h in children),
        "tokens_out": sum(int(h.attrs.get("tokens_out", 0) or 0) for h in children),
        "cost": await _ledger_cost(root.session_id),
        # 含根 span 本身：turn + llm + tool = 3
        "span_count": len([h for h in spans if h is root or not h.skipped]),
    }
    if error_kind:
        attrs["error_kind"] = error_kind
    return attrs


@asynccontextmanager
async def turn_span(
    tracer: Tracer,
    session_id: str,
    turn_index: int,
) -> AsyncIterator[SpanHandle]:
    """开一轮的根 span，并在退出时写入汇总。"""
    with bind_turn(tracer, session_id=session_id, turn_index=turn_index) as ctx:
        root = tracer.start_span("turn", "turn")
        status = "ok"
        error_kind: str | None = None
        try:
            yield root
        except BaseException as exc:
            status = "error"
            # 客户端断连不是业务失败，按 transient 记；其余按 fatal
            error_kind = "transient" if isinstance(exc, GeneratorExit) else "fatal"
            raise
        finally:
            attrs = await _aggregate(ctx.spans, root=root, error_kind=error_kind)
            tracer.end_span(root, status=status, attrs=attrs)


async def next_turn_index(graph: Any, config: dict[str, Any]) -> int:
    """run 前数历史 `HumanMessage`，返回本轮序号（从 1 开始）。

    拿不到状态（checkpointer 不支持 / 查询异常）时退回 0 并只 warning，不阻塞会话。
    """
    try:
        snapshot = await graph.aget_state(config)
        values = getattr(snapshot, "values", None)
        if not isinstance(values, dict):
            raise ValueError("aget_state 没有返回 values")
        messages = values.get("messages")
        if messages is None:
            # 新会话首次运行：checkpoint 还不存在，历史轮次就是 0
            return 1
        return sum(isinstance(message, HumanMessage) for message in messages) + 1
    except Exception as exc:  # noqa: BLE001 - 追踪不是业务依赖
        logger.warning("取 turn_index 失败，退化为 0：%s", exc)
        return 0


__all__ = ["next_turn_index", "turn_span"]
