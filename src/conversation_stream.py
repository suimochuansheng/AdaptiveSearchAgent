"""会话式 Agent 的 SSE 事件编排（阶段 A 修复）。

背景：原 `/chat/conversation` 只在整张图跑完后吐一条 `final`，首响时延 = 全流程耗时；
`deep_research` 一次要跑 1~3 分钟，这期间连接完全静默。语音场景里 TTFT 是第一约束，
所以把 LangGraph 的 `astream` 编排成五类事件：

- `token`：agent 节点模型输出的增量文本（首响由模型首 token 决定，不再是全流程）；
- `tool_call`：模型决定调工具时立即通知，客户端可显示"正在调研…"；
- `ping`：`heartbeat_seconds` 秒内没有事件时发心跳，长工具调用期间保持连接活性；
- `final`：完整回答（兼容只读 final 的客户端与 curl 脚本）；
- `error`：执行异常（与旧端点同款结构）；
- `interrupt`：外层 HITL 审批挂起（T9-2，仅 `ENABLE_HITL` 开启且命中审批规则时出现；
  载荷与判定见 `agent/adapters/api/hitl.py`）。出现该帧后本轮不发 `final`，等 `/resume`
  续跑；开关关闭 / 无中断时帧序列与改造前逐帧一致。

只转发 `langgraph_node == "agent"` 的 chunk，因此 `deep_research` 内部那张 workflow
子图（planner/writer 等节点）的模型输出不会污染外层会话的 token 流。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, AIMessageChunk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

Event = dict[str, Any]


class ConversationStreamError(RuntimeError):
    """图执行失败。error 事件已经推送给客户端，端点只需收尾状态机，别再补一条。"""


# 队列哨兵：区分"流结束"与"流异常"两种终止
_DONE = "__done__"
_FAILED = "__failed__"

# 流式转发时只认这一个节点（见模块 docstring 末段）
_AGENT_NODE = "agent"


def _sse(event: Event) -> str:
    """把事件字典编码成 SSE 帧（与 /chat/stream 保持同一格式）。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _tool_call_event(message: AIMessage) -> Event:
    """模型发起工具调用时的事件（只带名字和参数，不带工具结果）。"""
    calls = [
        {"name": call.get("name", ""), "args": call.get("args", {})} for call in message.tool_calls
    ]
    return {"type": "tool_call", "tools": calls}


def _agent_text(payload: Any) -> str | None:
    """判断 messages 流里的一个 chunk 是否属于 agent 节点，是则返回文本。"""
    try:
        chunk, metadata = payload
    except (TypeError, ValueError):
        return None
    if not isinstance(metadata, dict) or metadata.get("langgraph_node") != _AGENT_NODE:
        return None
    if not isinstance(chunk, AIMessageChunk):
        return None
    # 工具调用轮的 chunk 只有 tool_call_chunks，content 为空 → 不是给用户看的文本
    return str(chunk.content) if chunk.content else None


def _new_tool_calls(payload: Any, seen: int) -> tuple[Event | None, int]:
    """values 模式每个 superstep 吐全量状态：按"已转发条数"找出新的工具调用。

    Args:
        payload: values 流里的全量状态。
        seen: 已经检查过的消息条数。

    Returns:
        (tool_call 事件或 None, 新的 seen)
    """
    messages = payload.get("messages") or []
    if len(messages) <= seen:
        return None, seen
    found: Event | None = None
    for message in messages[seen:]:
        if isinstance(message, AIMessage) and message.tool_calls:
            found = _tool_call_event(message)
    return found, len(messages)


def _consume(
    kind: str, payload: Any, snapshot: dict[str, Any], answer_parts: list[str]
) -> str | None:
    """把一条流事件转成 SSE 帧（token / tool_call），没有可推送内容时返回 None。"""
    if kind == "messages":
        text = _agent_text(payload)
        if not text:
            return None
        answer_parts.append(text)
        return _sse({"type": "token", "content": text})

    snapshot["final"] = payload
    tool_event, snapshot["seen"] = _new_tool_calls(payload, snapshot["seen"])
    return _sse(tool_event) if tool_event else None


async def _next_event(
    queue: asyncio.Queue[tuple[str, Any]], timeout: float
) -> tuple[str, Any] | None:
    """取队列里的下一条事件；静默超时返回 None（调用方据此发心跳）。"""
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except TimeoutError:
        return None


def _remaining_until_ping(
    last_emit: float, heartbeat_seconds: float, clock: Callable[[], float]
) -> float:
    """距下一次必须发心跳还剩多少秒（<=0 表示已经该发了）。

    关键：等待时间必须用"剩余时间"，不能用固定 heartbeat——否则被过滤的事件
    在 9.9s 到达时既不发 ping、又重新等满 10s，下一条就被推迟到接近 20s。
    """
    return heartbeat_seconds - (clock() - last_emit)


async def _wait_for_event(
    queue: asyncio.Queue[tuple[str, Any]],
    *,
    last_emit: float,
    heartbeat_seconds: float,
    clock: Callable[[], float],
) -> tuple[str, Any] | None:
    """等到下一条事件；等满"剩余心跳时间"就返回 None（调用方发 ping）。

    已静默满一个周期时立即返回 None，不再排队等——否则一条 9.9s 到达的
    被过滤事件会把下一条 ping 推迟到接近 20s。
    """
    remaining = _remaining_until_ping(last_emit, heartbeat_seconds, clock)
    if remaining <= 0:
        return None
    return await _next_event(queue, remaining)


async def _baseline_messages(graph: Any, config: dict[str, Any]) -> int:
    """读取进入本次请求前 checkpoint 里已有的消息条数（无 checkpointer 时为 0）。"""
    getter = getattr(graph, "aget_state", None)
    if getter is None:
        return 0
    try:
        snapshot = await getter(config)
    except Exception:  # noqa: BLE001 - 读不到历史不影响本轮
        return 0
    values = getattr(snapshot, "values", None) or {}
    return len(values.get("messages") or [])


async def _stop_producer(producer: asyncio.Task[None]) -> None:
    """客户端断连/异常时让后台跑图任务退出，避免图继续跑却没人收结果。"""
    if producer.done():
        return
    producer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await producer


async def stream_conversation(  # noqa: C901 - 帧循环本身是状态机，拆开反而难读
    graph: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
    *,
    heartbeat_seconds: float = 10.0,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """把一次图执行编排成 SSE 事件流。

    Args:
        graph: 已编译的 LangGraph 图（需支持 `astream(stream_mode=[...])`）。
        inputs: 本轮输入，如 `{"messages": [HumanMessage(...)]}`。
        config: LangGraph 配置（thread_id 等）。
        heartbeat_seconds: 静默多久发一次 ping。
        is_disconnected: 客户端断连探测（传入 `request.is_disconnected`）。
        clock: 单调时钟（测试注入假时钟，避免用 sleep 制造时序）。

    Yields:
        SSE 帧字符串，形如 `data: {"type": "token", "content": "…"}\n\n`。
    """
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    # 流式状态：图的最新全量快照、已检查过的消息条数、已推送的 token 片段
    # seen 从"进入本次请求前的历史条数"起算：checkpoint 会把历史一起回放，
    # 从 0 起算会把上一轮的工具调用当成这一轮的事件重发给前端。
    snapshot: dict[str, Any] = {"final": None, "seen": await _baseline_messages(graph, config)}
    answer_parts: list[str] = []

    async def _produce() -> None:
        """后台跑图，把每个 superstep / chunk 塞进队列（生成器负责心跳与断连）。"""
        try:
            async for mode, payload in graph.astream(
                inputs,
                config=config,
                stream_mode=["messages", "values"],
            ):
                await queue.put((mode, payload))
            await queue.put((_DONE, None))
        except Exception as exc:  # noqa: BLE001 - 统一转成 error 事件，交由端点收尾
            await queue.put((_FAILED, exc))

    producer = asyncio.create_task(_produce())
    # "静默"的判据是"距上一帧多久"，不是"队列空不空"：
    # deep_research 子图的 token 会源源不断进队列，但被节点过滤掉、不产生帧，
    # 只按队列判空会导致子图连续出字时客户端全程无声（实测 272s 无 ping）。
    last_emit = clock()
    interrupted = False
    try:
        while True:
            item = await _wait_for_event(
                queue,
                last_emit=last_emit,
                heartbeat_seconds=heartbeat_seconds,
                clock=clock,
            )
            if item is None:
                # 静默满一个心跳周期（或被过滤的事件不算活跃）→ 补 ping
                yield _sse({"type": "ping"})
                last_emit = clock()
                continue

            kind, payload = item
            if kind == _DONE:
                break
            if kind == _FAILED:
                yield _sse({"type": "error", "message": f"服务异常：{payload}"})
                raise ConversationStreamError(str(payload))
            if is_disconnected is not None and await is_disconnected():
                return

            # 外层 HITL：LangGraph 的中断出现在 values chunk 的 `__interrupt__` 键里
            # （不是独立的 stream mode）。只有带 approval_id 的外层审批才编码成
            # interrupt 帧；子图成本确认等非外层中断跳过，不改变原有帧语义。
            if kind == "values" and isinstance(payload, dict) and payload.get("__interrupt__"):
                from agent.adapters.api.hitl import build_interrupt_frame

                interrupt_frame = build_interrupt_frame(payload)
                if interrupt_frame:
                    yield interrupt_frame
                    last_emit = clock()
                    interrupted = True
                    break
                continue

            frame = _consume(kind, payload, snapshot, answer_parts)
            if frame:
                yield frame
                last_emit = clock()
    finally:
        await _stop_producer(producer)

    if interrupted:
        # 本轮已挂起等人工审批：不发 final（恢复后由 /resume 继续吐后续帧）
        return

    final_state = snapshot["final"] or {}
    messages = final_state.get("messages") or []
    # 优先用最终快照里的完整回答；快照缺失时退回已推送的 token 拼接
    answer = str(messages[-1].content) if messages else "".join(answer_parts)
    yield _sse({"type": "final", "content": answer})
