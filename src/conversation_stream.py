"""会话式 Agent 的 SSE 事件编排（阶段 A 修复）。

背景：原 `/chat/conversation` 只在整张图跑完后吐一条 `final`，首响时延 = 全流程耗时；
`deep_research` 一次要跑 1~3 分钟，这期间连接完全静默。语音场景里 TTFT 是第一约束，
所以把 LangGraph 的 `astream` 编排成五类事件：

- `token`：agent 节点模型输出的增量文本（首响由模型首 token 决定，不再是全流程）；
- `tool_call`：模型决定调工具时立即通知，客户端可显示"正在调研…"；
- `ping`：`heartbeat_seconds` 秒内没有事件时发心跳，长工具调用期间保持连接活性；
- `final`：完整回答（兼容只读 final 的客户端与 curl 脚本）；
- `error`：执行异常（与旧端点同款结构）。

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


def _frame_or_ping(
    frame: str | None,
    *,
    last_emit: float,
    heartbeat_seconds: float,
    clock: Callable[[], float],
) -> tuple[str | None, float]:
    """有帧就发帧；没帧但已静默超时就补心跳。

    返回 (要发送的帧或 None, 新的"上次发送时间")。被过滤的事件（如 deep_research
    子图的 token）不产生帧，因此只能靠这里的静默计时兜住。
    """
    if frame:
        return frame, clock()
    if clock() - last_emit >= heartbeat_seconds:
        return _sse({"type": "ping"}), clock()
    return None, last_emit


async def _stop_producer(producer: asyncio.Task[None]) -> None:
    """客户端断连/异常时让后台跑图任务退出，避免图继续跑却没人收结果。"""
    if producer.done():
        return
    producer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await producer


async def stream_conversation(
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
    snapshot: dict[str, Any] = {"final": None, "seen": 0}
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
    try:
        while True:
            item = await _next_event(queue, heartbeat_seconds)
            if item is None:
                # 队列真空闲 → 发心跳，避免客户端/代理判定连接已死
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

            frame, last_emit = _frame_or_ping(
                _consume(kind, payload, snapshot, answer_parts),
                last_emit=last_emit,
                heartbeat_seconds=heartbeat_seconds,
                clock=clock,
            )
            if frame:
                yield frame
    finally:
        await _stop_producer(producer)

    final_state = snapshot["final"] or {}
    messages = final_state.get("messages") or []
    # 优先用最终快照里的完整回答；快照缺失时退回已推送的 token 拼接
    answer = str(messages[-1].content) if messages else "".join(answer_parts)
    yield _sse({"type": "final", "content": answer})
