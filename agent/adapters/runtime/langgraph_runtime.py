"""`AgentRuntime` 的 LangGraph 适配器（T-arch 样板切片）。

职责边界（本文件只做"翻译"，不重写主循环）：

- 主循环仍由现网 ``src.conversation_graph.build_conversation_graph()`` 构建；
- 流式编排仍由 ``src.conversation_stream.stream_conversation()`` 完成（它已经处理
  心跳、断连、interrupt 与 error 帧）；
- 本适配器只做两件事：把 ``session`` 翻译成 LangGraph 的 ``inputs`` / ``config``，
  再把 SSE 帧解析成 :class:`agent.app.ports.RuntimeEvent`。

**为什么循环不搬**：中断/恢复、checkpoint、流式事件这些轮子成熟且与框架强耦合；
把它们搬进应用层等于重写 LangGraph。适配器是"防腐层"：应用层看到的是
``RuntimeEvent``，框架类型（``Command``、``AIMessage``、``astream``）留在这里。

**hooks 只透传**：压缩 / 校验 / 预算策略不在这里做，``hooks`` 原样挂到
``configurable["runtime_hooks"]``，由应用层或图内节点按需取用。

契约见 ``docs/contracts/runtime-event.md``：``ping`` 是传输层心跳，解析时丢弃；
``kind`` 只允许六值，未知 ``type`` 静默忽略（与前端一致）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agent.app.ports import RuntimeEvent
from src.conversation_graph import build_conversation_graph
from src.conversation_stream import stream_conversation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

__all__ = ["LangGraphRuntime", "event_from_frame"]

#: SSE `type` -> RuntimeEvent.kind。`ping` 不在表里，因此会被丢弃。
_KIND_BY_FRAME_TYPE: dict[str, str] = {
    "token": "token",
    "tool_call": "tool_call",
    "tool_result": "tool_result",
    "interrupt": "interrupt",
    "final": "final",
    "error": "error",
}

#: `payload` 的键名（契约冻结，见 runtime-event.md §3）；不在表里的字段不往下传。
_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "token": ("content",),
    "tool_call": ("tools",),
    "tool_result": ("tool_call_id", "tool", "content"),
    "interrupt": ("approval",),
    "final": ("content",),
    "error": ("message",),
}


def event_from_frame(frame: str) -> RuntimeEvent | None:
    """把一条 SSE 帧解析成 ``RuntimeEvent``。

    Args:
        frame: ``stream_conversation`` 产出的帧，形如 ``data: {"type": ...}\\n\\n``。

    Returns:
        ``RuntimeEvent``；``ping``、非 ``data:`` 帧、坏 JSON、未知 ``type`` 均返回
        ``None``（静默忽略，与前端对未知 type 的处理一致）。
    """
    if not isinstance(frame, str):
        return None
    text = frame.strip()
    if not text.startswith("data: "):
        return None
    try:
        data = json.loads(text[len("data: ") :])
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    kind = _KIND_BY_FRAME_TYPE.get(str(data.get("type")))
    if kind is None:
        # ping / 未知 type：不制造 RuntimeEvent
        return None

    if kind == "interrupt":
        # SSE 帧是 {"type":"interrupt","data":{八字段}}；契约要求 payload.approval
        approval = data.get("data")
        payload: dict[str, Any] = {"approval": approval if isinstance(approval, dict) else {}}
    else:
        payload = {key: data.get(key) for key in _PAYLOAD_KEYS[kind]}

    return RuntimeEvent(kind=kind, payload=payload)  # type: ignore[arg-type]


def _session_value(session: Any, key: str, default: Any = None) -> Any:
    """从 dict / 对象两种 session 形态里取字段（适配器不绑定某一种会话实体）。"""
    if isinstance(session, Mapping):
        return session.get(key, default)
    return getattr(session, key, default)


class LangGraphRuntime:
    """``AgentRuntime`` 的 LangGraph 实现（结构上满足端口，不显式继承 Protocol）。

    构造参数全部可注入，便于契约测试用假图/假帧流，不连 Ollama：

    Args:
        graph: 已编译的 LangGraph 图；不传则首次 ``run_turn`` 时调用 ``graph_factory``。
        graph_factory: 图工厂，默认 ``build_conversation_graph``。
        stream_factory: 帧流工厂，默认 ``src.conversation_stream.stream_conversation``。
            测试注入假实现即可断言 `RuntimeEvent` 映射，不触发真实模型。
        heartbeat_seconds: 透传给帧流的心跳间隔（本层不解释）。
        clock: 单调时钟（透传，测试可注入假时钟）。
    """

    def __init__(
        self,
        graph: Any | None = None,
        *,
        graph_factory: Callable[[], Any] | None = None,
        stream_factory: Callable[..., AsyncIterator[str]] | None = None,
        heartbeat_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._graph = graph
        self._graph_factory = graph_factory or build_conversation_graph
        self._stream_factory = stream_factory or stream_conversation
        self._heartbeat_seconds = heartbeat_seconds
        self._clock = clock

    def _resolve_graph(self) -> Any:
        """惰性建图：没有注入图时首次调用才构建（避免 import 期连模型/Ollama）。"""
        if self._graph is None:
            self._graph = self._graph_factory()
        return self._graph

    def _inputs_for(self, session: Any) -> dict[str, Any]:
        """把 session 翻译成 LangGraph 的 inputs。"""
        raw_input = _session_value(session, "input")
        if isinstance(raw_input, dict):
            return raw_input
        messages = _session_value(session, "messages")
        if messages is not None:
            return {"messages": list(messages)}
        return {"messages": []}

    def _config_for(self, session: Any, hooks: Any | None) -> dict[str, Any]:
        """把 session / hooks 翻译成 LangGraph 的 config（hooks 只透传）。"""
        thread_id = _session_value(session, "thread_id") or _session_value(session, "session_id")
        if thread_id is None:
            raise ValueError("session 缺少 thread_id，无法定位会话")
        extra = _session_value(session, "config", {}) or {}
        configurable = dict(extra.get("configurable") or {})
        configurable["thread_id"] = str(thread_id)
        if hooks is not None:
            # 只挂载，不解释：压缩/校验/预算由应用层钩子自己消费
            configurable.setdefault("runtime_hooks", hooks)
        return {**extra, "configurable": configurable}

    async def _events(
        self,
        session: Any,
        inputs: Any,
        hooks: Any | None,
    ) -> AsyncIterator[RuntimeEvent]:
        """驱动帧流并逐帧翻译（`run_turn` 与 `resume` 共用）。"""
        graph = self._resolve_graph()
        config = self._config_for(session, hooks)
        async for frame in self._stream_factory(
            graph,
            inputs,
            config,
            heartbeat_seconds=self._heartbeat_seconds,
            clock=self._clock,
        ):
            event = event_from_frame(frame)
            if event is not None:
                yield event

    async def run_turn(
        self,
        session: Any,
        hooks: Any | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        """跑一轮对话：``stream_conversation`` 的帧 -> ``RuntimeEvent``。"""
        async for event in self._events(session, self._inputs_for(session), hooks):
            yield event

    async def resume(
        self,
        session: Any,
        command: Any,
    ) -> AsyncIterator[RuntimeEvent]:
        """从 HITL 挂起恢复：复用 G7 的恢复原语 ``Command(resume=...)``。

        与 ``agent/adapters/api/hitl.py::_resume_event_stream`` 走的是同一段
        ``stream_conversation``；本适配器不重复实现审批存储 / 锁 / HTTP 状态码矩阵
        （那些是入站适配器的职责），只保证恢复后的事件同样被翻译成 ``RuntimeEvent``。
        """
        from langgraph.types import Command

        inputs = command if isinstance(command, Command) else Command(resume=command)
        async for event in self._events(session, inputs, None):
            yield event
