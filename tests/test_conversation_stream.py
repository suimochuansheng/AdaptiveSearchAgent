"""会话式 Agent 的 SSE 事件编排测试（阶段 A 修复）。

覆盖：token 增量、tool_call 通知、非 agent 节点过滤、心跳 ping、error 收尾、断连短路。
不需要真实 LLM：用假图（_FakeGraph）回放 LangGraph 的 (mode, payload) 序列。
"""

import asyncio
import itertools
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.conversation_stream import ConversationStreamError, stream_conversation


class _FakeGraph:
    """回放预置事件的假图，接口与 LangGraph 编译图一致（只需要 astream）。"""

    def __init__(
        self,
        events: list[tuple[str, Any]],
        *,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._events = events
        self._delay = delay
        self._error = error
        self.stream_mode: Any = None

    async def astream(  # type: ignore[no-untyped-def]
        self, inputs: dict[str, Any], config: dict[str, Any] | None = None, stream_mode: Any = None
    ):
        self.stream_mode = stream_mode
        for mode, payload in self._events:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield mode, payload
        if self._error is not None:
            raise self._error


def _parse(frame: str) -> dict[str, Any]:
    """SSE 帧 → 事件字典，顺带断言帧格式。"""
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    return json.loads(frame[len("data: ") :])


async def _collect(graph: _FakeGraph, **kwargs: Any) -> list[dict[str, Any]]:
    frames = [frame async for frame in stream_conversation(graph, {"messages": []}, {}, **kwargs)]
    return [_parse(frame) for frame in frames]


def _chunk(text: str, node: str = "agent") -> tuple[str, Any]:
    return "messages", (AIMessageChunk(content=text), {"langgraph_node": node})


async def test_tokens_stream_then_final() -> None:
    """agent 节点的增量文本按序推送，末尾补一条完整 final。"""
    graph = _FakeGraph(
        [
            _chunk("1.7"),
            _chunk("7"),
            ("values", {"messages": [HumanMessage("q"), AIMessage("1.77")]}),
        ]
    )

    events = await _collect(graph)

    assert [e["type"] for e in events] == ["token", "token", "final"]
    assert "".join(e["content"] for e in events[:-1]) == "1.77"
    assert events[-1]["content"] == "1.77"


async def test_requests_both_stream_modes() -> None:
    """必须向图要 messages（token）与 values（快照）两种流。"""
    graph = _FakeGraph([("values", {"messages": [AIMessage("done")]})])

    await _collect(graph)

    assert graph.stream_mode == ["messages", "values"]


async def test_tool_call_is_announced_once() -> None:
    """模型发起工具调用时推 tool_call；同一批消息不会被重复通知。"""
    tool_call = {"name": "deep_research", "args": {"query": "对比 A 和 B"}, "id": "call-1"}
    graph = _FakeGraph(
        [
            ("values", {"messages": [HumanMessage("q")]}),
            ("values", {"messages": [HumanMessage("q"), AIMessage("", tool_calls=[tool_call])]}),
            (
                "values",
                {
                    "messages": [
                        HumanMessage("q"),
                        AIMessage("", tool_calls=[tool_call]),
                        ToolMessage("调研完成", tool_call_id="call-1"),
                        AIMessage("结论：A 更强"),
                    ]
                },
            ),
        ]
    )

    events = await _collect(graph)

    tool_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["tools"] == [{"name": "deep_research", "args": {"query": "对比 A 和 B"}}]
    assert events[-1] == {"type": "final", "content": "结论：A 更强"}


async def test_chunks_from_other_nodes_are_ignored() -> None:
    """deep_research 子图（writer 等节点）的输出不能混进外层 token 流。"""
    graph = _FakeGraph(
        [
            _chunk("子图内部报告", node="writer"),
            _chunk("外层回答"),
            ("values", {"messages": [AIMessage("外层回答")]}),
        ]
    )

    events = await _collect(graph)

    assert [e["type"] for e in events] == ["token", "final"]
    assert events[0]["content"] == "外层回答"


async def test_heartbeat_ping_while_tool_runs() -> None:
    """长工具调用期间没有事件时发 ping，避免连接被判死。"""
    graph = _FakeGraph(
        [
            _chunk("开始"),
            _chunk("结束"),
            ("values", {"messages": [AIMessage("结束")]}),
        ],
        delay=0.05,
    )

    events = await _collect(graph, heartbeat_seconds=0.01)

    assert any(e["type"] == "ping" for e in events)
    assert events[-1]["type"] == "final"


async def test_graph_error_yields_error_frame_then_raises() -> None:
    """图抛异常：先推 error 帧（客户端能收到），再抛 ConversationStreamError 让端点收尾。"""
    graph = _FakeGraph([_chunk("半截")], error=RuntimeError("tool exploded"))
    frames: list[str] = []

    with pytest.raises(ConversationStreamError):
        async for frame in stream_conversation(graph, {"messages": []}, {}):
            frames.append(frame)

    events = [_parse(f) for f in frames]
    assert events[-1]["type"] == "error"
    assert "tool exploded" in events[-1]["message"]
    # 失败时不应该再补一条 final
    assert all(e["type"] != "final" for e in events)


async def test_disconnect_stops_before_final() -> None:
    """客户端断开后立即收手，不再产生事件。"""

    async def _disconnected() -> bool:
        return True

    graph = _FakeGraph([_chunk("hi"), ("values", {"messages": [AIMessage("hi")]})])

    frames = [
        frame
        async for frame in stream_conversation(
            graph, {"messages": []}, {}, is_disconnected=_disconnected
        )
    ]

    assert frames == []


async def test_ping_while_only_filtered_subgraph_chunks_arrive() -> None:
    """回归：deep_research 子图连续出字期间（事件非空但被过滤）也必须按静默时长发 ping。

    这是验收里实测到的 bug：只看"队列是否为空"会把被过滤的事件当成活跃，
    子图连续输出多久，客户端就静默多久（实测 272s 无 ping）。
    用假时钟保证确定性，不依赖 sleep。
    """
    ticks = itertools.count()

    def _clock() -> float:
        return float(next(ticks))

    graph = _FakeGraph([_chunk("子图内部输出", node="writer") for _ in range(8)])

    events = await _collect(graph, heartbeat_seconds=2.0, clock=_clock)

    pings = [e for e in events if e["type"] == "ping"]
    assert len(pings) >= 2, f"被过滤的空转没有补心跳：{events}"
    # 全程没有 agent chunk → 不应出现 token，但 final 必须在
    assert all(e["type"] != "token" for e in events)
    assert events[-1]["type"] == "final"
