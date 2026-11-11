"""`AgentRuntime` 端口契约测试（T-arch 样板，只此一份）。

为什么只写一份：契约属于**端口**，不属于某个适配器。把同一组断言参数化到所有
实现上，换实现（LangGraph -> 别的框架）时只需要往参数表里加一行，断言不用抄。
这里挂两个实现：

- `LangGraphRuntime`：真实适配器 + 假图（不连 Ollama，不起子进程）；
- `_FakeRuntime`：最小内存实现，证明端口是**结构化**的（不要求继承某个基类）。

断言（每个实现都一样）：

1. `isinstance(impl, AgentRuntime)` 为真（`@runtime_checkable`）；
2. `run_turn` / `resume` 产出的 `kind` 只出现契约里的六值；
3. `ping`（SSE 心跳）**不得**出现在 `RuntimeEvent` 里——适配器必须丢弃；
4. 每个事件都是 `RuntimeEvent`，`payload` 是 dict。

真实 LangGraph 路径用假图 + 0.01s 心跳强制产生 `ping` 帧，再由适配器过滤，这样
"ping 不出现"不是靠实现自觉，而是被真实帧流打到。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from agent.adapters.runtime.langgraph_runtime import LangGraphRuntime
from agent.app.ports import AgentRuntime, RuntimeEvent

#: 契约冻结的六值（`docs/contracts/runtime-event.md` §2）。
SIX_KINDS = {"token", "tool_call", "tool_result", "interrupt", "final", "error"}

_SESSION: dict[str, Any] = {
    "thread_id": "t-contract",
    "input": {"messages": []},
}


class _FakeGraph:
    """最小 LangGraph 替身：只提供 `stream_conversation` 用到的两个方法。

    `astream` 先睡 0.03s（> 心跳 0.01s），强制帧流先发一条 `ping`，再发
    token / final；用来验证适配器确实把 ping 丢掉了。
    """

    async def aget_state(self, config: dict[str, Any]) -> Any:
        raise RuntimeError("假图没有 checkpointer")

    async def astream(self, inputs: Any, config: dict[str, Any], stream_mode: list[str]):
        await asyncio.sleep(0.03)
        yield (
            "messages",
            (AIMessageChunk(content="你好"), {"langgraph_node": "agent"}),
        )
        yield ("values", {"messages": [AIMessage(content="你好")]})


class _FakeRuntime:
    """端口的内存实现：不依赖任何框架，用来证明契约只约束行为。"""

    async def run_turn(self, session: Any, hooks: Any | None = None):
        yield RuntimeEvent(kind="token", payload={"content": "你"})
        yield RuntimeEvent(kind="tool_call", payload={"tools": [{"name": "t", "args": {}}]})
        yield RuntimeEvent(
            kind="tool_result",
            payload={"tool_call_id": "c1", "tool": "t", "content": "ok"},
        )
        yield RuntimeEvent(kind="interrupt", payload={"approval": {"approval_id": "0" * 32}})
        yield RuntimeEvent(kind="error", payload={"message": "示例错误"})
        yield RuntimeEvent(kind="final", payload={"content": "你好"})

    async def resume(self, session: Any, command: Any):
        yield RuntimeEvent(kind="final", payload={"content": "恢复完成"})


def _langgraph_runtime() -> LangGraphRuntime:
    """真实适配器 + 假图；心跳调到 0.01s 以便强制产生 ping。"""
    return LangGraphRuntime(_FakeGraph(), heartbeat_seconds=0.01)


@pytest.mark.parametrize(
    "runtime_factory",
    [_langgraph_runtime, _FakeRuntime],
    ids=["langgraph", "fake"],
)
async def test_agent_runtime_contract(runtime_factory: Any) -> None:
    """所有 `AgentRuntime` 实现必须满足的行为契约。"""
    runtime = runtime_factory()

    assert isinstance(runtime, AgentRuntime), "实现不满足 AgentRuntime 的结构化契约"

    for events in (
        [event async for event in runtime.run_turn(_SESSION, None)],
        [event async for event in runtime.resume(_SESSION, "approve")],
    ):
        assert events, "端口实现必须至少产出一个事件"
        assert all(isinstance(event, RuntimeEvent) for event in events)
        assert all(isinstance(event.payload, dict) for event in events)

        kinds = {event.kind for event in events}
        assert kinds <= SIX_KINDS, f"出现契约外 kind：{kinds - SIX_KINDS}"
        assert "ping" not in kinds, "ping 是 SSE 心跳，不得编码成 RuntimeEvent"
        assert "final" in kinds, "正常收尾必须产出 final"
