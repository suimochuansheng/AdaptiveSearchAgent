"""resume 路由的终止路径（reject 分档语义 v2，2026-11-24）。

`dangerous` 级工具被拒 -> 图在 tools 之后直接 END，SSE 只有一条 final。
本用例守两条**状态机**不变量（不改任何状态码）：

1. 终止路径结束后 `task_states` 必须回到 `completed` / `idle`——否则会话停在
   `interrupted`，用户下一条消息会拿 409；
2. Redis 会话锁必须释放——否则同一 thread 的下一次请求拿 429。

不连真实 Redis / LangGraph：图、锁、状态写入都用假对象替换。
"""

import json
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.adapters.api import hitl


class _FakeLock:
    """只记录释放次数。"""

    def __init__(self) -> None:
        self.released = 0

    async def release(self) -> None:
        self.released += 1


class _FakeRequest:
    """`is_disconnected` 恒 False：模拟客户端全程在线。"""

    async def is_disconnected(self) -> bool:
        return False


class _FakeGraph:
    """回放"拒绝后 tools 收尾"的最终快照（接口只需 astream + aget_state）。"""

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    async def aget_state(self, config: Any = None) -> Any:
        # 历史里已有 tool_call：checkpoint 回放时不该再补发 tool_call 帧
        return SimpleNamespace(values={"messages": self._state["messages"][:2]})

    async def astream(  # type: ignore[no-untyped-def]
        self, inputs: Any, config: Any = None, stream_mode: Any = None
    ):
        yield "values", self._state


def _rejected_state(tool: str = "deep_research") -> dict[str, Any]:
    """本轮被用户拒绝的最终快照（ToolMessage 内容 = 执行器拒绝分支的回执）。"""
    call = {"name": tool, "args": {"query": "q"}, "id": "call-r"}
    return {
        "messages": [
            HumanMessage("帮我调研一下"),
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(
                content="用户拒绝执行该工具，本次未调用。",
                tool_call_id="call-r",
                name=tool,
            ),
        ],
        "last_tool_outcome": "rejected_by_user",
    }


async def test_dangerous_reject_completes_task_and_releases_lock(monkeypatch) -> None:
    """危险工具被拒：一条 final、零 token，task_states=completed，锁释放（无 409/429）。"""
    written: list[str] = []

    async def _fake_set_state(thread_id: str, state: str, *, interrupt_time: Any = None) -> None:
        written.append(state)

    monkeypatch.setattr(hitl, "_set_task_state", _fake_set_state)
    lock = _FakeLock()

    frames = [
        frame
        async for frame in hitl._resume_event_stream(
            _FakeRequest(),  # type: ignore[arg-type]
            _FakeGraph(_rejected_state()),
            {},
            "tid-reject",
            "reject",
            lock,  # type: ignore[arg-type]
        )
    ]

    events = [json.loads(frame[len("data: ") :]) for frame in frames]
    assert [e["type"] for e in events] == ["final"], f"终止路径的帧序列不对: {events}"
    assert events[0]["content"] == "已拒绝执行 deep_research，本轮未产生内容"
    # 状态机：不在 interrupted / running 停留，否则下一条消息 409/429
    assert written and written[-1] == "completed", f"终止后 task_states 未回收: {written}"
    assert "interrupted" not in written, f"终止路径不应再写 interrupted: {written}"
    assert lock.released == 1, f"Redis 锁未释放: {lock.released}"
