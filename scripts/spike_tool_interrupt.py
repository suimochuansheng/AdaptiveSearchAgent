"""G7/T9-2 限时 spike：工具函数内 `interrupt()` 与 `Command(resume=...)` 的可行性。

只回答三个问题，不接入业务代码：

1. 在工具函数里调用 `langgraph.types.interrupt(payload)`，用 `ToolNode` 执行时，
   `astream`（生产帧循环用的 `["messages", "values"]`）里能否观察到 `__interrupt__`？
2. 拿到中断后，用 `Command(resume=值)` 重新 invoke 同一 `thread_id`，能否从断点继续、
   并让 `interrupt(...)` 返回那个值？
3. 中断期间，`src/conversation_stream.py` 的现有帧循环会不会卡死或抛异常？

运行：`poetry run python scripts/spike_tool_interrupt.py`
输出为 UTF-8；退出码 0 表示脚本跑完（结论由输出判定，不在此脚本里 assert）。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

# 允许 `poetry run python scripts/spike_tool_interrupt.py` 直接跑：把仓库根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

# 观测点：工具执行次数 / interrupt 收到的恢复值 / 中断载荷
EXECUTED = {"n": 0}
RESUME_VALUE = {"value": None}


async def danger_tool(query: str) -> str:
    """假的危险工具：先挂起等审批，恢复后按决定执行（仅用于 spike）。"""
    payload = {
        "approval_id": "0123456789abcdef0123456789abcdef",
        "tool_call_id": "call_spike_1",
        "tool": "danger_tool",
        "args_digest": f"query={query}",
        "cost_estimate": 0.02,
        "budget_remaining": 1.98,
        "permission": "dangerous",
        "expires_at": "2026-11-11T21:00:00+08:00",
    }
    decision = interrupt(payload)
    RESUME_VALUE["value"] = decision
    EXECUTED["n"] += 1
    return f"danger_tool 已执行，decision={decision}"


async def agent_node(state: MessagesState) -> dict[str, Any]:
    """固定发起一次 danger_tool 调用（不依赖真实 LLM）。"""
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "danger_tool",
                        "args": {"query": "spike"},
                        "id": "call_spike_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }


def build_graph() -> Any:
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([danger_tool]))
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "tools")
    graph.add_edge("tools", END)
    return graph.compile(checkpointer=InMemorySaver())


def _describe(chunk: Any) -> str:
    """把 stream chunk 压成一行摘要，避免把整段状态打出来。"""
    if not isinstance(chunk, tuple) or len(chunk) != 2:
        return f"<非 (mode, payload) 元组：{type(chunk).__name__}>"
    mode, payload = chunk
    if isinstance(payload, dict):
        keys = list(payload)
        if "__interrupt__" in payload:
            return f"mode={mode!r} dict keys={keys} __interrupt__={payload['__interrupt__']!r}"
        return f"mode={mode!r} dict keys={keys}"
    return f"mode={mode!r} payload_type={type(payload).__name__} value={payload!r}"


async def q1_observe_interrupt() -> dict[str, Any]:
    """问题 1：生产帧循环的 stream_mode 下能否看到 __interrupt__。"""
    print("\n=== Q1：astream 里观察 __interrupt__ ===")
    graph = build_graph()
    config = {"configurable": {"thread_id": "spike-q1"}}
    inputs = {"messages": [HumanMessage(content="开始")]}

    seen: list[str] = []
    interrupt_modes: list[str] = []
    try:
        async for mode, payload in graph.astream(
            inputs, config=config, stream_mode=["messages", "values"]
        ):
            seen.append(mode)
            if mode == "__interrupt__" or (
                isinstance(payload, dict) and "__interrupt__" in payload
            ):
                interrupt_modes.append(mode)
                print("  中断 chunk:", _describe((mode, payload)))
            elif mode == "values" and isinstance(payload, dict):
                print("  values keys:", list(payload))
    except Exception as exc:  # noqa: BLE001 - spike 需要把异常原样报出来
        print(f"  读取流时抛异常：{type(exc).__name__}: {exc}")
        return {"seen": seen, "interrupt": False, "error": repr(exc)}

    print("  观察到的 stream mode 序列：", seen)
    print("  是否观察到中断：", bool(interrupt_modes), "| 命中的 mode：", interrupt_modes)
    return {"seen": seen, "interrupt": bool(interrupt_modes)}


async def q2_resume() -> dict[str, Any]:
    """问题 2：Command(resume=...) 能否续跑并让 interrupt 返回该值。"""
    print("\n=== Q2：Command(resume=...) 续跑 ===")
    global EXECUTED
    EXECUTED = {"n": 0}
    RESUME_VALUE["value"] = None

    graph = build_graph()
    config = {"configurable": {"thread_id": "spike-q2"}}
    inputs = {"messages": [HumanMessage(content="开始")]}

    # 先跑到中断
    async for _mode, _payload in graph.astream(
        inputs, config=config, stream_mode=["messages", "values"]
    ):
        pass
    print(f"  中断后执行次数：{EXECUTED['n']}（应为 0）")

    state = await graph.aget_state(config)
    print("  中断后 next 节点：", state.next)

    try:
        result = await graph.ainvoke(Command(resume="approve"), config=config)
    except Exception as exc:  # noqa: BLE001
        print(f"  resume 抛异常：{type(exc).__name__}: {exc}")
        return {"ok": False, "error": repr(exc)}

    last = result["messages"][-1]
    print(f"  恢复后执行次数：{EXECUTED['n']}（应为 1）")
    print(f"  interrupt 收到的值：{RESUME_VALUE['value']!r}")
    print(f"  最后一条消息：{type(last).__name__} content={last.content!r}")

    ok = (
        EXECUTED["n"] == 1 and RESUME_VALUE["value"] == "approve" and "approve" in str(last.content)
    )
    print("  结论：", "可以从断点续跑，interrupt 返回 resume 值" if ok else "未按预期续跑")
    return {"ok": ok, "executed": EXECUTED["n"], "resume": RESUME_VALUE["value"]}


async def q3_existing_loop() -> dict[str, Any]:
    """问题 3：现有 conversation_stream 帧循环遇到中断会不会卡死 / 抛异常。"""
    print("\n=== Q3：现有 conversation_stream 帧循环 ===")
    graph = build_graph()
    config = {"configurable": {"thread_id": "spike-q3"}}
    inputs = {"messages": [HumanMessage(content="开始")]}

    from src.conversation_stream import ConversationStreamError, stream_conversation

    frames: list[str] = []
    error: BaseException | None = None
    started = time.monotonic()
    try:
        async for frame in stream_conversation(graph, inputs, config, heartbeat_seconds=1.0):
            frames.append(frame)
            if len(frames) <= 10:
                print("  帧：", frame.strip())
    except ConversationStreamError as exc:
        error = exc
        print(f"  帧循环抛 ConversationStreamError：{exc}")
    except Exception as exc:  # noqa: BLE001
        error = exc
        print(f"  帧循环抛异常：{type(exc).__name__}: {exc}")
    elapsed = time.monotonic() - started

    types = []
    for frame in frames:
        try:
            types.append(json.loads(frame.removeprefix("data: ").strip())["type"])
        except Exception:  # noqa: BLE001
            types.append("?")
    print(f"  用时：{elapsed:.2f}s（未卡死）| 帧数：{len(frames)} | 帧类型：{types}")
    print("  是否出现 interrupt 帧：", "interrupt" in types)
    return {
        "frames": len(frames),
        "types": types,
        "error": repr(error) if error else None,
        "elapsed": round(elapsed, 2),
    }


async def main() -> None:
    print("langgraph:", version("langgraph"))
    print("langchain-core:", version("langchain-core"))
    results = {
        "q1": await q1_observe_interrupt(),
        "q2": await q2_resume(),
        "q3": await q3_existing_loop(),
    }
    print("\n=== 汇总 ===")
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
