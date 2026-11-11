"""G7/T9-2 外层 HITL 自测：10 条验收 + 关键不变量（真实 Redis / 进程内 ASGI）。

运行：
    poetry run python "tests/test_agent_cases/T92-外层HITL/t92_checks.py"

依赖：Redis（`settings.REDIS_URL`，验收用本机 16379）。不依赖 Postgres / Ollama：
图用 `InMemorySaver` + 假工具，路由用 `httpx.ASGITransport` 在进程内调用。

覆盖（对应交接单第四节）：
    A1  开关关：与 G6 一致（无 interrupt、G5 拒绝文案、resume 404）
    A2  开关开 + 危险工具：interrupt 帧、八字段、32 位 hex
    A3  approve：200、工具恰好 1 次、流到 final
    A4  reject：工具 0 次、可读理由、不抛栈
    A5  同一 approval_id 再发：409
    A6  伪造 approval_id：404
    A7  TTL=1s，等 2s：410
    A8  非危险工具：零审批
    A9  并发两个 resume：一个 200 一个 409，工具仍 1 次
    A10 追踪：挂起 span=approval_required；拒绝 span=rejected_by_user
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import httpx
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from redis import asyncio as aioredis

from agent.adapters.api import hitl
from agent.adapters.stores import approval_store
from agent.adapters.tools import executor
from agent.adapters.tools.budget_memory import MemoryBudgetLedger
from agent.domain.tool import CostModel, ToolSpec
from agent.observability.langchain_handler import build_callbacks
from agent.observability.tracer import JsonlTracer
from agent.observability.turn import next_turn_index, turn_span
from config import settings
from src.conversation_stream import stream_conversation

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

APPROVAL_FIELDS = list(hitl.APPROVAL_FIELDS)
G5_REASON = "该工具需要人工审批后才能执行（HITL 未启用，本次已拒绝）"
REJECT_REASON = "用户拒绝执行该工具，本次未调用。"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}")


# ── 假工具与假图 ─────────────────────────────────────────────────────


@dataclass
class Counters:
    danger: int = 0
    safe: int = 0


COUNTERS = Counters()


async def danger(query: str) -> str:
    """危险工具（自测用）：执行一次计数一次。"""
    COUNTERS.danger += 1
    return f"danger executed: {query}"


async def safe(expr: str) -> str:
    """普通只读工具（自测用）：不应触发审批。"""
    COUNTERS.safe += 1
    return f"safe executed: {expr}"


def _spec(name: str, *, dangerous: bool) -> ToolSpec:
    if dangerous:
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        permission, cost, timeout, approval = "dangerous", 0.02, 5.0, True
    else:
        schema = {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        }
        permission, cost, timeout, approval = "read", 0.0, 5.0, False
    return ToolSpec(
        name=name,
        description=f"{name} 自测工具",
        input_schema=schema,
        permission=permission,  # type: ignore[arg-type]
        cost_model=CostModel(unit="call", unit_cost=cost),
        timeout_seconds=timeout,
        max_output_tokens=100,
        requires_approval=approval,
        feature_flag=None,
    )


DANGER_WRAPPED: Any = None
SAFE_WRAPPED: Any = None


def configure_tools() -> None:
    """`wrap_tool` 是否包装取决于 ENABLE_TOOL_POLICY，必须在 settings 设好后再包。"""
    global DANGER_WRAPPED, SAFE_WRAPPED
    DANGER_WRAPPED = executor.wrap_tool(danger, _spec("danger", dangerous=True))
    SAFE_WRAPPED = executor.wrap_tool(safe, _spec("safe", dangerous=False))


async def agent_node(state: MessagesState) -> dict[str, Any]:
    """收到工具结果就收尾；否则按用户消息选一个工具调用（不依赖 LLM）。"""
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content=f"完成：{last.content}")]}
    text = " ".join(
        str(getattr(m, "content", "")) for m in state["messages"] if isinstance(m, HumanMessage)
    )
    if "danger" in text:
        call = {
            "name": "danger",
            "args": {"query": text},
            "id": "call_danger_1",
            "type": "tool_call",
        }
    else:
        call = {
            "name": "safe",
            "args": {"expr": "1+1"},
            "id": "call_safe_1",
            "type": "tool_call",
        }
    return {"messages": [AIMessage(content="", tool_calls=[call])]}


async def tools_node(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """与生产 `conversation_graph.tools_node` 同款：用 tool_scope 包住 ToolNode。"""
    thread_id = str((config.get("configurable") or {}).get("thread_id", "default"))
    with executor.tool_scope(thread_id):
        return await ToolNode([DANGER_WRAPPED, SAFE_WRAPPED]).ainvoke(state)


def build_graph() -> Any:
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=InMemorySaver())


# ── 测试夹具 ─────────────────────────────────────────────────────────


class Harness:
    """一套图 + ASGI app + Redis 锁；每个场景用独立 thread_id。"""

    def __init__(self) -> None:
        configure_tools()
        self.graph = build_graph()
        self.app = FastAPI()
        self.app.include_router(hitl.router)
        self.redis: Any = None
        self._prev_load = hitl._load_graph
        self._prev_lock = hitl._acquire_thread_lock

    async def __aenter__(self) -> Harness:
        self.redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        graph = self.graph

        async def _load() -> Any:
            return graph

        async def _acquire(thread_id: str) -> Any | None:
            lock = self.redis.lock(f"agent_lock:{thread_id}", timeout=600)
            return lock if await lock.acquire(blocking=False) else None

        hitl._load_graph = _load  # type: ignore[assignment]
        hitl._acquire_thread_lock = _acquire  # type: ignore[assignment]
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.client.aclose()
        await self.redis.aclose()
        hitl._load_graph = self._prev_load  # type: ignore[assignment]
        hitl._acquire_thread_lock = self._prev_lock  # type: ignore[assignment]

    def thread(self, name: str) -> str:
        return f"t92-{name}-{uuid.uuid4().hex[:10]}"

    async def first_turn(
        self, thread_id: str, text: str, callbacks: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if callbacks:
            config["callbacks"] = callbacks
        frames = [
            frame
            async for frame in stream_conversation(
                self.graph,
                {"messages": [HumanMessage(content=text)]},
                config,
                heartbeat_seconds=5.0,
            )
        ]
        return [_parse(frame) for frame in frames]

    async def resume(
        self, thread_id: str, approval_id: str, decision: str, request_id: str
    ) -> httpx.Response:
        return await self.client.post(
            f"/chat/conversation/{thread_id}/resume",
            json={
                "approval_id": approval_id,
                "decision": decision,
                "request_id": request_id,
            },
        )


def _parse(frame: str) -> dict[str, Any]:
    assert frame.startswith("data: "), frame
    return json.loads(frame[len("data: ") :].strip())


def _interrupt(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((e for e in events if e.get("type") == "interrupt"), None)


def _final(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((e for e in events if e.get("type") == "final"), None)


def _sse_events(response: httpx.Response) -> list[dict[str, Any]]:
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :].strip()))
    return events


# ── A1：开关关 ───────────────────────────────────────────────────────


async def a1_switch_off(h: Harness) -> None:
    settings.ENABLE_HITL = False
    settings.ENABLE_TOOL_POLICY = True
    thread = h.thread("a1")
    events = await h.first_turn(thread, "danger please")
    types = [e.get("type") for e in events]
    final = _final(events) or {}
    ok = (
        types == ["tool_call", "final"]
        and final.get("content", "").endswith(G5_REASON)
        and COUNTERS.danger == 0
    )
    check(
        "A1 开关关：无 interrupt + G5 拒绝文案 + 工具 0 次",
        ok,
        f"frames={types}, danger={COUNTERS.danger}",
    )

    # 关闭时 resume 路由必须 404（即使凭据形似合法）
    resp = await h.resume(thread, "0" * 32, "approve", f"rid-{uuid.uuid4().hex}")
    check("A1b 开关关：resume 路由 404", resp.status_code == 404, f"status={resp.status_code}")


# ── A2 / A3：开关开 + approve ────────────────────────────────────────


async def a2_a3_approve(h: Harness) -> None:
    settings.ENABLE_HITL = True
    settings.ENABLE_TOOL_POLICY = True
    COUNTERS.danger = 0
    thread = h.thread("a2")
    events = await h.first_turn(thread, "danger please")
    types = [e.get("type") for e in events]
    interrupt = _interrupt(events)
    data = (interrupt or {}).get("data") or {}
    hex_ok = len(str(data.get("approval_id", ""))) == 32 and all(
        c in "0123456789abcdef" for c in str(data.get("approval_id", ""))
    )
    fields_ok = list(data) == APPROVAL_FIELDS
    detail = f"frames={types}, fields={list(data)}, hex32={hex_ok}"
    check(
        "A2 危险工具：interrupt 帧 + 八字段齐全 + 32 位 hex",
        bool(interrupt) and fields_ok and hex_ok,
        detail,
    )
    check("A2b 挂起时工具 0 次", COUNTERS.danger == 0, f"danger={COUNTERS.danger}")
    if not interrupt:
        return

    resp = await h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}")
    resumed = _sse_events(resp)
    check(
        "A3 approve：200 + 工具恰好 1 次 + 流到 final",
        resp.status_code == 200 and _final(resumed) is not None and COUNTERS.danger == 1,
        f"status={resp.status_code}, danger={COUNTERS.danger}, frames={[e.get('type') for e in resumed]}",
    )

    # A5：同一 approval_id 再发 -> 409
    again = await h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}")
    check("A5 同一 approval_id 再发：409", again.status_code == 409, f"status={again.status_code}")

    # A6：伪造 -> 404
    forged = await h.resume(thread, "f" * 32, "approve", f"rid-{uuid.uuid4().hex}")
    check("A6 伪造 approval_id：404", forged.status_code == 404, f"status={forged.status_code}")


# ── A4：reject ───────────────────────────────────────────────────────


async def a4_reject(h: Harness) -> None:
    settings.ENABLE_HITL = True
    COUNTERS.danger = 0
    thread = h.thread("a4")
    events = await h.first_turn(thread, "danger please")
    data = (_interrupt(events) or {}).get("data") or {}
    if not data:
        check("A4 reject", False, "没有拿到 interrupt 帧")
        return
    resp = await h.resume(thread, data["approval_id"], "reject", f"rid-{uuid.uuid4().hex}")
    resumed = _sse_events(resp)
    final = _final(resumed) or {}
    ok = (
        resp.status_code == 200
        and COUNTERS.danger == 0
        and REJECT_REASON in final.get("content", "")
        and not any(e.get("type") == "error" for e in resumed)
    )
    check(
        "A4 reject：工具 0 次 + 可读理由 + 不抛栈",
        ok,
        f"status={resp.status_code}, danger={COUNTERS.danger}, final={final.get('content', '')[:40]!r}",
    )


# ── A7：TTL 过期 410 ─────────────────────────────────────────────────


async def a7_expired(h: Harness) -> None:
    settings.ENABLE_HITL = True
    old_ttl = settings.APPROVAL_TTL_SECONDS
    settings.APPROVAL_TTL_SECONDS = 1
    approval_store.set_default_store(approval_store.ApprovalStore())
    try:
        thread = h.thread("a7")
        events = await h.first_turn(thread, "danger please")
        data = (_interrupt(events) or {}).get("data") or {}
        if not data:
            check("A7 过期：410", False, "没有拿到 interrupt 帧")
            return
        await asyncio.sleep(2.1)
        resp = await h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}")
        check(
            "A7 TTL=1s 等 2s：410",
            resp.status_code == 410,
            f"status={resp.status_code}, body={resp.text[:80]}",
        )
    finally:
        settings.APPROVAL_TTL_SECONDS = old_ttl
        approval_store.set_default_store(approval_store.ApprovalStore())


# ── A8：非危险工具零审批 ─────────────────────────────────────────────


async def a8_safe_tool(h: Harness) -> None:
    settings.ENABLE_HITL = True
    COUNTERS.safe = 0
    thread = h.thread("a8")
    events = await h.first_turn(thread, "safe please")
    types = [e.get("type") for e in events]
    ok = "interrupt" not in types and _final(events) is not None and COUNTERS.safe == 1
    check("A8 非危险工具：零审批且正常执行", ok, f"frames={types}, safe={COUNTERS.safe}")


# ── A9：并发 resume ──────────────────────────────────────────────────


async def a9_concurrent(h: Harness) -> None:
    settings.ENABLE_HITL = True
    COUNTERS.danger = 0
    thread = h.thread("a9")
    events = await h.first_turn(thread, "danger please")
    data = (_interrupt(events) or {}).get("data") or {}
    if not data:
        check("A9 并发 resume", False, "没有拿到 interrupt 帧")
        return
    first, second = await asyncio.gather(
        h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}"),
        h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}"),
    )
    statuses = sorted([first.status_code, second.status_code])
    ok = statuses == [200, 409] and COUNTERS.danger == 1
    check(
        "A9 并发两个 resume：一个 200 一个 409，工具仍 1 次",
        ok,
        f"statuses={statuses}, danger={COUNTERS.danger}",
    )


# ── A10：追踪标记 ────────────────────────────────────────────────────


async def a10_trace(h: Harness) -> None:
    settings.ENABLE_HITL = True
    settings.ENABLE_TRACE = True
    thread = h.thread("a10")
    with tempfile.TemporaryDirectory() as td:
        tracer = JsonlTracer(td)
        callbacks = build_callbacks(tracer)
        config = {"configurable": {"thread_id": thread}, "callbacks": callbacks}
        async with turn_span(tracer, thread, await next_turn_index(h.graph, config)):
            events = [
                _parse(frame)
                async for frame in stream_conversation(
                    h.graph,
                    {"messages": [HumanMessage(content="danger please")]},
                    config,
                    heartbeat_seconds=5.0,
                )
            ]
        data = (_interrupt(events) or {}).get("data") or {}

        tracer2 = JsonlTracer(td)
        config["callbacks"] = build_callbacks(tracer2)
        if data:
            with executor.approval_resume_scope():
                async with turn_span(tracer2, thread, await next_turn_index(h.graph, config)):
                    async for _frame in stream_conversation(
                        h.graph,
                        Command(resume="reject"),
                        config,
                        heartbeat_seconds=5.0,
                    ):
                        pass

        log = Path(td) / f"{date.today().isoformat()}.jsonl"
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        kinds = [
            r["attrs"].get("error_kind")
            for r in records
            if r["kind"] == "tool" and r["session_id"] == thread
        ]
        ok = "approval_required" in kinds and "rejected_by_user" in kinds
        check(
            "A10 追踪：挂起=approval_required / 拒绝=rejected_by_user",
            ok,
            f"tool error_kind={kinds}",
        )
    settings.ENABLE_TRACE = False


# ── 额外：thread 不匹配 409（契约） ──────────────────────────────────


async def extra_thread_mismatch(h: Harness) -> None:
    settings.ENABLE_HITL = True
    thread_a = h.thread("mismatch-a")
    thread_b = h.thread("mismatch-b")
    events = await h.first_turn(thread_a, "danger please")
    data = (_interrupt(events) or {}).get("data") or {}
    if not data:
        check("EXTRA thread 不匹配", False, "没有拿到 interrupt 帧")
        return
    resp = await h.resume(thread_b, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}")
    check("EXTRA 凭据换 thread 使用：409", resp.status_code == 409, f"status={resp.status_code}")


async def main() -> int:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    approval_store.set_default_store(approval_store.ApprovalStore())
    executor.set_default_ledger(MemoryBudgetLedger())

    async with Harness() as h:
        await a1_switch_off(h)
        await a2_a3_approve(h)
        await a4_reject(h)
        await a7_expired(h)
        await a8_safe_tool(h)
        await a9_concurrent(h)
        await a10_trace(h)
        await extra_thread_mismatch(h)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print(f"\nRESULT: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
