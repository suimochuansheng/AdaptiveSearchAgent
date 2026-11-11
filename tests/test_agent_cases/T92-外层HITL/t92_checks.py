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
import os
import subprocess
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
#: C 组（边界/反例）只记 open_issue，不计入 RESULT 与退出码
C_RESULTS: list[tuple[str, bool, str]] = []

REPO_ROOT = Path(__file__).resolve().parents[3]


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}")


def c_check(name: str, ok: bool, detail: str) -> None:
    """C 组证据行：格式与 A 组一致，但不参与退出码（open_issue 不判失败）。"""
    C_RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}")


def _is_hex32(value: str) -> bool:
    return len(value) == 32 and all(ch in "0123456789abcdef" for ch in value)


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


# ══════════════════════════════════════════════════════════════════════
# test-G7 独立补测（T92-101 ~ T92-114；不改上面开发写的 13 项）
# ══════════════════════════════════════════════════════════════════════


async def t92_101_switch_off(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = False
    COUNTERS.danger = 0
    thread = h.thread("101")
    events = await h.first_turn(thread, "danger please")
    types = [e.get("type") for e in events]
    final = _final(events) or {}
    resp = await h.resume(thread, "0" * 32, "approve", f"rid-{uuid.uuid4().hex}")
    ok = (
        types == ["tool_call", "final"]
        and final.get("content", "").endswith(G5_REASON)
        and COUNTERS.danger == 0
        and resp.status_code in (404, 405)
    )
    check(
        "T92-101 A1 开关关：帧序列同改造前 + 审批路由不可达",
        ok,
        f"frames={types}, danger={COUNTERS.danger}, resume={resp.status_code}",
    )


async def t92_102_interrupt_frame(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    COUNTERS.danger = 0
    thread = h.thread("102")
    events = await h.first_turn(thread, "danger please")
    intr = _interrupt(events)
    data = (intr or {}).get("data") or {}
    frame_keys = sorted(intr) if intr else []
    extra = sorted(set(data) - set(APPROVAL_FIELDS))
    missing = sorted(set(APPROVAL_FIELDS) - set(data))
    ok = (
        bool(intr)
        and frame_keys == ["data", "type"]
        and len(data) == 8
        and not extra
        and not missing
        and COUNTERS.danger == 0
    )
    check(
        "T92-102 A2 危险工具：interrupt 帧 + data 恰好八字段且无多余键",
        ok,
        f"frame_keys={frame_keys}, fields={list(data)}, extra={extra}, "
        f"missing={missing}, danger={COUNTERS.danger}",
    )


async def t92_103_approval_id(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    ids: list[str] = []
    for index in range(2):
        thread = h.thread(f"103-{index}")
        events = await h.first_turn(thread, "danger please")
        data = (_interrupt(events) or {}).get("data") or {}
        ids.append(str(data.get("approval_id", "")))
    ok = all(_is_hex32(value) for value in ids) and len(set(ids)) == 2
    check(
        "T92-103 A3 approval_id 为 32 位十六进制且每次不同",
        ok,
        f"ids={ids}, hex32={[_is_hex32(v) for v in ids]}, distinct={len(set(ids))}",
    )


async def t92_104_approve(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    COUNTERS.danger = 0
    thread = h.thread("104")
    events = await h.first_turn(thread, "danger please")
    data = (_interrupt(events) or {}).get("data") or {}
    if not data:
        check("T92-104 A4 approve", False, "没有拿到 interrupt 帧")
        return
    resp = await h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}")
    resumed = _sse_events(resp)
    ok = (
        resp.status_code == 200
        and COUNTERS.danger == 1
        and _final(resumed) is not None
        and not any(e.get("type") == "error" for e in resumed)
    )
    check(
        "T92-104 A4 approve：200 + 工具恰好 1 次 + 流到 final",
        ok,
        f"status={resp.status_code}, danger={COUNTERS.danger}, "
        f"frames={[e.get('type') for e in resumed]}",
    )


async def t92_105_reject(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    COUNTERS.danger = 0
    thread = h.thread("105")
    events = await h.first_turn(thread, "danger please")
    data = (_interrupt(events) or {}).get("data") or {}
    if not data:
        check("T92-105 A5 reject", False, "没有拿到 interrupt 帧")
        return
    resp = await h.resume(thread, data["approval_id"], "reject", f"rid-{uuid.uuid4().hex}")
    resumed = _sse_events(resp)
    final = _final(resumed) or {}
    body = resp.text
    ok = (
        resp.status_code == 200
        and COUNTERS.danger == 0
        and REJECT_REASON in final.get("content", "")
        and "Traceback" not in body
        and not any(e.get("type") == "error" for e in resumed)
    )
    check(
        "T92-105 A5 reject：工具 0 次 + 可读理由 + 无 Traceback",
        ok,
        f"status={resp.status_code}, danger={COUNTERS.danger}, "
        f"final={final.get('content', '')[:40]!r}, traceback={'Traceback' in body}",
    )


async def t92_106_replay_forge_mismatch(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    COUNTERS.danger = 0
    thread_a = h.thread("106-a")
    thread_b = h.thread("106-b")
    events = await h.first_turn(thread_a, "danger please")
    data = (_interrupt(events) or {}).get("data") or {}
    if not data:
        check("T92-106 A6 重放/伪造/换 thread", False, "没有拿到 interrupt 帧")
        return
    aid = data["approval_id"]
    mismatch = await h.resume(thread_b, aid, "approve", f"rid-{uuid.uuid4().hex}")
    approved = await h.resume(thread_a, aid, "approve", f"rid-{uuid.uuid4().hex}")
    replay = await h.resume(thread_a, aid, "approve", f"rid-{uuid.uuid4().hex}")
    forged = await h.resume(thread_a, "f" * 32, "approve", f"rid-{uuid.uuid4().hex}")
    ok = (
        mismatch.status_code == 409
        and approved.status_code == 200
        and replay.status_code == 409
        and forged.status_code == 404
        and COUNTERS.danger == 1
    )
    check(
        "T92-106 A6 重放 409 / 伪造 404 / 换 thread 409",
        ok,
        f"other_thread={mismatch.status_code}, approve={approved.status_code}, "
        f"replay={replay.status_code}, forged={forged.status_code}, danger={COUNTERS.danger}",
    )


async def t92_107_expired(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    old_ttl = settings.APPROVAL_TTL_SECONDS
    settings.APPROVAL_TTL_SECONDS = 1
    approval_store.set_default_store(approval_store.ApprovalStore())
    try:
        thread = h.thread("107")
        events = await h.first_turn(thread, "danger please")
        data = (_interrupt(events) or {}).get("data") or {}
        if not data:
            check("T92-107 A7 TTL 过期", False, "没有拿到 interrupt 帧")
            return
        await asyncio.sleep(2.1)
        resp = await h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}")
        check(
            "T92-107 A7 TTL=1s 等待 2s：410",
            resp.status_code == 410,
            f"status={resp.status_code}, body={resp.text[:80]}",
        )
    finally:
        settings.APPROVAL_TTL_SECONDS = old_ttl
        approval_store.set_default_store(approval_store.ApprovalStore())


async def t92_108_concurrent_same_id(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    COUNTERS.danger = 0
    thread = h.thread("108")
    events = await h.first_turn(thread, "danger please")
    data = (_interrupt(events) or {}).get("data") or {}
    if not data:
        check("T92-108 A8 并发同 approval_id", False, "没有拿到 interrupt 帧")
        return
    first, second = await asyncio.gather(
        h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}"),
        h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}"),
    )
    statuses = sorted([first.status_code, second.status_code])
    ok = statuses == [200, 409] and COUNTERS.danger == 1
    check(
        "T92-108 A8 并发两次同 approval_id：一个 200 一个 409，工具仍 1 次",
        ok,
        f"statuses={statuses}, danger={COUNTERS.danger}",
    )


async def t92_109_safe_tool(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    COUNTERS.safe = 0
    thread = h.thread("109")
    events = await h.first_turn(thread, "safe please")
    types = [e.get("type") for e in events]
    ok = "interrupt" not in types and _final(events) is not None and COUNTERS.safe == 1
    check(
        "T92-109 A9 非危险工具：开关开着也零审批",
        ok,
        f"frames={types}, safe={COUNTERS.safe}",
    )


def _build_subgraph_interrupt() -> Any:
    """真子图：内部节点调用 interrupt({type: cost_confirm})，复刻 hitl_confirm 载荷。"""
    from src.agents.hitl_confirm import hitl_confirm
    from src.state import AgentState

    graph = StateGraph(AgentState)
    graph.add_node("hitl_confirm", hitl_confirm)
    graph.add_edge(START, "hitl_confirm")
    graph.add_edge("hitl_confirm", END)
    return graph.compile(checkpointer=InMemorySaver())


def _build_parent_with_subgraph(subgraph: Any) -> Any:
    async def agent_node(state: MessagesState, config):  # noqa: ANN001 - LangGraph 按名注入
        await subgraph.ainvoke(
            {
                "user_query": "q",
                "plan": ["a", "b", "c", "d", "e"],
                "search_results": [],
                "iteration": 0,
                "pending_keywords": [],
                "retry_keywords": [],
            },
            config,
        )
        return {"messages": [AIMessage(content="子图已返回")]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=InMemorySaver())


async def t92_110_subgraph_interrupt(h: Harness) -> None:
    """构造一次真实子图中断；外层流不应产出 interrupt 帧，更不应弹审批卡。"""
    parent = _build_parent_with_subgraph(_build_subgraph_interrupt())
    thread = h.thread("110")
    frames = [
        _parse(frame)
        async for frame in stream_conversation(
            parent,
            {"messages": [HumanMessage(content="go")]},
            {"configurable": {"thread_id": thread}},
            heartbeat_seconds=5.0,
        )
    ]
    types = [frame.get("type") for frame in frames]
    snapshot = await parent.aget_state({"configurable": {"thread_id": thread}})
    interrupts = tuple(getattr(snapshot, "interrupts", ()) or ())
    values = [getattr(item, "value", None) for item in interrupts]
    subgraph_only = any(
        isinstance(value, dict)
        and value.get("type") == "cost_confirm"
        and "approval_id" not in value
        for value in values
    )
    ok = "interrupt" not in types and subgraph_only
    check(
        "T92-110 A10 子图中断留在子图：外层不消费、不弹外层审批",
        ok,
        f"outer_frames={types}, subgraph_interrupt={values}, next={snapshot.next}",
    )


class AppHarness:
    """真实 `/chat/conversation` + 真实 task_states（Postgres）+ 假图，用于状态机验收。"""

    def __init__(self) -> None:
        configure_tools()
        self.graph = build_graph()
        self.threads: list[str] = []
        self.api: Any = None
        self._prev: dict[str, Any] = {}

    async def __aenter__(self) -> AppHarness:
        import api_main
        from src.checkpointer import init_checkpointer

        self.api = api_main
        await init_checkpointer()
        await api_main.ensure_task_states_table()
        self._prev = {
            "graph": api_main.get_conversation_graph,
            "load": hitl._load_graph,
            "lock": hitl._acquire_thread_lock,
            "trace": settings.ENABLE_TRACE,
            "hitl": settings.ENABLE_HITL,
            "policy": settings.ENABLE_TOOL_POLICY,
        }
        settings.ENABLE_TRACE = False
        settings.ENABLE_HITL = True
        settings.ENABLE_TOOL_POLICY = True
        graph = self.graph

        async def _load() -> Any:
            return graph

        async def _acquire(thread_id: str) -> Any | None:
            lock = api_main.redis_client.lock(f"agent_lock:{thread_id}", timeout=600)
            return lock if await lock.acquire(blocking=False) else None

        api_main.get_conversation_graph = _load  # type: ignore[assignment]
        hitl._load_graph = _load  # type: ignore[assignment]
        hitl._acquire_thread_lock = _acquire  # type: ignore[assignment]
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_main.app), base_url="http://test"
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.client.aclose()
        api_main = self.api
        api_main.get_conversation_graph = self._prev["graph"]  # type: ignore[assignment]
        hitl._load_graph = self._prev["load"]  # type: ignore[assignment]
        hitl._acquire_thread_lock = self._prev["lock"]  # type: ignore[assignment]
        settings.ENABLE_TRACE = self._prev["trace"]
        settings.ENABLE_HITL = self._prev["hitl"]
        settings.ENABLE_TOOL_POLICY = self._prev["policy"]
        configure_tools()
        try:
            pool = api_main.get_business_pool()
            if pool is not None:
                async with pool.connection() as conn:
                    for thread_id in self.threads:
                        await conn.execute(
                            "DELETE FROM task_states WHERE thread_id = %s", (thread_id,)
                        )
        except Exception as exc:  # noqa: BLE001 - 清理失败不影响结论
            print(f"[INFO] AppHarness 清理 task_states 失败：{exc}")

    def thread(self, name: str) -> str:
        thread_id = f"t92-{name}-{uuid.uuid4().hex[:10]}"
        self.threads.append(thread_id)
        return thread_id

    async def chat(self, thread_id: str, text: str) -> httpx.Response:
        return await self.client.post(
            "/chat/conversation", json={"thread_id": thread_id, "message": text}
        )

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


async def t92_111_state_machine(ah: AppHarness) -> None:
    COUNTERS.danger = 0
    approve_thread = ah.thread("111-approve")
    approve_turn = await ah.chat(approve_thread, "danger please")
    approve_events = _sse_events(approve_turn)
    approve_data = (_interrupt(approve_events) or {}).get("data") or {}
    after_hang = await ah.api.get_task_state(approve_thread)
    approve_resume = await ah.resume(
        approve_thread, approve_data.get("approval_id", ""), "approve", f"rid-{uuid.uuid4().hex}"
    )
    after_approve = await ah.api.get_task_state(approve_thread)

    reject_thread = ah.thread("111-reject")
    reject_turn = await ah.chat(reject_thread, "danger please")
    reject_events = _sse_events(reject_turn)
    reject_data = (_interrupt(reject_events) or {}).get("data") or {}
    reject_hang = await ah.api.get_task_state(reject_thread)
    reject_resume = await ah.resume(
        reject_thread, reject_data.get("approval_id", ""), "reject", f"rid-{uuid.uuid4().hex}"
    )
    after_reject = await ah.api.get_task_state(reject_thread)

    ok = (
        bool(approve_data)
        and bool(reject_data)
        and after_hang == "interrupted"
        and approve_resume.status_code == 200
        and after_approve == "completed"
        and reject_hang == "interrupted"
        and reject_resume.status_code == 200
        and after_reject == "completed"
        and COUNTERS.danger == 1
    )
    check(
        "T92-111 A11 挂起=interrupted；approve/reject 后均=completed",
        ok,
        f"hang={after_hang}, approve={approve_resume.status_code}->{after_approve}, "
        f"reject={reject_resume.status_code}->{after_reject}, danger={COUNTERS.danger}",
    )


async def t92_112_interrupted_guard(ah: AppHarness) -> None:
    COUNTERS.danger = 0
    thread = ah.thread("112")
    turn = await ah.chat(thread, "danger please")
    events = _sse_events(turn)
    data = (_interrupt(events) or {}).get("data") or {}
    if not data:
        check("T92-112 A12 挂起态普通消息 409", False, "没有拿到 interrupt 帧")
        return
    before = COUNTERS.danger
    again = await ah.chat(thread, "再发一条普通消息")
    try:
        detail = str(again.json().get("detail", ""))
    except (ValueError, AttributeError):
        detail = again.text[:160]
    ok = (
        again.status_code == 409
        and again.status_code not in (429, 500)
        and "审批" in detail
        and COUNTERS.danger == before
    )
    check(
        "T92-112 A12 interrupted 时普通消息：409（非 429/500）且提示可读",
        ok,
        f"status={again.status_code}, detail={detail!r}, danger={COUNTERS.danger}",
    )


def _api_import_warning() -> tuple[bool, int, str]:
    env = os.environ.copy()
    env["ENABLE_HITL"] = "true"
    env["ENABLE_TOOL_POLICY"] = "false"
    proc = subprocess.run(
        [sys.executable, "-c", "import api_main"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = proc.stdout + proc.stderr
    warned = False
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        message = str(record.get("message", ""))
        if (
            record.get("levelname") == "WARNING"
            and "ENABLE_HITL" in message
            and "ENABLE_TOOL_POLICY" in message
        ):
            warned = True
    return warned, proc.returncode, output[-400:]


async def t92_113_flag_dependency(h: Harness) -> None:
    warned, returncode, tail = _api_import_warning()
    old_policy = settings.ENABLE_TOOL_POLICY
    old_hitl = settings.ENABLE_HITL
    try:
        settings.ENABLE_TOOL_POLICY = False
        settings.ENABLE_HITL = True
        configure_tools()
        COUNTERS.danger = 0
        thread = h.thread("113")
        events = await h.first_turn(thread, "danger please")
        types = [e.get("type") for e in events]
    finally:
        settings.ENABLE_TOOL_POLICY = old_policy
        settings.ENABLE_HITL = old_hitl
        configure_tools()
    ok = (
        warned
        and returncode == 0
        and "interrupt" not in types
        and COUNTERS.danger == 1
        and _final(events) is not None
    )
    check(
        "T92-113 A13 只开 HITL 关策略：启动 warning 且审批不触发（预期）",
        ok,
        f"warning={warned}, import_rc={returncode}, frames={types}, "
        f"danger={COUNTERS.danger}, tail={tail[-120:]!r}",
    )


async def t92_114_trace(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    settings.ENABLE_TRACE = True
    thread = h.thread("114")
    try:
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
            if data:
                tracer2 = JsonlTracer(td)
                config["callbacks"] = build_callbacks(tracer2)
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
                record["attrs"].get("error_kind")
                for record in records
                if record["kind"] == "tool" and record["session_id"] == thread
            ]
            ok = "approval_required" in kinds and "rejected_by_user" in kinds
            check(
                "T92-114 A14 追踪：挂起=approval_required / 拒绝=rejected_by_user",
                ok,
                f"tool error_kind={kinds}",
            )
    finally:
        settings.ENABLE_TRACE = False


# ── C 组：边界与反例（open_issue，不判失败） ─────────────────────────


class _BoomPipe:
    def set(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def execute(self) -> None:
        raise ConnectionError("redis down (probe)")


class _BoomRedis:
    """命令期抛异常的假 Redis：模拟"连接建立后服务不可用"。"""

    def pipeline(self, transaction: bool = True) -> _BoomPipe:
        return _BoomPipe()

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("redis down (probe)")

    async def set(self, *args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("redis down (probe)")

    async def eval(self, *args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("redis down (probe)")


async def c1_credential_gone(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    COUNTERS.danger = 0
    thread = h.thread("c1")
    events = await h.first_turn(thread, "danger please")
    data = (_interrupt(events) or {}).get("data") or {}
    if not data:
        c_check("C1 凭据消费后 Redis 无残留", False, "没有拿到 interrupt 帧")
        return
    resp = await h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}")
    aid = data["approval_id"]
    approval_key = approval_store.APPROVAL_KEY.format(approval_id=aid)
    tomb_key = approval_store.TOMBSTONE_KEY.format(approval_id=aid)
    redis: Any = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        raw = await redis.get(approval_key)
        tomb = await redis.get(tomb_key)
        exists = await redis.exists(approval_key)
    finally:
        await redis.aclose()
    ok = resp.status_code == 200 and raw is None and tomb == "consumed" and exists == 0
    c_check(
        "C1 凭据消费后 Redis 无可二次通过的记录（直接查键）",
        ok,
        f"status={resp.status_code}, approval_key={raw!r}, tombstone={tomb!r}, exists={exists}",
    )


async def c2_redis_unavailable(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    approval_store.set_default_store(
        approval_store.ApprovalStore(client=_BoomRedis())  # type: ignore[arg-type]
    )
    COUNTERS.danger = 0
    try:
        thread = h.thread("c2")
        events = await h.first_turn(thread, "danger please")
        types = [e.get("type") for e in events]
        final = _final(events) or {}
        resp = await h.resume(thread, "0" * 32, "approve", f"rid-{uuid.uuid4().hex}")
    finally:
        approval_store.set_default_store(approval_store.ApprovalStore())
    ok = (
        COUNTERS.danger == 0
        and "interrupt" not in types
        and "审批服务不可用" in final.get("content", "")
        and resp.status_code in (404, 409)
        and not any(e.get("type") == "error" for e in events)
    )
    deviation = "" if resp.status_code == 404 else "；偏离交接单（那里写 404）"
    c_check(
        "C2 Redis 不可用：审批一律拒绝（fail-closed）且会话不崩",
        ok,
        f"exec_status={types}, danger={COUNTERS.danger}, "
        f"final={final.get('content', '')[:50]!r}, resume={resp.status_code}{deviation}",
    )


async def c3_concurrent_resume(h: Harness) -> None:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    COUNTERS.danger = 0
    thread = h.thread("c3")
    events = await h.first_turn(thread, "danger please")
    data = (_interrupt(events) or {}).get("data") or {}
    if not data:
        c_check("C3 同会话并发 resume", False, "没有拿到 interrupt 帧")
        return
    second = await approval_store.create(
        thread_id=thread,
        tool="danger",
        args_digest="probe=1",
        cost_estimate=0.02,
        budget_remaining=99.0,
        permission="dangerous",
        tool_call_id="call-c3-second",
    )
    if second is None:
        c_check("C3 同会话并发 resume", False, "第二张凭据签发失败")
        return
    first_resp, second_resp = await asyncio.gather(
        h.resume(thread, data["approval_id"], "approve", f"rid-{uuid.uuid4().hex}"),
        h.resume(thread, second.approval_id, "approve", f"rid-{uuid.uuid4().hex}"),
    )
    statuses = sorted([first_resp.status_code, second_resp.status_code])
    ok = statuses == [200, 409] and COUNTERS.danger == 1
    c_check(
        "C3 同一会话两个 resume 并发：只有一个推进（另一个 409）",
        ok,
        f"statuses={statuses}, danger={COUNTERS.danger}",
    )
    # 清掉可能未被消费的第二张凭据，避免污染 Redis
    try:
        redis: Any = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            await redis.delete(
                approval_store.APPROVAL_KEY.format(approval_id=second.approval_id),
                approval_store.TOMBSTONE_KEY.format(approval_id=second.approval_id),
            )
        finally:
            await redis.aclose()
    except Exception:  # noqa: BLE001 - 清理失败不影响结论
        pass


async def c4_cleanup_task(h: Harness) -> None:
    """清理任务：调度器无法隔离验证；但可确定性地验证"残留锁释放"这一步。"""
    source = (REPO_ROOT / "api_main.py").read_text(encoding="utf-8")
    predicate = "INTERVAL '10 minutes'" in source and "state = 'interrupted'" in source
    key = f"agent_lock:c4-{uuid.uuid4().hex[:10]}"
    owner = h.redis.lock(key, timeout=30)
    await owner.acquire(blocking=False)
    other = h.redis.lock(key, timeout=30)
    locked = await other.locked()
    raised = ""
    try:
        await other.release()
    except Exception as exc:  # noqa: BLE001
        raised = type(exc).__name__
    still_there = await h.redis.exists(key)
    await owner.release()
    await h.redis.delete(key)
    lock_released = not still_there
    c_check(
        "C4 清理任务：状态转 failed 可成立，但残留锁释放失败（open_issue）",
        predicate and locked and lock_released,
        f"sql_predicate={predicate}, locked={locked}, release_raises={raised!r}, "
        f"lock_exists_after={still_there}；共享 8200 实例实测 state->failed、锁未释放；"
        "调度器固定 sleep 60s 且内联在 lifespan，未做隔离复跑",
    )


def c5_reject_text() -> None:
    ok = bool(REJECT_REASON) and "拒绝" in REJECT_REASON and "未调用" in REJECT_REASON
    c_check(
        "C5 拒绝文案可读性（人工判断）",
        ok,
        f"回灌模型文本={REJECT_REASON!r}；人工判断：语义明确、无技术堆栈",
    )


async def main() -> int:
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = True
    approval_store.set_default_store(approval_store.ApprovalStore())
    executor.set_default_ledger(MemoryBudgetLedger())

    async with Harness() as h:
        # 开发写的 13 项（保持原样）
        await a1_switch_off(h)
        await a2_a3_approve(h)
        await a4_reject(h)
        await a7_expired(h)
        await a8_safe_tool(h)
        await a9_concurrent(h)
        await a10_trace(h)
        await extra_thread_mismatch(h)
        # test-G7 独立补测 A1~A10、A13、A14
        await t92_101_switch_off(h)
        await t92_102_interrupt_frame(h)
        await t92_103_approval_id(h)
        await t92_104_approve(h)
        await t92_105_reject(h)
        await t92_106_replay_forge_mismatch(h)
        await t92_107_expired(h)
        await t92_108_concurrent_same_id(h)
        await t92_109_safe_tool(h)
        await t92_110_subgraph_interrupt(h)
        await t92_113_flag_dependency(h)
        await t92_114_trace(h)
        # C 组（不计入退出码）
        await c1_credential_gone(h)
        await c2_redis_unavailable(h)
        await c3_concurrent_resume(h)
        await c4_cleanup_task(h)

    # A11 / A12 需要真实 /chat/conversation + 真实 task_states
    async with AppHarness() as ah:
        await t92_111_state_machine(ah)
        await t92_112_interrupted_guard(ah)

    c5_reject_text()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    c_passed = sum(1 for _, ok, _ in C_RESULTS if ok)
    c_failed = len(C_RESULTS) - c_passed
    print(f"\nC-FINDINGS: {c_passed} passed, {c_failed} failed（open_issue，不计退出码）")
    print(f"\nRESULT: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
