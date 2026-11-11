"""外层 HITL 审批：中断帧编码 + `POST /chat/conversation/{thread_id}/resume`（T9-2）。

链路：

```text
工具策略 require_approval
  └─ executor 签发 ApprovalRequest -> interrupt(八字段)
       └─ LangGraph values chunk: {"messages": …, "__interrupt__": (Interrupt(value=…),)}
            └─ build_interrupt_frame() -> SSE {"type":"interrupt","data":{八字段}}
                 └─ 前端弹卡 -> POST /resume {approval_id, decision, request_id}
                      └─ 校验 -> Command(resume=decision) -> 续跑 -> 继续 SSE
```

**与 `deep_research` 子图中断的区分**（判据写死在这里）：

- 外层审批的 `Interrupt.value` 一定是 dict，且**含 `approval_id`**（服务端签发的一次性凭据）；
- 子图成本确认（`src/agents/hitl_confirm.py`）的 payload 含 `type="cost_confirm"` 且
  **没有 `approval_id`**，它属于子图内部状态机，留在子图里，本模块不编码、不弹卡。
- 因此这里只处理 `approval_id` 那种；其余一律返回 None（调用方按“非外层审批”忽略）。

`data` 严格等于 `docs/contracts/approval.md` 的八个字段，顺序也按契约表。
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import nullcontext
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from agent.adapters.tools import executor
from config import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agent.domain.roles import ApprovalRequest

logger = logging.getLogger(__name__)

router = APIRouter()

#: 契约 frozen：`data` 恰好这八个字段（顺序即审批卡片阅读顺序）
APPROVAL_FIELDS: tuple[str, ...] = (
    "approval_id",
    "tool_call_id",
    "tool",
    "args_digest",
    "cost_estimate",
    "budget_remaining",
    "permission",
    "expires_at",
)

_APPROVAL_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ResumeRequest(BaseModel):
    """resume body：恰好三个键（契约 `resume-http.md` §2）。"""

    approval_id: str
    decision: Literal["approve", "reject"]
    request_id: str


# ── 中断 -> SSE 帧 ───────────────────────────────────────────────────


def approval_payload(request: ApprovalRequest) -> dict[str, Any]:
    """`ApprovalRequest` -> 八字段 dict（不含 thread_id 等实现细节）。"""
    data = asdict(request)
    return {field: data[field] for field in APPROVAL_FIELDS}


def build_interrupt_frame(values_payload: dict[str, Any]) -> str | None:
    """从 LangGraph 的 values chunk 里取外层审批中断，编码成 SSE 帧。

    Args:
        values_payload: `astream(stream_mode=["messages", "values"])` 的 values 快照；
            中断时它形如 `{"messages": […], "__interrupt__": (Interrupt(...),)}`。

    Returns:
        `data: {"type":"interrupt","data":{八字段}}\\n\\n`；没有外层审批中断时返回 None。
    """
    interrupts = values_payload.get("__interrupt__") or ()
    for item in interrupts:
        value = getattr(item, "value", None)
        if not isinstance(value, dict) or "approval_id" not in value:
            # 子图 cost_confirm / 未知中断：留在子图，不弹外层卡片
            continue
        data = {field: value.get(field) for field in APPROVAL_FIELDS}
        if not _APPROVAL_ID_RE.match(str(data["approval_id"])):
            logger.warning("审批中断的 approval_id 不是 32 位 hex，忽略该帧（fail-closed）")
            return None
        if any(data[field] is None for field in APPROVAL_FIELDS):
            logger.warning("审批中断缺字段，忽略该帧（fail-closed）：%s", sorted(data))
            return None
        return f"data: {json.dumps({'type': 'interrupt', 'data': data}, ensure_ascii=False)}\n\n"
    return None


# ── 依赖（便于测试替换；生产走 api_main 的图 / Redis） ───────────────


async def _load_graph() -> Any:
    from api_main import get_conversation_graph

    return await get_conversation_graph()


async def _acquire_thread_lock(thread_id: str) -> Any | None:
    """复用 `/chat/conversation` 的 Redis 会话锁；拿不到返回 None（由调用方判 409）。"""
    from api_main import redis_client

    lock = redis_client.lock(f"agent_lock:{thread_id}", timeout=600)
    acquired = await lock.acquire(blocking=False)
    return lock if acquired else None


def _langfuse_handler() -> Any:
    from api_main import langfuse_handler

    return langfuse_handler


async def _pending_interrupt(graph: Any, config: dict[str, Any]) -> bool:
    """会话当前是否有挂起任务；没有时不允许 resume（防止把已完成会话重新驱动）。"""
    snapshot = await graph.aget_state(config)
    return bool(getattr(snapshot, "next", ()))


# ── resume 路由 ──────────────────────────────────────────────────────


@router.post("/chat/conversation/{thread_id}/resume")
async def resume_conversation(  # noqa: C901 - 状态码矩阵按契约逐条展开，拆开反而丢顺序
    thread_id: str, body: ResumeRequest, request: Request
) -> StreamingResponse:
    """恢复一次被外层 HITL 挂起的会话（状态码严格按 `resume-http.md` §4）。

    处理顺序：`request_id` 幂等 -> 凭据存在性/归属/TTL -> 会话挂起校验 -> 会话锁 ->
    原子消费 -> `Command(resume=decision)` 续跑并继续吐 SSE。
    """
    if not settings.ENABLE_HITL:
        raise HTTPException(status_code=404, detail="审批功能未启用")

    from agent.adapters.stores import approval_store as store

    try:
        # 1) request_id 24h 幂等：重复 -> 409
        if not await store.remember_request(body.request_id, thread_id):
            raise HTTPException(status_code=409, detail="该 request_id 已处理，请勿重复提交")

        # 2) 凭据存在性：consume 过的 -> 409；签发过但已过期 -> 410；从未存在 -> 404
        stored = await store.get(body.approval_id)
        if stored is None:
            await _raise_by_tombstone(store, body.approval_id, lock=None)

        # 3) 归属：换 thread 使用同一凭据视为重放 -> 409
        if stored.thread_id != thread_id:
            raise HTTPException(status_code=409, detail="审批凭据不属于该会话")

        graph = await _load_graph()
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if not await _pending_interrupt(graph, config):
            raise HTTPException(status_code=409, detail="该会话没有待审批的挂起任务")
    except HTTPException:
        raise
    except store.ApprovalStoreUnavailable as exc:
        # Redis 不可用：fail-closed，不执行；契约没有 503，沿用 404 的“未启用”语义
        logger.warning("审批存储不可用，resume 失败（fail-closed）：%s", exc)
        raise HTTPException(
            status_code=404, detail="审批服务不可用，已拒绝（fail-closed）"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - 读状态 / 读凭据失败一律不执行
        logger.exception("resume 前置校验失败（fail-closed）")
        raise HTTPException(status_code=404, detail=f"审批状态不可用：{exc}") from exc

    # 4) 并发保护：同一会话只允许一个 resume 推进
    try:
        lock = await _acquire_thread_lock(thread_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("resume 获取会话锁失败（fail-closed）：%s", exc)
        raise HTTPException(status_code=404, detail="会话锁不可用，已拒绝（fail-closed）") from exc
    if lock is None:
        raise HTTPException(status_code=409, detail="该会话正在恢复中")

    # 5) 原子消费：只有一个 resume 能拿到凭据
    try:
        consumed = await store.consume(body.approval_id, thread_id)
        if consumed is None:
            await _raise_by_tombstone(store, body.approval_id, lock=lock)
    except HTTPException:
        await lock.release()
        raise
    except store.ApprovalStoreUnavailable as exc:
        await lock.release()
        logger.warning("审批凭据消费失败（fail-closed）：%s", exc)
        raise HTTPException(
            status_code=404, detail="审批服务不可用，已拒绝（fail-closed）"
        ) from exc

    # 6) 200 + SSE：用 Command(resume=decision) 续跑同一 thread_id
    return StreamingResponse(
        _resume_event_stream(request, graph, config, thread_id, body.decision, lock),
        media_type="text/event-stream",
    )


async def _resume_event_stream(
    request: Request,
    graph: Any,
    config: dict[str, Any],
    thread_id: str,
    decision: str,
    lock: Any,
) -> AsyncIterator[str]:
    """驱动续跑并按与 `/chat/conversation` 同一套 SSE 帧继续推送。"""
    from agent.observability.langchain_handler import build_callbacks
    from agent.observability.tracer import get_tracer
    from agent.observability.turn import next_turn_index, turn_span
    from src.conversation_stream import ConversationStreamError, stream_conversation

    try:
        trace: Any = nullcontext()
        if settings.ENABLE_TRACE:
            config["callbacks"] = build_callbacks(get_tracer(), _langfuse_handler())
            trace = turn_span(get_tracer(), thread_id, await next_turn_index(graph, config))

        async with trace:
            with executor.approval_resume_scope():
                async for frame in stream_conversation(
                    graph,
                    Command(resume=decision),  # type: ignore[arg-type]
                    config,
                    is_disconnected=request.is_disconnected,
                ):
                    yield frame
    except ConversationStreamError:
        # error 帧已由 stream_conversation 推送，这里只记录
        logger.exception("恢复后的会话流执行错误（thread_id=%s）", thread_id)
    except Exception as exc:  # noqa: BLE001 - 已经 200，只能以 error 帧收尾
        logger.exception("恢复后的会话流异常（thread_id=%s）", thread_id)
        yield f"data: {json.dumps({'type': 'error', 'message': f'服务异常：{exc}'}, ensure_ascii=False)}\n\n"
    finally:
        await lock.release()


async def _raise_by_tombstone(store: Any, approval_id: str, *, lock: Any | None) -> NoReturn:
    """凭据缺失时按墓碑分流：已消费 409 / 过期 410 / 从未存在 404。"""
    state = await store.tombstone_state(approval_id)
    if lock is not None:
        await lock.release()
    if state == "consumed":
        raise HTTPException(status_code=409, detail="该审批已被处理（重放）")
    if state == "issued":
        raise HTTPException(status_code=410, detail="审批已过期，请重新发起")
    raise HTTPException(status_code=404, detail="未知的审批凭据")


__all__ = [
    "APPROVAL_FIELDS",
    "ResumeRequest",
    "approval_payload",
    "build_interrupt_frame",
    "router",
]
