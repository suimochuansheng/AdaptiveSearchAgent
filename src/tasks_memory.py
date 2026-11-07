"""画像抽取的 Celery 入口（阶段 C3b）。

Worker 进程没有 FastAPI lifespan，所以这里自己初始化 checkpointer；
从会话 checkpoint 读最近消息与摘要，抽取后写回 user_profiles。
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.tasks import celery_app
from src.utils.logger import get_logger

logger = get_logger()


async def _default_llm(prompt: str) -> str:
    """生产环境的模型调用：走统一入口（带主模型重试 + 备援）。"""
    from src.utils.llm_utils import llm_call_with_fallback

    content, _in_tok, _out_tok, _total = await llm_call_with_fallback(prompt, None)
    return content


async def run_profile_extraction(thread_id: str, user_id: str) -> dict[str, Any]:
    """从 checkpoint 取会话内容，抽取并落库。"""
    from config import settings
    from src.checkpointer import get_checkpointer, init_checkpointer
    from src.conversation_graph import build_conversation_graph
    from src.memory.profile_context import ensure_user_profiles_table
    from src.memory.profile_extraction import extract_and_store
    from src.memory.user_profile_store import get_user_profile_store

    await init_checkpointer()
    await ensure_user_profiles_table()

    graph = build_conversation_graph(checkpointer=get_checkpointer())
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values or {}

    store = await get_user_profile_store()
    return await extract_and_store(
        llm_fn=_default_llm,
        store=store,
        user_id=user_id,
        messages=values.get("messages") or [],
        summary=values.get("context_summary") or "",
        max_messages=settings.MEMORY_EXTRACT_MAX_MESSAGES,
    )


@celery_app.task(name="extract_profile")
def extract_profile_task(thread_id: str, user_id: str) -> dict[str, Any]:
    """Celery 入口：同步进程里跑一遍异步逻辑。"""
    logger.info("画像抽取任务开始", extra={"thread_id": thread_id, "user_id": user_id})
    return asyncio.run(run_profile_extraction(thread_id, user_id))


def enqueue_profile_extraction(thread_id: str, user_id: str) -> None:
    """把抽取任务扔进队列（调用方负责不要阻塞主流程）。"""
    extract_profile_task.delay(thread_id, user_id)
