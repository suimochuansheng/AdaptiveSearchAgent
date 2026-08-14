"""Celery 异步任务定义。

将耗时 Agent 图遍历从 FastAPI 请求线程中剥离，通过 Redis 消息队列
交由独立 Worker 进程执行。由 ENABLE_ASYNC_TASK 开关控制启用。
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from celery import Celery

from config import settings
from src.utils.logger import get_logger

logger = get_logger()

# ── 构造 Redis Broker URL（db=1，与分布式锁的 db=0 隔离）──────


def _build_broker_url() -> str:
    """从 settings.REDIS_URL 解析 host/port，构造 db=1 的 broker URL。"""
    if settings.REDIS_URL:
        parsed = urlparse(settings.REDIS_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
    else:
        host = "localhost"
        port = 6379

    password_part = f":{settings.REDIS_PASSWORD}@" if settings.REDIS_PASSWORD else ""
    return f"redis://{password_part}{host}:{port}/1"


# ── Celery 应用实例 ───────────────────────────────────────────

celery_app = Celery(
    "adaptive_search_tasks",
    broker=_build_broker_url(),
    backend=_build_broker_url(),  # 结果也存入 Redis（同 db=1）
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_soft_time_limit=580,
    task_time_limit=600,
    broker_connection_retry_on_startup=True,
    task_acks_late=True,  # Worker 崩溃后任务自动重新入队
)


# ── 异步 Agent 图遍历任务 ────────────────────────────────────


@celery_app.task(name="execute_agent_task", bind=True)
def execute_agent_task(self, thread_id: str, user_query: str) -> dict:
    """异步执行 LangGraph Agent 图遍历，完成后将 final_report 写入 Redis 后端。

    此任务由 Celery Worker 消费，不与 FastAPI 请求线程共享事件循环。

    Args:
        thread_id: 会话/用户唯一标识。
        user_query: 用户原始查询文本。

    Returns:
        包含 status, final_report, total_tokens, thread_id 的结构化字典。
    """
    logger.info(
        "Celery task started",
        extra={"task_id": self.request.id, "thread_id": thread_id},
    )

    async def _run() -> dict:
        from src.graph_factory import get_graph
        from src.state import AgentState

        graph = await get_graph()

        initial_state: AgentState = {
            "user_query": user_query,
            "plan": [],
            "search_results": [],
            "_search_accum": [],
            "confidence_score": 0.0,
            "missing_info": "",
            "retry_keywords": [],
            "iteration": 0,
            "final_report": "",
            "task_id": thread_id,
            "thread_id": thread_id,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "current_llm": "ollama",
            "pending_keywords": [],
            "_batch_keywords": [],
        }

        config = {"configurable": {"thread_id": thread_id}}

        async for event in graph.astream(initial_state, config):
            # 检查任务是否被撤销
            if celery_app.control.revoke(self.request.id, terminate=False):
                logger.warning("Task cancelled", extra={"task_id": self.request.id})
                return {"status": "cancelled", "thread_id": thread_id}

        final_state = await graph.aget_state(config)
        values = final_state.values if final_state else {}

        logger.info(
            "Celery task completed",
            extra={
                "task_id": self.request.id,
                "thread_id": thread_id,
                "report_len": len(values.get("final_report", "")),
            },
        )

        return {
            "status": "completed",
            "final_report": values.get("final_report", ""),
            "total_tokens": values.get("total_tokens", 0),
            "thread_id": thread_id,
        }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()
