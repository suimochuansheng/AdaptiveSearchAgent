"""会话式 Agent 的异步任务执行器（阶段 C1）。

链路：提交接口写一行 `async_tasks(queued)` → Celery 执行本文件的任务 →
每个 superstep 更新进度、检查取消 → 结果写 Redis（PG 只存引用）→ 状态置 completed。

可测性：真正的编排逻辑在 `run_conversation_task`，依赖（图工厂 / 状态存储 / 结果存储）
全部由参数注入，单元测试不需要 PG、Redis、Celery、LLM。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

from config import settings
from src.tasks import celery_app
from src.utils.logger import get_logger

logger = get_logger()

RESULT_KEY_TEMPLATE = "task:result:{task_id}"

# 进度不是精确百分比，只是让前端能区分"还在思考"和"在跑工具"
_PROGRESS_START = 5
_PROGRESS_PER_STEP = 10
_PROGRESS_MAX = 90


class TaskStore(Protocol):
    """状态存储接口（生产用 async_tasks_store，测试用内存实现）。"""

    async def get(self, task_id: str) -> dict[str, Any] | None: ...

    async def claim(self, task_id: str) -> bool: ...

    async def set_state(
        self,
        task_id: str,
        state: str,
        *,
        progress: int | None = None,
        error: str | None = None,
        result_ref: str | None = None,
    ) -> None: ...

    async def is_cancel_requested(self, task_id: str) -> bool: ...


class ResultStore(Protocol):
    """结果存储接口（生产用 Redis，测试用字典）。"""

    async def set(self, key: str, value: str, ttl: int) -> None: ...

    async def get(self, key: str) -> str | None: ...


def _answer_of(state: dict[str, Any] | None) -> str:
    """从最终状态里取最后一条消息的正文。"""
    if not state:
        return ""
    messages = state.get("messages") or []
    if not messages:
        return ""
    return str(messages[-1].content)


async def run_conversation_task(
    task_id: str,
    *,
    graph_factory: Any,
    store: TaskStore,
    result_store: ResultStore,
    ttl: int | None = None,
) -> dict[str, Any]:
    """执行一个异步会话任务。

    Args:
        task_id: 任务 id（async_tasks 主键）。
        graph_factory: 无参可等待函数，返回已编译的会话图。
        store: 状态存储。
        result_store: 结果存储。
        ttl: 结果在 Redis 的存活秒数，默认读配置。

    Returns:
        执行摘要（state / answer / steps / error），便于 Celery 日志与测试断言。
    """
    task = await store.get(task_id)
    if task is None:
        logger.warning("异步任务不存在", extra={"task_id": task_id})
        return {"task_id": task_id, "state": "missing"}

    # 幂等闸：Worker 重跑时已完成的任务直接返回，不再调模型
    if task["state"] == "completed":
        return {"task_id": task_id, "state": "completed", "idempotent": True}
    if task["state"] == "cancelled":
        return {"task_id": task_id, "state": "cancelled"}
    if task["state"] == "failed":
        return {"task_id": task_id, "state": "failed", "error": task.get("error"), "skipped": True}

    # 原子认领：两个 Worker 同时收到同一任务时，只有一个能把它从 queued 推进到 running
    if not await store.claim(task_id):
        current = await store.get(task_id) or {}
        logger.info("任务已被其他 Worker 认领或已结束", extra={"task_id": task_id})
        return {"task_id": task_id, "state": current.get("state", "unknown"), "skipped": True}

    message = (task.get("payload") or {}).get("message", "")

    # 排队期间就被取消的任务：连图都不要建，省一次模型调用
    if await store.is_cancel_requested(task_id):
        await store.set_state(task_id, "cancelled", progress=_PROGRESS_START)
        return {"task_id": task_id, "state": "cancelled", "steps": 0}

    final_state: dict[str, Any] | None = None
    steps = 0
    try:
        graph = await graph_factory()
        config = {"configurable": {"thread_id": task["thread_id"]}}
        async for event in graph.astream(
            _initial_input(message, task.get("user_id") or task["thread_id"]),
            config=config,
            stream_mode="values",
        ):
            final_state = event
            steps += 1
            # 取消只在 superstep 之间生效：强杀会写坏 checkpoint
            if await store.is_cancel_requested(task_id):
                await store.set_state(task_id, "cancelled", progress=min(5 + steps * 5, 90))
                logger.info("异步任务被取消", extra={"task_id": task_id, "steps": steps})
                return {"task_id": task_id, "state": "cancelled", "steps": steps}
            await store.set_state(
                task_id,
                "running",
                progress=min(_PROGRESS_START + steps * _PROGRESS_PER_STEP, _PROGRESS_MAX),
            )

        answer = _answer_of(final_state)
        result_ref = RESULT_KEY_TEMPLATE.format(task_id=task_id)
        payload = json.dumps(
            {"task_id": task_id, "thread_id": task["thread_id"], "answer": answer},
            ensure_ascii=False,
        )
        await result_store.set(result_ref, payload, ttl or settings.ASYNC_TASK_RESULT_TTL)
        await store.set_state(task_id, "completed", progress=100, result_ref=result_ref)
        logger.info("异步任务完成", extra={"task_id": task_id, "steps": steps})
        return {"task_id": task_id, "state": "completed", "answer": answer, "steps": steps}

    except Exception as exc:  # noqa: BLE001 - 任务失败要落状态，不能让 Worker 吞掉
        logger.exception("异步任务失败", extra={"task_id": task_id})
        await store.set_state(task_id, "failed", error=f"{type(exc).__name__}: {exc}")
        return {"task_id": task_id, "state": "failed", "error": str(exc), "steps": steps}


def _initial_input(message: str, user_id: str) -> dict[str, Any]:
    """构造图输入（延迟 import，避免测试环境拉起重依赖）。"""
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=message)], "user_id": user_id}


# ── 生产实现：DB / Redis / Celery ────────────────────────────────────────


class _DbTaskStore:
    """把 async_tasks_store 的模块函数包成对象，便于注入。"""

    async def get(self, task_id: str) -> dict[str, Any] | None:
        from src.async_tasks_store import get_task

        return await get_task(task_id)

    async def claim(self, task_id: str) -> bool:
        from src.async_tasks_store import claim_task

        return await claim_task(task_id)

    async def set_state(self, task_id: str, state: str, **kwargs: Any) -> None:
        from src.async_tasks_store import set_state

        await set_state(task_id, state, **kwargs)

    async def is_cancel_requested(self, task_id: str) -> bool:
        from src.async_tasks_store import is_cancel_requested

        return await is_cancel_requested(task_id)


class _RedisResultStore:
    """结果存 Redis，PG 里只留 key（大文本不进关系库）。"""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def _redis(self) -> Any:
        if self._client is None:
            from redis import asyncio as aioredis

            self._client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._client

    async def set(self, key: str, value: str, ttl: int) -> None:
        await self._redis().set(key, value, ex=ttl)

    async def get(self, key: str) -> str | None:
        return await self._redis().get(key)


async def _default_graph_factory() -> Any:
    """默认图工厂：复用全局 checkpointer，写回同一 thread 的存档。"""
    from src.checkpointer import get_checkpointer
    from src.conversation_graph import build_conversation_graph

    return build_conversation_graph(checkpointer=get_checkpointer())


@celery_app.task(name="conversation_task", bind=True)
def conversation_task(self: Any, task_id: str) -> dict[str, Any]:
    """Celery 入口：Worker 进程里没有 FastAPI lifespan，要自己初始化连接池。"""
    from src.async_tasks_store import ensure_async_tasks_table
    from src.checkpointer import init_checkpointer

    async def _main() -> dict[str, Any]:
        await init_checkpointer()
        await ensure_async_tasks_table()
        return await run_conversation_task(
            task_id,
            graph_factory=_default_graph_factory,
            store=_DbTaskStore(),
            result_store=_RedisResultStore(),
        )

    return asyncio.run(_main())


def enqueue_conversation_task(task_id: str) -> None:
    """把任务扔进队列（Celery broker）。"""
    conversation_task.delay(task_id)


async def read_result(task_id: str) -> str | None:
    """读取已完成任务的结果 JSON 字符串；未完成/过期返回 None。"""
    return await _RedisResultStore().get(RESULT_KEY_TEMPLATE.format(task_id=task_id))
