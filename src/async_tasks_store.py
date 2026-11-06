"""异步任务状态存储（阶段 C1）。

为什么新开一张表而不是复用 `task_states`：
- `task_states` 的主键是 `thread_id`（会话维度），一个会话可能先后提交多个任务；
- 异步任务需要按 `task_id` 查询，还要存进度、错误、结果引用、取消标记；
- 语义混在一张表里，旧端点的"僵尸锁清理"会把任务行当成会话行误判。

所以：新表 `async_tasks`，旧表一行不动、旧端点行为不变。
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from src.checkpointer import get_business_pool
from src.utils.logger import get_logger

logger = get_logger()

# 任务状态机：queued → running → completed / failed / cancelled
VALID_STATES = ("queued", "running", "completed", "failed", "cancelled")

_DDL_TABLE = """
CREATE TABLE IF NOT EXISTS async_tasks (
    task_id          TEXT PRIMARY KEY,
    thread_id        TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    kind             TEXT NOT NULL DEFAULT 'conversation',
    state            TEXT NOT NULL DEFAULT 'queued',
    progress         INT  NOT NULL DEFAULT 0,
    error            TEXT,
    result_ref       TEXT,
    payload          TEXT NOT NULL DEFAULT '{}',
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    attempts         INT  NOT NULL DEFAULT 0,
    created_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMP NOT NULL DEFAULT NOW()
)
"""

# 已存在的库升级：additive，不删列
_DDL_MIGRATIONS = (
    "ALTER TABLE async_tasks ADD COLUMN IF NOT EXISTS attempts INT NOT NULL DEFAULT 0",
    "ALTER TABLE async_tasks ADD COLUMN IF NOT EXISTS cancel_requested BOOLEAN NOT NULL DEFAULT FALSE",
)

_DDL_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_async_tasks_thread ON async_tasks (thread_id, created_at DESC)"
)


async def ensure_async_tasks_table() -> None:
    """建表（幂等，应用启动时调用）。"""
    async with get_business_pool().connection() as conn:
        # psycopg 不允许一次 execute 多条语句，DDL 拆开跑
        await conn.execute(_DDL_TABLE)
        for migration in _DDL_MIGRATIONS:
            await conn.execute(migration)
        await conn.execute(_DDL_INDEX)
    logger.info("async_tasks 表已就绪")


async def create_task(
    *,
    thread_id: str,
    user_id: str,
    message: str,
    kind: str = "conversation",
) -> str:
    """写入一条排队中的任务，返回 task_id。"""
    task_id = uuid.uuid4().hex
    async with get_business_pool().connection() as conn:
        await conn.execute(
            "INSERT INTO async_tasks (task_id, thread_id, user_id, kind, payload) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                task_id,
                thread_id,
                user_id,
                kind,
                json.dumps({"message": message}, ensure_ascii=False),
            ),
        )
    return task_id


async def get_task(task_id: str) -> dict[str, Any] | None:
    """读取任务行；不存在返回 None。"""
    async with get_business_pool().connection() as conn:
        result = await conn.execute(
            "SELECT task_id, thread_id, user_id, kind, state, progress, error, "
            "result_ref, payload, cancel_requested, attempts, updated_at "
            "FROM async_tasks WHERE task_id = %s",
            (task_id,),
        )
        row = await result.fetchone()
    if row is None:
        return None
    task = dict(row)
    try:
        task["payload"] = json.loads(task["payload"] or "{}")
    except json.JSONDecodeError:
        task["payload"] = {}
    return task


async def set_state(
    task_id: str,
    state: str,
    *,
    progress: int | None = None,
    error: str | None = None,
    result_ref: str | None = None,
) -> None:
    """更新任务状态；已处于 completed 的行不再被覆盖（幂等的第一道闸）。"""
    if state not in VALID_STATES:
        raise ValueError(f"非法状态: {state}")
    async with get_business_pool().connection() as conn:
        await conn.execute(
            "UPDATE async_tasks SET state = %s, "
            "progress = COALESCE(%s::int, progress), "
            "error = COALESCE(%s::text, error), "
            "result_ref = COALESCE(%s::text, result_ref), "
            "updated_at = NOW() "
            # 终态（completed/failed/cancelled）不再被覆盖：只有排队/运行中能迁移
            "WHERE task_id = %s AND state IN ('queued', 'running')",
            (state, progress, error, result_ref, task_id),
        )


async def claim_task(task_id: str) -> bool:
    """原子认领任务：queued → running。

    Worker 可能重复投递（acks_late），两个进程同时执行同一任务会重复调模型。
    用一条 UPDATE ... WHERE state='queued' 做乐观锁，只有一个能拿到。
    """
    async with get_business_pool().connection() as conn:
        result = await conn.execute(
            "UPDATE async_tasks SET state = 'running', progress = 5, updated_at = NOW() "
            "WHERE task_id = %s AND state = 'queued'",
            (task_id,),
        )
        return result.rowcount > 0


async def request_cancel(task_id: str) -> bool:
    """标记取消请求；任务不存在返回 False。

    真正的停止发生在执行器"两个 superstep 之间"——这里只落标记，
    不做进程级强杀（强杀会把 checkpoint 写坏）。
    """
    async with get_business_pool().connection() as conn:
        result = await conn.execute(
            "UPDATE async_tasks SET cancel_requested = TRUE, updated_at = NOW() "
            "WHERE task_id = %s AND state IN ('queued', 'running')",
            (task_id,),
        )
        return result.rowcount > 0


async def is_cancel_requested(task_id: str) -> bool:
    """执行器在每个 superstep 之间检查一次。"""
    async with get_business_pool().connection() as conn:
        result = await conn.execute(
            "SELECT cancel_requested FROM async_tasks WHERE task_id = %s",
            (task_id,),
        )
        row = await result.fetchone()
    return bool(row and row["cancel_requested"])


async def requeue_stale(stale_seconds: int, max_attempts: int) -> list[str]:
    """回收"运行中但长时间没进展"的任务：重试次数未用尽就重新排队。

    场景：Worker 被 kill -9 / 机器重启，任务永远停在 running。
    每个 superstep 都会 touch updated_at，所以长时间不更新 = 进程没了。

    Returns:
        需要重新入队的 task_id 列表。
    """
    async with get_business_pool().connection() as conn:
        result = await conn.execute(
            "UPDATE async_tasks "
            "SET state = 'queued', progress = 0, attempts = attempts + 1, updated_at = NOW() "
            "WHERE state = 'running' "
            "  AND updated_at < NOW() - (%s * INTERVAL '1 second') "
            "  AND attempts < %s "
            "RETURNING task_id",
            (stale_seconds, max_attempts),
        )
        rows = await result.fetchall()
        if rows:
            logger.warning("回收僵尸异步任务 %d 个，重新入队", len(rows))
        return [row["task_id"] for row in rows]


async def fail_stale_exhausted(stale_seconds: int, max_attempts: int) -> int:
    """重试次数用尽的僵尸任务直接置 failed，避免无限重跑。"""
    async with get_business_pool().connection() as conn:
        result = await conn.execute(
            "UPDATE async_tasks "
            "SET state = 'failed', "
            "    error = COALESCE(error, 'worker 中断：超过阈值无进展且重试次数已用尽'), "
            "    updated_at = NOW() "
            "WHERE state = 'running' "
            "  AND updated_at < NOW() - (%s * INTERVAL '1 second') "
            "  AND attempts >= %s",
            (stale_seconds, max_attempts),
        )
        return result.rowcount


async def purge_finished(older_than_days: int = 7) -> int:
    """清理终态任务行（结果本身在 Redis 有 TTL，这里清的是数据库记录）。"""
    async with get_business_pool().connection() as conn:
        result = await conn.execute(
            "DELETE FROM async_tasks "
            "WHERE state IN ('completed', 'failed', 'cancelled') "
            "  AND updated_at < NOW() - (%s * INTERVAL '1 day')",
            (older_than_days,),
        )
        return result.rowcount
