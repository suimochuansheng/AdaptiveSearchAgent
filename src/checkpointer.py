# 全局连接池与 Checkpointer 管理
# src/checkpointer.py
from contextlib import asynccontextmanager

import asyncpg
from langgraph.checkpoint.postgres import AsyncPostgresSaver

from config import settings

_global_pool: asyncpg.Pool | None = None


async def init_checkpointer() -> None:
    """在应用启动时调用，初始化全局连接池"""
    global _global_pool
    if _global_pool is not None:
        return
    _global_pool = await asyncpg.create_pool(
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
        host=settings.postgres_host,
        port=settings.postgres_port,
        min_size=settings.postgres_pool_min_size,
        max_size=settings.postgres_pool_max_size,
        command_timeout=60,
    )
    # 初始化 checkpoint 表结构（如果表不存在）
    async with _global_pool.acquire() as conn:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()


async def close_checkpointer() -> None:
    """应用关闭时释放连接池"""
    global _global_pool
    if _global_pool:
        await _global_pool.close()
        _global_pool = None


@asynccontextmanager
async def get_checkpointer():
    """提供一个获取 checkpointer 的上下文管理器（每次从池中借连接）"""
    if _global_pool is None:
        raise RuntimeError("Checkpointer 尚未初始化。请先调用 init_checkpointer() 函数。")
    async with _global_pool.acquire() as conn:
        saver = AsyncPostgresSaver(conn)
        yield saver
