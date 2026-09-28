"""Checkpointer 管理模块 —— 连接池与 Saver 初始化。"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from config import settings

# ============================================================
# 全局实例
# ============================================================
_global_saver: AsyncPostgresSaver | None = None
_saver_pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None
_business_pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None


async def init_checkpointer() -> None:
    """初始化连接池和 Saver（应用启动时调用）。"""
    global _global_saver, _saver_pool, _business_pool
    if _global_saver is not None:
        return

    conninfo = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

    # 1. Saver 专用连接池 —— 只给 AsyncPostgresSaver 用
    _saver_pool = AsyncConnectionPool[AsyncConnection[DictRow]](
        conninfo,
        min_size=2,
        max_size=10,
        max_waiting=30,
        timeout=30.0,
        max_lifetime=600,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    await _saver_pool.open()  # ✅ 显式打开

    # 2. 业务专用连接池 —— 只给 task_states 等业务查询用
    _business_pool = AsyncConnectionPool[AsyncConnection[DictRow]](
        conninfo,
        min_size=1,
        max_size=5,
        max_waiting=10,
        timeout=10.0,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    await _business_pool.open()  # ✅ 显式打开

    # 3. 创建 Saver（使用 Saver 专用池）
    _global_saver = AsyncPostgresSaver(_saver_pool)
    await _global_saver.setup()


async def close_checkpointer() -> None:
    global _global_saver, _saver_pool, _business_pool
    if _saver_pool is not None:
        await _saver_pool.close()
        _saver_pool = None
    if _business_pool is not None:
        await _business_pool.close()
        _business_pool = None
    _global_saver = None


def get_checkpointer() -> AsyncPostgresSaver:
    """返回全局 Saver 实例。"""
    if _global_saver is None:
        raise RuntimeError("Checkpointer 尚未初始化")
    return _global_saver


def get_business_pool() -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    """返回业务专用连接池（供 task_states 等查询使用）。"""
    if _business_pool is None:
        raise RuntimeError("业务连接池尚未初始化")
    return _business_pool
