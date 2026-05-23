# src/checkpointer.py
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from config import settings

# 全局 Saver 实例（供 langgraph checkpoint 使用）
_global_saver: AsyncPostgresSaver | None = None
# psycopg 异步连接池 —— 同时供 AsyncPostgresSaver 和 task_states CRUD 使用
# DictRow = dict[str, Any]，与 AsyncPostgresSaver 要求的类型一致
_global_pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None


async def init_checkpointer() -> None:
    """应用启动时创建异步连接池和全局 Saver。"""
    global _global_saver, _global_pool
    if _global_saver is not None:
        return

    # 构建 psycopg 连接字符串
    conninfo = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

    # 创建 psycopg 异步连接池（langgraph AsyncPostgresSaver 原生支持）
    _global_pool = AsyncConnectionPool[AsyncConnection[DictRow]](
        conninfo,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )

    # 创建 AsyncPostgresSaver 并初始化表结构（只会执行一次）
    _global_saver = AsyncPostgresSaver(_global_pool)
    await _global_saver.setup()


async def close_checkpointer() -> None:
    """应用关闭时释放连接池。"""
    global _global_saver, _global_pool
    if _global_pool is not None:
        await _global_pool.close()
        _global_pool = None
    _global_saver = None


def get_checkpointer() -> AsyncPostgresSaver:
    """返回全局唯一的 Saver 实例。"""
    if _global_saver is None:
        raise RuntimeError("Checkpointer 尚未初始化。请先调用 init_checkpointer()。")
    return _global_saver
