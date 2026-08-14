# 提供 get_graph() 异步工厂函数，封装图的构建与编译。
# src/graph_factory.py
import asyncio

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from config import settings
from src.agents.evaluator import evaluator
from src.agents.parallel_searcher import parallel_searcher, route_to_search_workers
from src.agents.planner import planner
from src.agents.search_worker import search_worker
from src.agents.writer import writer
from src.checkpointer import get_checkpointer
from src.state import AgentState

# 全局 graph 实例占位符（在异步启动后赋值）
_compiled_graph = None
_graph_lock = asyncio.Lock()


async def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        async with _graph_lock:
            if _compiled_graph is None:
                saver = get_checkpointer()
                _compiled_graph = build_graph(checkpointer=saver)
    return _compiled_graph


def should_continue_wave(state: AgentState) -> str:
    """条件边路由：评估后决定继续搜索或生成报告。

    优先级（从上到下短路）：
    1. iteration >= max_iterations → 强制 writer
    2. confidence_score >= threshold → writer（高置信度直接生成）
    3. 有 pending_keywords → parallel_searcher（继续搜索）
    4. 否则 → planner（重试新关键词）
    """
    if state["iteration"] >= settings.max_iterations:
        return "writer"

    if state["confidence_score"] >= settings.confidence_threshold:
        return "writer"

    if state.get("pending_keywords"):
        return "parallel_searcher"

    return "planner"


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """构建全自动 LangGraph 工作流（无人工审批）。

    图结构：
        planner → parallel_searcher → workers → evaluator
            ↑                                    │
            └────────────── retry ───────────────┤
                                                 ↓ (conf >= threshold | iter >= max)
                                               writer → END

    Send 扇出机制：
        parallel_searcher 写入 _batch_keywords，
        route_to_search_workers 返回 list[Send] 并行扇出 search_worker。
    """
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("planner", planner)
    builder.add_node("parallel_searcher", parallel_searcher)
    builder.add_node("search_worker", search_worker)
    builder.add_node("evaluator", evaluator)
    builder.add_node("writer", writer)

    # 图结构
    builder.set_entry_point("planner")
    builder.add_edge("planner", "parallel_searcher")

    # parallel_searcher → 条件扇出到 workers 或直接 evaluator
    builder.add_conditional_edges(
        "parallel_searcher",
        route_to_search_workers,
        {"evaluator": "evaluator"},
    )
    builder.add_edge("search_worker", "evaluator")

    # evaluator → 条件路由：继续搜索 / 重试 / 生成报告
    builder.add_conditional_edges(
        "evaluator",
        should_continue_wave,
        {
            "parallel_searcher": "parallel_searcher",
            "planner": "planner",
            "writer": "writer",
        },
    )

    # writer → END
    builder.add_edge("writer", END)

    graph = builder.compile(checkpointer=checkpointer, debug=settings.DEBUG)
    graph = graph.with_config(recursion_limit=50)
    return graph


# LangGraph Studio 入口
def get_graph_for_studio():
    """Studio 入口：使用 PostgreSQL checkpointer。"""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow, dict_row
    from psycopg_pool import AsyncConnectionPool

    conninfo = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

    pool = AsyncConnectionPool[AsyncConnection[DictRow]](
        conninfo,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )

    checkpointer = AsyncPostgresSaver(pool)
    return build_graph(checkpointer=checkpointer)
