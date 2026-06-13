# 提供 get_graph() 异步工厂函数，封装图的构建与编译。
# src/graph_factory.py
import asyncio

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from config import settings
from src.agents.evaluator import evaluator
from src.agents.human_approval import after_approval_node, human_approval_node
from src.agents.parallel_searcher import parallel_searcher, route_to_search_workers
from src.agents.planner import planner
from src.agents.search_worker import search_worker
from src.agents.writer import writer
from src.checkpointer import get_checkpointer
from src.state import AgentState

# 全局 graph 实例占位符（在异步启动后赋值）
# 函数式轻量单例，舍弃了完整的单例类，借助 Python 模块全局特性，用一个变量 + 判断逻辑，
# 实现了 “全局唯一、只初始化一次” 的核心效果，非常适合 LangGraph 这类单一实例缓存场景。
_compiled_graph = None
_graph_lock = asyncio.Lock()  # 异步锁


async def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        async with _graph_lock:
            # 双重判空：防止锁等待期间其他协程已完成初始化
            if _compiled_graph is None:
                saver = get_checkpointer()
                _compiled_graph = build_graph(checkpointer=saver)
    return _compiled_graph


def should_continue(state: AgentState) -> str:
    """条件边判断：是否继续重试或进入报告生成。"""
    # 置信度达标 或 达到最大迭代次数
    if (
        state["confidence_score"] >= settings.confidence_threshold
        or state["iteration"] >= settings.max_iterations
    ):
        return "writer"
    else:
        return "retry"


def should_continue_wave(state: AgentState) -> str:
    """综合判断：先检查置信度/迭代，再处理未完成的 wave。

    优先级：
    1. 置信度达标或达到最大迭代 → 直接 human_approval，丢弃剩余 pending 关键词
    2. 有 pending_keywords → 继续下一批
    3. 否则回到 planner 重试
    """
    # 置信度达标或达最大迭代 → 提前结束，无需处理剩余关键词
    if (
        state["confidence_score"] >= settings.confidence_threshold
        or state["iteration"] >= settings.max_iterations
    ):
        return "human_approval"

    # 如果还有未搜索的关键词，继续下一批（不经过 planner）
    if state.get("pending_keywords"):
        return "parallel_searcher"

    # 置信度不足且无 pending → 回 planner 生成新关键词
    return "planner"


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """构建 LangGraph 工作流（状态机执行图）-并行 search_worker 版本，包含人工审批节点

    图结构：
        planner → parallel_searcher → (条件边: Send 扇出到 search_worker
          或跳过搜索直接到 evaluator)
        search_worker → evaluator → (条件边: parallel_searcher 继续下一波
          / planner 重试 / writer 结束)
        writer → END

    Send 扇出机制：
        parallel_searcher 节点负责准备关键词批次（写入 _batch_keywords），
        随后的 add_conditional_edges 调用 route_to_search_workers 路由函数，
        该函数返回 list[Send] 实现 search_worker 的并行扇出。
        所有 search_worker 完成后通过固定边汇聚到 evaluator。

    Args:
        checkpointer: 可选，AsyncSqliteSaver 实例，传入则启用 checkpoint 持久化。
    """
    builder = StateGraph(AgentState)
    # 加载节点
    builder.add_node("planner", planner)
    builder.add_node("parallel_searcher", parallel_searcher)
    builder.add_node("search_worker", search_worker)
    builder.add_node("evaluator", evaluator)
    builder.add_node("writer", writer)
    builder.add_node("human_approval", human_approval_node)  # 新增人工审批节点
    # 图结构
    builder.set_entry_point("planner")
    builder.add_edge("planner", "parallel_searcher")

    # 条件边：parallel_searcher 之后，根据关键词批次决定 Send 扇出或跳过
    builder.add_conditional_edges(
        "parallel_searcher",
        route_to_search_workers,
        {
            "evaluator": "evaluator",  # 无关键词时直接进入评估
        },
    )
    # search_worker 完成后汇聚到 evaluator
    builder.add_edge("search_worker", "evaluator")

    # 条件边：从 evaluator 出发，根据 should_continue_wave 决定下一节点
    builder.add_conditional_edges(
        "evaluator",
        should_continue_wave,
        {
            "parallel_searcher": "parallel_searcher",  # 继续下一批搜索
            "planner": "planner",  # 重试：回到 planner
            "human_approval": "human_approval",  # 指向人工审核节点
        },
    )
    # 审批后的路由函数
    builder.add_conditional_edges(
        "human_approval",
        after_approval_node,
        {
            "writer": "writer",
            END: END,
        },
    )
    # 编译时绑定 checkpointer
    graph = builder.compile(checkpointer=checkpointer, debug=True)
    return graph


# 专门为 LangGraph Studio 添加的无参工厂
def get_graph_for_studio():
    """Studio 入口：使用 PostgreSQL checkpointer，支持中断恢复和持久化调试。

    LangGraph Studio 通过 langgraph.json 中配置的此函数获取编译后的图。
    使用数据库持久化 checkpoint 后：
    - Studio 中运行到 human_approval 节点的 interrupt() 会自动挂起
    - 在 Studio UI 中可查看中断状态并手动恢复
    - 刷新页面后中断状态不丢失（InMemorySaver 会丢失）
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow, dict_row
    from psycopg_pool import AsyncConnectionPool

    conninfo = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

    # 为 Studio 创建独立的连接池（不影响 api_main.py 中的全局连接池）
    pool = AsyncConnectionPool[AsyncConnection[DictRow]](
        conninfo,
        min_size=1,
        max_size=5,  # Studio 只用单用户调试，不需要大连接池
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )

    checkpointer = AsyncPostgresSaver(pool)
    return build_graph(checkpointer=checkpointer)
