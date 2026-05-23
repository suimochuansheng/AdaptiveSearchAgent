# 提供 get_graph() 异步工厂函数，封装图的构建与编译。
# src/graph_factory.py
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
_compiled_graph = None


async def get_graph():
    """异步获取全局 graph 实例（懒加载），绑定 PostgreSQL checkpointer。"""
    global _compiled_graph
    if _compiled_graph is None:
        saver = get_checkpointer()  # 从 checkpointer 模块获取全局唯一的 Saver
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
    """
    综合判断：先处理未完成的 wave（pending_keywords），
    若无 pending 则根据置信度和迭代次数决定是重试还是结束。
    """
    # 如果还有未搜索的关键词，继续下一批（不经过 planner）
    if state.get("pending_keywords"):
        return "parallel_searcher"

    # 所有批次完成，条件边判断：置信度达标或达到最大迭代次数 -> human_approval，否则 planner
    if (
        state["confidence_score"] >= settings.confidence_threshold
        or state["iteration"] >= settings.max_iterations
    ):
        return "human_approval"
    else:
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
