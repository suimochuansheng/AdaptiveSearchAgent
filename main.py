"""AdaptiveSearchAgent 测试入口模块。

用于快速验证 Planner 节点的功能。
后续会替换为完整的 LangGraph StateGraph 工作流编排。
"""

import asyncio
import uuid

from langgraph.graph import END, StateGraph

from config import settings
from src.agents.evaluator import evaluator
from src.agents.planner import planner
from src.agents.search_worker import search_worker
from src.agents.writer import writer
from src.state import AgentState
from src.utils.metrics import metrics


async def test_planner() -> None:
    """测试 Planner 节点：输入查询，打印生成的搜索关键词列表。

    构造一个模拟的 AgentState 字典，调用 planner 异步节点函数，
    验证 LLM 是否能正确生成搜索计划。
    """
    # 模拟 LangGraph 运行时传入的初始状态
    state: AgentState = {
        "user_query": "Qwen2.5 性能评测",
        "iteration": 0,
        "missing_info": "",
        "plan": [],
        "search_results": [],
        "confidence_score": 0.0,
        "retry_keywords": [],
        "final_report": "",
        "human_approved": False,
        "task_id": "test",
        "total_tokens": 0,
    }

    print(f"输入查询: {state['user_query']}")
    print("正在调用 Planner 节点生成搜索计划...")

    # 调用 Planner 节点 —— 这是 LangGraph 中的单个节点函数
    # 返回的是部分状态更新 {"plan": [...]}，LangGraph 会自动合并
    result = await planner(state)

    print(f"生成的搜索关键词: {result['plan']}")


def should_continue(state: AgentState) -> str:
    """条件边判断：是否继续重试或进入报告生成。"""
    # 置信度达标 或 达到最大迭代次数
    if (
        state["confidence_score"] >= settings.confidence_threshold
        or state["iteration"] >= settings.max_iterations
    ):
        # 记录本次查询收敛所需的迭代次数
        metrics.record_iterations(state["iteration"])
        return "writer"
    else:
        return "retry"


def build_graph():
    """
    构建 LangGraph 工作流（状态机执行图）
    作用：定义 Agent 的执行节点、跳转逻辑、判断条件，最终编译成可运行的工作流
    """
    # 1. 创建 StateGraph 构建器，绑定共享状态类型 AgentState
    # 所有节点之间共享、读写同一个 AgentState 数据
    builder = StateGraph(AgentState)

    # 2. 添加工作流节点（node = 一个执行函数/步骤）
    # 参数：(节点名称, 节点对应的执行函数)
    builder.add_node("planner", planner)  # 规划节点：生成搜索关键词
    builder.add_node("search", search_worker)  # 搜索节点：执行联网搜索
    builder.add_node("evaluator", evaluator)  # 评估节点：判断信息是否足够
    builder.add_node("writer", writer)  # 写作节点：生成最终报告

    # 3. 设置工作流【入口起点】
    # 程序启动后第一个执行的节点：planner
    builder.set_entry_point("planner")

    # 4. 添加【固定顺序边】(无条件直接跳转)
    # planner 执行完 → 自动跳转到 search
    builder.add_edge("planner", "search")
    # search 执行完 → 自动跳转到 evaluator
    builder.add_edge("search", "evaluator")

    # 5. 添加【条件边】(根据判断结果决定跳转到哪里)
    # 这是循环重试的核心！
    builder.add_conditional_edges(
        source="evaluator",  # 来源节点：从 evaluator 出发
        path=should_continue,  # 条件判断函数：返回 "retry" 或 "writer"
        path_map={  # 映射关系：返回值 → 目标节点
            "retry": "planner",  # 返回 retry → 回到 planner 重新搜索
            "writer": "writer",  # 返回 writer → 进入最终报告生成
        },
    )

    # 6. writer 执行完成 → 结束整个工作流（END 是 LangGraph 内置结束标记）
    builder.add_edge("writer", END)

    # 7. 编译生成可运行的工作流实例（类似编译成可执行程序）
    return builder.compile()


async def run_agent(user_query: str) -> str:
    """运行 Agent，返回最终报告。"""
    # 初始化全局状态，包含用户查询和其他必要字段
    initial_state: AgentState = {
        "user_query": user_query,
        "plan": [],
        "search_results": [],
        "confidence_score": 0.0,
        "missing_info": "",
        "retry_keywords": [],
        "iteration": 0,
        "final_report": "",
        "human_approved": True,  # 暂时自动批准
        "task_id": str(uuid.uuid4()),
        "total_tokens": 0,
    }
    # 构建 LangGraph 工作流
    graph = build_graph()
    # graph.ainvoke() → 异步非阻塞（必须搭配 await）
    final_state = await graph.ainvoke(initial_state)
    # 从最终状态中提取报告文本
    report: str = final_state["final_report"]
    return report


if __name__ == "__main__":
    # =============测试======================
    # asyncio.run(test_planner())
    # ======================单轮循环==========================
    query = input("请输入您的问题：").strip()
    if not query:
        query = "LangGraph 和 LangChain 的区别"
    # 启动异步程序的「总开关」
    report = asyncio.run(run_agent(query))
    print("\n" + "=" * 60)
    print(report)
    print("\n" + "=" * 60)
    print(f"Metrics 数据已保存至: {metrics.data_path}")
    # ================================================================
