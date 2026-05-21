"""AdaptiveSearchAgent 测试入口模块。

用于快速验证 Planner 节点的功能。
后续会替换为完整的 LangGraph StateGraph 工作流编排。
"""

import asyncio
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from config import settings
from src.agents.evaluator import evaluator
from src.agents.parallel_searcher import parallel_searcher, route_to_search_workers
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
        "input_tokens": 0,
        "output_tokens": 0,
        "current_llm": "ollama",
        "pending_keywords": [],
        "_batch_keywords": [],
        "thread_id": "test",
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
        # metrics.record_iterations(state["iteration"])
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

    # 所有批次完成，判断是否需要重试或结束
    if (
        state["confidence_score"] >= settings.confidence_threshold
        or state["iteration"] >= settings.max_iterations
    ):
        return "human_approval"
    else:
        return "planner"


async def human_approval(state: AgentState) -> dict:
    """人工审批节点：在生成报告前请求用户确认"""
    # preview = state.get("final_report", "")[:200] if state.get("final_report") else "（暂无预览）"
    search_count = len(state.get("search_results", []))
    confidence = state.get("confidence_score", 0)
    preview = f"已搜索 {search_count} 个关键词，置信度 {confidence:.0%}"

    # 挂起图，等待用户输入
    prompt = f"请审批是否生成最终报告？\n预览：{preview}\n输入 'yes' 批准，'no' 拒绝："
    # interrupt() 是 LangGraph 官方提供的「中断当前节点执行、挂起整个图、等待外部输入后恢复执行」的专用函数
    # interrupt() 是强制再写入一次（最终快照）
    user_input = interrupt(prompt)
    approved = user_input.strip().lower() == "yes"
    return {"human_approved": approved}


def after_approval(state: AgentState) -> str:
    """审批后的路由：批准则 writer，否则结束"""
    return "writer" if state.get("human_approved") else END  # type: ignore[no-any-return]


def build_graph(checkpointer: AsyncSqliteSaver | None = None):
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
    builder.add_node("human_approval", human_approval)  # 新增人工审批节点

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
        after_approval,
        {
            "writer": "writer",
            END: END,
        },
    )
    graph = builder.compile(checkpointer=checkpointer, debug=True)
    return graph


async def run_agent(user_query: str) -> tuple[str, dict]:
    """运行 Agent，返回 (最终报告, KPI 数据字典)。

    返回的 KPI 字典可直接传给 print_kpi_dashboard() 生成 CLI 指标表格。
    """
    import time as time_module

    from src.utils.llm_utils import get_model_switch_count, reset_model_switch_count

    # 初始化全局状态，包含用户查询和其他必要字段
    # 1. 生成唯一 thread_id，唯一的对话会话编号，不是Python 多线程、异步任务线程
    thread_id = str(uuid.uuid4())

    # 重置模型切换计数器（每个新会话从 0 开始）
    reset_model_switch_count()

    # 2. 定义 LangGraph 标准 config（和 main 里完全一致）
    config = {
        "configurable": {
            "thread_id": thread_id,
            "llm_provider": "ollama",  # 可改为 "deepseek"
        }
    }
    # 3. 初始化全局状态
    initial_state: AgentState = {
        "user_query": user_query,
        "plan": [],
        "search_results": [],
        "confidence_score": 0.0,
        "missing_info": "",
        "retry_keywords": [],
        "iteration": 0,
        "final_report": "",
        "human_approved": False,  # 等待人工审批
        "thread_id": thread_id,  # 每次运行生成唯一线程 ID
        "task_id": thread_id,  # 任务唯一标识，与 thread_id 一致
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "current_llm": "ollama",
        "pending_keywords": [],
        "_batch_keywords": [],
        # 注意：state 内部不需要存 config，config 是传给 ainvoke 的参数
    }
    # 4. 构建图（带异步 SQLite checkpointer）并执行
    import aiosqlite

    #  perf_counter()系统最高精度计时器,记下开始时间，后续计算总耗时用
    start_time = time_module.perf_counter()
    # 注意：aiosqlite 连接需要在 async 环境中创建，因此放在 graph.ainvoke 外层
    async with aiosqlite.connect("data/checkpoints.db") as conn:
        # 兼容性修补：aiosqlite 0.22.x 缺少 is_alive()，langgraph 2.0.x 需要它
        conn.is_alive = lambda: True  # type: ignore[method-assign, attr-defined]
        # 中断前那一瞬间的最新状态会被 checkpointer 自动保存到 SQLite 数据库中，方便后续查询和恢复
        checkpointer = AsyncSqliteSaver(conn)
        # checkpointer传入 build_graph，启用图的持久化功能，图的执行状态会被自动保存到 SQLite 数据库中
        graph = build_graph(checkpointer=checkpointer)

        # 首次执行：运行到 human_approval 的 interrupt() 处自动挂起
        final_state = await graph.ainvoke(initial_state, config)

        # 检查 state 中是否有中断标记（Human-in-the-loop）
        if "__interrupt__" in final_state:
            user_input = input(">>> 请审批 (yes/no): ").strip()
            # Command 是 LangGraph 官方专门用来「控制工作流执行、中断后恢复、传递任务指令」的专用数据结构
            # Command 是控制指令，告诉langgraph从上面的interrupt处继续执行，resume=user_input 是把用户输入的审批结果传回去，供 human_approval 节点使用
            # Command 恢复中断前那一瞬间的最新状态
            final_state = await graph.ainvoke(Command(resume=user_input), config)
    # 计算总耗时
    elapsed = time_module.perf_counter() - start_time

    # 5. 组装 KPI 数据（供 CLI 仪表盘使用）
    kpi_data: dict = {
        "total_tokens": final_state.get("total_tokens", 0),
        "input_tokens": final_state.get("input_tokens", 0),
        "output_tokens": final_state.get("output_tokens", 0),
        "current_llm": final_state.get("current_llm", "ollama"),
        "model_switches": get_model_switch_count(),
        "thread_id": thread_id,
        "elapsed_seconds": elapsed,
        "human_approved": final_state.get("human_approved"),
        "confidence_score": final_state.get("confidence_score", 0.0),
    }

    # 6. 返回报告 + KPI
    return final_state["final_report"], kpi_data


if __name__ == "__main__":
    # =============测试======================
    # asyncio.run(test_planner())
    # ======================单轮循环==========================
    from src.utils.cli_report import print_kpi_dashboard

    query = input("请输入您的问题：").strip()
    if not query:
        query = "LangGraph 和 LangChain 的区别"
    # 启动异步程序的「总开关」，同时获取报告和 KPI 数据
    report, kpi_data = asyncio.run(run_agent(query))
    print("\n" + "=" * 60)
    print(report)
    print("\n" + "=" * 60)
    print(f"Metrics 数据已保存至: {metrics.data_path}")
    # 输出 CLI KPI 仪表盘
    print_kpi_dashboard(kpi_data)
    # ========================================================
    # ========================================================


# =============================================================================
# LangGraph CLI 入口：供 langgraph dev / langgraph up / langgraph build 使用
# =============================================================================
# build_graph() 返回一个已编译的 StateGraph（含所有节点、边、条件路由）。
# LangGraph CLI 读取此变量后，会用自己的 checkpointer（如 PostgresSaver）
# 重新编译以适配 dev/生产环境的持久化需求，开发者无需手动传入 checkpointer。
cli_graph = build_graph()
