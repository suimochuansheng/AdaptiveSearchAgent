# """
# 调用公共 execute_agent_stream 或直接使用 graph.astreaminvoke 的示例入口模块。
# """

# import asyncio
# from langgraph.types import Command
# from src.agents.planner import planner
# from src.state import AgentState
# from src.utils.metrics import metrics
# from src.graph_factory import build_graph, get_graph


# async def test_planner() -> None:
#     """测试 Planner 节点：输入查询，打印生成的搜索关键词列表。

#     构造一个模拟的 AgentState 字典，调用 planner 异步节点函数，
#     验证 LLM 是否能正确生成搜索计划。
#     """
#     # 模拟 LangGraph 运行时传入的初始状态
#     state: AgentState = {
#         "user_query": "Qwen2.5 性能评测",
#         "iteration": 0,
#         "missing_info": "",
#         "plan": [],
#         "search_results": [],
#         "confidence_score": 0.0,
#         "retry_keywords": [],
#         "final_report": "",
#         "human_approved": False,
#         "task_id": "test",
#         "total_tokens": 0,
#         "input_tokens": 0,
#         "output_tokens": 0,
#         "current_llm": "ollama",
#         "pending_keywords": [],
#         "_batch_keywords": [],
#         "thread_id": "test",
#     }

#     print(f"输入查询: {state['user_query']}")
#     print("正在调用 Planner 节点生成搜索计划...")

#     # 调用 Planner 节点 —— 这是 LangGraph 中的单个节点函数
#     # 返回的是部分状态更新 {"plan": [...]}，LangGraph 会自动合并
#     result = await planner(state)

#     print(f"生成的搜索关键词: {result['plan']}")

# async def run_agent(user_query: str,thread_id: str) -> tuple[str, dict]:
#     """运行 Agent，返回 (最终报告, KPI 数据字典)。

#     返回的 KPI 字典可直接传给 print_kpi_dashboard() 生成 CLI 指标表格。
#     """
#     import time as time_module

#     from src.utils.llm_utils import get_model_switch_count, reset_model_switch_count

#     # 初始化全局状态，包含用户查询和其他必要字段
#     graph = await get_graph()
#     # 重置模型切换计数器（每个新会话从 0 开始）
#     reset_model_switch_count()

#     # 2. 定义 LangGraph 标准 config（和 main 里完全一致）
#     config = {
#         "configurable": {
#             "thread_id": thread_id,
#             "llm_provider": "ollama",  # 可改为 "deepseek"
#         }
#     }
#     # 3. 初始化全局状态
#     initial_state: AgentState = {
#         "user_query": user_query,
#         "plan": [],
#         "search_results": [],
#         "confidence_score": 0.0,
#         "missing_info": "",
#         "retry_keywords": [],
#         "iteration": 0,
#         "final_report": "",
#         "human_approved": False,  # 等待人工审批
#         "thread_id": thread_id,  # 每次运行生成唯一线程 ID
#         "total_tokens": 0,
#         "input_tokens": 0,
#         "output_tokens": 0,
#         "current_llm": "ollama",
#         "pending_keywords": [],
#         "_batch_keywords": [],
#         # 注意：state 内部不需要存 config，config 是传给 ainvoke 的参数
#     }
#     # 4. 构建图（带异步 SQLite checkpointer）并执行
#     import aiosqlite
#     #  perf_counter()系统最高精度计时器,记下开始时间，后续计算总耗时用
#     start_time = time_module.perf_counter()
#     # 注意：aiosqlite 连接需要在 async 环境中创建，因此放在 graph.ainvoke 外层
#     async with aiosqlite.connect("data/checkpoints.db") as conn:
#         # 兼容性修补：aiosqlite 0.22.x 缺少 is_alive()，langgraph 2.0.x 需要它
#         conn.is_alive = lambda: True  # type: ignore[method-assign]
#         # 中断前那一瞬间的最新状态会被 checkpointer 自动保存到 SQLite 数据库中，方便后续查询和恢复
#         checkpointer = AsyncSqliteSaver(conn)
#         # checkpointer传入 build_graph，启用图的持久化功能，图的执行状态会被自动保存到 SQLite 数据库中
#         graph = build_graph(checkpointer=checkpointer)

#         # 首次执行：运行到 human_approval 的 interrupt() 处自动挂起
#         final_state = await graph.ainvoke(initial_state, config)

#         # 检查 state 中是否有中断标记（Human-in-the-loop）
#         if "__interrupt__" in final_state:
#             user_input = input(">>> 请审批 (yes/no): ").strip()
#             # Command 是 LangGraph 官方专门用来「控制工作流执行、中断后恢复、传递任务指令」的专用数据结构
#             # Command 是控制指令，告诉langgraph从上面的interrupt处继续执行，resume=user_input 是把用户输入的审批结果传回去，供 human_approval 节点使用
#             # Command 恢复中断前那一瞬间的最新状态
#             final_state = await graph.ainvoke(Command(resume=user_input), config)
#     # 计算总耗时
#     elapsed = time_module.perf_counter() - start_time

#     # 5. 组装 KPI 数据（供 CLI 仪表盘使用）
#     kpi_data: dict = {
#         "total_tokens": final_state.get("total_tokens", 0),
#         "input_tokens": final_state.get("input_tokens", 0),
#         "output_tokens": final_state.get("output_tokens", 0),
#         "current_llm": final_state.get("current_llm", "ollama"),
#         "model_switches": get_model_switch_count(),
#         "thread_id": thread_id,
#         "elapsed_seconds": elapsed,
#         "human_approved": final_state.get("human_approved"),
#         "confidence_score": final_state.get("confidence_score", 0.0),
#     }

#     # 6. 返回报告 + KPI
#     return final_state["final_report"], kpi_data


# if __name__ == "__main__":
#     # =============测试======================
#     # asyncio.run(test_planner())
#     # ======================单轮循环==========================
#     from src.utils.cli_report import print_kpi_dashboard

#     query = input("请输入您的问题：").strip()
#     if not query:
#         query = "LangGraph 和 LangChain 的区别"
#     # 启动异步程序的「总开关」，同时获取报告和 KPI 数据
#     report, kpi_data = asyncio.run(run_agent(query))
#     print("\n" + "=" * 60)
#     print(report)
#     print("\n" + "=" * 60)
#     print(f"Metrics 数据已保存至: {metrics.data_path}")
#     # 输出 CLI KPI 仪表盘
#     print_kpi_dashboard(kpi_data)
#     # ========================================================
#     # ========================================================


# # =============================================================================
# # LangGraph CLI 入口：供 langgraph dev / langgraph up / langgraph build 使用
# # =============================================================================
# # build_graph() 返回一个已编译的 StateGraph（含所有节点、边、条件路由）。
# # LangGraph CLI 读取此变量后，会用自己的 checkpointer（如 PostgresSaver）
# # 重新编译以适配 dev/生产环境的持久化需求，开发者无需手动传入 checkpointer。
# cli_graph = build_graph()
