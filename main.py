"""AdaptiveSearchAgent 测试入口模块。

用于快速验证 Planner 节点的功能。
后续会替换为完整的 LangGraph StateGraph 工作流编排。
"""

import asyncio
from typing import TYPE_CHECKING

from src.agents.planner import planner

if TYPE_CHECKING:
    from src.state import AgentState


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
    }

    print(f"输入查询: {state['user_query']}")
    print("正在调用 Planner 节点生成搜索计划...")

    # 调用 Planner 节点 —— 这是 LangGraph 中的单个节点函数
    # 返回的是部分状态更新 {"plan": [...]}，LangGraph 会自动合并
    result = await planner(state)

    print(f"生成的搜索关键词: {result['plan']}")


if __name__ == "__main__":
    asyncio.run(test_planner())
