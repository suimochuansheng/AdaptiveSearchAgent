"""搜索工作节点：执行计划中的所有关键词搜索，串行累加结果。"""

from typing import Any

from src.state import AgentState
from src.tools.search import search_tavily
from src.utils.logger import log_node


# 装饰器：给这个节点打日志，标记名字叫 search_worker（方便调试、看执行记录）
@log_node("search_worker")
# 定义异步函数：智能体的“搜索工人”节点
async def search_worker(state: AgentState) -> dict:
    """文档说明：遍历 plan 列表，对每个关键词执行 Tavily 搜索，返回结果列表。"""

    # 1. 从智能体状态里 取出 plan（就是你之前的搜索关键词列表）
    plan: list[str] = state["plan"]

    # 2. 创建空列表，用来存放所有搜索结果
    results: list[dict[str, Any]] = []

    # 3. 遍历每个关键词，挨个搜索
    for keyword in plan:
        # 异步调用 Tavily 搜索（不阻塞程序）
        content = await search_tavily(keyword)
        # 把【关键词 + 搜索内容】存起来
        results.append(
            {
                "keyword": keyword,  # 搜的什么词
                "content": content,  # 搜到的内容
            }
        )

    # 4. 返回搜索结果，存入智能体状态的 search_results 字段
    return {"search_results": results}
