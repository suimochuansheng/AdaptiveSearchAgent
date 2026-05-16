"""搜索工作节点：执行计划中的所有关键词搜索，串行累加结果。"""

from src.tools.search import search_tavily
from src.utils.logger import log_node


# 装饰器：给这个节点打日志，标记名字叫 search_worker（方便调试、看执行记录）
@log_node("search_worker")
# 定义异步函数：智能体的“搜索工人”节点
async def search_worker(state: dict) -> dict:
    """单个关键词搜索节点，由 parallel_searcher 通过 Send 调用。

    LangGraph 的 Send API 会将当前全局状态与 Send.arg 合并后传入，
    因此 state 是 AgentState + {"keyword": str} 的超集，
    使用 dict 类型以兼容 TypedDict 无法表达的扩展字段。
    """

    keyword = state["keyword"]  # 从输入状态中提取关键词
    content = await search_tavily(keyword)

    # 4. 返回搜索结果，存入智能体状态的 search_results 字段
    return {"search_results": [{"keyword": keyword, "content": content}]}
