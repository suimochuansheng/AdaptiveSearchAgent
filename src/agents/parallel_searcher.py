"""并行搜索分发节点：将关键词列表拆分为多个 Send 任务，实现搜索并行化。"""

from langgraph.types import Send

from config import settings
from src.state import AgentState
from src.utils.logger import log_node


@log_node("parallel_searcher")
async def parallel_searcher(state: AgentState) -> list[Send]:
    """根据待搜索关键词列表，分批分发搜索任务（只分发，不搜索）。

    关键词来源优先级：
    1. pending_keywords — 上一波未处理完的剩余关键词
    2. retry_keywords — Evaluator 建议的补充搜索关键词
    3. plan — Planner 生成的初始搜索计划

    自动对已搜索过的关键词去重，每次最多分发 max_concurrent_searches 个任务，
    剩余关键词写入 state["pending_keywords"] 供下一轮继续处理。

    Args:
        state: 当前全局状态。

    Returns:
        list[Send]: 发给 search_worker 节点的并行任务列表。
    """
    # 确定关键词来源
    if state.get("pending_keywords"):
        keywords = state["pending_keywords"]
        source = "pending"
    elif state.get("retry_keywords"):
        keywords = state["retry_keywords"]
        source = "retry"
    else:
        keywords = state["plan"]
        source = "plan"

    # 去重：过滤掉已经搜索过的关键词（基于 search_results 中的 keyword）
    existing_keywords = {res["keyword"] for res in state.get("search_results", [])}
    # keywords不区分source标记，因为无论是来自 pending、retry 还是 plan，只要之前搜索过了都不应该再搜索了。
    new_keywords = [kw for kw in keywords if kw not in existing_keywords]

    if not new_keywords:  # 为空 意味着当前来源的关键词都已搜索过了，
        # 为了避免下次进入 parallel_searcher 时再次读取这些已经处理过的关键词（导致无限循环），需要主动清空 state["pending_keywords"]
        if source == "pending":
            state["pending_keywords"] = []
        return []

    # 分批：每次最多发送 max_concurrent_searches 个
    batch = new_keywords[: settings.max_concurrent_searches]
    # 剩余的关键词留到下一轮继续处理
    remaining = new_keywords[settings.max_concurrent_searches :]

    # 更新 state 中的 pending_keywords
    if remaining:
        state["pending_keywords"] = remaining
    else:
        # 如果没有剩余且来源是 pending，说明当前批次的 pending_keywords 已经全部发送完毕，清空该字段，避免下次错误读取。
        if source == "pending":
            state["pending_keywords"] = []
        # 注意：如果来源是 retry 或 plan，剩余关键词应当作为新的 pending 等待下一 wave
        # 但上述代码已经将 remaining 赋给了 pending_keywords，正确。
        #

    # 生成 Send 任务
    sends = [Send("search_worker", {"keyword": kw}) for kw in batch]
    return sends
