"""并行搜索分发节点：将关键词列表拆分为多个 Send 任务，实现搜索并行化。

本模块包含两个协作部分：
1. parallel_searcher 节点 — 准备搜索关键词批次，更新 state
2. route_to_search_workers 路由函数 — 由 add_conditional_edges 调用，生成 Send 扇出

LangGraph 1.1.x 要求：固定边连接的节点必须返回 dict，Send 扇出只能由
conditional_edges 的路由函数返回。因此将"分批 + 扇出"拆分为两步。
"""

from langgraph.types import Send

from config import settings
from src.state import AgentState
from src.utils.logger import log_node


@log_node("parallel_searcher")
async def parallel_searcher(state: AgentState) -> dict:
    """准备搜索关键词批次，写入 state 供路由函数使用。

    关键词来源优先级：
    1. pending_keywords — 上一波未处理完的剩余关键词
    2. retry_keywords — Evaluator 建议的补充搜索关键词
    3. plan — Planner 生成的初始搜索计划

    自动对已搜索过的关键词去重，将本批次关键词和剩余关键词分别写入
    state["_batch_keywords"] 和 state["pending_keywords"]。
    实际的 Send 扇出由 route_to_search_workers 完成。

    Args:
        state: 当前全局状态。

    Returns:
        dict: 包含 _batch_keywords / pending_keywords 的部分状态更新。
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

    # 去重：过滤掉已经搜索过的关键词
    existing_keywords = {res["keyword"] for res in state.get("search_results", [])}
    new_keywords = [kw for kw in keywords if kw not in existing_keywords]

    if not new_keywords:
        # 当前来源无可搜索关键词，清空 pending 防止死循环
        return {"_batch_keywords": [], "pending_keywords": []}

    # 分批：每次最多发送 max_concurrent_searches 个
    batch = new_keywords[: settings.max_concurrent_searches]
    remaining = new_keywords[settings.max_concurrent_searches :]

    # 先定义变量，统一类型
    new_pending: list[str]
    # 计算新的 pending_keywords（通过返回 dict 更新，不依赖原地修改）
    if remaining:
        new_pending = remaining
    elif source == "pending":
        # pending 来源且无剩余 → 清空
        new_pending = []
    else:
        # plan/retry 来源且无剩余 → 保持原有 pending（可能已有之前遗留的）
        new_pending = state.get("pending_keywords", [])

    return {"_batch_keywords": batch, "pending_keywords": new_pending}


def route_to_search_workers(state: AgentState) -> list[Send] | str:
    """条件边路由函数：根据 _batch_keywords 生成 Send 扇出任务。

    由 add_conditional_edges("parallel_searcher", ...) 调用。
    当有本批次关键词时返回 list[Send] 实现并行扇出；
    无关键词时返回 "evaluator" 跳过搜索直接进入评估。

    Args:
        state: 当前全局状态（parallel_searcher 节点已更新 _batch_keywords）。

    Returns:
        list[Send] 或字符串 "evaluator"。
    """
    batch = state.get("_batch_keywords", [])
    if not batch:
        return "evaluator"
    # print(f"路由函数生成 Send 任务，关键词批次: {batch}")
    Sends = [Send("search_worker", {"keyword": kw}) for kw in batch]
    return Sends
