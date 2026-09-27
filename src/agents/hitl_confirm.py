"""HITL 高成本搜索前置确认节点。

在 parallel_searcher 执行前，估算本批次搜索成本。当 iteration == 0 且
当前批次关键词数量 >= 5 时，调用 interrupt() 暂停图执行，由用户在
前端确认是否继续，避免在未经确认的情况下产生高额搜索费用。

注意：此处复刻了 parallel_searcher 的批次计算规则
（pending > retry > plan，去重后按 max_concurrent_searches 切片），
保证「确认的对象」与「即将执行的批次」完全一致。
"""

from langgraph.types import interrupt

from config import settings
from src.state import AgentState
from src.utils.logger import log_node

# 触发确认的关键词数量阈值
_CONFIRM_THRESHOLD = 5
# 单次搜索估算成本（元/关键词）
_COST_PER_KEYWORD = 0.02


def _compute_batch(state: AgentState) -> list[str]:
    """按 parallel_searcher 相同规则计算本批次待搜索关键词。"""
    if state.get("pending_keywords"):
        keywords = state["pending_keywords"]
    elif state.get("retry_keywords"):
        keywords = state["retry_keywords"]
    else:
        keywords = state["plan"]

    existing = {res["keyword"] for res in state.get("search_results", [])}
    new_keywords = [kw for kw in keywords if kw not in existing]
    return new_keywords[: settings.max_concurrent_searches]


@log_node("hitl_confirm")
async def hitl_confirm(state: AgentState) -> dict:
    """检查本批次搜索成本，必要时请求用户确认。

    Returns:
        {"hitl_decision": "confirmed" | "cancelled"}
        - confirmed：放行到 parallel_searcher
        - cancelled：路由到 writer 返回取消消息
    """
    batch = _compute_batch(state)
    iteration = state.get("iteration", 0)

    if not (len(batch) >= _CONFIRM_THRESHOLD and iteration == 0):
        return {"hitl_decision": "confirmed"}

    count = len(batch)
    payload = {
        "type": "cost_confirm",
        "keyword_count": count,
        "estimated_cost": count * _COST_PER_KEYWORD,
        "message": (
            f"本次需要搜索 {count} 个关键词，预计消耗约 {count * _COST_PER_KEYWORD} 元，是否继续？"
        ),
    }

    decision = interrupt(payload)
    decision = str(decision).strip().lower()
    return {"hitl_decision": "confirmed" if decision == "confirmed" else "cancelled"}
