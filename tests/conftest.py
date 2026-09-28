"""pytest 共享夹具和工具函数。"""

from typing import Any

from src.state import AgentState


def create_test_state(overrides: dict[str, Any] | None = None) -> AgentState:
    """构造一个字段齐全的 AgentState，仅需覆盖测试关心的字段。

    Args:
        overrides: 需要覆盖的字段字典，与默认值合并。

    Returns:
        合法的 AgentState 字典，所有字段均已填充。
    """
    defaults: AgentState = {
        "user_query": "",
        "plan": [],
        "search_results": [],
        "confidence_score": 0.0,
        "missing_info": "",
        "retry_keywords": [],
        "iteration": 0,
        "final_report": "",
        "human_approved": False,
        "task_id": "test-id",
        "total_tokens": 0,
        "pending_keywords": [],
    }
    if overrides:
        defaults.update(overrides)  # type: ignore[typeddict-item]
    return defaults
