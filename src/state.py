"""AdaptiveSearchAgent 的 LangGraph 状态定义模块。

定义 Agent 在整个搜索工作流中传递和累积的状态数据结构。
LangGraph 中各节点通过读取和写入此状态来协作完成搜索任务，
节点仅返回需要更新的字段字典，框架自动完成状态合并。
"""

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

# ── 搜索结果的哨兵值，表示“清空累加器” ──
_RESET_SENTINEL = "__RESET__"


def reduce_accum(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """自定义聚合器：支持并行合并 + 重置信号。

    行为：
    1. 如果 right[0] 等于 _RESET_SENTINEL → 直接返回空列表 []（清零）。
    2. 否则执行标准追加：(left or []) + right。
    """
    if right and isinstance(right[0], str) and right[0] == _RESET_SENTINEL:
        return []
    return (left or []) + right


class AgentState(TypedDict):
    """LangGraph Agent 全局状态结构（搜索工作流核心数据载体）。

    Attributes:
        user_query:     原始用户查询，全程不变。
        iteration:      当前迭代轮次，从 0 开始递增。
        plan:           Planner 生成的搜索关键词列表。
        missing_info:   Evaluator 判定的信息缺失项。
        retry_keywords: 基于缺失信息生成的补充搜索关键词。
        confidence_score: 结果置信度评分（0~1）。
        search_results: 累积搜索结果（普通 list，可直接赋值覆盖）。
        _search_accum:  并行 worker 临时累加器（自定义 reducer，支持信号清空）。
        final_report:   最终生成的结构化回答/报告。
    """

    # 原始输入（全程只读）
    user_query: str

    # 计划与迭代控制
    iteration: int
    plan: list[str]
    retry_keywords: list[str]

    # 搜索执行 — 双列表解耦架构
    search_results: list[dict[str, Any]]  # 可覆写（planner 首轮清零）
    _search_accum: Annotated[list[dict[str, Any]], reduce_accum]  # 并行合并 + 信号归零

    # 结果评估
    confidence_score: float
    missing_info: str

    # 最终输出
    final_report: str

    # 可观测性与监控
    task_id: str
    total_tokens: Annotated[int, operator.add]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    current_llm: str

    # 分批 wave 所需字段
    pending_keywords: list[str]
    _batch_keywords: list[str]

    thread_id: str
