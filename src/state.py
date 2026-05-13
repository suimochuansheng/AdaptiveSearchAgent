"""AdaptiveSearchAgent 的 LangGraph 状态定义模块。

定义 Agent 在整个搜索工作流中传递和累积的状态数据结构。
LangGraph 中各节点通过读取和写入此状态来协作完成搜索任务，
节点仅返回需要更新的字段字典，框架自动完成状态合并。
"""

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


# 可以理解为 AgentState 就是整个搜索工作流的“共享内存”，所有节点都从这里拿数据、写数据
# LangGraph 自动帮你合并、累加、传递！
class AgentState(TypedDict):
    """LangGraph Agent 全局状态结构（搜索工作流核心数据载体）。

    所有节点（Planner/搜索执行器/Evaluator/报告生成器）共享此状态，
    通过状态读写实现流程协作、数据传递与循环重试控制。
    状态更新遵循“增量合并”规则：仅更新指定字段，未指定字段保持不变。

    Attributes:
        user_query: 原始用户查询，全程不变，作为所有决策的依据
        iteration: 当前迭代轮次，从 0 开始递增，用于控制 Planner→Evaluator 循环重试次数
        plan: Planner 节点生成的**结构化搜索计划**，包含本轮需要执行的搜索关键词/查询列表
        missing_info: Evaluator 判定的**信息缺失项**，描述当前结果不足的内容，用于指导下一轮补充搜索
        retry_keywords: 基于缺失信息生成的**补充搜索关键词**，专门用于修复信息缺口
        confidence_score: 结果置信度评分（0~1），Evaluator 根据信息完整度计算，达到阈值则终止迭代
        search_results: 多轮/多节点**累积搜索结果**，使用 Annotated + operator.add 实现自动列表拼接
                        并行节点返回的结果会自动合并，不会覆盖原有数据（扇入聚合）
        final_report: 最终生成的结构化回答/报告，满足用户查询的完整输出内容
        human_approved: 人工干预审批标记（HIL 预留），True=人工确认通过，False=需要修改/拒绝
        task_id: 任务唯一标识，用于日志追踪、链路监控与结果存储
        total_tokens: 全流程累计 Token 消耗量，用于成本统计与限流监控
    """

    # 原始输入（全程只读）
    user_query: str

    # 计划与迭代控制
    iteration: int  # 循环第几轮了
    plan: list[str]  # Planner 生成的搜索关键词列表
    retry_keywords: list[str]  # 评估器建议“重新搜索”的词

    # 搜索执行（自动合并）
    search_results: Annotated[list[dict[str, Any]], operator.add]

    # 结果评估
    confidence_score: float
    missing_info: str

    # 最终输出
    final_report: str
    human_approved: bool

    # 可观测性与监控
    task_id: str
    total_tokens: int
