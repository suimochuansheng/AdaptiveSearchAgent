"""AdaptiveSearchAgent 的 LangGraph 状态定义模块。

定义 Agent 在整个搜索工作流中传递和累积的状态数据结构。
LangGraph 中各节点通过读取和写入此状态来协作完成搜索任务。
"""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """LangGraph Agent 的全局状态结构。

    每个 LangGraph 节点函数接收完整的 AgentState，
    返回部分字段的更新字典，LangGraph 自动合并到全局状态中。

    Attributes:
        user_query: 用户输入的原始查询字符串。
        iteration: 当前迭代次数，用于控制 Planner → Evaluator 的重试循环。
        plan: Planner 节点生成的搜索关键词列表。
        missing_info: Evaluator 节点发现的信息缺失描述，用于下一轮重试补充。
        search_results: 并行搜索结果的累积列表。
            使用 Annotated[list[str], operator.add] 实现：
            LangGraph 在合并多个并行搜索节点的返回值时，
            自动调用 operator.add 将各节点的搜索结果列表拼接起来（扇入同步）。
    """

    user_query: str
    iteration: int
    plan: list[str]
    missing_info: str
    search_results: Annotated[list[str], operator.add]
