"""搜索计划生成节点（Planner 节点）模块。

LangGraph 工作流中的第一个处理节点，负责将用户的自然语言查询
转化为一组精确的搜索关键词，驱动后续的并行搜索节点。
"""

from langchain_core.runnables import RunnableConfig

from src.state import AgentState
from src.utils.json_parser import robust_json_parse
from src.utils.llm_factory import get_llm


async def planner(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """LangGraph 节点函数：根据用户查询（以及缺失信息）生成搜索关键词列表。

    作为 LangGraph 有向图中的 Planner 节点，此函数接收当前全局状态，
    通过 LLM 分析用户查询并生成 3-5 个高价值搜索关键词。

    工作流程：
    1. 从 state 中提取用户查询和上一轮缺失信息（如有）
    2. 通过请求级 config 动态选择 LLM provider
    3. 构造提示词，告知 LLM 搜索意图并指定 JSON 输出格式
    4. 异步调用 LLM 生成关键词列表
    5. 使用鲁棒 JSON 解析器提取结构化结果
    6. 返回部分状态更新，LangGraph 自动合并到全局状态

    Args:
        state: LangGraph 传递的当前全局状态（AgentState 字典）。
               包含 user_query、iteration、missing_info 等字段。
        config: LangGraph RunnableConfig，其中的 configurable.llm_provider
               指定本节点使用的 LLM（deepseek / ollama）。
               由 graph.ainvoke 调用方在请求级传入，节点不依赖全局状态。

    Returns:
        dict: 部分状态更新字典，仅包含 {"plan": [...]}。
              LangGraph 自动将此更新合并到全局 AgentState 中。
    """
    query = state["user_query"]
    missing = state.get("missing_info", "")
    iteration = state.get("iteration", 0)

    # 根据请求级 config 动态创建 LLM 实例
    llm = get_llm(temperature=0, config=config)

    if missing and iteration > 0:
        prompt = f"""用户原始问题：{query}
    上一轮搜索后缺少的信息：{missing}
    请根据缺少的信息，生成 3~5 个更精准的搜索关键词。
    只输出 JSON，格式：{{"plan": ["关键词1", "关键词2", ...]}}"""
    else:
        prompt = f"""用户问题：{query}
    请为这个问题生成 3~5 个不同的搜索关键词，覆盖不同角度。
    只输出 JSON，格式：{{"plan": ["关键词1", "关键词2", ...]}}"""

    response = await llm.ainvoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    parsed = await robust_json_parse(content)
    plan = parsed.get("plan", [query])
    plan = list(dict.fromkeys(plan))[:5]
    return {"plan": plan}
