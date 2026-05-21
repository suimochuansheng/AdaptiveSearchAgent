"""搜索计划生成节点（Planner 节点）模块。

LangGraph 工作流中的第一个处理节点，负责将用户的自然语言查询
转化为一组精确的搜索关键词，驱动后续的并行搜索节点。
"""

from langchain_core.runnables import RunnableConfig

from src.state import AgentState
from src.utils.json_parser import robust_json_parse
from src.utils.llm_utils import llm_call_with_fallback


async def planner(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """根据用户查询（以及缺失信息）生成搜索关键词列表。

    通过 llm_call_with_fallback 统一调用 LLM，自动获得备援与 Token 计数。
    返回：搜索计划 + Token 增量 + 当前 LLM 提供商。

    Args:
        state: LangGraph 传递的当前全局状态。
        config: LangGraph RunnableConfig，其中的 configurable.llm_provider
               指定本节点使用的 LLM。

    Returns:
        部分状态更新字典，含 plan / total_tokens / input_tokens /
        output_tokens / current_llm。
    """
    query = state["user_query"]
    missing = state.get("missing_info", "")
    iteration = state.get("iteration", 0)

    if missing and iteration > 0:
        prompt = f"""用户原始问题：{query}
    上一轮搜索后缺少的信息：{missing}
    请根据缺少的信息，生成 3~5 个更精准的搜索关键词。
    只输出 JSON，格式：{{"plan": ["关键词1", "关键词2", ...]}}"""
    else:
        prompt = f"""用户问题：{query}
    请为这个问题生成 3~5 个不同的搜索关键词，覆盖不同角度。
    只输出 JSON，格式：{{"plan": ["关键词1", "关键词2", ...]}}"""

    # 统一调用入口：自带备援 + Token 计数 + 并发限流
    content, in_tok, out_tok, total_tok = await llm_call_with_fallback(prompt, config)

    # 从 config 中读取当前实际使用的 provider（可能已被备援逻辑切换）
    current_provider = (
        config["configurable"].get("llm_provider", "ollama")
        if config and "configurable" in config
        else "ollama"
    )

    # 使用鲁棒 JSON 解析器提取关键词列表
    parsed = await robust_json_parse(content)
    # 从 AI 回复中提取 plan 字段，如果没有用原始查询作为 plan
    plan = parsed.get("plan", [query])
    # dict.fromkeys() 是 Python 里唯一能保序的去重方法
    plan = list(dict.fromkeys(plan))[:5]
    print(f"Planner 输出原始内容: {content}")
    return {
        "plan": plan,
        "total_tokens": total_tok,  # operator.add 自动累加
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "current_llm": current_provider,
    }
