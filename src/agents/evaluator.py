"""Evaluator 节点：评估搜索结果充分性，给出置信度和缺失信息。"""

from langchain_core.runnables import RunnableConfig

from src.state import AgentState
from src.utils.json_parser import robust_json_parse
from src.utils.llm_utils import llm_call_with_fallback
from src.utils.logger import log_node


@log_node("evaluator")
async def evaluator(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """评估当前搜索结果是否足够回答用户问题。

    通过 llm_call_with_fallback 统一调用 LLM，自动获得备援与 Token 计数。
    返回：置信度、缺失信息、建议重试关键词、递增后的迭代轮次、
          Token 计数增量、当前 LLM 提供商。

    Args:
        state: 当前全局状态，含 user_query、search_results。
        config: LangGraph RunnableConfig，其中的 configurable.llm_provider
               指定本节点使用的 LLM。

    Returns:
        部分状态更新字典。
    """
    query = state["user_query"]
    # 合并本轮的并行 worker 结果到 search_results
    new_results = state.get("_search_accum", [])
    results = state.get("search_results", []) + new_results

    results_summary = "\n\n".join(
        f"关键词：{r['keyword']}\n内容：{r['content'][:500]}" for r in results[:5]
    )

    prompt = f"""用户问题：{query}
搜索结果：
{results_summary}

请判断这些搜索结果是否足以回答用户问题。

判定标准（重要）：
- 如果搜索结果中**已包含用户问题的核心概念定义**，且至少有一条相关说明，
  则置信度 ≥ 0.80，判定为"已满足"。
- 如果搜索结果中**部分覆盖**问题但缺少关键细节（如数据、时间、对比），
  置信度 0.50~0.79，需补充搜索。
- 如果搜索结果**与问题完全无关**或为空，置信度 < 0.50。

输出 JSON 格式如下：
{{
    "confidence_score": 0.85,
    "missing_info": "还缺少功耗数据",
    "retry_keywords": ["RTX 5090 功耗"]
}}
如果信息已足够，missing_info 为空字符串，retry_keywords 为空列表。
只输出 JSON，不要额外文字。"""

    # 统一调用入口：自带备援 + Token 计数 + 并发限流
    content, in_tok, out_tok, total_tok = await llm_call_with_fallback(prompt, config)

    # 从 config 中读取当前实际使用的 provider
    current_provider = (
        config["configurable"].get("llm_provider", "ollama")
        if config and "configurable" in config
        else "ollama"
    )
    # 从结果中鲁棒性提取JSON字段
    parsed = await robust_json_parse(content)
    confidence = float(parsed.get("confidence_score", 0.0))
    missing = parsed.get("missing_info", "")
    retry = parsed.get("retry_keywords", [])

    return {
        "confidence_score": confidence,
        "missing_info": missing,
        "retry_keywords": retry,
        "search_results": results,  # 持久化合并结果
        "_search_accum": ["__RESET__"],  # 通过自定义 reducer 信号清零
        "iteration": state.get("iteration", 0) + 1,  # 本轮评估完成，递增轮次
        "total_tokens": total_tok,  # operator.add 自动累加（API 直接返回 total）
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "current_llm": current_provider,
    }
