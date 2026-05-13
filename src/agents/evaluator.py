"""Evaluator 节点：评估搜索结果充分性，给出置信度和缺失信息。"""

from langchain_ollama import ChatOllama

from config import settings
from src.state import AgentState
from src.utils.json_parser import robust_json_parse
from src.utils.logger import log_node

llm = ChatOllama(model=settings.ollama_model_name, temperature=0)


@log_node("evaluator")
async def evaluator(state: AgentState) -> dict:
    """
    评估当前搜索结果是否足够回答用户问题
    返回：置信度、缺失信息、建议重试关键词、迭代次数+1
    """
    # 从状态中获取核心输入：用户问题、搜索结果、当前迭代次数
    query = state["user_query"]
    results = state.get("search_results", [])
    iteration = state.get("iteration", 0)

    # 构造搜索结果摘要：最多取前5条，每条只保留500字符，避免过长
    # 格式：关键词 + 内容片段，用双换行分隔
    results_summary = "\n\n".join(
        f"关键词：{r['keyword']}\n内容：{r['content'][:500]}" for r in results[:5]
    )

    # 构造给大模型的提示词：明确要求输出JSON格式，定义字段含义
    prompt = f"""用户问题：{query}
搜索结果：
{results_summary}

请判断这些搜索结果是否足以回答用户问题。输出 JSON 格式如下：
{{
    "confidence_score": 0.85,    // 0~1 之间的浮点数，表示回答置信度
    "missing_info": "还缺少功耗数据",   // 信息不足时，描述缺少的内容
    "retry_keywords": ["RTX 5090 功耗", "RTX 5090 TDP"]  // 建议补充搜索的关键词
}}
如果信息已足够，missing_info 为空字符串，retry_keywords 为空列表。
只输出 JSON，不要额外文字。"""

    # 异步调用大模型获取评估结果
    response = await llm.ainvoke(prompt)

    # 统一转为字符串，确保后续解析正常（兼容大模型不同返回格式）
    content = response.content if isinstance(response.content, str) else str(response.content)

    # 稳健解析JSON（自动处理大模型常见的格式错误、多余字符）
    parsed = await robust_json_parse(content)

    # 安全提取JSON字段，设置默认值防止键不存在报错
    confidence = float(parsed.get("confidence_score", 0.0))
    missing = parsed.get("missing_info", "")
    retry = parsed.get("retry_keywords", [])

    # 返回更新后的状态：置信度、缺失信息、重试关键词、迭代次数+1
    return {
        "confidence_score": confidence,
        "missing_info": missing,
        "retry_keywords": retry,
        "iteration": iteration + 1,
    }
