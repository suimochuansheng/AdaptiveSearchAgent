"""Writer 节点：LLM 生成专业调研报告（去噪 + 提炼 + 格式化）。

包含上下文长度保护：搜索结果超过 MAX_CONTEXT_TOKENS 时自动截断。
"""

import logging

import tiktoken
from langchain_core.runnables import RunnableConfig

from config import settings
from src.state import AgentState
from src.utils.llm_utils import llm_call_with_fallback
from src.utils.logger import log_node

logger = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")

# ── Writer 系统提示词 ──────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的技术文档分析师与报告撰写专家。
你的任务是根据给定的参考资料（包括本地知识库和在线搜索结果），
针对用户的提问撰写一份专业、严谨、逻辑清晰的调研报告。

在撰写时，你必须严格遵守以下规则：

1. 真实性与去噪：
   - 只能基于参考资料中"确实与用户问题高度相关"的内容进行整合。
   - 如果参考资料中包含与主题无关的噪声（例如无关网页、浏览器操作指南、
     原始 CSS 代码段、无意义的占位文本等），你必须在撰写时将其【彻底忽略】，
     绝对不能写入报告。

2. 报告格式规范：
   - 使用 Markdown 格式。
   - 严禁在报告中输出任何 RAG 系统内部的调试信息，包括但不限于：
     相似度分数（如 0.8000）、文档文件名（如 test_rag_spec.md）、
     检索源标记（如 [本地知识库] 或 [在线搜索]）。

3. 语言风格：
   - 使用专业、流畅、连贯的技术语言，将碎片化的参考资料有机融合为
     一篇通顺的文章，严禁进行简单的"段落拼接"或"生搬硬套"。
   - 报告应包含：引言、核心概念/定义、关键要点分节、总结。
"""


def _clean_content(text: str) -> str:
    """清洗检索结果中的系统标记和元数据噪声。"""
    # 移除系统内部标记
    for tag in ("[本地知识库]", "[在线搜索]"):
        text = text.replace(tag, "")
    # 移除相似度行（如 "【文件名】(相似度: 0.95)"）
    import re

    text = re.sub(r"【.+?】\(相似度: [\d.]{4,6}\)", "", text)
    return text.strip()


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _prepare_context(results: list[dict], max_tokens: int) -> str:
    """将搜索结果清洗后拼接为 LLM 上下文，控制 token 总量。"""
    parts: list[str] = []
    used = 0
    for r in results:
        keyword = r.get("keyword", "")
        content = _clean_content(r.get("content", ""))
        if not content.strip():
            continue
        block = f"[来源关键词：{keyword}]\n{content}\n"
        tok = _count_tokens(block)
        if used + tok > max_tokens:
            remaining = max_tokens - used
            if remaining > 100:
                truncated = _ENCODER.decode(_ENCODER.encode(block)[:remaining])
                parts.append(truncated + "…")
            break
        parts.append(block)
        used += tok
    return "\n".join(parts)


@log_node("writer")
async def writer(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """LLM 驱动的报告生成：读取搜索结果 → 清洗噪声 → LLM 提炼 → Markdown 报告。"""
    query = state["user_query"]
    results = state.get("search_results", [])

    # 清洗并裁剪上下文
    max_tokens = settings.MAX_CONTEXT_TOKENS
    context = _prepare_context(results, max_tokens) if results else "（无参考资料）"

    logger.info(
        "Writer 上下文: results=%d, context_tokens≈%d, max=%d",
        len(results),
        _count_tokens(context),
        max_tokens,
    )

    # 构建 prompt
    user_prompt = (
        f"用户提问：{query}\n\n" f"参考资料：\n{context}\n\n" f"请根据以上参考资料，撰写调研报告。"
    )

    prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    # LLM 生成报告
    report, in_tok, out_tok, total_tok = await llm_call_with_fallback(prompt, config)

    total_tokens = state.get("total_tokens", 0)
    input_tokens = state.get("input_tokens", 0)
    output_tokens = state.get("output_tokens", 0)
    current_llm = state.get("current_llm", "unknown")

    # 追加成本统计
    report += (
        f"\n\n---\n\n"
        f"**成本统计**\n\n"
        f"- 模型：{current_llm}\n"
        f"- 输入 Token：{input_tokens}\n"
        f"- 输出 Token：{output_tokens}\n"
        f"- 总计：{total_tokens} tokens\n"
    )

    logger.info("Writer 完成报告: 长度=%d 字符", len(report))
    return {
        "final_report": report,
        "total_tokens": total_tok,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }
