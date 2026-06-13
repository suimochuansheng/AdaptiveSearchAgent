"""Writer 节点：生成最终报告，附上 token 成本统计。"""

import logging

from src.state import AgentState
from src.utils.logger import log_node

logger = logging.getLogger(__name__)


@log_node("writer")
async def writer(state: AgentState) -> dict:
    """根据所有搜索结果生成 Markdown 格式报告。"""
    query = state["user_query"]
    results = state.get("search_results", [])
    total_tokens = state.get("total_tokens", 0)
    input_tokens = state.get("input_tokens", 0)
    output_tokens = state.get("output_tokens", 0)
    current_llm = state.get("current_llm", "unknown")

    logger.info(
        "Writer 开始生成报告: query=%s, search_results_count=%d, tokens=%d",
        query[:50],
        len(results),
        total_tokens,
    )

    report = f"# 调研报告：{query}\n\n"

    for idx, res in enumerate(results, 1):
        keyword = res.get("keyword", f"结果{idx}")
        content = res.get("content", "无内容")
        # 简单格式化：标题 + 内容 + 分割线
        report += f"## {idx}. {keyword}\n\n{content}\n\n---\n\n"

    report += (
        f"\n**成本统计**\n"
        f"- 模型：{current_llm}\n"
        f"- 输入 Token：{input_tokens}\n"
        f"- 输出 Token：{output_tokens}\n"
        f"- 总计：{total_tokens} tokens\n"
    )

    logger.info("Writer 完成报告: 长度=%d 字符", len(report))
    return {"final_report": report}
