"""Writer 节点：生成最终报告，附上 token 成本统计。"""

from src.state import AgentState
from src.utils.logger import log_node


@log_node("writer")
async def writer(state: AgentState) -> dict:
    """根据所有搜索结果生成 Markdown 格式报告。"""
    query = state["user_query"]
    results = state.get("search_results", [])
    total_tokens = state.get("total_tokens", 0)

    report = f"# 调研报告：{query}\n\n"

    for idx, res in enumerate(results, 1):
        keyword = res.get("keyword", f"结果{idx}")
        content = res.get("content", "无内容")
        report += f"## {idx}. {keyword}\n\n{content}\n\n---\n\n"

    report += f"\n**成本统计**：本次任务共消耗约 {total_tokens} tokens（本地模型不计费）。"
    return {"final_report": report}
