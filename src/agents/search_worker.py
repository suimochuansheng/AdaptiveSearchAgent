"""搜索工作节点：并行检索本地 RAG 知识库和 Tavily 在线搜索，合并结果返回。

由 parallel_searcher 通过 Send API 并行调用。
"""

import asyncio

from src.tools.rag import search_knowledge
from src.tools.search import search_tavily
from src.utils.logger import log_node


@log_node("search_worker")
async def search_worker(state: dict) -> dict:
    """单个关键词搜索节点 — RAG + Tavily 并行检索。

    LangGraph Send API 将 AgentState + {"keyword": str} 合并传入。
    """
    keyword: str = state["keyword"]
    kb_id: str = state.get("kb_id", "default")

    # 并行检索：本地知识库 + 在线搜索同时发起
    rag_task = search_knowledge(keyword, top_k=3, kb_id=kb_id)
    web_task = search_tavily(keyword)

    rag_result, web_result = await asyncio.gather(
        rag_task,
        web_task,
        return_exceptions=True,
    )

    # 合并结果（截断过长的单条文本，防止 Token 暴涨）
    parts: list[str] = []
    if rag_result and not isinstance(rag_result, Exception):
        parts.append(f"[本地知识库]\n{rag_result[:800]}")
    if web_result and not isinstance(web_result, Exception):
        parts.append(f"[在线搜索]\n{web_result[:800]}")

    content = "\n\n".join(parts) if parts else f"未找到关于 '{keyword}' 的相关信息"

    return {"_search_accum": [{"keyword": keyword, "content": content}]}
