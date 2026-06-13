"""搜索工作节点：执行关键词搜索，优先查本地 RAG 知识库，无结果时走 Tavily。

由 parallel_searcher 通过 Send API 并行调用。
"""

from src.tools.rag import search_knowledge
from src.tools.search import search_tavily
from src.utils.logger import log_node


@log_node("search_worker")
async def search_worker(state: dict) -> dict:
    """单个关键词搜索节点。

    LangGraph Send API 将 AgentState + {"keyword": str} 合并传入。
    优先检索本地 RAG 知识库（knowledge_embeddings），
    无匹配时回退到 Tavily 在线搜索。
    """
    keyword: str = state["keyword"]

    # 1. 先查本地 RAG 知识库
    rag_result = await search_knowledge(keyword)

    if rag_result:
        content = f"[本地知识库]\n{rag_result}"
    else:
        # 2. 知识库无匹配 → 回退到 Tavily 在线搜索
        content = await search_tavily(keyword)

    return {"search_results": [{"keyword": keyword, "content": content}]}
