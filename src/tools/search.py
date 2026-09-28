"""Tavily 搜索工具封装，带指数退避重试。"""

from tavily import TavilyClient
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings

# 初始化 Tavily 客户端
client = TavilyClient(api_key=settings.tavily_api_key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
)
async def search_tavily(keyword: str) -> str:
    """执行 Tavily 搜索，返回结果摘要文本。

    Args:
        keyword: 搜索关键词

    Returns:
        搜索结果的纯文本摘要，最多 3000 字符。
    """
    # Tavily 客户端是同步的，但在协程中用 run_in_executor 避免阻塞
    import asyncio

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.search(
            query=keyword,
            search_depth="basic",
            max_results=3,
            include_answer=False,
            include_raw_content=False,
        ),
    )
    results = response.get("results", [])
    texts = [r["content"] for r in results if "content" in r]
    combined = "\n\n".join(texts)
    return combined[:3000] if combined else f"未找到关于 '{keyword}' 的相关信息。"
