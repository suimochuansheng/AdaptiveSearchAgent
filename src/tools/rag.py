"""RAG 检索工具 — 通过 HTTP 调用 rag-ingest-gateway 子服务进行语义检索。"""

import logging
import uuid

import httpx

from config import settings

logger = logging.getLogger(__name__)


async def search_knowledge(
    query: str,
    top_k: int = 3,
    kb_id: str = "default",
    model: str = "nomic-embed",  # ★ 新增 model 参数
) -> str:
    """调用子服务 /api/v1/search 进行向量检索。
    通过 HTTP 调用子服务的 /search 接口进行 RAG 检索。
    超时 10 秒，若失败则返回空字符串（降级处理）。
    Args:
        query:  检索查询文本。
        top_k:  返回的最相关结果数量。
        kb_id:  目标知识库 ID。
        model: Embedding 模型 (nomic-embed 或 bge-m3)

    Returns:
        子服务返回的 result 字段；失败时降级返回空字符串。
    """
    request_id = str(uuid.uuid4())
    url = f"{settings.INGEST_SERVICE_URL}/api/v1/search"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                json={"query": query, "top_k": top_k, "kb_id": kb_id, "model": model},
                headers={"X-Request-ID": request_id},
            )
            resp.raise_for_status()
            result = resp.json().get("result", "")
            # 过滤无结果标记，触发 search_worker 的 Tavily fallback
            if not result or "未找到相关内容" in result:
                logger.info(
                    "RAG 无结果 (request_id=%s, query=%r)",
                    request_id,
                    query,
                )
                return ""
            logger.info(
                "RAG 子服务调用成功 (request_id=%s, query=%r, hits=%d bytes)",
                request_id,
                query,
                len(result),
            )
            return result

    except httpx.TimeoutException:
        logger.warning("RAG 子服务超时 (request_id=%s)", request_id)
    except httpx.HTTPStatusError as e:
        logger.warning(
            "RAG 子服务返回错误 (request_id=%s, status=%d): %s",
            request_id,
            e.response.status_code,
            e.response.text[:200],
        )
    except httpx.RequestError as e:
        logger.warning("RAG 子服务连接失败 (request_id=%s): %s", request_id, e)

    return ""
