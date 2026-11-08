"""RAG 检索工具 — 通过 HTTP 调用 rag-ingest-gateway 子服务进行语义检索。"""

import logging
import time
import uuid

import httpx

from config import settings

logger = logging.getLogger(__name__)

# ── 熔断器（阶段 C 运维补丁）────────────────────────────────────────────
# RAG 子服务没起时，每个搜索 worker 都会等一次连接/超时失败再重试，
# 一次深调研能拖到 10 分钟。连接级失败（连不上/超时）后冷却一段时间，
# 期间直接返回空字符串——search_worker 会走 Tavily 兜底，不必反复撞墙。
_clock = time.monotonic
_unavailable_until: float = 0.0


def _now() -> float:
    return _clock()


def _rag_available() -> bool:
    """熔断期是否已过。"""
    return _now() >= _unavailable_until


def _mark_unavailable() -> None:
    global _unavailable_until
    _unavailable_until = _now() + settings.RAG_COOLDOWN_SECONDS
    logger.warning(
        "RAG 子服务不可用，%.0fs 内跳过检索（走 Tavily 兜底）",
        settings.RAG_COOLDOWN_SECONDS,
    )


def _reset_breaker() -> None:
    global _unavailable_until
    _unavailable_until = 0.0


async def search_knowledge(
    query: str,
    top_k: int = 3,
    kb_id: str = "default",
    model: str = "bge-m3",  # 仅 bge-m3（nomic 已下线）
    search_mode: str = "hybrid",  # ★ 检索模式：hybrid(BM25+向量RRF+网关侧Reranker) / vector(纯向量)
    skip_reranker: bool = False,  # ★ 跳过网关侧 Reranker（离线评估用）
) -> str:
    """调用子服务 /api/v1/search 进行向量检索。
    通过 HTTP 调用子服务的 /search 接口进行 RAG 检索。
    超时 10 秒，若失败则返回空字符串（降级处理）。
    Args:
        query:  检索查询文本。
        top_k:  返回的最相关结果数量。
        kb_id:  目标知识库 ID。
        model: Embedding 模型（仅 bge-m3）
        search_mode: 检索模式 (hybrid 或 vector)

    Returns:
        子服务返回的 result 字段；失败时降级返回空字符串。
    """
    request_id = str(uuid.uuid4())

    if not _rag_available():
        logger.debug("RAG 熔断中，跳过检索 (query=%r)", query)
        return ""

    url = f"{settings.INGEST_SERVICE_URL}/api/v1/search"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                json={
                    "query": query,
                    "top_k": top_k,
                    "kb_id": kb_id,
                    "model": model,
                    "search_mode": search_mode,
                    "skip_reranker": skip_reranker,
                },
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
            _reset_breaker()
            return result

    except httpx.TimeoutException:
        logger.warning("RAG 子服务超时 (request_id=%s)", request_id)
        _mark_unavailable()
    except httpx.HTTPStatusError as e:
        logger.warning(
            "RAG 子服务返回错误 (request_id=%s, status=%d): %s",
            request_id,
            e.response.status_code,
            e.response.text[:200],
        )
    except httpx.RequestError as e:
        logger.warning("RAG 子服务连接失败 (request_id=%s): %s", request_id, e)
        _mark_unavailable()

    return ""
