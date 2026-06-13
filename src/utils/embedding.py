"""嵌入模型工具 — 基于 Ollama nomic-embed-text。

零额外依赖（复用已有 Ollama 基础设施），httpx 异步调用。
"""

import httpx

from config import settings

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768  # nomic-embed-text 输出维度


async def embed_text(text: str) -> list[float]:
    """将文本转为向量嵌入（768 维）。"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding: list[float] = data["embedding"]
        return embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """批量嵌入（逐个调用，Ollama 无批量 API）。"""
    results = []
    for text in texts:
        results.append(await embed_text(text))
    return results
