"""RAG 检索工具 — 从 knowledge_embeddings 表检索知识片段。"""

from psycopg.rows import dict_row

from src import checkpointer
from src.utils.embedding import embed_text


async def search_knowledge(query: str, top_k: int = 3) -> str:
    """从 knowledge_embeddings 表中检索最相关的知识片段。

    Args:
        query: 检索查询文本。
        top_k: 返回的最相关结果数量。

    Returns:
        拼接后的知识文本，以 "---" 分隔。无结果时返回空字符串。
    """
    pool = checkpointer._global_pool
    assert pool is not None, "数据库连接池未初始化"

    query_vec = await embed_text(query)

    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """SELECT title, content,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM knowledge_embeddings
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (query_vec, query_vec, top_k),
        )
        rows = await cur.fetchall()

    if not rows:
        return ""

    parts = []
    for r in rows:
        parts.append(f"【{r['title']}】(相关度: {r['similarity']:.2f})\n{r['content']}")
    return "\n\n---\n\n".join(parts)
