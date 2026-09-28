#!/usr/bin/env python3
"""数学知识库一次性导入脚本。

用法：
    cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent
    eval $(poetry env activate)
    python scripts/ingest_math_knowledge.py

读取 data/math/ 目录下所有 .md 文件，按 ## 标题切分为知识点，
通过 Ollama nomic-embed-text 向量化，写入 PostgreSQL math_embeddings 表。
"""

import asyncio
import re
import sys
from pathlib import Path

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from config import settings
from src.utils.embedding import embed_text

MATH_DIR = Path(__file__).resolve().parent.parent / "data" / "math"
BATCH_SIZE = 5  # 每批嵌入的数量，避免 Ollama 过载


def parse_markdown(file_path: Path) -> list[dict]:
    """解析 Markdown 文件，按 ## 标题切分为知识点。

    每个知识点 = {"title": "二次函数顶点公式", "content": "...", "source_file": "xxx.md"}
    """
    text = file_path.read_text(encoding="utf-8")
    source = file_path.name
    # 按 ## 标题切分（保留标题行作为 title）
    blocks = re.split(r"\n(?=## )", text)
    chunks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # 提取第一行 ## 标题
        match = re.match(r"^## (.+)", block)
        title = match.group(1).strip() if match else source.replace(".md", "")
        # 内容去掉标题行
        content = re.sub(r"^## .+\n", "", block).strip()
        if content:
            chunks.append({"title": title, "content": content, "source_file": source})
    return chunks


async def main():
    # ── 1. 连接数据库 ─────────────────────────────────────
    conninfo = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )
    conn = await AsyncConnection.connect(conninfo, row_factory=dict_row, autocommit=True)

    # ── 2. 解析所有 Markdown 文件 ──────────────────────────
    if not MATH_DIR.exists():
        print(f"[错误] 知识库目录不存在: {MATH_DIR}")
        print("请创建 data/math/ 并放入 .md 格式的数学知识文档。")
        sys.exit(1)

    all_chunks = []
    for f in sorted(MATH_DIR.glob("*.md")):
        print(f"[解析] {f.name}")
        all_chunks.extend(parse_markdown(f))

    if not all_chunks:
        print("[错误] 未找到任何知识点，请检查 data/math/ 下的 .md 文件格式。")
        sys.exit(1)

    print(f"[总计] {len(all_chunks)} 个知识点")

    # ── 3. 批量向量化并写入 ───────────────────────────────
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [f"{c['title']}\n{c['content']}" for c in batch]

        # 逐个嵌入（Ollama 无 batch API）
        embeddings = []
        for j, text in enumerate(texts):
            emb = await embed_text(text)
            embeddings.append(emb)
            print(f"  [嵌入] {i + j + 1}/{len(all_chunks)}: {batch[j]['title'][:40]}")

        # 批量写入 PostgreSQL
        for chunk, emb in zip(batch, embeddings, strict=True):
            await conn.execute(
                """INSERT INTO math_embeddings (title, content, embedding, source_file)
                   VALUES (%s, %s, %s::vector, %s)""",
                (chunk["title"], chunk["content"], emb, chunk["source_file"]),
            )
        print(f"  [写入] 第 {i // BATCH_SIZE + 1} 批完成")

    await conn.close()
    print(f"\n[完成] 共导入 {len(all_chunks)} 个知识点到 math_embeddings 表。")


if __name__ == "__main__":
    asyncio.run(main())
