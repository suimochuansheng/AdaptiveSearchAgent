#!/usr/bin/env python3
# ruff: noqa: E402  — sys.path 必须在项目导入之前修改
"""知识库一次性导入脚本。

用法:
    eval $(poetry env activate)
    python scripts/ingest_knowledge.py

从 data/company_notes/ 目录读取 .txt 文件，按【标签】格式切分；
从 data/math/ 目录读取 .md 文件，按 ## 标题切分。
通过 Ollama nomic-embed-text 向量化，写入 PostgreSQL knowledge_embeddings 表。
"""

import asyncio
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from config import settings
from src.utils.embedding import embed_text

SOURCE_DIRS = [
    ROOT / "data" / "company_notes",  # .txt — 按【标签】切分
    ROOT / "data" / "math",  # .md  — 按 ## 标题切分
]
BATCH_SIZE = 5


def parse_markdown(file_path: Path) -> list[dict]:
    """解析 .md 文件：按 ## 标题切分。"""
    text = file_path.read_text(encoding="utf-8")
    source = file_path.name
    blocks = re.split(r"\n(?=## )", text)
    chunks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        match = re.match(r"^## (.+)", block)
        title = match.group(1).strip() if match else source.replace(".md", "")
        content = re.sub(r"^## .+\n", "", block).strip()
        if content:
            chunks.append({"title": title, "content": content, "source_file": source})
    return chunks


def parse_tagged_txt(file_path: Path) -> list[dict]:
    """解析 .txt 文件：按【标签】内容 格式切分，以公司名作为 title。"""
    text = file_path.read_text(encoding="utf-8")
    source = file_path.name
    # 按空行分隔公司条目
    entries = re.split(r"\n\n+", text)
    chunks = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        # 提取第一个人名/公司名标签作为 title
        match = re.search(r"【公司名称】(.+)", entry)
        title = match.group(1).strip() if match else "未知条目"
        chunks.append({"title": title, "content": entry, "source_file": source})
    return chunks


async def main():
    conninfo = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )
    conn = await AsyncConnection.connect(conninfo, row_factory=dict_row, autocommit=True)

    all_chunks = []
    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            print(f"[跳过] 目录不存在: {src_dir}")
            continue
        for f in sorted(src_dir.glob("*.md")):
            print(f"[解析 MD] {f.name}")
            all_chunks.extend(parse_markdown(f))
        for f in sorted(src_dir.glob("*.txt")):
            print(f"[解析 TXT] {f.name}")
            all_chunks.extend(parse_tagged_txt(f))

    if not all_chunks:
        print("[错误] 未找到任何知识点。请放入 .md 或 .txt 文件。")
        sys.exit(1)

    print(f"[总计] {len(all_chunks)} 个知识点")

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [f"{c['title']}\n{c['content']}" for c in batch]

        embeddings = []
        for j, text in enumerate(texts):
            emb = await embed_text(text)
            embeddings.append(emb)
            print(f"  [嵌入] {i + j + 1}/{len(all_chunks)}: {batch[j]['title'][:40]}")

        for chunk, emb in zip(batch, embeddings, strict=True):
            await conn.execute(
                """INSERT INTO knowledge_embeddings (title, content, embedding, source_file)
                   VALUES (%s, %s, %s::vector, %s)""",
                (chunk["title"], chunk["content"], emb, chunk["source_file"]),
            )
        print(f"  [写入] 第 {i // BATCH_SIZE + 1} 批完成")

    await conn.close()
    print(f"\n[完成] 共导入 {len(all_chunks)} 个知识点到 knowledge_embeddings 表。")


if __name__ == "__main__":
    asyncio.run(main())
