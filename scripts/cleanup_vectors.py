#!/usr/bin/env python3
"""清理 PostgreSQL 向量库脚本。

用法:
    # 查看当前向量库状态（只读，不删除）
    python scripts/cleanup_vectors.py --dry-run

    # 清理所有向量数据（需要确认）
    python scripts/cleanup_vectors.py

    # 按知识库ID清理
    python scripts/cleanup_vectors.py --kb-id test

    # 按来源文件清理
    python scripts/cleanup_vectors.py --source-file data/sample.md

    # 跳过确认，直接执行
    python scripts/cleanup_vectors.py --yes

    # 清理后重建索引
    python scripts/cleanup_vectors.py --reindex

涉及表:
    - knowledge_embeddings  (Agent 对话助手 + rag_data_dispose 共用)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

# 将项目根目录加入 sys.path，以便导入 config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings


def _build_conninfo() -> str:
    """根据 config.settings 构建 PostgreSQL 连接串。"""
    return (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


async def show_current_state(conn: psycopg.AsyncConnection) -> None:
    """展示 knowledge_embeddings 表当前状态。"""
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT COUNT(*) AS total FROM knowledge_embeddings")
        row = await cur.fetchone()
        total = row["total"] if row else 0
        print(f"  总行数: {total}")

        if total > 0:
            await cur.execute("""
                SELECT kb_id, source_file, COUNT(*) AS cnt
                FROM knowledge_embeddings
                GROUP BY kb_id, source_file
                ORDER BY kb_id, cnt DESC
            """)
            rows = await cur.fetchall()
            print("  分布:")
            for r in rows:
                print(f"    kb_id={r['kb_id']}  source={r['source_file']}  rows={r['cnt']}")


async def cleanup_vectors(
    conninfo: str,
    *,
    kb_id: str | None = None,
    source_file: str | None = None,
    reindex: bool = False,
    dry_run: bool = False,
) -> int:
    """清理向量数据，返回被删除的行数。

    Args:
        conninfo: PostgreSQL 连接串
        kb_id: 按知识库ID过滤清理（None = 全部）
        source_file: 按来源文件过滤清理（None = 全部）
        reindex: 是否在清理后重建 ivfflat 索引
        dry_run: 仅展示将要删除的数据，不实际执行

    Returns:
        被删除的行数
    """
    conn = await psycopg.AsyncConnection.connect(conninfo, autocommit=True)

    try:
        # ── 1. 构建 WHERE 条件 ──
        conditions: list[str] = []
        params: list[str] = []
        if kb_id:
            conditions.append("kb_id = %s")
            params.append(kb_id)
        if source_file:
            conditions.append("source_file = %s")
            params.append(source_file)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # ── 2. 查询待删除行数 ──
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"SELECT COUNT(*) AS cnt FROM knowledge_embeddings {where_clause}",
                params,
            )
            row = await cur.fetchone()
            delete_count = row["cnt"] if row else 0

        if delete_count == 0:
            print("✅ 没有需要清理的数据。")
            return 0

        # ── 3. 展示待删除数据分布 ──
        print(f"\n📊 待删除: {delete_count} 行")
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"""
                SELECT kb_id, source_file, COUNT(*) AS cnt
                FROM knowledge_embeddings
                {where_clause}
                GROUP BY kb_id, source_file
                ORDER BY kb_id, cnt DESC
                """,
                params,
            )
            for r in await cur.fetchall():
                print(f"   → kb_id={r['kb_id']}  source={r['source_file']}  rows={r['cnt']}")

        if dry_run:
            print("\n🔍 [dry-run 模式] 未实际删除数据。")
            print("   去掉 --dry-run 参数即可执行真实删除。")
            return delete_count

        # ── 4. 执行删除 ──
        async with conn.cursor() as cur:
            await cur.execute(
                f"DELETE FROM knowledge_embeddings {where_clause}",
                params,
            )
        print(f"\n🗑️  已删除 {delete_count} 行。")

        # ── 5. 重置序列（全表清理时） ──
        if kb_id is None and source_file is None:
            async with conn.cursor() as cur:
                await cur.execute("ALTER SEQUENCE knowledge_embeddings_id_seq RESTART WITH 1")
            print("🔄 已重置 id 序列 (RESTART WITH 1)。")

        # ── 6. 重建向量索引（可选） ──
        if reindex:
            async with conn.cursor() as cur:
                print("🔨 正在重建向量索引（ivfflat）...")
                await cur.execute("DROP INDEX IF EXISTS idx_knowledge_embedding")
                await cur.execute("""
                    CREATE INDEX idx_knowledge_embedding
                    ON knowledge_embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 10)
                """)
            print("✅ 向量索引重建完成。")

        # ── 7. 验证 ──
        print("\n📊 清理后状态:")
        await show_current_state(conn)

        return delete_count

    finally:
        await conn.close()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="清理 PostgreSQL pgvector 向量库 (knowledge_embeddings 表)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --dry-run              # 查看当前状态
  %(prog)s --yes                   # 清理全部向量数据
  %(prog)s --kb-id test --yes      # 按知识库ID清理
  %(prog)s --reindex --yes         # 清理全部 + 重建索引
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读模式：展示将要删除的数据，不实际执行删除",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="跳过确认提示，直接执行",
    )
    parser.add_argument(
        "--kb-id",
        type=str,
        default=None,
        help="按知识库ID过滤清理（不指定则清理全部）",
    )
    parser.add_argument(
        "--source-file",
        type=str,
        default=None,
        help="按来源文件名过滤清理（不指定则清理全部）",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="清理后重建 ivfflat 向量索引",
    )
    args = parser.parse_args()

    conninfo = _build_conninfo()

    # ── 展示当前状态 ──
    print("=" * 60)
    print("  PostgreSQL 向量库清理工具")
    print("  目标表: knowledge_embeddings")
    print("=" * 60)
    print("\n📊 当前状态:")
    conn = await psycopg.AsyncConnection.connect(conninfo, autocommit=True)
    try:
        await show_current_state(conn)
    finally:
        await conn.close()

    # ── 确认 ──
    if args.dry_run:
        print("\n--- 以下为 dry-run 预览 ---")
    else:
        filter_desc = []
        if args.kb_id:
            filter_desc.append(f"kb_id={args.kb_id!r}")
        if args.source_file:
            filter_desc.append(f"source_file={args.source_file!r}")
        scope = " AND ".join(filter_desc) if filter_desc else "全部数据"

        if not args.yes:
            print("\n⚠️  即将清理 knowledge_embeddings 表中符合条件的数据")
            print(f"   条件: {scope}")
            resp = input("\n确认执行? [y/N]: ").strip().lower()
            if resp not in ("y", "yes"):
                print("❌ 已取消。")
                sys.exit(0)

    # ── 执行清理 ──
    deleted = await cleanup_vectors(
        conninfo,
        kb_id=args.kb_id,
        source_file=args.source_file,
        reindex=args.reindex,
        dry_run=args.dry_run,
    )

    if not args.dry_run and deleted > 0:
        print(f"\n🎉 清理完成，共删除 {deleted} 行。")


if __name__ == "__main__":
    asyncio.run(main())
