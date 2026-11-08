"""
RAG 质量评估脚本 — 使用 Ragas 量化检索 + 生成质量。

当前评测口径（2026-11 起）：
  - Embedding 仅 bge-m3（nomic-embed 已下线）
  - 黄金集默认 qa_dataset_v3.yaml
  - 文档归因拆成 hit@1 / hit@k / hit@k(精排后)，hit@1 只作参考项
  - 数据集 sha256 写进报告，保证基线可复现

用法:
    poetry run python evals/rag_quality.py --kb_id eval_v3
    poetry run python evals/rag_quality.py --dataset evals/datasets/qa_dataset_v3.yaml

输出:
    ../文档/项目/AdaptiveSearchAgent/评测报告/ragas_evaluation_bge-m3_bge.json / .csv
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
import types
from collections import Counter
from pathlib import Path

import httpx
import yaml  # type: ignore[import-untyped]

# 确保项目根在 sys.path
_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

from dotenv import load_dotenv

load_dotenv(_PROJ / ".env")

# ── Monkey-patch: 桥接 Ragas 0.4.x 硬依赖 ──
import langchain_community.chat_models as _chat_models
from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI  # type: ignore[attr-defined]

_chat_models.ChatVertexAI = _ChatVertexAI  # type: ignore[attr-defined]
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = _ChatVertexAI  # type: ignore[attr-defined]
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

import torch
from datasets import Dataset
from langchain_community.embeddings import OllamaEmbeddings
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import answer_relevancy, context_recall, faithfulness
from ragas.run_config import RunConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import settings
from src.tools.rag import search_knowledge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("ragas-eval")

# ── 阿里云百炼配置（裁判 + 生成）─────────────────────────────
QWEN_MODEL = settings.ALIYUN_QWEN_MODEL_NAME
QWEN_API_KEY = settings.ALIYUN_QWEN_API_KEY
QWEN_BASE_URL = settings.ALIYUN_QWEN_BASE_URL

# ── 评测口径 ──────────────────────────────────────────────
EMBEDDING_MODEL = "bge-m3"
OLLAMA_EMBEDDING = "bge-m3"
DEFAULT_DATASET = _PROJ / "evals" / "datasets" / "qa_dataset_v3.yaml"
METRIC_COLUMNS = ["context_recall", "faithfulness", "answer_relevancy"]


def _build_llm_and_embeddings():
    """构建裁判员 LLM 与对齐的向量 Embedding 模型（仅 bge-m3）。"""
    qwen_client = OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)
    llm = llm_factory(
        QWEN_MODEL,
        client=qwen_client,
        max_tokens=2048,
        temperature=0.0,
        extra_body={  # 关闭思考模式
            "enable_thinking": False,
        },
    )

    lc_embed = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=OLLAMA_EMBEDDING,
    )
    emb = LangchainEmbeddingsWrapper(lc_embed)

    logger.info("🤖 裁判 LLM: %s | 🧬 Ragas 评估 Embedding: %s", QWEN_MODEL, OLLAMA_EMBEDDING)
    return llm, emb


def _build_answer_prompt(question: str, context: str) -> str:
    return (
        "请根据以下参考文档，简洁、严谨地回答用户的问题。如果文档中未提及相关信息，请直接回答无法回答。\n\n"
        f"【参考文档】：\n{context}\n\n"
        f"【用户问题】：\n{question}\n\n"
        "【回答】："
    )


async def _generate_answer_ollama(prompt: str) -> str:
    """用本地 Ollama 模型生成回答（被测 RAG 系统的生成器）。"""
    payload = {
        "model": settings.ollama_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,  # qwen3.5 9B 支持 thinking，评测时关掉，避免把思考内容算进答案
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        resp.raise_for_status()
        message = resp.json().get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise RuntimeError("Ollama 返回空 content")
        return content


async def _generate_answer(question: str, context: str) -> str:
    """根据检索到的上下文，用本地 Ollama 模型回答。"""
    if not context or not context.strip():
        return "抱歉，根据已知文档内容，无法回答该问题。"

    prompt = _build_answer_prompt(question, context)
    try:
        return await _generate_answer_ollama(prompt)
    except Exception as e:  # noqa: BLE001
        logger.warning("⚠️ 本地生成失败，降级使用上下文: %s", e)
        return context


def load_test_data(yaml_path: Path):
    """加载 YAML 测试集，返回 (test_data, metadata_fields, sha256)。"""
    if not yaml_path.exists():
        raise FileNotFoundError(f"未找到测试集文件: {yaml_path}")

    raw = yaml_path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"测试集为空或格式不对: {yaml_path}")
    logger.info("从 YAML 加载 %d 条测试问题: %s (sha256=%s)", len(data), yaml_path, sha256[:12])

    # 提取元数据字段名（核心字段之外的键）
    core_keys = {"question", "ground_truth", "source_file", "source_files", "id"}
    metadata_fields: set[str] = set()
    metadata_fields = set(data[0].keys()) - core_keys
    return data, metadata_fields, sha256


def _expected_sources(item: dict) -> list[str]:
    """支持 source_files: [..] 列表；兼容旧的 source_file 单值。"""
    files = item.get("source_files")
    if files:
        return [str(f) for f in files] if isinstance(files, list) else [str(files)]
    one = item.get("source_file")
    return [str(one)] if one else []


def _source_matches(source: str, expected: list[str]) -> bool:
    if not source or not expected:
        return False
    source_name = Path(source).name
    return any(source == e or source_name == e or source_name == Path(e).name for e in expected)


def _hit_at(passages: list[dict], expected: list[str], k: int) -> bool:
    return any(_source_matches(p["source"], expected) for p in passages[:k])


def _format_context(passages: list[dict]) -> str:
    parts: list[str] = []
    for p in passages:
        if p.get("source"):
            parts.append(f"【{p['source']}】\n{p['content']}")
        else:
            parts.append(p["content"])
    return "\n\n".join(parts)


# ============================================================
# BGE Reranker（离线精排，不修改子服务架构）
# ============================================================
_bge_model = None
_bge_tokenizer = None


def _get_bge_reranker():
    """延迟加载 BGE Reranker 模型（单例）。"""
    global _bge_model, _bge_tokenizer
    if _bge_model is None:
        model_name = "BAAI/bge-reranker-v2-m3"
        # 优先只读本地缓存：离线/网络受限时 from_pretrained 会卡在 hub 检查，
        # 模型已缓存的场景下 local_files_only=True 能把加载从分钟级降到秒级。
        try:
            _bge_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            _bge_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            logger.info("本地缓存未命中，尝试联网加载 BGE Reranker: %s", model_name)
            _bge_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _bge_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if torch.cuda.is_available():
            _bge_model = _bge_model.cuda()
        _bge_model.eval()
        logger.info("✅ BGE Reranker 已加载: %s", model_name)
    return _bge_model, _bge_tokenizer


def _rerank_with_bge(query: str, passages: list[dict], top_k: int = 3) -> list[dict]:
    """对 passages 用 BGE 做重排序，返回 Top-K（保留 source/content 结构）。"""
    if not passages:
        return passages

    model, tokenizer = _get_bge_reranker()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    texts = [p["content"] for p in passages]
    inputs = tokenizer(
        [query] * len(texts),
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        scores = model(**inputs, return_dict=True).logits.view(-1).cpu().tolist()

    order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [passages[i] for i in order]


_RERANK_PREFIX = re.compile(r"^\[Reranked[^\]]*\]\s*\n?")
_CHUNK_SPLIT = re.compile(r"\n{2,}(?=【)")


def _parse_passages_from_context(context: str) -> list[dict]:
    """从 RAG 检索返回的格式化字符串中提取 (source, content)。

    网关格式: 【source_file】(RRF分数: x)\\ncontent\\n\\n【...】
    注意：这里必须用真实换行 \\n，不要写成字面量反斜杠+n（历史 bug）。
    """
    if not context:
        return []
    context = _RERANK_PREFIX.sub("", context.strip())
    passages: list[dict] = []
    for block in _CHUNK_SPLIT.split(context):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        header = lines[0].strip()
        content = lines[1].strip() if len(lines) >= 2 else lines[0].strip()
        match = re.match(r"^【([^】]+)】", header)
        source = match.group(1).strip() if match else ""
        if content:
            passages.append({"source": source, "content": content})
    return passages


async def evaluate_rag(args=None) -> dict:  # noqa: C901
    """离线评估 RAG 系统：检索上下文 → 生成答案 → Ragas 指标（支持并发）。"""
    model_name = getattr(args, "model", EMBEDDING_MODEL)
    if model_name != EMBEDDING_MODEL:
        raise ValueError(f"仅支持 {EMBEDDING_MODEL}（nomic-embed 已下线），收到: {model_name}")
    top_k = getattr(args, "top_k", 3)
    max_workers = getattr(args, "max_workers", 4)
    eval_timeout = getattr(args, "timeout", 120)
    kb_id = str(getattr(args, "kb_id", None) or "eval_v3")
    reranker = getattr(args, "reranker", "bge")

    # BGE Reranker 模式下，先取更大候选池再精排回 top_k
    retrieval_top_k = top_k * 3 if reranker == "bge" else top_k

    # 1. 加载 YAML 测试集
    dataset_arg = getattr(args, "dataset", None)
    dataset_path = Path(dataset_arg) if dataset_arg else DEFAULT_DATASET
    if not dataset_path.is_absolute():
        dataset_path = _PROJ / dataset_path
    test_data, metadata_fields, dataset_sha256 = load_test_data(dataset_path)
    logger.info(
        "加载 %d 条测试问题（模型: %s, kb_id: %s, top_k: %d, max_workers: %d, timeout: %ds）",
        len(test_data),
        model_name,
        kb_id,
        top_k,
        max_workers,
        eval_timeout,
    )

    # 2. 构建 LLM 与对齐的 Embeddings
    llm, emb = _build_llm_and_embeddings()

    # 3. 检索 + 生成回答
    t_retrieval_start = time.perf_counter()
    ragas_rows: list[dict] = []
    meta_rows: list[dict] = []
    retrieval_errors = 0
    generation_fallbacks = 0
    context_sizes: list[int] = []

    for idx, item in enumerate(test_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_sources = _expected_sources(item)

        logger.info("[%d/%d] 检索并生成回答: %s", idx + 1, len(test_data), question[:50])

        # A. 检索（hybrid=BM25+向量RRF，跳过网关侧 Reranker，由本脚本离线精排）
        context = await search_knowledge(
            question,
            top_k=retrieval_top_k,
            kb_id=kb_id,
            model=model_name,
            search_mode="hybrid",
            skip_reranker=True,
        )
        passages = _parse_passages_from_context(context)
        if not passages:
            retrieval_errors += 1

        # B. 归因：网关原始顺序上的 hit@1 / hit@k
        hit_at_1 = _hit_at(passages, expected_sources, 1)
        hit_at_k = _hit_at(passages, expected_sources, top_k)

        # C. BGE 精排（若启用）
        if reranker == "bge" and len(passages) > top_k:
            reranked = _rerank_with_bge(question, passages, top_k=top_k)
            if reranked:
                passages = reranked
        final_passages = passages[:top_k]
        hit_at_k_reranked = _hit_at(final_passages, expected_sources, len(final_passages))
        context_sizes.append(len(final_passages))

        final_context = _format_context(final_passages)

        # D. 让 LLM 基于上下文生成最终回答
        generated_answer = await _generate_answer(question, final_context)
        if generated_answer == final_context:  # _generate_answer 降级返回了原始上下文
            generation_fallbacks += 1

        ragas_rows.append(
            {
                "user_input": question,
                "response": generated_answer,
                "retrieved_contexts": [p["content"] for p in final_passages] or [""],
                "reference": ground_truth,
            }
        )
        meta_rows.append(
            {
                "id": item.get("id", ""),
                "expected_sources": expected_sources,
                "retrieved_sources": [p["source"] for p in final_passages],
                "doc_hit_at_1": int(hit_at_1),
                "doc_hit_at_k": int(hit_at_k),
                "doc_hit_at_k_reranked": int(hit_at_k_reranked),
                "retrieved_chunk_count": len(final_passages),
                "category": item.get("category", ""),
                "difficulty": item.get("difficulty", ""),
                "tags": item.get("tags", []),
            }
        )

    t_retrieval_elapsed = time.perf_counter() - t_retrieval_start
    logger.info(
        "检索+生成完成: %d 条, 耗时 %.1fs, 检索空结果 %d 次, 生成降级 %d 次",
        len(ragas_rows),
        t_retrieval_elapsed,
        retrieval_errors,
        generation_fallbacks,
    )

    # 4. 构建 Ragas Dataset（只放 Ragas 认识的 4 列，元信息单独存）
    ragas_ds = Dataset.from_list(ragas_rows).rename_columns(
        {
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": "contexts",
            "reference": "ground_truth",
        }
    )

    # 5. 运行评估（并发 + 串行降级）
    run_config = RunConfig(max_workers=max_workers, timeout=eval_timeout)
    logger.info(
        "开始 Ragas 评估 (%d 条, 3 项指标, max_workers=%d, timeout=%ds)...",
        len(ragas_rows),
        max_workers,
        eval_timeout,
    )
    t_eval_start = time.perf_counter()

    try:
        result = evaluate(
            dataset=ragas_ds,
            metrics=[context_recall, faithfulness, answer_relevancy],
            llm=llm,
            embeddings=emb,
            run_config=run_config,
        )
        actual_workers = max_workers
        logger.info("Ragas 并发评估完成 (workers=%d)", actual_workers)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "⚠️ 并发评估失败 (workers=%d, error=%s)，自动降级为串行 (max_workers=1)",
            max_workers,
            str(e)[:120],
        )
        try:
            fallback_config = RunConfig(max_workers=1, timeout=eval_timeout)
            result = evaluate(
                dataset=ragas_ds,
                metrics=[context_recall, faithfulness, answer_relevancy],
                llm=llm,
                embeddings=emb,
                run_config=fallback_config,
            )
            actual_workers = 1
            logger.info("✅ 串行降级评估完成")
        except Exception as e2:  # noqa: BLE001
            logger.exception("❌ 串行降级评估也失败")
            raise RuntimeError(f"Ragas 评估失败（并发与串行均失败）: {e2}") from e2

    t_eval_elapsed = time.perf_counter() - t_eval_start
    t_total = t_retrieval_elapsed + t_eval_elapsed

    # 6. 报告输出
    from src.utils.report_paths import eval_report_dir

    report_dir = eval_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)

    df = result.to_pandas()
    for col in reversed(list(meta_rows[0].keys())):
        df.insert(0, col, [row[col] for row in meta_rows])

    # 生成器固定为本地 Ollama，报告统一带 _ollama，避免和旧的 Qwen 参考报告混淆
    report_suffix = (
        f"{model_name}_{reranker}_ollama" if reranker != "ms-marco" else f"{model_name}_ollama"
    )
    csv_path = report_dir / f"ragas_evaluation_{report_suffix}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    hit1_values = [row["doc_hit_at_1"] for row in meta_rows]
    hitk_values = [row["doc_hit_at_k"] for row in meta_rows]
    hitkr_values = [row["doc_hit_at_k_reranked"] for row in meta_rows]
    summary: dict = {
        "model_evaluated": model_name,
        "embedding_model": EMBEDDING_MODEL,
        "judge_model": QWEN_MODEL,
        "generator_model": settings.ollama_model_name,
        "reranker": reranker,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "kb_id": kb_id,
        "total_questions": len(ragas_rows),
        "retrieval_top_k": retrieval_top_k,
        "max_workers": actual_workers,
        "timeout_seconds": eval_timeout,
        "elapsed_seconds": {
            "retrieval": round(t_retrieval_elapsed, 1),
            "evaluation": round(t_eval_elapsed, 1),
            "total": round(t_total, 1),
        },
        "retrieval_errors": retrieval_errors,
        "generation_fallbacks": generation_fallbacks,
        # 归因只作参考项：golden 集的 source_file 是单值，且同一答案可能来自多篇文档
        "doc_attribution_reference_only": True,
        "doc_attribution_accuracy": round(sum(hit1_values) / len(hit1_values), 4)
        if hit1_values
        else 0,
        "doc_hit_at_1": round(sum(hit1_values) / len(hit1_values), 4) if hit1_values else 0,
        "doc_hit_at_k": round(sum(hitk_values) / len(hitk_values), 4) if hitk_values else 0,
        "doc_hit_at_k_reranked": round(sum(hitkr_values) / len(hitkr_values), 4)
        if hitkr_values
        else 0,
        "doc_hit_k": top_k,
        "retrieved_chunk_count": {
            "min": min(context_sizes) if context_sizes else 0,
            "max": max(context_sizes) if context_sizes else 0,
            "avg": round(sum(context_sizes) / len(context_sizes), 2) if context_sizes else 0,
        },
        "metrics": {},
        "metadata_stats": {},
    }

    # 元数据统计
    if metadata_fields:
        for field in sorted(metadata_fields):
            values = [item.get(field) for item in test_data if field in item]
            if values and isinstance(values[0], list):
                flat = [t for tags in values for t in (tags if isinstance(tags, list) else [tags])]
                summary["metadata_stats"][field] = dict(Counter(flat))
            else:
                summary["metadata_stats"][field] = dict(Counter(values))

    for col in METRIC_COLUMNS:
        if col not in df.columns:
            continue
        val = df[col].dropna()
        if len(val) == 0:
            continue
        summary["metrics"][col] = {
            "mean": round(float(val.mean()), 4),
            "median": round(float(val.median()), 4),
            "min": round(float(val.min()), 4),
            "max": round(float(val.max()), 4),
        }

    json_path = report_dir / f"ragas_evaluation_{report_suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(
        f"RAG 质量评估汇总 ({model_name}, reranker={reranker}, kb_id={kb_id}, "
        f"answer={settings.ollama_model_name})"
    )
    print(f"数据集: {dataset_path.name}  sha256={dataset_sha256[:12]}...")
    print("=" * 64)
    for name, stats in summary["metrics"].items():
        print(f"  {name:30s}  mean={stats['mean']:.4f}  median={stats['median']:.4f}")
    print("-" * 64)
    print(
        f"⏱️  检索耗时: {t_retrieval_elapsed:.1f}s  |  评估耗时: {t_eval_elapsed:.1f}s  |  "
        f"总耗时: {t_total:.1f}s"
    )
    print(
        f"🔧 并发 workers: {actual_workers}  |  检索空结果: {retrieval_errors} 次  |  "
        f"生成降级: {generation_fallbacks} 次"
    )
    print(
        f"🎯 归因（参考项）: hit@1={summary['doc_hit_at_1']:.2%}  "
        f"hit@{top_k}={summary['doc_hit_at_k']:.2%}  "
        f"hit@{top_k}(精排后)={summary['doc_hit_at_k_reranked']:.2%}"
    )
    print(f"📦 上下文切片: {summary['retrieved_chunk_count']}")

    if summary["metadata_stats"]:
        print("-" * 64)
        print("📋 元数据分布 (Metadata Stats):")
        for field, dist in sorted(summary["metadata_stats"].items()):
            print(f"  [{field}] {dist}")

    logger.info("📄 CSV 报告已生成: %s", csv_path)
    logger.info("📊 JSON 报告已生成: %s", json_path)
    logger.info(
        "⏱️  总耗时: %.1fs (检索 %.1fs + 评估 %.1fs, workers=%d)",
        t_total,
        t_retrieval_elapsed,
        t_eval_elapsed,
        actual_workers,
    )
    return summary


async def main():
    parser = argparse.ArgumentParser(description="RAG 系统质量评估（仅 bge-m3）")
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
        choices=[EMBEDDING_MODEL],
        help="Embedding 模型（仅 bge-m3；nomic-embed 已下线）",
    )
    parser.add_argument("--top_k", type=int, default=3, help="最终送入生成/评估的切片数")
    parser.add_argument(
        "--kb_id",
        type=str,
        default="eval_v3",
        help="指定测试的知识库 ID（默认 eval_v3，黄金集专用库）",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Ragas 评估并发 worker 数（建议 2~10）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="单个评估任务超时时间（秒）",
    )
    parser.add_argument(
        "--reranker",
        default="bge",
        choices=["ms-marco", "bge"],
        help="ms-marco=只取网关 RRF 前 top_k；bge=候选池再用 BGE 精排回 top_k",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="测试集 YAML 路径（默认 evals/datasets/qa_dataset_v3.yaml）",
    )
    args = parser.parse_args()
    await evaluate_rag(args=args)


if __name__ == "__main__":
    asyncio.run(main())
