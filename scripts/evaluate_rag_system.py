"""
RAG 质量评估脚本 — 使用 Ragas 量化检索 + 生成质量。

用法:
    poetry run python scripts/evaluate_rag_system.py --model nomic-embed
    poetry run python scripts/evaluate_rag_system.py --model bge-m3

输出:
    docs/test_reports/ragas_evaluation_nomic-embed.json / .csv
    docs/test_reports/ragas_evaluation_bge-m3.json / .csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import types
from collections import Counter
from pathlib import Path

import yaml

# 确保项目根在 sys.path
_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

from dotenv import load_dotenv

load_dotenv(_PROJ / ".env")

# ── Monkey-patch: 桥接 Ragas 0.4.x 硬依赖 ──
import langchain_community.chat_models as _chat_models
from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI

_chat_models.ChatVertexAI = _ChatVertexAI
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = _ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

# BGE Reranker 依赖（离线评估对比用）
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

# ── 阿里云百炼配置 ────────────────────────────────────────────
QWEN_MODEL = settings.ALIYUN_QWEN_MODEL_NAME
QWEN_API_KEY = settings.ALIYUN_QWEN_API_KEY
QWEN_BASE_URL = settings.ALIYUN_QWEN_BASE_URL


def _build_llm_and_embeddings(model_key: str = "nomic-embed"):
    """构建裁判员 LLM 与对齐的向量 Embedding 模型。"""
    qwen_client = OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)
    # 获取思考深度配置，并传递给 llm_factory
    reasoning_effort = getattr(settings, "ALIYUN_QWEN_MODEL_REASONING_EFFORT", "medium")
    llm = llm_factory(
        QWEN_MODEL,
        client=qwen_client,
        max_tokens=2048,
        temperature=0.0,
        # 在这里添加参数
        model_kwargs={
            "reasoning_effort": reasoning_effort,
            "enable_thinking": True,  # 评估场景建议开启
        },
    )

    # 映射命令行参数到 Ollama 模型名称（避免硬编码错位）
    model_map = {"nomic-embed": "nomic-embed-text", "bge-m3": "bge-m3"}
    ollama_model = model_map.get(model_key, "nomic-embed-text")

    lc_embed = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=ollama_model,
    )
    emb = LangchainEmbeddingsWrapper(lc_embed)

    logger.info(f"🤖 裁判 LLM: {QWEN_MODEL} | 🧬 Ragas 评估 Embedding: {ollama_model}")
    return llm, emb, qwen_client


async def _generate_answer(qwen_client: OpenAI, question: str, context: str) -> str:
    """根据检索到的上下文，让 LLM 生成真实的回答。"""
    if not context or not context.strip():
        return "抱歉，根据已知文档内容，无法回答该问题。"

    prompt = (
        f"请根据以下参考文档，简洁、严谨地回答用户的问题。如果文档中未提及相关信息，请直接回答无法回答。\n\n"
        f"【参考文档】：\n{context}\n\n"
        f"【用户问题】：\n{question}\n\n"
        f"【回答】："
    )
    try:
        resp = qwen_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"⚠️ 生成回答失败，降级使用上下文: {e}")
        return context


def load_test_data(yaml_path: Path):
    """
    加载 YAML 格式测试集。

    返回 (test_data, metadata_fields) 元组：
      - test_data: list[dict]  测试用例列表
      - metadata_fields: set[str]  YAML 中额外的元数据字段名（如 category, difficulty, tags）
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"未找到测试集文件: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.info("从 YAML 加载 %d 条测试问题: %s", len(data), yaml_path)

    # 提取元数据字段名（question / ground_truth / source_file / id 之外的键）
    core_keys = {"question", "ground_truth", "source_file", "id"}
    metadata_fields: set[str] = set()
    if data:
        metadata_fields = set(data[0].keys()) - core_keys
    return data, metadata_fields


async def evaluate_rag(args=None) -> dict:
    """离线评估 RAG 系统：检索上下文 → 模拟答案 → Ragas 指标（支持并发）。"""
    model_name = getattr(args, "model", "nomic-embed")
    top_k = getattr(args, "top_k", 3)
    max_workers = getattr(args, "max_workers", 4)
    eval_timeout = getattr(args, "timeout", 120)
    kb_id = getattr(args, "kb_id", None)
    reranker = getattr(args, "reranker", "ms-marco")

    # BGE Reranker 模式下，先取更大候选池再精排回 top_k
    retrieval_top_k = top_k * 3 if reranker == "bge" else top_k

    # 1. 加载 YAML 测试集
    dataset_path = _PROJ / "data" / "eval" / "qa_dataset_v2.yaml"
    test_data, _metadata_fields = load_test_data(dataset_path)
    logger.info(
        "加载 %d 条测试问题（模型: %s, top_k: %d, max_workers: %d, timeout: %ds）",
        len(test_data),
        model_name,
        top_k,
        max_workers,
        eval_timeout,
    )

    # 2. 构建 LLM 与对齐的 Embeddings
    llm, emb, qwen_client = _build_llm_and_embeddings(model_key=model_name)

    # 3. 检索 + 生成回答
    t_retrieval_start = time.perf_counter()
    eval_rows: list[dict] = []
    retrieval_errors = 0
    doc_hit_count = 0  # ★ 文档归因命中计数
    for idx, item in enumerate(test_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        target_source_file = item.get("source_file")  # ★ 期望的来源文件名

        logger.info("[%d/%d] 检索并生成回答: %s", idx + 1, len(test_data), question[:50])

        # A. 检索上下文
        context = await search_knowledge(
            question, top_k=retrieval_top_k, kb_id=kb_id, model=model_name
        )

        # BGE Reranker 离线重排（解析格式化字符串 → 精排 → 重建上下文）
        if reranker == "bge" and context:
            passages = _parse_passages_from_context(context)
            if len(passages) > top_k:
                reranked = _rerank_with_bge(question, passages, top_k=top_k)
                if reranked:
                    context = "[Reranked by BGE]\n" + "\n\n".join(reranked)

        # ★ 文档归因校验：检查检索结果第一行是否包含期望的源文件名
        if target_source_file and context:
            first_line = context.split("\n")[0] if "\n" in context else context
            if target_source_file in first_line:
                doc_hit_count += 1

        # B. 让 LLM 基于上下文生成最终回答
        generated_answer = await _generate_answer(qwen_client, question, context)
        if generated_answer == context:  # _generate_answer 降级返回了原始上下文
            retrieval_errors += 1

        eval_rows.append(
            {
                "user_input": question,
                "response": generated_answer,
                "retrieved_contexts": [context] if context else [""],
                "reference": ground_truth,
            }
        )
    t_retrieval_elapsed = time.perf_counter() - t_retrieval_start
    logger.info(
        "检索+生成完成: %d 条, 耗时 %.1fs, 检索异常降级 %d 次",
        len(eval_rows),
        t_retrieval_elapsed,
        retrieval_errors,
    )

    # 4. 构建 Ragas Dataset
    ragas_ds = Dataset.from_list(eval_rows)
    ragas_ds = ragas_ds.rename_columns(
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
        len(eval_rows),
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
    except Exception as e:
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
        except Exception as e2:
            logger.exception("❌ 串行降级评估也失败")
            raise RuntimeError(f"Ragas 评估失败（并发与串行均失败）: {e2}") from e2

    t_eval_elapsed = time.perf_counter() - t_eval_start
    t_total = t_retrieval_elapsed + t_eval_elapsed

    # 6. 带模型后缀的单独报告输出（防止互相覆盖）
    report_dir = _PROJ / "docs" / "test_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    df = result.to_pandas()
    report_suffix = f"{model_name}_{reranker}" if reranker != "ms-marco" else model_name
    csv_path = report_dir / f"ragas_evaluation_{report_suffix}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary: dict = {
        "model_evaluated": model_name,
        "reranker": reranker,  # 🆕 记录使用的 Reranker
        "total_questions": len(eval_rows),
        "retrieval_top_k": retrieval_top_k,  # 🆕 实际检索候选数
        "max_workers": actual_workers,  # ★ 记录实际使用的并发数
        "timeout_seconds": eval_timeout,  # ★ 记录超时配置
        "elapsed_seconds": {  # ★ 分段耗时
            "retrieval": round(t_retrieval_elapsed, 1),
            "evaluation": round(t_eval_elapsed, 1),
            "total": round(t_total, 1),
        },
        "retrieval_errors": retrieval_errors,  # ★ 记录异常降级次数
        "doc_attribution_accuracy": round(doc_hit_count / len(test_data), 4)
        if test_data
        else 0,  # ★ 文档归因准确率
        "metrics": {},
        "metadata_stats": {},  # ★ 元数据分布（YAML 特有，JSON 回退时为空）
    }
    # ★ 计算元数据统计
    if _metadata_fields:
        for field in sorted(_metadata_fields):
            values = [item.get(field) for item in test_data if field in item]
            if values and isinstance(values[0], list):
                flat = [t for tags in values for t in (tags if isinstance(tags, list) else [tags])]
                summary["metadata_stats"][field] = dict(Counter(flat))
            else:
                summary["metadata_stats"][field] = dict(Counter(values))
    for col in df.columns:
        if col not in ("user_input", "response", "retrieved_contexts", "reference"):
            val = df[col].dropna()
            summary["metrics"][col] = {
                "mean": round(float(val.mean()), 4),
                "median": round(float(val.median()), 4),
                "min": round(float(val.min()), 4),
                "max": round(float(val.max()), 4),
            }

    json_path = report_dir / f"ragas_evaluation_{report_suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"RAG 质量评估汇总 ({model_name}, reranker={reranker})")
    print("=" * 60)
    for name, stats in summary["metrics"].items():
        print(f"  {name:30s}  mean={stats['mean']:.4f}  median={stats['median']:.4f}")
    print("-" * 60)
    print(
        f"⏱️  检索耗时: {t_retrieval_elapsed:.1f}s  |  评估耗时: {t_eval_elapsed:.1f}s  |  总耗时: {t_total:.1f}s"
    )
    print(f"🔧 并发 workers: {actual_workers}  |  检索异常降级: {retrieval_errors} 次")
    print(
        f"🎯 文档精准归因召回率 (Doc Attribution Accuracy): {doc_hit_count}/{len(test_data)} = {doc_hit_count / len(test_data) * 100:.1f}%"
    )

    # ★ 打印元数据分布
    if summary["metadata_stats"]:
        print("-" * 60)
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
    parser = argparse.ArgumentParser(description="RAG 系统质量评估（支持并发）")
    parser.add_argument(
        "--model",
        default="nomic-embed",
        choices=["nomic-embed", "bge-m3"],
        help="指定使用的 Embedding 模型",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="指定检索的文档切片数量 top_k",
    )
    parser.add_argument(
        "--kb_id",
        type=str,
        default=None,
        help="指定测试的知识库 ID",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Ragas 评估并发 worker 数（设置过高可能触发 API 限流，建议 2~10）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="单个评估任务超时时间（秒）",
    )
    parser.add_argument(
        "--reranker",
        default="ms-marco",
        choices=["ms-marco", "bge"],
        help="指定 Reranker 模型用于离线对比评估（ms-marco / bge）",
    )
    args = parser.parse_args()
    await evaluate_rag(args=args)


# ============================================================
# BGE Reranker（离线评估对比用，不修改子服务架构）
# ============================================================
_bge_model = None
_bge_tokenizer = None


def _get_bge_reranker():
    """延迟加载 BGE Reranker 模型（单例）。"""
    global _bge_model, _bge_tokenizer
    if _bge_model is None:
        model_name = "BAAI/bge-reranker-v2-m3"
        _bge_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _bge_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if torch.cuda.is_available():
            _bge_model = _bge_model.cuda()
        _bge_model.eval()
        logger.info("✅ BGE Reranker 已加载: %s", model_name)
    return _bge_model, _bge_tokenizer


def _rerank_with_bge(query: str, passages: list[str], top_k: int = 3) -> list[str]:
    """对 passages 用 BGE 做重排序，返回 Top-K 文本列表。"""
    if not passages:
        return passages

    model, tokenizer = _get_bge_reranker()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = [[query, p] for p in passages]
    inputs = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        scores = model(**inputs, return_dict=True).logits.view(-1).cpu().tolist()

    sorted_pairs = sorted(zip(passages, scores, strict=False), key=lambda x: x[1], reverse=True)
    return [p for p, _ in sorted_pairs[:top_k]]


def _parse_passages_from_context(context: str) -> list[str]:
    """从 RAG 检索返回的格式化字符串中提取各 passage 文本。

    格式: 【source_file】(分数)\\ncontent\\n\\n
    """
    if not context:
        return []
    # 去掉可能的 Reranker 前缀标记
    context = re.sub(r"^\\[Reranked[^\\]]*\\]\\s*\\n?", "", context)
    blocks = re.split(r"\\n\\n+", context.strip())
    passages: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\\n", 1)
        if len(lines) >= 2:
            content = lines[1].strip()
        else:
            content = lines[0].strip()
        if content:
            passages.append(content)
    return passages


if __name__ == "__main__":
    asyncio.run(main())
