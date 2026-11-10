#!/usr/bin/env python
"""T01 环境搭建 + Ragas 基线快照：可执行检查脚本（只读，不写任何文件）。

为什么文件名不是 test_*.py：
    一旦放进 pytest 收集范围，会让 `pytest --collect-only` 从 227/231 变成 227+N，
    破坏 docs/baseline.md 里冻结的 nodeids 基线哈希。这里做成独立脚本，
    既保留自动化能力，又不影响测试快照。

用法（在仓库根目录）：
    poetry run python "tests/test_agent_cases/T01-环境搭建+ragas基线快照/t01_checks.py"

退出码：0 = 全部通过；1 = 有失败项。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO.parent / "文档" / "项目" / "AdaptiveSearchAgent" / "评测报告"
BASELINE = REPORT_DIR / "baseline_8100"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def http_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - 本机服务
        return json.loads(resp.read().decode("utf-8"))


def check_env() -> None:
    # T01-001 虚拟环境
    venv = Path.home() / ".cache/pypoetry/virtualenvs/adsearchagent-w97WTAdM-py3.11"
    check("T01-001 Poetry 虚拟环境存在", venv.exists(), str(venv))

    # T01-002 评测依赖（只探测模块是否可寻，不实际 import 重包）
    missing = [m for m in ("ragas", "datasets", "torch") if importlib.util.find_spec(m) is None]
    check(
        "T01-002 评测依赖齐备",
        not missing,
        "缺少: " + ",".join(missing) if missing else "ragas/datasets/torch",
    )

    # T01-003 docker 容器
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=False
        ).stdout
        names = out.split()
        need = ["postgres_dev", "redis_cache"]
        ok = all(n in names for n in need)
        check("T01-003 PG/Redis 容器在跑", ok, ",".join(n for n in need if n in names))
    except Exception as exc:  # noqa: BLE001
        check("T01-003 PG/Redis 容器在跑", False, f"{type(exc).__name__}: {exc}")

    # T01-004 Ollama 模型
    try:
        tags = http_json("http://localhost:11434/api/tags")
        models = [m.get("name", "") for m in tags.get("models", [])]
        ok = any("qwen35-9b-official" in m for m in models) and any("bge-m3" in m for m in models)
        check("T01-004 Ollama 模型可用", ok, f"{len(models)} 个模型")
    except Exception as exc:  # noqa: BLE001
        check("T01-004 Ollama 模型可用", False, f"{type(exc).__name__}: {exc}")

    # T01-005 RAG 网关健康
    try:
        with urllib.request.urlopen("http://localhost:8100/health", timeout=5) as resp:  # noqa: S310
            code = resp.status
        check("T01-005 ingest_dev 8100 健康", code == 200, f"HTTP {code}")
    except Exception as exc:  # noqa: BLE001
        check("T01-005 ingest_dev 8100 健康", False, f"{type(exc).__name__}: {exc}")


def check_rag_latency() -> None:
    """T01-006 单次检索必须非空且低于 RAG_TIMEOUT_SECONDS。"""
    sys.path.insert(0, str(REPO))
    try:
        from src.tools.rag import search_knowledge

        t0 = time.perf_counter()
        text = asyncio.run(search_knowledge("PostgreSQL 写入吞吐量", kb_id="eval_v3"))
        elapsed = time.perf_counter() - t0
        check(
            "T01-006 RAG 检索非空且 <30s",
            len(text) > 0 and elapsed < 30,
            f"{len(text)} 字符 / {elapsed:.1f}s",
        )
    except Exception as exc:  # noqa: BLE001
        check("T01-006 RAG 检索非空且 <30s", False, f"{type(exc).__name__}: {exc}")


def check_baseline_report() -> None:
    """T01-010~013/017 基线报告的口径与指标。"""
    json_path = BASELINE / "ragas_evaluation_bge-m3_bge_ollama.json"
    meta_path = BASELINE / "run_meta.txt"
    check("T01-010 基线报告存在", json_path.exists() and meta_path.exists(), str(BASELINE))
    if not json_path.exists() or not meta_path.exists():
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    recall = (metrics.get("context_recall") or {}).get("mean")
    check("T01-010 context_recall = 1.0", recall == 1.0, str(recall))

    faith = (metrics.get("faithfulness") or {}).get("mean")
    relev = (metrics.get("answer_relevancy") or {}).get("mean")
    check(
        "T01-011 指标落在基线值",
        faith is not None
        and relev is not None
        and abs(faith - 0.8030) < 0.05
        and abs(relev - 0.8339) < 0.05,
        f"faith={faith} relev={relev}（基线 0.8030 / 0.8339）",
    )
    check(
        "T01-012 空检索=0 且生成降级=0",
        data.get("retrieval_errors") == 0 and data.get("generation_fallbacks") == 0,
        f"errors={data.get('retrieval_errors')} fallbacks={data.get('generation_fallbacks')}",
    )
    check(
        "T01-011 裁判为 max-0902",
        data.get("judge_model") == "qwen3.8-max-0902",
        str(data.get("judge_model")),
    )

    meta = meta_path.read_text(encoding="utf-8")
    check(
        "T01-013 元数据含关键字段",
        all(k in meta for k in ("项目 commit", "裁判 LLM", "RAG 子服务", "并发 workers")),
    )
    check("T01-017 基线走容器网关", "http://localhost:8100" in meta, "run_meta 中 RAG 子服务")
    check(
        "T01-015 裁判来自 .env（非硬编码）",
        "settings.ALIYUN_QWEN_MODEL_NAME"
        in (REPO / "evals/agent_quality.py").read_text(encoding="utf-8"),
    )


def main() -> int:
    print(f"仓库: {REPO}")
    check_env()
    check_rag_latency()
    check_baseline_report()

    failed = [r for r in results if not r[1]]
    print("\n" + "=" * 60)
    print(f"合计 {len(results)} 项，通过 {len(results) - len(failed)} 项，失败 {len(failed)} 项")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
