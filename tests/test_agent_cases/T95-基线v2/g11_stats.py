#!/usr/bin/env python3
"""G11 三组评测报告汇总：从 `agent_quality.py` 产出的 JSON 里取 §8 需要的列。

用法：

    poetry run python tests/test_agent_cases/T95-基线v2/g11_stats.py \
        <报告A.json> <报告B.json> <报告C.json>

输出每组：均分 / 有效题数 / 失败数 / 分数分布 / 平均耗时 / 平均工具调用数 / judge_model，
以及可直接贴进 `docs/baseline.md` §8 的 markdown 行（审批次数列留空，由调用方按日志填）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BUCKETS = ("10", "9", "8", "7", "<=6")


def _bucket(score: float) -> str:
    if score >= 10:
        return "10"
    if score >= 9:
        return "9"
    if score >= 8:
        return "8"
    if score >= 7:
        return "7"
    return "<=6"


def stats(report: dict) -> dict[str, Any]:
    results = report["results"]
    scored = [item for item in results if item.get("score") is not None]
    scores = [float(item["score"]) for item in scored]
    dist = dict.fromkeys(BUCKETS, 0)
    for score in scores:
        dist[_bucket(score)] += 1
    elapsed = round(
        sum(float(item.get("elapsed_seconds") or 0) for item in results) / max(len(results), 1), 1
    )
    tools = round(sum(len(item.get("tools") or []) for item in results) / max(len(results), 1), 2)
    errors = sum(1 for item in results if item.get("errors"))
    return {
        "label": report.get("label"),
        "judge_model": report.get("judge_model"),
        "mean": round(sum(scores) / len(scores), 2) if scores else None,
        "scored": len(scored),
        "failed": len(results) - len(scored),
        "total": len(results),
        "dist": dist,
        "avg_elapsed": elapsed,
        "total_elapsed": round(sum(float(item.get("elapsed_seconds") or 0) for item in results), 1),
        "avg_tools": tools,
        "items_with_errors": errors,
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    for path in argv[1:]:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        s = stats(report)
        dist = " / ".join(f"{name}:{count}" for name, count in s["dist"].items())
        print(f"== {s['label']} ==")
        print(f"judge_model = {s['judge_model']}")
        print(
            f"均分 = {s['mean']} | 有效/失败 = {s['scored']}/{s['failed']} | min/max = {s['min']}/{s['max']}"
        )
        print(f"分布(10/9/8/7/<=6) = {dist}")
        print(
            f"平均耗时 = {s['avg_elapsed']}s（累计 {s['total_elapsed']}s） | 平均工具调用 = {s['avg_tools']}"
        )
        print(f"带 errors 的题数 = {s['items_with_errors']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
