"""把一次 run + 门禁结果写成 markdown 报告。

报告结构固定五段：

1. 元信息表（时间 / 配置 / 模型 / 裁判 / 样本 / commit / 开关 / schema 指纹）
2. 配置矩阵分数表（同一数据集的 mean / 方差 / 样本数）
3. 与上一次 run 的分题 diff 表
4. 结论段（判据原文 + gate 生成的结论句）
5. 失败 case 清单（<= 10 条，带归因分类）

结论句只从 `GateResult.reason` 取，不在本模块手写。
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any

from evals import store
from evals.gate import GateInputError, GateResult, Run

if TYPE_CHECKING:
    from pathlib import Path


def _scores(run: Run) -> list[float]:
    """取出有效分数；坏分数跳过，报告不因脏数据崩。"""
    scores: list[float] = []
    for case in run.per_case:
        value = case.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        scores.append(float(value))
    return scores


def _mean_text(run: Run) -> str:
    scores = _scores(run)
    return f"{sum(scores) / len(scores):.4f}" if scores else "—"


def _variance_text(run: Run) -> str:
    scores = _scores(run)
    if len(scores) < 2:
        return "—"
    return f"{statistics.pvariance(scores):.4f}"


def _cell(value: Any) -> str:
    text = "—" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _score_text(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _delta_text(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}"


def _matrix_runs(candidate: Run) -> list[Run]:
    """配置矩阵只看同一份数据集；没有 sha 时退回全部 run。"""
    sha = candidate.meta.get("dataset_sha256")
    runs = store.list_runs()
    if not sha:
        return runs
    same = [run for run in runs if run.meta.get("dataset_sha256") == sha]
    return same or runs


def _meta_section(run: Run, report_path: Path) -> list[str]:
    meta = run.meta
    flags = meta.get("flags") or {}
    flag_text = " ".join(f"{key}={value}" for key, value in flags.items()) or "—"
    scored = len(_scores(run))
    lines = [
        "## 1. 元信息",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| run_id | `{_cell(run.run_id)}` |",
        f"| 时间 | {_cell(meta.get('created_at'))} |",
        f"| 配置 | {_cell(meta.get('config_id'))} {_cell(meta.get('config_name'))} |",
        f"| 模型 | {_cell(meta.get('model'))} |",
        f"| 裁判 | {_cell(meta.get('judge_model'))} |",
        f"| 样本数 | {len(run.per_case)}（有效 {scored}） |",
        f"| 代码版本 | `{_cell(meta.get('git_commit'))}` |",
        f"| 数据集 | {_cell(meta.get('dataset'))} |",
        f"| 数据集 sha256 | `{_cell(str(meta.get('dataset_sha256') or '')[:16])}…` |",
        f"| 开关快照 | `{_cell(flag_text)}` |",
        f"| schema 指纹 | `{_cell(meta.get('schema_sha256'))}` |",
        f"| 报告文件 | `{report_path}` |",
    ]
    return lines


def _matrix_section(candidate: Run) -> list[str]:
    lines = [
        "## 2. 配置矩阵分数表",
        "",
        "同一数据集下已落库的 run（均值按有效分数，方差为总体方差）：",
        "",
        "| 配置 | run_id | 时间 | 均值 | 方差 | 样本数 |",
        "|---|---|---|---|---|---|",
    ]
    for run in _matrix_runs(candidate):
        scored = len(_scores(run))
        lines.append(
            f"| {_cell(run.meta.get('config_id'))} | `{_cell(run.run_id)}` "
            f"| {_cell(run.meta.get('created_at'))} | {_mean_text(run)} "
            f"| {_variance_text(run)} | {scored}/{len(run.per_case)} |"
        )
    return lines


def _diff_section(result: GateResult | None) -> list[str]:
    lines = ["## 3. 与上一次 run 的 diff", ""]
    if result is None:
        lines.append("本次运行没有产生可比较的 diff（无对比基线，或对比输入不合法）。")
        return lines

    lines += [
        f"- 基线：`{result.baseline_run_id}`（mean={result.baseline_mean:.4f}）",
        f"- 候选：`{result.candidate_run_id}`（mean={result.candidate_mean:.4f}）",
        f"- 共同有效题：{result.common_cases}",
        "",
        "| case | base | cand | diff | 归因 |",
        "|---|---|---|---|---|",
    ]
    for item in result.evidence:
        lines.append(
            f"| {_cell(item.case_id)} | {_score_text(item.baseline_score)} "
            f"| {_score_text(item.candidate_score)} | {_delta_text(item.delta)} "
            f"| {_cell(item.attribution or '—')} |"
        )
    return lines


def _conclusion_section(result: GateResult | None) -> list[str]:
    lines = ["## 4. 结论", ""]
    criterion = (
        result.criterion
        if result is not None
        else ("同口径两次均值差 <= 记录方差（默认 0.0408）；单题差值只做归因，不单独判失败")
    )
    lines.append(f"> 判据原文：{criterion}")
    lines.append("")
    if result is None:
        lines.append("**未判定**：本次运行没有可用的对比基线（或对比输入不合法），不生成门禁结论。")
    else:
        lines.append(f"**{result.reason}**")
    return lines


def _failures_section(result: GateResult | None) -> list[str]:
    lines = ["## 5. 失败 case 清单（退化 / 报错 / 超时）", ""]
    if result is None or not result.failures:
        lines.append("无退化、报错或超时的 case。")
        return lines

    shown = result.failures[:10]
    lines += [
        "| case | base | cand | diff | 归因 | 详情 |",
        "|---|---|---|---|---|---|",
    ]
    for item in shown:
        lines.append(
            f"| {_cell(item.case_id)} | {_score_text(item.baseline_score)} "
            f"| {_score_text(item.candidate_score)} | {_delta_text(item.delta)} "
            f"| {_cell(item.attribution)} | {_cell(item.detail)} |"
        )
    hidden = len(result.failures) - len(shown)
    if hidden:
        lines.append("")
        lines.append(f"另有 {hidden} 条失败未列出（报告最多列 10 条）。")
    return lines


def write_markdown(run_id: str, compare_result: GateResult | None) -> Path:
    """生成 `evals/results/<run_id>/report.md`，返回路径。

    Args:
        run_id: 候选 run id（报告主体）。
        compare_result: gate 结果；None 表示本次没有可比较基线。

    Raises:
        StoreError: run 不存在或读不了。
        ValueError: compare_result 与 run_id 不匹配。
    """
    run = store.load_run(run_id)
    if compare_result is not None and compare_result.candidate_run_id != run_id:
        raise GateInputError(
            f"compare_result 的 candidate={compare_result.candidate_run_id!r} "
            f"与报告 run_id={run_id!r} 不一致"
        )

    report_path = store.RESULTS_DIR / run_id / "report.md"
    lines = [
        f"# 评测报告 — {run_id}",
        "",
        f"> 配置 {run.meta.get('config_id')} / 数据集 {run.meta.get('dataset')}；"
        "本报告由 `evals/report.py` 自动生成，结论句来自 `evals/gate.py`。",
        "",
    ]
    lines += _meta_section(run, report_path)
    lines += [""]
    lines += _matrix_section(run)
    lines += [""]
    lines += _diff_section(compare_result)
    lines += [""]
    lines += _conclusion_section(compare_result)
    lines += [""]
    lines += _failures_section(compare_result)
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
