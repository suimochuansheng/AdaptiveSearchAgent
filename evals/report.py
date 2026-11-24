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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from evals import store
from evals.gate import CRITERION, GateInputError, GateResult, Run

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


def _environment_text(meta: Mapping[str, Any]) -> str:
    environment = meta.get("environment")
    if not isinstance(environment, Mapping):
        return "未记录（历史 run）"
    parts = []
    for key in ("rag_ok", "model_ok", "pg_ok", "redis_ok"):
        value = environment.get(key)
        label = key.removesuffix("_ok")
        parts.append(f"{label}={'ok' if value is True else 'fail'}")
    checked_at = environment.get("checked_at")
    if checked_at:
        parts.append(f"checked_at={checked_at}")
    return " ".join(parts)


def _validity_text(meta: Mapping[str, Any]) -> str:
    valid = meta.get("valid_for_quality")
    if valid is True:
        return "✅ 可用于质量结论"
    if valid is False:
        return "⚠️ 不可用于质量结论（链路验证）"
    return "未记录（历史 run）"


def _partial_text(meta: Mapping[str, Any]) -> str:
    partial = meta.get("partial")
    n_cases = meta.get("n_cases")
    dataset_size = meta.get("dataset_size")
    if partial is True:
        return f"非全量（n_cases={n_cases} / dataset_size={dataset_size}）"
    if partial is False:
        return "全量"
    if isinstance(n_cases, int) and isinstance(dataset_size, int) and n_cases < dataset_size:
        return f"非全量（n_cases={n_cases} / dataset_size={dataset_size}，历史未标注）"
    return "未记录（历史 run）"


def _publishable_text(meta: Mapping[str, Any]) -> str:
    publishable = meta.get("publishable")
    if publishable is True:
        return "是"
    if publishable is False:
        return "否"
    return "未记录（历史 run）"


def _is_partial(meta: Mapping[str, Any]) -> bool:
    partial = meta.get("partial")
    if isinstance(partial, bool):
        return partial
    n_cases = meta.get("n_cases")
    dataset_size = meta.get("dataset_size")
    return isinstance(n_cases, int) and isinstance(dataset_size, int) and n_cases < dataset_size


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
        f"| 有效性 | {_validity_text(meta)} |",
        f"| 全量性 | {_partial_text(meta)} |",
        f"| 可发布 | {_publishable_text(meta)} |",
        f"| 环境探活 | {_cell(_environment_text(meta))} |",
        f"| 数据集 | {_cell(meta.get('dataset'))} |",
        f"| 数据集 sha256 | `{_cell(str(meta.get('dataset_sha256') or '')[:16])}…` |",
        f"| 开关快照 | `{_cell(flag_text)}` |",
        f"| schema 指纹 | `{_cell(meta.get('schema_sha256'))}` |",
        f"| 报告文件 | `{report_path}` |",
    ]
    return lines


def _matrix_section(candidate: Run, result: GateResult | None = None) -> list[str]:
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
    lines.append("")
    if result is None:
        lines.append("单题当量：— ｜ 有效阈值：—（无对比基线）")
        lines.append("逐题抖动：—")
        return lines
    significance = "不显著" if result.sign_test_p > 0.05 else "显著"
    lines.append(
        f"单题当量：{result.one_case_weight:.4f} ｜ "
        f"有效阈值：{result.threshold_used:.4f}"
        f"（方差 {result.recorded_variance:.4f} 与当量取大）"
    )
    lines.append(
        f"逐题抖动：升 {result.flips.get('up', 0)} / 降 {result.flips.get('down', 0)} / "
        f"不变 {result.flips.get('unchanged', 0)} ｜ "
        f"符号检验 p={result.sign_test_p:.2f}（{'<=0.05' if not result.sign_test_p > 0.05 else '>0.05'}，{significance}）"
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
    ]
    if result.missing_in_candidate:
        lines.append(f"- 候选缺题：{', '.join(result.missing_in_candidate)}")
    if result.missing_in_baseline:
        lines.append(f"- 基线缺题：{', '.join(result.missing_in_baseline)}")
    lines += [
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
    criterion = result.criterion if result is not None else CRITERION
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
    ]
    warnings: list[str] = []
    if run.meta.get("valid_for_quality") is False:
        warnings.append("本次为链路验证 run，环境未就绪，分数不可用于质量结论")
    if _is_partial(run.meta):
        warnings.append(
            f"非全量 run（n_cases={run.meta.get('n_cases')} / "
            f"dataset_size={run.meta.get('dataset_size')}），结论不可用于发布或写简历"
        )
    if compare_result is not None:
        if compare_result.quality_warning:
            warnings.append(compare_result.quality_warning)
        if compare_result.code_version_changed:
            warnings.append(
                f"两次 run 代码版本不同（{compare_result.baseline_git} vs "
                f"{compare_result.candidate_git}），分数差异不能归因于配置开关"
            )
        if compare_result.partial_compare:
            warnings.append(
                f"仅比较共同题（compared_cases={compare_result.compared_cases} / "
                f"skipped={compare_result.skipped}），结论不可跨版本比较"
            )
    for warning in warnings:
        lines += [f"> ⚠️ {warning}", ""]
    lines += [
        f"> 配置 {run.meta.get('config_id')} / 数据集 {run.meta.get('dataset')}；"
        "本报告由 `evals/report.py` 自动生成，结论句来自 `evals/gate.py`。",
        "",
    ]
    lines += _meta_section(run, report_path)
    lines += [""]
    lines += _matrix_section(run, compare_result)
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
