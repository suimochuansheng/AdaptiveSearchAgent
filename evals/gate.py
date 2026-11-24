"""评测门禁判定 —— 纯函数，零 IO 副作用（CI 可跑）。

判据（三态，替代原来的双侧 `abs()`）：

    delta = mean(candidate) - mean(baseline)
    delta < -variance          -> FAIL_DEGRADED（退化超出记录噪声）
    delta > +variance          -> WARN_IMPROVED（提升超出记录噪声，需人工复核）
    其余                       -> PASS（同口径无漂移）

`compare()` 只吃两个 run 的 meta + per_case，返回 `GateResult`；不读写文件、
不调模型。CLI 才做 IO（加载 run、打印结论），退出码：

    0 = PASS / WARN_IMPROVED
    1 = FAIL_DEGRADED
    2 = 输入不合法（缺字段、样本不足、方差非正）

命令行：

    python -m evals.gate --baseline <run_id|path> --candidate <run_id|path> [--variance 0.0408]
    python -m evals.gate --compare last
    python -m evals.gate --compare <run_id|path>   # 指定基线，候选取最近一次 run

无效 run / 题集不一致默认拒绝对比（退出码 2）；确需继续时显式加
`--allow-invalid-baseline`、`--allow-invalid-candidate`、`--allow-partial`，
结果会带 quality_warning / compared_cases / skipped 标记并写进报告。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

#: docs/baseline.md §4 实测记录方差（faithfulness 两次最大差）。
DEFAULT_RECORDED_VARIANCE = 0.0408

#: 少于 2 道共同题时均值比较没有意义：给可读错误，不猜。
MIN_COMMON_CASES = 2

#: 单题耗时超过这个值按"超时"归因（与 agent_quality 默认单题超时 300s 对齐）。
DEFAULT_TIMEOUT_MS = 300_000

#: 失败归因的固定分类与展示顺序。
ATTRIBUTION_LABELS = ("工具未调用", "答案退化", "超时", "报错")

#: 判据原文（报告里原样引用）。
CRITERION = (
    "candidate - baseline 落在 ±有效阈值内为 PASS；"
    "有效阈值 = max(记录方差, 单题当量)；"
    "低于 -有效阈值为 FAIL_DEGRADED；高于 +有效阈值为 WARN_IMPROVED"
)

GateStatus = Literal["PASS", "WARN_IMPROVED", "FAIL_DEGRADED"]


class GateInputError(ValueError):
    """输入不合法：缺字段、样本不足、方差非正、不是 Run。CLI 映射为退出码 2。"""


@dataclass(frozen=True)
class Run:
    """一次评测 run：meta 是元信息，per_case 是逐题结果。"""

    run_id: str
    meta: Mapping[str, Any]
    per_case: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CaseEvidence:
    """单题对比证据。attribution 为空表示这题不算失败。"""

    case_id: str
    baseline_score: float | None
    candidate_score: float | None
    delta: float | None
    attribution: str
    detail: str


@dataclass(frozen=True)
class GateResult:
    """门禁结论 + 每项证据；报告只从这些字段拼结论，不手写。"""

    status: GateStatus
    passed: bool
    baseline_run_id: str
    candidate_run_id: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    threshold: float
    score_step: float
    n_cases: int
    recorded_variance: float
    one_case_weight: float
    threshold_used: float
    delta_in_cases: float
    flips: Mapping[str, int]
    sign_test_p: float
    common_cases: int
    compared_cases: int
    skipped: int
    missing_in_candidate: tuple[str, ...]
    missing_in_baseline: tuple[str, ...]
    partial_compare: bool
    code_version_changed: bool
    baseline_git: str
    candidate_git: str
    quality_warning: str | None
    evidence: tuple[CaseEvidence, ...]
    failures: tuple[CaseEvidence, ...]
    attribution_counts: Mapping[str, int]
    criterion: str
    reason: str

    @property
    def verdict(self) -> str:
        """向后兼容别名：旧调用方读 `verdict` 时等于三态 `status`。"""
        return self.status


def _as_float(value: Any, *, field: str, case_id: str) -> float:
    """把分数转成有限 float；bool / 字符串 / NaN 都给可读错误。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateInputError(f"case {case_id} 的 {field} 必须是数字，实际 {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise GateInputError(f"case {case_id} 的 {field} 不是有限数：{value!r}")
    return number


def _validate_run(run: Run, role: str) -> None:
    # 用属性判定而不是 isinstance：`python -m evals.gate` 会让模块以 __main__ 和
    # evals.gate 两名各加载一次，两个 Run 类不是同一个类，isinstance 会误杀。
    missing = [attr for attr in ("run_id", "meta", "per_case") if not hasattr(run, attr)]
    if missing:
        raise GateInputError(
            f"{role} 必须是 evals.gate.Run（缺 {', '.join(missing)}），实际 {type(run).__name__}"
        )
    if not run.per_case:
        raise GateInputError(f"{role} run {run.run_id} 没有 per_case 数据")


def _case_map(run: Run) -> dict[str, Mapping[str, Any]]:
    """按 case_id 建表；重复 case_id / 缺 score 字段都报可读错误。"""
    result: dict[str, Mapping[str, Any]] = {}
    for index, case in enumerate(run.per_case):
        if not isinstance(case, Mapping):
            raise GateInputError(f"run {run.run_id} 第 {index + 1} 条 per_case 不是对象")
        if "case_id" not in case or not str(case.get("case_id") or "").strip():
            raise GateInputError(f"run {run.run_id} 第 {index + 1} 条 per_case 缺 case_id")
        case_id = str(case["case_id"]).strip()
        if case_id in result:
            raise GateInputError(f"run {run.run_id} 的 case_id 重复：{case_id}")
        if "score" not in case:
            raise GateInputError(f"run {run.run_id} 的 case {case_id} 缺字段 score")
        result[case_id] = case
    return result


def _score(case: Mapping[str, Any] | None) -> float | None:
    """取有效分数；None 表示这题没有可比较的分数。"""
    if case is None:
        return None
    value = case.get("score")
    if value is None:
        return None
    return _as_float(value, field="score", case_id=str(case.get("case_id")))


def _tool_count(case: Mapping[str, Any]) -> int:
    tools = case.get("tool_calls")
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
        return len(tools)
    if isinstance(tools, bool):
        return 0
    if isinstance(tools, int):
        return tools
    return 0


def _is_timeout(case: Mapping[str, Any]) -> bool:
    error = str(case.get("error") or "").lower()
    if "timeout" in error or "timed out" in error or "超时" in error:
        return True
    latency = case.get("latency_ms")
    return (
        isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and latency > DEFAULT_TIMEOUT_MS
    )


def _truncate(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _attribute(
    case_id: str,
    base_case: Mapping[str, Any] | None,
    cand_case: Mapping[str, Any] | None,
    base_score: float | None,
    cand_score: float | None,
    delta: float | None,
) -> tuple[str, str]:
    """按固定分类给单题归因；不改变 PASS/FAIL，只进失败清单。"""
    if cand_case is None:
        return "报错", "候选 run 缺少该题"
    if base_case is None:
        return "报错", "基线 run 缺少该题"
    if cand_score is None:
        if _is_timeout(cand_case):
            return "超时", _truncate(str(cand_case.get("error") or "超过单题超时"))
        error = str(cand_case.get("error") or "没有分数（裁判未返回）")
        return "报错", _truncate(error)
    if _is_timeout(cand_case):
        latency = cand_case.get("latency_ms")
        detail = str(cand_case.get("error") or "")
        if not detail:
            detail = f"耗时 {latency}ms > {DEFAULT_TIMEOUT_MS}ms"
        return "超时", _truncate(detail)
    error = str(cand_case.get("error") or "")
    if error:
        return "报错", _truncate(error)
    if delta is not None and delta < 0:
        base_tools = _tool_count(base_case) if base_case is not None else 0
        cand_tools = _tool_count(cand_case)
        if cand_tools == 0 and base_tools > 0:
            return "工具未调用", f"候选 0 次工具调用，基线 {base_tools} 次"
        return "答案退化", f"分数 {base_score:.2f} -> {cand_score:.2f}"
    return "", ""


def _resolve_variance(baseline: Run, variance: float | None) -> float:
    """记录方差：显式参数 > baseline.meta > 0.0408；允许 0（有效阈值另有下限）。"""
    raw: Any = variance
    if raw is None:
        raw = baseline.meta.get("recorded_variance", DEFAULT_RECORDED_VARIANCE)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise GateInputError(f"记录方差必须是数字，实际 {raw!r}")
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise GateInputError(f"记录方差必须 >= 0，实际 {raw!r}")
    return value


def _infer_score_step(evidence: Sequence[CaseEvidence]) -> float:
    """从逐题分数差推最小步长；推不出时用默认 1.0。"""
    diffs = [
        abs(item.delta) for item in evidence if item.delta is not None and abs(item.delta) > 1e-12
    ]
    if not diffs:
        return 1.0
    step = min(diffs)
    return step if step > 0 else 1.0


def _sign_test_p(up: int, down: int) -> float:
    """双侧符号检验：H0 升/降等概率；p=2*sum_{i<=min(up,down)} C(n,i)/2^n。"""
    n = up + down
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(up, down) + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def _dataset_case_count(baseline: Run, fallback: int) -> int:
    """单题当量里的 n：优先 meta.dataset_size（全量题数），否则共同题数。"""
    raw = baseline.meta.get("dataset_size")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    return fallback


def _guard_quality(
    baseline: Run,
    candidate: Run,
    *,
    allow_invalid_baseline: bool,
    allow_invalid_candidate: bool,
) -> str | None:
    """校验两次 run 的有效性；默认拒绝，放行时返回 quality_warning。"""
    baseline_invalid = baseline.meta.get("valid_for_quality") is False
    candidate_invalid = candidate.meta.get("valid_for_quality") is False
    if baseline_invalid and not allow_invalid_baseline:
        raise GateInputError(
            "基线 run 是链路验证（valid_for_quality=false），不可作为质量基线；"
            "如确需对比加 --allow-invalid-baseline"
        )
    if candidate_invalid and not allow_invalid_candidate:
        raise GateInputError(
            "候选 run 是链路验证（valid_for_quality=false），不可作为质量候选；"
            "如确需对比加 --allow-invalid-candidate"
        )
    parts: list[str] = []
    if baseline_invalid:
        parts.append("基线 run 是链路验证（valid_for_quality=false），对比结论仅作链路参考")
    if candidate_invalid:
        parts.append("候选 run 是链路验证（valid_for_quality=false），对比结论仅作链路参考")
    return "；".join(parts) or None


def _guard_partial(
    missing_in_candidate: tuple[str, ...],
    missing_in_baseline: tuple[str, ...],
    *,
    allow_partial: bool,
) -> None:
    """题集不一致时默认拒绝；错误里列出两侧缺失题号。"""
    if allow_partial or not (missing_in_candidate or missing_in_baseline):
        return
    parts: list[str] = []
    if missing_in_candidate:
        parts.append(f"候选缺 {len(missing_in_candidate)} 题：{','.join(missing_in_candidate)}")
    if missing_in_baseline:
        parts.append(f"基线缺 {len(missing_in_baseline)} 题：{','.join(missing_in_baseline)}")
    raise GateInputError(f"题集不一致：{'；'.join(parts)}；如确需按共同题比较加 --allow-partial")


def _decide_status(
    delta: float, threshold: float, *, code_version_changed: bool
) -> tuple[GateStatus, str]:
    """三态 + 结论文案（零差值单独措辞；版本漂移时改为无法归因）。"""
    epsilon = 1e-12
    if delta < -threshold - epsilon:
        status: GateStatus = "FAIL_DEGRADED"
        conclusion = "退化超出记录噪声"
    elif delta > threshold + epsilon:
        status = "WARN_IMPROVED"
        conclusion = "提升超出记录噪声，需人工复核（可能是工具未触发/裁判漂移/答案变短）"
    else:
        status = "PASS"
        conclusion = "同口径无漂移"
    if abs(delta) < 1e-9:
        conclusion = "无差异（两均值完全相同）"
    if code_version_changed and status != "PASS":
        conclusion = "含代码版本差异，无法归因，需同版本复跑"
    return status, conclusion


def compare(
    baseline: Run,
    candidate: Run,
    variance: float | None = None,
    *,
    score_step: float | None = None,
    allow_invalid_baseline: bool = False,
    allow_invalid_candidate: bool = False,
    allow_partial: bool = False,
) -> GateResult:
    """比较两个 run 的均值差是否落在记录方差内；纯函数，无 IO。

    Args:
        baseline: 基线 run。
        candidate: 候选 run。
        variance: 记录方差；None 时读 `baseline.meta["recorded_variance"]`，
            再缺省用 0.0408；有效阈值还会取 max(方差, 单题当量)。
        score_step: 分数步长；None 时从 per_case 分数差推断（推不出用 1.0）。
        allow_invalid_baseline: 允许 `valid_for_quality=false` 的基线继续比较。
        allow_invalid_candidate: 允许无效候选继续比较。
        allow_partial: 题集不一致时允许只比共同题。

    Returns:
        GateResult：三态结论 + 两均值、阈值、差值、逐题归因、有效性/题集/版本标记。

    Raises:
        GateInputError: 输入不合法（无效 run 未放行 / 题集不一致未放行 /
            缺字段 / 共同题 < 2 / 方差为负 / 步长非正）。
    """
    _validate_run(baseline, "baseline")
    _validate_run(candidate, "candidate")
    quality_warning = _guard_quality(
        baseline,
        candidate,
        allow_invalid_baseline=allow_invalid_baseline,
        allow_invalid_candidate=allow_invalid_candidate,
    )

    base_cases = _case_map(baseline)
    cand_cases = _case_map(candidate)
    missing_in_candidate = tuple(sorted(set(base_cases) - set(cand_cases)))
    missing_in_baseline = tuple(sorted(set(cand_cases) - set(base_cases)))
    _guard_partial(
        missing_in_candidate,
        missing_in_baseline,
        allow_partial=allow_partial,
    )

    all_ids = sorted(set(base_cases) | set(cand_cases))
    evidence: list[CaseEvidence] = []
    common_scores: list[tuple[float, float]] = []

    for case_id in all_ids:
        base_score = _score(base_cases.get(case_id))
        cand_score = _score(cand_cases.get(case_id))
        delta = (
            round(cand_score - base_score, 4)
            if base_score is not None and cand_score is not None
            else None
        )
        if base_score is not None and cand_score is not None:
            common_scores.append((base_score, cand_score))
        attribution, detail = _attribute(
            case_id,
            base_cases.get(case_id),
            cand_cases.get(case_id),
            base_score,
            cand_score,
            delta,
        )
        evidence.append(
            CaseEvidence(
                case_id=case_id,
                baseline_score=base_score,
                candidate_score=cand_score,
                delta=delta,
                attribution=attribution,
                detail=detail,
            )
        )

    if len(common_scores) < MIN_COMMON_CASES:
        raise GateInputError(
            f"共同有效题只有 {len(common_scores)} 道，少于 {MIN_COMMON_CASES} 道，"
            "均值比较无意义；请用 --limit 2 以上复跑或换同题集 run"
        )

    compared_cases = len(common_scores)
    skipped = len(all_ids) - compared_cases
    partial_compare = skipped > 0
    baseline_git = str(baseline.meta.get("git_commit") or "")
    candidate_git = str(candidate.meta.get("git_commit") or "")
    code_version_changed = bool(baseline_git and candidate_git and baseline_git != candidate_git)

    up = sum(1 for base, cand in common_scores if cand > base + 1e-12)
    down = sum(1 for base, cand in common_scores if cand < base - 1e-12)
    unchanged = len(common_scores) - up - down
    flips = {"up": up, "down": down, "unchanged": unchanged}

    baseline_mean = sum(item[0] for item in common_scores) / len(common_scores)
    candidate_mean = sum(item[1] for item in common_scores) / len(common_scores)
    delta = candidate_mean - baseline_mean
    recorded_variance = _resolve_variance(baseline, variance)
    if score_step is None:
        step = _infer_score_step(evidence)
    else:
        if isinstance(score_step, bool) or not isinstance(score_step, (int, float)):
            raise GateInputError(f"score_step 必须是数字，实际 {score_step!r}")
        step = float(score_step)
    if not math.isfinite(step) or step <= 0:
        raise GateInputError(f"score_step 必须为正数，实际 {step!r}")
    n_cases = _dataset_case_count(baseline, compared_cases)
    one_case_weight = step / n_cases
    threshold_used = max(recorded_variance, one_case_weight)
    delta_in_cases = delta / one_case_weight
    sign_test_p = _sign_test_p(up, down)
    status, conclusion = _decide_status(
        delta, threshold_used, code_version_changed=code_version_changed
    )
    passed = status != "FAIL_DEGRADED"

    failures = tuple(item for item in evidence if item.attribution)
    counts = dict.fromkeys(ATTRIBUTION_LABELS, 0)
    for item in failures:
        counts[item.attribution] = counts.get(item.attribution, 0) + 1

    comparison = (
        f"delta={delta:+.4f}（candidate-baseline；净值 {abs(delta_in_cases):.1f} 题当量）；"
        f"有效阈值 {threshold_used:.4f}（方差 {recorded_variance:.4f} 与单题当量 "
        f"{one_case_weight:.4f} 取大）"
    )
    failure_note = ""
    if failures:
        parts = [f"{label} {counts[label]}" for label in ATTRIBUTION_LABELS if counts.get(label)]
        failure_note = f"；失败归因：{'、'.join(parts)}"
    partial_note = ""
    if partial_compare:
        partial_note = (
            f"；仅比较共同题（compared_cases={compared_cases} / skipped={skipped}），"
            "结论不可跨版本比较"
        )
    significance = "不显著" if sign_test_p > 0.05 else "显著"
    flip_note = (
        f"；逐题 {up} 升 / {down} 降 / {unchanged} 不变，"
        f"符号检验 p={sign_test_p:.2f}（{significance}）"
    )
    reason = f"{status}：{comparison}，{conclusion}{partial_note}{failure_note}{flip_note}"

    return GateResult(
        status=status,
        passed=passed,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_mean=round(baseline_mean, 4),
        candidate_mean=round(candidate_mean, 4),
        delta=round(delta, 4),
        threshold=threshold_used,
        score_step=step,
        n_cases=n_cases,
        recorded_variance=recorded_variance,
        one_case_weight=one_case_weight,
        threshold_used=threshold_used,
        delta_in_cases=round(delta_in_cases, 4),
        flips=flips,
        sign_test_p=sign_test_p,
        common_cases=compared_cases,
        compared_cases=compared_cases,
        skipped=skipped,
        missing_in_candidate=missing_in_candidate,
        missing_in_baseline=missing_in_baseline,
        partial_compare=partial_compare,
        code_version_changed=code_version_changed,
        baseline_git=baseline_git,
        candidate_git=candidate_git,
        quality_warning=quality_warning,
        evidence=tuple(evidence),
        failures=failures,
        attribution_counts=counts,
        criterion=CRITERION,
        reason=reason,
    )


def _print_result(result: GateResult, *, as_json: bool) -> None:
    if as_json:
        payload = {
            "status": result.status,
            "verdict": result.verdict,
            "passed": result.passed,
            "baseline_run_id": result.baseline_run_id,
            "candidate_run_id": result.candidate_run_id,
            "baseline_mean": result.baseline_mean,
            "candidate_mean": result.candidate_mean,
            "delta": result.delta,
            "threshold": result.threshold,
            "score_step": result.score_step,
            "n_cases": result.n_cases,
            "recorded_variance": result.recorded_variance,
            "one_case_weight": result.one_case_weight,
            "threshold_used": result.threshold_used,
            "delta_in_cases": result.delta_in_cases,
            "flips": result.flips,
            "sign_test_p": result.sign_test_p,
            "common_cases": result.common_cases,
            "compared_cases": result.compared_cases,
            "skipped": result.skipped,
            "missing_in_candidate": list(result.missing_in_candidate),
            "missing_in_baseline": list(result.missing_in_baseline),
            "partial_compare": result.partial_compare,
            "code_version_changed": result.code_version_changed,
            "baseline_git": result.baseline_git,
            "candidate_git": result.candidate_git,
            "quality_warning": result.quality_warning,
            "criterion": result.criterion,
            "reason": result.reason,
            "attribution_counts": result.attribution_counts,
            "failures": [
                {
                    "case_id": item.case_id,
                    "baseline_score": item.baseline_score,
                    "candidate_score": item.candidate_score,
                    "delta": item.delta,
                    "attribution": item.attribution,
                    "detail": item.detail,
                }
                for item in result.failures
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"baseline : {result.baseline_run_id}  mean={result.baseline_mean:.4f}")
    print(f"candidate: {result.candidate_run_id}  mean={result.candidate_mean:.4f}")
    print(f"共同题   : {result.common_cases}")
    print(f"状态     : {result.status}")
    if result.quality_warning:
        print(f"质量警告 : {result.quality_warning}")
    if result.partial_compare:
        print(
            f"题集     : 仅共同题 compared_cases={result.compared_cases} skipped={result.skipped}"
        )
    if result.missing_in_candidate:
        print(f"缺题     : 候选缺 {','.join(result.missing_in_candidate)}")
    if result.missing_in_baseline:
        print(f"缺题     : 基线缺 {','.join(result.missing_in_baseline)}")
    if result.code_version_changed:
        print(f"代码版本 : {result.baseline_git} vs {result.candidate_git}（含版本差异）")
    print(f"判据     : {result.criterion}")
    print(
        f"阈值     : 有效 {result.threshold_used:.4f}"
        f"（方差 {result.recorded_variance:.4f} 与单题当量 {result.one_case_weight:.4f} 取大）"
    )
    print(f"证据     : delta={result.delta:+.4f}（{result.delta_in_cases:+.1f} 题当量）")
    print(
        f"逐题抖动 : 升 {result.flips.get('up', 0)} / 降 {result.flips.get('down', 0)} / "
        f"不变 {result.flips.get('unchanged', 0)}，符号检验 p={result.sign_test_p:.2f}"
    )
    for item in result.failures:
        base = "—" if item.baseline_score is None else f"{item.baseline_score:.2f}"
        cand = "—" if item.candidate_score is None else f"{item.candidate_score:.2f}"
        print(f"  [{item.attribution}] {item.case_id}: {base} -> {cand} | {item.detail}")
    print(result.reason)


def _mean_score(run: Run) -> float | None:
    scores = [
        float(case["score"])
        for case in run.per_case
        if isinstance(case.get("score"), (int, float)) and not isinstance(case.get("score"), bool)
    ]
    return sum(scores) / len(scores) if scores else None


def _auto_variance(baseline: Run, store_module: Any) -> float:
    """同 config / 同 commit / 同数据集的重复 run：用均值差绝对值当方差。"""
    config = baseline.meta.get("config_id")
    git = baseline.meta.get("git_commit")
    sha = baseline.meta.get("dataset_sha256")
    candidates = [
        run
        for run in store_module.list_runs()
        if run.meta.get("config_id") == config
        and run.meta.get("git_commit") == git
        and run.meta.get("dataset_sha256") == sha
        and run.meta.get("valid_for_quality") is not False
        and run.meta.get("partial") is not True
    ]
    if len(candidates) < 2:
        print(
            f"未找到可用的重复 run，回退到 {DEFAULT_RECORDED_VARIANCE}",
            file=sys.stderr,
        )
        return DEFAULT_RECORDED_VARIANCE
    first, second = candidates[-2], candidates[-1]
    first_mean = _mean_score(first)
    second_mean = _mean_score(second)
    if first_mean is None or second_mean is None:
        print(
            f"未找到可用的重复 run，回退到 {DEFAULT_RECORDED_VARIANCE}",
            file=sys.stderr,
        )
        return DEFAULT_RECORDED_VARIANCE
    value = abs(second_mean - first_mean)
    print(
        f"variance auto: {value:.4f}（{first.run_id} vs {second.run_id}）",
        file=sys.stderr,
    )
    return value


def _parse_variance(raw: str | None, baseline: Run, store_module: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text == "auto":
        return _auto_variance(baseline, store_module)
    try:
        return float(text)
    except ValueError:
        raise GateInputError(f"--variance 只接受数字或 auto，实际 {raw!r}") from None


def main(argv: list[str] | None = None) -> int:
    """CLI：加载 run、比较、打印；退出码 0/1/2。"""
    parser = argparse.ArgumentParser(
        description="评测门禁：三态判定（PASS / WARN_IMPROVED / FAIL_DEGRADED）"
    )
    parser.add_argument("--baseline", default="", help="基线 run_id 或路径（meta.json / run 目录）")
    parser.add_argument("--candidate", default="", help="候选 run_id 或路径")
    parser.add_argument(
        "--compare",
        nargs="?",
        const="last",
        default=None,
        help="快捷比较：last=最近两次；也可给基线 run_id/路径（候选取最近一次）",
    )
    parser.add_argument(
        "--variance",
        default=None,
        help=(
            f"记录方差：数字、auto 或留空（读 baseline.meta / {DEFAULT_RECORDED_VARIANCE}）；"
            "auto=找同 config/同 commit/同数据集的两次 run，用均值差绝对值"
        ),
    )
    parser.add_argument(
        "--allow-invalid-baseline",
        action="store_true",
        help="允许 valid_for_quality=false 的基线继续比较（报告带 quality_warning）",
    )
    parser.add_argument(
        "--allow-invalid-candidate",
        action="store_true",
        help="允许 valid_for_quality=false 的候选继续比较（报告带 quality_warning）",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="题集不一致时允许只比较共同题（结果标记 partial_compare/skipped）",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="输出 JSON 供机器读")
    args = parser.parse_args(argv)

    from evals import store  # 延迟导入：compare() 本身保持纯函数、零 IO

    try:
        if args.compare is not None:
            if args.baseline or args.candidate:
                raise GateInputError("--compare 不能与 --baseline/--candidate 同时使用")
            runs = store.list_runs()
            if len(runs) < 2:
                raise GateInputError(f"--compare last 需要至少 2 次 run，当前只有 {len(runs)} 次")
            candidate = runs[-1]
            if args.compare == "last":
                # 只与同一份数据集的历史 run 比，避免跨题集/跨口径误判
                sha = candidate.meta.get("dataset_sha256")
                same_dataset = [
                    run for run in runs[:-1] if not sha or run.meta.get("dataset_sha256") == sha
                ]
                if not same_dataset:
                    raise GateInputError("最近一次 run 之前没有同数据集的历史 run")
                baseline = same_dataset[-1]
            else:
                baseline = store.resolve_run_ref(args.compare)
        else:
            if not args.baseline or not args.candidate:
                raise GateInputError("需要同时给 --baseline 和 --candidate，或用 --compare last")
            baseline = store.resolve_run_ref(args.baseline)
            candidate = store.resolve_run_ref(args.candidate)

        variance_value = _parse_variance(args.variance, baseline, store)
        result = compare(
            baseline,
            candidate,
            variance=variance_value,
            allow_invalid_baseline=args.allow_invalid_baseline,
            allow_invalid_candidate=args.allow_invalid_candidate,
            allow_partial=args.allow_partial,
        )
    except (GateInputError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"输入不合法：{exc}", file=sys.stderr)
        return 2

    _print_result(result, as_json=args.as_json)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
