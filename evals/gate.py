"""评测门禁判定 —— 纯函数，零 IO 副作用（CI 可跑）。

判据（沿用 docs/baseline.md §4 的既有口径，不另发明）：

    同口径两次均值差 <= 记录方差（默认 0.0408）
    单题差值只做归因，不单独判失败

`compare()` 只吃两个 run 的 meta + per_case，返回 `GateResult`；不读写文件、
不调模型。CLI 才做 IO（加载 run、打印结论），退出码：

    0 = PASS
    1 = FAIL
    2 = 输入不合法（缺字段、样本不足、方差非正）

命令行：

    python -m evals.gate --baseline <run_id|path> --candidate <run_id|path> [--variance 0.0408]
    python -m evals.gate --compare last
    python -m evals.gate --compare <run_id|path>   # 指定基线，候选取最近一次 run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: docs/baseline.md §4 实测记录方差（faithfulness 两次最大差）。
DEFAULT_RECORDED_VARIANCE = 0.0408

#: 少于 2 道共同题时均值比较没有意义：给可读错误，不猜。
MIN_COMMON_CASES = 2

#: 单题耗时超过这个值按"超时"归因（与 agent_quality 默认单题超时 300s 对齐）。
DEFAULT_TIMEOUT_MS = 300_000

#: 失败归因的固定分类与展示顺序。
ATTRIBUTION_LABELS = ("工具未调用", "答案退化", "超时", "报错")

#: 判据原文（报告里原样引用）。
CRITERION = "同口径两次均值差 <= 记录方差（默认 0.0408）；单题差值只做归因，不单独判失败"


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

    verdict: str
    passed: bool
    baseline_run_id: str
    candidate_run_id: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    threshold: float
    common_cases: int
    evidence: tuple[CaseEvidence, ...]
    failures: tuple[CaseEvidence, ...]
    attribution_counts: Mapping[str, int]
    criterion: str
    reason: str


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


def _resolve_threshold(baseline: Run, variance: float | None) -> float:
    """判据阈值：显式参数 > baseline.meta > 0.0408；必须为正。"""
    raw: Any = variance
    if raw is None:
        raw = baseline.meta.get("recorded_variance", DEFAULT_RECORDED_VARIANCE)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise GateInputError(f"记录方差必须是数字，实际 {raw!r}")
    threshold = max(float(raw), 0.0)
    if not math.isfinite(threshold) or threshold <= 0:
        raise GateInputError(
            f"记录方差必须 > 0，实际 {raw!r}；方差为 0 时任何噪声都会被判失败，判据无判别力"
        )
    return threshold


def compare(
    baseline: Run,
    candidate: Run,
    variance: float | None = None,
) -> GateResult:
    """比较两个 run 的均值差是否落在记录方差内；纯函数，无 IO。

    Args:
        baseline: 基线 run。
        candidate: 候选 run。
        variance: 记录方差；None 时读 `baseline.meta["recorded_variance"]`，
            再缺省用 0.0408。

    Returns:
        GateResult：PASS / FAIL + 两均值、阈值、差值、逐题归因。

    Raises:
        GateInputError: 输入不合法（缺字段 / 共同题 < 2 / 方差非正）。
    """
    _validate_run(baseline, "baseline")
    _validate_run(candidate, "candidate")
    base_cases = _case_map(baseline)
    cand_cases = _case_map(candidate)

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

    baseline_mean = sum(item[0] for item in common_scores) / len(common_scores)
    candidate_mean = sum(item[1] for item in common_scores) / len(common_scores)
    delta = candidate_mean - baseline_mean
    threshold = _resolve_threshold(baseline, variance)
    passed = abs(delta) <= threshold + 1e-12

    failures = tuple(item for item in evidence if item.attribution)
    counts = dict.fromkeys(ATTRIBUTION_LABELS, 0)
    for item in failures:
        counts[item.attribution] = counts.get(item.attribution, 0) + 1

    verdict = "PASS" if passed else "FAIL"
    comparison = (
        f"|{candidate_mean:.4f} - {baseline_mean:.4f}| = {abs(delta):.4f} "
        f"{'<=' if passed else '>'} 记录方差 {threshold:.4f}"
    )
    failure_note = ""
    if failures:
        parts = [f"{label} {counts[label]}" for label in ATTRIBUTION_LABELS if counts.get(label)]
        failure_note = f"；失败归因：{'、'.join(parts)}"
    reason = f"{verdict}：{comparison}{failure_note}"

    return GateResult(
        verdict=verdict,
        passed=passed,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_mean=round(baseline_mean, 4),
        candidate_mean=round(candidate_mean, 4),
        delta=round(delta, 4),
        threshold=threshold,
        common_cases=len(common_scores),
        evidence=tuple(evidence),
        failures=failures,
        attribution_counts=counts,
        criterion=CRITERION,
        reason=reason,
    )


def _print_result(result: GateResult, *, as_json: bool) -> None:
    if as_json:
        payload = {
            "verdict": result.verdict,
            "passed": result.passed,
            "baseline_run_id": result.baseline_run_id,
            "candidate_run_id": result.candidate_run_id,
            "baseline_mean": result.baseline_mean,
            "candidate_mean": result.candidate_mean,
            "delta": result.delta,
            "threshold": result.threshold,
            "common_cases": result.common_cases,
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
    print(f"判据     : {result.criterion}")
    print(f"证据     : delta={result.delta:+.4f} 阈值={result.threshold:.4f}")
    for item in result.failures:
        base = "—" if item.baseline_score is None else f"{item.baseline_score:.0f}"
        cand = "—" if item.candidate_score is None else f"{item.candidate_score:.0f}"
        print(f"  [{item.attribution}] {item.case_id}: {base} -> {cand} | {item.detail}")
    print(result.reason)


def main(argv: list[str] | None = None) -> int:
    """CLI：加载 run、比较、打印；退出码 0/1/2。"""
    parser = argparse.ArgumentParser(description="评测门禁：两次 run 的均值差是否 <= 记录方差")
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
        type=float,
        default=None,
        help=f"记录方差，默认读 baseline.meta 或 {DEFAULT_RECORDED_VARIANCE}",
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

        result = compare(baseline, candidate, variance=args.variance)
    except (GateInputError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"输入不合法：{exc}", file=sys.stderr)
        return 2

    _print_result(result, as_json=args.as_json)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
