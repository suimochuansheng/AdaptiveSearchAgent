#!/usr/bin/env python
"""离线冒烟：不跑模型，验证三态门禁 + 报告 + 探活拒绝 + 边界错误。

覆盖：
- 三态 fixture：降质 -1.5 -> FAIL_DEGRADED、持平 -> PASS、+0.03 -> PASS；
- 三态补丁：+0.5 -> WARN_IMPROVED（退出码 0）、-0.5 -> FAIL_DEGRADED（退出码 1）；
- report.md 生成，含均值、三态 status 与 gate 生成的结论句；
- `valid_for_quality=false` 时报告顶部有链路验证警告；
- 探活失败（RAG 指向不存在端口）-> `evals.run` 退出码 2、提示“环境未就绪：rag”；
- 3 个边界：方差为 0、只有 1 题、per_case 缺字段 -> 可读错误而不是崩溃。

用法（任意 cwd 均可）：

    python scripts/smoke_eval_gate.py
    python scripts/smoke_eval_gate.py --results-dir evals/results   # CI 产出 artifact

默认把 fixture 写到临时目录，不污染 `evals/results/`。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evals import gate, report, store  # noqa: E402
from evals.gate import GateInputError, Run  # noqa: E402

_STAMP = time.strftime("%Y%m%d-%H%M%S")
_CASE_IDS = ("q001", "q002", "q003")
_TWENTY_IDS = tuple(f"q{i:03d}" for i in range(1, 21))

LINK_ONLY_WARNING = "⚠️ 本次为链路验证 run，环境未就绪，分数不可用于质量结论"


class SmokeFailure(AssertionError):
    """冒烟断言失败；由 run_check() 收集成 [FAIL]。"""


def _case(
    case_id: str,
    score: float | None,
    *,
    tools: list[str] | None = None,
    error: str | None = None,
    latency_ms: int = 1200,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "question": f"fixture question {case_id}",
        "answer_digest": hashlib.sha256(f"{case_id}:{score}".encode()).hexdigest()[:16],
        "score": score,
        "latency_ms": latency_ms,
        "tokens": 100,
        "tool_calls": ["search_knowledge"] if tools is None else tools,
        "error": error,
    }


def _make_run(
    run_id: str,
    scores: Sequence[float | None],
    *,
    drop_tools: bool = False,
    valid_for_quality: bool = True,
    case_ids: tuple[str, ...] = _CASE_IDS,
    git_commit: str = "fixture",
    dataset_sha: str = "fixture-sha",
    partial: bool = False,
    dataset_size: int | None = None,
) -> Run:
    per_case = []
    for index, score in enumerate(scores):
        case_id = case_ids[index]
        if score is None:
            per_case.append(
                _case(
                    case_id,
                    None,
                    tools=[],
                    error="TimeoutError: read timeout",
                    latency_ms=gate.DEFAULT_TIMEOUT_MS + 1,
                )
            )
        else:
            per_case.append(_case(case_id, score, tools=[] if drop_tools else None))
    meta = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_id": "FIXTURE",
        "config_name": "fixture",
        "model": "fixture-model",
        "judge_model": "fixture-judge",
        "n_cases": len(per_case),
        "dataset_size": dataset_size if dataset_size is not None else len(per_case),
        "git_commit": git_commit,
        "flags": {"ENABLE_TOOL_POLICY": "0"},
        "schema_sha256": "0" * 64,
        "dataset": "fixture",
        "dataset_sha256": dataset_sha,
        "recorded_variance": gate.DEFAULT_RECORDED_VARIANCE,
        "environment": {
            "rag_ok": True,
            "model_ok": True,
            "pg_ok": True,
            "redis_ok": True,
            "checked_at": "fixture",
        },
        "valid_for_quality": valid_for_quality,
        "partial": partial,
        "publishable": not partial,
    }
    store.save_run(run_id, meta, per_case)
    return store.load_run(run_id)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _gate_exit(baseline: Run, candidate: Run, *extra: str) -> tuple[int, str]:
    """走真实 CLI 入口，断言退出码；stdout 一并返回供错误信息使用。"""
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gate.main(
            [
                "--baseline",
                str(store.RESULTS_DIR / baseline.run_id),
                "--candidate",
                str(store.RESULTS_DIR / candidate.run_id),
                *extra,
            ]
        )
    return code, out.getvalue() + err.getvalue()


def check_degraded(base: Run) -> str:
    candidate = _make_run(f"{_STAMP}-fixture-degraded", [7, 7, None], drop_tools=True)
    result = gate.compare(base, candidate)
    _assert(
        result.status == "FAIL_DEGRADED",
        f"降质应为 FAIL_DEGRADED，实际 {result.status}",
    )
    _assert(result.passed is False, "FAIL_DEGRADED 的 passed 必须为 False")
    _assert(result.attribution_counts["工具未调用"] == 2, "应归因 2 题工具未调用")
    _assert(result.attribution_counts["超时"] == 1, "应归因 1 题超时")

    path = report.write_markdown(candidate.run_id, result)
    text = path.read_text(encoding="utf-8")
    _assert("FAIL_DEGRADED" in text, "report.md 缺 FAIL_DEGRADED 结论")
    _assert(f"{result.baseline_mean:.4f}" in text, "report.md 缺基线均值")
    _assert(result.reason in text, "report.md 缺 gate 生成的结论句")
    return f"FAIL_DEGRADED，均值 {result.baseline_mean:.4f}->{result.candidate_mean:.4f}"


def check_flat(base: Run) -> str:
    candidate = _make_run(f"{_STAMP}-fixture-flat", [9, 8, 9])
    result = gate.compare(base, candidate)
    _assert(result.status == "PASS", f"持平应为 PASS，实际 {result.status}")
    _assert(result.passed is True, "PASS 的 passed 必须为 True")
    _assert(result.delta == 0, f"持平 delta 应为 0，实际 {result.delta}")
    path = report.write_markdown(candidate.run_id, result)
    _assert(result.reason in path.read_text(encoding="utf-8"), "持平报告缺结论句")
    return f"PASS，delta={result.delta:+.4f}"


def check_improved_in_band(base: Run) -> str:
    candidate = _make_run(f"{_STAMP}-fixture-improved", [9.03, 8.03, 9.03])
    result = gate.compare(base, candidate)
    _assert(result.status == "PASS", f"小幅提升应为 PASS，实际 {result.status}")
    _assert(result.delta > 0, f"提升 delta 应 > 0，实际 {result.delta}")
    return f"PASS，delta={result.delta:+.4f}（在记录方差内）"


def check_warn_improved(base: Run) -> str:
    candidate = _make_run(f"{_STAMP}-fixture-warn", [9.5, 8.5, 9.5])
    result = gate.compare(base, candidate)
    _assert(result.status == "WARN_IMPROVED", f"+0.5 应为 WARN_IMPROVED，实际 {result.status}")
    _assert(result.passed is True, "WARN_IMPROVED 的 passed 必须为 True")
    _assert("人工复核" in result.reason, f"WARN 原因缺人工复核提示：{result.reason}")
    path = report.write_markdown(candidate.run_id, result)
    _assert("WARN_IMPROVED" in path.read_text(encoding="utf-8"), "报告缺 WARN_IMPROVED")
    code, _ = _gate_exit(base, candidate)
    _assert(code == 0, f"WARN_IMPROVED 的 CLI 退出码应为 0，实际 {code}")
    return f"WARN_IMPROVED，delta={result.delta:+.4f}，CLI=0"


def check_fail_degraded(base: Run) -> str:
    candidate = _make_run(f"{_STAMP}-fixture-fail", [8.5, 7.5, 8.5])
    result = gate.compare(base, candidate)
    _assert(result.status == "FAIL_DEGRADED", f"-0.5 应为 FAIL_DEGRADED，实际 {result.status}")
    _assert(result.passed is False, "FAIL_DEGRADED 的 passed 必须为 False")
    code, _ = _gate_exit(base, candidate)
    _assert(code == 1, f"FAIL_DEGRADED 的 CLI 退出码应为 1，实际 {code}")
    return f"FAIL_DEGRADED，delta={result.delta:+.4f}，CLI=1"


def check_invalid_report(base: Run) -> str:
    candidate = _make_run(f"{_STAMP}-fixture-invalid", [9, 8, 9], valid_for_quality=False)
    result = gate.compare(base, candidate, allow_invalid_candidate=True)
    text = report.write_markdown(candidate.run_id, result).read_text(encoding="utf-8")
    _assert(LINK_ONLY_WARNING in text, "valid_for_quality=false 的报告缺顶部警告")
    _assert("不可用于质量结论" in text, "报告缺不可用质量结论标注")
    return "链路验证警告已写入 report.md 顶部"


def check_zero_variance(base: Run) -> str:
    result = gate.compare(base, base, variance=0)
    _assert(
        result.recorded_variance == 0, f"recorded_variance 应为 0，实际 {result.recorded_variance}"
    )
    _assert(
        result.threshold_used == result.one_case_weight,
        f"方差 0 时有效阈值应取单题当量，实际 {result.threshold_used}",
    )
    _assert(result.status == "PASS", f"零差值应 PASS，实际 {result.status}")
    return f"方差 0 -> 有效阈值 {result.threshold_used:.4f}（单题当量）"


def check_single_case(base: Run) -> str:
    single = Run(
        run_id="fixture-single",
        meta={"recorded_variance": gate.DEFAULT_RECORDED_VARIANCE},
        per_case=tuple(_case(case_id, 9 if case_id == "q001" else None) for case_id in _CASE_IDS),
    )
    try:
        gate.compare(base, single)
    except GateInputError as exc:
        text = str(exc)
        _assert("2 道" in text or "共同有效题" in text, f"错误信息不可读：{text}")
        return text
    raise SmokeFailure("只有 1 题时应给可读错误，实际通过了比较")


def check_missing_field(base: Run) -> str:
    broken = Run(
        run_id="fixture-missing-score",
        meta={"recorded_variance": gate.DEFAULT_RECORDED_VARIANCE},
        per_case=({"case_id": "q001", "question": "缺 score"},),
    )
    try:
        gate.compare(base, broken)
    except GateInputError as exc:
        text = str(exc)
        _assert("score" in text, f"错误信息应指出缺 score：{text}")
        return text
    except (KeyError, TypeError, AttributeError) as exc:  # pragma: no cover - 只用于定位
        raise SmokeFailure(
            f"缺字段应给 GateInputError，实际崩溃：{type(exc).__name__}: {exc}"
        ) from exc
    raise SmokeFailure("缺字段时应给可读错误，实际通过了比较")


def check_net_one_case_down() -> str:
    base = _make_run(f"{_STAMP}-fixture-20-base", [9.0] * 20, case_ids=_TWENTY_IDS)
    candidate = _make_run(f"{_STAMP}-fixture-20-down1", [9.0] * 19 + [8.0], case_ids=_TWENTY_IDS)
    result = gate.compare(base, candidate)
    _assert(result.status == "PASS", f"净降 1 题应 PASS，实际 {result.status}")
    _assert(
        abs(result.threshold_used - 0.05) < 1e-9,
        f"有效阈值应为 0.05，实际 {result.threshold_used}",
    )
    _assert("1.0 题当量" in result.reason, f"reason 缺题当量：{result.reason}")
    _assert("符号检验" in result.reason, f"reason 缺符号检验：{result.reason}")
    return f"PASS，delta={result.delta:+.4f}，阈值={result.threshold_used:.4f}"


def check_net_three_case_down() -> str:
    base = _make_run(f"{_STAMP}-fixture-20-base3", [9.0] * 20, case_ids=_TWENTY_IDS)
    candidate = _make_run(
        f"{_STAMP}-fixture-20-down3", [9.0] * 17 + [8.0] * 3, case_ids=_TWENTY_IDS
    )
    result = gate.compare(base, candidate)
    _assert(result.status == "FAIL_DEGRADED", f"净降 3 题应 FAIL_DEGRADED，实际 {result.status}")
    _assert(result.delta == -0.15, f"delta 应为 -0.15，实际 {result.delta}")
    return f"FAIL_DEGRADED，delta={result.delta:+.4f}，阈值={result.threshold_used:.4f}"


def check_flip_evidence() -> str:
    base = _make_run(f"{_STAMP}-fixture-flip-base", [9.0] * 20, case_ids=_TWENTY_IDS)
    scores = [10.0, 10.0, 8.0, 8.0] + [9.0] * 16
    candidate = _make_run(f"{_STAMP}-fixture-flip", scores, case_ids=_TWENTY_IDS)
    result = gate.compare(base, candidate)
    _assert(result.status == "PASS", f"净 0 应 PASS，实际 {result.status}")
    _assert(result.flips == {"up": 2, "down": 2, "unchanged": 16}, f"flips 错误：{result.flips}")
    _assert(result.sign_test_p == 1.0, f"符号检验 p 应为 1.00，实际 {result.sign_test_p}")
    _assert("2 升 / 2 降 / 16 不变" in result.reason, f"reason 缺抖动：{result.reason}")
    _assert("符号检验 p=1.00" in result.reason, f"reason 缺 p 值：{result.reason}")
    text = report.write_markdown(candidate.run_id, result).read_text(encoding="utf-8")
    _assert("单题当量：0.0500" in text, "报告缺单题当量行")
    _assert("有效阈值：0.0500" in text, "报告缺有效阈值行")
    _assert("逐题抖动：升 2 / 降 2 / 不变 16" in text, "报告缺逐题抖动行")
    _assert("符号检验 p=1.00" in text, "报告缺符号检验 p 值")
    return "PASS，flips=2/2/16，p=1.00，报告已呈现"


def check_pass_attribution_wording() -> str:
    base = _make_run(f"{_STAMP}-fixture-note-base", [9.0] * 20, case_ids=_TWENTY_IDS)
    candidate = _make_run(f"{_STAMP}-fixture-note-down", [9.0] * 19 + [8.0], case_ids=_TWENTY_IDS)
    result = gate.compare(base, candidate)
    _assert(result.status == "PASS", f"净降 1 题应 PASS，实际 {result.status}")
    _assert("变化归因" in result.reason, f"PASS 应写变化归因：{result.reason}")
    _assert("失败归因" not in result.reason, f"PASS 不应写失败归因：{result.reason}")
    text = report.write_markdown(candidate.run_id, result).read_text(encoding="utf-8")
    _assert("变化归因" in text, "报告表头应写变化归因")
    _assert("失败归因" not in text, "PASS 报告不应出现失败归因")
    return "PASS reason/报告表头均为变化归因"


def check_inband_wording() -> str:
    base = _make_run(f"{_STAMP}-fixture-inband-base", [9.0] * 20, case_ids=_TWENTY_IDS)
    candidate = _make_run(f"{_STAMP}-fixture-inband-up", [9.0] * 19 + [10.0], case_ids=_TWENTY_IDS)
    result = gate.compare(base, candidate)
    _assert(result.status == "PASS", f"净升 1 题应 PASS，实际 {result.status}")
    _assert("差异在单题当量内" in result.reason, f"缺当量内措辞：{result.reason}")
    return f"PASS，reason 含差异在单题当量内（{result.delta_in_cases:+.1f} 题）"


def check_small_sample_warning() -> str:
    case_ids = ("q001", "q002")
    base = _make_run(f"{_STAMP}-fixture-n2-base", [9.0, 8.0], case_ids=case_ids, dataset_size=2)
    candidate = _make_run(
        f"{_STAMP}-fixture-n2-cand", [9.0, 8.0], case_ids=case_ids, dataset_size=2
    )
    result = gate.compare(base, candidate)
    _assert(result.sample_warning is not None, "n=2 应有 sample_warning")
    _assert("样本过小" in result.reason, f"reason 缺样本过小：{result.reason}")
    text = report.write_markdown(candidate.run_id, result).read_text(encoding="utf-8")
    _assert("样本过小" in text, "报告顶部缺样本过小警告")
    return f"sample_warning={result.sample_warning}"


def check_mixed_dataset() -> str:
    base = _make_run(f"{_STAMP}-fixture-ds-base", [9.0, 8.0, 9.0])
    candidate = _make_run(f"{_STAMP}-fixture-ds-other", [9.0, 8.0, 9.0], dataset_sha="other-sha")
    try:
        gate.compare(base, candidate)
    except GateInputError as exc:
        text = str(exc)
        _assert("数据集不同" in text, f"错误应说明数据集不同：{text}")
    else:
        raise SmokeFailure("数据集不同默认应拒绝，实际通过了比较")
    code, _ = _gate_exit(base, candidate)
    _assert(code == 2, f"数据集不同默认退出码应为 2，实际 {code}")
    result = gate.compare(base, candidate, allow_mixed_dataset=True)
    _assert(result.dataset_mismatch is True, "放行后 dataset_mismatch 应为 True")
    text = report.write_markdown(candidate.run_id, result).read_text(encoding="utf-8")
    _assert("数据集不同" in text, "报告缺数据集不同提示")
    code2, _ = _gate_exit(base, candidate, "--allow-mixed-dataset")
    _assert(code2 == 0, f"加 --allow-mixed-dataset 后退出码应为 0，实际 {code2}")
    return "数据集不同默认 exit 2；放行后 dataset_mismatch=true"


def check_probe_failure() -> str:
    """把 RAG 指向不存在端口，断言 run.py 拒绝开跑、退出码 2。"""
    import evals.run as run_module

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    dead_port = sock.getsockname()[1]
    sock.close()

    old_url = run_module.settings.INGEST_SERVICE_URL
    old_model = run_module._probe_model
    old_pg = run_module._probe_pg
    old_redis = run_module._probe_redis
    out = io.StringIO()
    err = io.StringIO()
    run_module.settings.INGEST_SERVICE_URL = f"http://127.0.0.1:{dead_port}"
    # 只测 RAG 失败分支：其余探活打成桩，避免冒烟依赖模型 / PG / Redis
    run_module.__dict__["_probe_model"] = lambda: (True, "stub")
    run_module.__dict__["_probe_pg"] = lambda: (True, "stub")
    run_module.__dict__["_probe_redis"] = lambda: (True, "stub")
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_module.main(["--configs", "A", "--limit", "1"])
    finally:
        run_module.settings.INGEST_SERVICE_URL = old_url
        run_module.__dict__["_probe_model"] = old_model
        run_module.__dict__["_probe_pg"] = old_pg
        run_module.__dict__["_probe_redis"] = old_redis

    text = out.getvalue() + err.getvalue()
    _assert(code == 2, f"探活失败应退出码 2，实际 {code}")
    _assert("环境未就绪：rag" in text, f"缺少可读原因：{text[-300:]}")
    return "RAG 不可达 -> 退出码 2，提示 环境未就绪：rag"


def check_invalid_baseline(base: Run) -> str:
    invalid = _make_run(f"{_STAMP}-fixture-invalid-base", [9, 8, 9], valid_for_quality=False)
    try:
        gate.compare(invalid, base)
    except GateInputError as exc:
        text = str(exc)
        _assert("valid_for_quality" in text, f"错误应说明链路验证：{text}")
        _assert("allow-invalid-baseline" in text, f"错误应给出放行开关：{text}")
    else:
        raise SmokeFailure("无效基线默认应拒绝，实际通过了比较")

    code, _ = _gate_exit(invalid, base)
    _assert(code == 2, f"无效基线默认退出码应为 2，实际 {code}")
    result = gate.compare(invalid, base, allow_invalid_baseline=True)
    warning = result.quality_warning or ""
    _assert(bool(warning), "放行后 quality_warning 不能为空")
    _assert("链路验证" in warning, "quality_warning 应说明链路验证")
    code2, _ = _gate_exit(invalid, base, "--allow-invalid-baseline")
    _assert(code2 == 0, f"加 --allow-invalid-baseline 后退出码应为 0，实际 {code2}")
    return "无效基线默认 exit 2；放行后 quality_warning 非空、exit 0"


def check_partial_set(base: Run) -> str:
    partial = _make_run(
        f"{_STAMP}-fixture-partial-set",
        [9, 8],
        case_ids=("q001", "q002"),
    )
    try:
        gate.compare(base, partial)
    except GateInputError as exc:
        text = str(exc)
        _assert("缺 1 题" in text, f"错误应写缺题数量：{text}")
        _assert("q003" in text, f"错误应列出缺失题号：{text}")
        _assert("allow-partial" in text, f"错误应给出放行开关：{text}")
    else:
        raise SmokeFailure("题集不一致默认应拒绝，实际通过了比较")

    code, _ = _gate_exit(base, partial)
    _assert(code == 2, f"题集不一致默认退出码应为 2，实际 {code}")
    result = gate.compare(base, partial, allow_partial=True)
    _assert(result.compared_cases == 2, f"compared_cases 应为 2，实际 {result.compared_cases}")
    _assert(result.skipped == 1, f"skipped 应为 1，实际 {result.skipped}")
    _assert(result.partial_compare is True, "partial_compare 应为 True")
    _assert("仅比较共同题" in result.reason, f"reason 应写明仅比较共同题：{result.reason}")
    code2, _ = _gate_exit(base, partial, "--allow-partial")
    _assert(code2 == 0, f"加 --allow-partial 后退出码应为 0，实际 {code2}")
    return "题集不一致默认 exit 2；放行后 compared_cases=2 / skipped=1"


def check_zero_delta(base: Run) -> str:
    same = _make_run(f"{_STAMP}-fixture-zero", [9, 8, 9])
    result = gate.compare(base, same)
    _assert(result.status == "PASS", f"零差值应为 PASS，实际 {result.status}")
    _assert(result.delta == 0, f"零差值 delta 应为 0，实际 {result.delta}")
    _assert("无差异" in result.reason, f"零差值 reason 应写无差异：{result.reason}")
    _assert("同口径无漂移" not in result.reason, "零差值不应写同口径无漂移")
    return f"PASS，reason={result.reason!r}"


def check_partial_run(base: Run) -> str:
    import evals.run as run_module

    partial = _make_run(
        f"{_STAMP}-fixture-partial-run",
        [9, 8, 9],
        partial=True,
        dataset_size=20,
    )
    _assert(partial.meta.get("partial") is True, "非全量 run 的 meta.partial 应为 True")
    _assert(partial.meta.get("publishable") is False, "非全量 run 的 meta.publishable 应为 False")
    result = gate.compare(base, partial)
    text = report.write_markdown(partial.run_id, result).read_text(encoding="utf-8")
    _assert("非全量 run" in text, "report 缺非全量提示")
    _assert("n_cases=3 / dataset_size=20" in text, "report 缺 n_cases/dataset_size")
    _assert("不可用于发布" in text, "report 缺不可用于发布提示")
    _assert(run_module._partial_flags(1, 20) == (True, False), "_partial_flags(1,20) 错误")
    _assert(run_module._partial_flags(20, 20) == (False, True), "_partial_flags(20,20) 错误")
    return "meta.partial=true / publishable=false；report 有非全量警告"


def check_code_version(base: Run) -> str:
    drift = _make_run(f"{_STAMP}-fixture-git-drift", [9, 8, 9], git_commit="bbbbbbb")
    result = gate.compare(base, drift)
    _assert(result.code_version_changed is True, "两 run git 不同应置 code_version_changed=True")
    text = report.write_markdown(drift.run_id, result).read_text(encoding="utf-8")
    _assert("代码版本不同" in text, "report 缺代码版本提示")
    _assert("fixture" in text and "bbbbbbb" in text, "report 未写出两侧 git 短哈希")
    _assert("不能归因于配置开关" in text, "report 缺不能归因于配置开关提示")

    drift_bad = _make_run(
        f"{_STAMP}-fixture-git-drift-bad",
        [8.5, 7.5, 8.5],
        git_commit="bbbbbbb",
    )
    bad = gate.compare(base, drift_bad)
    _assert(bad.status == "FAIL_DEGRADED", f"版本漂移 + 退化应为 FAIL_DEGRADED，实际 {bad.status}")
    _assert("含代码版本差异" in bad.reason, f"reason 应写含版本差异：{bad.reason}")
    return "code_version_changed=True；report 提示版本不同，退化原因改为无法归因"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评测门禁离线冒烟（不调模型）")
    parser.add_argument(
        "--results-dir",
        default="",
        help="fixture 产物目录；默认临时目录，CI 传 evals/results/ci-smoke 以产出 artifact",
    )
    args = parser.parse_args(argv)

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.results_dir:
        store.RESULTS_DIR = Path(args.results_dir).expanduser().resolve()
        store.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        location = store.RESULTS_DIR
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="eval-gate-smoke-")
        store.RESULTS_DIR = Path(temp_dir.name)
        location = store.RESULTS_DIR

    base = _make_run(f"{_STAMP}-fixture-base", [9, 8, 9])

    checks: list[tuple[str, Any]] = [
        ("降质 fixture -> FAIL_DEGRADED", lambda: check_degraded(base)),
        ("持平 fixture -> PASS", lambda: check_flat(base)),
        ("提升 +0.03（方差内）-> PASS", lambda: check_improved_in_band(base)),
        ("提升 +0.5（超方差）-> WARN_IMPROVED，退出码 0", lambda: check_warn_improved(base)),
        ("退化 -0.5（超方差）-> FAIL_DEGRADED，退出码 1", lambda: check_fail_degraded(base)),
        ("链路验证 run 报告顶部警告", lambda: check_invalid_report(base)),
        ("无效基线 -> 默认拒绝，放行后 quality_warning", lambda: check_invalid_baseline(base)),
        ("题集不一致 -> 默认拒绝，放行后 compared_cases", lambda: check_partial_set(base)),
        ("零差值 -> reason 写无差异", lambda: check_zero_delta(base)),
        ("非全量 run -> meta.partial/publishable + report 警告", lambda: check_partial_run(base)),
        ("代码版本不同 -> code_version_changed + report 提示", lambda: check_code_version(base)),
        ("净降 1 题（delta=-0.05, n=20）-> PASS", check_net_one_case_down),
        ("净降 3 题（delta=-0.15）-> FAIL_DEGRADED", check_net_three_case_down),
        ("2 升 2 降 -> PASS + 符号检验证据", check_flip_evidence),
        ("PASS -> 变化归因（不写失败归因）", check_pass_attribution_wording),
        ("delta 非零且在阈值内 -> 差异在单题当量内", check_inband_wording),
        ("n=2 -> sample_warning 非空", check_small_sample_warning),
        ("数据集不同 -> exit 2；放行后 dataset_mismatch", check_mixed_dataset),
        ("探活失败 -> 退出码 2", lambda: check_probe_failure()),
        ("边界：方差为 0 -> 阈值取单题当量", lambda: check_zero_variance(base)),
        ("边界：只有 1 题 -> 可读错误", lambda: check_single_case(base)),
        ("边界：per_case 缺 score -> 可读错误", lambda: check_missing_field(base)),
    ]

    failures = 0
    print(f"结果目录：{location}")
    for name, check in checks:
        try:
            detail = check()
        except Exception as exc:  # noqa: BLE001 - 冒烟脚本要汇总所有失败
            failures += 1
            print(f"[FAIL] {name} — {type(exc).__name__}: {exc}")
        else:
            print(f"[PASS] {name} — {detail}")

    print(f"RESULT: {len(checks) - failures} passed, {failures} failed")
    if temp_dir is not None:
        temp_dir.cleanup()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
