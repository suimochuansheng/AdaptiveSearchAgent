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
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evals import gate, report, store  # noqa: E402
from evals.gate import GateInputError, Run  # noqa: E402

_STAMP = time.strftime("%Y%m%d-%H%M%S")
_CASE_IDS = ("q001", "q002", "q003")

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
    scores: list[float | None],
    *,
    drop_tools: bool = False,
    valid_for_quality: bool = True,
) -> Run:
    per_case = []
    for index, score in enumerate(scores):
        case_id = _CASE_IDS[index]
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
        "git_commit": "fixture",
        "flags": {"ENABLE_TOOL_POLICY": "0"},
        "schema_sha256": "0" * 64,
        "dataset": "fixture",
        "dataset_sha256": "fixture-sha",
        "recorded_variance": gate.DEFAULT_RECORDED_VARIANCE,
        "environment": {
            "rag_ok": True,
            "model_ok": True,
            "pg_ok": True,
            "redis_ok": True,
            "checked_at": "fixture",
        },
        "valid_for_quality": valid_for_quality,
    }
    store.save_run(run_id, meta, per_case)
    return store.load_run(run_id)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _gate_exit(baseline: Run, candidate: Run) -> tuple[int, str]:
    """走真实 CLI 入口，断言退出码；stdout 一并返回供错误信息使用。"""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = gate.main(
            [
                "--baseline",
                str(store.RESULTS_DIR / baseline.run_id),
                "--candidate",
                str(store.RESULTS_DIR / candidate.run_id),
            ]
        )
    return code, out.getvalue()


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
    result = gate.compare(base, candidate)
    text = report.write_markdown(candidate.run_id, result).read_text(encoding="utf-8")
    _assert(LINK_ONLY_WARNING in text, "valid_for_quality=false 的报告缺顶部警告")
    _assert("不可用于质量结论" in text, "报告缺不可用质量结论标注")
    return "链路验证警告已写入 report.md 顶部"


def check_zero_variance(base: Run, candidate: Run) -> str:
    try:
        gate.compare(base, candidate, variance=0)
    except GateInputError as exc:
        text = str(exc)
        _assert("方差" in text, f"错误信息应提到方差：{text}")
        return text
    raise SmokeFailure("方差为 0 时应给可读错误，实际通过了比较")


def check_single_case(base: Run) -> str:
    single = Run(
        run_id="fixture-single",
        meta={"recorded_variance": gate.DEFAULT_RECORDED_VARIANCE},
        per_case=(_case("q001", 9),),
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
        ("探活失败 -> 退出码 2", lambda: check_probe_failure()),
        ("边界：方差为 0 -> 可读错误", lambda: check_zero_variance(base, base)),
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
