#!/usr/bin/env python
"""TEG 轻量评测门禁 —— 测试窗口独立验收（不照抄开发冒烟脚本）。

卡号：TEG「轻量评测门禁」（补丁基线 83bf142：三态判据 + 评测前探活 + 有效性标记）
目的：自己构造 fixture，逐条验收 A1~A13（功能有效性 + 判据修正）、
      B1~B10（不变量 + 措辞一致性）、C1~C10（守卫断言 + 边界观察，观察项只记 open_issue）。
运行：poetry run python "tests/test_agent_cases/TEG-评测门禁/teg_checks.py"
      （cwd 任意；脚本自己定位仓库根）
数据与编码：fixture 全部写 tempfile；A1/A6 会真跑一次最小评测（--limit 1），
      产物落 gitignore 的 evals/results/；台账只读，本脚本不改台账。
      本脚本 UTF-8；台账 GB18030（显式 encoding）。
只读边界：不改 evals/ scripts/ .github/ docs/ 与业务代码；不启动常驻服务。
退出码：0 = 全部通过；1 = 有 [FAIL]。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
_RESULTS = _REPO / "evals" / "results"
_LEDGER = _REPO / "tests" / "test_agent_cases" / "台账用例.csv"
_DATASETS = _REPO / "evals" / "datasets"
_SCHEMA_SHA = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
_META_REQUIRED = (
    "run_id",
    "created_at",
    "config_id",
    "model",
    "judge_model",
    "n_cases",
    "git_commit",
    "flags",
    "schema_sha256",
    "environment",
    "valid_for_quality",
)
_ENV_KEYS = ("rag_ok", "model_ok", "pg_ok", "redis_ok", "checked_at")
_CASE_KEYS = (
    "case_id",
    "question",
    "answer_digest",
    "score",
    "latency_ms",
    "tokens",
    "tool_calls",
    "error",
)
_MISSING = object()

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evals import gate, report, store  # noqa: E402

_WORK = Path(tempfile.mkdtemp(prefix="teg-checks-"))
_RES = _WORK / "results"
_RES.mkdir(parents=True, exist_ok=True)
store.RESULTS_DIR = _RES  # 本脚本内存里的 report/gate 只认临时结果库

_A1: dict[str, Any] = {}


class CheckFailure(AssertionError):
    """单条检查失败；由 run_all() 汇总成 [FAIL]。"""


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _case(
    case_id: str,
    score: Any = _MISSING,
    *,
    tools: list[str] | None = None,
    error: str | None = None,
    latency_ms: int = 1200,
) -> dict[str, Any]:
    """造一题 per_case；score=_MISSING 表示故意不写 score 字段。"""
    item: dict[str, Any] = {
        "case_id": case_id,
        "question": f"fixture question {case_id}",
        "answer_digest": _sha16(case_id),
        "latency_ms": latency_ms,
        "tokens": 100,
        "tool_calls": ["search_knowledge"] if tools is None else list(tools),
        "error": error,
    }
    if score is not _MISSING:
        item["score"] = score
    return item


def _make_fixture(
    run_id: str,
    cases: list[dict[str, Any]],
    *,
    root: Path | None = None,
    valid: bool = True,
    git_commit: str = "aaaaaaa",
    dataset_sha: str = "teg-fixture-sha",
    created_at: str = "2026-11-23T00:00:00",
) -> Path:
    """直接写 meta.json + per_case.json：fixture 不经 store.save_run，避免用被测代码造数据。"""
    scored = sum(1 for item in cases if isinstance(item.get("score"), (int, float)))
    meta: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at,
        "config_id": "FIX",
        "config_name": "fixture",
        "model": "fixture-model",
        "judge_model": "fixture-judge",
        "n_cases": len(cases),
        "n_scored": scored,
        "git_commit": git_commit,
        "flags": {"ENABLE_TOOL_POLICY": "0", "ENABLE_TRACE": "0", "ENABLE_HITL": "0"},
        "schema_sha256": "f" * 64,
        "dataset": "fixture://teg",
        "dataset_sha256": dataset_sha,
        "dataset_size": len(cases),
        "recorded_variance": gate.DEFAULT_RECORDED_VARIANCE,
        "environment": {
            "rag_ok": True,
            "model_ok": True,
            "pg_ok": True,
            "redis_ok": True,
            "checked_at": "2026-11-23T00:00:00",
        },
        "valid_for_quality": valid,
        "invalid_reason": "" if valid else "环境未就绪：rag: fixture 故意标坏",
    }
    run_dir = (root or _RES) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "per_case.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir


def _run(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    merged.setdefault("PRE_COMMIT_HOME", str(_REPO / ".pre-commit-cache"))
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, *args],
        cwd=_REPO,
        env=merged,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _gate_cli(
    *args: str, results: Path | None = None, timeout: float = 120.0
) -> subprocess.CompletedProcess[str]:
    return _run(
        ["-m", "evals.gate", *args],
        env={"EVAL_RESULTS_DIR": str(results or _RES)},
        timeout=timeout,
    )


def _run_dir(run_id: str) -> str:
    return str(_RES / run_id)


def _assert(condition: object, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def _no_traceback(text: str) -> None:
    _assert(
        "Traceback (most recent call last)" not in text, "出现 Python traceback（应给可读错误）"
    )


def _dirs(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {item.name for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")}


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode("utf-8"))
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_ledger() -> list[dict[str, str]]:
    """独立按 GB18030 解析台账（不用 evals.backfill 的实现）。"""
    text = _LEDGER.read_bytes().decode("gb18030")
    return list(csv.DictReader(io.StringIO(text)))


# --------------------------------------------------------------------------- A1


def check_a1() -> str:
    """真跑一次最小评测：三件套 + meta 必填字段齐全且可信。"""
    before = _dirs(_RESULTS)
    proc = _run(["-m", "evals.run", "--configs", "A", "--limit", "1"], timeout=900.0)
    combined = proc.stdout + proc.stderr
    _no_traceback(combined)
    _assert(
        proc.returncode == 0, f"最小评测退出码应为 0，实际 {proc.returncode}：{combined[-500:]}"
    )

    new = _dirs(_RESULTS) - before
    _assert(len(new) == 1, f"应新增 1 个 run 目录，实际 {sorted(new)}")
    run_id = new.pop()
    run_dir = _RESULTS / run_id
    for name in ("meta.json", "per_case.json", "report.md"):
        path = run_dir / name
        _assert(path.is_file() and path.stat().st_size > 0, f"缺少或空文件：{name}")

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    per_case = json.loads((run_dir / "per_case.json").read_text(encoding="utf-8"))
    missing = [key for key in _META_REQUIRED if key not in meta]
    _assert(not missing, f"meta 缺字段：{missing}")
    _assert(meta["run_id"] == run_id, f"meta.run_id={meta['run_id']} 与目录名 {run_id} 不一致")
    try:
        time.strptime(str(meta["created_at"]), "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        raise CheckFailure(f"created_at 不是可解析时间：{meta['created_at']!r}") from None
    _assert(meta["config_id"] == "A", f"config_id 应为 A，实际 {meta['config_id']!r}")
    _assert(str(meta["model"]).strip(), "model 为空")
    _assert(str(meta["judge_model"]).strip(), "judge_model 为空")
    _assert(
        isinstance(meta["n_cases"], int) and meta["n_cases"] == len(per_case) == 1,
        f"n_cases={meta['n_cases']} 与 per_case={len(per_case)} 不一致（--limit 1 应为 1）",
    )
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    _assert(
        meta["git_commit"] == head, f"git_commit={meta['git_commit']!r} 与 HEAD {head!r} 不一致"
    )
    _assert(
        isinstance(meta["flags"], dict) and meta["flags"], f"flags 开关快照为空：{meta['flags']!r}"
    )
    for key in ("ENABLE_TOOL_POLICY", "ENABLE_TRACE", "ENABLE_HITL", "ENABLE_SKILLS"):
        _assert(key in meta["flags"], f"flags 缺开关 {key}")
    _assert(
        meta["schema_sha256"] == _SCHEMA_SHA, f"schema 指纹不是 c48c9d48…：{meta['schema_sha256']}"
    )
    env = meta["environment"]
    _assert(isinstance(env, dict), f"environment 不是对象：{env!r}")
    for key in _ENV_KEYS:
        _assert(key in env, f"environment 缺 {key}")
    _assert(
        all(env[key] is True for key in ("rag_ok", "model_ok", "pg_ok", "redis_ok")),
        f"环境探活并非四项全 ok，但 run 仍记 valid：{env}",
    )
    _assert(
        meta["valid_for_quality"] is True,
        f"环境全 ok 时 valid_for_quality 应为 true：{meta['valid_for_quality']!r}",
    )

    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    _assert("## 1. 元信息" in report_text, "report.md 缺元信息段")
    _assert(_SCHEMA_SHA in report_text, "report.md 未记录 schema 指纹")
    _A1.update(
        {
            "run_id": run_id,
            "meta": meta,
            "report": report_text,
            "report_path": run_dir / "report.md",
        }
    )
    return (
        f"run={run_id}，meta {len(_META_REQUIRED)} 项 + environment {len(_ENV_KEYS)} 项齐全，"
        f"git={meta['git_commit']}，n_cases=1，valid=true"
    )


# --------------------------------------------------------------------------- A2


def check_a2() -> str:
    """三态：+0.5 -> WARN_IMPROVED(0)、-0.5 -> FAIL_DEGRADED(1)、+0.01 -> PASS(0)。"""
    cases = [
        ("teg-up", 0.5, "WARN_IMPROVED", 0),
        ("teg-down", -0.5, "FAIL_DEGRADED", 1),
        ("teg-flat", 0.01, "PASS", 0),
    ]
    evidence: list[str] = []
    for run_id, delta, expected, exit_code in cases:
        proc = _gate_cli(
            "--baseline", _run_dir("teg-base"), "--candidate", _run_dir(run_id), "--json"
        )
        _no_traceback(proc.stdout + proc.stderr)
        _assert(
            proc.returncode == exit_code,
            f"{run_id} 退出码应为 {exit_code}，实际 {proc.returncode}：{proc.stderr[-300:]}",
        )
        payload = json.loads(proc.stdout)
        _assert(
            payload["status"] == expected,
            f"{run_id} status 应为 {expected}，实际 {payload['status']}",
        )
        _assert(
            abs(payload["delta"] - delta) < 1e-9,
            f"{run_id} delta 应为 {delta:+.2f}，实际 {payload['delta']}",
        )
        want_passed = expected != "FAIL_DEGRADED"
        _assert(
            payload["passed"] is want_passed,
            f"{run_id} passed={payload['passed']} 与 status={expected} 不一致",
        )
        evidence.append(f"{delta:+.2f}->{expected}(exit {exit_code}, passed={payload['passed']})")
    return "；".join(evidence)


# --------------------------------------------------------------------------- A3


def check_a3() -> str:
    """四个边界：方差为负 / 只有 1 题 / 缺 score / 题集不一致 —— 可读错误 + 退出码 2。

    注意：本轮判据修正后「方差为 0」是合法输入（有效阈值另有单题当量下限），
    旧断言已移出本检查，改由 A13 断言新行为。
    """
    scenarios = [
        (
            "方差为负",
            [
                "--baseline",
                _run_dir("teg-base"),
                "--candidate",
                _run_dir("teg-up"),
                "--variance",
                "-0.1",
            ],
            ("输入不合法", "方差"),
        ),
        (
            "只有 1 题",
            ["--baseline", _run_dir("teg-single"), "--candidate", _run_dir("teg-single")],
            ("输入不合法", "共同"),
        ),
        (
            "per_case 缺 score",
            ["--baseline", _run_dir("teg-base"), "--candidate", _run_dir("teg-noscore")],
            ("输入不合法", "score"),
        ),
        (
            "题集完全不一致",
            ["--baseline", _run_dir("teg-base"), "--candidate", _run_dir("teg-disjoint")],
            ("输入不合法", "题集不一致"),
        ),
    ]
    evidence: list[str] = []
    for name, args, needles in scenarios:
        proc = _gate_cli(*args)
        combined = proc.stdout + proc.stderr
        _no_traceback(combined)
        _assert(
            proc.returncode == 2,
            f"{name} 退出码应为 2，实际 {proc.returncode}：{combined[-300:]}",
        )
        for needle in needles:
            _assert(needle in combined, f"{name} 错误信息缺 {needle!r}：{combined[-300:]}")
        first = combined.strip().splitlines()[0][:70]
        evidence.append(f"{name} -> 2（{first}）")
    return "；".join(evidence)


# --------------------------------------------------------------------------- A4


def check_a4() -> str:
    """报告五段齐全；换 fixture 后结论句跟着变（证明不是手写）。"""
    base = store.load_run("teg-base")
    down = store.load_run("teg-down")
    res_down = gate.compare(base, down)
    text_down = report.write_markdown("teg-down", res_down).read_text(encoding="utf-8")

    for section in (
        "## 1. 元信息",
        "## 2. 配置矩阵分数表",
        "## 3. 与上一次 run 的 diff",
        "## 4. 结论",
        "## 5. 失败 case 清单",
    ):
        _assert(section in text_down, f"report.md 缺段落：{section}")
    for header in ("均值", "方差", "样本数"):
        _assert(header in text_down, f"矩阵表缺列：{header}")
    _assert("| case | base | cand | diff | 归因 |" in text_down, "缺分题 diff 表头")
    _assert("| q001 |" in text_down, "分题 diff 未列出 q001")
    _assert("判据原文" in text_down and gate.CRITERION in text_down, "结论段未引用判据原文")
    _assert(res_down.reason in text_down, "结论段未使用 gate 生成的结论句")
    _assert("答案退化" in text_down, "失败清单缺归因分类")

    up = store.load_run("teg-up")
    res_up = gate.compare(base, up)
    text_up = report.write_markdown("teg-up", res_up).read_text(encoding="utf-8")
    _assert(res_down.reason != res_up.reason, "换 fixture 后结论句没变（疑似手写）")
    _assert(
        res_down.reason in text_down and res_down.reason not in text_up, "退化结论串到了提升报告"
    )
    _assert(res_up.reason in text_up and res_up.reason not in text_down, "提升结论串到了退化报告")
    return f"五段齐全；{res_down.status} 与 {res_up.status} 结论句各自不同"


# --------------------------------------------------------------------------- A5


def check_a5() -> str:
    """valid_for_quality=false：报告顶部警告；且 gate 默认拒绝把它当基线。"""
    base = store.load_run("teg-base")
    invalid = store.load_run("teg-invalid")
    res = gate.compare(base, invalid, allow_invalid_candidate=True)
    text = report.write_markdown("teg-invalid", res).read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:6])
    _assert("链路验证" in head, f"report.md 顶部缺链路验证警告：{head}")
    _assert("不可用于质量结论" in head, "report.md 顶部缺不可用于质量结论标注")

    proc = _gate_cli("--baseline", _run_dir("teg-invalid"), "--candidate", _run_dir("teg-base"))
    combined = proc.stdout + proc.stderr
    _assert(proc.returncode == 2, f"无效基线默认应被拒绝（exit 2），实际 {proc.returncode}")
    _assert("不可作为质量基线" in combined, f"缺可读拒绝原因：{combined[-300:]}")
    return "报告顶部有链路验证警告；无效基线默认被 gate 拒绝（exit 2），OI-TEG-001 已闭合"


# --------------------------------------------------------------------------- A6


def _dead_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def check_a6() -> str:
    """RAG 指向不存在端口：默认拒绝且不落目录；--allow-degraded 才跑并标 invalid。"""
    dead = f"http://127.0.0.1:{_dead_port()}"
    env = {"INGEST_SERVICE_URL": dead}

    before = _dirs(_RESULTS)
    refuse = _run(["-m", "evals.run", "--configs", "A", "--limit", "1"], env=env, timeout=300.0)
    combined = refuse.stdout + refuse.stderr
    _no_traceback(combined)
    _assert(
        refuse.returncode == 2, f"探活失败应退出 2，实际 {refuse.returncode}：{combined[-400:]}"
    )
    _assert("环境未就绪：rag" in combined, f"缺可读提示 环境未就绪：rag：{combined[-400:]}")
    after_refuse = _dirs(_RESULTS)
    _assert(after_refuse == before, f"拒绝开跑却新增了 run 目录：{sorted(after_refuse - before)}")

    before_deg = _dirs(_RESULTS)
    degraded = _run(
        ["-m", "evals.run", "--configs", "A", "--set", "smoke", "--limit", "1", "--allow-degraded"],
        env=env,
        timeout=600.0,
    )
    combined = degraded.stdout + degraded.stderr
    _no_traceback(combined)
    _assert(
        degraded.returncode == 0,
        f"--allow-degraded 应允许跑完，实际退出 {degraded.returncode}：{combined[-500:]}",
    )
    new = _dirs(_RESULTS) - before_deg
    _assert(len(new) == 1, f"--allow-degraded 应新增 1 个 run，实际 {sorted(new)}")
    run_id = new.pop()
    meta = json.loads((_RESULTS / run_id / "meta.json").read_text(encoding="utf-8"))
    _assert(
        meta["valid_for_quality"] is False,
        f"降级 run 的 valid_for_quality 应为 false：{meta['valid_for_quality']!r}",
    )
    reason = str(meta.get("invalid_reason") or "")
    _assert("rag" in reason and "未就绪" in reason, f"invalid_reason 未记录 RAG 原因：{reason!r}")
    report_text = (_RESULTS / run_id / "report.md").read_text(encoding="utf-8")
    _assert(
        "链路验证" in "\n".join(report_text.splitlines()[:6]), "降级 run 报告顶部缺链路验证警告"
    )
    return f"拒绝->exit 2 且无新目录；--allow-degraded->run {run_id} valid=false，invalid_reason={reason[:60]}"


# --------------------------------------------------------------------------- A7


def check_a7() -> str:
    """backfill：GB18030 解析、候选数=非 pass 行数、带来源 case_id、不改 datasets。"""
    rows = _read_ledger()
    _assert(len(rows) > 0, "台账解析出 0 行（GB18030 解析失败？）")
    _assert(len({len(row) for row in rows}) == 1, "台账列数不一致")
    non_pass = [row for row in rows if row["status"].strip().lower() != "pass"]
    _assert(len(rows) >= 200, f"台账行数异常：{len(rows)}")

    ledger_before = hashlib.sha256(_LEDGER.read_bytes()).hexdigest()
    datasets_before = _tree_hash(_DATASETS)
    out = _WORK / "candidates.json"
    proc = _run(
        ["-m", "evals.backfill", "--ledger", str(_LEDGER), "--out", str(out)], timeout=120.0
    )
    combined = proc.stdout + proc.stderr
    _no_traceback(combined)
    _assert(
        proc.returncode == 0, f"backfill 退出码应为 0，实际 {proc.returncode}：{combined[-400:]}"
    )
    _assert(out.is_file(), "backfill 未生成候选文件")

    payload = json.loads(out.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    _assert(
        payload["ledger_rows"] == len(rows), f"ledger_rows={payload['ledger_rows']} != {len(rows)}"
    )
    _assert(
        len(candidates) == len(non_pass),
        f"候选数 {len(candidates)} 应等于台账非 pass 行数 {len(non_pass)}",
    )
    expected_ids = [row["case_id"].strip() for row in non_pass]
    actual_ids = [item["source_case_id"] for item in candidates]
    _assert(actual_ids == expected_ids, f"来源 case_id 不一致：{actual_ids} != {expected_ids}")
    for item in candidates:
        _assert(
            item["candidate_id"] == f"REG-{item['source_case_id']}", "candidate_id 未带来源 case_id"
        )
        _assert(item["source_status"].lower() != "pass", "候选里混入了 pass 行")

    _assert(
        hashlib.sha256(_LEDGER.read_bytes()).hexdigest() == ledger_before, "backfill 改动了台账"
    )
    _assert(
        _tree_hash(_DATASETS) == datasets_before, "backfill 自动改动了 datasets（应只写 --out）"
    )
    return f"台账 {len(rows)} 行 -> 非 pass {len(non_pass)} 行 -> 候选 {len(candidates)} 条，datasets 未变"


# --------------------------------------------------------------------------- A8


def check_a8() -> str:
    """同一对 run 跑两次 gate：退出码、stdout、结论、报告逐字节一致。"""
    args1 = ("--baseline", _run_dir("teg-base"), "--candidate", _run_dir("teg-down"), "--json")
    first = _gate_cli(*args1)
    second = _gate_cli(*args1)
    _assert(first.returncode == second.returncode, "两次 gate 退出码不同")
    _assert(first.stdout == second.stdout, "两次 gate stdout 不同")
    _assert(first.stderr == second.stderr, "两次 gate stderr 不同")

    base = store.load_run("teg-base")
    down = store.load_run("teg-down")
    text1 = report.write_markdown("teg-down", gate.compare(base, down)).read_text(encoding="utf-8")
    text2 = report.write_markdown("teg-down", gate.compare(base, down)).read_text(encoding="utf-8")
    _assert(text1 == text2, "两次生成的 report.md 不一致")
    return f"两次 gate 输出一致（exit={first.returncode}），report.md 逐字节一致"


# --------------------------------------------------------------------------- A9


def check_a9() -> str:
    """阈值下限：n=20、step=1 时单题当量 0.05；--variance 0.01 不降阈。"""
    payload = json.loads(
        _gate_cli(
            "--baseline",
            _run_dir("teg-n20-base"),
            "--candidate",
            _run_dir("teg-n20-d1"),
            "--variance",
            "0.01",
            "--json",
        ).stdout
    )
    _assert(
        abs(payload["one_case_weight"] - 0.05) < 1e-9,
        f"单题当量应为 0.05，实际 {payload['one_case_weight']}",
    )
    _assert(
        abs(payload["recorded_variance"] - 0.01) < 1e-9,
        f"记录方差应为 0.01，实际 {payload['recorded_variance']}",
    )
    _assert(
        abs(payload["threshold_used"] - 0.05) < 1e-9,
        f"有效阈值应为 0.05（取大），实际 {payload['threshold_used']}",
    )
    return "n=20/step=1 -> one_case_weight=0.05；variance=0.01 时 threshold_used 仍为 0.05"


# --------------------------------------------------------------------------- A10


def check_a10() -> str:
    """净降 1 题（-0.05）-> PASS；净降 3 题（-0.15）-> FAIL_DEGRADED。"""
    one = _gate_cli(
        "--baseline", _run_dir("teg-n20-base"), "--candidate", _run_dir("teg-n20-d1"), "--json"
    )
    payload_one = json.loads(one.stdout)
    _assert(one.returncode == 0, f"净降 1 题应 exit 0，实际 {one.returncode}：{one.stderr[-300:]}")
    _assert(payload_one["status"] == "PASS", f"净降 1 题应为 PASS，实际 {payload_one['status']}")
    _assert(
        abs(payload_one["delta"] + 0.05) < 1e-9,
        f"delta 应为 -0.05，实际 {payload_one['delta']}",
    )

    three = _gate_cli(
        "--baseline", _run_dir("teg-n20-base"), "--candidate", _run_dir("teg-n20-d3"), "--json"
    )
    payload_three = json.loads(three.stdout)
    _assert(
        three.returncode == 1, f"净降 3 题应 exit 1，实际 {three.returncode}：{three.stderr[-300:]}"
    )
    _assert(
        payload_three["status"] == "FAIL_DEGRADED",
        f"净降 3 题应为 FAIL_DEGRADED，实际 {payload_three['status']}",
    )
    _assert(
        abs(payload_three["delta"] + 0.15) < 1e-9,
        f"delta 应为 -0.15，实际 {payload_three['delta']}",
    )
    return "净降 1 题：exit 0 / PASS / delta -0.05；净降 3 题：exit 1 / FAIL_DEGRADED / delta -0.15"


# --------------------------------------------------------------------------- A11


def check_a11() -> str:
    """逐题抖动：flips{up,down,unchanged} 之和 = n_cases；sign_test_p ∈ [0,1]。"""
    mix = gate.compare(store.load_run("teg-n20-base"), store.load_run("teg-n20-mix"))
    _assert(
        mix.flips["up"] == 10 and mix.flips["down"] == 10,
        f"混合 fixture flips 不对：{dict(mix.flips)}",
    )
    _assert(
        sum(mix.flips.values()) == mix.n_cases == 20,
        f"flips 之和 {sum(mix.flips.values())} 应等于 n_cases {mix.n_cases}",
    )
    _assert(0.0 <= mix.sign_test_p <= 1.0, f"sign_test_p 越界：{mix.sign_test_p}")

    one = gate.compare(store.load_run("teg-n20-base"), store.load_run("teg-n20-d1"))
    _assert(
        sum(one.flips.values()) == one.n_cases,
        f"单题退化 flips 之和 {sum(one.flips.values())} 应等于 n_cases {one.n_cases}",
    )
    _assert(0.0 <= one.sign_test_p <= 1.0, f"sign_test_p 越界：{one.sign_test_p}")
    return (
        f"混合 flips={dict(mix.flips)} 求和 20，p={mix.sign_test_p:.2f}；"
        f"单题退化 flips={dict(one.flips)} 求和 20"
    )


# --------------------------------------------------------------------------- A12


def check_a12() -> str:
    """--variance auto：同 config/commit/数据集有重复 run 时给实测方差，否则回退 0.0408。"""
    hit = _gate_cli(
        "--baseline",
        _run_dir("teg-n20-base"),
        "--candidate",
        _run_dir("teg-n20-d1"),
        "--variance",
        "auto",
        "--json",
    )
    _assert(
        hit.returncode == 0, f"auto 命中时 exit 应为 0，实际 {hit.returncode}：{hit.stderr[-300:]}"
    )
    payload_hit = json.loads(hit.stdout)
    _assert("variance auto:" in hit.stderr, f"stderr 未打印实测方差：{hit.stderr[-300:]}")
    _assert(
        abs(payload_hit["recorded_variance"] - 0.5) < 1e-9,
        f"实测方差应为 0.5（auto-a 9.0 vs auto-b 8.5），实际 {payload_hit['recorded_variance']}",
    )

    miss = _gate_cli(
        "--baseline",
        _run_dir("teg-unique-sha"),
        "--candidate",
        _run_dir("teg-n20-d1"),
        "--variance",
        "auto",
        "--json",
    )
    payload_miss = json.loads(miss.stdout)
    _assert("回退到 0.0408" in miss.stderr, f"未命中时缺回退提示：{miss.stderr[-300:]}")
    _assert(
        abs(payload_miss["recorded_variance"] - 0.0408) < 1e-9,
        f"未命中应回退 0.0408，实际 {payload_miss['recorded_variance']}",
    )
    return (
        "命中：stderr variance auto: 0.5000，recorded_variance=0.5；未命中：回退 0.0408 并打印提示"
    )


# --------------------------------------------------------------------------- A13


def check_a13() -> str:
    """--variance 0 合法（旧「方差 0 报错」边界已废弃）：阈值取单题当量。"""
    proc = _gate_cli(
        "--baseline",
        _run_dir("teg-n20-base"),
        "--candidate",
        _run_dir("teg-n20-d1"),
        "--variance",
        "0",
        "--json",
    )
    combined = proc.stdout + proc.stderr
    _no_traceback(combined)
    _assert(proc.returncode in (0, 1), f"--variance 0 不应报错（退出码 2），实际 {proc.returncode}")
    payload = json.loads(proc.stdout)
    _assert(
        abs(payload["recorded_variance"]) < 1e-12,
        f"recorded_variance 应为 0，实际 {payload['recorded_variance']}",
    )
    _assert(
        abs(payload["threshold_used"] - 0.05) < 1e-9,
        f"方差 0 时阈值应取单题当量 0.05，实际 {payload['threshold_used']}",
    )
    return f"--variance 0 合法：exit={proc.returncode}，recorded_variance=0，threshold_used=0.05"


# --------------------------------------------------------------------------- B1


def check_b1() -> str:
    """pytest 225 passed；--collect-only 229/233（检查脚本未被收集）。"""
    proc = _run(["-m", "pytest", "-q"], timeout=900.0)
    tail = [line for line in proc.stdout.splitlines() if line.strip()][-1]
    match = re.search(r"(\d+) passed", tail)
    if match is None:
        raise CheckFailure(f"pytest 末行无法解析：{tail}")
    _assert(int(match.group(1)) == 225, f"pytest passed 应为 225，实际 {match.group(1)}：{tail}")

    collect = _run(["-m", "pytest", "--collect-only", "-q"], timeout=300.0)
    collected = [line for line in collect.stdout.splitlines() if "tests collected" in line]
    _assert(bool(collected), f"collect-only 输出异常：{collect.stdout[-300:]}")
    match = re.search(r"(\d+)/(\d+) tests collected", collected[-1])
    if match is None:
        raise CheckFailure(f"collect-only 末行无法解析：{collected[-1]}")
    _assert(
        (match.group(1), match.group(2)) == ("229", "233"),
        f"收集数应为 229/233，实际 {match.group(1)}/{match.group(2)}",
    )
    return f"pytest: {tail}；collect-only: {match.group(1)}/{match.group(2)}"


# --------------------------------------------------------------------------- B2


def check_b2() -> str:
    """schema 指纹 MATCH（c48c9d48…）；import-linter 3 kept。"""
    schema = _run(["scripts/smoke_tools_schema.py"], timeout=180.0)
    _no_traceback(schema.stdout + schema.stderr)
    _assert(schema.returncode == 0, f"schema 脚本退出码 {schema.returncode}")
    _assert("status: MATCH" in schema.stdout, f"schema 指纹非 MATCH：{schema.stdout[-300:]}")
    _assert(_SCHEMA_SHA in schema.stdout, "schema 指纹不是 c48c9d48…")

    lint_bin = Path(sys.executable).with_name("lint-imports")
    command = [str(lint_bin)] if lint_bin.is_file() else ["lint-imports"]
    lint = subprocess.run(
        command, cwd=_REPO, capture_output=True, text=True, timeout=300.0, check=False
    )
    plain = re.sub(r"\x1b\[[0-9;]*m", "", lint.stdout + lint.stderr)
    _assert(lint.returncode == 0, f"lint-imports 退出码 {lint.returncode}：{plain[-300:]}")
    _assert("3 kept, 0 broken" in plain, f"import-linter 不是 3 kept：{plain[-300:]}")
    return f"schema MATCH（{_SCHEMA_SHA[:8]}…）；import-linter 3 kept, 0 broken"


# --------------------------------------------------------------------------- B3


def _git_diff_quiet(*paths: str) -> bool:
    proc = subprocess.run(
        ["git", "diff", "--quiet", "origin/main", "--", *paths],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def check_b3() -> str:
    """ci.yml 零改动；新增 workflow 只在 workflow_dispatch/schedule 触发。"""
    _assert(_git_diff_quiet(".github/workflows/ci.yml"), "ci.yml 相对 origin/main 有改动")
    new_workflow = _REPO / ".github" / "workflows" / "eval-gate.yml"
    _assert(new_workflow.is_file(), "缺少 eval-gate.yml")
    import yaml  # 项目已有依赖

    data = yaml.safe_load(new_workflow.read_text(encoding="utf-8")) or {}
    triggers = data.get("on") or data.get(True) or {}
    keys = set(triggers) if isinstance(triggers, dict) else {str(triggers)}
    _assert(
        keys <= {"workflow_dispatch", "schedule"},
        f"新增 workflow 触发方式超出允许范围：{sorted(keys)}",
    )
    _assert(
        "workflow_dispatch" in keys and "schedule" in keys,
        f"新增 workflow 缺触发方式：{sorted(keys)}",
    )
    return f"ci.yml 无差异；eval-gate.yml 仅 {sorted(keys)} 触发"


# --------------------------------------------------------------------------- B4


def check_b4() -> str:
    """无新增依赖：pyproject.toml / poetry.lock 相对 origin/main 无差异。"""
    _assert(_git_diff_quiet("pyproject.toml"), "pyproject.toml 有改动（疑似新增依赖）")
    _assert(_git_diff_quiet("poetry.lock"), "poetry.lock 有改动（疑似新增依赖）")
    return "pyproject.toml / poetry.lock 均无差异"


# --------------------------------------------------------------------------- B5


def check_b5() -> str:
    """无前端产物、无服务启动代码。"""
    added = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=A", "origin/main"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    everything = [path for path in (*added, *untracked) if path.strip()]
    frontend = [
        path
        for path in everything
        if path.lower().endswith((".html", ".vue", ".js", ".ts", ".css"))
    ]
    _assert(not frontend, f"新增了前端产物：{frontend[:5]}")
    servers = []
    for path in everything:
        if not path.endswith(".py"):
            continue
        text = (_REPO / path).read_text(encoding="utf-8", errors="replace")
        if "uvicorn.run(" in text or ".listen(" in text:
            servers.append(path)
    _assert(not servers, f"新增文件含服务启动代码：{servers}")
    return f"新增/未跟踪 {len(everything)} 个文件，无前端产物、无服务启动代码"


# --------------------------------------------------------------------------- B6


def check_b6() -> str:
    """docs/baseline.md 既有数字未被改动（相对 origin/main 零 diff）。"""
    baseline = _REPO / "docs" / "baseline.md"
    _assert(baseline.is_file(), "docs/baseline.md 不存在")
    _assert(_git_diff_quiet("docs/baseline.md"), "docs/baseline.md 相对 origin/main 有改动")
    return "docs/baseline.md 零 diff（冻结数字未动）"


# --------------------------------------------------------------------------- B7


def check_b7() -> str:
    """台账里没有开发窗口写的 TEG 行：TEG 行只能由 test-TEG 追加。"""
    rows = _read_ledger()
    teg = [row for row in rows if row["case_id"].strip().upper().startswith("TEG-")]
    outsiders = [row["case_id"] for row in teg if row.get("run_by", "").strip() != "test-TEG"]
    _assert(not outsiders, f"发现非 test-TEG 写的 TEG 行：{outsiders}")
    return f"TEG 行 {len(teg)} 条，全部 run_by=test-TEG"


# --------------------------------------------------------------------------- B8


def check_b8() -> str:
    """PASS 时 reason 不得出现「失败归因」，应写「变化归因」。"""
    res = gate.compare(store.load_run("teg-n20-base"), store.load_run("teg-n20-mix"))
    _assert(res.status == "PASS", f"混合 fixture 应为 PASS，实际 {res.status}")
    _assert(
        "失败归因" not in res.reason,
        f"PASS reason 仍含「失败归因」：{res.reason}",
    )
    _assert("变化归因" in res.reason, f"PASS reason 未写「变化归因」：{res.reason}")
    return f"PASS 措辞合规：{res.reason[:120]}"


# --------------------------------------------------------------------------- B9


def check_b9() -> str:
    """delta==0 写「无差异」；delta!=0 且小于阈值不得写「无漂移」。"""
    zero = gate.compare(store.load_run("teg-base"), store.load_run("teg-same"))
    _assert(abs(zero.delta) < 1e-12, f"零差值 fixture delta 应为 0，实际 {zero.delta}")
    _assert("无差异" in zero.reason, f"零差值 reason 未写「无差异」：{zero.reason}")

    # 显式给 0.06 记录方差，让 |delta|=0.05 严格小于有效阈值（0.06），
    # 专门检验「非零差值但落在阈值内」的措辞，而不是卡在等号边界。
    within = gate.compare(
        store.load_run("teg-n20-base"), store.load_run("teg-n20-d1"), variance=0.06
    )
    _assert(abs(within.delta) > 1e-12, f"非零差值 fixture delta 不应为 0：{within.delta}")
    _assert(within.status == "PASS", f"净降 1 题应为 PASS，实际 {within.status}")
    _assert(
        abs(within.delta) < within.threshold_used,
        f"fixture 应满足 |delta| < 阈值：{within.delta} vs {within.threshold_used}",
    )
    _assert(
        "无漂移" not in within.reason,
        f"非零差值（|delta| 小于阈值）仍写「无漂移」：{within.reason}",
    )
    return f"零差值写「无差异」；非零差值 reason 未写「无漂移」：{within.reason[:100]}"


# --------------------------------------------------------------------------- B10


def check_b10() -> str:
    """真实复算两条命令：符号检验与题当量必须同时出现在输出里。"""
    pair = _gate_cli("--baseline", _run_dir("teg-n20-base"), "--candidate", _run_dir("teg-n20-d1"))
    last = _gate_cli("--compare", "last")
    for name, proc in (("显式两 run", pair), ("--compare last", last)):
        _no_traceback(proc.stdout + proc.stderr)
        _assert(
            proc.returncode in (0, 1),
            f"{name} 退出码异常：{proc.returncode}：{proc.stderr[-300:]}",
        )
        _assert("符号检验" in proc.stdout, f"{name} 输出缺「符号检验」：{proc.stdout[-300:]}")
        _assert("题当量" in proc.stdout, f"{name} 输出缺「题当量」：{proc.stdout[-300:]}")
    return "两条命令（显式 --baseline/--candidate 与 --compare last）均同时输出符号检验与题当量"


# --------------------------------------------------------------------------- C1


def check_c1() -> str:
    """无效基线：默认拒绝（exit 2）；--allow-invalid-baseline 放行且 quality_warning 非空。"""
    refused = _gate_cli("--baseline", _run_dir("teg-invalid"), "--candidate", _run_dir("teg-base"))
    combined = refused.stdout + refused.stderr
    _no_traceback(combined)
    _assert(refused.returncode == 2, f"无效基线默认应 exit 2，实际 {refused.returncode}")
    _assert("不可作为质量基线" in combined, f"拒绝原因缺可读说明：{combined[-300:]}")

    allowed = _gate_cli(
        "--baseline",
        _run_dir("teg-invalid"),
        "--candidate",
        _run_dir("teg-base"),
        "--allow-invalid-baseline",
        "--json",
    )
    _no_traceback(allowed.stdout + allowed.stderr)
    _assert(
        allowed.returncode == 0,
        f"放行后应 exit 0，实际 {allowed.returncode}：{allowed.stderr[-300:]}",
    )
    payload = json.loads(allowed.stdout)
    warning = str(payload.get("quality_warning") or "")
    _assert(bool(warning.strip()), "放行后 quality_warning 为空")
    _assert("链路验证" in warning, f"quality_warning 未说明链路验证：{warning!r}")
    return (
        f"默认 exit=2（不可作为质量基线）；--allow-invalid-baseline 放行，"
        f"quality_warning={warning[:48]}"
    )


# --------------------------------------------------------------------------- C2


def check_c2() -> str:
    """题集不一致：默认拒绝并列出两侧缺题；--allow-partial 标记 compared/skipped。"""
    refused = _gate_cli("--baseline", _run_dir("teg-base"), "--candidate", _run_dir("teg-partial"))
    combined = refused.stdout + refused.stderr
    _no_traceback(combined)
    _assert(refused.returncode == 2, f"题集不一致默认应 exit 2，实际 {refused.returncode}")
    for needle in ("题集不一致", "候选缺 1 题：q003", "基线缺 1 题：q004"):
        _assert(needle in combined, f"拒绝原因缺 {needle!r}：{combined[-300:]}")

    allowed = _gate_cli(
        "--baseline",
        _run_dir("teg-base"),
        "--candidate",
        _run_dir("teg-partial"),
        "--allow-partial",
        "--json",
    )
    _no_traceback(allowed.stdout + allowed.stderr)
    _assert(
        allowed.returncode == 0,
        f"放行后应 exit 0，实际 {allowed.returncode}：{allowed.stderr[-300:]}",
    )
    payload = json.loads(allowed.stdout)
    _assert(
        payload["compared_cases"] == 2,
        f"compared_cases 应为 2，实际 {payload['compared_cases']}",
    )
    _assert(payload["skipped"] == 2, f"skipped 应为 2，实际 {payload['skipped']}")
    _assert(payload["partial_compare"] is True, "partial_compare 应为 true")
    _assert(
        payload["missing_in_candidate"] == ["q003"],
        f"missing_in_candidate={payload['missing_in_candidate']}",
    )
    _assert(
        payload["missing_in_baseline"] == ["q004"],
        f"missing_in_baseline={payload['missing_in_baseline']}",
    )
    return "默认 exit=2（候选缺 q003/基线缺 q004）；--allow-partial 后 compared_cases=2、skipped=2"


# --------------------------------------------------------------------------- C3


def check_c3() -> str:
    """零差值：reason 必须含「无差异」。"""
    res = gate.compare(store.load_run("teg-base"), store.load_run("teg-same"))
    _assert(res.delta == 0, f"全同分数 delta 应为 0，实际 {res.delta}")
    _assert("无差异" in res.reason, f"零差值 reason 未含 无差异：{res.reason!r}")
    out = _gate_cli("--baseline", _run_dir("teg-base"), "--candidate", _run_dir("teg-same")).stdout
    _assert("无差异" in out, "CLI 输出未含 无差异")
    return f"delta=0.0000，reason={res.reason!r}"


# --------------------------------------------------------------------------- C4


def check_c4() -> str:
    """--limit 1：meta partial=true/publishable=false，report 顶部不可用于发布。"""
    meta = _A1.get("meta") or {}
    text = _A1.get("report") or ""
    _assert(bool(meta), "A1 未产出 meta，无法断言 --limit 1")
    _assert(meta.get("partial") is True, f"meta.partial 应为 true，实际 {meta.get('partial')!r}")
    _assert(
        meta.get("publishable") is False,
        f"meta.publishable 应为 false，实际 {meta.get('publishable')!r}",
    )
    _assert(meta.get("dataset_size") == 20 and meta.get("n_cases") == 1, "预期 20 题里跑 1 题")
    head = "\n".join(text.splitlines()[:8])
    _assert("不可用于发布" in head, f"report 顶部缺 不可用于发布 警告：{head}")
    _assert("非全量" in head, f"report 顶部缺 非全量 标注：{head}")
    return "meta partial=true/publishable=false；report 顶部含 非全量/不可用于发布"


# --------------------------------------------------------------------------- C5


def check_c5() -> str:
    """两 run git 不同：code_version_changed=true，报告写不能归因于配置开关。"""
    base = store.load_run("teg-base")
    drift = store.load_run("teg-git-drift")
    res = gate.compare(base, drift)
    _assert(res.code_version_changed is True, "code_version_changed 应为 true")
    _assert(
        res.baseline_git == "aaaaaaa" and res.candidate_git == "bbbbbbb",
        f"git 字段不对：{res.baseline_git} vs {res.candidate_git}",
    )
    payload = json.loads(
        _gate_cli(
            "--baseline",
            _run_dir("teg-base"),
            "--candidate",
            _run_dir("teg-git-drift"),
            "--json",
        ).stdout
    )
    _assert(payload["code_version_changed"] is True, "JSON code_version_changed 应为 true")
    text = report.write_markdown("teg-git-drift", res).read_text(encoding="utf-8")
    _assert("不能归因于配置开关" in text, "report 缺 不能归因于配置开关 提示")
    return "code_version_changed=true（aaaaaaa vs bbbbbbb）；report 含 不能归因于配置开关"


# --------------------------------------------------------------------------- C6


def check_c6() -> str:
    """失败 case 超 10 条：report 是否截断并说明。"""
    base = store.load_run("teg-twelve-base")
    cand = store.load_run("teg-twelve-cand")
    res = gate.compare(base, cand)
    _assert(len(res.failures) == 12, f"应造出 12 条失败，实际 {len(res.failures)}")
    text = report.write_markdown("teg-twelve-cand", res).read_text(encoding="utf-8")
    failures_part = text.split("## 5.", 1)[1] if "## 5." in text else ""
    rows = [
        line for line in failures_part.splitlines() if line.startswith("| q") and "答案退化" in line
    ]
    truncated = "另有 2 条失败未列出" in text
    _assert(len(rows) <= 10, f"失败清单应最多 10 条，实际 {len(rows)}")
    _assert(truncated, "失败清单超过 10 条时未说明截断")
    return f"{len(res.failures)} 条失败 -> 列出 {len(rows)} 条，注明另有 2 条未列出"


# --------------------------------------------------------------------------- C7


def check_c7() -> str:
    """results 下 ci-smoke/ 这类子目录：latest_run() 不应误选。"""
    root = _WORK / "c7-results"
    _make_fixture(
        "20261123-100000-direct", [_case("q001", 9)], root=root, created_at="2026-11-23T10:00:00"
    )
    nested = root / "ci-smoke"
    _make_fixture(
        "20261123-120000-nested",
        [_case("q001", 9)],
        root=nested,
        created_at="2026-11-23T12:00:00",
    )
    (nested / "20261123-130000-broken").mkdir(parents=True, exist_ok=True)
    (nested / "20261123-130000-broken" / "meta.json").write_text("{}", encoding="utf-8")

    saved = store.RESULTS_DIR
    store.RESULTS_DIR = root
    try:
        runs = store.list_runs()
        latest = store.latest_run()
    finally:
        store.RESULTS_DIR = saved
    if latest is None:
        raise CheckFailure("latest_run 返回 None")
    _assert(
        latest.run_id == "20261123-100000-direct",
        f"latest_run 误选了子目录/坏目录：{latest.run_id}",
    )
    _assert(
        len(runs) == 1,
        f"list_runs 应只认直接子目录里带 meta.json 的 run，实际 {[run.run_id for run in runs]}",
    )
    return f"latest_run -> {latest.run_id}（ci-smoke 子目录未被误选），list_runs=1"


# --------------------------------------------------------------------------- C8


def check_c8() -> str:
    """观察：n_cases=2 时单题当量 0.5，几乎必然 PASS，是否提示样本过小。"""
    res = gate.compare(store.load_run("teg-n2-base"), store.load_run("teg-n2-cand"))
    _assert(res.n_cases == 2, f"n_cases 应为 2，实际 {res.n_cases}")
    _assert(
        abs(res.one_case_weight - 0.5) < 1e-9,
        f"n=2 单题当量应为 0.5，实际 {res.one_case_weight}",
    )
    _assert(res.status == "PASS", f"delta=-0.5 仍应为 PASS，实际 {res.status}")
    hints = [w for w in ("样本过小", "判别力", "题数过少") if w in res.reason]
    suffix = "" if hints else " -> 未提示样本过小/判据无判别力，记 OI-TEG-006"
    return f"n=2 当量 0.5，delta=-0.5 仍 PASS；提示={hints or '无'}{suffix}"


# --------------------------------------------------------------------------- C9


def check_c9() -> str:
    """观察：非整数分数（0~1）时 score_step 如何推断，是否写进 reason。"""
    payload = json.loads(
        _gate_cli(
            "--baseline",
            _run_dir("teg-unit-base"),
            "--candidate",
            _run_dir("teg-unit-cand"),
            "--json",
        ).stdout
    )
    step = float(payload["score_step"])
    _assert(abs(step - 0.1) < 1e-9, f"0~1 分数应推出 step=0.1，实际 {step}")
    in_reason = ("步长" in payload["reason"]) or ("score_step" in payload["reason"])
    suffix = "" if in_reason else " -> score_step 未写进 reason（只写单题当量），记 OI-TEG-009"
    return (
        f"score_step={step}（由最小非零分数差推出），"
        f"one_case_weight={payload['one_case_weight']}{suffix}"
    )


# --------------------------------------------------------------------------- C10


def check_c10() -> str:
    """观察：两次 run 数据集 sha256 不同时的行为。"""
    proc = _gate_cli(
        "--baseline", _run_dir("teg-base"), "--candidate", _run_dir("teg-other-dataset")
    )
    combined = proc.stdout + proc.stderr
    _no_traceback(combined)
    silent = proc.returncode in (0, 1) and "数据集" not in combined
    suffix = " -> 直接对比不校验 dataset_sha256（静默按均值比较），记 OI-TEG-010" if silent else ""
    return f"sha 不同时 exit={proc.returncode}，输出{'未' if silent else '已'}提及数据集{suffix}"


# --------------------------------------------------------------------------- 主流程


CHECKS: list[tuple[str, Any]] = [
    ("A1 run/store 落库三件套 + meta 完整性", check_a1),
    ("A2 gate 三态 + passed 与 status 一致", check_a2),
    ("A3 gate 四个边界可读错误 + 退出码 2", check_a3),
    ("A4 report 五段 + 结论句随 fixture 变化", check_a4),
    ("A5 链路验证标记 + gate 基线行为记录", check_a5),
    ("A6 探活拒绝不落目录 + --allow-degraded 标 invalid", check_a6),
    ("A7 backfill GB18030/候选数/来源/不动 datasets", check_a7),
    ("A8 gate 幂等", check_a8),
    ("A9 阈值下限：单题当量 0.05，variance 0.01 不降阈", check_a9),
    ("A10 净降 1 题 PASS / 净降 3 题 FAIL", check_a10),
    ("A11 逐题抖动 flips 求和 = n，符号检验 p ∈ [0,1]", check_a11),
    ("A12 --variance auto 命中实测方差 / 未命中回退", check_a12),
    ("A13 --variance 0 合法，阈值取单题当量", check_a13),
    ("B1 pytest 225 + collect-only 229/233", check_b1),
    ("B2 schema MATCH + lint-imports 3 kept", check_b2),
    ("B3 ci.yml 零改动 + 新 workflow 仅手动/定时", check_b3),
    ("B4 无新增依赖", check_b4),
    ("B5 无前端产物/无服务启动代码", check_b5),
    ("B6 docs/baseline.md 数字未动", check_b6),
    ("B7 台账无开发窗口写的 TEG 行", check_b7),
    ("B8 PASS 时不得写「失败归因」（应写「变化归因」）", check_b8),
    ("B9 非零差值小于阈值不得写「无漂移」", check_b9),
    ("B10 两条真实命令同时输出符号检验与题当量", check_b10),
    ("C1 无效基线默认拒绝 + 放行记 quality_warning", check_c1),
    ("C2 题集不一致默认拒绝 + --allow-partial 标记", check_c2),
    ("C3 零差值 reason 含「无差异」", check_c3),
    ("C4 --limit 1 标 partial/publishable + 报告警告", check_c4),
    ("C5 git 版本不同 code_version_changed + 报告提示", check_c5),
    ("C6 观察：失败 case 超 10 条是否截断", check_c6),
    ("C7 观察：ci-smoke 子目录是否被 latest_run 误选", check_c7),
    ("C8 观察：n=2 当量 0.5 是否提示样本过小", check_c8),
    ("C9 观察：非整数分数的 score_step 推断与措辞", check_c9),
    ("C10 观察：数据集 sha256 不同时的行为", check_c10),
]


def _build_fixtures() -> None:
    base_scores = [9, 8, 9]
    _make_fixture(
        "teg-base", [_case(f"q{index:03d}", score) for index, score in enumerate(base_scores, 1)]
    )
    _make_fixture(
        "teg-up",
        [_case(f"q{index:03d}", score + 0.5) for index, score in enumerate(base_scores, 1)],
    )
    _make_fixture(
        "teg-down",
        [_case(f"q{index:03d}", score - 0.5) for index, score in enumerate(base_scores, 1)],
    )
    _make_fixture(
        "teg-flat",
        [_case(f"q{index:03d}", score + 0.01) for index, score in enumerate(base_scores, 1)],
    )
    _make_fixture(
        "teg-same", [_case(f"q{index:03d}", score) for index, score in enumerate(base_scores, 1)]
    )
    _make_fixture(
        "teg-invalid",
        [_case(f"q{index:03d}", score) for index, score in enumerate(base_scores, 1)],
        valid=False,
    )
    _make_fixture("teg-single", [_case("q001", 9)])
    _make_fixture("teg-noscore", [_case("q001", _MISSING), _case("q002", 8)])
    _make_fixture(
        "teg-disjoint",
        [_case(f"q{index:03d}", score) for index, score in enumerate(base_scores, 101)],
    )
    _make_fixture("teg-partial", [_case("q001", 9), _case("q002", 8), _case("q004", 9)])
    _make_fixture(
        "teg-git-drift",
        [_case(f"q{index:03d}", score) for index, score in enumerate(base_scores, 1)],
        git_commit="bbbbbbb",
    )
    _make_fixture("teg-twelve-base", [_case(f"q{index:03d}", 9) for index in range(1, 13)])
    _make_fixture(
        "teg-twelve-cand",
        [_case(f"q{index:03d}", 5) for index in range(1, 13)],
        created_at="2026-11-23T00:01:00",
    )

    # ---- 判据修正（本轮）：20 题 / 单题当量 0.05 ----
    _make_fixture("teg-n20-base", [_case(f"q{index:03d}", 9) for index in range(1, 21)])
    _make_fixture(
        "teg-n20-d1",
        [_case("q001", 8), *[_case(f"q{index:03d}", 9) for index in range(2, 21)]],
    )
    _make_fixture(
        "teg-n20-d3",
        [
            *[_case(f"q{index:03d}", 8) for index in range(1, 4)],
            *[_case(f"q{index:03d}", 9) for index in range(4, 21)],
        ],
    )
    _make_fixture(
        "teg-n20-mix",
        [_case(f"q{index:03d}", 8.9 if index <= 10 else 9.1) for index in range(1, 21)],
    )
    _make_fixture(
        "teg-auto-a",
        [_case(f"q{index:03d}", 9) for index in range(1, 21)],
        created_at="2026-11-23T03:00:00",
    )
    _make_fixture(
        "teg-auto-b",
        [_case(f"q{index:03d}", 8.5) for index in range(1, 21)],
        created_at="2026-11-23T03:01:00",
    )
    _make_fixture(
        "teg-unique-sha",
        [_case(f"q{index:03d}", 9) for index in range(1, 21)],
        dataset_sha="teg-unique-sha",
        created_at="2026-11-23T00:02:00",
    )
    _make_fixture(
        "teg-n2-base",
        [_case("q001", 9), _case("q002", 9)],
        created_at="2026-11-23T00:00:30",
    )
    _make_fixture(
        "teg-n2-cand",
        [_case("q001", 8), _case("q002", 9)],
        created_at="2026-11-23T00:00:31",
    )
    _make_fixture(
        "teg-unit-base",
        [_case(f"q{index:03d}", 0.8) for index in range(1, 21)],
        created_at="2026-11-23T00:00:32",
    )
    _make_fixture(
        "teg-unit-cand",
        [_case("q001", 0.7), *[_case(f"q{index:03d}", 0.8) for index in range(2, 21)]],
        created_at="2026-11-23T00:00:33",
    )
    _make_fixture(
        "teg-other-dataset",
        [_case(f"q{index:03d}", score) for index, score in enumerate(base_scores, 1)],
        dataset_sha="teg-other-sha",
        created_at="2026-11-23T00:00:40",
    )


def main() -> int:
    print(f"临时结果库：{_RES}")
    print(f"仓库：{_REPO}")
    _build_fixtures()
    failures = 0
    for name, check in CHECKS:
        started = time.time()
        try:
            detail = check()
        except Exception as exc:  # noqa: BLE001 - 汇总所有失败，不中断后续检查
            failures += 1
            print(f"[FAIL] {name} — {type(exc).__name__}: {exc}")
        else:
            print(f"[PASS] {name} — {detail}（{time.time() - started:.1f}s）")

    total = len(CHECKS)
    print(f"RESULT: {total - failures} passed, {failures} failed")
    print(f"临时目录：{_WORK}（保留供复核，可手动删除）")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
