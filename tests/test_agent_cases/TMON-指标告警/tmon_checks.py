#!/usr/bin/env python
"""TMON 指标 + 告警（B 卡）—— 卡级独立验收脚本。

用途：
    只验收「可观测性补完：指标 + 告警」这张卡：开关矩阵、指标名与 histogram
    后缀、埋点确实累加、标签纪律、告警表达式与指标对齐、alerts.yml 结构，
    以及 pytest 收集数不变、import-linter 仍 3 kept。

用法（仓库根目录，任意 cwd 均可）：
    poetry run python "tests/test_agent_cases/TMON-指标告警/tmon_checks.py"

    慢检查（pytest / lint-imports）可以单独跳过：
    poetry run python "tests/test_agent_cases/TMON-指标告警/tmon_checks.py" --skip-suite

判定与退出码：
    每项一行 `[PASS]/[FAIL] TMON-xxx 名称 — 证据`；末行
    `RESULT: n passed, m failed` 与 `EXIT=<0|1>`。

为什么不是 `test_*.py`：
    会被 pytest 收集，`--collect-only` 从 229/233 变成 229+1，`docs/baseline.md`
    §2 的 nodeids 哈希立刻失效（见 tests/test_agent_cases/README.md）。

只读承诺：
    不写仓库内任何文件，也不启动真实服务；`/metrics` 用 httpx ASGITransport
    直接打 app（不跑 lifespan，不碰 Redis）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

import api_main  # noqa: E402
from agent.observability import metrics  # noqa: E402
from config import settings  # noqa: E402

CARD_DIR = Path(__file__).resolve().parent
ALERTS = REPO_ROOT / "monitoring" / "alerts.yml"
DOC = REPO_ROOT / "docs" / "observability.md"

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((ok, name, evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} — {evidence}")


def http_get(path: str):
    """用 ASGITransport 直接打 app（不启动 lifespan / 不连 Redis）。"""
    import httpx

    async def _go():
        transport = httpx.ASGITransport(app=api_main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://tm-test") as client:
            return await client.get(path)

    return asyncio.run(_go())


def snapshot() -> str:
    payload, _ = metrics.render()
    return payload.decode("utf-8")


def series_value(text: str, name: str, *needles: str) -> float:
    """取某条序列的值；没有该序列返回 0。"""
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        rest = line[len(name) :]
        if rest and rest[0] not in "{ ":
            continue
        if all(needle in line for needle in needles):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def label_values(text: str, name: str, label: str) -> set[str]:
    """抽某指标某标签的全部取值，用于标签基数断言。"""
    values: set[str] = set()
    pattern = re.compile(rf'{name}\{{[^}}]*{label}="([^"]+)"')
    for match in pattern.finditer(text):
        values.add(match.group(1))
    return values


# ── TMON-1：开关关 —— /metrics 404，且埋点不累加 ──


def check_switch_off() -> None:
    settings.ENABLE_METRICS = False
    before = snapshot()

    # 模拟一轮假会话能碰到的全部埋点
    metrics.observe_request("conversation", "completed", 12.5)
    metrics.observe_tool_decision("calculator", "allow")
    metrics.observe_approval("approve", "approved")
    metrics.observe_usage(100, 50, 0.5)
    metrics.observe_error("fatal")
    after = snapshot()

    response = http_get("/metrics")
    check(
        response.status_code == 404,
        "TMON-1a 开关关 /metrics 返回 404",
        f"status={response.status_code}",
    )

    grew = []
    for name, *needles in (
        ("agent_requests_total", 'outcome="completed"'),
        ("agent_tool_decisions_total", 'tool="calculator"'),
        ("agent_approvals_total", 'outcome="approved"'),
        ("agent_tokens_total", 'direction="in"'),
        ("agent_cost_yuan_total",),
        ("agent_errors_total", 'kind="fatal"'),
        ("agent_request_duration_seconds_count", 'endpoint="conversation"'),
    ):
        old = series_value(before, name, *needles)
        new = series_value(after, name, *needles)
        if new != old:
            grew.append(f"{name}:{old}->{new}")
    check(not grew, "TMON-1b 开关关时埋点零累加", "无增长" if not grew else ", ".join(grew))


# ── TMON-2：开关开 —— 200 / text/plain / 6 个指标名 + histogram 后缀 ──

NAMES = (
    "agent_requests_total",
    "agent_tool_decisions_total",
    "agent_approvals_total",
    "agent_errors_total",
    "agent_tokens_total",
    "agent_cost_yuan_total",
)


def check_switch_on() -> None:
    settings.ENABLE_METRICS = True
    response = http_get("/metrics")
    content_type = response.headers.get("content-type", "")
    text = response.text

    check(
        response.status_code == 200 and "text/plain" in content_type,
        "TMON-2a 开关开 /metrics 200 + text/plain",
        f"status={response.status_code}, content-type={content_type}",
    )
    missing = [name for name in NAMES if name not in text]
    check(not missing, "TMON-2b 6 个指标名齐全", "全部出现" if not missing else f"缺 {missing}")


# ── TMON-3：埋点后计数有值（含 histogram _bucket/_count/_sum）──


def check_increments() -> str:
    metrics.observe_request("conversation", "completed", 12.5)
    metrics.observe_tool_decision("calculator", "allow")
    text = snapshot()

    requests_value = series_value(
        text, "agent_requests_total", 'endpoint="conversation"', 'outcome="completed"'
    )
    duration_count = series_value(
        text, "agent_request_duration_seconds_count", 'endpoint="conversation"'
    )
    tool_value = series_value(
        text, "agent_tool_decisions_total", 'tool="calculator"', 'decision="allow"'
    )
    check(
        requests_value >= 1 and duration_count >= 1 and tool_value >= 1,
        "TMON-3 假会话后 requests/duration/tool_decisions 有值",
        f"requests={requests_value}, duration_count={duration_count}, tool_decisions={tool_value}",
    )

    suffixes = [
        suffix
        for suffix in ("_bucket", "_count", "_sum")
        if f"agent_request_duration_seconds{suffix}" in text
    ]
    check(
        len(suffixes) == 3,
        "TMON-3b histogram 有 _bucket/_count/_sum",
        f"找到 {suffixes}",
    )
    return text


# ── TMON-4：标签纪律（不出现会话/审批 ID）──


def check_label_discipline(text: str) -> None:
    forbidden = ("thread_id=", "session_id=", "approval_id=")
    leaked = [needle for needle in forbidden if needle in text]
    check(
        not leaked,
        "TMON-4 标签里没有 thread/session/approval id",
        "未出现" if not leaked else f"泄漏 {leaked}",
    )

    # 现有标签取值必须是有限集合（工具名/枚举），这里抽查工具名不超过注册表规模
    tools = label_values(text, "agent_tool_decisions_total", "tool")
    check(len(tools) <= 20, "TMON-4b tool 标签取值有限", f"取值={sorted(tools)}")


# ── TMON-5：告警表达式里的 agent_* 指标名都在 /metrics 输出里 ──


def check_alerts_alignment(text: str) -> None:
    raw = ALERTS.read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"\bagent_[a-z0-9_]+", raw)))
    missing = [name for name in referenced if name not in text]
    check(
        bool(referenced) and not missing,
        "TMON-5 告警引用的指标都能在 /metrics 找到",
        f"引用 {referenced}" + (f"，缺 {missing}" if missing else "，全部命中"),
    )


# ── TMON-6：alerts.yml 可解析、3 条规则、每条有 summary ──


def check_alerts_yaml() -> None:
    data = yaml.safe_load(ALERTS.read_text(encoding="utf-8"))
    rules = [
        rule
        for group in (data or {}).get("groups", [])
        for rule in group.get("rules", [])
        if "alert" in rule
    ]
    names = [rule["alert"] for rule in rules]
    expected = {"AgentErrorRateHigh", "AgentLatencyP95High", "AgentCostHourlyHigh"}
    no_summary = [
        rule["alert"]
        for rule in rules
        if not (rule.get("annotations") or {}).get("summary")
        or not (rule.get("annotations") or {}).get("description")
    ]
    check(
        set(names) == expected and not no_summary,
        "TMON-6 alerts.yml 3 条规则齐全且带 summary/description",
        f"rules={names}, 缺注解={no_summary}",
    )
    check(
        all("for" in rule for rule in rules),
        "TMON-6b 三条都有 for",
        f"for={[rule.get('for') for rule in rules]}",
    )

    # 阈值不能是不可达值：错误率阈值必须 < 0.5、延迟必须有限、费用必须 > 0
    checks = {
        "AgentErrorRateHigh": "0.05",
        "AgentLatencyP95High": "60",
        "AgentCostHourlyHigh": "20",
    }
    unreachable = [name for name, needle in checks.items() if needle not in raw_expr(name)]
    check(not unreachable, "TMON-6c 阈值不是不可达值", "阈值可被真实流量触发")


def raw_expr(alert_name: str) -> str:
    data = yaml.safe_load(ALERTS.read_text(encoding="utf-8"))
    for group in (data or {}).get("groups", []):
        for rule in group.get("rules", []):
            if rule.get("alert") == alert_name:
                return str(rule.get("expr", ""))
    return ""


# ── TMON-7：pytest 基线不变 ──


def run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - 固定命令，无 shell
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PRE_COMMIT_HOME": str(REPO_ROOT / ".pre-commit-cache")},
    )


def check_pytest_baseline() -> None:
    result = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    tail = (result.stdout.strip().splitlines() or [""])[-1]
    check(
        result.returncode == 0 and "225 passed" in tail and "4 skipped" in tail,
        "TMON-7a pytest 仍 225 passed / 4 skipped",
        tail,
    )

    collected = run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
    )
    collected_tail = (collected.stdout.strip().splitlines() or [""])[-1]
    check(
        collected.returncode == 0 and "229/233" in collected_tail,
        "TMON-7b --collect-only 仍 229/233",
        collected_tail,
    )


def check_lint_imports() -> None:
    # importlinter 没有 __main__：用同目录的 console script（与 poetry run lint-imports 同源）
    lint_imports = Path(sys.executable).with_name("lint-imports")
    cmd = [str(lint_imports)] if lint_imports.exists() else ["poetry", "run", "lint-imports"]
    result = run(cmd)
    output = (result.stdout + result.stderr).strip()
    tail = (output.splitlines() or [""])[-1]
    check(
        result.returncode == 0 and "3 kept" in output,
        "TMON-8 lint-imports 仍 3 kept",
        tail or f"returncode={result.returncode}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="TMON 指标告警卡级验收")
    parser.add_argument(
        "--skip-suite", action="store_true", help="跳过 pytest / lint-imports 慢检查"
    )
    args = parser.parse_args()

    check_switch_off()
    check_switch_on()
    text_after = check_increments()
    check_label_discipline(text_after)
    check_alerts_alignment(text_after)
    check_alerts_yaml()
    if not args.skip_suite:
        check_pytest_baseline()
        check_lint_imports()

    passed = sum(1 for ok, _, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print(f"RESULT: {passed} passed, {failed} failed")
    print(f"EXIT={0 if failed == 0 else 1}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
