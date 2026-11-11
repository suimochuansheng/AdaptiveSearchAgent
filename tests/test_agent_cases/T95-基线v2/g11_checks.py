#!/usr/bin/env python3
"""G11（T9-5 基线 v2）的验收检查：端口口径 + 评测专用自动放行开关。

不对应 pytest 用例（文件名不是 `test_*.py`，不影响 `docs/baseline.md` §2 冻结的
nodeids 哈希），用法：

    cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent
    poetry run python tests/test_agent_cases/T95-基线v2/g11_checks.py

检查项：
1. `BudgetLedger` 只在 `agent/app/ports.py` 声明，`agent/domain/budget.py` 不含 Protocol；
2. `EVAL_AUTO_APPROVE=1` 且 `ENV=prod` 时被忽略（仍走“需审批未启用 -> 拒绝”分支），
   且不计数；同一场景 `ENV=dev` 时直接放行并计数 +1。
"""

from __future__ import annotations

import ast
import asyncio
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent.adapters.tools import executor  # noqa: E402
from agent.app.ports import BudgetLedger  # noqa: E402
from agent.domain.tool import CostModel, ToolSpec  # noqa: E402
from config import settings  # noqa: E402

_ROOT = REPO


class CheckFailed(AssertionError):
    """任一检查不通过时抛出，脚本以非 0 退出。"""


def _spec() -> ToolSpec:
    return ToolSpec(
        name="g11_auto_approve_probe",
        description="G11 验收探针（需审批）",
        input_schema={"properties": {"query": {"type": "string"}}, "type": "object"},
        permission="dangerous",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=5.0,
        max_output_tokens=100,
        requires_approval=True,
        feature_flag=None,
    )


def _imports_protocol(tree: ast.Module) -> bool:
    """AST 判定：是否 import 了 `typing.Protocol` 或继承 `Protocol`。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "Protocol" for alias in node.names):
                return True
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "Protocol":
                    return True
                if isinstance(base, ast.Attribute) and base.attr == "Protocol":
                    return True
    return False


def check_domain_has_no_protocol() -> None:
    domain_files = glob.glob(str(_ROOT / "agent" / "domain" / "*.py"))
    offenders = [
        Path(path).name
        for path in domain_files
        if _imports_protocol(ast.parse(Path(path).read_text(encoding="utf-8")))
    ]
    if offenders:
        raise CheckFailed(f"domain 里仍有 Protocol：{offenders}")
    # 运行时属性判定（docstring 里保留"已迁走"的说明不算声明）
    import agent.domain.budget as domain_budget

    if hasattr(domain_budget, "BudgetLedger") or hasattr(domain_budget, "NoopLedger"):
        raise CheckFailed("agent/domain/budget.py 仍导出 BudgetLedger / NoopLedger")
    budget_ledger = BudgetLedger
    if not hasattr(budget_ledger, "add") or not hasattr(budget_ledger, "snapshot"):
        raise CheckFailed("app.ports.BudgetLedger 端口缺方法")
    print("PASS 1/2：BudgetLedger 只在 app/ports.py；domain 无 Protocol")


async def _run_probe(*, env: str, hitl: bool) -> tuple[str, int, int]:
    """构造一次“需审批”调用，返回（结果文本, fn 实际被调次数, auto_approved 计数）。"""
    settings.ENABLE_TOOL_POLICY = True
    settings.ENABLE_HITL = hitl
    settings.EVAL_AUTO_APPROVE = True
    settings.ENV = env
    executor.reset_auto_approved_count()

    calls: list[str] = []

    async def probe(query: str) -> str:
        calls.append(query)
        return "PROBE_EXECUTED"

    wrapped = executor.wrap_tool(probe, _spec())
    with executor.tool_scope("g11-check"):
        result = await wrapped(query="x")
    return str(result), len(calls), executor.auto_approved_count()


def check_prod_ignores_switch() -> None:
    old = (
        settings.ENABLE_TOOL_POLICY,
        settings.ENABLE_HITL,
        settings.EVAL_AUTO_APPROVE,
        settings.ENV,
    )
    try:
        result, called, count = asyncio.run(_run_probe(env="prod", hitl=False))
        if called != 0 or count != 0 or "人工审批" not in result:
            raise CheckFailed(
                f"ENV=prod 未忽略 EVAL_AUTO_APPROVE：result={result!r} called={called} count={count}"
            )
        result, called, count = asyncio.run(_run_probe(env="dev", hitl=False))
        if called != 1 or count != 1 or result != "PROBE_EXECUTED":
            raise CheckFailed(
                f"ENV=dev 自动放行未生效：result={result!r} called={called} count={count}"
            )
        print("PASS 2/2：ENV=prod 忽略 EVAL_AUTO_APPROVE；ENV=dev 直接放行并计数 +1")
    finally:
        (
            settings.ENABLE_TOOL_POLICY,
            settings.ENABLE_HITL,
            settings.EVAL_AUTO_APPROVE,
            settings.ENV,
        ) = old


def main() -> int:
    check_domain_has_no_protocol()
    check_prod_ignores_switch()
    print("G11 checks: 2/2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
