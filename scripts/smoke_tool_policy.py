#!/usr/bin/env python
"""T6-3 冒烟：工具级权限与预算的离线演示（**不连模型、不连 Redis**）。

跑三组东西，全部在本进程内完成：

1. **开关关闭**：`to_lc_tools()` 返回原函数（`is` 同一对象）、schema 哈希 = 基线；
2. **开关开启**：工具被包装，但签名跟随 `__wrapped__`，schema 哈希仍 = 基线；
3. **5 个工具 × 3 种决策**：`decide()` 的纯函数结果表；
   再用内存账本演示执行器的 allow / deny / require_approval / timeout / 异常
   与“3 次调用后 calls==3、cost==3*unit_cost”的账本口径。

用法（任意 cwd 均可，建议仓库根目录）：

    poetry run python scripts/smoke_tool_policy.py

基线哈希与 `scripts/smoke_tools_schema.py` 同一口径：
`json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()` 的 sha256。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import agent / src
    sys.path.insert(0, str(REPO))

# 警告与演示输出走同一个流，避免 stderr 抢先、输出看起来像乱序
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="[log] %(message)s")

from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.adapters.tools import executor, registry
from agent.adapters.tools.budget_memory import MemoryBudgetLedger
from agent.domain.budget import BudgetSnapshot
from agent.domain.tool_policy import PolicyContext, decide
from config import settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"

_SCENARIOS: list[tuple[str, Callable[[], BudgetSnapshot]]] = [
    ("预算内", lambda: BudgetSnapshot()),
    ("调用超限", lambda: BudgetSnapshot(calls=settings.TOOL_MAX_CALLS_PER_TURN)),
    ("费用超限", lambda: BudgetSnapshot(cost=settings.TOOL_SESSION_COST_LIMIT + 0.5)),
]


def _original(name: str) -> Callable[..., Any]:
    """取工具原本的函数对象（冒烟脚本用私有 `_resolve`，只为对比身份）。"""
    return registry._resolve(name)


def _schema_sha256(tools: list[Callable[..., Any]]) -> tuple[str, int]:
    schemas = [convert_to_openai_tool(tool) for tool in tools]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest(), len(canonical)


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _print_switch_off() -> None:
    settings.ENABLE_TOOL_POLICY = False
    tools = registry.to_lc_tools()
    digest, size = _schema_sha256(tools)

    _section("1. 开关关闭（默认 ENABLE_TOOL_POLICY=False）")
    print(f"{'工具':<20}{'identity(原函数，is)':<24}{'__wrapped__ 属性'}")
    for name, tool in zip(registry.TOOL_SPECS, tools, strict=True):
        identity = tool is _original(name)
        print(f"{name:<20}{str(identity):<24}{hasattr(tool, '__wrapped__')}")
    print(
        "注：search_tavily 的原函数自身带 __wrapped__（tenacity 重试装饰器），"
        "判“是否被本卡包装”要看 identity/is，不看属性。"
    )
    print(f"\nschema sha256: {digest}")
    print(f"schema bytes : {size}")
    print(f"基线一致     : {digest == BASELINE_SHA256}")


def _print_switch_on() -> None:
    settings.ENABLE_TOOL_POLICY = True
    tools = registry.to_lc_tools()
    digest, size = _schema_sha256(tools)

    _section("2. 开关开启（ENABLE_TOOL_POLICY=True：策略 + 超时 + 记账包装）")
    print(f"{'工具':<20}{'已包装(is not)':<16}{'签名一致':<12}{'identity(同一对象)'}")
    for name, tool in zip(registry.TOOL_SPECS, tools, strict=True):
        original = _original(name)
        same_sig = inspect.signature(tool) == inspect.signature(original)
        print(f"{name:<20}{str(tool is not original):<16}{str(same_sig):<12}{tool is original}")
    print(f"\nschema sha256: {digest}")
    print(f"schema bytes : {size}")
    print(f"基线一致     : {digest == BASELINE_SHA256}")


def _print_decision_matrix() -> None:
    _section("3. 5 个工具 × 3 种决策（decide 纯函数，不执行任何工具）")
    header = f"{'工具':<20}{'权限':<12}{'场景':<12}{'结论':<18}{'code':<20}理由"
    print(header)
    print("-" * 110)
    for spec in registry.all_specs():
        for label, make_snapshot in _SCENARIOS:
            ctx = PolicyContext(
                scope="smoke",
                max_calls=settings.TOOL_MAX_CALLS_PER_TURN,
                cost_limit=settings.TOOL_SESSION_COST_LIMIT,
            )
            decision = decide(spec, ctx, make_snapshot())
            print(
                f"{spec.name:<20}{spec.permission:<12}{label:<12}"
                f"{decision.kind:<18}{decision.code:<20}{decision.reason}"
            )
        print("-" * 110)


async def _print_executor_demo() -> None:
    settings.ENABLE_TOOL_POLICY = True
    _section("4. 执行器行为（内存账本，scope=demo / ledger-demo）")

    # ── allow：正常执行并记账 ──
    ledger = MemoryBudgetLedger()
    calculator_spec = registry.spec("calculator")
    wrapped_calc = executor.wrap_tool(_original("calculator"), calculator_spec, ledger)
    with executor.tool_scope("demo"):
        result = await wrapped_calc(expression="1 + 1")
    print(f"[allow]    calculator('1 + 1') -> {result}")
    print(f"           账本 demo: {ledger.snapshot('demo')}")

    # ── deny：第 9 次调用（上限 8），fn 不被调用 ──
    calls = {"n": 0}

    async def counted(keyword: str) -> str:
        calls["n"] += 1
        return "ok"

    tavily_spec = registry.spec("search_tavily")
    deny_ledger = MemoryBudgetLedger()
    deny_ledger.add("demo-deny", calls=settings.TOOL_MAX_CALLS_PER_TURN)
    deny_wrapped = executor.wrap_tool(counted, tavily_spec, deny_ledger)
    with executor.tool_scope("demo-deny"):
        deny_result = await deny_wrapped(keyword="第 9 次")
    print(f"\n[deny]     第 9 次调用 -> {deny_result}")
    print(f"           fn 未被调用: {calls['n'] == 0}")

    # ── require_approval：deep_research，fn 不被调用（不碰网络） ──
    research_calls = {"n": 0}

    async def fake_research(query: str) -> str:
        research_calls["n"] += 1
        return "不应该被执行"

    wrapped_research = executor.wrap_tool(fake_research, registry.spec("deep_research"), ledger)
    with executor.tool_scope("demo-approval"):
        approval_result = await wrapped_research(query="演示")
    print(f"\n[approval] deep_research -> {approval_result}")
    print(f"           fn 未被调用: {research_calls['n'] == 0}")

    # ── timeout：0.1s 超时 + 睡 1s ──
    async def slow() -> str:
        await asyncio.sleep(1.0)
        return "太慢了"

    slow_spec = replace(tavily_spec, name="slow_tool", timeout_seconds=0.1, requires_approval=False)
    wrapped_slow = executor.wrap_tool(slow, slow_spec, ledger)
    with executor.tool_scope("demo-timeout"):
        timeout_result = await wrapped_slow()
    print(f"\n[timeout]  slow_tool -> {timeout_result}")
    print(f"           账本 demo-timeout: {ledger.snapshot('demo-timeout')}")

    # ── 异常：归一为可读文本，不抛栈 ──
    async def broken() -> str:
        raise RuntimeError("模拟工具内部炸了")

    broken_spec = replace(tavily_spec, name="broken_tool", requires_approval=False)
    wrapped_broken = executor.wrap_tool(broken, broken_spec, ledger)
    with executor.tool_scope("demo-error"):
        error_result = await wrapped_broken()
    print(f"\n[error]    broken_tool -> {error_result}")
    print(f"           账本 demo-error: {ledger.snapshot('demo-error')}")

    # ── 账本口径：3 次调用后 calls==3、cost==3*unit_cost ──
    _section("5. 账本口径（search_tavily 的 spec，unit_cost=0.02）")
    unit_cost = tavily_spec.cost_model.unit_cost

    async def fake_search(keyword: str) -> str:
        return f"结果：{keyword}"

    ledger_demo = MemoryBudgetLedger()
    wrapped_search = executor.wrap_tool(fake_search, tavily_spec, ledger_demo)
    with executor.tool_scope("ledger-demo"):
        for i in range(3):
            await wrapped_search(keyword=f"关键词 {i + 1}")

    snap = ledger_demo.snapshot("ledger-demo")
    print(f"3 次调用后 snapshot: {snap}")
    print(f"calls == 3          : {snap.calls == 3}")
    print(f"cost  == 3*unit_cost: {snap.cost == 3 * unit_cost}  （{snap.cost} vs {3 * unit_cost}）")


async def main() -> int:
    _print_switch_off()
    _print_switch_on()
    _print_decision_matrix()
    await _print_executor_demo()
    _section("冒烟结束：策略 / 执行器 / 账本三件套均可在默认关闭下零影响")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
