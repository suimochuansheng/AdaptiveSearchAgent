#!/usr/bin/env python
"""T63 权限与预算（T6-3）—— 测试窗口 G5 独立验收脚本。

用途：
    只验收 T6-3 的工具级权限与预算：A 类 10 条验收（A1~A10）、B 类 4 条不变量
    （B1~B4）、C 类 5 条边界 / 反例（C1~C5）。**不读开发窗口自测结论**，每条都从
    当前代码与命令输出独立取证；不改 `agent/`、`src/`、`config.py`。

用法（仓库根目录）：
    poetry run python "tests/test_agent_cases/T63-权限与预算/t63_checks.py"

判定与退出码：
    每项一行 `[PASS]/[FAIL] T63-xxx 名称 — 证据`；末行 `RESULT: n passed, m failed`
    与 `EXIT=<0|1>`。`0` = 20 项全过；`1` = 至少一项失败。C 类失败额外打印
    `OPEN_ISSUE:`，仍计入 failed（本卡的 C 项如失败需要另立问题单）。

为什么不是 `test_*.py`：
    会被 pytest 收集，`--collect-only` 从 227/231 变成 227+1，`docs/baseline.md`
    §2 的 nodeids 哈希立刻失效（见 tests/test_agent_cases/README.md）。

只读承诺：
    本脚本不写仓库内任何文件；台账 `台账用例.csv` 由测试窗口在脚本全绿后另行追加。
"""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import hashlib
import inspect
import io
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import agent / src / config
    sys.path.insert(0, str(REPO))
#: 卡级基线（pytest 通过数 / 收集数 / nodeids）唯一来源：`docs/baseline.md` §2。
CASE_ROOT = REPO / "tests" / "test_agent_cases"
if str(CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CASE_ROOT))

import _baseline  # noqa: E402

LEDGER = REPO / "tests" / "test_agent_cases" / "台账用例.csv"
DEV_COMMIT = "423c1ea"
BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
BASELINE_BYTES = 2558
EXPECTED_ORDER = [
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
]


# ── 基础设施 ─────────────────────────────────────────────────────────


class Checker:
    """每项一行 `[PASS]/[FAIL]`；异常信息即失败证据。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.issues: list[str] = []

    def run(self, case_id: str, title: str, fn: Any, *, c_class: bool = False) -> None:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - 验收脚本要汇总所有失败项
            self.failed += 1
            print(f"[FAIL] {case_id} {title} — {type(exc).__name__}: {exc}")
            if c_class:
                self.issues.append(f"{case_id} {type(exc).__name__}: {exc}")
                print(f"OPEN_ISSUE: {case_id} {type(exc).__name__}: {exc}")
        else:
            self.passed += 1
            print(f"[PASS] {case_id} {title} — {detail}")


def _run(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(  # noqa: S603 - 参数固定、shell=False
        args,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        check=False,
    )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args])


def _last_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


@contextmanager
def _settings(**changes: Any) -> Any:
    """临时改 settings 字段，退出时恢复（pydantic Settings 允许属性赋值）。"""
    from config import settings

    old = {key: getattr(settings, key) for key in changes}
    for key, value in changes.items():
        setattr(settings, key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(settings, key, value)


def _spec(name: str, **changes: Any) -> Any:
    from agent.adapters.tools import registry

    base = registry.spec(name)
    return dataclasses.replace(base, **changes) if changes else base


def _sha_tools(tools: list[Any]) -> tuple[str, int]:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schemas = [convert_to_openai_tool(tool) for tool in tools]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest(), len(canonical)


def _imports_of(path: Path) -> list[str]:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


# ── A 类：10 条验收 ──────────────────────────────────────────────────


def check_001() -> str:
    """A1：flag 关时 to_lc_tools() 每项与原函数是同一对象。"""
    from agent.adapters.tools import registry

    with _settings(ENABLE_TOOL_POLICY=False):
        from src.conversation_graph import calculator, deep_research
        from src.tools.offload import read_offload
        from src.tools.rag import search_knowledge
        from src.tools.search import search_tavily

        origins = {
            "search_knowledge": search_knowledge,
            "search_tavily": search_tavily,
            "calculator": calculator,
            "deep_research": deep_research,
            "read_offload": read_offload,
        }
        tools = registry.to_lc_tools()

    assert len(tools) == 5, f"工具数不是 5：{len(tools)}"
    bad = [
        name for name, tool in zip(EXPECTED_ORDER, tools, strict=True) if tool is not origins[name]
    ]
    assert not bad, f"不是同一对象：{bad}"
    return "5/5 identity True（flag=False，is 同一对象）"


def check_002() -> str:
    """A2（前半）：flag 关时 schema 哈希与字节数仍是基线。"""
    from agent.adapters.tools import registry

    with _settings(ENABLE_TOOL_POLICY=False):
        sha, size = _sha_tools(registry.to_lc_tools())
    assert sha == BASELINE_SHA256, f"schema 哈希漂移：{sha}"
    assert size == BASELINE_BYTES, f"schema 字节数漂移：{size}"
    return f"sha256={sha[:8]}..{sha[-4:]} bytes={size}（基线一致，flag=False）"


def check_003() -> str:
    """A2（后半）：全量 pytest 与 `docs/baseline.md` §2 一致（数字比较，skipped 不比）。"""
    proc = _run([sys.executable, "-m", "pytest", "-q"])
    last = _last_line(proc.stdout)
    assert proc.returncode == 0, f"pytest 退出码 {proc.returncode}：{last}"
    ok, detail = _baseline.pytest_matches(last)
    assert ok, detail
    return f"pytest -q 末行：{last}"


def check_004() -> str:
    """A3：flag 开 + read 工具正常执行，账本 calls=1。"""

    async def scenario() -> str:
        from agent.adapters.tools import registry
        from agent.adapters.tools.budget_memory import MemoryBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool
        from src.conversation_graph import calculator

        ledger = MemoryBudgetLedger()
        wrapped = wrap_tool(calculator, registry.spec("calculator"), ledger)
        with tool_scope("t63-a3"):
            result = await wrapped("2 * 21")
        snap = ledger.snapshot("t63-a3")
        assert result == "2 * 21 = 42", f"返回异常：{result!r}"
        assert snap.calls == 1, f"账本 calls 不是 1：{snap.calls}"
        return f"calculator('2 * 21') -> {result!r}；账本 calls={snap.calls}"

    with _settings(ENABLE_TOOL_POLICY=True):
        return asyncio.run(scenario())


def check_005() -> str:
    """A4：flag 开 + 0.1s 超时 + 1s sleep 的假工具 -> 可读错误且无 Traceback。"""

    async def scenario() -> str:
        from agent.adapters.tools.budget_memory import MemoryBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool

        async def slow_tool() -> str:
            await asyncio.sleep(1.0)
            return "never"

        ledger = MemoryBudgetLedger()
        wrapped = wrap_tool(slow_tool, _spec("calculator", timeout_seconds=0.1), ledger)
        before = set(asyncio.all_tasks())
        start = time.monotonic()
        with tool_scope("t63-a4"):
            result = await wrapped()
        elapsed = time.monotonic() - start
        await asyncio.sleep(0)
        residual = set(asyncio.all_tasks()) - before

        assert isinstance(result, str), f"返回值不是字符串：{type(result)!r}"
        assert "超时" in result, f"没有超时文案：{result!r}"
        assert "Traceback" not in result, f"泄漏了调用栈：{result!r}"
        assert ledger.snapshot("t63-a4").calls == 1, "超时未记 1 次"
        assert elapsed < 0.8, f"没有及时返回：{elapsed:.2f}s"
        assert not residual, f"有残留任务：{residual}"
        return f"返回 {result!r}；无 Traceback；calls=1；耗时 {elapsed:.2f}s；残留任务 0"

    with _settings(ENABLE_TOOL_POLICY=True):
        return asyncio.run(scenario())


def check_006() -> str:
    """A5：flag 开 + 第 9 次调用（上限 8）-> denied_calls，假工具 0 次。"""

    async def scenario() -> tuple[str, int, int, str]:
        from agent.adapters.tools.budget_memory import MemoryBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool
        from agent.domain.budget import BudgetSnapshot
        from agent.domain.tool_policy import PolicyContext, decide

        counter = 0

        async def fake_tool() -> str:
            nonlocal counter
            counter += 1
            return "ok"

        ledger = MemoryBudgetLedger()
        ledger.add("t63-a5", calls=8)
        wrapped = wrap_tool(fake_tool, _spec("calculator"), ledger)
        with tool_scope("t63-a5"):
            result = await wrapped()
        snap = ledger.snapshot("t63-a5")
        decision = decide(
            _spec("calculator"),
            PolicyContext(scope="t63-a5", max_calls=8, cost_limit=2.0),
            BudgetSnapshot(calls=8),
        )
        return result, counter, snap.calls, decision.code

    with _settings(ENABLE_TOOL_POLICY=True, TOOL_MAX_CALLS_PER_TURN=8):
        result, counter, calls, code = asyncio.run(scenario())
    assert "上限" in result, f"没有次数上限文案：{result!r}"
    assert code == "denied_calls", f"决策码不是 denied_calls：{code}"
    assert counter == 0, f"被拒后假工具仍被调用 {counter} 次"
    assert calls == 8, f"账本被多记：{calls}"
    return f"第 9 次返回 {result!r}；假工具计数 0；decision.code={code}"


def check_007() -> str:
    """A6：flag 开 + 费用将超上限 -> denied_cost，假工具 0 次。"""

    async def scenario() -> tuple[str, int, str]:
        from agent.adapters.tools.budget_memory import MemoryBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool
        from agent.domain.budget import BudgetSnapshot
        from agent.domain.tool import CostModel
        from agent.domain.tool_policy import PolicyContext, decide

        counter = 0

        async def fake_tool() -> str:
            nonlocal counter
            counter += 1
            return "ok"

        spec = _spec("search_tavily", cost_model=CostModel(unit="call", unit_cost=0.02))
        ledger = MemoryBudgetLedger()
        ledger.add("t63-a6", cost=0.04)
        wrapped = wrap_tool(fake_tool, spec, ledger)
        with tool_scope("t63-a6"):
            result = await wrapped()
        decision = decide(
            spec,
            PolicyContext(scope="t63-a6", max_calls=8, cost_limit=0.05),
            BudgetSnapshot(cost=0.04),
        )
        return result, counter, decision.code

    with _settings(ENABLE_TOOL_POLICY=True, TOOL_SESSION_COST_LIMIT=0.05):
        result, counter, code = asyncio.run(scenario())
    assert "费用" in result, f"没有费用上限文案：{result!r}"
    assert code == "denied_cost", f"决策码不是 denied_cost：{code}"
    assert counter == 0, f"被拒后假工具仍被调用 {counter} 次"
    return f"返回 {result!r}；假工具计数 0；decision.code={code}"


def check_008() -> str:
    """A7：flag 开 + deep_research -> approval_required，假实现 0 次。"""

    async def scenario() -> tuple[str, int, int, str]:
        from agent.adapters.tools.budget_memory import MemoryBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool
        from agent.domain.budget import BudgetSnapshot
        from agent.domain.tool_policy import PolicyContext, decide

        counter = 0

        async def fake_research() -> str:
            nonlocal counter
            counter += 1
            return "should-not-run"

        spec = _spec("deep_research")
        ledger = MemoryBudgetLedger()
        wrapped = wrap_tool(fake_research, spec, ledger)
        with tool_scope("t63-a7"):
            result = await wrapped()
        decision = decide(
            spec,
            PolicyContext(scope="t63-a7", max_calls=8, cost_limit=2.0),
            BudgetSnapshot(),
        )
        return result, counter, ledger.snapshot("t63-a7").calls, decision.code

    with _settings(ENABLE_TOOL_POLICY=True):
        result, counter, calls, code = asyncio.run(scenario())
    assert "审批" in result, f"没有审批文案：{result!r}"
    assert code == "approval_required", f"决策码不是 approval_required：{code}"
    assert counter == 0, f"审批被拒后假实现仍被调用 {counter} 次"
    assert calls == 0, f"审批拒绝不应记账：{calls}"
    return f"返回 {result!r}；假实现计数 0；decision.code={code}"


def check_009() -> str:
    """A8：工具抛异常 -> 可读错误字符串，不抛栈，账本记 1 次。"""

    async def scenario() -> tuple[str, int]:
        from agent.adapters.tools.budget_memory import MemoryBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool

        async def broken_tool() -> str:
            raise ValueError("boom")

        ledger = MemoryBudgetLedger()
        wrapped = wrap_tool(broken_tool, _spec("calculator"), ledger)
        with tool_scope("t63-a8"):
            result = await wrapped()
        return result, ledger.snapshot("t63-a8").calls

    with _settings(ENABLE_TOOL_POLICY=True):
        result, calls = asyncio.run(scenario())
    assert isinstance(result, str), f"返回值不是字符串：{type(result)!r}"
    assert "执行出错" in result and "ValueError" in result and "boom" in result, (
        f"错误文案不可读：{result!r}"
    )
    assert "Traceback" not in result, f"泄漏了调用栈：{result!r}"
    assert calls == 1, f"异常未记 1 次：{calls}"
    return f"返回 {result!r}；无 Traceback；calls={calls}"


def check_010() -> str:
    """A9：账本 3 次调用 -> calls=3、cost=3*unit_cost。"""

    async def scenario() -> tuple[int, float]:
        from agent.adapters.tools.budget_memory import MemoryBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool

        async def ok_tool() -> str:
            return "ok"

        ledger = MemoryBudgetLedger()
        wrapped = wrap_tool(ok_tool, _spec("search_tavily"), ledger)
        with tool_scope("t63-a9"):
            for _ in range(3):
                await wrapped()
        snap = ledger.snapshot("t63-a9")
        return snap.calls, snap.cost

    with _settings(ENABLE_TOOL_POLICY=True):
        calls, cost = asyncio.run(scenario())
    assert calls == 3, f"calls 不是 3：{calls}"
    assert abs(cost - 3 * 0.02) < 1e-9, f"cost 不是 3*0.02：{cost}"
    return f"calls={calls} cost={cost:.4f}（3 x unit_cost=0.02）"


def check_011() -> str:
    """A10：Redis 不可用（抛异常的假对象）-> 一次 warning + 退化 Noop，调用仍成功。"""

    async def scenario() -> tuple[str, str, int, float, int]:
        from agent.adapters.tools.budget_redis import RedisBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool

        class BrokenRedis:
            async def ping(self) -> bool:
                raise ConnectionError("redis down")

            async def hgetall(self, key: str) -> dict[str, Any]:
                raise ConnectionError("redis down")

            def pipeline(self, transaction: bool = False) -> Any:
                raise ConnectionError("redis down")

        logger = logging.getLogger("agent.adapters.tools.budget_redis")
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = Capture()
        old_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        counter = 0

        async def fine_tool() -> str:
            nonlocal counter
            counter += 1
            return "fine"

        # 故意注入抛异常的假客户端（类型上不是 redis.Redis）
        ledger = RedisBudgetLedger(
            client=BrokenRedis(),  # type: ignore[arg-type]
            ttl_seconds=60,
        )
        wrapped = wrap_tool(fine_tool, _spec("calculator"), ledger)
        try:
            with tool_scope("t63-a10"):
                first = await wrapped()
                second = await wrapped()
            snap = await ledger.snapshot("t63-a10")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

        warnings = [
            record
            for record in records
            if record.levelno == logging.WARNING and "降级" in record.getMessage()
        ]
        return first, second, counter, snap.calls, len(warnings)

    with _settings(ENABLE_TOOL_POLICY=True):
        first, second, counter, calls, warns = asyncio.run(scenario())
    assert first == "fine" and second == "fine", f"工具调用未成功：{first!r} {second!r}"
    assert counter == 2, f"工具应执行 2 次，实际 {counter}"
    assert warns == 1, f"降级 warning 不是 1 条：{warns}"
    assert calls == 0, f"降级后快照应为 0：calls={calls}"
    return f"两次调用均返回 'fine'；降级 warning={warns} 条；快照 calls={calls}（Noop 语义）"


# ── B 类：4 条不变量 ─────────────────────────────────────────────────


def check_012() -> str:
    """B1：src/tools 零改动；conversation_graph diff 仅 contextvar 相关行。

    第一条断言看 ``DEV_COMMIT`` 的历史（T6-3 开发提交本身没碰 src/tools）；
    第二条断言判的是**未提交**改动（``git status``），不是与 ``DEV_COMMIT`` 的历史
    差异——后者会因后续合法提交（如 c0fefec 改 offload.py）永久变红。
    """
    changed = _git("show", "--name-only", "--format=", DEV_COMMIT).stdout.splitlines()
    touched = [name for name in changed if name.startswith("src/tools/")]
    assert not touched, f"src/tools 被改动：{touched}"

    dirty = _git("status", "--porcelain", "--", "src/tools").stdout.splitlines()
    assert not dirty, f"工作区 src/tools 有未提交改动：{dirty}"

    diff = _git("show", DEV_COMMIT, "--", "src/conversation_graph.py").stdout
    added = [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    removed = [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    allowed_added = {
        "from agent.adapters.tools import executor, registry",
        "        with executor.tool_scope(thread_id):",
        "            result = await tool_node.ainvoke(state)",
    }
    allowed_removed = {
        "from agent.adapters.tools import registry",
        "        result = await tool_node.ainvoke(state)",
    }
    assert len(added) == 3, f"新增行数不是 3：{added}"
    assert set(added) <= allowed_added, f"出现 contextvar 之外的改动：{added}"
    assert len(removed) == 2, f"删除行数不是 2：{removed}"
    assert set(removed) <= allowed_removed, f"出现 contextvar 之外的删除：{removed}"
    assert "        thread_id = str(" in diff, "thread_id 取值行应保持为上下文（未改）"
    return (
        "src/tools 无未提交改动；conversation_graph +3/-2（import/with/result），thread_id 行未变"
    )


def check_013() -> str:
    """B2：lint-imports 3 kept；tool_policy/budget 无 redis/langchain import。"""
    lint = Path(sys.executable).with_name("lint-imports")
    assert lint.exists(), f"找不到 lint-imports：{lint}"
    proc = _run([str(lint)])
    combined = proc.stdout + proc.stderr
    assert "3 kept" in combined and "0 broken" in combined, (
        f"import-linter 不是 3 kept / 0 broken：{_last_line(combined)}"
    )

    banned = ("redis", "langchain", "langgraph")
    checked = [
        REPO / "agent" / "domain" / "tool_policy.py",
        REPO / "agent" / "domain" / "budget.py",
    ]
    offenders: list[str] = []
    for path in checked:
        for module in _imports_of(path):
            if module.split(".")[0] in banned:
                offenders.append(f"{path.name}:{module}")
    assert not offenders, f"domain 层出现框架/Redis 依赖：{offenders}"
    return "Contracts: 3 kept, 0 broken；2 个 domain 文件 0 条 redis/langchain/langgraph import"


def check_014() -> str:
    """B3：pytest / collect 与 `docs/baseline.md` §2 一致；新增脚本不被收集。"""
    exp = _baseline.expected()
    assert not Path(__file__).name.startswith("test_"), "检查脚本不能叫 test_*.py"
    collect = _run([sys.executable, "-m", "pytest", "--collect-only", "-q"])
    last_collect = _last_line(collect.stdout)
    assert exp["collect"] in last_collect, f"收集数不是 {exp['collect']}：{last_collect}"
    assert "4 deselected" in last_collect, f"deselected 不是 4：{last_collect}"
    assert "t63_checks" not in collect.stdout and "T63" not in collect.stdout, (
        "新增脚本被 pytest 收集了"
    )

    full = _run([sys.executable, "-m", "pytest", "-q"])
    last_full = _last_line(full.stdout)
    assert full.returncode == 0, f"pytest 退出码 {full.returncode}：{last_full}"
    ok, detail = _baseline.pytest_matches(last_full)
    assert ok, detail
    return f"collect {exp['collect']}；pytest {last_full}；t63_checks.py 未被收集"


def check_015() -> str:
    """B4：开发提交没有写 T63 台账行；工作区 T63 行只能由 test-G5 或 D15 授权行写入。"""
    commit_text = _git("show", f"{DEV_COMMIT}:tests/test_agent_cases/台账用例.csv").stdout
    assert "T63" not in commit_text, "开发提交里已经出现 T63 行"

    changed = _git("show", "--name-only", "--format=", DEV_COMMIT).stdout
    assert "台账用例.csv" not in changed, "开发提交改动了台账"

    rows = list(csv.reader(io.StringIO(LEDGER.read_text(encoding="gb18030"))))
    t63_rows = [row for row in rows if row and row[0].startswith("T63-")]
    # 例外：计划窗口 D15 授权行（run_by 非 test-G5，但 notes 带授权标记）。
    foreign = [
        row[0]
        for row in t63_rows
        if len(row) > 12
        and row[12] != "test-G5"
        and "计划窗口授权（D15）" not in (row[13] if len(row) > 13 else "")
    ]
    assert not foreign, f"T63 行不是本窗口或 D15 授权写的：{foreign}"
    return (
        f"dev 提交 T63 行 0、未改台账；工作区 T63 行 {len(t63_rows)}"
        "（run_by 为 test-G5 或 D15 授权）"
    )


# ── C 类：5 条边界 / 反例 ────────────────────────────────────────────


def check_016() -> str:
    """C1：同一 scope 两次并发，账本允许 +-1，但不得为 0 或翻倍。"""

    async def scenario() -> tuple[Any, int, int]:
        from agent.adapters.tools.budget_memory import MemoryBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool

        counter = 0

        async def slow_tool() -> str:
            nonlocal counter
            counter += 1
            await asyncio.sleep(0.05)
            return "ok"

        ledger = MemoryBudgetLedger()
        wrapped = wrap_tool(slow_tool, _spec("calculator", timeout_seconds=5.0), ledger)

        async def one() -> str:
            with tool_scope("t63-c1"):
                return await wrapped()

        results = await asyncio.gather(one(), one())
        return results, counter, ledger.snapshot("t63-c1").calls

    with _settings(ENABLE_TOOL_POLICY=True, TOOL_MAX_CALLS_PER_TURN=8):
        results, counter, calls = asyncio.run(scenario())
    assert results == ["ok", "ok"], f"结果异常：{results}"
    assert counter == 2, f"并发两次应都执行，实际 {counter}"
    assert 1 <= calls <= 3, f"账本 calls={calls}，超出 2 的 +-1 容差"
    return f"2 并发 -> 工具执行 {counter} 次；账本 calls={calls}（容差 1..3，未为 0 未翻倍）"


def check_017() -> str:
    """C2：wrap_tool 包装后 signature / __name__ / __doc__ 未变。"""
    from agent.adapters.tools import registry

    with _settings(ENABLE_TOOL_POLICY=True):
        tools = registry.to_lc_tools()
        origins = [registry._resolve(name) for name in registry.TOOL_SPECS]

    problems: list[str] = []
    for tool, origin in zip(tools, origins, strict=True):
        same_object = tool is origin
        sig_ok = inspect.signature(tool) == inspect.signature(origin)
        name_ok = tool.__name__ == origin.__name__
        doc_ok = tool.__doc__ == origin.__doc__
        if same_object or not sig_ok or not name_ok or not doc_ok:
            problems.append(
                f"{origin.__name__}: is={same_object} sig={sig_ok} name={name_ok} doc={doc_ok}"
            )
    assert not problems, f"包装改变了元数据：{problems}"
    return "5/5 signature/__name__/__doc__ 一致（包装后 is 不同）"


def check_018() -> str:
    """C3：超时后 asyncio.all_tasks() 里没有残留的假工具任务。"""

    async def scenario() -> tuple[str, int]:
        from agent.adapters.tools.budget_memory import MemoryBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool

        async def slow_tool() -> str:
            await asyncio.sleep(1.0)
            return "never"

        ledger = MemoryBudgetLedger()
        wrapped = wrap_tool(slow_tool, _spec("calculator", timeout_seconds=0.1), ledger)
        before = set(asyncio.all_tasks())
        with tool_scope("t63-c3"):
            result = await wrapped()
        await asyncio.sleep(0)
        residual = set(asyncio.all_tasks()) - before
        return result, len(residual)

    with _settings(ENABLE_TOOL_POLICY=True):
        result, residual = asyncio.run(scenario())
    assert "超时" in result, f"没有超时文案：{result!r}"
    assert residual == 0, f"超时后残留 {residual} 个任务"
    return f"超时返回 {result!r}；all_tasks 新增残留任务 {residual}"


def check_019() -> str:
    """C4：TOOL_SESSION_COST_LIMIT=0 时第一次调用就被 deny（边界）。"""

    async def scenario() -> tuple[str, int, str, int]:
        from agent.adapters.tools.budget_memory import MemoryBudgetLedger
        from agent.adapters.tools.executor import tool_scope, wrap_tool

        counter = 0

        async def fake_tool() -> str:
            nonlocal counter
            counter += 1
            return "ok"

        ledger = MemoryBudgetLedger()
        wrapped = wrap_tool(fake_tool, _spec("search_tavily"), ledger)
        with tool_scope("t63-c4"):
            denied = await wrapped()

        boundary_counter = 0

        async def boundary_tool() -> str:
            nonlocal boundary_counter
            boundary_counter += 1
            return "ok"

        boundary_ledger = MemoryBudgetLedger()
        boundary = wrap_tool(boundary_tool, _spec("search_tavily"), boundary_ledger)
        with tool_scope("t63-c4-equal"):
            allowed = await boundary()
        return denied, counter, allowed, boundary_counter

    with _settings(ENABLE_TOOL_POLICY=True, TOOL_SESSION_COST_LIMIT=0.0):
        denied, counter, _, _ = asyncio.run(scenario())
    with _settings(ENABLE_TOOL_POLICY=True, TOOL_SESSION_COST_LIMIT=0.02):
        _, _, allowed, boundary_counter = asyncio.run(scenario())
    assert "费用" in denied, f"limit=0 首次没有费用拒绝文案：{denied!r}"
    assert counter == 0, f"limit=0 时假工具仍被调用 {counter} 次"
    assert allowed == "ok" and boundary_counter == 1, (
        f"恰好等于上限的边界应允许：{allowed!r} counter={boundary_counter}"
    )
    return f"limit=0 首次返回 {denied!r}、假工具 0 次；limit=0.02 恰好等值 -> 允许（边界）"


def check_020() -> str:
    """C5：flag 关 + 工具抛异常 -> 与改造前一致，异常原样冒泡。"""

    async def boom() -> str:
        raise ValueError("boom-off")

    with _settings(ENABLE_TOOL_POLICY=False):
        from agent.adapters.tools.executor import wrap_tool

        wrapped = wrap_tool(boom, _spec("calculator"))
        assert wrapped is boom, "flag 关时 wrap_tool 必须原样返回原函数"
        try:
            asyncio.run(wrapped())
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("flag 关时异常被吞掉了")

    assert message == "boom-off", f"异常内容被改写：{message!r}"
    return "flag 关：wrap_tool 返回原函数（is）；ValueError('boom-off') 原样冒泡"


# ── 入口 ─────────────────────────────────────────────────────────────


def main() -> int:
    checker = Checker()
    checker.run("T63-001", "A1 flag 关 to_lc_tools 与原函数同一对象", check_001)
    checker.run("T63-002", "A2 flag 关 schema 哈希与字节数不变", check_002)
    checker.run("T63-003", "A2 flag 关全量 pytest 223/4/4", check_003)
    checker.run("T63-004", "A3 flag 开 read 工具执行且账本 calls=1", check_004)
    checker.run("T63-005", "A4 超时返回可读错误且无 Traceback", check_005)
    checker.run("T63-006", "A5 第 9 次调用 denied_calls 且假工具 0 次", check_006)
    checker.run("T63-007", "A6 费用将超上限 denied_cost 且假工具 0 次", check_007)
    checker.run("T63-008", "A7 deep_research 审批拒绝且假实现 0 次", check_008)
    checker.run("T63-009", "A8 工具异常归一为可读字符串且记账 1 次", check_009)
    checker.run("T63-010", "A9 3 次调用 calls=3 cost=3xunit", check_010)
    checker.run("T63-011", "A10 Redis 不可用降级 Noop 且 warning 一次", check_011)
    checker.run("T63-012", "B1 src/tools 零改动且 graph 仅 contextvar", check_012)
    checker.run("T63-013", "B2 lint-imports 3 kept 且 domain 无框架依赖", check_013)
    checker.run("T63-014", "B3 pytest 223 且 collect 227/231", check_014)
    checker.run("T63-015", "B4 开发提交未写 T63 台账行", check_015)
    checker.run("T63-016", "C1 同 scope 两次并发账本在容差内", check_016, c_class=True)
    checker.run("T63-017", "C2 包装后签名/名称/文档未变", check_017, c_class=True)
    checker.run("T63-018", "C3 超时后无残留 asyncio 任务", check_018, c_class=True)
    checker.run("T63-019", "C4 费用上限 0 时首次即拒绝", check_019, c_class=True)
    checker.run("T63-020", "C5 flag 关异常原样冒泡", check_020, c_class=True)

    if checker.issues:
        print(f"C-FINDINGS: {len(checker.issues)} issue")
    print(f"RESULT: {checker.passed} passed, {checker.failed} failed")
    code = 0 if checker.failed == 0 else 1
    print(f"EXIT={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
