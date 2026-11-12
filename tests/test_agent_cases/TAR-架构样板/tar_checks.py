#!/usr/bin/env python3
"""G10 / T-arch 架构样板 —— 测试窗口（test-G10）独立验收脚本。

用法（仓库根目录，任意 cwd 均可）：

    export PRE_COMMIT_HOME="$PWD/.pre-commit-cache"
    poetry run python "tests/test_agent_cases/TAR-架构样板/tar_checks.py"

覆盖（A/B 为判分必过项，C 只记 open_issue、不判失败）：

    TAR-001  A1 七个端口齐全、均 @runtime_checkable、docstring 三段契约
    TAR-002  A2 RuntimeEvent.kind 只有六值；ping 不是 RuntimeEvent
    TAR-003  A3 isinstance(LangGraphRuntime, AgentRuntime)；仅用 Fake 跑通 run_turn
    TAR-004  A4 ports.py 与 agent/domain 不 import langchain/langgraph
    TAR-005  A5 端口没有 TokenCounter / UserProfileStore
    TAR-006  A6 architecture.md 四段齐全（映射表/依赖方向/策略上移/被砍项）
    TAR-007  B1 pytest 225 passed、collect-only 229/233
    TAR-008  B2 nodeids sha256 与 baseline.md 新值 8c3cab4d… 一致
    TAR-009  B3 baseline.md 写明"因新增契约测试（N=2）而变"
    TAR-010  B4 旧值 a57008dd… 仍保留（历史对比）
    TAR-011  B5 _baseline.py 解析出的三个值与实测一致（0a 自验证）
    TAR-012  C1 LangGraphRuntime.resume 是否重复实现 G7 凭据存储
    TAR-013  C2 smoke_tools_schema.py 双指纹不一致时是否醒目提示
    TAR-014  0a/0b 修 OI-P-008（t11/t93 改读 baseline.md）+ 契约测试 N=2
    TAR-015  C3/C4 端口方法参数数量；_baseline 对坏 baseline.md 是否报可读错误

退出码：
    0 = 15 条检查全部 PASS；1 = 存在 FAIL。逐条打印 `[PASS]/[FAIL] 名称 — 证据`，
    末行固定为 `RESULT: n passed, m failed`。

只读承诺：
    不写业务代码、不改 docs/、不改 tests/contracts/；唯一副作用是 pytest 自己的
    `.pytest_cache/`（已 gitignore）。C 类发现只写进证据与台账 open_issue，不影响退出码。

为什么不是 `test_*.py`：
    一旦被 pytest 收集，`docs/baseline.md` §2 的收集数与 nodeids 哈希立刻失效
    （当前 229/233、230 行），本卡自己的 B1/B5 就会红。详见
    `tests/test_agent_cases/README.md` §2。

编码：
    本文件 UTF-8；台账 `台账用例.csv` 是 GB18030，两者不要混用。
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import re
import subprocess
import sys
import typing
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parent
ROOT = HERE.parents[2]
for _path in (str(ROOT), str(CASE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import _baseline  # noqa: E402

PORTS = [
    "AgentRuntime",
    "ModelClient",
    "SessionStore",
    "ArtifactStore",
    "TaskStore",
    "MemoryStore",
    "EventBus",
    "BudgetLedger",
]
SIX_KINDS = {"token", "tool_call", "tool_result", "interrupt", "final", "error"}
CONTRACT_TEST = ROOT / "tests" / "contracts" / "test_agent_runtime_contract.py"
OLD_NODEIDS_SHA = "a57008dd1a22cb16ff5096dda448dcee641f266ce63786dc96eb7bfe25657334"
NEW_NODEIDS_PREFIX = "8c3cab4d"

#: B1 实测结果，供 B5 复用（避免重复跑 pytest）
_MEASURED: dict[str, Any] = {}


# ── 通用工具 ──────────────────────────────────────────────────────────


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _decorator_names(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in node.decorator_list:
        if isinstance(dec, ast.Call):
            dec = dec.func
        if isinstance(dec, ast.Attribute):
            names.add(dec.attr)
        elif isinstance(dec, ast.Name):
            names.add(dec.id)
    return names


def _protocol_classes() -> dict[str, ast.ClassDef]:
    tree = ast.parse(_read(ROOT / "agent" / "app" / "ports.py"))
    result: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
            for base in node.bases
        }
        if "Protocol" in bases:
            result[node.name] = node
    return result


def _pytest(args: list[str], timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _last_line(output: str, needle: str) -> str:
    for line in reversed(output.splitlines()):
        if needle in line:
            return line.strip()
    return "未找到统计行"


# ── A 类：端口与适配器 ────────────────────────────────────────────────


def check_a1() -> tuple[bool, str]:
    """八个端口齐全、都是 @runtime_checkable、docstring 有三段契约。"""
    classes = _protocol_classes()
    missing = [name for name in PORTS if name not in classes]
    extra = sorted(set(classes) - set(PORTS))
    no_checkable = [
        name
        for name in PORTS
        if name in classes and "runtime_checkable" not in _decorator_names(classes[name])
    ]
    no_sections: dict[str, list[str]] = {}
    for name in PORTS:
        if name not in classes:
            continue
        doc = ast.get_docstring(classes[name]) or ""
        lack = [section for section in ("幂等性", "错误语义", "并发安全") if section not in doc]
        if lack:
            no_sections[name] = lack
    ok = not missing and not extra and not no_checkable and not no_sections
    total = len(PORTS)
    evidence = (
        f"Protocol {len(classes)} 个（期望 {total}）；@runtime_checkable "
        f"{total - len(no_checkable)}/{total}；三段契约 {total - len(no_sections)}/{total}"
    )
    if missing:
        evidence += f"；缺端口 {missing}"
    if extra:
        evidence += f"；多出 {extra}"
    if no_sections:
        evidence += f"；缺段 {no_sections}"
    return ok, evidence


def check_a2() -> tuple[bool, str]:
    """RuntimeEvent.kind 只有六值；ping 不出现在 RuntimeEvent 里。"""
    from agent.app.ports import RuntimeEvent, RuntimeKind

    kinds = set(typing.get_args(RuntimeKind))
    fields = [field.name for field in dataclasses.fields(RuntimeEvent)]
    ok = kinds == SIX_KINDS and "ping" not in kinds and fields == ["kind", "payload"]
    evidence = (
        f"kind={sorted(kinds)}（六值={kinds == SIX_KINDS}）；"
        f"ping 不在={'ping' not in kinds}；字段={fields}"
    )
    return ok, evidence


def check_a3() -> tuple[bool, str]:
    """isinstance(LangGraphRuntime, AgentRuntime)；仅用 Fake 就能跑通 run_turn。"""
    from agent.adapters.runtime.langgraph_runtime import LangGraphRuntime
    from agent.app.ports import AgentRuntime, RuntimeEvent

    real = LangGraphRuntime(graph=object(), heartbeat_seconds=0.01)
    real_ok = isinstance(real, AgentRuntime)

    class _FakeRuntime:
        """最小内存实现：不 import 任何框架，证明端口是结构化的。"""

        async def run_turn(self, session: Any, hooks: Any | None = None):
            yield RuntimeEvent(kind="token", payload={"content": "你"})
            yield RuntimeEvent(kind="final", payload={"content": "你好"})

        async def resume(self, session: Any, command: Any):
            yield RuntimeEvent(kind="final", payload={"content": "恢复完成"})

    fake = _FakeRuntime()
    fake_isinstance = isinstance(fake, AgentRuntime)

    async def _collect() -> list[RuntimeEvent]:
        return [event async for event in fake.run_turn({"thread_id": "tar"})]

    events = asyncio.run(_collect())
    run_ok = (
        bool(events)
        and all(isinstance(event, RuntimeEvent) for event in events)
        and events[-1].kind == "final"
    )
    ok = real_ok and fake_isinstance and run_ok
    evidence = (
        f"LangGraphRuntime isinstance={real_ok}；Fake isinstance={fake_isinstance}；"
        f"Fake.run_turn 事件={[event.kind for event in events]}"
    )
    return ok, evidence


def check_a4() -> tuple[bool, str]:
    """agent/app/ports.py 与 agent/domain 不 import langchain/langgraph。"""
    files = [ROOT / "agent" / "app" / "ports.py", *sorted((ROOT / "agent" / "domain").glob("*.py"))]
    hits: list[str] = []
    for path in files:
        tree = ast.parse(_read(path), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.startswith(("langchain", "langgraph")):
                    hits.append(f"{path.name}:{module}")
    ok = not hits
    evidence = f"扫 {len(files)} 个文件（ports.py + domain/*.py）；框架 import {len(hits)} 处"
    if hits:
        evidence += f"；{hits}"
    return ok, evidence


def check_a5() -> tuple[bool, str]:
    """端口里没有 TokenCounter、没有 UserProfileStore。"""
    classes = _protocol_classes()
    tree = ast.parse(_read(ROOT / "agent" / "app" / "ports.py"))
    exported: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
            and isinstance(node.value, ast.List)
        ):
            exported = {
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
    forbidden = {"TokenCounter", "UserProfileStore"}
    hits = sorted((forbidden & set(classes)) | (forbidden & exported))
    ok = not hits
    evidence = f"端口 {len(classes)} 个；TokenCounter/UserProfileStore 命中 {hits or '0'}"
    return ok, evidence


def check_a6() -> tuple[bool, str]:
    """architecture.md 四段齐全：映射表 / 依赖方向 / 循环留策略上移 / 被砍项。"""
    text = _read(ROOT / "docs" / "architecture.md")
    required = {
        "端口映射表": "## 端口清单与实现映射",
        "依赖方向": "## 依赖方向",
        "循环留策略上移": '## 为什么"循环留在适配器、策略上移到应用层"',
        "被砍项与理由": "## 被砍项与取舍",
    }
    missing = [label for label, token in required.items() if token not in text]
    mapped = [name for name in PORTS if re.search(rf"^\|\s*`{name}`\s*\|", text, re.MULTILINE)]
    ok = not missing and len(mapped) == len(PORTS)
    evidence = f"四段 {len(required) - len(missing)}/4；端口映射行 {len(mapped)}/7"
    if missing:
        evidence += f"；缺 {missing}"
    return ok, evidence


# ── B 类：基线一致性 ──────────────────────────────────────────────────


def check_b1() -> tuple[bool, str]:
    """pytest = 225、--collect-only = 229/233。"""
    exp = _baseline.expected()
    collect_code, collect_out = _pytest(["--collect-only", "-q"], timeout=600)
    collect_line = _last_line(collect_out, "collected")
    ratio_match = re.search(r"(\d+/\d+) tests collected", collect_line)
    ratio = ratio_match.group(1) if ratio_match else "未匹配"

    full_code, full_out = _pytest(["-q"], timeout=900)
    summary = _last_line(full_out, " passed")
    passed_match = re.search(r"(\d+) passed", summary)
    passed = int(passed_match.group(1)) if passed_match else -1

    _MEASURED.update({"summary": summary, "passed": passed, "collect": ratio})
    ok = (
        collect_code == 0
        and ratio == exp["collect"]
        and full_code == 0
        and passed == exp["pytest_passed"]
    )
    evidence = (
        f"pytest 末行「{summary}」；collect 末行「{collect_line}」"
        f"（baseline.md: {exp['pytest_passed']} passed / {exp['collect']}）"
    )
    return ok, evidence


def check_b2() -> tuple[bool, str]:
    """nodeids sha256 与 baseline.md 新值 8c3cab4d… 一致。"""
    exp = _baseline.expected()
    try:
        digest = _baseline.assert_nodeids_sha()
    except _baseline.BaselineError as exc:
        return False, f"nodeids 校验失败：{exc}"
    prefix_ok = digest.startswith(NEW_NODEIDS_PREFIX)
    ok = digest == exp["nodeids_sha256"] and prefix_ok
    evidence = (
        f"实测 sha256={digest}（{_baseline.expected()['nodeids_lines']} 行）；"
        f"baseline.md={exp['nodeids_sha256']}；新值前缀 {NEW_NODEIDS_PREFIX}= {prefix_ok}"
    )
    return ok, evidence


def check_b3() -> tuple[bool, str]:
    """baseline.md 写明"因新增契约测试（N=2 例）而变"。"""
    text = _read(ROOT / "docs" / "baseline.md")
    match = re.search(r"## 2\. 测试基线(.*?)(?:\n## |\Z)", text, re.DOTALL)
    section = match.group(1) if match else ""
    has_contract = "契约测试" in section
    has_n2 = bool(re.search(r"N\s*=\s*2|\*\*2\*\*\s*例|2\s*例", section))
    has_numbers = all(number in section for number in ("225", "229", "227", "223"))
    ok = has_contract and has_n2 and has_numbers
    evidence = (
        f"§2 含契约测试={has_contract}；N=2 例={has_n2}；"
        f"新旧数字 223/225/227/229 齐全={has_numbers}"
    )
    return ok, evidence


def check_b4() -> tuple[bool, str]:
    """旧值 a57008dd… 没有被删除（保留历史对比）。"""
    text = _read(ROOT / "docs" / "baseline.md")
    ok = OLD_NODEIDS_SHA in text
    position = text.find(OLD_NODEIDS_SHA)
    context = (
        text[max(0, position - 40) : position + 12].replace("\n", " ") if position >= 0 else ""
    )
    evidence = f"旧 sha256 保留={ok}；上下文「{context}」"
    return ok, evidence


def check_b5() -> tuple[bool, str]:
    """_baseline.py 解析出的三个值与实测一致（0a 的自我验证）。"""
    exp = _baseline.expected()
    if not _MEASURED:
        return False, "B1 未先运行，无法比较实测值"
    try:
        digest = _baseline.assert_nodeids_sha()
    except _baseline.BaselineError as exc:
        return False, f"nodeids 校验失败：{exc}"
    pytest_ok = exp["pytest"] in _MEASURED["summary"]
    collect_ok = exp["collect"] == _MEASURED["collect"]
    sha_ok = digest == exp["nodeids_sha256"]
    ok = pytest_ok and collect_ok and sha_ok
    evidence = (
        f"解析 pytest「{exp['pytest']}」/ collect「{exp['collect']}」/ "
        f"sha256 {digest[:8]}…；与实测一致 pytest={pytest_ok} "
        f"collect={collect_ok} sha={sha_ok}"
    )
    return ok, evidence


# ── C 类：边界与反例（只记 open_issue，不判失败）─────────────────────


def check_c1() -> tuple[bool, str]:
    """C1：resume 是否复用 G7 凭据存储（重复实现就是隐患）。"""
    runtime_src = _read(ROOT / "agent" / "adapters" / "runtime" / "langgraph_runtime.py")
    hitl_src = _read(ROOT / "agent" / "adapters" / "api" / "hitl.py")
    duplicate = "approval_store" in runtime_src
    uses_command = "Command(resume=" in runtime_src
    hitl_owns = "approval_store" in hitl_src and ".consume(" in hitl_src
    ok = not duplicate and uses_command and hitl_owns
    evidence = (
        f"resume 走 Command(resume=…)={uses_command}；运行时适配器 import approval_store="
        f"{duplicate}；凭据消费在 hitl.py={hitl_owns}；"
        "未重复实现；直接调用适配器时需调用方先校验（OI-TAR-001）"
    )
    return ok, evidence


def check_c2() -> tuple[bool, str]:
    """C2：双指纹不一致时 smoke 脚本是否有醒目提示。"""
    smoke = _read(ROOT / "scripts" / "smoke_tools_schema.py")
    has_diff = "status: DIFF" in smoke
    has_only = "snapshot_only" in smoke and "current_only" in smoke
    ok = has_diff and has_only
    evidence = (
        f"status: DIFF 行={has_diff}；snapshot_only/current_only 差异清单={has_only}；"
        "开关打开时允许不等并逐项打印（文本提示，非 stderr）"
    )
    return ok, evidence


def check_c3() -> tuple[bool, str]:
    """C3：端口方法参数超过 5 个的（该打包 Context/Options）。"""
    worst = ""
    max_params = 0
    for node in ast.walk(ast.parse(_read(ROOT / "agent" / "app" / "ports.py"))):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        count = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
        if count:
            count -= 1  # self
        if count > max_params:
            max_params = count
            worst = node.name
    ok = max_params <= 5
    evidence = f"端口方法最大参数 {max_params} 个（{worst} 或并列）；>5 的 0 个"
    if not ok:
        evidence = f"端口方法最大参数 {max_params} 个（{worst}），超过 5 个（OI-TAR-002）"
    return ok, evidence


def check_c4() -> tuple[bool, str]:
    """C4：baseline.md 被改坏时 _baseline 是否抛可读错误。"""
    text = _read(ROOT / "docs" / "baseline.md")
    broken_heading = text.replace("## 2. 测试基线", "## 2. 基线（临时改坏）", 1)
    broken_nodeids = "\n".join(
        line for line in text.splitlines() if ".pytest_cache/v/cache/nodeids" not in line
    )
    probes: list[str] = []
    ok = True
    for label, broken in (("坏标题", broken_heading), ("缺 nodeids 行", broken_nodeids)):
        try:
            _baseline.parse(broken)
        except _baseline.BaselineError as exc:
            readable = "baseline.md 结构变了，改解析器" in str(exc)
            ok = ok and readable
            probes.append(f"{label} -> BaselineError（可读={readable}）")
        except Exception as exc:  # noqa: BLE001 - 反例探针要看见所有异常类型
            ok = False
            probes.append(f"{label} -> {type(exc).__name__}（不是 BaselineError）")
        else:
            ok = False
            probes.append(f"{label} -> 未抛错（静默通过）")
    return ok, "；".join(probes)


def check_tar014() -> tuple[bool, str]:
    """0a/0b：t11/t93 改读 baseline.md；契约测试确实挂了 2 个实现。"""
    t11 = _read(CASE_ROOT / "T11-三层骨架+依赖规则" / "t11_checks.py")
    t93 = _read(CASE_ROOT / "T93-MCP工具" / "t93_checks.py")
    t11_forbidden = [
        name for name in ("BASELINE_PASSED", "BASELINE_COLLECT", "BASELINE_NODEIDS") if name in t11
    ]
    t93_forbidden = [
        name
        for name in ("EXPECTED_PASSED", "EXPECTED_COLLECT", "NODEIDS_SHA256", "NODEIDS_LINES")
        if name in t93
    ]
    imports = "import _baseline" in t11 and "import _baseline" in t93
    stale_sha = OLD_NODEIDS_SHA in t11 or OLD_NODEIDS_SHA in t93

    contract = _read(CONTRACT_TEST)
    parametrized: list[int] = []
    for node in ast.walk(ast.parse(contract)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            is_parametrize = (isinstance(func, ast.Attribute) and func.attr == "parametrize") or (
                isinstance(func, ast.Name) and func.id == "parametrize"
            )
            if is_parametrize and len(dec.args) >= 2 and isinstance(dec.args[1], ast.List):
                parametrized.append(len(dec.args[1].elts))
    n_impls = max(parametrized) if parametrized else 0
    has_both = "LangGraphRuntime" in contract and "_FakeRuntime" in contract

    ok = (
        imports
        and not t11_forbidden
        and not t93_forbidden
        and not stale_sha
        and n_impls >= 2
        and has_both
    )
    evidence = (
        f"t11/t93 import _baseline={imports}；t11 残留硬编码={t11_forbidden or '无'}；"
        f"t93 残留硬编码={t93_forbidden or '无'}；旧 sha 残留={stale_sha}；"
        f"契约测试 parametrize N={n_impls}（LangGraphRuntime + Fake={has_both}）"
    )
    return ok, evidence


CHECKS: list[tuple[str, Any]] = [
    ("TAR-001 A1 八端口齐全 + @runtime_checkable + 三段契约", check_a1),
    ("TAR-002 A2 RuntimeEvent.kind 六值 / ping 不入事件", check_a2),
    ("TAR-003 A3 LangGraphRuntime 满足端口 / Fake 跑通 run_turn", check_a3),
    ("TAR-004 A4 ports.py 与 domain 零框架依赖", check_a4),
    ("TAR-005 A5 端口无 TokenCounter / UserProfileStore", check_a5),
    ("TAR-006 A6 architecture.md 四段齐全", check_a6),
    ("TAR-007 B1 pytest 225 / collect-only 229/233", check_b1),
    ("TAR-008 B2 nodeids 哈希与 baseline.md 新值一致", check_b2),
    ("TAR-009 B3 baseline.md 写明因新增契约测试而变", check_b3),
    ("TAR-010 B4 旧 nodeids 值保留（历史对比）", check_b4),
    ("TAR-011 B5 _baseline 解析值与实测一致", check_b5),
    ("TAR-012 C1 resume 未重复实现 G7 凭据存储", check_c1),
    ("TAR-013 C2 smoke 双指纹 DIFF 有醒目提示", check_c2),
    ("TAR-014 0a/0b 修 OI-P-008 + 契约测试 N=2", check_tar014),
    ("TAR-015 C3/C4 端口参数数量 + baseline 坏输入报错", lambda: _combine_c3_c4()),
]


def _combine_c3_c4() -> tuple[bool, str]:
    c3_ok, c3_evidence = check_c3()
    c4_ok, c4_evidence = check_c4()
    return c3_ok and c4_ok, f"C3 {c3_evidence}；C4 {c4_evidence}"


def main() -> int:
    passed = 0
    failed = 0
    print(f"仓库：{ROOT}")
    print(f"基线文件：{_baseline.BASELINE_PATH}")
    print("-" * 78, flush=True)
    for name, check in CHECKS:
        try:
            ok, evidence = check()
        except Exception as exc:  # noqa: BLE001 - 单条异常不应吞掉其余检查
            ok, evidence = False, f"异常 {type(exc).__name__}: {exc}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name} — {evidence}", flush=True)
        if ok:
            passed += 1
        else:
            failed += 1
    print("-" * 78)
    print(f"RESULT: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
