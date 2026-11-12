#!/usr/bin/env python3
"""T1-1 验收脚本：三层骨架 + 依赖规则（T11-001 ~ T11-010）。

用法：
    cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent
    poetry run python "tests/test_agent_cases/T11-三层骨架+依赖规则/t11_checks.py"

退出码：
    0 = 10 条用例全部 PASS；1 = 存在 FAIL。逐条打印 `[PASS]/[FAIL] 用例名 — 证据`，
    末行固定为 `RESULT: n passed, m failed`。

为什么不是 test_*.py：
    `pytest --collect-only` 会把 `test_*.py` / `*_test.py` 全部收集。本目录的卡级检查脚本
    一旦被收进去，`docs/baseline.md` §2 的收集数与 nodeids 哈希立刻失效（当前基线
    229/233、230 行），每张卡"开工前三件事"第 3 条就过不了；且验收脚本带着环境依赖
    （poetry / 真实仓库路径）混进单测，CI 上必挂。详见 `tests/test_agent_cases/README.md` §2。

基线来源：
    T11-009 / T11-010 不再硬编码数字，统一读 `tests/test_agent_cases/_baseline.py`
    （解析 `docs/baseline.md` §2）；断言语义是"与本卡基线文件一致"，改基线只改一处。

编码：
    本文件 UTF-8；同目录台账 `台账用例.csv` 是 GB18030，两者不要混用。

副作用与还原：
    T11-002 / T11-003 会**临时注入违规**（验证 lint 真的拦得住），均在 try/finally 中还原：
    T11-002 先把 `agent/domain/__init__.py` 原始字节存起来，测完写回并校验字节一致；
    T11-003 新增临时探针 `agent/app/_t11_violation.py`，测完删除并清 __pycache__。
    两条都断言 `git status --short -- <路径>` 为空，确保工作区无残留。其余用例只读。
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AGENT = ROOT / "agent"

#: 卡级基线的唯一事实来源是 `docs/baseline.md` §2（见 `_baseline.py`）。
CASE_ROOT = HERE.parent
if str(CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CASE_ROOT))

import _baseline  # noqa: E402 - 必须先把 tests/test_agent_cases 加进 sys.path

CONTRACTS = ["core-no-framework", "app-independent", "adapters-only-ports"]
PORTS = [
    "AgentRuntime",
    "ModelClient",
    "SessionStore",
    "ArtifactStore",
    "TaskStore",
    "MemoryStore",
    "EventBus",
]
DOMAIN_EXPECTED = {
    "__init__.py",
    "tool.py",
    "task.py",
    "roles.py",
    "context.py",
    "message.py",
    "memory.py",
    "session.py",
}
APP_EXPECTED = {"__init__.py", "ports.py", "hooks.py"}
ADAPTER_SUBPKGS = ["runtime", "models", "stores", "locks", "tasks", "events", "api", "tools"]
ARCH_TOKENS = ["## 端口清单", "## 依赖方向", "## 组合根", "待启用规则"]

#: 已知的 domain 内 Protocol 例外：T6-3 的预算账本（不是六边形端口）。
#: 新增任何 domain Protocol 都会红，必须先把例外登记到这里并说明理由。
KNOWN_DOMAIN_PROTOCOLS = {"budget.py:BudgetLedger"}

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _clean(text: str) -> str:
    """去掉 lint-imports 输出的 ANSI 颜色码，便于比对。"""
    return ANSI.sub("", text)


def _run(args: list[str], timeout: int = 600) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, _clean((proc.stdout or "") + (proc.stderr or ""))


def _lint_cmd() -> list[str]:
    """优先按文档口径走 `poetry run lint-imports`，退路用当前解释器所在环境的脚本。"""
    if shutil.which("poetry"):
        return ["poetry", "run", "lint-imports"]
    exe = Path(sys.executable).parent / "lint-imports"
    if exe.exists():
        return [str(exe)]
    return [sys.executable, "-m", "importlinter.cli"]


def _pytest_cmd(*args: str) -> list[str]:
    if shutil.which("poetry"):
        return ["poetry", "run", "pytest", *args]
    return [sys.executable, "-m", "pytest", *args]


def _last_summary(output: str) -> str:
    """取 pytest 的统计末行（形如 `225 passed, 4 skipped ... in 38.8s`）。"""
    for line in reversed(output.splitlines()):
        if " passed" in line and " in " in line:
            return line.strip()
    return "未找到统计行"


def check_001() -> tuple[bool, str]:
    """lint-imports 退出码 0，且输出含三条 contract 名。"""
    code, out = _run(_lint_cmd())
    missing = [name for name in CONTRACTS if name not in out]
    summary = "Contracts: 3 kept, 0 broken." in out
    ok = code == 0 and not missing and summary
    evidence = f"EXIT={code}；contract 名 {3 - len(missing)}/3；3 kept 0 broken={'是' if summary else '否'}"
    if missing:
        evidence += f"；缺 {missing}"
    return ok, evidence


def check_002() -> tuple[bool, str]:
    """违规必拦：domain 注入 langchain_core 后 lint 非 0 且指向该文件。"""
    target = AGENT / "domain" / "__init__.py"
    rel = "agent/domain/__init__.py"
    original = target.read_bytes()
    code = -1
    out = ""
    try:
        target.write_bytes(original + b"\nimport langchain_core\n")
        code, out = _run(_lint_cmd())
    finally:
        target.write_bytes(original)

    restored = target.read_bytes() == original
    git_code, git_out = _run(["git", "status", "--short", "--", rel])
    clean = git_code == 0 and git_out.strip() == ""
    points = "agent.domain" in out and "langchain_core" in out and "core-no-framework" in out
    ok = code != 0 and points and restored and clean
    return ok, (
        f"注入后 EXIT={code}；指向 {rel}={'是' if points else '否'}；"
        f"还原字节一致={restored}；git status 该文件为空={clean}"
    )


def check_003() -> tuple[bool, str]:
    """app 越界必拦：app 下临时 import agent.adapters 后 lint 非 0，随后还原。"""
    probe = AGENT / "app" / "_t11_violation.py"
    code = -1
    out = ""
    try:
        probe.write_text(
            '"""临时违规探针（T11-003），脚本结束即删除。"""\n\n'
            "import agent.adapters  # noqa: F401\n",
            encoding="utf-8",
        )
        code, out = _run(_lint_cmd())
    finally:
        probe.unlink(missing_ok=True)
        shutil.rmtree(AGENT / "app" / "__pycache__", ignore_errors=True)

    gone = not probe.exists()
    git_code, git_out = _run(["git", "status", "--short", "--", "agent/app"])
    clean = git_code == 0 and git_out.strip() == ""
    points = "_t11_violation" in out and "app-independent" in out
    ok = code != 0 and points and gone and clean
    return ok, (
        f"注入后 EXIT={code}；输出指向探针={'是' if points else '否'}；"
        f"探针已删除={gone}；git status agent/app 为空={clean}"
    )


def check_004() -> tuple[bool, str]:
    """agent/domain 下无端口类、无 `@runtime_checkable`；Protocol 只允许已登记例外。"""
    files = sorted((AGENT / "domain").glob("*.py"))
    protocols: set[str] = set()
    port_classes: list[str] = []
    checkable: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                for base in node.bases
            }
            if "Protocol" in bases:
                protocols.add(f"{path.name}:{node.name}")
            if re.fullmatch(r"\w*Port\w*", node.name):
                port_classes.append(f"{path.name}:{node.name}")
            decorators = {
                dec.attr if isinstance(dec, ast.Attribute) else getattr(dec, "id", "")
                for dec in node.decorator_list
            }
            if "runtime_checkable" in decorators:
                checkable.append(f"{path.name}:{node.name}")

    unknown = sorted(protocols - KNOWN_DOMAIN_PROTOCOLS)
    ok = not unknown and not port_classes and not checkable
    evidence = (
        f"扫 {len(files)} 个文件；Protocol {len(protocols)} 个"
        f"（已知例外 {sorted(protocols & KNOWN_DOMAIN_PROTOCOLS)}）；"
        f"*Port 类 {len(port_classes)}；@runtime_checkable {len(checkable)}"
    )
    if unknown:
        evidence += f"；未登记 Protocol {unknown}（先登记例外并说明理由）"
    if port_classes:
        evidence += f"；{port_classes}"
    if checkable:
        evidence += f"；{checkable}"
    return ok, evidence


def check_005() -> tuple[bool, str]:
    """端口清单：ports.py 的 `__all__` 有 7 个端口，architecture.md 有对应映射行。"""
    ports_text = (AGENT / "app" / "ports.py").read_text(encoding="utf-8")
    tree = ast.parse(ports_text, filename="agent/app/ports.py")
    declared: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            )
            and isinstance(node.value, ast.List)
        ):
            declared = {
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
    missing_in_ports = [name for name in PORTS if name not in declared]

    arch_text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    mapped = [
        name
        for name in PORTS
        if re.search(rf"^\|\s*`{re.escape(name)}`\s*\|", arch_text, re.MULTILINE)
    ]

    ok = not missing_in_ports and len(mapped) == len(PORTS)
    evidence = (
        f"ports.__all__ 端口 {len(PORTS) - len(missing_in_ports)}/7；"
        f"architecture.md 映射行 {len(mapped)}/7"
    )
    if missing_in_ports:
        evidence += f"；ports 缺 {missing_in_ports}"
    if len(mapped) != len(PORTS):
        evidence += f"；architecture 缺 {sorted(set(PORTS) - set(mapped))}"
    return ok, evidence


def check_006() -> tuple[bool, str]:
    """agent/ 目录树包含卡面必需项（domain 8 / app 3 / adapters 8 子包）。"""
    domain = {path.name for path in (AGENT / "domain").glob("*.py")}
    app = {path.name for path in (AGENT / "app").glob("*.py")}
    adapters_init = (AGENT / "adapters" / "__init__.py").exists()
    subpkgs = [
        name for name in ADAPTER_SUBPKGS if (AGENT / "adapters" / name / "__init__.py").exists()
    ]
    total = len(list(AGENT.rglob("*.py")))

    problems: list[str] = []
    missing_domain = sorted(DOMAIN_EXPECTED - domain)
    if missing_domain:
        problems.append(f"domain 缺必需文件 {missing_domain}")
    if app != APP_EXPECTED:
        problems.append(f"app 多 {sorted(app - APP_EXPECTED)} 缺 {sorted(APP_EXPECTED - app)}")
    if not adapters_init:
        problems.append("缺 adapters/__init__.py")
    if len(subpkgs) != len(ADAPTER_SUBPKGS):
        problems.append(f"adapters 子包缺 {sorted(set(ADAPTER_SUBPKGS) - set(subpkgs))}")

    ok = not problems
    evidence = (
        f"domain {len(domain)}（必需 {len(DOMAIN_EXPECTED)} 全在={not missing_domain}）/ "
        f"app {len(app)} / adapters {len(subpkgs)} 子包；agent 共 {total} 个 .py"
    )
    if problems:
        evidence += "；" + "；".join(problems)
    return ok, evidence


def check_007() -> tuple[bool, str]:
    """docs/architecture.md 含端口清单、依赖方向、组合根、待启用规则。"""
    text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    missing = [token for token in ARCH_TOKENS if token not in text]
    ok = not missing
    evidence = f"四要素 {len(ARCH_TOKENS) - len(missing)}/{len(ARCH_TOKENS)}"
    if missing:
        evidence += f"；缺 {missing}"
    return ok, evidence


def check_008() -> tuple[bool, str]:
    """pyproject.toml 的 lint 配置与 architecture.md 描述一致（三条 + 一条待启用）。"""
    pytext = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    mdtext = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    blocks = len(re.findall(r"^\[\[tool\.importlinter\.contracts\]\]", pytext, re.MULTILINE))
    py_names = all(name in pytext for name in CONTRACTS)
    md_names = all(name in mdtext for name in CONTRACTS)
    pending_py = "组合根规则" in pytext and "暂不启用" in pytext
    pending_md = "待启用规则" in mdtext
    ok = blocks == 3 and py_names and md_names and pending_py and pending_md
    return ok, (
        f"pyproject contract {blocks} 条（期望 3）；三条名 pyproject={py_names} md={md_names}；"
        f"待启用组合根规则 pyproject={pending_py} md={pending_md}"
    )


def check_009() -> tuple[bool, str]:
    """pytest -q 末行与 `docs/baseline.md` §2 一致（数字比较，skipped 不比）。"""
    code, out = _run(_pytest_cmd("-q"), timeout=900)
    summary = _last_summary(out)
    matched, detail = _baseline.pytest_matches(summary)
    ok = code == 0 and matched
    return ok, f"EXIT={code}；{detail}；末行 {summary}"


def check_010() -> tuple[bool, str]:
    """pytest --collect-only -q 与 nodeids 哈希均与 `docs/baseline.md` §2 一致。"""
    exp = _baseline.expected()
    code, out = _run(_pytest_cmd("--collect-only", "-q"), timeout=600)
    match = re.search(r"(\d+)/(\d+) tests collected", out)
    ratio = f"{match.group(1)}/{match.group(2)}" if match else "未匹配"
    collected = "t11_checks" in out

    nodeids = ROOT / ".pytest_cache" / "v" / "cache" / "nodeids"
    try:
        digest = _baseline.assert_nodeids_sha(nodeids)
        hash_state = f"nodeids 一致 {digest[:8]}…"
        hash_ok = True
    except _baseline.BaselineError as exc:
        hash_state = f"nodeids 不一致：{exc}"
        hash_ok = False

    ok = code == 0 and ratio == exp["collect"] and not collected and hash_ok
    return ok, (
        f"EXIT={code}；收集 {ratio}（baseline.md: {exp['collect']}）；"
        f"t11_checks 被收集={'是' if collected else '否'}；{hash_state}"
    )


CHECKS: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
    ("T11-001 lint-imports 三条 contract 全 KEPT", check_001),
    ("T11-002 违规必拦：domain 注入 langchain_core", check_002),
    ("T11-003 违规必拦：app 越界 import agent.adapters", check_003),
    ("T11-004 domain 无端口实现泄漏（Protocol 例外已登记）", check_004),
    ("T11-005 端口清单与 architecture 映射一致", check_005),
    ("T11-006 agent/ 目录树必需项齐全", check_006),
    ("T11-007 architecture.md 四要素齐备", check_007),
    ("T11-008 pyproject 与 architecture 口径一致", check_008),
    ("T11-009 pytest -q 与 baseline.md 一致", check_009),
    ("T11-010 collect-only / nodeids 与 baseline.md 一致", check_010),
]


def main() -> int:
    passed = 0
    failed = 0
    for name, check in CHECKS:
        try:
            ok, evidence = check()
        except Exception as exc:  # noqa: BLE001 - 单条异常不应吞掉其余用例
            ok, evidence = False, f"异常 {type(exc).__name__}: {exc}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name} — {evidence}")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"RESULT: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
