#!/usr/bin/env python
"""T60 工具注册表（T6-0）—— 测试窗口 G4 独立验收脚本。

用法（仓库根目录）：

    poetry run python "tests/test_agent_cases/T60-工具注册表/t60_g4_checks.py"

与开发窗口 `t60_checks.py` 的关系：
    `t60_checks.py` 是开发自检（T60-001 ~ T60-010）；
    本脚本是**独立验收**（T60-011 ~ T60-020），不改业务代码、不读开发脚本的判定结果，
    只复用它的冻结取值表 `BASELINE_SPECS` 作为“契约取值”对照物。
    新增 OI-P-005 断言：spec.description / spec.input_schema 必须与
    `convert_to_openai_tool(实际函数)` 逐字节一致，防止 spec 静默漂移。

退出码：
    0 = T60-011 ~ T60-020 全部通过（C 类发现不影响退出码，只打 [C-ISSUE]）
    1 = 至少一项 A/B 检查失败

为什么不是 `test_*.py`：
    会被 pytest 收集，`--collect-only` 从 227/231 变成 227+1，
    `docs/baseline.md` §2 的 nodeids 哈希失效（见 tests/test_agent_cases/README.md）。

只读承诺：本脚本不写仓库内任何文件；B4 的假工具只登记在内存里，结束前 pop 掉。
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / "docs" / "contracts" / "tool-spec.md"
CONVERSATION_GRAPH = REPO / "src" / "conversation_graph.py"
LEDGER = REPO / "tests" / "test_agent_cases" / "台账用例.csv"
DEV_CHECKS = Path(__file__).with_name("t60_checks.py")
COMMIT = "0a4eaa0"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASELINE_NAMES = [
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
]
BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
BASELINE_BYTES = 2558
TOOL_FIELDS = [
    "name",
    "description",
    "input_schema",
    "permission",
    "cost_model",
    "timeout_seconds",
    "max_output_tokens",
    "requires_approval",
    "feature_flag",
]
PERMISSIONS = {"read", "external", "write", "dangerous"}
COST_UNITS = {"call", "keyword", "token"}
LEDGER_HEADER = [
    "case_id",
    "module",
    "sub_feature",
    "title",
    "level",
    "auto",
    "priority",
    "precondition",
    "steps",
    "expected",
    "status",
    "last_run",
    "run_by",
    "notes",
    "open_issue",
]


class Checker:
    """每项一行 `[PASS]/[FAIL]`；C 类单独走 `[C-OK]/[C-ISSUE]`，不计入失败数。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.c_ok = 0
        self.c_issues: list[str] = []

    def run(self, case_id: str, title: str, fn: Any) -> None:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - 验收脚本要汇总所有失败项
            self.failed += 1
            print(f"[FAIL] {case_id} {title}: {type(exc).__name__}: {exc}")
        else:
            self.passed += 1
            print(f"[PASS] {case_id} {title}: {detail}")

    def probe(self, tag: str, title: str, fn: Any) -> None:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - C 类只记录，不阻断
            self.c_issues.append(f"{tag} {type(exc).__name__}: {exc}")
            print(f"[C-ISSUE] {tag} {title}: {type(exc).__name__}: {exc}")
        else:
            self.c_ok += 1
            print(f"[C-OK] {tag} {title}: {detail}")


def _run(args: list[str], timeout: int = 240) -> subprocess.CompletedProcess[str]:
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


def _sha_of(tools: list[Any]) -> tuple[str, int]:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schemas = [convert_to_openai_tool(tool) for tool in tools]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest(), len(canonical)


def _contract_fields(marker: str) -> list[str]:
    text = CONTRACT.read_text(encoding="utf-8")
    block = re.search(
        rf"<!-- t12:fields {marker} -->(.*?)<!-- /t12:fields {marker} -->", text, re.S
    )
    assert block is not None, f"契约缺少 t12:fields {marker} 机读块"
    return re.findall(r"^\| `([a-z_]+)` \|", block.group(1), re.M)


def _dev_baseline() -> dict[str, dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("t60_dev_checks_baseline", DEV_CHECKS)
    assert spec is not None and spec.loader is not None, "无法加载 t60_checks.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BASELINE_SPECS


# --------------------------------------------------------------------------------------
# A 组：行为零变更
# --------------------------------------------------------------------------------------


def check_a1_smoke_hash() -> str:
    proc = _run([sys.executable, "scripts/smoke_tools_schema.py"])
    assert proc.returncode == 0, f"smoke 退出码 {proc.returncode}: {proc.stderr[-300:]}"
    digest = re.search(r"sha256\s*[:：]\s*([0-9a-f]{64})", proc.stdout)
    size = re.search(r"bytes\s*[:：]\s*(\d+)", proc.stdout)
    assert digest is not None, f"smoke 输出无 sha256：{proc.stdout[-300:]}"
    assert size is not None, f"smoke 输出无 bytes：{proc.stdout[-300:]}"
    assert digest.group(1) == BASELINE_SHA256, f"sha256 漂移：{digest.group(1)}"
    assert int(size.group(1)) == BASELINE_BYTES, f"bytes 漂移：{size.group(1)}"
    return f"sha256={digest.group(1)[:12]}... bytes={size.group(1)}（独立进程跑 scripts/）"


def check_a2_names_order() -> str:
    from src.conversation_graph import TOOLS

    names = [tool.__name__ for tool in TOOLS]
    assert names == BASELINE_NAMES, f"名字/顺序不一致：{names}"
    return f"5/5 同名同序：{' / '.join(names)}"


def check_a3_identity() -> str:
    from agent.adapters.tools import registry
    from src.conversation_graph import TOOLS

    live = registry.to_lc_tools()
    assert len(live) == len(TOOLS) == 5, f"长度不一致：TOOLS={len(TOOLS)} registry={len(live)}"
    pairs = [a is b for a, b in zip(TOOLS, live, strict=True)]
    assert all(pairs), f"identity 不成立：{pairs}"
    assert [t.__name__ for t in live] == BASELINE_NAMES
    return "5/5 identity True（TOOLS[i] is registry.to_lc_tools()[i]）"


def check_a4_commit_scope() -> str:
    stat = _run(["git", "show", "--name-only", "--format=", COMMIT])
    assert stat.returncode == 0, f"git show 失败：{stat.stderr[-300:]}"
    touched = [line.strip() for line in stat.stdout.splitlines() if line.strip()]
    tools_touched = [p for p in touched if p.startswith("src/tools/")]
    assert not tools_touched, f"src/tools/* 被改动：{tools_touched}"

    patch = _run(["git", "show", "--format=", COMMIT, "--", "src/conversation_graph.py"])
    assert patch.returncode == 0, f"取 conversation_graph diff 失败：{patch.stderr[-300:]}"
    added, removed = [], []
    for line in patch.stdout.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:].strip())
        elif line.startswith("-"):
            removed.append(line[1:].strip())

    added_ok = [
        re.fullmatch(r"from agent\.adapters\.tools import registry", s)
        or re.fullmatch(
            r"from src\.tools\.(offload|rag|search) import .*# noqa: F401 - 保持既有导出", s
        )
        or re.fullmatch(r"# ── 工具列表：模型每轮只能从这 5 个工具里选.*registry\.py.*──", s)
        or re.fullmatch(r"TOOLS: list\[Callable\[\.\.\., Any\]\] = registry\.to_lc_tools\(\)", s)
        for s in added
    ]
    removed_ok = [
        re.fullmatch(r"from src\.tools\.(offload|rag|search) import .*", s)
        or re.fullmatch(r"# ── 工具列表：模型每轮只能从这 4 个工具里选.*", s)
        or re.fullmatch(r"TOOLS: list\[Callable\[\.\.\., Any\]\] = \[", s)
        or re.fullmatch(
            r"(search_knowledge|search_tavily|calculator|deep_research|read_offload),", s
        )
        or re.fullmatch(r"\]", s)
        for s in removed
    ]
    assert all(added_ok), f"conversation_graph 出现允许范围外的 + 行：{added}"
    assert all(removed_ok), f"conversation_graph 出现允许范围外的 - 行：{removed}"
    assert any("5 个工具" in s for s in added), "未见「4 个工具 -> 5 个工具」注释订正"
    assert not any("4 个工具" in s for s in added), "新增行仍写「4 个工具」"

    hash_before = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    assert hash_before.startswith(COMMIT), f"HEAD 不是 {COMMIT}：{hash_before}"
    return f"{len(touched)} 文件改动，src/tools/* 零改动，conversation_graph +{len(added)}/-{len(removed)} 全在允许集"


def check_a5_lint_imports() -> str:
    exe = shutil.which("lint-imports") or str(Path(sys.executable).parent / "lint-imports")
    assert Path(exe).exists() or shutil.which("lint-imports"), f"找不到 lint-imports：{exe}"
    proc = _run([exe])
    out = proc.stdout + proc.stderr
    match = re.search(r"(\d+)\s+kept,\s*(\d+)\s+broken", out)
    assert match is not None, f"无法解析 lint-imports 输出：{out[-500:]}"
    kept, broken = int(match.group(1)), int(match.group(2))
    assert broken == 0, f"broken={broken}"
    assert kept == 3, f"kept={kept}，基线 3"
    return f"{kept} kept, {broken} broken"


# --------------------------------------------------------------------------------------
# B 组：注册表语义
# --------------------------------------------------------------------------------------


def check_b1_specs_vs_contract() -> str:
    from agent.adapters.tools import registry
    from agent.domain.tool import ApprovalThreshold, CostModel, ToolSpec

    baseline = _dev_baseline()
    assert _contract_fields("ToolSpec") == TOOL_FIELDS, "契约字段与冻结九字段不一致"
    assert [f.name for f in dataclasses.fields(ToolSpec)] == TOOL_FIELDS

    for name in BASELINE_NAMES:
        item = registry.spec(name)
        assert isinstance(item, ToolSpec), f"{name}: 类型错误"
        assert dataclasses.is_dataclass(item), f"{name}: 不是 dataclass"
        for field in TOOL_FIELDS:
            assert hasattr(item, field), f"{name}: 缺字段 {field}"
        assert isinstance(item.description, str) and item.description, f"{name}: description 空"
        assert isinstance(item.input_schema, dict) and item.input_schema, f"{name}: schema 空"
        assert item.input_schema.get("type") == "object", f"{name}: schema 非 object"
        assert item.permission in PERMISSIONS, f"{name}: permission 非法"
        assert isinstance(item.cost_model, CostModel), f"{name}: cost_model 类型错"
        assert item.cost_model.unit in COST_UNITS, f"{name}: unit 非法"
        assert isinstance(item.timeout_seconds, float), f"{name}: timeout 非 float"
        assert isinstance(item.max_output_tokens, int), f"{name}: max_output_tokens 非 int"
        assert isinstance(item.requires_approval, (bool, ApprovalThreshold)), (
            f"{name}: requires_approval 形态非法"
        )
        assert item.feature_flag is None or isinstance(item.feature_flag, str)

        expected = baseline[name]
        assert item.permission == expected["permission"], f"{name}: permission 漂移"
        assert item.cost_model.unit == expected["unit"], f"{name}: unit 漂移"
        assert item.cost_model.unit_cost == expected["unit_cost"], f"{name}: unit_cost 漂移"
        assert item.timeout_seconds == expected["timeout_seconds"], f"{name}: timeout 漂移"
        assert item.max_output_tokens == expected["max_output_tokens"], (
            f"{name}: max_output_tokens 漂移"
        )
        assert item.requires_approval is expected["requires_approval"], (
            f"{name}: requires_approval 漂移"
        )
        assert item.feature_flag == expected["feature_flag"], f"{name}: feature_flag 漂移"

    assert registry.spec("deep_research").requires_approval is True, "deep_research 未恒审批"
    return "5/5 九字段 + 取值一致，deep_research.requires_approval is True"


def check_b2_spec_matches_live_tool() -> str:
    """OI-P-005：spec 必须与“模型实际看到的” description/parameters 完全一致。"""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from agent.adapters.tools import registry

    live = registry.to_lc_tools()
    assert len(live) == 5
    for fn in live:
        name = fn.__name__
        spec = registry.spec(name)
        converted = convert_to_openai_tool(fn)["function"]
        assert converted["name"] == spec.name, f"{name}: name 不一致"
        assert converted["description"] == spec.description, (
            f"{name}: description 漂移（spec 与真实函数 docstring 不同）"
        )
        assert converted["parameters"] == spec.input_schema, (
            f"{name}: input_schema 与真实函数 parameters 不同"
        )
    return "5/5 description 与 parameters 均逐字节等于 convert_to_openai_tool(实际函数)"


def check_b3_copy_semantics() -> str:
    from agent.adapters.tools import registry

    before = list(registry.TOOL_SPECS)
    first = registry.to_lc_tools()
    assert len(first) == 5, f"首次返回 {len(first)} 个"
    first.append(lambda: "g4")  # 外部 append
    first[0] = lambda: "g4"  # 外部替换
    second = registry.to_lc_tools()
    assert second is not first, "两次调用返回同一 list 对象（不是副本）"
    assert len(second) == 5, f"第二次返回 {len(second)} 个，被外部 append 污染"
    assert second[0].__name__ == "search_knowledge", "第二次首项被外部替换污染"
    assert [t.__name__ for t in second] == BASELINE_NAMES, "第二次名字/顺序被污染"
    assert list(registry.TOOL_SPECS) == before, "TOOL_SPECS 被外部列表操作污染"
    return "返回新 list；外部 append/替换不影响下一次调用与 TOOL_SPECS"


def check_b4_register() -> str:
    from agent.adapters.tools import registry
    from agent.domain.tool import CostModel, ToolSpec

    def fake_tool() -> str:
        return "g4"

    def make(name: str) -> ToolSpec:
        return ToolSpec(
            name=name,
            description="G4 验收假工具",
            input_schema={"properties": {}, "required": [], "type": "object"},
            permission="read",
            cost_model=CostModel(unit="call", unit_cost=0.0),
            timeout_seconds=1.0,
            max_output_tokens=10,
            requires_approval=False,
            feature_flag=None,
        )

    # 1) 重名
    try:
        registry.register("search_knowledge", fake_tool, make("search_knowledge"))
    except ValueError:
        pass
    else:
        raise AssertionError("重名 register 未抛 ValueError")

    # 2) spec.name 与注册名不一致
    try:
        registry.register("tmp_g4_mismatch", fake_tool, make("tmp_g4_other"))
    except ValueError:
        pass
    else:
        raise AssertionError("spec.name 不一致时未抛 ValueError")
    assert "tmp_g4_mismatch" not in registry.TOOL_SPECS, "失败登记污染了 TOOL_SPECS"

    # 3) 正常登记：末尾多一个
    try:
        registry.register("tmp_g4_ok", fake_tool, make("tmp_g4_ok"))
        tools = registry.to_lc_tools()
        assert len(tools) == 6, f"登记后应为 6，实际 {len(tools)}"
        assert tools[-1] is fake_tool, "登记的工具不在末尾"
        assert [t.__name__ for t in tools[:5]] == BASELINE_NAMES, "内置顺序被扰动"
        assert registry.spec("tmp_g4_ok").name == "tmp_g4_ok"
    finally:
        registry.TOOL_SPECS.pop("tmp_g4_ok", None)
        registry._TOOL_FNS.pop("tmp_g4_ok", None)  # noqa: SLF001 - 验收要复原内存态

    assert len(registry.to_lc_tools()) == 5, "复原失败：TOOL_SPECS 仍有假工具"
    return "重名/不一致均 ValueError，登记后 5 -> 6 且追加末尾，复原 6 -> 5"


def check_b5_cold_start() -> str:
    code = "\n".join(
        [
            "import sys",
            "import agent.adapters.tools.registry as r",
            "assert 'src.conversation_graph' not in sys.modules, 'import registry 时就拖入了业务模块'",
            "tools = r.to_lc_tools()",
            "assert 'src.conversation_graph' in sys.modules, 'to_lc_tools 未按需导入'",
            "print(len(tools), [t.__name__ for t in tools])",
        ]
    )
    proc = _run([sys.executable, "-c", code])
    assert proc.returncode == 0, f"冷启动失败：{proc.stderr[-500:]}"
    assert "5 ['search_knowledge'" in proc.stdout, f"输出异常：{proc.stdout!r}"
    return (
        "新进程 import registry -> to_lc_tools() 正常 5 个，且 import 期不拖入 conversation_graph"
    )


# --------------------------------------------------------------------------------------
# C 组：边界 / 反例（只记 open_issue，不影响退出码）
# --------------------------------------------------------------------------------------


_C1_REPORT = "\n".join(
    [
        "import hashlib, json",
        "from langchain_core.utils.function_calling import convert_to_openai_tool",
        "print(json.dumps({",
        "    'names': [f.__name__ for f in cg.TOOLS],",
        "    'same': [a is b for a, b in zip(cg.TOOLS, tools)],",
        "    'hash': hashlib.sha256(json.dumps(",
        "        [convert_to_openai_tool(t) for t in cg.TOOLS],",
        "        sort_keys=True, ensure_ascii=False).encode()).hexdigest(),",
        "}))",
    ]
)


def check_c1_import_order() -> str:
    # 路径 A：先直接 import src.conversation_graph（其内部先 import registry）
    path_a = "\n".join(
        [
            "import src.conversation_graph as cg",
            "import agent.adapters.tools.registry as r",
            "tools = r.to_lc_tools()",
            _C1_REPORT,
        ]
    )
    # 路径 B：先 import registry 并先调用 to_lc_tools()，再 import conversation_graph
    path_b = "\n".join(
        [
            "import agent.adapters.tools.registry as r",
            "tools = r.to_lc_tools()",
            "import src.conversation_graph as cg",
            _C1_REPORT,
        ]
    )
    proc_a = _run([sys.executable, "-c", path_a])
    proc_b = _run([sys.executable, "-c", path_b])
    assert proc_a.returncode == 0, f"路径 A 失败：{proc_a.stderr[-400:]}"
    assert proc_b.returncode == 0, f"路径 B 失败：{proc_b.stderr[-400:]}"
    result_a = json.loads(proc_a.stdout.strip().splitlines()[-1])
    result_b = json.loads(proc_b.stdout.strip().splitlines()[-1])

    assert result_a["names"] == result_b["names"] == BASELINE_NAMES, "两条路径名字/顺序不一致"
    assert all(result_a["same"]), f"路径 A identity 不成立：{result_a['same']}"
    assert all(result_b["same"]), f"路径 B identity 不成立：{result_b['same']}"
    assert result_a["hash"] == result_b["hash"] == BASELINE_SHA256, (
        f"两条路径 schema 哈希不同：{result_a['hash']} / {result_b['hash']}"
    )
    return "两条路径（先 conversation_graph / 先 registry）名字、顺序、identity、schema 哈希全一致"


def check_c2_ledger() -> str:
    raw = LEDGER.read_bytes()
    text = raw.decode("gb18030")  # 解码失败即抛 UnicodeDecodeError
    assert text.encode("gb18030") == raw, "GB18030 往返不一致"
    rows = list(csv.reader(io.StringIO(text, newline="")))
    assert rows, "台账为空"
    assert rows[0] == LEDGER_HEADER, f"表头不符：{rows[0]}"
    body = rows[1:]
    bad_cols = [(i + 2, len(r)) for i, r in enumerate(body) if len(r) != 15]
    assert not bad_cols, f"列数不是 15 的行：{bad_cols[:5]}"
    ids = [r[0] for r in body]
    assert len(ids) == len(set(ids)), "case_id 有重复"

    by_id = {r[0]: r for r in body}
    for seq in range(1, 11):
        cid = f"T60-{seq:03d}"
        assert cid in by_id, f"缺开发窗口的 {cid}"
        row = by_id[cid]
        assert row[11] == "2026/11/11", f"{cid} last_run={row[11]}"
        assert row[12] == "dev-G4", f"{cid} run_by={row[12]}"
        notes = row[13].strip()
        assert notes not in {"PASS", "pass", "通过", "PASSED"}, f"{cid} notes 是自引用"
        assert len(notes) >= 10, f"{cid} notes 太短（无证据）：{notes}"
        assert re.search(r"sha256|bytes|PASS；|T60-|/5|=", notes), (
            f"{cid} notes 无证据痕迹：{notes}"
        )

    mine = [f"T60-{n:03d}" for n in range(11, 21)]
    present = [cid for cid in mine if cid in by_id]
    if present:
        assert len(present) == 10, f"G4 区间不完整：{present}"
        for cid in mine:
            row = by_id[cid]
            assert row[12] == "test-G4", f"{cid} run_by={row[12]}"
            assert row[11] == "2026/11/11", f"{cid} last_run={row[11]}"
            assert row[13].strip(), f"{cid} notes 为空"
    return f"{len(body)} 行 x {len(LEDGER_HEADER)} 列 / GB18030 往返 OK / T60-001~010 证据行 OK / G4 区间 {len(present)}/10"


_WRITE_ATTRS = frozenset(
    {
        "write_text",
        "write_bytes",
        "touch",
        "unlink",
        "mkdir",
        "rmdir",
        "rename",
        "replace",
        "remove",
        "makedirs",
        "rmtree",
        "move",
        "copy",
        "copyfile",
        "writelines",
    }
)


def _write_op(node: ast.Call) -> str | None:
    """`ast.Call` -> 写操作描述；只读调用返回 None。"""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "open":
        mode = ""
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            mode = str(node.args[1].value)
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = str(kw.value.value)
        return f"open(mode={mode!r})@{node.lineno}" if any(f in mode for f in "wax+") else None
    label = ""
    if isinstance(func, ast.Name):
        label = func.id
    elif isinstance(func, ast.Attribute):
        label = func.attr
    return f"{label}()@{node.lineno}" if label in _WRITE_ATTRS else None


def _scan_write_ops(source: str) -> list[str]:
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    ops = (_write_op(node) for node in calls)
    return [op for op in ops if op is not None]


def _repo_snapshot() -> dict[str, tuple[int, int]]:
    snap: dict[str, tuple[int, int]] = {}
    for base in ("src", "agent", "scripts", "tests", "docs", "evals"):
        root = REPO / base
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                stat = path.stat()
                snap[str(path.relative_to(REPO))] = (stat.st_size, stat.st_mtime_ns)
    return snap


def check_c3_dev_checks_readonly() -> str:
    offenders = _scan_write_ops(DEV_CHECKS.read_text(encoding="utf-8"))
    assert not offenders, f"t60_checks.py 疑似写文件：{offenders}"

    before = _repo_snapshot()
    proc = _run([sys.executable, str(DEV_CHECKS)])
    assert proc.returncode == 0, f"t60_checks.py 退出码 {proc.returncode}: {proc.stderr[-300:]}"
    after = _repo_snapshot()
    changed = sorted(set(before) ^ set(after)) + sorted(
        key for key in before.keys() & after.keys() if before[key] != after[key]
    )
    assert not changed, f"跑 t60_checks.py 改动了仓库文件：{changed[:5]}"
    assert not DEV_CHECKS.name.startswith("test_"), "t60_checks.py 会被 pytest 收集"
    assert "RESULT: 10 passed, 0 failed" in proc.stdout, "t60_checks 末行不是 10 passed"
    return f"AST 无写操作；实跑前后 {len(before)} 个文件 size/mtime 全等"


# --------------------------------------------------------------------------------------


def main() -> int:
    checker = Checker()
    checker.run("T60-011", "smoke sha256/bytes = 基线（A1）", check_a1_smoke_hash)
    checker.run("T60-012", "名字与顺序 = 基线（A2）", check_a2_names_order)
    checker.run("T60-013", "TOOLS 与 to_lc_tools 同一批函数对象（A3）", check_a3_identity)
    checker.run("T60-014", "提交范围：src/tools 零改动（A4）", check_a4_commit_scope)
    checker.run("T60-015", "lint-imports 3 kept / 0 broken（A5）", check_a5_lint_imports)
    checker.run("T60-016", "5 条 spec 与契约九字段一致（B1）", check_b1_specs_vs_contract)
    checker.run(
        "T60-017",
        "spec 与真实函数 description/parameters 一致（B2/OI-P-005）",
        check_b2_spec_matches_live_tool,
    )
    checker.run("T60-018", "to_lc_tools 返回副本（B3）", check_b3_copy_semantics)
    checker.run("T60-019", "register 重名/不一致/追加语义（B4）", check_b4_register)
    checker.run("T60-020", "冷启动 import 安全与惰性导入（B5）", check_b5_cold_start)

    checker.probe("C1", "两条导入路径 TOOLS 一致", check_c1_import_order)
    checker.probe("C2", "台账 86x15 / GB18030 / 证据 notes", check_c2_ledger)
    checker.probe("C3", "t60_checks.py 只读", check_c3_dev_checks_readonly)

    print(f"C-FINDINGS: {checker.c_ok} ok, {len(checker.c_issues)} issue")
    for issue in checker.c_issues:
        print(f"OPEN_ISSUE: {issue}")
    print(f"RESULT: {checker.passed} passed, {checker.failed} failed")
    code = 0 if checker.failed == 0 else 1
    print(f"EXIT={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
