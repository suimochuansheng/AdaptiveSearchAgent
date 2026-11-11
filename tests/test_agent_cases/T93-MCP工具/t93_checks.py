#!/usr/bin/env python
"""G8/T9-3 MCP 工具适配器 —— 测试窗口（test-G8）**独立验收**。

用法（仓库根目录，任意 cwd 均可）：

    export PRE_COMMIT_HOME="$PWD/.pre-commit-cache"
    poetry run python "tests/test_agent_cases/T93-MCP工具/t93_checks.py"

覆盖（判分标准 A/B 必过，C 只记 open_issue）：

    A1  开关关：注册表 5 个、schema 哈希 c48c9d48…2a99、to_lc_tools 与内置函数 identity
    A2  开关开 + Server 正常：注册表 7 个、to_lc_tools 7 个、两个新名字追加在末尾
    A3  **重建图之后**假模型 bind_tools 真收到的工具名集合含 word_count / unit_convert
    A4  新工具 ToolSpec 九字段齐全、permission="external"、requires_approval is False
    A5  word_count / unit_convert 各调一次结果正确；非法输入返回可读错误
    A6  杀掉 Server 再调：可读"不可用"错误，同一会话其它工具照常可用
    A7  Server 启动失败（命令改 false）：一条 warning、启动继续、5 个内置可用
    A8  关开关重启（全新子进程）：回到 5 个、无残留、无子进程
    B1  pytest / collect / nodeids 与 `docs/baseline.md` §2 一致（不再硬编码数字）
    B2  原 5 个工具 schema 哈希不变（新增不改旧）
    B3  lint-imports 3 kept；pyproject.toml、poetry.lock 零改动
    B4  未改 src/tools/*、frontend/、hitl.py、approval_store.py
    B5  台账里没有开发窗口写的 T93 行
    C1  Server 卡住不响应（sleep 30）：MCP_TIMEOUT_SECONDS 生效
    C2  两次加载同名工具：第二次 register 报重名，不静默覆盖、无旁路写表
    C3  reset 后 TOOLS 兼容别名与 current_tools() 是否一致
    C4  Server 往 stdout 打非协议内容：客户端不崩
    C5  关 flag 时子进程确实没被拉起（扫 /proc 子进程 + 日志）

退出码：
    0 = A1~A8 / B1~B5 全部通过（C 类发现只打 [C-ISSUE]，不影响退出码）
    1 = A/B 至少一项失败

为什么不是 `test_*.py`：
    会被 pytest 收集，`--collect-only` 的收集数改变，
    `docs/baseline.md` §2 的 nodeids 哈希失效（见 tests/test_agent_cases/README.md）。

只读承诺：
    不写仓库内任何业务文件；不 import 开发窗口的自测脚本、不复用它的判定结果。
    副作用只有 pytest 的 `.pytest_cache/`（已 gitignore）与 /tmp 下的慢 Server/脏输出 Server
    临时脚本（用 `-c` 内联传码，**不落盘**）；内存里登记的工具结束时全部撤回。
"""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: 卡级基线（pytest 通过数 / 收集数 / nodeids）唯一来源：`docs/baseline.md` §2。
CASE_ROOT = Path(__file__).resolve().parents[1]
if str(CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(CASE_ROOT))

import _baseline  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from agent.adapters.tools import mcp_client, registry  # noqa: E402
from agent.domain.tool import ToolSpec  # noqa: E402
from config import settings  # noqa: E402
from src import conversation_graph as cg  # noqa: E402

# ── 冻结基线（docs/baseline.md §2 / T6-0 契约）────────────────────────────
BASELINE_NAMES = [
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
]
BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
NEW_TOOLS = ["word_count", "unit_convert"]
EXPECTED_NAMES_WITH_MCP = BASELINE_NAMES + NEW_TOOLS

#: 本卡开工 HEAD（交接单第 1 节）：所有"零改动"类断言都拿它当基准
BASELINE_COMMIT = "7d02f24"

#: 契约 `ToolSpec` 九字段（docs/contracts/tool-spec.md §1，顺序即 dataclass 顺序）
CONTRACT_FIELDS = [
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

#: 内置工具的真实定义处（identity 对照用，故意不经 registry 的私有映射）
BUILTIN_IMPORTS: dict[str, tuple[str, str]] = {
    "search_knowledge": ("src.tools.rag", "search_knowledge"),
    "search_tavily": ("src.tools.search", "search_tavily"),
    "calculator": ("src.conversation_graph", "calculator"),
    "deep_research": ("src.conversation_graph", "deep_research"),
    "read_offload": ("src.tools.offload", "read_offload"),
}

LEDGER = REPO / "tests" / "test_agent_cases" / "台账用例.csv"

#: 结果三元组（case_id, 标题, 是否通过, 证据）
RESULTS: list[tuple[str, str, bool, str]] = []
#: C 类探针（不参与 RESULT 计数）
C_PROBES: list[tuple[str, bool, str]] = []


# ── 打印与记录 ────────────────────────────────────────────────────────


def record(case_id: str, title: str, ok: bool, evidence: str) -> bool:
    RESULTS.append((case_id, title, ok, evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {case_id} {title} — {evidence}", flush=True)
    return ok


def record_c(case_id: str, ok: bool, evidence: str) -> None:
    C_PROBES.append((case_id, ok, evidence))
    tag = "C-PASS" if ok else "C-ISSUE"
    print(f"[{tag}] {case_id} — {evidence}", flush=True)


def skip(case_id: str, title: str, reason: str) -> None:
    record(case_id, title, False, f"未验证：{reason}")


# ── 小工具 ────────────────────────────────────────────────────────────


def tool_names(tools: list[Any]) -> list[str]:
    return [getattr(tool, "__name__", "?") for tool in tools]


def schema_hash(tools: list[Any]) -> str:
    """与 scripts/smoke_tools_schema.py 同一口径：OpenAI function schema 的规范化 sha256。"""
    schemas = [convert_to_openai_tool(tool) for tool in tools]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def baseline_tools() -> list[Any]:
    """只取前 5 个内置工具（按登记顺序）。"""
    live = registry.to_lc_tools()
    return [tool for tool in live if getattr(tool, "__name__", "") in BASELINE_NAMES]


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def git_out(*args: str) -> str:
    return git(*args).stdout.strip()


def children_of(pid: int) -> list[tuple[int, str]]:
    """扫 /proc 找直接子进程（cmdline），用于"子进程是否被拉起 / 是否还在"。"""
    found: list[tuple[int, str]] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return found
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text().split()
            if int(stat[3]) != pid:
                continue
            cmdline = Path(f"/proc/{entry}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            found.append((int(entry), cmdline))
        except (OSError, ValueError, IndexError):
            continue
    return found


def kill_children(pid: int, marker: str) -> list[int]:
    """按 cmdline 关键字杀掉子进程，返回被杀 pid（清理卡住的 Server 用）。"""
    killed: list[int] = []
    for child_pid, cmdline in children_of(pid):
        if marker in cmdline:
            try:
                os.kill(child_pid, signal.SIGKILL)
                killed.append(child_pid)
            except OSError:
                continue
    return killed


def wait_gone(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return True
        time.sleep(0.05)
    return not Path(f"/proc/{pid}").exists()


def drop_mcp_tools() -> None:
    """把热注册进来的工具从注册表撤回（模拟关开关重启），只动非基线名字。"""
    for name in [name for name in registry.TOOL_SPECS if name not in BASELINE_NAMES]:
        registry.TOOL_SPECS.pop(name, None)
        registry._TOOL_FNS.pop(name, None)  # noqa: SLF001 - 测试脚本撤回动态登记
    import_api_main().reset_conversation_graph()
    mcp_client.reset_unavailable()


async def close_mcp_quietly(timeout: float = 10.0) -> None:
    try:
        await asyncio.wait_for(mcp_client.close_mcp_session(), timeout=timeout)
    except BaseException as exc:  # noqa: BLE001 - 清理失败不该影响判定
        print(f"    （清理 MCP 会话时忽略异常：{type(exc).__name__}: {exc}）", flush=True)


def import_api_main() -> Any:
    import api_main  # noqa: PLC0415 - 延迟到 A 类检查再导入，避免污染 A1 前的模块状态

    return api_main


class _LogCapture(logging.Handler):
    """抓 `adaptive_agent` 的日志记录（A7 只允许一条 warning）。"""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def capture_graph_bind() -> list[Any]:
    """真建一次图，用假模型接住 bind_tools 的**实参**（不是只看注册表）。"""
    captured: dict[str, list[Any]] = {}

    class _CaptureModel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def bind_tools(self, tools: list[Any]) -> _CaptureModel:
            captured["tools"] = list(tools)
            return self

        async def ainvoke(self, _messages: object) -> object:  # pragma: no cover
            from langchain_core.messages import AIMessage

            return AIMessage(content="ok")

    original = cg.ChatOllama
    setattr(cg, "ChatOllama", _CaptureModel)  # noqa: B010 - 验收脚本临时替换模型类
    try:
        cg.build_conversation_graph()
    finally:
        setattr(cg, "ChatOllama", original)  # noqa: B010
    return captured.get("tools", [])


# ── B 类：不变量（静态，先跑，避免受内存态影响）───────────────────────


def check_b3() -> bool:
    """B3：lint-imports 3 kept；pyproject.toml / poetry.lock 零改动。"""
    exe = shutil.which("lint-imports")
    if exe is None:
        record("T93-011", "B3 lint-imports 3 kept / 依赖文件零改动", False, "找不到 lint-imports")
        return False
    proc = subprocess.run([exe], cwd=REPO, capture_output=True, text=True, timeout=180, check=False)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
    match = re.search(r"Contracts:\s*(\d+) kept,\s*(\d+) broken", plain)
    kept, broken = (match.group(1), match.group(2)) if match else ("?", "?")
    diffs = git_out(
        "diff", "--name-only", f"{BASELINE_COMMIT}..HEAD", "--", "poetry.lock", "pyproject.toml"
    )
    dirty = git_out("status", "--porcelain", "--", "pyproject.toml", "poetry.lock")
    ok = proc.returncode == 0 and (kept, broken) == ("3", "0") and not diffs and not dirty
    detail = f"lint-imports exit={proc.returncode} kept={kept} broken={broken}；依赖文件 diff='{diffs}' 工作树='{dirty}'"
    return record("T93-011", "B3 lint-imports 3 kept / 依赖文件零改动", ok, detail)


def check_b4() -> bool:
    """B4：未改 src/tools/*、frontend/、hitl.py、approval_store.py。"""
    changed = [
        line
        for line in git_out("diff", "--name-only", f"{BASELINE_COMMIT}..HEAD").splitlines()
        if line
    ]
    forbidden_re = re.compile(
        r"^(src/tools/|frontend/|src/agents/hitl\.py|src/approval_store\.py|"
        r"agent/domain/hitl|agent/adapters/stores/approval)"
    )
    hit = [path for path in changed if forbidden_re.search(path)]
    forbidden_grep = git_out(
        "diff",
        "--name-only",
        f"{BASELINE_COMMIT}..HEAD",
        "--",
        "src/tools",
        "frontend",
        "*hitl.py",
        "*approval_store.py",
    )
    handoff_allowed = {
        "agent/adapters/tools/mcp_client.py",
        "api_main.py",
        "config.py",
        "scripts/mcp_server.py",
        "scripts/smoke_mcp.py",
        "src/conversation_graph.py",
    }
    extra = [
        path
        for path in changed
        if path not in handoff_allowed and not path.startswith("tests/test_agent_cases/")
    ]
    ok = not hit and not forbidden_grep
    detail = (
        f"改动 {len(changed)} 个文件：{sorted(changed)}；"
        f"禁改命中={hit or '无'}；清单外改动={extra or '无'}"
    )
    if extra:
        record_c("C7", True, f"交接单清单外还有改动（非禁改路径，仅提示）：{extra}")
    return record("T93-012", "B4 未改 src/tools / frontend / hitl / approval_store", ok, detail)


def check_b5() -> bool:
    """B5：台账里没有开发窗口写的 T93 行（测试窗口自己的行 run_by=test-G8 除外）。"""
    rows: list[dict[str, str]] = []
    with LEDGER.open(encoding="gb18030", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
    t93 = [row for row in rows if (row.get("case_id") or "").startswith("T93")]
    foreign = [row["case_id"] for row in t93 if (row.get("run_by") or "").strip() != "test-G8"]
    commits = [
        line
        for line in git_out(
            "log", "--format=%s", f"{BASELINE_COMMIT}..HEAD", "--", str(LEDGER.relative_to(REPO))
        ).splitlines()
        if line
    ]
    foreign_commit = [subject for subject in commits if not subject.startswith("test(cases)")]
    ok = not foreign and not foreign_commit
    detail = (
        f"台账 T93 行={len(t93)}（非 test-G8 写的={foreign or '无'}）；"
        f"开工后动过台账的提交={commits or '无'}"
    )
    return record("T93-013", "B5 台账无开发窗口的 T93 行", ok, detail)


def check_b1() -> bool:
    """B1：pytest 通过数 / 收集数 / nodeids 与 `docs/baseline.md` §2 一致。"""
    exp = _baseline.expected()
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    collect_line = ""
    for line in reversed(collect.stdout.splitlines()):
        if "collected" in line:
            collect_line = line.strip()
            break
    nodeids = REPO / ".pytest_cache" / "v" / "cache" / "nodeids"
    node_hash = "missing"
    node_lines = -1
    if nodeids.exists():
        raw = nodeids.read_bytes()
        node_hash = hashlib.sha256(raw).hexdigest()
        node_lines = raw.count(b"\n")  # 与 `wc -l` 同口径
    full = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    tail = ""
    for line in reversed(full.stdout.splitlines()):
        if re.search(r"\d+ passed", line):
            tail = line.strip()
            break
    passed_match = re.search(r"(\d+) passed", tail)
    passed = int(passed_match.group(1)) if passed_match else -1
    ok = (
        full.returncode == 0
        and passed == exp["pytest_passed"]
        and collect.returncode == 0
        and exp["collect"] in collect_line
        and node_hash == exp["nodeids_sha256"]
        and node_lines == exp["nodeids_lines"]
    )
    detail = (
        f"pytest 末行「{tail}」；collect「{collect_line}」；"
        f"nodeids {node_lines} 行 sha256={node_hash[:8]}…"
        f"（baseline.md: {exp['pytest_passed']} passed / {exp['collect']} / "
        f"{exp['nodeids_lines']} 行 / {exp['nodeids_sha256'][:8]}…）"
    )
    return record("T93-009", "B1 pytest/收集数/nodeids 与 baseline.md 一致", ok, detail)


# ── A 类：热插拔 ──────────────────────────────────────────────────────


def check_a1() -> bool:
    """A1：开关关 → 注册表 5 个、哈希基线、to_lc_tools 与内置函数 identity。"""
    live = registry.to_lc_tools()
    names = tool_names(live)
    identity: list[bool] = []
    sources: list[str] = []
    for name, tool in zip(names, live, strict=False):
        module_name, attr = BUILTIN_IMPORTS[name]
        fn = getattr(importlib.import_module(module_name), attr)
        identity.append(tool is fn)
        sources.append(f"{name}={'is' if tool is fn else 'NOT is'}")
    digest = schema_hash(live)
    tools_alias_same = [a is b for a, b in zip(cg.TOOLS, cg.current_tools(), strict=False)]
    ok = (
        names == BASELINE_NAMES
        and all(identity)
        and digest == BASELINE_SHA256
        and settings.ENABLE_MCP is False
        and all(tools_alias_same)
    )
    detail = (
        f"ENABLE_MCP={settings.ENABLE_MCP}；注册表 {len(names)} 个={names}；"
        f"schema sha256={digest[:8]}…{digest[-4:]}（基线 c48c9d48…2a99）；"
        f"identity={sum(identity)}/{len(identity)}；TOOLS 与 current_tools 同对象={all(tools_alias_same)}"
    )
    return record("T93-001", "A1 开关关基线（5 个 / 哈希 / identity）", ok, detail)


async def check_a2() -> bool:
    """A2：开关开 + Server 正常 → 7 个，新增两个追加在末尾，图缓存被重置。"""
    api_main = import_api_main()
    api_main._conversation_graph = object()  # 伪造一个"旧图缓存"，验证热注册会重置它
    settings.ENABLE_MCP = True
    started = time.time()
    try:
        await asyncio.wait_for(api_main._register_mcp_tools(), timeout=60)
    except BaseException as exc:  # noqa: BLE001
        record(
            "T93-002",
            "A2 开关开热注册（7 个 / 追加末尾）",
            False,
            f"_register_mcp_tools 抛了 {type(exc).__name__}: {exc}",
        )
        return False
    elapsed = time.time() - started
    live = registry.to_lc_tools()
    names = tool_names(live)
    ok = (
        names == EXPECTED_NAMES_WITH_MCP
        and len(registry.TOOL_SPECS) == 7
        and len(live) == 7
        and api_main._conversation_graph is None
    )
    detail = (
        f"注册表 {len(registry.TOOL_SPECS)} 个 / to_lc_tools {len(live)} 个：{names}；"
        f"新增={names[5:]}；图缓存已重置={api_main._conversation_graph is None}；耗时 {elapsed:.2f}s"
    )
    return record("T93-002", "A2 开关开热注册（7 个 / 追加末尾）", ok, detail)


def check_b2() -> bool:
    """B2：注册后前 5 个工具 schema 哈希仍等于基线（新增不改旧）。"""
    first_five = baseline_tools()
    digest = schema_hash(first_five)
    ok = tool_names(first_five) == BASELINE_NAMES and digest == BASELINE_SHA256
    detail = (
        f"前 5 个={tool_names(first_five)}；sha256={digest[:8]}…{digest[-4:]}（基线 c48c9d48…2a99）"
    )
    return record("T93-010", "B2 原 5 个工具 schema 不变", ok, detail)


def check_a3() -> bool:
    """A3：reset 后重建图，模型 bind_tools 真收到 7 个且含两个新工具（看实参，不看注册表）。"""
    api_main = import_api_main()
    api_main._conversation_graph = object()
    api_main.reset_conversation_graph()
    received = capture_graph_bind()
    names = tool_names(received)
    live = registry.to_lc_tools()
    same_objects = len(received) == len(live) and all(
        a is b for a, b in zip(received, live, strict=False)
    )
    ok = names == EXPECTED_NAMES_WITH_MCP and same_objects and api_main._conversation_graph is None
    detail = (
        f"bind_tools 收到 {len(names)} 个：{names}；含两个新工具="
        f"{all(name in names for name in NEW_TOOLS)}；与注册表同一批对象={same_objects}"
    )
    return record("T93-003", "A3 重建图后模型可见两个新工具", ok, detail)


def _spec_problems(name: str) -> list[str]:  # noqa: C901 - 逐字段对照契约，拆开反而更难读
    """单个新工具的 ToolSpec 校验（A4 的逐字段断言）。"""
    spec = registry.TOOL_SPECS.get(name)
    if spec is None:
        return [f"{name} 没登记"]
    problems: list[str] = []
    absent = [field for field in CONTRACT_FIELDS if not hasattr(spec, field)]
    if absent:
        problems.append(f"{name} 缺字段 {absent}")
    if spec.name != name:
        problems.append(f"{name} spec.name={spec.name}")
    if spec.permission != "external":
        problems.append(f"{name} permission={spec.permission!r}")
    if spec.requires_approval is not False:
        problems.append(f"{name} requires_approval={spec.requires_approval!r}（不是 False 本体）")
    if spec.feature_flag != "ENABLE_MCP":
        problems.append(f"{name} feature_flag={spec.feature_flag!r}")
    if spec.timeout_seconds != settings.MCP_TIMEOUT_SECONDS:
        problems.append(f"{name} timeout={spec.timeout_seconds}")
    if spec.max_output_tokens != settings.MCP_MAX_OUTPUT_TOKENS:
        problems.append(f"{name} max_output_tokens={spec.max_output_tokens}")
    if not spec.description:
        problems.append(f"{name} description 为空")
    schema = spec.input_schema if isinstance(spec.input_schema, dict) else {}
    if schema.get("type") != "object" or not schema.get("properties"):
        problems.append(f"{name} input_schema 不是带 properties 的对象：{schema}")
    if spec.cost_model.unit != "call" or spec.cost_model.unit_cost != 0.0:
        problems.append(f"{name} cost_model={spec.cost_model!r}")
    return problems


def check_a4() -> bool:
    """A4：新工具 ToolSpec 九字段齐全、permission=external、requires_approval is False。"""
    problems: list[str] = []
    runtime_fields = [field.name for field in dataclasses.fields(ToolSpec)]
    if runtime_fields != CONTRACT_FIELDS:
        problems.append(f"ToolSpec 字段与契约不一致：{runtime_fields}")
    if settings.MCP_TIMEOUT_SECONDS <= 0:
        problems.append("MCP_TIMEOUT_SECONDS 非正数")
    for name in NEW_TOOLS:
        problems.extend(_spec_problems(name))
    specs = {name: registry.TOOL_SPECS.get(name) for name in NEW_TOOLS}
    detail = (
        f"字段 {len(CONTRACT_FIELDS)}/9={CONTRACT_FIELDS}；"
        + "；".join(
            f"{name}(permission={spec.permission}, requires_approval={spec.requires_approval!r}, "
            f"flag={spec.feature_flag}, timeout={spec.timeout_seconds})"
            for name, spec in specs.items()
            if spec is not None
        )
        + (f"；问题={problems}" if problems else "")
    )
    ok = record("T93-004", "A4 新工具 ToolSpec 九字段 / permission / 审批", not problems, detail)
    _probe_spec_vs_visible_schema()
    return ok


def _probe_spec_vs_visible_schema() -> None:
    """C 类：注册表里的 `input_schema` / `description` 与"模型真正看到"的是不是同一份。"""
    drifts: list[str] = []

    def visible_of(name: str) -> dict[str, Any] | None:
        fn = next(
            (tool for tool in registry.to_lc_tools() if getattr(tool, "__name__", "") == name), None
        )
        return convert_to_openai_tool(fn)["function"] if fn is not None else None

    for name in NEW_TOOLS:
        spec = registry.TOOL_SPECS.get(name)
        visible = visible_of(name)
        if spec is None or visible is None:
            continue
        params = visible.get("parameters") or {}
        if params != spec.input_schema:
            extra_keys = sorted(set(spec.input_schema) - set(params))
            prop_diff = sorted(
                key
                for key in (spec.input_schema.get("properties") or {})
                if spec.input_schema["properties"][key] != (params.get("properties") or {}).get(key)
            )
            drifts.append(
                f"{name} 参数 schema 不一致（spec 多出 {extra_keys or '无'}；属性差异 {prop_diff}）"
            )
        if visible.get("description") != spec.description:
            drifts.append(
                f"{name} description 不一致（模型只见摘要 {len(visible.get('description') or '')} 字，"
                f"spec 存了 {len(spec.description)} 字含 Args/Returns）"
            )

    # 对照：内置 5 个工具是否本来就"spec == 模型可见"（若是，则上述漂移是本卡引入的）
    builtin_equal = []
    for name in BASELINE_NAMES:
        spec = registry.TOOL_SPECS.get(name)
        visible = visible_of(name)
        if spec is None or visible is None:
            builtin_equal.append(f"{name}=无法对照")
            continue
        builtin_equal.append(
            f"{name}={'一致' if visible.get('parameters') == spec.input_schema and visible.get('description') == spec.description else '不一致'}"
        )

    record_c(
        "C6",
        not drifts,
        f"内置 5 个 spec/模型可见对照：{builtin_equal}；新工具漂移：{drifts or '无'}；"
        "契约只要求 properties 即参数名，故记 OI 不判失败",
    )


async def check_a5() -> bool:
    """A5：两个新工具各调一次结果正确；非法输入返回可读错误。"""
    by_name = {getattr(tool, "__name__", ""): tool for tool in registry.to_lc_tools()}
    word_count = by_name.get("word_count")
    unit_convert = by_name.get("unit_convert")
    if word_count is None or unit_convert is None:
        return record("T93-005", "A5 两个新工具各调一次 / 非法输入", False, "工具没登记上")

    try:
        wc = await asyncio.wait_for(word_count(text="hello world"), timeout=30)
        uc = await asyncio.wait_for(unit_convert(1, "km", "m"), timeout=30)
        temp = await asyncio.wait_for(unit_convert(100, "c", "f"), timeout=30)
        bad_unit = await asyncio.wait_for(unit_convert(1, "xx", "yy"), timeout=30)
        bad_dim = await asyncio.wait_for(unit_convert(1, "km", "kg"), timeout=30)
        bad_args = await asyncio.wait_for(unit_convert(1), timeout=30)
    except BaseException as exc:  # noqa: BLE001
        return record(
            "T93-005",
            "A5 两个新工具各调一次 / 非法输入",
            False,
            f"调用抛了 {type(exc).__name__}: {exc}",
        )

    def readable(text: str) -> bool:
        return isinstance(text, str) and bool(text.strip()) and "Traceback" not in text

    ok = (
        wc == "字符数：11，词数：2，行数：1"
        and uc == "1 km = 1000 m"
        and temp == "100 c = 212 f"
        and readable(bad_unit)
        and "不支持" in bad_unit
        and readable(bad_dim)
        and "跨量纲" in bad_dim
        and readable(bad_args)
        and "不合法" in bad_args
    )
    detail = (
        f"word_count('hello world')={wc!r}；unit_convert(1,km,m)={uc!r}；"
        f"unit_convert(100,c,f)={temp!r}；非法单位={bad_unit!r}；跨量纲={bad_dim!r}；缺参={bad_args!r}"
    )
    return record("T93-005", "A5 两个新工具各调一次 / 非法输入", ok, detail)


async def check_a6() -> bool:
    """A6：杀掉 Server 再调 → 可读"不可用"；同一会话其它工具照常可用。"""
    by_name = {getattr(tool, "__name__", ""): tool for tool in registry.to_lc_tools()}
    word_count = by_name.get("word_count")
    calculator = by_name.get("calculator")
    if word_count is None or calculator is None:
        return record("T93-006", "A6 停 Server 后降级", False, "工具没登记上")

    targets = [pid for pid, cmdline in children_of(os.getpid()) if "mcp_server.py" in cmdline]
    if not targets:
        return record(
            "T93-006",
            "A6 停 Server 后降级",
            False,
            "找不到 MCP Server 子进程，无法制造'停服'（子进程未拉起或已被回收）",
        )
    for pid in targets:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as exc:
            return record("T93-006", "A6 停 Server 后降级", False, f"杀子进程失败：{exc}")
    gone = all(wait_gone(pid) for pid in targets)

    started = time.time()
    first = await asyncio.wait_for(word_count(text="after kill"), timeout=40)
    first_elapsed = time.time() - started
    second = await asyncio.wait_for(word_count(text="after kill again"), timeout=40)
    fallback = await asyncio.wait_for(calculator("1+1"), timeout=30)
    graph_tools = tool_names(capture_graph_bind())
    elapsed = time.time() - started

    ok = (
        gone
        and isinstance(first, str)
        and "不可用" in first
        and "Traceback" not in first
        and isinstance(second, str)
        and "不可用" in second
        and "= 2" in str(fallback)
        and graph_tools == EXPECTED_NAMES_WITH_MCP
    )
    detail = (
        f"杀子进程 {targets}（已退出={gone}）；首次调用（{first_elapsed:.2f}s）={first!r}；"
        f"再次调用={second!r}；同会话 calculator('1+1')={fallback!r}；"
        f"重建图仍可建（{len(graph_tools)} 个工具）；总耗时 {elapsed:.2f}s"
    )
    ok = record("T93-006", "A6 停 Server 后降级", ok, detail)
    record_c(
        "C1c",
        first_elapsed < 2.0,
        f"Server 已被杀，word_count 首次调用 {first_elapsed:.2f}s 才返回"
        + (
            "（连接已断也等满 MCP_TIMEOUT_SECONDS 才降级：单次卡顿时延 = 超时值）"
            if first_elapsed >= 2.0
            else "（断开即报错，符合预期）"
        ),
    )
    return ok


async def check_a7() -> bool:
    """A7：Server 启动失败（命令改 false）→ 一条 warning、不抛、5 个内置可用。"""
    api_main = import_api_main()
    logger = logging.getLogger("adaptive_agent")
    capture = _LogCapture()
    logger.addHandler(capture)
    saved_cmd, saved_args = settings.MCP_SERVER_CMD, settings.MCP_SERVER_ARGS
    settings.ENABLE_MCP = True
    settings.MCP_SERVER_CMD = "false"
    settings.MCP_SERVER_ARGS = ""
    error: BaseException | None = None
    started = time.time()
    try:
        await asyncio.wait_for(api_main._register_mcp_tools(), timeout=60)
    except BaseException as exc:  # noqa: BLE001
        error = exc
    finally:
        settings.MCP_SERVER_CMD, settings.MCP_SERVER_ARGS = saved_cmd, saved_args
        logger.removeHandler(capture)
    elapsed = time.time() - started

    warnings = [
        item
        for item in capture.records
        if item.levelno == logging.WARNING and "MCP 工具加载失败" in item.getMessage()
    ]
    other_warnings = [
        item.getMessage() for item in capture.records if item.levelno == logging.WARNING
    ]
    nodes = tool_names(registry.to_lc_tools())
    calculator = next(
        (tool for tool in registry.to_lc_tools() if tool.__name__ == "calculator"), None
    )
    try:
        still_usable = await asyncio.wait_for(calculator("2*3"), timeout=30) if calculator else None
    except BaseException as exc:  # noqa: BLE001
        still_usable = f"{type(exc).__name__}: {exc}"
    ok = (
        error is None
        and len(warnings) == 1
        and nodes == BASELINE_NAMES
        and "= 6" in str(still_usable)
        and elapsed < 30
    )
    detail = (
        f"命令=false 时未抛异常={error is None}；warning 条数={len(warnings)}"
        f"（文案={warnings[0].getMessage() if warnings else '无'}）；"
        f"全部 warning={other_warnings}；注册表回到 {len(nodes)} 个={nodes}；"
        f"内置 calculator('2*3')={still_usable!r}；耗时 {elapsed:.2f}s"
    )
    return record("T93-007", "A7 Server 启动失败不阻塞（一条 warning）", ok, detail)


CHILD_NO_FLAG = r"""
import asyncio, hashlib, json, os, sys
sys.path.insert(0, @@REPO@@)
from langchain_core.utils.function_calling import convert_to_openai_tool
from agent.adapters.tools import registry
from config import settings
import api_main

def digest(tools):
    schemas = [convert_to_openai_tool(t) for t in tools]
    return hashlib.sha256(
        json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()

async def main():
    await api_main._register_mcp_tools()
    tools = registry.to_lc_tools()
    kids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = open("/proc/%s/stat" % entry).read().split()
            if int(stat[3]) == os.getpid():
                kids.append(
                    open("/proc/%s/cmdline" % entry, "rb").read().replace(b"\0", b" ").decode()
                )
        except Exception:
            pass
    print("T93JSON:" + json.dumps({
        "names": [t.__name__ for t in tools],
        "hash": digest(tools),
        "flag": settings.ENABLE_MCP,
        "children": kids,
        "specs": list(registry.TOOL_SPECS),
    }))

asyncio.run(main())
"""


def check_a8_and_c5() -> None:
    """A8：关开关重启（全新子进程）回到 5 个；C5：此时子进程确实没被拉起。"""
    code = CHILD_NO_FLAG.replace("@@REPO@@", repr(str(REPO)))
    env = {key: value for key, value in os.environ.items() if key != "ENABLE_MCP"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    payload: dict[str, Any] | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("T93JSON:"):
            payload = json.loads(line[len("T93JSON:") :])
    if payload is None:
        record(
            "T93-008",
            "A8 关开关重启回到 5 个",
            False,
            f"子进程没输出结果（exit={proc.returncode}）stderr 尾：{proc.stderr[-300:]}",
        )
        record_c("C5", False, f"关 flag 子进程判定失败（exit={proc.returncode}）")
        return

    ok = (
        payload["flag"] is False
        and payload["names"] == BASELINE_NAMES
        and payload["hash"] == BASELINE_SHA256
        and payload["specs"] == BASELINE_NAMES
        and proc.returncode == 0
    )
    hash_note = "基线一致" if payload["hash"] == BASELINE_SHA256 else "与基线不一致"
    detail = (
        f"新进程 ENABLE_MCP={payload['flag']}；注册表 {len(payload['names'])} 个={payload['names']}；"
        f"sha256={payload['hash'][:8]}…{payload['hash'][-4:]}（{hash_note}）；"
        f"spec 键={payload['specs']}"
    )
    record("T93-008", "A8 关开关重启回到 5 个", ok, detail)

    mcp_children = [cmd for cmd in payload["children"] if "mcp_server.py" in cmd]
    load_logs = proc.stderr.count("MCP 工具加载完成")
    c5_ok = not mcp_children and load_logs == 0
    record_c(
        "C5",
        c5_ok,
        f"关 flag 启动后子进程={mcp_children or '无'}；'MCP 工具加载完成' 日志={load_logs} 条"
        + ("" if c5_ok else "（本应是 0：关 flag 不该拉起 Server）"),
    )


# ── C 类：边界与反例 ─────────────────────────────────────────────────

SLOW_SERVER = r"""
import time
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("t93-slow", log_level="WARNING")

@mcp.tool()
def slow_tool(seconds: float = 1.0) -> str:
    time.sleep(seconds)
    return "done"

@mcp.tool()
def fast_tool() -> str:
    return "fast-ok"

mcp.run(transport="stdio")
"""

NOISY_SERVER_CALL = r"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("t93-noisy", log_level="WARNING")

@mcp.tool()
def noisy() -> str:
    print("T93 GARBAGE: not JSON-RPC", flush=True)
    return "ok"

@mcp.tool()
def ping() -> str:
    return "pong"

mcp.run(transport="stdio")
"""

NOISY_SERVER_IMPORT = r"""
import sys
print("T93 GARBAGE AT IMPORT", flush=True)
sys.stdout.write("T93 MORE GARBAGE\n")
sys.stdout.flush()
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("t93-noisy-import", log_level="WARNING")

@mcp.tool()
def ping() -> str:
    return "pong"

mcp.run(transport="stdio")
"""


async def check_c1() -> None:
    """C1：Server 卡住 30s，MCP_TIMEOUT_SECONDS 是否生效。"""
    saved_timeout = settings.MCP_TIMEOUT_SECONDS
    settings.MCP_TIMEOUT_SECONDS = 2.0
    try:
        items = await asyncio.wait_for(
            mcp_client.load_mcp_tools(sys.executable, ["-c", SLOW_SERVER]), timeout=60
        )
        specs = {spec.name: (spec, fn) for spec, fn in items}
        spec = specs["slow_tool"][0]
        slow_fn = specs["slow_tool"][1]
        fast_fn = specs["fast_tool"][1]
        started = time.time()
        result = await asyncio.wait_for(slow_fn(seconds=30), timeout=40)
        elapsed = time.time() - started
        timeout_ok = (
            spec.timeout_seconds == 2.0
            and isinstance(result, str)
            and "超时" in result
            and "不可用" in result
            and 1.5 <= elapsed <= 6.0
        )
        record_c(
            "C1",
            timeout_ok,
            f"MCP_TIMEOUT_SECONDS=2 透传到 spec.timeout_seconds={spec.timeout_seconds}；"
            f"slow_tool(30s) 在 {elapsed:.2f}s 返回={result!r}",
        )

        started = time.time()
        other = await asyncio.wait_for(fast_fn(), timeout=40)
        other_elapsed = time.time() - started
        record_c(
            "C1b",
            "超时" not in str(other),
            f"同一 Server 的另一个瞬时工具 fast_tool 在 slow_tool 超时后调用："
            f"{other!r}（{other_elapsed:.2f}s）→ "
            + (
                "整条会话被在途调用占住，其它工具也超时（mark_unavailable 只标了超时的那一个）"
                if "超时" in str(other)
                else "其它工具不受影响"
            ),
        )
    except BaseException as exc:  # noqa: BLE001
        record_c("C1", False, f"探针异常 {type(exc).__name__}: {exc}")
    finally:
        settings.MCP_TIMEOUT_SECONDS = saved_timeout
        kill_children(os.getpid(), "t93-slow")
        await close_mcp_quietly()


async def check_c2() -> None:
    """C2：两次加载同名工具，第二次 register 必须报重名、不静默覆盖、无旁路写表。"""
    await close_mcp_quietly()
    try:
        first = await asyncio.wait_for(
            mcp_client.load_mcp_tools("python", ["scripts/mcp_server.py"]), timeout=60
        )
        for spec, fn in first:
            registry.register(spec.name, fn, spec)
        fn_before = registry._TOOL_FNS["word_count"]  # noqa: SLF001
        spec_before = registry.TOOL_SPECS["word_count"]
        count_before = len(registry.to_lc_tools())

        second = await asyncio.wait_for(
            mcp_client.load_mcp_tools("python", ["scripts/mcp_server.py"]), timeout=60
        )
        again = {spec.name: (spec, fn) for spec, fn in second}
        dup_errors: dict[str, str] = {}
        overwritten: list[str] = []
        for name in NEW_TOOLS:
            spec, fn = again[name]
            try:
                registry.register(name, fn, spec)
            except ValueError as exc:
                dup_errors[name] = str(exc)
            except BaseException as exc:  # noqa: BLE001 - 非 ValueError 也要暴露出来
                dup_errors[name] = f"{type(exc).__name__}: {exc}"
            else:
                overwritten.append(name)
        identity_kept = (
            registry._TOOL_FNS["word_count"] is fn_before  # noqa: SLF001
            and registry.TOOL_SPECS["word_count"] is spec_before
        )
        count_after = len(registry.to_lc_tools())
        dup_ok = (
            len(dup_errors) == len(NEW_TOOLS)
            and all("不能重复" in message for message in dup_errors.values())
            and identity_kept
            and not overwritten
            and count_after == count_before
        )

        # 无旁路：TOOL_SPECS / _TOOL_FNS 只允许在 registry.py 里写
        search = git(
            "grep", "-n", "-E", r"TOOL_SPECS\[[^]]+\]\s*=|_TOOL_FNS\[[^]]+\]\s*=", "--", "*.py"
        )
        bypass_hits = [
            line
            for line in search.stdout.splitlines()
            if line and "agent/adapters/tools/registry.py" not in line
        ]
        record_c(
            "C2",
            dup_ok and not bypass_hits,
            f"第二次 register 报重名={dup_errors}；静默覆盖={overwritten or '无'}；"
            f"原 fn/spec 未被覆盖={identity_kept}；注册表数量 {count_before}->{count_after}；"
            f"旁路直接写表={bypass_hits or '无'}",
        )
    except BaseException as exc:  # noqa: BLE001
        record_c("C2", False, f"探针异常 {type(exc).__name__}: {exc}")
    finally:
        drop_mcp_tools()
        await close_mcp_quietly()


async def check_c3() -> None:
    """C3：reset 后重建，TOOLS 兼容别名与 current_tools() 是否一致（需 MCP 工具已登记）。"""
    live_tools = cg.current_tools()
    alias = cg.TOOLS
    same_len = len(alias) == len(live_tools)
    same_objs = same_len and all(a is b for a, b in zip(alias, live_tools, strict=False))
    record_c(
        "C3",
        same_objs,
        f"TOOLS 快照 {len(alias)} 个={tool_names(alias)}；current_tools() {len(live_tools)} 个="
        f"{tool_names(live_tools)}；同一批对象={same_objs}"
        + (
            "；热注册后兼容别名仍是旧快照（已在源码注释标记 deprecated，"
            "但 scripts/smoke_tools_schema.py 等仍读 TOOLS，会打印与模型可见不一致的指纹）"
            if not same_objs
            else ""
        ),
    )


async def check_c4() -> None:
    """C4：Server 往 stdout 打非协议内容 → 客户端不崩（调用仍能拿到结果）。"""
    problems: list[str] = []
    try:
        items = await asyncio.wait_for(
            mcp_client.load_mcp_tools(sys.executable, ["-c", NOISY_SERVER_CALL]), timeout=60
        )
        by_name = {spec.name: fn for spec, fn in items}
        noisy = await asyncio.wait_for(by_name["noisy"](), timeout=30)
        ping = await asyncio.wait_for(by_name["ping"](), timeout=30)
        if noisy != "ok" or ping != "pong":
            problems.append(f"脏输出后调用结果异常：noisy={noisy!r} ping={ping!r}")
    except BaseException as exc:  # noqa: BLE001
        problems.append(f"调用期脏输出让客户端异常：{type(exc).__name__}: {exc}")
    await close_mcp_quietly()

    import_problem = ""
    try:
        items = await asyncio.wait_for(
            mcp_client.load_mcp_tools(sys.executable, ["-c", NOISY_SERVER_IMPORT]), timeout=60
        )
        by_name = {spec.name: fn for spec, fn in items}
        if await asyncio.wait_for(by_name["ping"](), timeout=30) != "pong":
            import_problem = "握手前脏输出后调用结果异常"
    except BaseException as exc:  # noqa: BLE001
        import_problem = f"握手前脏输出让加载失败：{type(exc).__name__}: {exc}"
    await close_mcp_quietly()

    record_c(
        "C4",
        not problems,
        f"调用期脏输出（工具 print 到 stdout）：{problems or '客户端容忍，noisy=ok / ping=pong'}；"
        f"握手前脏输出（import 期 print）：{import_problem or '容忍，ping=pong'}；"
        "注：SDK 把 'Failed to parse JSONRPC message' 的 traceback 打到 stderr，属噪声不崩",
    )


# ── 主流程 ────────────────────────────────────────────────────────────


async def main() -> int:
    started = time.time()
    head = git_out("rev-parse", "--short", "HEAD")
    print(f"仓库：{REPO}")
    print(f"HEAD：{head}（本卡开工基线 {BASELINE_COMMIT}）")
    print("-" * 78, flush=True)

    # 静态不变量先跑
    check_b3()
    check_b4()
    check_b5()

    a1_ok = check_a1()
    a2_ok = False
    if not a1_ok:
        skip("T93-002", "A2 开关开热注册（7 个 / 追加末尾）", "A1 未通过，基线不干净")
    else:
        a2_ok = await check_a2()

    if a2_ok:
        check_b2()
        check_a4()
        check_a3()
        await check_a5()
        await check_a6()
        await check_c3()
    else:
        for case_id, title in [
            ("T93-010", "B2 原 5 个工具 schema 不变"),
            ("T93-003", "A3 重建图后模型可见两个新工具"),
            ("T93-004", "A4 新工具 ToolSpec 九字段 / permission / 审批"),
            ("T93-005", "A5 两个新工具各调一次 / 非法输入"),
            ("T93-006", "A6 停 Server 后降级"),
        ]:
            skip(case_id, title, "A2 未通过（热注册没成功）")
        record_c("C3", False, "MCP 工具未登记，无法比较 TOOLS 与 current_tools")

    drop_mcp_tools()
    await close_mcp_quietly()

    check_a8_and_c5()
    await check_a7()

    # C 类独立探针（各自清理）
    await check_c1()
    await check_c2()
    await check_c4()

    # 兜底清理：内存里不留 MCP 工具 / 不留子进程
    drop_mcp_tools()
    await close_mcp_quietly()
    leftover = [cmdline for _, cmdline in children_of(os.getpid()) if "mcp_server.py" in cmdline]
    if leftover:
        record_c("C8", False, f"收尾后仍有 MCP 子进程残留：{leftover}")

    check_b1()

    settings.ENABLE_MCP = False  # 还原进程内开关（不影响子进程）

    failed_cases = [case_id for case_id, _, ok, _ in RESULTS if not ok]
    passed = len(RESULTS) - len(failed_cases)
    issues = [case_id for case_id, ok, _ in C_PROBES if not ok]
    print("-" * 78)
    print(
        f"耗时 {time.time() - started:.1f}s；A/B 共 {len(RESULTS)} 项，C 类探针 {len(C_PROBES)} 项"
    )
    if failed_cases:
        print(f"失败：{', '.join(failed_cases)}")
    print(f"C 类 open_issue：{', '.join(issues) if issues else '无'}")
    print(f"RESULT: {passed} passed, {len(failed_cases)} failed")
    return 1 if failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
