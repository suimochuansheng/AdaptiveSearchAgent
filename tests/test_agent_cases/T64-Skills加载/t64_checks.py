#!/usr/bin/env python
"""G13 / T6-4 Skills 加载器 —— 测试窗口（test-G13）**独立验收**。

用法（仓库根目录，任意 cwd 均可）：

    export PRE_COMMIT_HOME="$PWD/.pre-commit-cache"
    poetry run python "tests/test_agent_cases/T64-Skills加载/t64_checks.py"

覆盖（判分标准 A/B 必过，C 只记 open_issue）：

    A1  开关关：注册表 5 个、无 list_skills / load_skill、schema 哈希 c48c9d48…2a99
    A2  开关开：7 个工具、to_lc_tools() 末尾是 list_skills / load_skill
    A3  list_skills() 条目数 = skills/ 下有效技能数；每条 name + when_to_use
    A4  load_skill('rag-answer-format') 返回正文，且不含 frontmatter
    A5  未知名字 -> 可读错误且列出可用名字
    A6  热加载：新建 SKILL.md 不重启即 list 到；删除后 list 不到
    A7  路径穿越四种全拒（../、../../、绝对路径、含 / 的名字）
    A8  正文 > SKILL_MAX_BODY_TOKENS -> 截断且带可辨认标记
    A9  坏文件隔离：无 frontmatter 的 SKILL.md 被跳过 + warning，其余技能仍可用
    B1  src/conversation_graph.py 零改动（v1 不做索引注入，属设计取舍）
    B2  没有把技能正文拼进系统提示（b59be16 diff 无 SYSTEM_PROMPT / stable 注入）
    B3  pytest 225 / collect 229/233 / lint-imports 3 kept
    B4  台账里没有开发窗口写的 T64 行（本卡开发未写台账）
    B5  未改契约 / 前端 / SUBAGENT_* 配置；config.py 只新增 4 个 SKILLS 字段
    C1  两个技能同名：scan 是否只留一个并 warning
    C2  when_to_use 缺失时 list 展示是否可读
    C3  SKILLS_DIR 指向不存在的目录：可读提示而非崩
    C4  名字大小写混用（RAG-answer-format）：拒绝或规范化，二者其一
    C5  技能文件含非 UTF-8 字节：scan 跳过而不是抛异常

退出码：
    0 = A1~A9 / B1~B5 全部通过（C 类发现只打 [C-PASS]/[C-ISSUE]，不影响退出码）
    1 = A/B 至少一项失败

为什么不是 `test_*.py`：
    会被 pytest 收集，`--collect-only` 的收集数改变，`docs/baseline.md` §2 的
    nodeids 哈希失效（见 tests/test_agent_cases/README.md）。

只读承诺：
    不改业务代码，不 import 开发窗口的 `scripts/smoke_skills.py`，不复用它的断言；
    所有判定都由本脚本自己构造的输入 + 当前代码的返回值取证。
    唯一写动作：A6/A8/A9/C1/C2/C5 在 `skills/` 下建临时技能目录
    （`tmp-probe` / `tmp-t64-*`），每个子进程 finally 自删，父进程开跑与收尾各清一遍；
    不碰 3 个正式技能，不写台账。
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]
LEDGER = REPO / "tests" / "test_agent_cases" / "台账用例.csv"
LEDGER_REL = "tests/test_agent_cases/台账用例.csv"
SKILLS_ROOT = REPO / "skills"

#: 本卡开工 HEAD（含 0a `OI-T94-001`）与交付提交
BASE_COMMIT = "889bb73"
CARD_COMMIT = "b59be16"

#: 开关关闭时的 schema 指纹（T6-0 冻结，见 scripts/smoke_tools_schema.py）
BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
BASELINE_BYTES = 2558
BASELINE_NAMES = [
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
]
SKILLS_TOOL_NAMES = ("list_skills", "load_skill")

#: 本脚本可能创建的临时技能目录（只删这些，且只删未被 git 跟踪的）
TEMP_DIRS = (
    "tmp-probe",
    "tmp-t64-big",
    "tmp-t64-bad",
    "tmp-t64-dup-a",
    "tmp-t64-dup-b",
    "tmp-t64-nowtu",
    "tmp-t64-utf8",
)

TRUNCATED_HINT = "技能正文已截断"

RESULTS: list[tuple[str, str, bool, str]] = []
C_PROBES: list[tuple[str, bool, str]] = []

TITLES: dict[str, str] = {
    "T64-001": "A1 开关关：注册表 5 个 / 无 list_skills+load_skill / schema 哈希不变",
    "T64-002": "A2 开关开：7 个工具且 to_lc_tools() 末尾是 list_skills / load_skill",
    "T64-003": "A3 list_skills() 条目数 = 有效技能数且每条含 name + when_to_use",
    "T64-004": "A4 load_skill('rag-answer-format') 返回正文且不含 frontmatter",
    "T64-005": "A5 未知技能名 -> 可读错误且列出可用名字",
    "T64-006": "A6 热加载：新建即 list 到，删除即消失（不重启）",
    "T64-007": "A7 路径穿越四种全拒",
    "T64-008": "A8 超大正文截断到 SKILL_MAX_BODY_TOKENS 且带标记",
    "T64-009": "A9 坏文件隔离：跳过 + warning，其余技能可用",
    "T64-010": "B1 src/conversation_graph.py 零改动（v1 不做索引注入）",
    "T64-011": "B2 未把技能正文拼进系统提示（只允许工具形式）",
    "T64-012": "B3 pytest 225 / collect 229/233 / lint-imports 3 kept",
    "T64-013": "B4 台账无开发窗口写的 T64 行",
    "T64-014": "B5 未改契约/前端/SUBAGENT_*，config 只新增 4 个 SKILLS 字段",
}


# ── 结果输出 ──────────────────────────────────────────────────────────


def record(case_id: str, title: str, ok: bool, evidence: str) -> None:
    """记一条 A/B 结果并立即打印。"""
    RESULTS.append((case_id, title, ok, evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {case_id} {title} — {evidence}")


def record_c(probe: str, ok: bool, evidence: str) -> None:
    """记一条 C 类探针；C 类发现只打 [C-ISSUE]，不影响退出码。"""
    C_PROBES.append((probe, ok, evidence))
    print(f"[{'C-PASS' if ok else 'C-ISSUE'}] {probe} — {evidence}")


def guarded(case_id: str, fn: Any) -> None:
    """跑一项 A/B 检查：异常转成 [FAIL]，不让一项问题吞掉整份报告。"""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - 验收脚本要指出“哪项失败”，不是打栈
        record(
            case_id, TITLES.get(case_id, case_id), False, f"脚本异常：{type(exc).__name__}: {exc}"
        )


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ── 外部命令 / 子进程 ────────────────────────────────────────────────


def git(*args: str) -> str:
    """在仓库根目录跑 git，返回 stdout+stderr（去首尾空白）。"""
    proc = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=180, check=False
    )
    return (proc.stdout + proc.stderr).strip()


def run_cmd(cmd: list[str], timeout: int) -> tuple[int, str]:
    """跑一个外部命令，返回 (exit_code, stdout+stderr)。"""
    proc = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def last_lines(text: str, count: int) -> str:
    """取最后 count 行非空行，用 ` | ` 连接（用于回贴末行）。"""
    lines = [line for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-count:])


#: 子进程序言：emit() 打印机器可读载荷；capture_warnings() 抓 "adaptive_agent" 日志。
CHILD_EMIT = """
import json
import logging


def emit(obj):
    print("__T64__" + json.dumps(obj, ensure_ascii=True))


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def capture_warnings():
    handler = _Capture()
    logging.getLogger("adaptive_agent").addHandler(handler)
    return handler
"""


def emit_child(
    code: str, env_extra: dict[str, str] | None = None, timeout: int = 240
) -> dict[str, Any]:
    """在干净子进程里跑一段取证代码，解析它打印的 `__T64__{json}` 载荷。

    每个状态都用全新解释器：`config.settings` 与环境变量在导入时求值，
    同进程里切开关会污染后续判据。
    """
    env = os.environ.copy()
    for key in ("ENABLE_SKILLS", "SKILLS_DIR", "SKILLS_INDEX_MAX", "SKILL_MAX_BODY_TOKENS"):
        env.pop(key, None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__T64__"):
            return json.loads(line[len("__T64__") :])
    raise RuntimeError(
        f"子进程没有输出 __T64__ 载荷（exit={proc.returncode}）："
        f"{(proc.stderr or proc.stdout)[-600:]}"
    )


# ── 独立取证用的本地解析（不 import 开发窗口的 loader 判定）──────────


def iter_skill_files() -> list[Path]:
    """枚举 `skills/*/SKILL.md`（与实现同一约定，但由本脚本自己解析）。"""
    return sorted(SKILLS_ROOT.glob("*/SKILL.md"))


def independent_index() -> dict[str, dict[str, str]]:
    """独立解析 frontmatter，返回“有效技能”索引：name -> {description, when_to_use}。

    有效 = 有 `---` 包裹的 frontmatter、name 匹配 `[a-z0-9-]+`、description 与
    when_to_use 都非空。数量即 A3 的期望条目数。
    """
    index: dict[str, dict[str, str]] = {}
    for path in iter_skill_files():
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?", text, re.S)
        if not match:
            continue
        meta = yaml.safe_load(match.group(1))
        if not isinstance(meta, dict):
            continue
        name = str(meta.get("name") or "").strip()
        description = str(meta.get("description") or "").strip()
        when_to_use = str(meta.get("when_to_use") or "").strip()
        if not re.fullmatch(r"[a-z0-9-]+", name):
            continue
        if not description or not when_to_use:
            continue
        index[name] = {"description": description, "when_to_use": when_to_use}
    return index


def independent_body(path: Path) -> str:
    """独立切出正文（去掉 frontmatter），用于 A4 对照。"""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\r?\n.*?\r?\n---\r?\n?(.*)$", text, re.S)
    return (match.group(1) if match else text).strip()


def cleanup_temp() -> list[str]:
    """删掉本脚本可能残留的临时技能目录；被 git 跟踪的目录绝不删。"""
    removed: list[str] = []
    for name in TEMP_DIRS:
        path = SKILLS_ROOT / name
        if not path.exists():
            continue
        tracked = git("ls-files", "--", str(path.relative_to(REPO)))
        if tracked.strip():
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(name)
    return removed


# ── A1 / A2：开关矩阵 ────────────────────────────────────────────────

CHILD_FLAG_OFF = (
    CHILD_EMIT
    + """
import hashlib

from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.adapters.tools import registry
from config import settings

schemas = [convert_to_openai_tool(tool) for tool in registry.to_lc_tools()]
canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
emit({
    "flag": settings.ENABLE_SKILLS,
    "count": len(registry.TOOL_SPECS),
    "names": [schema["function"]["name"] for schema in schemas],
    "sha256": hashlib.sha256(canonical).hexdigest(),
    "bytes": len(canonical),
})
"""
)

CHILD_FLAG_ON = (
    CHILD_EMIT
    + """
import hashlib

from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.adapters.tools import registry
from agent.adapters.tools.skills_tools import register_skills_tools
from config import settings

first = register_skills_tools()
second = register_skills_tools()
schemas = [convert_to_openai_tool(tool) for tool in registry.to_lc_tools()]
canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
emit({
    "flag": settings.ENABLE_SKILLS,
    "first": first,
    "second": second,
    "count": len(registry.TOOL_SPECS),
    "tools": len(schemas),
    "names": [schema["function"]["name"] for schema in schemas],
    "sha256": hashlib.sha256(canonical).hexdigest(),
})
"""
)


def check_a1() -> None:
    """开关关：5 个内置工具、无 Skills 工具、schema 哈希仍是冻结基线。"""
    data = emit_child(CHILD_FLAG_OFF, {"ENABLE_SKILLS": "0"})
    names = list(data["names"])
    invisible = [name for name in SKILLS_TOOL_NAMES if name in names]
    ok = (
        data["flag"] is False
        and data["count"] == 5
        and names == BASELINE_NAMES
        and data["sha256"] == BASELINE_SHA256
        and data["bytes"] == BASELINE_BYTES
        and not invisible
    )
    record(
        "T64-001",
        TITLES["T64-001"],
        ok,
        f"flag={data['flag']}；注册表={data['count']}；names={','.join(names)}；"
        f"skills 工具可见={invisible or '无'}；sha256={data['sha256'][:12]}…；"
        f"bytes={data['bytes']}（基线 {BASELINE_BYTES}）",
    )


def check_a2() -> None:
    """开关开：注册表 7 个，两个 Skills 工具追加在 to_lc_tools() 末尾。"""
    data = emit_child(CHILD_FLAG_ON, {"ENABLE_SKILLS": "1"})
    names = list(data["names"])
    ok = (
        data["flag"] is True
        and data["first"] is True
        and data["count"] == 7
        and data["tools"] == 7
        and names[:5] == BASELINE_NAMES
        and tuple(names[-2:]) == SKILLS_TOOL_NAMES
        and len(names) == 7
    )
    record(
        "T64-002",
        TITLES["T64-002"],
        ok,
        f"注册表={data['count']}；to_lc_tools={data['tools']}；注册返回={data['first']}/"
        f"重复注册={data['second']}；last={tuple(names[-2:])}；names={','.join(names)}",
    )


# ── A3 ~ A5、A7、C4：索引 / 加载 / 错误文案 ──────────────────────────

CHILD_LIST = (
    CHILD_EMIT
    + """
from agent.adapters.tools.skills_tools import list_skills
from config import settings

emit({
    "flag": settings.ENABLE_SKILLS,
    "index_max": settings.SKILLS_INDEX_MAX,
    "out": list_skills(),
})
"""
)

CHILD_LOAD = (
    CHILD_EMIT
    + """
from agent.adapters.tools.skills_tools import load_skill
from config import settings

names = [
    "rag-answer-format",
    "no-such-skill",
    "../etc/passwd",
    "../../etc/passwd",
    "/etc/passwd",
    "foo/bar",
    "RAG-answer-format",
]
emit({
    "flag": settings.ENABLE_SKILLS,
    "cap": settings.SKILL_MAX_BODY_TOKENS,
    "results": {name: load_skill(name) for name in names},
})
"""
)


def check_a3() -> None:
    """索引条目数 = 磁盘有效技能数；每行都是 `name — when_to_use`。"""
    data = emit_child(CHILD_LIST, {"ENABLE_SKILLS": "1"})
    out = data["out"]
    index = independent_index()
    lines = [line for line in out.splitlines() if line.strip()]
    parsed: dict[str, str] = {}
    bad_lines: list[str] = []
    for line in lines:
        match = re.match(r"^([a-z0-9-]+) — (.+)$", line)
        if match:
            parsed[match.group(1)] = match.group(2).strip()
        else:
            bad_lines.append(line)
    missing_wtu = [name for name, when in parsed.items() if not when]
    when_mismatch = [
        name
        for name in parsed
        if name in index and parsed[name] != " ".join(index[name]["when_to_use"].split())
    ]
    ok = (
        data["flag"] is True
        and len(index) == 3
        and not bad_lines
        and len(parsed) == len(index)
        and set(parsed) == set(index)
        and not missing_wtu
        and not when_mismatch
    )
    record(
        "T64-003",
        TITLES["T64-003"],
        ok,
        f"list 输出 {len(parsed)} 条；磁盘有效技能 {len(index)} 个"
        f"（{'/'.join(sorted(index))}）；name+when_to_use 齐备={not missing_wtu}；"
        f"与磁盘 when_to_use 一致={not when_mismatch}；坏行={bad_lines or '无'}；"
        f"SKILLS_INDEX_MAX={data['index_max']}",
    )


def check_a4() -> None:
    """load_skill 返回正文：无 frontmatter 分隔符行、无 name/description 键。"""
    data = emit_child(CHILD_LOAD, {"ENABLE_SKILLS": "1"})
    out = str(data["results"]["rag-answer-format"])
    lines = out.splitlines()
    separator_lines = [line for line in lines if line.startswith("---")]
    key_lines = [line for line in lines if re.match(r"^(name|description|when_to_use)\s*:", line)]
    expected = independent_body(SKILLS_ROOT / "rag-answer-format" / "SKILL.md")
    ok = (
        bool(out.strip())
        and not separator_lines
        and not key_lines
        and out.strip() == expected
        and out.lstrip().startswith("#")
    )
    record(
        "T64-004",
        TITLES["T64-004"],
        ok,
        f"返回 {len(out)} 字；首行={lines[0][:24] if lines else '空'!r}；"
        f"--- 开头行={len(separator_lines)}；name/description 键行={len(key_lines)}；"
        f"与文件正文一致={out.strip() == expected}",
    )


def check_a5() -> None:
    """未知名字：可读错误 + 列出全部可用名字 + 指路 list_skills。"""
    data = emit_child(CHILD_LOAD, {"ENABLE_SKILLS": "1"})
    out = str(data["results"]["no-such-skill"])
    index = independent_index()
    names_listed = [name for name in index if name in out]
    ok = (
        "未知技能" in out
        and "no-such-skill" in out
        and len(names_listed) == len(index)
        and "list_skills" in out
        and "Traceback" not in out
    )
    record(
        "T64-005",
        TITLES["T64-005"],
        ok,
        f"文案={out[:74]}…；含未知名且列全 {len(names_listed)}/{len(index)} 个可用名；"
        f"指路 list_skills={'list_skills' in out}",
    )


def check_a7() -> None:
    """路径穿越四种输入全拒，且没有读到 /etc/passwd。"""
    data = emit_child(CHILD_LOAD, {"ENABLE_SKILLS": "1"})
    cases = ["../etc/passwd", "../../etc/passwd", "/etc/passwd", "foo/bar"]
    results = {name: str(data["results"][name]) for name in cases}
    verdicts = []
    for name in cases:
        out = results[name]
        rejected = out.startswith("已拒绝") or out.startswith("未知技能")
        leaked = "root:" in out or out.lstrip().startswith("# ") and "passwd" in out.lower()
        verdicts.append((name, rejected, leaked))
    ok = all(rejected and not leaked for _, rejected, leaked in verdicts)
    record(
        "T64-007",
        TITLES["T64-007"],
        ok,
        "；".join(
            f"{name!r}->{'拒绝' if rejected else '未拒'}"
            f"{'/读到内容' if leaked else ''}"
            f"（{results[name][:18]}…）"
            for name, rejected, leaked in verdicts
        ),
    )


# ── A6 / A8 / A9：热加载 / 截断 / 坏文件 ─────────────────────────────

CHILD_HOT = (
    CHILD_EMIT
    + r"""
import os
import shutil
from pathlib import Path

from agent.adapters.tools.skills_tools import list_skills

probe = Path("skills") / "tmp-probe"
payload = {}
try:
    payload["before"] = list_skills()
    if probe.exists():
        shutil.rmtree(probe)
    probe.mkdir(parents=True)
    (probe / "SKILL.md").write_text(os.environ["T64_SKILL_TEXT"], encoding="utf-8")
    payload["after_create"] = list_skills()
finally:
    shutil.rmtree(probe, ignore_errors=True)
payload["after_delete"] = list_skills()
emit(payload)
"""
)

CHILD_BIG = (
    CHILD_EMIT
    + r"""
import os
import shutil
from pathlib import Path

from agent.adapters.skills import loader
from config import settings
from src.context_manager import estimate_tokens

probe = Path("skills") / "tmp-t64-big"
payload = {}
try:
    probe.mkdir(parents=True, exist_ok=True)
    body = os.environ["T64_BIG_BODY"]
    content = os.environ["T64_FRONTMATTER"] + "\n" + body + "\n"
    (probe / "SKILL.md").write_text(content, encoding="utf-8")
    text = loader.load("tmp-t64-big")
    payload["tokens"] = estimate_tokens(text)
    payload["cap"] = settings.SKILL_MAX_BODY_TOKENS
    payload["marker"] = "技能正文已截断" in text
    payload["body_chars"] = len(body)
    payload["out_chars"] = len(text)
    payload["head_ok"] = text.lstrip().startswith(body[:12])
    payload["tail"] = text[-40:]
finally:
    shutil.rmtree(probe, ignore_errors=True)
emit(payload)
"""
)

CHILD_BAD = (
    CHILD_EMIT
    + r"""
import os
import shutil
from pathlib import Path

from agent.adapters.skills import loader

capture = capture_warnings()
probe = Path("skills") / "tmp-t64-bad"
payload = {}
try:
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "SKILL.md").write_text(os.environ["T64_BAD_TEXT"], encoding="utf-8")
    payload["before"] = [spec.name for spec in loader.scan()]
    payload["after"] = [spec.name for spec in loader.scan()]
    payload["cost_guard"] = loader.load("cost-guard")
finally:
    shutil.rmtree(probe, ignore_errors=True)
payload["warnings"] = capture.messages
emit(payload)
"""
)


def check_a6() -> None:
    """同一进程里新建 -> list 立刻多一条，删除 -> list 立刻少一条。"""
    probe_text = "\n".join(
        [
            "---",
            "name: tmp-probe",
            "description: 热加载探测用临时技能",
            "when_to_use: 热加载探测",
            "---",
            "",
            "# 探测正文",
            "",
            "这行只用于验证不重启进程即可看到新技能。",
            "",
        ]
    )
    data = emit_child(CHILD_HOT, {"ENABLE_SKILLS": "1", "T64_SKILL_TEXT": probe_text})
    before = [line for line in data["before"].splitlines() if line.strip()]
    after_create = [line for line in data["after_create"].splitlines() if line.strip()]
    after_delete = [line for line in data["after_delete"].splitlines() if line.strip()]
    created_line = next((line for line in after_create if line.startswith("tmp-probe — ")), "")
    ok = (
        len(before) == 3
        and not any("tmp-probe" in line for line in before)
        and len(after_create) == 4
        and created_line == "tmp-probe — 热加载探测"
        and after_delete == before
    )
    record(
        "T64-006",
        TITLES["T64-006"],
        ok,
        f"新建前={len(before)} 条；新建后={len(after_create)} 条且命中"
        f"{created_line or '未命中'}；删除后={len(after_delete)} 条且与新建前相同"
        f"={after_delete == before}（同一子进程，未重启）",
    )


def check_a8() -> None:
    """正文远超上限：截断到 cap 内，保留正文开头，带可辨认标记。"""
    body = "技能正文占位" * 600  # 3600 个 CJK 字符 -> 约 3600 tokens，> 默认 2000
    frontmatter = "\n".join(
        [
            "---",
            "name: tmp-t64-big",
            "description: 超大正文探测",
            "when_to_use: 截断探测",
            "---",
        ]
    )
    data = emit_child(
        CHILD_BIG,
        {"ENABLE_SKILLS": "1", "T64_BIG_BODY": body, "T64_FRONTMATTER": frontmatter},
    )
    ok = (
        data["cap"] == 2000
        and data["tokens"] <= data["cap"]
        and data["marker"] is True
        and data["out_chars"] < data["body_chars"]
        and data["head_ok"] is True
    )
    record(
        "T64-008",
        TITLES["T64-008"],
        ok,
        f"原文 {data['body_chars']} 字 -> 返回 {data['out_chars']} 字 / "
        f"{data['tokens']} tokens <= cap={data['cap']}；"
        f"标记「{TRUNCATED_HINT}」={data['marker']}；保留开头={data['head_ok']}；"
        f"尾部={data['tail']!r}",
    )


def check_a9() -> None:
    """无 frontmatter 的 SKILL.md：scan 跳过 + 一条 warning，其余技能照常可读。"""
    bad_text = "# 坏文件\n\n这个 SKILL.md 故意没有 frontmatter，用于验证坏文件隔离。\n"
    data = emit_child(CHILD_BAD, {"ENABLE_SKILLS": "1", "T64_BAD_TEXT": bad_text})
    before = list(data["before"])
    after = list(data["after"])
    warnings = [str(msg) for msg in data["warnings"]]
    hit = [msg for msg in warnings if "tmp-t64-bad" in msg and "frontmatter" in msg]
    cost_guard = str(data["cost_guard"])
    ok = (
        len(after) == len(before) == 3
        and "tmp-t64-bad" not in " ".join(after)
        and bool(hit)
        and bool(cost_guard.strip())
        and "Traceback" not in cost_guard
    )
    record(
        "T64-009",
        TITLES["T64-009"],
        ok,
        f"含坏文件时 scan 仍返回 {len(after)} 条（{','.join(after)}）；"
        f"坏目录未入索引={'tmp-t64-bad' not in ' '.join(after)}；"
        f"warning 命中={hit[0][:60] if hit else '无'}…；cost-guard 正文可读="
        f"{bool(cost_guard.strip())}（{len(cost_guard)} 字）",
    )


# ── C 类探针 ────────────────────────────────────────────────────────

CHILD_DUP = (
    CHILD_EMIT
    + r"""
import os
import shutil
from pathlib import Path

from agent.adapters.skills import loader

capture = capture_warnings()
dirs = [Path("skills") / "tmp-t64-dup-a", Path("skills") / "tmp-t64-dup-b"]
payload = {}
try:
    for index, probe in enumerate(dirs):
        probe.mkdir(parents=True, exist_ok=True)
        (probe / "SKILL.md").write_text(
            os.environ[f"T64_DUP_TEXT_{index}"], encoding="utf-8"
        )
    specs = loader.scan()
    payload["names"] = [spec.name for spec in specs]
    payload["dup"] = [spec.name for spec in specs].count("dup-probe")
    payload["dup_when"] = next(
        (spec.when_to_use for spec in specs if spec.name == "dup-probe"), ""
    )
finally:
    for probe in dirs:
        shutil.rmtree(probe, ignore_errors=True)
payload["warnings"] = capture.messages
emit(payload)
"""
)

CHILD_NOWTU = (
    CHILD_EMIT
    + r"""
import os
import shutil
from pathlib import Path

from agent.adapters.skills import loader
from agent.adapters.tools.skills_tools import list_skills

capture = capture_warnings()
probe = Path("skills") / "tmp-t64-nowtu"
payload = {}
try:
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "SKILL.md").write_text(os.environ["T64_NOWTU_TEXT"], encoding="utf-8")
    payload["names"] = [spec.name for spec in loader.scan()]
    payload["listing"] = list_skills()
    payload["cost_guard"] = loader.load("cost-guard")
finally:
    shutil.rmtree(probe, ignore_errors=True)
payload["warnings"] = capture.messages
emit(payload)
"""
)

CHILD_NO_DIR = (
    CHILD_EMIT
    + """
from agent.adapters.skills import loader
from agent.adapters.tools.skills_tools import list_skills, load_skill
from config import settings

settings.SKILLS_DIR = "skills-does-not-exist-t64"
emit({
    "scan": [spec.name for spec in loader.scan()],
    "listing": list_skills(),
    "load": load_skill("rag-answer-format"),
})
"""
)

CHILD_NON_UTF8 = (
    CHILD_EMIT
    + r"""
import shutil
from pathlib import Path

from agent.adapters.skills import loader

capture = capture_warnings()
probe = Path("skills") / "tmp-t64-utf8"
payload = {}
try:
    probe.mkdir(parents=True, exist_ok=True)
    raw = (
        b"---\nname: tmp-t64-utf8\ndescription: probe\nwhen_to_use: probe\n---\n"
        + b"\xff\xfe\x80\x81"
    )
    (probe / "SKILL.md").write_bytes(raw)
    payload["names"] = [spec.name for spec in loader.scan()]
    payload["cost_guard"] = loader.load("cost-guard")
finally:
    shutil.rmtree(probe, ignore_errors=True)
payload["warnings"] = capture.messages
emit(payload)
"""
)


def check_c() -> None:
    """C1~C5：边界行为探针，只记录现象，不判 A/B 失败。"""
    # C1 同名技能
    dup_texts = [
        "\n".join(
            [
                "---",
                "name: dup-probe",
                "description: 重名探测 A",
                "when_to_use: 重名探测 A",
                "---",
                "",
                "# A",
                "",
            ]
        ),
        "\n".join(
            [
                "---",
                "name: dup-probe",
                "description: 重名探测 B",
                "when_to_use: 重名探测 B",
                "---",
                "",
                "# B",
                "",
            ]
        ),
    ]
    dup = emit_child(
        CHILD_DUP,
        {"ENABLE_SKILLS": "1", "T64_DUP_TEXT_0": dup_texts[0], "T64_DUP_TEXT_1": dup_texts[1]},
    )
    dup_warnings = [str(msg) for msg in dup["warnings"]]
    dup_hit = [msg for msg in dup_warnings if "重名" in msg]
    record_c(
        "T64-C1",
        bool(dup_hit) and dup["dup"] == 1 and len(dup["names"]) == 4,
        f"两个同名目录 -> scan 返回 {len(dup['names'])} 条，dup-probe 出现 {dup['dup']} 次；"
        f"保留的 when_to_use={dup['dup_when']!r}；warning={dup_hit[0][:70] if dup_hit else '无'}…",
    )

    # C2 缺 when_to_use
    nowtu_text = "\n".join(
        [
            "---",
            "name: nowtu-probe",
            "description: 只有 description，没有 when_to_use",
            "---",
            "",
            "# 缺字段",
            "",
        ]
    )
    nowtu = emit_child(CHILD_NOWTU, {"ENABLE_SKILLS": "1", "T64_NOWTU_TEXT": nowtu_text})
    nowtu_warnings = [str(msg) for msg in nowtu["warnings"]]
    nowtu_hit = [msg for msg in nowtu_warnings if "when_to_use" in msg]
    listing = str(nowtu["listing"])
    record_c(
        "T64-C2",
        bool(nowtu_hit) and len(nowtu["names"]) == 3 and "nowtu-probe" not in listing,
        f"缺 when_to_use -> scan {len(nowtu['names'])} 条（不回显空字段）；"
        f"list 不含 nowtu-probe={'nowtu-probe' not in listing}；"
        f"warning={nowtu_hit[0][:70] if nowtu_hit else '无'}…；"
        f"其余技能可读={bool(str(nowtu['cost_guard']).strip())}",
    )

    # C3 SKILLS_DIR 不存在
    no_dir = emit_child(CHILD_NO_DIR, {"ENABLE_SKILLS": "1"})
    listing = str(no_dir["listing"])
    load_out = str(no_dir["load"])
    record_c(
        "T64-C3",
        no_dir["scan"] == [] and "暂无可用技能" in listing and "未知技能" in load_out,
        f"SKILLS_DIR 指向不存在目录 -> scan={no_dir['scan']}；list={listing!r}；"
        f"load={load_out[:60]}…（可读提示，无异常）",
    )

    # C4 大小写混用（复用 A7 那次子进程的取证，单独判）
    load_data = emit_child(CHILD_LOAD, {"ENABLE_SKILLS": "1"})
    upper = str(load_data["results"]["RAG-answer-format"])
    rejected = upper.startswith("已拒绝") or upper.startswith("未知技能")
    record_c(
        "T64-C4",
        rejected,
        f"'RAG-answer-format' -> {'明确拒绝' if upper.startswith('已拒绝') else '未知/未规范化'}"
        f"（{upper[:60]}…）；行为二选一已满足且文案含可用名={'rag-answer-format' in upper}",
    )

    # C5 非 UTF-8 字节
    non_utf8 = emit_child(CHILD_NON_UTF8, {"ENABLE_SKILLS": "1"})
    utf8_warnings = [str(msg) for msg in non_utf8["warnings"]]
    decode_hit = [msg for msg in utf8_warnings if "读取失败" in msg or "UnicodeDecodeError" in msg]
    record_c(
        "T64-C5",
        bool(decode_hit) and len(non_utf8["names"]) == 3,
        f"非 UTF-8 文件 -> scan 返回 {len(non_utf8['names'])} 条（跳过坏文件，未抛异常）；"
        f"warning={decode_hit[0][:70] if decode_hit else '无'}…；"
        f"cost-guard 仍可读={bool(str(non_utf8['cost_guard']).strip())}",
    )


# ── B1 ~ B5：不变量 ─────────────────────────────────────────────────


def check_b1() -> None:
    """src/conversation_graph.py 区间与工作树都零改动（v1 不注入索引）。"""
    graph_range = git(
        "diff", "--name-only", f"{BASE_COMMIT}..{CARD_COMMIT}", "--", "src/conversation_graph.py"
    )
    src_range = git("diff", "--name-only", f"{BASE_COMMIT}..{CARD_COMMIT}", "--", "src/")
    graph_status = git("status", "--porcelain", "--", "src/conversation_graph.py")
    src_status = git("status", "--porcelain", "--", "src/")
    ok = not graph_range and not src_range and not graph_status and not src_status
    record(
        "T64-010",
        TITLES["T64-010"],
        ok,
        f"{BASE_COMMIT}..{CARD_COMMIT} 的 src/ 改动={src_range or '无'}；"
        f"conversation_graph.py 区间 diff={'有' if graph_range else '空'}；"
        f"工作树 status={'有' if (graph_status or src_status) else '干净'}",
    )


def check_b2() -> None:
    """b59be16 的 diff 没有往 SYSTEM_PROMPT / stable 区注入技能正文。

    判据（结构 + 文本双查）：
    1. 交付提交没有触碰 `src/`（系统提示的所在层）；
    2. 新增行里没有 `SYSTEM_PROMPT` / `system_prompt` 字样；
    3. 技能加载代码的引用点只有 `agent/adapters/skills/`、`skills_tools.py` 与
       `api_main._register_skills_tools()`（工具形式登记），`src/` 里零引用。
    """
    changed = [
        line
        for line in git("diff", "--name-only", f"{BASE_COMMIT}..{CARD_COMMIT}").splitlines()
        if line
    ]
    src_changed = [path for path in changed if path.startswith("src/")]
    diff_text = git("show", "--format=", CARD_COMMIT)
    added = [
        line
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    prompt_hits = [line for line in added if re.search(r"SYSTEM_PROMPT|system_prompt", line)]
    body_injection = [
        line
        for line in added
        if re.search(r"(skill|SKILL)", line)
        and re.search(r"(system|stable)", line)
        and not line.lstrip("+ ").startswith("#")
        and "不" not in line
    ]
    refs: list[str] = []
    scan_targets = list((REPO / "src").rglob("*.py"))
    scan_targets += list((REPO / "agent").rglob("*.py"))
    scan_targets.append(REPO / "api_main.py")
    for path in scan_targets:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"adapters\.skills|skills_tools|load_skill\(|list_skills\(", line):
                refs.append(f"{path.relative_to(REPO)}:{lineno}")
    allowed_prefixes = (
        "agent/adapters/skills/",
        "agent/adapters/tools/skills_tools.py",
        "api_main.py",
    )
    illegal_refs = [ref for ref in refs if not ref.startswith(allowed_prefixes)]
    ok = not src_changed and not prompt_hits and not body_injection and not illegal_refs
    record(
        "T64-011",
        TITLES["T64-011"],
        ok,
        f"交付提交触碰 src/={src_changed or '无'}；新增行含 SYSTEM_PROMPT/system_prompt="
        f"{prompt_hits or '无'}；可疑“技能->系统提示”新增行={body_injection or '无'}；"
        f"技能模块引用点={[ref.split(':', 1)[0] for ref in refs] or '无'}；"
        f"越界引用={illegal_refs or '无'}",
    )


def baseline_expectations() -> tuple[str, str, str]:
    """从 docs/baseline.md §2 读期望值，读不到用交接单口径兜底。"""
    passed, collect, contracts = "225 passed", "229/233", "3 kept"
    baseline = REPO / "docs" / "baseline.md"
    if baseline.exists():
        text = baseline.read_text(encoding="utf-8")
        match = re.search(r"`pytest -q`\s*\|\s*\*\*(\d+) passed", text)
        if match:
            passed = f"{match.group(1)} passed"
        match = re.search(r"`pytest --collect-only -q`\s*\|\s*\*\*(\d+)/(\d+) collected", text)
        if match:
            collect = f"{match.group(1)}/{match.group(2)}"
    return passed, collect, contracts


def check_b3() -> None:
    """pytest / collect / lint-imports 与 docs/baseline.md §2 一致。"""
    passed, collect, contracts = baseline_expectations()
    code, pytest_out = run_cmd([sys.executable, "-m", "pytest", "-q"], timeout=1200)
    pytest_tail = last_lines(pytest_out, 1)
    collect_code, collect_out = run_cmd(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"], timeout=600
    )
    collect_tail = last_lines(collect_out, 1)
    collect_match = re.search(r"(\d+)/(\d+) tests collected", collect_out)
    lint_bin = Path(sys.executable).parent / "lint-imports"
    lint_cmd = [str(lint_bin)] if lint_bin.exists() else ["lint-imports"]
    lint_code, lint_out = run_cmd(lint_cmd, timeout=600)
    lint_tail = last_lines(lint_out, 1)
    ok = (
        code == 0
        and passed in pytest_tail
        and collect_code == 0
        and collect_match is not None
        and f"{collect_match.group(1)}/{collect_match.group(2)}" == collect
        and lint_code == 0
        and contracts in lint_out
        and "0 broken" in lint_out
    )
    record(
        "T64-012",
        TITLES["T64-012"],
        ok,
        f"pytest 末行「{pytest_tail}」；collect 末行「{collect_tail}」；"
        f"lint-imports 末行「{lint_tail}」；baseline 期望 {passed} / {collect} / {contracts}",
    )


def ledger_commit_verdict(sha: str) -> tuple[bool, str]:
    """判一个动过台账的提交是否合规（与 T94 B5 同一套细化口径）。

    允许集合：`test(cases)` 开头的验收提交，或（`fix` 开头且含 `OI-`）的修复提交；
    且该提交对台账只能新增行（numstat 删除数必须为 0）。其余类型动台账 -> 不合规。
    """
    subject = git("show", "-s", "--format=%s", sha).strip()
    numstat = git("show", "--numstat", "--format=", sha, "--", LEDGER_REL)
    added = deleted = 0
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        if parts[0].isdigit():
            added += int(parts[0])
        if parts[1].isdigit():
            deleted += int(parts[1])
    test_window = subject.startswith("test(cases)")
    fix_with_oi = subject.startswith("fix") and "OI-" in subject
    if not (test_window or fix_with_oi):
        return False, f"subject 不在允许集合：{subject}"
    if deleted != 0:
        return False, f"删除 {deleted} 行：{subject}"
    return True, f"{subject} (+{added}/-{deleted})"


def check_b4() -> None:
    """台账里没有开发窗口写的 T64 行；交付提交没碰台账；动台账的提交只追加。"""
    dev_numstat = git("show", "--numstat", "--format=", CARD_COMMIT, "--", LEDGER_REL)
    if not LEDGER.exists():
        record("T64-013", TITLES["T64-013"], False, f"台账不存在：{LEDGER}")
        return
    with LEDGER.open(encoding="gb18030", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    t64 = [row for row in rows if (row.get("case_id") or "").startswith("T64")]
    foreign = [row.get("case_id") for row in t64 if (row.get("run_by") or "").strip() != "test-G13"]
    shas = [
        line
        for line in git(
            "log", "--no-merges", "--format=%H", f"{BASE_COMMIT}..HEAD", "--", LEDGER_REL
        ).splitlines()
        if line
    ]
    verdicts = [ledger_commit_verdict(sha) for sha in shas]
    bad_commits = [reason for ok, reason in verdicts if not ok]
    ok = not dev_numstat and not foreign and not bad_commits
    record(
        "T64-013",
        TITLES["T64-013"],
        ok,
        f"{CARD_COMMIT} 对台账 numstat={dev_numstat or '空（开发未写台账）'}；"
        f"台账 {len(rows)} 行，T64 行 {len(t64)} 条，非 test-G13 写的={foreign or '无'}；"
        f"动台账提交={[reason for _, reason in verdicts] or '无'}；不合规={bad_commits or '无'}",
    )


def check_b5() -> None:
    """契约 / 前端 / SUBAGENT_* 零改动；config.py 只新增 4 个 SKILLS 字段。"""
    contract_changed = git(
        "diff", "--name-only", f"{BASE_COMMIT}..{CARD_COMMIT}", "--", "docs/contracts", "frontend"
    )
    dirty = git("status", "--porcelain", "--", "docs/contracts", "frontend", "config.py")
    config_diff = git("diff", f"{BASE_COMMIT}..{CARD_COMMIT}", "--", "config.py")
    added = [
        line
        for line in config_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    removed = [
        line
        for line in config_diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    fields = [
        match.group(1)
        for line in added
        for match in [re.match(r"^\+\s{4}([A-Z][A-Z0-9_]*)\s*:", line)]
        if match
    ]
    expected_fields = {
        "ENABLE_SKILLS",
        "SKILLS_DIR",
        "SKILLS_INDEX_MAX",
        "SKILL_MAX_BODY_TOKENS",
    }
    subagent_lines = [line for line in added + removed if "SUBAGENT_" in line]
    ok = (
        not contract_changed
        and not dirty
        and not removed
        and set(fields) == expected_fields
        and len(fields) == 4
        and not subagent_lines
    )
    record(
        "T64-014",
        TITLES["T64-014"],
        ok,
        f"契约/前端区间改动={contract_changed or '无'}；工作树 status={dirty or '干净'}；"
        f"config.py 新增字段={fields}（删除行 {len(removed)}）；"
        f"SUBAGENT_* 改动={subagent_lines or '无'}",
    )


# ── 主流程 ──────────────────────────────────────────────────────────


def main() -> int:
    started = time.monotonic()
    leftovers = cleanup_temp()
    print("=" * 74)
    print("G13 / T6-4 Skills 加载器 —— 测试窗口独立验收（test-G13）")
    print(f"仓库={REPO}")
    print(f"开工 HEAD={BASE_COMMIT}  交付提交={CARD_COMMIT}")
    print(f"清理残留临时技能目录={leftovers or '无'}")
    print("=" * 74)

    try:
        section("A1 ~ A2：开关矩阵")
        guarded("T64-001", check_a1)
        guarded("T64-002", check_a2)

        section("A3 ~ A5 / A7：索引、加载、错误文案、路径穿越")
        guarded("T64-003", check_a3)
        guarded("T64-004", check_a4)
        guarded("T64-005", check_a5)
        guarded("T64-007", check_a7)

        section("A6 / A8 / A9：热加载、截断、坏文件隔离")
        guarded("T64-006", check_a6)
        guarded("T64-008", check_a8)
        guarded("T64-009", check_a9)

        cleanup_temp()
        section("B1 ~ B5：设计约束与不变量")
        guarded("T64-010", check_b1)
        guarded("T64-011", check_b2)
        guarded("T64-012", check_b3)
        guarded("T64-013", check_b4)
        guarded("T64-014", check_b5)

        section("C1 ~ C5：边界探针（open_issue 用，不判失败）")
        check_c()
    finally:
        final_leftover = cleanup_temp()
        print(f"\n收尾清理临时技能目录={final_leftover or '无'}")

    passed = [case_id for case_id, _, ok, _ in RESULTS if ok]
    failed = [(case_id, evidence) for case_id, _, ok, evidence in RESULTS if not ok]
    issues = [probe for probe, ok, _ in C_PROBES if not ok]
    print("\n" + "-" * 74)
    print(
        f"耗时 {time.monotonic() - started:.1f}s；A/B {len(RESULTS)} 项；"
        f"C 类探针 {len(C_PROBES)} 项"
    )
    if failed:
        print(f"失败：{', '.join(case_id for case_id, _ in failed)}")
    print(f"C 类 open_issue：{', '.join(issues) if issues else '无'}")
    print(f"RESULT: {len(passed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
