#!/usr/bin/env python
"""T6-4 / G13 冒烟：Skills 渐进披露（list / load / 热加载 / 路径安全 / 截断）。

用**本地技能目录**演示五件事，不连 Ollama / RAG / Redis：

1. 开关关闭：注册表仍是 5 个内置工具、schema 哈希 == 基线 `c48c9d48…`；
2. 开关打开：注册表 +2，末尾是 `list_skills` / `load_skill`；
3. `list_skills()` 返回 3 条索引（name + when_to_use），`SKILLS_INDEX_MAX` 生效；
4. `load_skill("rag-answer-format")` 返回正文，且不含 frontmatter；
5. 未知名字 / 路径穿越被拒；新建 SKILL.md 不重启即可见；坏文件跳过并 warning；
   超大正文截断并带标记。

跑法（仓库根目录，任意 cwd 均可）：

    poetry run python scripts/smoke_skills.py

退出码：0 = 全部断言通过；1 = 有失败（输出里逐条 [FAIL]）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import 项目模块
    sys.path.insert(0, str(REPO))

from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from agent.adapters.skills import loader  # noqa: E402
from agent.adapters.tools import registry  # noqa: E402
from agent.adapters.tools.skills_tools import (  # noqa: E402
    LIST_SKILLS_TOOL_NAME,
    LOAD_SKILL_TOOL_NAME,
    list_skills,
    load_skill,
    register_skills_tools,
)
from config import settings  # noqa: E402
from src.context_manager import estimate_tokens  # noqa: E402

BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
BASELINE_NAMES = (
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
)

_PASS = 0
_FAIL = 0
_LOGS: list[str] = []


class _Capture(logging.Handler):
    """收集 adaptive_agent 的 warning，用于断言坏文件确实被跳过。"""

    def emit(self, record: logging.LogRecord) -> None:
        _LOGS.append(record.getMessage())


def check(label: str, ok: bool, detail: str = "") -> None:
    """打印一条断言结果并累计计数。"""
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"[PASS] {label}" + (f" — {detail}" if detail else ""))
    else:
        _FAIL += 1
        print(f"[FAIL] {label}" + (f" — {detail}" if detail else ""))


def _fingerprint(tools: list[Any]) -> tuple[str, tuple[str, ...], int]:
    """按开工基线口径算 schema 指纹（与 scripts/smoke_tools_schema.py 一致）。"""
    schemas = [convert_to_openai_tool(tool) for tool in tools]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    return (
        hashlib.sha256(canonical).hexdigest(),
        tuple(schema["function"]["name"] for schema in schemas),
        len(canonical),
    )


def _write_skill(directory: Path, name: str, when: str, body: str) -> None:
    """在临时技能目录写一个最小 SKILL.md。"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: 冒烟用临时技能\nwhen_to_use: {when}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def main() -> int:
    logging.getLogger("adaptive_agent").addHandler(_Capture())

    # ── 1. 开关关闭：5 个 / 基线哈希 ──────────────────────────────────
    settings.ENABLE_SKILLS = False
    before_sha, before_names, _ = _fingerprint(registry.to_lc_tools())
    check(
        "开关关：注册表 5 个且 schema 哈希 == 基线",
        before_names == BASELINE_NAMES and before_sha == BASELINE_SHA256,
        f"names={before_names}；sha={before_sha[:12]}…",
    )
    check("开关关：register_skills_tools() 返回 False", register_skills_tools() is False)
    check("开关关：注册表仍 5 个", len(registry.TOOL_SPECS) == 5, f"{len(registry.TOOL_SPECS)} 个")

    # ── 2. 开关打开：+2 ─────────────────────────────────────────────
    settings.ENABLE_SKILLS = True
    registered = register_skills_tools()
    after_tools = registry.to_lc_tools()
    after_names = tuple(convert_to_openai_tool(t)["function"]["name"] for t in after_tools)
    check("开关开：register_skills_tools() 返回 True", registered is True)
    check("开关开：注册表 5 -> 7", len(registry.TOOL_SPECS) == 7, f"{len(registry.TOOL_SPECS)} 个")
    check(
        "开关开：末尾是 list_skills / load_skill",
        after_names[-2:] == (LIST_SKILLS_TOOL_NAME, LOAD_SKILL_TOOL_NAME),
        f"last={after_names[-2:]}",
    )
    for tool_name in (LIST_SKILLS_TOOL_NAME, LOAD_SKILL_TOOL_NAME):
        spec = registry.spec(tool_name)
        check(
            f"{tool_name} spec 口径",
            spec.permission == "read"
            and spec.cost_model.unit == "call"
            and spec.cost_model.unit_cost == 0.0
            and spec.timeout_seconds == 5.0
            and spec.max_output_tokens == settings.SKILL_MAX_BODY_TOKENS
            and spec.requires_approval is False
            and spec.feature_flag == "ENABLE_SKILLS",
            f"permission={spec.permission} timeout={spec.timeout_seconds} "
            f"max={spec.max_output_tokens} flag={spec.feature_flag}",
        )

    # ── 3. list：3 条，每条 name 与 when_to_use ─────────────────────
    specs = loader.scan()
    listing = list_skills()
    lines = [line for line in listing.splitlines() if line.strip()]
    check("list_skills() 返回 3 条", len(lines) == 3, f"{len(lines)} 条")
    check(
        "list_skills() 每条含 name 与 when_to_use",
        all(f"{spec.name} — {spec.when_to_use}" in listing for spec in specs),
        listing.replace("\n", " | "),
    )

    old_max = settings.SKILLS_INDEX_MAX
    settings.SKILLS_INDEX_MAX = 2
    limited = list_skills()
    settings.SKILLS_INDEX_MAX = old_max
    check(
        "SKILLS_INDEX_MAX=2 时最多显示 2 条",
        len(limited.splitlines()) == 2 and all(" — " in line for line in limited.splitlines()),
        limited.replace("\n", " | "),
    )

    # ── 4. load：正文不含 frontmatter ───────────────────────────────
    body = load_skill("rag-answer-format")
    check(
        "load_skill('rag-answer-format') 返回正文且不含 frontmatter",
        "---\nname:" not in body and body.startswith("# 内部知识库答案格式"),
        body[:40].replace("\n", " "),
    )
    check("正文内容正确（含“结论先行”）", "结论先行" in body)
    check(
        "正文未超 SKILL_MAX_BODY_TOKENS",
        estimate_tokens(body) <= settings.SKILL_MAX_BODY_TOKENS,
        f"{estimate_tokens(body)} <= {settings.SKILL_MAX_BODY_TOKENS}",
    )

    # ── 5. 未知名字 / 路径穿越 ─────────────────────────────────────
    unknown = load_skill("no-such-skill")
    check(
        "未知名字：可读错误 + 列出可用名字",
        "未知技能" in unknown and "rag-answer-format" in unknown and "citation-policy" in unknown,
        unknown,
    )
    for bad in ("../etc/passwd", "../../etc/passwd", "/etc/passwd"):
        result = load_skill(bad)
        check(
            f"路径穿越被拒：{bad}",
            "已拒绝" in result and "root:" not in result,
            result[:90],
        )

    # ── 6. 热加载：新建 / 修改后不重启即可见 ───────────────────────
    hot_dir = loader.skills_root() / "tmp-demo"
    try:
        _write_skill(hot_dir, "tmp-demo", "冒烟热加载第一版", "临时技能正文。")
        first = list_skills()
        check("新建 SKILL.md 后不重启 list 立刻多一条", "tmp-demo — 冒烟热加载第一版" in first)

        _write_skill(hot_dir, "tmp-demo", "冒烟热加载第二版", "临时技能正文，已修改。")
        second = list_skills()
        check(
            "修改 SKILL.md 后不重启 list 读到新 when_to_use",
            "tmp-demo — 冒烟热加载第二版" in second and "第一版" not in second,
        )
    finally:
        shutil.rmtree(hot_dir, ignore_errors=True)
    check("删除临时技能后 list 不再出现", "tmp-demo" not in list_skills())

    # ── 7. 损坏文件：跳过 + warning，其余技能正常 ───────────────────
    broken_dir = loader.skills_root() / "broken-demo"
    try:
        broken_dir.mkdir(parents=True, exist_ok=True)
        (broken_dir / "SKILL.md").write_text("没有 frontmatter 的坏文件\n", encoding="utf-8")
        _LOGS.clear()
        clean = loader.scan()
        broken_warned = any(
            "broken-demo" in message and "frontmatter" in message for message in _LOGS
        )
        check(
            "坏文件被跳过：scan 仍返回 3 条",
            len(clean) == 3 and "broken-demo" not in [item.name for item in clean],
            f"{len(clean)} 条；warning={broken_warned}",
        )
        check("坏文件产生一条 warning", broken_warned)
        check("坏文件不影响 list_skills", "broken-demo" not in list_skills())
    finally:
        shutil.rmtree(broken_dir, ignore_errors=True)

    # ── 8. 超大正文：截断 + 标记 ───────────────────────────────────
    big_dir = loader.skills_root() / "oversize-demo"
    try:
        _write_skill(big_dir, "oversize-demo", "冒烟超大正文", "这是超长技能正文。" * 1000)
        big = load_skill("oversize-demo")
        big_tokens = estimate_tokens(big)
        check(
            "超大正文截断到 SKILL_MAX_BODY_TOKENS 内且带标记",
            big_tokens <= settings.SKILL_MAX_BODY_TOKENS and "已截断" in big,
            f"{big_tokens} <= {settings.SKILL_MAX_BODY_TOKENS}；标记={'已截断' in big}",
        )
    finally:
        shutil.rmtree(big_dir, ignore_errors=True)

    print("-" * 74)
    if _FAIL:
        print(f"RESULT: {_PASS} passed, {_FAIL} failed")
        return 1
    print(f"RESULT: {_PASS} passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
