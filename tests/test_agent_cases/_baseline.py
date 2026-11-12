#!/usr/bin/env python3
"""卡级基线读取器：唯一事实来源是 `docs/baseline.md` §2 的测试基线表。

**为什么存在**：T11 / T93 两张老卡把 `pytest` 通过数、`--collect-only` 收集数和
`nodeids` 哈希硬编码在脚本里。基线一旦有意更新（例如 T-arch 新增 2 例契约测试），
两条护栏就永远红，且"改哪几个数字"散落在多个脚本里。本模块把三个值收敛到
`docs/baseline.md` §2，脚本只断言"**与 baseline.md 一致**"。

**不叫 `test_*.py`**：本文件不是 pytest 用例，放在 `tests/test_agent_cases/` 下但
不会被收集，因此不会污染 `docs/baseline.md` §2 冻结的 nodeids 哈希。

用法::

    from _baseline import expected, assert_nodeids_sha

    exp = expected()                       # {'pytest': '225 passed, ...', ...}
    assert_nodeids_sha()                   # 与 baseline.md §2 比对，不一致抛 BaselineError

解析失败一律抛 `BaselineError`，错误信息写明"baseline.md 结构变了，改解析器"，
**绝不静默返回空值**（否则护栏会假装通过）。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

__all__ = [
    "BASELINE_PATH",
    "DEFAULT_NODEIDS",
    "BaselineError",
    "assert_nodeids_sha",
    "expected",
    "parse",
    "parse_file",
    "pytest_matches",
    "pytest_numbers",
]

#: `tests/test_agent_cases/_baseline.py` -> 仓库根 -> `docs/baseline.md`
BASELINE_PATH = Path(__file__).resolve().parents[2] / "docs" / "baseline.md"
#: pytest 收集缓存；sha256 的比对对象
DEFAULT_NODEIDS = Path(__file__).resolve().parents[2] / ".pytest_cache" / "v" / "cache" / "nodeids"

#: §2 的标题（改了标题就必须改这里，解析器会明确报错）
SECTION_TITLE = "## 2. 测试基线"

_HEX64 = re.compile(r"\b([0-9a-f]{64})\b")
_RATIO = re.compile(r"(\d+)\s*/\s*(\d+)")
_PASSED = re.compile(r"(\d+)\s+passed")
_LINES = re.compile(r"(\d+)\s*行")
#: pytest 末行的四个计数项；缺项按 0 处理（短形式如 "225 passed, 4 deselected"）。
_COUNTS = {
    "passed": re.compile(r"(\d+)\s+passed"),
    "failed": re.compile(r"(\d+)\s+failed"),
    "skipped": re.compile(r"(\d+)\s+skipped"),
    "deselected": re.compile(r"(\d+)\s+deselected"),
}


class BaselineError(RuntimeError):
    """`docs/baseline.md` §2 缺失/结构变化/取值不合法时抛出。"""


def _fail(reason: str) -> BaselineError:
    return BaselineError(
        "baseline.md 结构变了，改解析器（tests/test_agent_cases/_baseline.py）："
        f"{reason}；文件 {BASELINE_PATH}"
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BaselineError(f"读不到基线文件 {path}：{exc}") from exc


def _section_lines(text: str) -> list[str]:
    """切出 `## 2. 测试基线` 到下一个二级标题之间的行。"""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == SECTION_TITLE), None)
    if start is None:
        raise _fail(f"找不到二级标题「{SECTION_TITLE}」")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    section = lines[start:end]
    if not section:
        raise _fail(f"「{SECTION_TITLE}」小节为空")
    return section


def _cell(section: list[str], label: str) -> str:
    """取 §2 表格中首列包含 `label` 的那一行第二列。"""
    for line in section:
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        if label in cells[0]:
            return cells[1]
    raise _fail(f"§2 表格里找不到「{label}」行")


def _bold(raw: str, label: str) -> str:
    match = re.search(r"\*\*(.+?)\*\*", raw)
    if match is None:
        raise _fail(f"「{label}」行的期望值缺少 **加粗** 标记：{raw!r}")
    return match.group(1).strip()


def parse(text: str) -> dict[str, Any]:
    """解析 baseline.md 文本，返回 §2 的三个基线值（含拆好的数字）。

    Raises:
        BaselineError: 标题/表格行/加粗/哈希/行数任一缺失或格式不对。
    """
    section = _section_lines(text)

    pytest_cell = _cell(section, "pytest -q")
    pytest_text = _bold(pytest_cell, "pytest -q")
    passed_match = _PASSED.search(pytest_text)
    if passed_match is None:
        raise _fail(f"「pytest -q」加粗值里找不到「N passed」：{pytest_text!r}")

    collect_cell = _cell(section, "pytest --collect-only -q")
    collect_text = _bold(collect_cell, "pytest --collect-only -q")
    ratio_match = _RATIO.search(collect_text)
    if ratio_match is None:
        raise _fail(f"「pytest --collect-only -q」加粗值里找不到「N/M」：{collect_text!r}")

    nodeids_cell = _cell(section, ".pytest_cache/v/cache/nodeids")
    sha_match = _HEX64.search(nodeids_cell)
    if sha_match is None:
        raise _fail(f"nodeids 行里找不到 64 位 sha256：{nodeids_cell!r}")
    lines_match = _LINES.search(nodeids_cell)
    if lines_match is None:
        raise _fail(f"nodeids 行里找不到「N 行」：{nodeids_cell!r}")

    return {
        # 原文串，供脚本做 `in summary` 断言
        "pytest": pytest_text,
        "collect": f"{ratio_match.group(1)}/{ratio_match.group(2)}",
        # 拆好的数字，供精确断言
        "pytest_passed": int(passed_match.group(1)),
        "collect_collected": int(ratio_match.group(1)),
        "collect_total": int(ratio_match.group(2)),
        "nodeids_sha256": sha_match.group(1),
        "nodeids_lines": int(lines_match.group(1)),
        "source": str(BASELINE_PATH),
    }


def parse_file(path: str | Path | None = None) -> dict[str, Any]:
    """读取并解析 baseline.md；`path` 仅用于测试解析器自身。"""
    target = Path(path) if path is not None else BASELINE_PATH
    result = parse(_read(target))
    result["source"] = str(target)
    return result


def expected(path: str | Path | None = None) -> dict[str, Any]:
    """返回 `docs/baseline.md` §2 的基线值（每次重读，避免进程内缓存过期）。"""
    return parse_file(path)


def pytest_numbers(summary: str) -> dict[str, int]:
    """从 pytest 末行解析 passed / failed / skipped / deselected（缺省为 0）。"""
    numbers = {"passed": 0, "failed": 0, "skipped": 0, "deselected": 0}
    for name, pattern in _COUNTS.items():
        match = pattern.search(summary)
        if match is not None:
            numbers[name] = int(match.group(1))
    return numbers


def pytest_matches(summary: str) -> tuple[bool, str]:
    """把末行与 docs/baseline.md §2 的 pytest 值比较。

    判据：failed == 0 且 passed == 基线 passed 且 deselected == 基线 deselected。
    skipped **不参与比较**（它是环境函数：CI 缺服务会多跳），只放进返回的说明里。
    返回 (是否一致, 说明串)。
    """
    base = pytest_numbers(str(expected()["pytest"]))
    got = pytest_numbers(summary)
    ok = (
        got["failed"] == 0
        and got["passed"] == base["passed"]
        and got["deselected"] == base["deselected"]
    )
    detail = (
        f"passed={got['passed']}/{base['passed']}"
        f" skipped={got['skipped']}（环境，不比）"
        f" failed={got['failed']}"
        f" deselected={got['deselected']}/{base['deselected']}"
    )
    return ok, detail


def assert_nodeids_sha(path: str | Path | None = None) -> str:
    """断言 nodeids 文件的 sha256 与 baseline.md §2 一致，返回实测哈希。

    Args:
        path: nodeids 路径；默认 `.pytest_cache/v/cache/nodeids`。

    Raises:
        BaselineError: 文件缺失、或哈希/行数与 baseline.md 不一致。
    """
    exp = expected()
    target = Path(path) if path is not None else DEFAULT_NODEIDS
    if not target.exists():
        raise BaselineError(f"nodeids 缺失：{target}；先跑 `pytest --collect-only -q` 生成收集缓存")
    raw = target.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    actual_lines = raw.count(b"\n")  # 与 `wc -l` 同口径
    if digest != exp["nodeids_sha256"]:
        raise BaselineError(
            "nodeids 与 baseline.md §2 不一致："
            f"实测 {digest}（{actual_lines} 行），"
            f"基线 {exp['nodeids_sha256']}（{exp['nodeids_lines']} 行）。"
            "先确认收集数是否仍为 "
            f"{exp['collect']}；若是有意新增/删除用例，按 §2 的说明更新基线。"
        )
    if actual_lines != exp["nodeids_lines"]:
        raise BaselineError(
            "nodeids 哈希一致但行数不一致（异常）："
            f"实测 {actual_lines} 行，基线 {exp['nodeids_lines']} 行"
        )
    return digest
