#!/usr/bin/env python
"""台账体检 —— CI 门禁用（只读，仅标准库）。

校验 ``tests/test_agent_cases/台账用例.csv``（GB18030，15 列）的结构与状态一致性：

1. 能以 GB18030 解码（且不是 UTF-8）；
2. 表头 15 列且列名与约定一致；每个数据行 15 列；
3. ``case_id`` 非空、唯一、匹配 ``^[A-Z][A-Z0-9]*-\\d{3}$``；
4. ``status`` ∈ {pass, fail, block}；
5. ``status != pass`` 的行必须填写 ``open_issue``；
6. ``last_run`` 形如 ``YYYY/MM/DD`` 或为空。

输出每项一行 ``[OK]/[BAD] 项 — 证据``（行级问题额外打印 ``[BAD] 第 N 行 ...``），
末行 ``LEDGER: n rows, m problems``；有问题退出 1，全绿退出 0。

用法（仓库根目录）::

    poetry run python scripts/ci_check_ledger.py
    poetry run python scripts/ci_check_ledger.py --file /tmp/bad.csv

只读承诺：不写任何文件。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO / "tests" / "test_agent_cases" / "台账用例.csv"

#: 台账固定表头（15 列，顺序即列序）。
EXPECTED_HEADER = [
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

#: 卡号形如 ``T01-001`` / ``TSEC-016``：字母开头、允许数字、连字符后 3 位数字。
#: 注：需求书写的是 ``^[A-Z]+-\\d{3}$``，但那样会误杀 182 行 ``T01-xxx`` 这类
#: 合法卡号（前缀本身带数字），故按实际台账放宽前缀为 ``[A-Z][A-Z0-9]*``。
CASE_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d{3}$")
LAST_RUN_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")
VALID_STATUS = ("pass", "fail", "block")

COL_CASE_ID = 0
COL_STATUS = 10
COL_LAST_RUN = 11
COL_OPEN_ISSUE = 13

#: (物理行号, 行内容)；行号从 2 起（第 1 行是表头）。
Row = tuple[int, list[str]]


class Report:
    """按项汇总 ``[OK]/[BAD]`` 与行级证据；problems = ``[BAD]`` 行数。"""

    def __init__(self) -> None:
        self.problems = 0
        self._lines: list[str] = []

    def item(self, name: str, ok: bool, evidence: str) -> None:
        if not ok:
            self.problems += 1
        prefix = "[OK]" if ok else "[BAD]"
        self._lines.append(f"{prefix} {name} — {evidence}")

    def detail(self, line: str) -> None:
        """行级问题：计入 problems。"""
        self.problems += 1
        self._lines.append(f"[BAD] {line}")

    def emit(self) -> None:
        for line in self._lines:
            print(line)


def _cell(row: list[str], index: int) -> str:
    return row[index] if len(row) > index else ""


def _load(path: Path) -> tuple[bytes, str, list[list[str]]]:
    raw = path.read_bytes()
    text = raw.decode("gb18030")
    rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text))]
    return raw, text, rows


def _check_encoding(raw: bytes, text: str, report: Report) -> None:
    utf8_text: str | None = None
    with contextlib.suppress(UnicodeDecodeError):
        utf8_text = raw.decode("utf-8")
    if utf8_text is not None and any(byte >= 0x80 for byte in raw):
        report.item("解码", False, "文件是 UTF-8，台账约定 GB18030")
        return
    report.item(
        "解码",
        True,
        f"GB18030 解码通过（{len(raw)} 字节，{len(text.splitlines())} 行）",
    )


def _check_header(header: list[str], report: Report) -> None:
    if header == EXPECTED_HEADER:
        report.item("表头", True, "15 列，与约定表头逐列一致")
        return
    mismatch = [
        f"第 {i + 1} 列 {got!r}≠{want!r}"
        for i, (got, want) in enumerate(zip(header, EXPECTED_HEADER, strict=False))
        if got != want
    ]
    report.item(
        "表头",
        False,
        f"实际 {len(header)} 列、期望 15 列" + (f"；{'；'.join(mismatch[:5])}" if mismatch else ""),
    )


def _check_width(rows: list[Row], report: Report) -> None:
    bad = [(line_no, len(row)) for line_no, row in rows if len(row) != 15]
    report.item(
        "行宽",
        not bad,
        f"{len(rows) - len(bad)}/{len(rows)} 行 15 列"
        if not bad
        else f"{len(bad)}/{len(rows)} 行不是 15 列",
    )
    for line_no, width in bad:
        report.detail(f"第 {line_no} 行 行宽 — {width} 列，期望 15 列")


def _check_case_ids(rows: list[Row], report: Report) -> None:
    seen: dict[str, int] = {}
    bad: list[tuple[int, str, str]] = []
    for line_no, row in rows:
        case_id = _cell(row, COL_CASE_ID)
        if not case_id:
            bad.append((line_no, case_id, "为空"))
        elif not CASE_ID_RE.match(case_id):
            bad.append((line_no, case_id, "不匹配 ^[A-Z][A-Z0-9]*-\\d{3}$"))
        elif case_id in seen:
            bad.append((line_no, case_id, f"与第 {seen[case_id]} 行重复"))
        else:
            seen[case_id] = line_no
    report.item(
        "case_id",
        not bad,
        f"{len(rows)} 个，唯一、非空、匹配 ^[A-Z][A-Z0-9]*-\\d{{3}}$"
        if not bad
        else f"{len(bad)}/{len(rows)} 个不合法",
    )
    for line_no, case_id, reason in bad:
        report.detail(f"第 {line_no} 行 ({case_id or '空'}) case_id — {reason}")


def _check_status(rows: list[Row], report: Report) -> None:
    statuses = Counter(_cell(row, COL_STATUS) for _, row in rows)
    bad = [
        (line_no, _cell(row, COL_STATUS))
        for line_no, row in rows
        if _cell(row, COL_STATUS) not in VALID_STATUS
    ]
    evidence = "，".join(f"{name}={count}" for name, count in sorted(statuses.items()))
    report.item(
        "status",
        not bad,
        f"{evidence}（取值均合法）"
        if not bad
        else f"{len(bad)}/{len(rows)} 行取值非法；{evidence}",
    )
    for line_no, status in bad:
        report.detail(f"第 {line_no} 行 status — {status!r} 不在 {{pass, fail, block}}")


def _check_open_issue(rows: list[Row], report: Report) -> None:
    non_pass = [
        (line_no, row) for line_no, row in rows if _cell(row, COL_STATUS) in ("fail", "block")
    ]
    bad = [(line_no, row) for line_no, row in non_pass if not _cell(row, COL_OPEN_ISSUE)]
    report.item(
        "open_issue",
        not bad,
        f"非 pass 行 {len(non_pass) - len(bad)}/{len(non_pass)} 已填写"
        if not bad
        else f"{len(bad)}/{len(non_pass)} 行非 pass 却留空",
    )
    for line_no, row in bad:
        report.detail(
            f"第 {line_no} 行 ({_cell(row, COL_CASE_ID)}) open_issue — "
            f"status={_cell(row, COL_STATUS)} 但 open_issue 为空"
        )


def _check_last_run(rows: list[Row], report: Report) -> None:
    values = [(line_no, _cell(row, COL_LAST_RUN)) for line_no, row in rows]
    bad = [(line_no, value) for line_no, value in values if value and not LAST_RUN_RE.match(value)]
    filled = sum(1 for _, value in values if value)
    report.item(
        "last_run",
        not bad,
        f"{filled}/{len(rows)} 行非空，均为 YYYY/MM/DD（其余为空）"
        if not bad
        else f"{len(bad)}/{len(rows)} 行格式非法",
    )
    for line_no, value in bad:
        report.detail(f"第 {line_no} 行 last_run — {value!r} 不是 YYYY/MM/DD")


def check(path: Path) -> int:
    """执行全部校验并打印报告，返回退出码（0/1）。"""
    report = Report()
    try:
        raw, text, rows = _load(path)
    except OSError as exc:
        print(f"[BAD] 读取 — 无法读取 {path}：{exc}")
        print("LEDGER: 0 rows, 1 problems")
        return 1
    except UnicodeDecodeError as exc:
        print(f"[BAD] 解码 — GB18030 解码失败：{exc}")
        print("LEDGER: 0 rows, 1 problems")
        return 1

    header = rows[0] if rows else []
    data_rows: list[Row] = [
        (line_no, row) for line_no, row in enumerate(rows[1:], start=2) if row and any(row)
    ]

    _check_encoding(raw, text, report)
    _check_header(header, report)
    _check_width(data_rows, report)
    _check_case_ids(data_rows, report)
    _check_status(data_rows, report)
    _check_open_issue(data_rows, report)
    _check_last_run(data_rows, report)

    report.emit()
    print(f"LEDGER: {len(data_rows)} rows, {report.problems} problems")
    return 0 if report.problems == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="台账体检（GB18030 / 15 列 / 状态一致性）")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_LEDGER,
        help=f"台账路径，默认 {DEFAULT_LEDGER.relative_to(REPO)}",
    )
    args = parser.parse_args(argv)
    return check(args.file)


if __name__ == "__main__":
    sys.exit(main())
