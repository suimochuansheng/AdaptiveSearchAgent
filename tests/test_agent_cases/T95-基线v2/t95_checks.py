#!/usr/bin/env python3
"""T95 / G11 基线 v2 —— 测试窗口（test-G11）独立验收脚本（只读）。

用法（仓库根目录，任意 cwd 均可）：

    export PRE_COMMIT_HOME="$PWD/.pre-commit-cache"
    poetry run python "tests/test_agent_cases/T95-基线v2/t95_checks.py"

覆盖（T95-001 ~ T95-012，全部 PASS 才返回 0）：

    T95-001  三份 JSON 可读：各 20 题、q001..q020 顺序、git_commit=24a916c、0 errors
    T95-002  A 组均分从 results 独立复算，与 docs/baseline.md §8 表 9.70 一致
    T95-003  B 组均分独立复算，与 §8 表 9.75 一致
    T95-004  C 组均分独立复算，与 §8 表 9.85 一致
    T95-005  三组分数分布（10/9/8/7/<=6）与 §8 表逐格一致
    T95-006  三份 JSON 的 judge_model 均为 qwen3.8-max-0902
    T95-007  有效/失败、平均耗时、平均工具调用与 §8 表一致
    T95-008  §7 raw 9.65 的差异算式成立：A +0.05 / B +0.10 / C +0.20，且 <= 阈值 0.3
    T95-009  台账 T01-018 = pass，notes 指向 §8
    T95-010  台账非 T95 行与 101b69b 逐字段一致（189 行不变），T95 行 0 或 12 行
    T95-011  git diff --stat 101b69b~1 101b69b -- evals/ 为空，题集 sha256 未变
    T95-012  C 组 20 题未调用 deep_research -> 审批次数 0；只记局限，不判失败

退出码：
    0 = 12 条检查全部 PASS；1 = 存在 FAIL。
    逐条打印 ``[PASS]/[FAIL] 名称 — 证据``，末行固定 ``RESULT: n passed, m failed``。

只读承诺：
    只读三份 JSON、docs/baseline.md、台账 CSV 与 git 历史；不写任何文件，
    不起后端、不出网、不调 LLM。台账 T95 行由调用方另行追加。

编码：
    本脚本、baseline.md、报告 JSON 均为 UTF-8；台账 CSV 为 GB18030（显式 encoding）。

为什么不是 test_*.py：
    一旦被 pytest 收集，docs/baseline.md §2 的收集数与 nodeids 哈希立刻失效
    （见 tests/test_agent_cases/README.md §2）。本卡用独立进程与退出码表达结论。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

REPO = Path(__file__).resolve().parents[3]
BASELINE = REPO / "docs" / "baseline.md"
LEDGER = REPO / "tests" / "test_agent_cases" / "台账用例.csv"
DATASET = REPO / "evals" / "datasets" / "qa_dataset_v3.yaml"
DEFAULT_REPORTS = REPO.parent / "文档" / "项目" / "AdaptiveSearchAgent" / "评测报告"

REPORTS = Path(os.environ.get("T95_REPORTS_DIR", str(DEFAULT_REPORTS)))
BASE_COMMIT = os.environ.get("T95_BASE_COMMIT", "101b69b")
EVAL_COMMIT = "24a916c"
JUDGE = "qwen3.8-max-0902"
BUCKETS = ("10", "9", "8", "7", "<=6")
GROUP_KEYS = ("A", "B", "C")
REPORT_FILES = {
    "A": "agent_eval_G11_A_off_24a916c_20261111-2300.json",
    "B": "agent_eval_G11_B_governance_24a916c_20261111-2310.json",
    "C": "agent_eval_G11_C_hitl_24a916c_20261111-2323.json",
}


class CheckFailed(AssertionError):
    """任一检查不通过时抛出，脚本以非 0 退出。"""


@dataclass(frozen=True)
class GroupStats:
    """从一份 JSON 报告独立复算出来的统计量。"""

    key: str
    path: Path
    label: str
    judge_model: str
    commit: str
    dataset_sha256: str
    dataset_size: int
    total: int
    ids: tuple[str, ...]
    scored: int
    failed: int
    mean: Decimal
    dist: dict[str, int]
    avg_elapsed: Decimal
    avg_tools: Decimal
    tools: tuple[str, ...]
    error_items: tuple[str, ...]


_STATS: dict[str, GroupStats] = {}
_BASELINE_TEXT: str | None = None


def _fmt1(value: Decimal) -> str:
    return f"{value:.1f}"


def _fmt2(value: Decimal) -> str:
    return f"{value:.2f}"


def _bucket(score: Decimal) -> str:
    if score >= 10:
        return "10"
    if score >= 9:
        return "9"
    if score >= 8:
        return "8"
    if score >= 7:
        return "7"
    return "<=6"


def _load_group(key: str) -> GroupStats:
    path = REPORTS / REPORT_FILES[key]
    if not path.is_file():
        raise CheckFailed(f"报告不存在：{path}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = list(data.get("results") or [])
    if not results:
        raise CheckFailed(f"{path.name} 的 results 为空")
    scores = [Decimal(str(row["score"])) for row in results if row.get("score") is not None]
    if not scores:
        raise CheckFailed(f"{path.name} 没有可计分题目")
    dist = dict.fromkeys(BUCKETS, 0)
    for score in scores:
        dist[_bucket(score)] += 1
    elapsed = [Decimal(str(row.get("elapsed_seconds") or 0)) for row in results]
    tools = tuple(name for row in results for name in (row.get("tools") or []))
    return GroupStats(
        key=key,
        path=path,
        label=str(data.get("label")),
        judge_model=str(data.get("judge_model")),
        commit=str(data.get("git_commit")),
        dataset_sha256=str(data.get("dataset_sha256")),
        dataset_size=int(data.get("dataset_size") or 0),
        total=len(results),
        ids=tuple(str(row.get("id")) for row in results),
        scored=len(scores),
        failed=len(results) - len(scores),
        mean=sum(scores, Decimal(0)) / len(scores),
        dist=dist,
        avg_elapsed=sum(elapsed, Decimal(0)) / len(results),
        avg_tools=Decimal(sum(len(row.get("tools") or []) for row in results)) / len(results),
        tools=tools,
        error_items=tuple(str(row.get("id")) for row in results if row.get("errors")),
    )


def stats() -> dict[str, GroupStats]:
    if not _STATS:
        for key in GROUP_KEYS:
            _STATS[key] = _load_group(key)
    return _STATS


def baseline_text() -> str:
    global _BASELINE_TEXT
    if _BASELINE_TEXT is None:
        if not BASELINE.is_file():
            raise CheckFailed(f"缺少 {BASELINE}")
        _BASELINE_TEXT = BASELINE.read_text(encoding="utf-8")
    return _BASELINE_TEXT


def _section(number: int) -> str:
    text = baseline_text()
    marker = f"## {number}."
    start = text.find(marker)
    if start < 0:
        raise CheckFailed(f"baseline.md 缺 {marker} 章节")
    rest = text[start + len(marker) :]
    nxt = re.search(r"^## \d+\.", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _section8() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for line in _section(8).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 8 or cells[0][:1] not in GROUP_KEYS:
            continue
        key = cells[0][:1]
        if key in table:
            continue
        dist = [int(part) for part in cells[3].split("/")]
        if len(dist) != len(BUCKETS):
            raise CheckFailed(f"§8 {key} 组分布列无法解析：{cells[3]}")
        elapsed = re.search(r"([\d.]+)s", cells[4])
        approvals = re.match(r"\d+", cells[6])
        if not elapsed or not approvals:
            raise CheckFailed(f"§8 {key} 组耗时/审批列无法解析：{cells[4]} / {cells[6]}")
        table[key] = {
            "mean": Decimal(cells[1].replace("*", "")),
            "valid_failed": cells[2],
            "dist": dict(zip(BUCKETS, dist, strict=True)),
            "avg_elapsed": Decimal(elapsed.group(1)),
            "avg_tools": Decimal(cells[5]),
            "approvals": int(approvals.group(0)),
        }
    missing = [key for key in GROUP_KEYS if key not in table]
    if missing:
        raise CheckFailed(f"§8 表缺组：{missing}")
    return table


def _raw_mean() -> Decimal:
    matched = re.search(r"\|\s*均分\s*\|\s*\*\*([\d.]+)\*\*", _section(7))
    if not matched:
        raise CheckFailed("§7 未找到 raw 均分")
    return Decimal(matched.group(1))


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=False)


def _parse_csv(raw: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(raw.decode("gb18030"), newline="")))


def _mean_check(key: str) -> str:
    group = stats()[key]
    table = _section8()[key]
    computed = _fmt2(group.mean)
    declared = _fmt2(table["mean"])
    if computed != declared or group.mean != table["mean"]:
        raise CheckFailed(
            f"{key} 组均分不一致：复算 {computed}（sum="
            f"{group.mean * group.scored:.0f}/{group.scored}）；§8 表 {declared}"
        )
    return (
        f"{key} 组复算 sum={group.mean * group.scored:.0f}/{group.scored}={computed}；"
        f"§8 表 {declared}（小数位一致）"
    )


def check_t95_001() -> str:
    expected_ids = tuple(f"q{number:03d}" for number in range(1, 21))
    for key in GROUP_KEYS:
        group = stats()[key]
        if group.total != 20 or group.dataset_size != 20:
            raise CheckFailed(
                f"{key} 组题数不符：dataset_size={group.dataset_size} results={group.total}"
            )
        if group.ids != expected_ids:
            raise CheckFailed(f"{key} 组题目序列不符：{group.ids}")
        if group.commit != EVAL_COMMIT:
            raise CheckFailed(f"{key} 组 git_commit={group.commit}，期望 {EVAL_COMMIT}")
        if group.error_items:
            raise CheckFailed(f"{key} 组存在 errors：{group.error_items}")
    return "A/B/C 各 20 题（dataset_size=20、results=20、q001..q020 顺序一致）；git_commit=24a916c；0 errors"


def check_t95_002() -> str:
    return _mean_check("A")


def check_t95_003() -> str:
    return _mean_check("B")


def check_t95_004() -> str:
    return _mean_check("C")


def check_t95_005() -> str:
    parts = []
    for key in GROUP_KEYS:
        group = stats()[key]
        table = _section8()[key]
        if group.dist != table["dist"]:
            raise CheckFailed(f"{key} 组分布不一致：复算 {group.dist}；§8 表 {table['dist']}")
        parts.append(f"{key} " + "/".join(str(group.dist[bucket]) for bucket in BUCKETS))
    return "；".join(parts) + "（与 §8 表逐格一致）"


def check_t95_006() -> str:
    bad = {key: stats()[key].judge_model for key in GROUP_KEYS if stats()[key].judge_model != JUDGE}
    if bad:
        raise CheckFailed(f"judge_model 不符：{bad}，期望 {JUDGE}")
    return f"三份 JSON 的 judge_model 均为 {JUDGE}"


def check_t95_007() -> str:
    parts = []
    for key in GROUP_KEYS:
        group = stats()[key]
        table = _section8()[key]
        valid_failed = f"{group.scored}/{group.failed}"
        if valid_failed != table["valid_failed"]:
            raise CheckFailed(f"{key} 组有效/失败 {valid_failed} != §8 {table['valid_failed']}")
        if _fmt1(group.avg_elapsed) != _fmt1(table["avg_elapsed"]) or abs(
            group.avg_elapsed - table["avg_elapsed"]
        ) > Decimal("0.05"):
            raise CheckFailed(
                f"{key} 组平均耗时 {_fmt1(group.avg_elapsed)}s != §8 {_fmt1(table['avg_elapsed'])}s"
            )
        if (
            _fmt2(group.avg_tools) != _fmt2(table["avg_tools"])
            or group.avg_tools != table["avg_tools"]
        ):
            raise CheckFailed(
                f"{key} 组平均工具调用 {_fmt2(group.avg_tools)} != §8 {_fmt2(table['avg_tools'])}"
            )
        parts.append(f"{key} {valid_failed} {_fmt1(group.avg_elapsed)}s {_fmt2(group.avg_tools)}")
    return "；".join(parts) + "（与 §8 表一致）"


def check_t95_008() -> str:
    text8 = _section(8)
    raw = _raw_mean()
    matched_a = re.search(r"A\s+([\d.]+)\s+与\s+raw\s+([\d.]+)\s+差\s+\*\*([+-][\d.]+)\*\*", text8)
    matched_bc = re.search(r"B\s+([\d.]+)（([+-][\d.]+)）、C\s+([\d.]+)（([+-][\d.]+)）", text8)
    matched_threshold = re.search(r"阈值\s*([\d.]+)", text8)
    if not matched_a or not matched_bc or not matched_threshold:
        raise CheckFailed("§8 差异算式文本未能解析")
    if Decimal(matched_a.group(2)) != raw:
        raise CheckFailed(f"§8 算式里的 raw={matched_a.group(2)}，§7 表 {raw}")
    declared = {"A": matched_a.group(3), "B": matched_bc.group(2), "C": matched_bc.group(4)}
    threshold = Decimal(matched_threshold.group(1))
    actual: dict[str, str] = {}
    for key in GROUP_KEYS:
        diff = stats()[key].mean - raw
        actual[key] = f"{diff:+.2f}"
        if actual[key] != declared[key]:
            raise CheckFailed(f"{key} 组差异不一致：复算 {actual[key]}；§8 写 {declared[key]}")
        if abs(diff) > threshold:
            raise CheckFailed(f"{key} 组差异 {actual[key]} 超出阈值 {threshold}")
    return (
        f"§7 raw {raw}；复算差 A {actual['A']} / B {actual['B']} / C {actual['C']}，"
        f"与 §8 文字/算式一致，均 <= 阈值 {threshold}"
    )


def check_t95_009() -> str:
    if not LEDGER.is_file():
        raise CheckFailed(f"缺少台账 {LEDGER}")
    rows = _parse_csv(LEDGER.read_bytes())
    header = rows[0]
    if "case_id" not in header:
        raise CheckFailed(f"台账表头异常：{header}")
    for row in rows[1:]:
        if not row or row[0] != "T01-018":
            continue
        record = dict(zip(header, row, strict=True))
        if record.get("status") != "pass":
            raise CheckFailed(f"T01-018 status={record.get('status')}，期望 pass")
        notes = record.get("notes", "")
        if "§8" not in notes or "baseline.md" not in notes:
            raise CheckFailed(f"T01-018 notes 未指向 §8：{notes}")
        return f"T01-018 status=pass，last_run={record.get('last_run')}，notes={notes}"
    raise CheckFailed("台账缺 T01-018 行")


def _check_ledger_tail(current_data: list[list[str]], base_count: int) -> list[list[str]]:
    """T95 行必须整块追加在末尾，且是 0 行（未追加）或完整的 T95-001..T95-012。"""
    tail = current_data[base_count:]
    if any(not row[0].startswith("T95-") for row in tail):
        raise CheckFailed("T95 行未追加在台账末尾")
    expected_ids = [f"T95-{number:03d}" for number in range(1, 13)]
    tail_ids = [row[0] for row in tail]
    if tail_ids not in ([], expected_ids):
        raise CheckFailed(f"T95 行不完整或乱序：{tail_ids}")
    return tail


def _check_ledger_shape(rows: list[list[str]]) -> None:
    if any(len(row) != 15 for row in rows):
        raise CheckFailed("台账存在非 15 列的行")
    all_ids = [row[0] for row in rows]
    if len(set(all_ids)) != len(all_ids):
        raise CheckFailed("台账 case_id 有重复")


def check_t95_010() -> str:
    if not LEDGER.is_file():
        raise CheckFailed(f"缺少台账 {LEDGER}")
    proc = _git("show", f"{BASE_COMMIT}:tests/test_agent_cases/台账用例.csv")
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise CheckFailed(f"git show {BASE_COMMIT} 台账失败：{detail}")
    base_rows = _parse_csv(proc.stdout)
    current_rows = _parse_csv(LEDGER.read_bytes())
    if len(base_rows[0]) != 15 or len(current_rows[0]) != 15 or base_rows[0] != current_rows[0]:
        raise CheckFailed("台账表头与基线的 15 列不一致")
    base_data = base_rows[1:]
    current_data = current_rows[1:]
    base_non_t95 = [row for row in base_data if not row[0].startswith("T95-")]
    current_non_t95 = [row for row in current_data if not row[0].startswith("T95-")]
    if current_non_t95 != base_non_t95:
        raise CheckFailed(
            f"非 T95 行被改动：base {len(base_non_t95)} 行，now {len(current_non_t95)} 行"
        )
    if len(base_non_t95) != 189:
        raise CheckFailed(f"{BASE_COMMIT} 台账数据行={len(base_non_t95)}，期望 189")
    tail = _check_ledger_tail(current_data, len(current_non_t95))
    _check_ledger_shape(current_data)
    return (
        f"台账数据行 {len(base_data)} -> {len(current_data)}（T95 +{len(tail)}）；"
        f"非 T95 行与 {BASE_COMMIT} 逐字段一致，15 列、case_id 唯一"
    )


def check_t95_011() -> str:
    proc = _git("diff", "--stat", f"{BASE_COMMIT}~1", BASE_COMMIT, "--", "evals/")
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise CheckFailed(f"git diff 失败：{detail}")
    stat = proc.stdout.decode("utf-8", "replace").strip()
    if stat:
        raise CheckFailed(f"evals/ 在 {BASE_COMMIT}~1..{BASE_COMMIT} 有改动：{stat}")
    if not DATASET.is_file():
        raise CheckFailed(f"缺少题集 {DATASET}")
    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    bad = {
        key: stats()[key].dataset_sha256
        for key in GROUP_KEYS
        if stats()[key].dataset_sha256 != digest
    }
    if bad:
        raise CheckFailed(f"题集 sha256 与 JSON 不符：{bad}；实际 {digest}")
    return (
        f"git diff --stat {BASE_COMMIT}~1 {BASE_COMMIT} -- evals/ 输出为空；"
        f"qa_dataset_v3.yaml sha256={digest[:16]}… 与三份 JSON 一致"
    )


def check_t95_012() -> str:
    group = stats()["C"]
    counts = Counter(group.tools)
    if counts.get("deep_research", 0) != 0 or counts.get("search_knowledge", 0) != group.total:
        raise CheckFailed(f"C 组工具调用与“审批未触发”不符：{dict(counts)}")
    for key in GROUP_KEYS:
        if _section8()[key]["approvals"] != 0:
            raise CheckFailed(f"§8 {key} 组审批次数非 0：{_section8()[key]['approvals']}")
    return (
        "C 组 20/20 题仅调 search_knowledge、deep_research=0，§8 审批次数 0 "
        "-> 审批分支未进入；审批执行成本本轮无样本，留待 I3 手测（OI-T95-001，不判失败）"
    )


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("T95-001 三份 JSON 完整性与同题同 commit", check_t95_001),
    ("T95-002 A 组均分独立复算", check_t95_002),
    ("T95-003 B 组均分独立复算", check_t95_003),
    ("T95-004 C 组均分独立复算", check_t95_004),
    ("T95-005 三组分数分布", check_t95_005),
    ("T95-006 judge_model 三组一致", check_t95_006),
    ("T95-007 有效失败/平均耗时/平均工具调用", check_t95_007),
    ("T95-008 与 §7 raw 9.65 的差异算式", check_t95_008),
    ("T95-009 台账 T01-018 转 pass 且指向 §8", check_t95_009),
    ("T95-010 台账其余行未改（189 行基线）", check_t95_010),
    ("T95-011 evals/ 零改动与题集 sha256", check_t95_011),
    ("T95-012 C 组审批分支未进入（局限记录）", check_t95_012),
]


def main() -> int:
    passed = 0
    failed = 0
    for name, check in CHECKS:
        try:
            evidence = check()
        except CheckFailed as exc:
            failed += 1
            print(f"[FAIL] {name} — {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - 任何异常都按失败记录，不中断其余检查
            failed += 1
            print(f"[FAIL] {name} — 未预期异常 {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"[PASS] {name} — {evidence}", flush=True)
    print(f"RESULT: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
