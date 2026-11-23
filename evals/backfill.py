"""坏 case 回流（半自动）：台账非 pass 行 -> 回归集候选模板。

只生成候选，**不自动入集**：写入 `evals/datasets/regression/candidates.json`，
每条带 `source_case_id`，由人确认后再转成正式 YAML 题。

台账约定（tests/test_agent_cases/README.md）：**GB18030**、15 列、CRLF。

用法：

    python -m evals.backfill \
        --ledger tests/test_agent_cases/台账用例.csv \
        --out evals/datasets/regression/candidates.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[1]

DEFAULT_LEDGER = _PROJ / "tests" / "test_agent_cases" / "台账用例.csv"
DEFAULT_OUT = _PROJ / "evals" / "datasets" / "regression" / "candidates.json"

REQUIRED_COLUMNS = ("case_id", "title", "status", "expected")


class BackfillError(ValueError):
    """台账读不了 / 列不全；CLI 以退出码 2 结束。"""


def read_ledger(path: Path) -> list[dict[str, str]]:
    """按 GB18030 读台账；返回全部数据行（含 pass）。"""
    try:
        text = path.read_bytes().decode("gb18030")
    except FileNotFoundError:
        raise BackfillError(f"找不到台账：{path}") from None
    except UnicodeDecodeError as exc:
        raise BackfillError(f"台账不是 GB18030 编码：{path}（{exc}）") from None

    reader = csv.DictReader(io.StringIO(text))
    header = [name.strip() for name in (reader.fieldnames or [])]
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise BackfillError(f"台账缺列：{', '.join(missing)}；实际表头：{header}")
    rows: list[dict[str, str]] = []
    for row in reader:
        cleaned = {str(key).strip(): (value or "").strip() for key, value in row.items() if key}
        if cleaned.get("case_id"):
            rows.append(cleaned)
    return rows


def build_candidates(rows: list[dict[str, str]], ledger: Path) -> list[dict[str, str]]:
    """台账 `status != pass` 的行 -> 候选 case 模板（一条不多、一条不少）。"""
    candidates: list[dict[str, str]] = []
    for row in rows:
        status = row.get("status", "").strip()
        if status.lower() == "pass":
            continue
        source_case_id = row.get("case_id", "").strip()
        candidates.append(
            {
                "candidate_id": f"REG-{source_case_id}",
                "source_case_id": source_case_id,
                "source_status": status,
                "source_module": row.get("module", ""),
                "source_level": row.get("level", ""),
                "source_priority": row.get("priority", ""),
                "question": row.get("title", ""),
                "ground_truth": row.get("expected", ""),
                "steps": row.get("steps", ""),
                "observed": row.get("notes", ""),
                "open_issue": row.get("open_issue", ""),
                "ledger": str(ledger),
                "review": "pending",
                "note": "候选模板；需人工确认后转成 YAML 加入黄金集/回归集，本脚本不自动入集",
            }
        )
    return candidates


def write_candidates(
    out: Path, ledger: Path, rows: list[dict[str, str]]
) -> tuple[dict[str, object], list[dict[str, str]]]:
    candidates = build_candidates(rows, ledger)
    payload: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ledger": str(ledger),
        "ledger_rows": len(rows),
        "non_pass_rows": len(candidates),
        "count": len(candidates),
        "candidates": candidates,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="台账非 pass 行 -> 回归集候选模板（不自动入集）")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="台账 CSV（GB18030）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="候选 JSON 输出路径")
    args = parser.parse_args(argv)

    try:
        rows = read_ledger(args.ledger)
        payload, candidates = write_candidates(args.out, args.ledger, rows)
    except BackfillError as exc:
        print(f"输入不合法：{exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"写候选失败：{exc}", file=sys.stderr)
        return 2

    ids = [candidate["source_case_id"] for candidate in candidates]
    print(
        f"台账 {payload['ledger_rows']} 行，非 pass {payload['non_pass_rows']} 行 -> "
        f"{payload['count']} 条候选：{args.out}"
    )
    print(f"来源 case_id：{', '.join(ids) if ids else '（无）'}")
    print("仅生成候选，未自动入集；人工确认后再转成正式 YAML 题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
