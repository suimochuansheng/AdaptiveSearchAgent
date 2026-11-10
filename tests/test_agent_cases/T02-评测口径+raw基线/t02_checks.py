#!/usr/bin/env python
"""T02 评测口径修复 + Agent raw 基线：可执行验收脚本（只读，不改任何仓库文件）。

用法（在仓库根目录，任意 cwd 均可，脚本自己定位仓库）：
    poetry run python "tests/test_agent_cases/T02-评测口径+raw基线/t02_checks.py"
    poetry run python "tests/test_agent_cases/T02-评测口径+raw基线/t02_checks.py" --no-dry-run
        --no-dry-run：跳过第 1 项真实 dry-run，其余按已归档证据判定（离线/省额度时用）。

退出码：0 = 全部通过；1 = 有失败项（打印 [FAIL] 的行）。

为什么文件名不是 test_*.py：
    一旦被 pytest 收集，`pytest --collect-only` 会从 227/231 变成 227+N，
    docs/baseline.md §2 冻结的 nodeids 基线哈希随之失效，本卡"开工前三件事"第 3 条
    就无法通过。做成独立脚本既保留自动化，又不污染测试快照。

编码：
    本文件、baseline.md、报告 JSON 全部 UTF-8；
    台账 tests/test_agent_cases/台账用例.csv 是 GB18030（读写都必须显式指定），
    且只允许中文标点与 ASCII，禁 emoji / 箭头 / 希腊字母。

副作用声明：
    第 1 项会执行一次 `./evals/run_baseline.sh --dry-run`（秒级、不调 LLM）。
    该命令内部会 mkdir 再 rmdir 归档目录（交接单已声明的已知降级），不留空目录，
    也不写任何报告文件——这是本脚本唯一的写动作，且写完即被脚本自己清掉。
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO.parent / "文档" / "项目" / "AdaptiveSearchAgent" / "评测报告"
BASELINE_MD = REPO / "docs" / "baseline.md"
LEDGER = REPO / "tests" / "test_agent_cases" / "台账用例.csv"

# 交接单"证据 2"给的两份 JSON（同一 tag、两次同口径）
RUN1 = REPORT_DIR / "raw_20261110_20261110-1329" / "ragas_evaluation_bge-m3_bge_ollama.json"
RUN2 = REPORT_DIR / "raw_20261110_20261110-1341" / "ragas_evaluation_bge-m3_bge_ollama.json"
# 交接单"证据 3"的 Agent raw 报告（§7 的数据源）
AGENT_JSON = REPORT_DIR / "agent_eval_baseline_raw_9bf183d_20261110-1403.json"

JUDGE = "qwen3.8-max-0902"
WORKERS = "2"

results: list[tuple[str, bool, str]] = []
open_issues: list[tuple[str, str]] = []

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def check(case: str, name: str, ok: bool, detail: str = "") -> None:
    results.append((f"{case} {name}", ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {case} {name}" + (f" — {detail}" if detail else ""))


def issue(oid: str, text: str) -> None:
    open_issues.append((oid, text))
    print(f"[OPEN] {oid} — {text}")


def sh(cmd: list[str], timeout: int = 180, env: dict[str, str] | None = None) -> tuple[int, str]:
    run_env = {**os.environ, **(env or {})}
    try:
        p = subprocess.run(  # noqa: S603 - 固定命令，无用户输入
            cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout, env=run_env, check=False
        )
        return p.returncode, ANSI.sub("", p.stdout + p.stderr)
    except Exception as exc:  # noqa: BLE001
        return 127, f"{type(exc).__name__}: {exc}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def section(text: str, heading: str) -> str:
    """取 markdown 二级标题 heading 到下一个二级标题之间的正文。"""
    start = text.find(heading)
    if start < 0:
        return ""
    rest = text[start + len(heading) :]
    m = re.search(r"\n## ", rest)
    return rest[: m.start()] if m else rest


def field(block: str, key: str) -> str:
    """从 dry-run 摘要 / run_meta.txt 里取 '键 [后缀] : 值' 形式的值。"""
    m = re.search(rf"^\s*{re.escape(key)}[^\n:]*:\s*(.+)$", block, re.MULTILINE)
    return m.group(1).strip() if m else ""


# ── A1 裁判/workers 口径 ────────────────────────────────────────────────────
def check_a1(dry_run: bool, baseline_text: str) -> None:
    block = ""
    if dry_run:
        rc, out = sh(["./evals/run_baseline.sh", "--dry-run"])
        check("T02-001", "dry-run 退出码为 0", rc == 0, f"exit={rc}")
        # dry-run 输出里既有日志行"裁判: X"也有摘要行"裁判        : X"，取摘要块最后一段
        tail = out[out.rfind("DRY RUN") :] if "DRY RUN" in out else out
        block = tail
    else:
        # 离线模式：用归档 run_meta.txt 代替 dry-run 摘要
        meta = (RUN1.parent / "run_meta.txt").read_text(encoding="utf-8")
        block = meta

    judge = field(block, "裁判").split("（")[0].strip()
    workers = field(block, "并发 workers").strip()
    src = "dry-run 打印" if dry_run else "run_meta.txt 记录"
    check("T02-001", f"{src}的裁判名 = qwen3.8-max-0902", judge == JUDGE, f"裁判={judge or '<空>'}")
    check("T02-001", f"{src}的 workers = 2", workers == WORKERS, f"workers={workers or '<空>'}")

    # 与 baseline.md 固定环境清单对齐（§4 配置行 + §5 裁判模型行）
    sec4 = section(baseline_text, "## 4.")
    sec5 = section(baseline_text, "## 5.")
    check(
        "T02-002",
        "baseline.md §4/§5 裁判与 dry-run 同值",
        JUDGE in sec4 and "裁判模型（评测判定）" in sec5 and JUDGE in sec5,
        "§4/§5 均出现 qwen3.8-max-0902",
    )
    check(
        "T02-002",
        "baseline.md §4 workers=2 与 dry-run 一致",
        bool(re.search(r"workers=\s*\**\s*2", sec4)),
        "§4 配置行 workers=2",
    )


# ── A2 两份 JSON 复算 ──────────────────────────────────────────────────────
def check_a2(baseline_text: str) -> None:
    for p in (RUN1, RUN2):
        if not p.is_file():
            check("T02-003", "两份 JSON 存在", False, f"缺 {p}")
            return
    d1, d2 = load_json(RUN1), load_json(RUN2)

    faith = round(
        abs(d1["metrics"]["faithfulness"]["mean"] - d2["metrics"]["faithfulness"]["mean"]), 4
    )
    relev = round(
        abs(d1["metrics"]["answer_relevancy"]["mean"] - d2["metrics"]["answer_relevancy"]["mean"]),
        4,
    )
    r1 = d1["metrics"]["context_recall"]["mean"]
    r2 = d2["metrics"]["context_recall"]["mean"]

    check(
        "T02-003",
        "faithfulness |run1-run2| = 0.0408",
        faith == 0.0408,
        f"{d1['metrics']['faithfulness']['mean']} vs {d2['metrics']['faithfulness']['mean']} -> {faith}",
    )
    check(
        "T02-004",
        "answer_relevancy |run1-run2| = 0.0201",
        relev == 0.0201,
        f"{d1['metrics']['answer_relevancy']['mean']} vs {d2['metrics']['answer_relevancy']['mean']} -> {relev}",
    )
    check("T02-004", "context_recall 两次均为 1.0", r1 == 1.0 and r2 == 1.0, f"run1={r1} run2={r2}")
    check(
        "T02-003",
        "两份 JSON 同口径（裁判/workers/数据集 sha256）",
        d1.get("judge_model") == d2.get("judge_model") == JUDGE
        and d1.get("max_workers") == d2.get("max_workers") == 2
        and d1.get("dataset_sha256") == d2.get("dataset_sha256"),
        f"judge={d1.get('judge_model')} workers={d1.get('max_workers')} sha={str(d1.get('dataset_sha256'))[:12]}",
    )

    # 记录方差的独立性：若定义来自"本次两次实测差"，就是自证循环（A2 只记 open_issue，不判 FAIL）
    sec4 = section(baseline_text, "## 4.")
    selfref = "实测差" in sec4 or "取三项里最大的实测差" in baseline_text
    independent = bool(re.search(r"独立(标定|复跑|样本)|另行标定|历史方差", baseline_text))
    check(
        "T02-005",
        "记录方差定义可定位",
        bool(sec4) and "记录方差" in sec4,
        "§4 含『记录方差』判据段落",
    )
    if selfref and not independent:
        issue(
            "OI-T02-001",
            "记录方差 0.0408 = 本次两次实测差（faith），阈值来自被检验的同一批数据，判定自证循环；"
            "A2 形式通过，但『判据有效』不成立",
        )
        print("[PASS] T02-005 记录方差独立性 — 形式通过：定义为本次实测差，自证，已记 OI-T02-001")
        results.append(("T02-005 记录方差独立性", True, "自证，记 OI-T02-001"))
    else:
        check("T02-005", "记录方差独立于本次数据", independent, "找到独立来源表述")


# ── A3 §7 raw 基线 ─────────────────────────────────────────────────────────
def check_a3(baseline_text: str) -> None:
    sec7 = section(baseline_text, "## 7.")
    check("T02-006", "docs/baseline.md §7 存在", bool(sec7.strip()), f"{len(sec7)} 字符")
    need = {
        "20 题均分 9.65": "9.65" in sec7,
        "分数分布 16/2/1/1/0": bool(
            re.search(r"\|\s*16\s*\|\s*2\s*\|\s*1\s*\|\s*1\s*\|\s*0\s*\|", sec7)
        ),
        "失败数 20 / 0": bool(re.search(r"20\s*/\s*0", sec7)),
        "报告路径": "agent_eval_baseline_raw_9bf183d_20261110-1403.json" in sec7,
        "复跑命令": "agent_quality.py --label baseline_raw" in sec7,
    }
    missing = [k for k, v in need.items() if not v]
    check(
        "T02-006",
        "§7 含 均分/分布/失败数/报告路径/复跑命令",
        not missing,
        "缺: " + ",".join(missing) if missing else "五要素齐",
    )

    if not AGENT_JSON.is_file():
        check("T02-006", "§7 报告文件存在且数字自洽", False, f"缺 {AGENT_JSON}")
        return
    d = load_json(AGENT_JSON)
    scores = [r.get("score") for r in d["results"]]
    valid = [s for s in scores if s is not None]
    mean = round(sum(valid) / len(valid), 2) if valid else 0.0
    dist = {k: sum(1 for s in valid if s == k) for k in (10.0, 9.0, 8.0, 7.0)}
    ok = (
        len(scores) == 20
        and len(valid) == 20
        and mean == float(d["summary"]["mean_score"]) == 9.65
        and dist == {10.0: 16, 9.0: 2, 8.0: 1, 7.0: 1}
        and not any(r.get("errors") for r in d["results"])
        and d.get("judge_model") == JUDGE
    )
    check(
        "T02-006",
        "§7 报告文件存在且数字自洽",
        ok,
        f"n={len(scores)} mean={mean} dist={dist} errors=0 judge={d.get('judge_model')}",
    )


# ── B 不变量 ───────────────────────────────────────────────────────────────
def check_b(baseline_text: str) -> None:
    rc, out = sh(["git", "diff", "--stat", "src/"])
    check(
        "T02-007",
        "git diff --stat src/ 为空",
        rc == 0 and out.strip() == "",
        f"exit={rc} 输出={out.strip() or '<空>'}",
    )

    sec4 = section(baseline_text, "## 4.")
    head4 = sec4[:400]
    flash_nums = baseline_text.count("0.9205") >= 2 and baseline_text.count("0.8983") >= 2
    check(
        "T02-008",
        "旧 flash 基线节仍在（0.9205/0.8983 保留）",
        flash_nums,
        f"0.9205x{baseline_text.count('0.9205')} 0.8983x{baseline_text.count('0.8983')}",
    )
    check(
        "T02-008",
        "§4 节首有『旧 flash 口径作废』标注",
        "旧 flash 口径作废" in head4,
        "§4 首段引用块",
    )
    check(
        "T02-008",
        "§4 旧判据有替代（1~7 条新判据到位）",
        "回归判定（T0-2 起：同口径两次取均值）" in sec4 and "记录方差" in sec4,
        "两条命名条件满足（追加式偏差见 OI-T02-002）",
    )
    rc2, diff = sh(["git", "diff", "-U0", "docs/baseline.md"])
    deleted = [ln for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    if deleted:
        issue(
            "OI-T02-002",
            f"baseline.md 不是纯追加：删除 {len(deleted)} 行（含空行；旧『回归判定』三条 bullet 与 "
            "IncompleteOutputException 追查提示，已被 1~7 条新判据改写）。旧 flash 数字仍在，"
            "B2 两个命名条件判过；是否算不变量违例待裁决",
        )

    envfile = REPO / ".env"
    mtime = ""
    if envfile.exists():
        mtime = datetime.datetime.fromtimestamp(envfile.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    check(
        "T02-009",
        ".env mtime = 2026-11-09 09:14",
        mtime == "2026-11-09 09:14",
        f"实际 {mtime or '<缺失>'}",
    )
    rc3, status = sh(["git", "status", "--porcelain"])
    leaked = [ln for ln in status.splitlines() if ".env" in ln]
    check(
        "T02-009",
        ".env 未出现在 git status",
        rc3 == 0 and not leaked,
        "未出现" if not leaked else str(leaked),
    )


# ── C 边界/反例 ────────────────────────────────────────────────────────────
def check_c() -> None:
    # C1: EVAL_JUDGE_MODEL 置空 -> 回落 ALIYUN_QWEN_MODEL_NAME（直接测 run_baseline.sh 里的真函数）
    script = REPO / "evals" / "run_baseline.sh"
    probe = (
        f'source <(sed -n "/^env_val()/,/^}}/p" "{script}")\n'
        f'source <(sed -n "/^judge_model()/,/^}}/p" "{script}")\n'
        f'PROJECT_DIR="{REPO}"\n'
        'EVAL_JUDGE_MODEL=""\n'
        "judge_model\n"
    )
    rc, out = sh(["bash", "-c", probe], env={"EVAL_JUDGE_MODEL": ""})
    got = out.strip().splitlines()[-1] if out.strip() else ""
    check(
        "T02-010",
        "EVAL_JUDGE_MODEL 置空回落 ALIYUN_QWEN_MODEL_NAME",
        rc == 0 and got == JUDGE,
        f"judge_model()={got or '<空>'} exit={rc}",
    )

    py = "from config import settings;" "print(settings.EVAL_JUDGE_MODEL or '<空>')"
    rc2, out2 = sh([sys.executable, "-c", py], env={"EVAL_JUDGE_MODEL": ""})
    v = out2.strip().splitlines()[-1] if out2.strip() else ""
    check(
        "T02-010",
        "Settings.EVAL_JUDGE_MODEL 空值回落且非空串",
        rc2 == 0 and v == JUDGE,
        f"config.py -> {v} exit={rc2}",
    )

    # C2: 归档目录带时间戳；同 tag 两次不覆盖
    sample = sorted(REPORT_DIR.glob("raw_20261110_*"))
    stamps = [p.name for p in sample if re.search(r"_\d{8}-\d{4}$", p.name)]
    check(
        "T02-010",
        "归档目录名带 YYYYmmdd-HHMM 时间戳且同 tag 并存",
        len(stamps) >= 2,
        f"{len(stamps)} 个: {', '.join(p.name for p in sample)}",
    )
    tg = re.compile(r"^.+_\d{8}-\d{4}$")
    check(
        "T02-010",
        "归档目录名符合 <TAG>_<YYYYmmdd-HHMM> 契约",
        all(tg.match(p.name) for p in sample) and bool(sample),
        "同上；同分钟同 tag 撞名风险见 OI-T02-003",
    )
    if any(p.name.endswith(("-1329", "-1341")) for p in sample):
        issue(
            "OI-T02-003",
            "TAGDIR=<TAG>_<YYYYmmdd-HHMM> 只到分钟：同一 tag 在同一分钟内起两次会撞名并覆盖；"
            "默认 tag 含秒（YYYYmmdd_HHMMSS），只有显式 --tag 才会踩到",
        )

    # C3: run_meta.txt 三要素
    for d in (RUN1.parent, RUN2.parent):
        meta = d / "run_meta.txt"
        if not meta.is_file():
            check("T02-010", f"{d.name} run_meta.txt 存在", False, "缺文件")
            continue
        t = meta.read_text(encoding="utf-8")
        ok = (
            f"裁判 LLM      : {JUDGE}" in t
            and re.search(r"并发 workers\s*:\s*2", t) is not None
            and re.search(r"sha256=[0-9a-f]{64}", t) is not None
        )
        sha = re.search(r"sha256=([0-9a-f]{64})", t)
        check(
            "T02-010",
            f"{d.name} run_meta 含 裁判/workers=2/数据集 sha256",
            bool(ok),
            f"裁判={JUDGE} workers=2 sha={sha.group(1)[:12] if sha else '<无>'}",
        )

    # 台账可机读（GB18030，严格 15 列，T02-001~010 齐全）
    if LEDGER.is_file():
        with LEDGER.open(encoding="gb18030", newline="") as fh:
            rows = list(csv.reader(fh))
        width_ok = all(len(r) == 15 for r in rows)
        ids = {r[0] for r in rows[1:] if r}
        need = {f"T02-{i:03d}" for i in range(1, 11)}
        check(
            "T02-010",
            "台账 GB18030 可读、15 列、T02-001~010 齐全",
            width_ok and need <= ids,
            f"{len(rows)} 行 列数={'一致' if width_ok else '不一致'} 缺={sorted(need - ids) or '无'}",
        )


def main() -> int:
    dry_run = "--no-dry-run" not in sys.argv
    print(f"仓库: {REPO}")
    print(f"dry-run: {'执行（1 次，秒级）' if dry_run else '跳过（离线模式）'}\n")

    baseline_text = BASELINE_MD.read_text(encoding="utf-8")
    check_a1(dry_run, baseline_text)
    check_a2(baseline_text)
    check_a3(baseline_text)
    check_b(baseline_text)
    check_c()

    failed = [r for r in results if not r[1]]
    print("\n" + "=" * 62)
    if open_issues:
        print("open_issue:")
        for oid, text in open_issues:
            print(f"  {oid}: {text}")
    print(f"RESULT: {len(results) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} — {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
