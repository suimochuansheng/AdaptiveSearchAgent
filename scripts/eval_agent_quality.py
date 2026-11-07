#!/usr/bin/env python3
"""Agent 端到端质量评估（阶段 B4）。

和 `evaluate_rag_system.py` 的分工：
- 那个评的是**检索层**（context_recall / faithfulness / answer_relevancy，直接调 RAG 网关）；
- 这个评的是**Agent 主循环**：把同一批题发给 `/chat/conversation`，拿最终回答，
  用 LLM judge 按 1~10 分打"答案正确性"，用于两个 commit 之间的前后对比。

用法：
    # 1) 先跑基线（在基线提交上，起后端）
    poetry run python scripts/eval_agent_quality.py --label baseline

    # 2) 切到当前分支，起后端，再跑一次
    poetry run python scripts/eval_agent_quality.py --label current

    # 3) 对比（退出码非 0 表示平均分下降超过阈值，可直接接 CI）
    poetry run python scripts/eval_agent_quality.py --compare <baseline.json>

说明：
- judge 默认用阿里云百炼的 qwen 模型（settings.ALIYUN_QWEN_*），与被测模型解耦；
- 每题独立 thread_id，避免上下文串味；
- 不依赖 Ragas：judge 只对比"回答 vs 参考答案"，不要求提供 contexts。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

import httpx  # noqa: E402
import yaml  # noqa: E402
from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402

DEFAULT_DATASET = _PROJ / "data" / "eval" / "qa_dataset_v3.yaml"
# 必须与 Ragas 基线报告里的 judge_model 一致，否则两次分数不可比
DEFAULT_JUDGE_MODEL = "qwen3.8-flash"
REPORT_DIR = _PROJ / "docs" / "test_reports"

JUDGE_SYSTEM_PROMPT = """你是严格的答案评分员。给你：问题、参考答案、被测系统的回答。
请按 1~10 分给"回答正确性"打分：
- 10：与参考答案一致且完整，没有多余错误
- 7~9：核心正确，细节缺失或表述不同但不等错
- 4~6：部分正确，漏掉关键结论或含少量错误
- 1~3：错误、答非所问或编造
只输出 JSON：{"score": <1~10 的整数>, "reason": "<20 字以内的理由>"}
不要输出解释、不要代码块。"""


# ── 纯函数（可单测，不依赖网络）────────────────────────────────────────


def parse_sse_frames(lines: list[str]) -> dict[str, Any]:
    """解析 SSE 行，返回 final 内容 / 错误 / 工具调用序列。"""
    answer = ""
    errors: list[str] = []
    tools: list[str] = []

    for line in lines:
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "final":
            answer = str(event.get("content") or "")
        elif kind == "error":
            errors.append(str(event.get("message") or "unknown error"))
        elif kind == "tool_call":
            tools.extend(str(tool.get("name")) for tool in event.get("tools") or [])

    return {"answer": answer, "errors": errors, "tools": tools}


def build_judge_prompt(question: str, ground_truth: str, answer: str) -> str:
    return (
        f"【问题】\n{question}\n\n"
        f"【参考答案】\n{ground_truth}\n\n"
        f"【被测系统的回答】\n{answer or '（空回答）'}\n\n"
        "请按评分标准输出 JSON。"
    )


def parse_judge_score(text: str) -> tuple[float | None, str]:
    """从 judge 输出里取分数；容忍代码块和多余文字，取不到就返回 None。"""
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            score = float(data["score"])
            return max(1.0, min(10.0, score)), str(data.get("reason", ""))[:60]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw)
    if match:
        score = float(match.group(1))
        return max(1.0, min(10.0, score)), "（从文本中提取）"
    return None, ""


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总一次评估的结果。"""
    scored = [item["score"] for item in results if item.get("score") is not None]
    failed = [item for item in results if item.get("score") is None]
    return {
        "total": len(results),
        "scored": len(scored),
        "failed": len(failed),
        "mean_score": round(sum(scored) / len(scored), 2) if scored else None,
        "min_score": min(scored) if scored else None,
        "max_score": max(scored) if scored else None,
    }


def compare_reports(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """按题目 id 对比两次评估。"""
    base_items = {item["id"]: item for item in baseline.get("results", [])}
    current_items = {item["id"]: item for item in current.get("results", [])}

    rows: list[dict[str, Any]] = []
    for qid, curr in current_items.items():
        base = base_items.get(qid)
        base_score = base.get("score") if base else None
        curr_score = curr.get("score")
        delta = (
            round(curr_score - base_score, 2)
            if base_score is not None and curr_score is not None
            else None
        )
        rows.append({"id": qid, "baseline": base_score, "current": curr_score, "delta": delta})

    comparable = [row["delta"] for row in rows if row["delta"] is not None]
    base_mean = baseline.get("summary", {}).get("mean_score")
    curr_mean = current.get("summary", {}).get("mean_score")
    mean_delta = (
        round(curr_mean - base_mean, 2) if base_mean is not None and curr_mean is not None else None
    )
    return {
        "rows": rows,
        "mean_delta": mean_delta,
        "improved": sum(1 for d in comparable if d > 0),
        "regressed": sum(1 for d in comparable if d < 0),
        "unchanged": sum(1 for d in comparable if d == 0),
    }


# ── 网络部分 ────────────────────────────────────────────────────────────


async def ask_agent(
    client: httpx.AsyncClient, base_url: str, thread_id: str, question: str
) -> dict:
    """发一条消息给会话端点，收集 SSE。"""
    lines: list[str] = []
    async with client.stream(
        "POST",
        f"{base_url.rstrip('/')}/chat/conversation",
        json={"thread_id": thread_id, "user_id": f"eval-{thread_id}", "message": question},
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                lines.append(line)
    return parse_sse_frames(lines)


def judge_answer(client: OpenAI, model: str, question: str, truth: str, answer: str) -> dict:
    """调 judge 模型打分。"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_judge_prompt(question, truth, answer),
                },
            ],
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - judge 失败要记录，不要中断整批
        return {"score": None, "reason": f"judge 调用失败: {exc}"}

    score, reason = parse_judge_score(text)
    return {"score": score, "reason": reason}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJ,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


async def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    raw = dataset_path.read_bytes()
    questions = yaml.safe_load(raw) or []

    if args.limit:
        questions = questions[: args.limit]

    judge_client = OpenAI(
        base_url=settings.ALIYUN_QWEN_BASE_URL,
        api_key=settings.ALIYUN_QWEN_API_KEY,
    )
    run_id = uuid.uuid4().hex[:8]
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:

        async def _one(item: dict[str, Any]) -> None:
            async with semaphore:
                qid = str(item.get("id"))
                question = str(item.get("question"))
                truth = str(item.get("ground_truth", ""))
                started = time.perf_counter()
                try:
                    answer_info = await ask_agent(
                        client, args.base_url, f"eval-{run_id}-{qid}", question
                    )
                except Exception as exc:  # noqa: BLE001
                    answer_info = {"answer": "", "errors": [str(exc)], "tools": []}

                judged = {"score": None, "reason": "未启用 judge"}
                if not args.no_judge:
                    judged = judge_answer(
                        judge_client, args.judge_model, question, truth, answer_info["answer"]
                    )

                results.append(
                    {
                        "id": qid,
                        "question": question,
                        "ground_truth": truth,
                        "answer": answer_info["answer"],
                        "tools": answer_info["tools"],
                        "errors": answer_info["errors"],
                        "score": judged["score"],
                        "reason": judged["reason"],
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                )
                print(f"[{len(results):02d}/{len(questions)}] {qid} score={judged['score']}")

        await asyncio.gather(*(_one(item) for item in questions))

    results.sort(key=lambda item: item["id"])
    report = {
        "label": args.label,
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit(),
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_size": len(results),
        "base_url": args.base_url,
        "judge_model": None if args.no_judge else args.judge_model,
        "results": results,
        "summary": summarize(results),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 端到端质量评估（LLM judge 1~10 分）")
    parser.add_argument("--base-url", default="http://localhost:8000", help="后端地址")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="题目 YAML")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    parser.add_argument("--concurrency", type=int, default=3, help="并发题数")
    parser.add_argument("--timeout", type=float, default=300.0, help="单题超时（秒）")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="裁判模型（必须与基线一致，默认 qwen3.8-flash）",
    )
    parser.add_argument("--no-judge", action="store_true", help="只收集回答，不打分")
    parser.add_argument("--label", default="current", help="这次跑的标签，如 baseline/current")
    parser.add_argument("--output", default="", help="报告输出路径（默认按提交+时间命名）")
    parser.add_argument("--compare", default="", help="拿这份报告与指定基线对比")
    parser.add_argument(
        "--max-drop", type=float, default=2.0, help="平均分允许下降幅度，超出则退出码 1"
    )
    args = parser.parse_args()

    if args.compare:
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        current_path = Path(args.output) if args.output else _latest_report()
        if current_path is None:
            print("找不到当前报告，请用 --output 指定")
            return 2
        current = json.loads(current_path.read_text(encoding="utf-8"))
        diff = compare_reports(baseline, current)
        print(
            f"\n基线均值 {baseline['summary']['mean_score']} → 当前 {current['summary']['mean_score']}"
        )
        print(
            f"均值变化 {diff['mean_delta']}（提升 {diff['improved']} 题 / 下降 {diff['regressed']} 题）"
        )
        for row in diff["rows"]:
            if row["delta"] is not None and row["delta"] < 0:
                print(f"  下降: {row['id']} {row['baseline']} → {row['current']} ({row['delta']})")
        if diff["mean_delta"] is not None and diff["mean_delta"] < -args.max_drop:
            print(f"❌ 平均分下降超过阈值 {args.max_drop}")
            return 1
        print("✅ 未超过退化阈值")
        return 0

    report = asyncio.run(run_eval(args))
    output = Path(args.output) if args.output else _default_output(args.label)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(
        f"\n平均分 {summary['mean_score']} / 有效 {summary['scored']} 题 / 失败 {summary['failed']} 题"
    )
    print(f"报告已写入 {output}")
    return 0


def _default_output(label: str) -> Path:
    return REPORT_DIR / f"agent_eval_{label}_{_git_commit()}.json"


def _latest_report() -> Path | None:
    candidates = sorted(REPORT_DIR.glob("agent_eval_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


if __name__ == "__main__":
    raise SystemExit(main())
