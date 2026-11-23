"""真实评测执行器：按配置矩阵跑、落库、对比、出报告。

边界（见 docs/eval-gate.md）：
- 本脚本**真跑模型**（起本地后端 + Ollama / RAG 网关 + 裁判），只在本地或发版时用；
- CI 不跑本脚本，只跑 `scripts/smoke_eval_gate.py`（fixture 数据，零模型调用）；
- 每个 config 一个独立后端进程，启动时把 `evals/configs.py` 的 env 直接注入，
  跑完即停，避免开关串味；
- 开跑前探活 RAG / 模型 / PG / Redis；失败默认拒绝开跑（退出码 2），
  显式 `--allow-degraded` 才继续，并在 meta 标 `valid_for_quality=false`；
- `--limit N < dataset_size` 时 meta 标 `partial=true / publishable=false`，
  报告顶部提示“非全量，不可用于发布”。

用法：

    python -m evals.run --configs A --limit 1
    python -m evals.run --configs A,B,C --set golden20
    python -m evals.run --configs A,B --compare last
    python -m evals.run --configs A --limit 1 --allow-degraded
    python -m evals.run --configs A,B --compare last --allow-invalid

产物：`evals/results/<run_id>/{meta.json,per_case.json,report.md}`。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_PROJ = Path(__file__).resolve().parents[1]
if str(_PROJ) not in sys.path:  # 允许从任意 cwd 运行
    sys.path.insert(0, str(_PROJ))

import httpx  # noqa: E402
import yaml  # noqa: E402
from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402
from evals import configs as config_matrix  # noqa: E402
from evals import gate, report, store  # noqa: E402
from evals.agent_quality import DEFAULT_JUDGE_MODEL, judge_answer, parse_sse_frames  # noqa: E402

#: `--set` 到数据集文件的映射。黄金集 = docs/baseline.md §7/§8 的 20 题。
DATASET_SETS: dict[str, Path] = {
    "golden20": _PROJ / "evals" / "datasets" / "qa_dataset_v3.yaml",
    "smoke": _PROJ / "evals" / "datasets" / "qa_dataset_smoke.yaml",
}


class EvalRunError(RuntimeError):
    """可读的评测执行错误（不打印栈，CLI 以退出码 2 结束）。"""


def _model_name() -> str:
    provider = str(getattr(settings, "llm_provider", "ollama") or "ollama")
    if provider == "deepseek":
        return settings.DEEPSEEK_MODEL_NAME or "deepseek(unknown)"
    return settings.ollama_model_name or "ollama(unknown)"


def _health_ok(base_url: str, timeout: float = 1.0) -> bool:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


#: 评测前探活的单项超时（秒）；模型首次加载可能慢，单独放宽。
PREFLIGHT_TIMEOUT = 5.0
MODEL_PROBE_TIMEOUT = 30.0


def _probe_rag() -> tuple[bool, str]:
    base = (settings.INGEST_SERVICE_URL or "").rstrip("/")
    if not base:
        return False, "INGEST_SERVICE_URL 为空"
    try:
        response = httpx.get(f"{base}/health", timeout=PREFLIGHT_TIMEOUT)
    except httpx.HTTPError as exc:
        return False, f"{base}/health 不可达（{type(exc).__name__}: {exc}）"
    if response.status_code != 200:
        return False, f"{base}/health -> HTTP {response.status_code}"
    return True, f"{base}/health -> 200"


def _probe_model() -> tuple[bool, str]:
    """一次极短调用，确认模型真的能出 token（不是只看端口）。"""
    provider = str(getattr(settings, "llm_provider", "ollama") or "ollama")
    try:
        if provider == "deepseek":
            base = (settings.openai_base_url or "").rstrip("/")
            model = settings.DEEPSEEK_MODEL_NAME
            if not base or not model or not settings.deepseek_api_key:
                return False, "DeepSeek 配置不完整（openai_base_url / model / api_key）"
            client = OpenAI(base_url=base, api_key=settings.deepseek_api_key)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
                timeout=MODEL_PROBE_TIMEOUT,
            )
            return True, f"{model} 可调用"

        base = (settings.ollama_base_url or "").rstrip("/")
        model = settings.ollama_model_name
        if not base or not model:
            return False, "Ollama 配置不完整（ollama_base_url / ollama_model_name）"
        response = httpx.post(
            f"{base}/api/generate",
            json={
                "model": model,
                "prompt": "ping",
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=MODEL_PROBE_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - 探活失败要变成可读原因，不抛栈
        return False, f"模型调用失败（{type(exc).__name__}: {exc}）"
    if response.status_code != 200:
        return False, f"Ollama /api/generate -> HTTP {response.status_code}: {response.text[:120]}"
    return True, f"{model} 可调用"


def _tcp_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_pg() -> tuple[bool, str]:
    host = settings.postgres_host or ""
    port = settings.postgres_port or 5432
    if not host:
        return False, "postgres_host 为空"
    ok = _tcp_reachable(host, port)
    return ok, f"{host}:{port} {'可达' if ok else '不可达'}"


def _probe_redis() -> tuple[bool, str]:
    raw = settings.REDIS_URL or ""
    if not raw:
        return False, "REDIS_URL 为空"
    parsed = urlparse(raw)
    host = parsed.hostname or ""
    port = parsed.port or 6379
    if not host:
        return False, f"REDIS_URL 解析不到 host：{raw}"
    ok = _tcp_reachable(host, port)
    return ok, f"{host}:{port} {'可达' if ok else '不可达'}"


def _probe_environment() -> tuple[dict[str, Any], dict[str, str]]:
    """探活 RAG / 模型 / PG / Redis；返回 meta.environment 与失败原因。"""
    checks = (
        ("rag", _probe_rag),
        ("model", _probe_model),
        ("pg", _probe_pg),
        ("redis", _probe_redis),
    )
    environment: dict[str, Any] = {}
    details: dict[str, str] = {}
    for name, probe in checks:
        ok, detail = probe()
        environment[f"{name}_ok"] = ok
        if not ok:
            details[name] = detail
    environment["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return environment, details


def _tail(path: Path, lines: int = 20) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _schema_sha256(env: dict[str, str]) -> str:
    """复用 scripts/smoke_tools_schema.py 的口径，读它打印的 current_sha256。"""
    merged = os.environ.copy()
    merged.update(env)
    try:
        result = subprocess.run(
            [sys.executable, str(_PROJ / "scripts" / "smoke_tools_schema.py")],
            cwd=_PROJ,
            env=merged,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"warn: 取 schema 指纹失败：{exc}", file=sys.stderr)
        return "unknown"
    match = re.search(r"current_sha256\s*:\s*([0-9a-f]{64})", result.stdout)
    if match:
        return match.group(1)
    print(
        f"warn: schema 脚本没有输出指纹（exit={result.returncode}）：{result.stderr.strip()[:200]}",
        file=sys.stderr,
    )
    return "unknown"


def _load_questions(dataset: Path, limit: int) -> tuple[list[dict[str, Any]], bytes, int]:
    try:
        raw = dataset.read_bytes()
    except OSError as exc:
        raise EvalRunError(f"读数据集失败：{dataset}（{exc}）") from None
    try:
        questions = yaml.safe_load(raw) or []
    except yaml.YAMLError as exc:
        raise EvalRunError(f"数据集 YAML 解析失败：{dataset}（{exc}）") from None
    if not isinstance(questions, list) or not questions:
        raise EvalRunError(f"数据集必须是非空 YAML 列表：{dataset}")
    total = len(questions)
    if limit > 0:
        questions = questions[:limit]
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict) or not item.get("question"):
            raise EvalRunError(f"数据集第 {index} 题缺 question：{dataset}")
        if not item.get("id"):
            item["id"] = f"q{index:03d}"
    return questions, raw, total


def _start_backend(
    config: config_matrix.EvalConfig,
    base_url: str,
    startup_timeout: float,
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], Any]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    env = os.environ.copy()
    env.update(config.env)
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api_main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(_PROJ),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_file.close()
            raise EvalRunError(
                f"后端启动失败（config {config.id}，exit={proc.returncode}）：\n{_tail(log_path)}"
            )
        if _health_ok(base_url):
            print(f"[{config.id}] 后端已就绪：{base_url}（log={log_path}）", flush=True)
            return proc, log_file
        time.sleep(0.5)

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_file.close()
    raise EvalRunError(
        f"后端 {startup_timeout:.0f}s 内未通过 /health（config {config.id}）：\n{_tail(log_path)}"
    )


def _stop_backend(proc: subprocess.Popen[bytes] | None, log_file: Any, base_url: str) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if log_file is not None:
        log_file.close()
    for _ in range(20):
        if not _health_ok(base_url):
            return
        time.sleep(0.25)


async def _ask_agent(
    client: httpx.AsyncClient,
    base_url: str,
    thread_id: str,
    question: str,
) -> tuple[dict[str, Any], int | None]:
    """发一条消息，收 SSE；额外从 kpi 帧取 total_tokens。"""
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

    info = parse_sse_frames(lines)
    tokens: int | None = None
    for line in lines:
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if event.get("type") != "kpi":
            continue
        data = event.get("data")
        if isinstance(data, dict):
            value = data.get("total_tokens")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                tokens = int(value)
    return info, tokens


async def _evaluate(
    config: config_matrix.EvalConfig,
    run_id: str,
    questions: list[dict[str, Any]],
    judge_client: OpenAI,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index, item in enumerate(questions, start=1):
            case_id = str(item.get("id"))
            question = str(item.get("question") or "")
            truth = str(item.get("ground_truth") or "")
            started = time.perf_counter()
            answer = ""
            tools: list[str] = []
            tokens: int | None = None
            error: str | None = None
            try:
                info, tokens = await _ask_agent(
                    client, args.base_url, f"eval-{run_id}-{case_id}", question
                )
                answer = str(info.get("answer") or "")
                tools = [str(name) for name in info.get("tools") or []]
                errors = [str(item) for item in info.get("errors") or []]
                if errors:
                    error = "; ".join(errors)
            except Exception as exc:  # noqa: BLE001 - 单题失败要落库，不中断整批
                error = f"{type(exc).__name__}: {exc}"

            latency_ms = int(round((time.perf_counter() - started) * 1000))
            judged: dict[str, Any] = {"score": None, "reason": "无回答，跳过裁判"}
            if answer:
                judged = judge_answer(judge_client, args.judge_model, question, truth, answer)
            if judged.get("score") is None and not error:
                error = f"judge: {judged.get('reason') or '未返回分数'}"

            results.append(
                {
                    "case_id": case_id,
                    "question": question,
                    "answer_digest": hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16],
                    "score": judged.get("score"),
                    "latency_ms": latency_ms,
                    "tokens": tokens,
                    "tool_calls": tools,
                    "error": error,
                }
            )
            print(
                f"[{config.id}] [{index}/{len(questions)}] {case_id} "
                f"score={judged.get('score')} {latency_ms}ms tools={len(tools)}",
                flush=True,
            )
    results.sort(key=lambda item: item["case_id"])
    return results


def _partial_flags(limit: int, dataset_size: int) -> tuple[bool, bool]:
    """返回 (partial, publishable)：--limit 小于数据集题数即非全量。"""
    partial = bool(limit > 0 and limit < dataset_size)
    return partial, not partial


def _previous_run(
    exclude_run_id: str, dataset_sha256: str, *, include_invalid: bool = False
) -> gate.Run | None:
    runs = [
        run
        for run in store.list_runs()
        if run.run_id != exclude_run_id
        and (not dataset_sha256 or run.meta.get("dataset_sha256") == dataset_sha256)
        and (include_invalid or run.meta.get("valid_for_quality") is not False)
    ]
    return runs[-1] if runs else None


def _print_diff(result: gate.GateResult) -> None:
    print("")
    print(f"分题对比（{result.candidate_run_id} vs {result.baseline_run_id}）：")
    print("case      base  cand  diff    归因")
    for item in result.evidence:
        base = "—" if item.baseline_score is None else f"{item.baseline_score:.2f}"
        cand = "—" if item.candidate_score is None else f"{item.candidate_score:.2f}"
        delta = "—" if item.delta is None else f"{item.delta:+.2f}"
        print(f"{item.case_id:<9} {base:<5} {cand:<5} {delta:<7} {item.attribution or '—'}")
    if result.quality_warning:
        print(f"质量警告：{result.quality_warning}")
    if result.partial_compare:
        print(f"题集：仅比较共同题 compared_cases={result.compared_cases} skipped={result.skipped}")
    if result.code_version_changed:
        print(f"代码版本不同：{result.baseline_git} vs {result.candidate_git}")
    print(result.reason)


def _run_one(
    config: config_matrix.EvalConfig,
    questions: list[dict[str, Any]],
    dataset: Path,
    dataset_sha256: str,
    dataset_size: int,
    args: argparse.Namespace,
    judge_client: OpenAI,
    environment: dict[str, Any],
    valid_for_quality: bool,
    invalid_reason: str,
) -> tuple[str, bool]:
    run_id = store.new_run_id()
    log_path = store.RESULTS_DIR / ".logs" / f"{run_id}.log"
    proc: subprocess.Popen[bytes] | None = None
    log_file: Any = None

    if _health_ok(args.base_url):
        if not args.use_running_backend:
            raise EvalRunError(
                f"{args.base_url} 已有服务在跑，无法保证开关注入；"
                "请先停掉它，或用 --use-running-backend 明确复用（不注入开关）"
            )
        print(f"warn: 复用已在运行的后端 {args.base_url}；config {config.id} 的开关未注入")
    else:
        if args.use_running_backend:
            raise EvalRunError(f"--use-running-backend 指定了复用，但 {args.base_url} 没有服务")
        proc, log_file = _start_backend(config, args.base_url, args.startup_timeout, log_path)

    try:
        results = asyncio.run(_evaluate(config, run_id, questions, judge_client, args))
    finally:
        _stop_backend(proc, log_file, args.base_url)

    partial, publishable = _partial_flags(args.limit, dataset_size)
    meta: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_id": config.id,
        "config_name": config.name,
        "config_description": config.description,
        "model": _model_name(),
        "judge_model": args.judge_model,
        "n_cases": len(results),
        "n_scored": sum(1 for item in results if item.get("score") is not None),
        "git_commit": store.git_commit(),
        "flags": dict(config.env),
        "schema_sha256": _schema_sha256(dict(config.env)),
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha256,
        "dataset_size": dataset_size,
        "partial": partial,
        "publishable": publishable,
        "base_url": args.base_url,
        "recorded_variance": (
            args.variance if args.variance is not None else gate.DEFAULT_RECORDED_VARIANCE
        ),
        "backend": "external" if args.use_running_backend else "managed",
        "environment": environment,
        "valid_for_quality": valid_for_quality,
        "invalid_reason": invalid_reason,
    }
    store.save_run(run_id, meta, results)
    print(f"[{config.id}] 已落库：{store.RESULTS_DIR / run_id}", flush=True)

    compare_result: gate.GateResult | None = None
    if args.compare == "last":
        previous = _previous_run(run_id, dataset_sha256, include_invalid=args.allow_invalid)
        if previous is None:
            print(f"[{config.id}] 没有同数据集的历史 run，本次不生成 diff", flush=True)
        else:
            try:
                compare_result = gate.compare(
                    previous,
                    store.load_run(run_id),
                    args.variance,
                    allow_invalid_baseline=args.allow_invalid,
                    allow_invalid_candidate=args.allow_invalid,
                )
            except gate.GateInputError as exc:
                print(f"[{config.id}] 未判定：{exc}", flush=True)

    report_path = report.write_markdown(run_id, compare_result)
    print(f"[{config.id}] 报告：{report_path}", flush=True)
    if compare_result is not None:
        _print_diff(compare_result)
    return run_id, bool(compare_result and not compare_result.passed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按配置矩阵跑真实评测并落库")
    parser.add_argument("--configs", default="A", help="逗号分隔的配置 id，如 A,B,C")
    parser.add_argument("--set", default="golden20", choices=sorted(DATASET_SETS), help="数据集")
    parser.add_argument("--dataset", default="", help="自定义数据集 YAML（覆盖 --set）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0 = 全部）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="后端地址")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="裁判模型")
    parser.add_argument("--timeout", type=float, default=300.0, help="单题超时（秒）")
    parser.add_argument("--startup-timeout", type=float, default=90.0, help="后端启动超时（秒）")
    parser.add_argument("--variance", type=float, default=None, help="记录方差，默认 0.0408")
    parser.add_argument(
        "--compare",
        choices=["last"],
        default=None,
        help="跑完后与同数据集的上一次 run 比较并写进报告",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="对比时纳入并允许 valid_for_quality=false 的 run（默认过滤）",
    )
    parser.add_argument(
        "--use-running-backend",
        action="store_true",
        help="复用已在运行的后端（不注入 config 开关，仅排障用）",
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="环境探活失败仍开跑（只验链路；meta.valid_for_quality=false）",
    )
    args = parser.parse_args(argv)

    try:
        selected = config_matrix.get_configs(config_matrix.parse_config_ids(args.configs))
    except ValueError as exc:
        print(f"输入不合法：{exc}", file=sys.stderr)
        return 2

    dataset = Path(args.dataset).expanduser() if args.dataset else DATASET_SETS[args.set]
    try:
        questions, raw, dataset_size = _load_questions(dataset, args.limit)
    except EvalRunError as exc:
        print(f"输入不合法：{exc}", file=sys.stderr)
        return 2

    environment, failures = _probe_environment()
    probe_text = " ".join(
        f"{key.removesuffix('_ok')}={'ok' if value else 'fail'}"
        for key, value in environment.items()
        if key.endswith("_ok")
    )
    print(f"环境探活：{probe_text}", flush=True)
    if failures:
        print("环境未就绪：" + ", ".join(failures), file=sys.stderr)
        for name, detail in failures.items():
            print(f"  - {name}: {detail}", file=sys.stderr)
        if not args.allow_degraded:
            print(
                "拒绝开跑；如只验链路，可加 --allow-degraded（meta 会标 valid_for_quality=false）",
                file=sys.stderr,
            )
            return 2
        valid_for_quality = False
        invalid_reason = "环境未就绪：" + "; ".join(
            f"{name}: {detail}" for name, detail in failures.items()
        )
        print(
            f"warn: --allow-degraded 已开启，meta.valid_for_quality=false；{invalid_reason}",
            flush=True,
        )
    else:
        valid_for_quality = True
        invalid_reason = ""

    judge_client = OpenAI(
        base_url=settings.ALIYUN_QWEN_BASE_URL,
        api_key=settings.ALIYUN_QWEN_API_KEY,
    )
    print(
        f"数据集：{dataset}（{len(questions)} 题）"
        f" | 裁判：{args.judge_model} | 后端：{args.base_url}",
        flush=True,
    )

    any_failed = False
    for config in selected:
        print(f"\n=== config {config.id}（{config.name}）===", flush=True)
        try:
            _, failed = _run_one(
                config,
                questions,
                dataset,
                hashlib.sha256(raw).hexdigest(),
                dataset_size,
                args,
                judge_client,
                environment,
                valid_for_quality,
                invalid_reason,
            )
        except EvalRunError as exc:
            print(f"评测失败：{exc}", file=sys.stderr)
            return 2
        any_failed = any_failed or failed

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
