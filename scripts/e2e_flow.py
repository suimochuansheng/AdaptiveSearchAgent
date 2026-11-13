#!/usr/bin/env python
"""跨层端到端校验（E2E）：一次真实请求同时断言五处一致。

一条命令跑完 5 个场景，对**同一次真实请求**同时核对：

| 层 | 证据来源 |
|---|---|
| 传输层 | SSE 帧（`token` / `tool_call` / `ping` / `final` / `error` / `interrupt`） |
| 状态机 | PostgreSQL `task_states` + LangGraph `checkpoints` |
| 凭据层 | Redis `approval:<id>`（签发 / 消费 / 重放） |
| 追踪层 | `logs/trace/<今天>.jsonl`（kind / 父子 / attrs 白名单） |
| 指标层 | `GET /metrics`（requests / approvals 计数增量、标签基数） |

场景：

- S1 普通会话：帧集合合法、有 final 无 interrupt、state=completed、checkpoints>0、requests +1；
- S2 审批全链路：interrupt 八字段 -> Redis 凭据 -> state=interrupted -> resume 200 + final ->
  凭据消失 -> state=completed -> tool span `error_kind=approval_required` -> approvals +1；
- S3 重放与过期：同一 `approval_id` 再 resume -> 409 且凭据仍不存在（410 需重启调 TTL，本卡 SKIP）；
- S4 技能（模型依赖）：trace 的 tool 名集合含 `list_skills` + `load_skill`；模型没调工具 -> SKIP；
- S5 指标与 trace 自洽（只读）：无 `thread_id=`/`session_id=`/`approval_id=` 标签、
  kind 六枚举、`parent_span_id` 可达、turn attrs 白名单。

用法（仓库根目录）：

```bash
poetry run python scripts/e2e_flow.py
poetry run python scripts/e2e_flow.py --api http://localhost:8300   # 或 E2E_API=...
```

前置（不满足时打印指引并退出 2）：

1. 后端 `GET {api}/health` 返回 200（默认 `http://localhost:8200`，可用 `--api` / `E2E_API` 覆盖）；
2. PostgreSQL 可达（默认 `docker exec postgres_dev psql -U langgraph_user -d langgraph_db_dev`）；
3. Redis 可达（默认 `docker exec redis_cache redis-cli`，密码取 `E2E_REDIS_PASSWORD` /
   `REDIS_PASSWORD` / 仓库 `.env` 的 `REDIS_PASSWORD`）。

环境变量覆盖：

- `E2E_API`：后端地址；
- `E2E_PSQL`：psql 命令**前缀**（脚本在末尾补 `-v`，SQL 走 stdin；`docker exec` 需带 `-i`），
  例如 `E2E_PSQL="docker exec -i postgres_dev psql -U langgraph_user -d langgraph_db_dev -tA"`；
- `E2E_REDIS`：redis-cli 命令**前缀**（脚本在末尾补子命令），例如
  `E2E_REDIS="docker exec redis_cache redis-cli --no-auth-warning -a ***"`；
- `E2E_PG_CONTAINER` / `E2E_PG_USER` / `E2E_PG_DB` / `E2E_REDIS_CONTAINER`：参数化容器与库名。

退出码：0 = 全过；1 = 有 FAIL；2 = 前置不满足；3 = 有 SKIP 但无 FAIL。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from unicodedata import east_asian_width

import httpx

REPO = Path(__file__).resolve().parents[1]
TRACE_DIR = REPO / "logs" / "trace"

API_DEFAULT = "http://localhost:8200"
PG_CONTAINER_DEFAULT = "postgres_dev"
PG_USER_DEFAULT = "langgraph_user"
PG_DB_DEFAULT = "langgraph_db_dev"
REDIS_CONTAINER_DEFAULT = "redis_cache"

#: 审批契约 frozen：`interrupt.data` 恰好八个字段（`docs/contracts/approval.md` §1）
APPROVAL_FIELDS: tuple[str, ...] = (
    "approval_id",
    "tool_call_id",
    "tool",
    "args_digest",
    "cost_estimate",
    "budget_remaining",
    "permission",
    "expires_at",
)

#: 普通会话允许出现的帧类型（interrupt 单列断言，不在这个集合里）
CHAT_FRAME_TYPES: frozenset[str] = frozenset({"token", "tool_call", "ping", "final", "error"})

#: trace 契约：kind 六枚举 / attrs 六键（turn 额外两键）
SPAN_KINDS: frozenset[str] = frozenset({"turn", "llm", "tool", "compress", "offload", "approval"})
SPAN_ATTRS: frozenset[str] = frozenset(
    {"tool", "tokens_in", "tokens_out", "cost", "error_kind", "approval_id"}
)
TURN_ATTRS: frozenset[str] = SPAN_ATTRS | {"tools", "span_count"}

APPROVAL_ID_RE = re.compile(r"^[0-9a-f]{32}$")
THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_COUNTER_RE = re.compile(r"^(?P<name>[a-z_]+)\{(?P<labels>[^}]*)\}\s+(?P<value>[-+0-9.eE]+)$")
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')
_FORBIDDEN_LABEL_RE = re.compile(r"(?:session_id|thread_id|approval_id)\s*=")

#: 深调研（S2）最长等 10 分钟（工具 timeout 600s）；普通会话也共用同一读超时
READ_TIMEOUT = 900.0

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# 运行期配置（`configure()` 写入）
_API = API_DEFAULT
_PSQL_PREFIX: list[str] = []
_REDIS_PREFIX: list[str] = []


# ── 配置与前置 ───────────────────────────────────────────────────────


def configure(api: str | None = None) -> None:
    """解析 `--api` / `E2E_*` 覆盖，构造 psql / redis-cli 命令前缀。"""
    global _API, _PSQL_PREFIX, _REDIS_PREFIX

    _API = (api or os.environ.get("E2E_API") or API_DEFAULT).rstrip("/")

    explicit_psql = os.environ.get("E2E_PSQL")
    if explicit_psql:
        _PSQL_PREFIX = shlex.split(explicit_psql)
    else:
        _PSQL_PREFIX = [
            "docker",
            "exec",
            "-i",
            os.environ.get("E2E_PG_CONTAINER", PG_CONTAINER_DEFAULT),
            "psql",
            "-U",
            os.environ.get("E2E_PG_USER", PG_USER_DEFAULT),
            "-d",
            os.environ.get("E2E_PG_DB", PG_DB_DEFAULT),
            "-tA",
        ]

    explicit_redis = os.environ.get("E2E_REDIS")
    if explicit_redis:
        _REDIS_PREFIX = shlex.split(explicit_redis)
    else:
        _REDIS_PREFIX = [
            "docker",
            "exec",
            os.environ.get("E2E_REDIS_CONTAINER", REDIS_CONTAINER_DEFAULT),
            "redis-cli",
            "--no-auth-warning",
        ]
        password = _redis_password()
        if password:
            _REDIS_PREFIX += ["-a", password]


def _redis_password() -> str:
    """读 Redis 密码：环境变量优先，其次仓库 `.env`（不打印到任何输出）。"""
    for key in ("E2E_REDIS_PASSWORD", "REDIS_PASSWORD"):
        value = os.environ.get(key)
        if value:
            return value
    env_path = REPO / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        key, sep, value = line.partition("=")
        if not sep:
            continue
        if key.strip() == "REDIS_PASSWORD":
            return value.strip().strip('"').strip("'")
    for line in lines:
        if line.startswith("REDIS_URL="):
            match = re.match(r"redis://:([^@]+)@", line[len("REDIS_URL=") :].strip())
            if match:
                return match.group(1)
    return ""


def _run(cmd: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    """执行外部命令（不经过 shell），返回 CompletedProcess。"""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def probe() -> list[str]:
    """前置检查；返回可读错误列表（空 = 全部就绪）。"""
    errors: list[str] = []

    try:
        resp = httpx.get(f"{_API}/health", timeout=5.0)
        if resp.status_code != 200:
            errors.append(
                f"后端未就绪：GET {_API}/health -> HTTP {resp.status_code}\n"
                f"   起服务（仓库根目录）：ENABLE_TOOL_POLICY=true ENABLE_HITL=true "
                f"ENABLE_TRACE=true poetry run uvicorn api_main:app --port 8200"
            )
    except Exception as exc:  # noqa: BLE001 - 前置检查要把所有失败原因收齐
        errors.append(
            f"后端不可达：GET {_API}/health -> {type(exc).__name__}: {exc}\n"
            f"   确认端口/地址（--api 或 E2E_API），并先起服务："
            f"poetry run uvicorn api_main:app --port 8200"
        )

    try:
        out = _psql("SELECT 1")
        if "1" not in out:
            errors.append(
                f"PostgreSQL 不可达：psql 返回 {out!r}\n"
                f"   起容器：docker compose up -d db_dev redis；"
                f"或设 E2E_PSQL / E2E_PG_CONTAINER 指向正确容器"
                f"（E2E_PSQL 覆盖时 docker exec 需带 -i，SQL 走 stdin）"
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"PostgreSQL 检查失败：{type(exc).__name__}: {exc}\n"
            f"   psql 前缀：{' '.join(_PSQL_PREFIX)}（可用 E2E_PSQL 覆盖）"
        )

    try:
        proc = _run([*_REDIS_PREFIX, "PING"])
        if proc.returncode != 0 or proc.stdout.strip() != "PONG":
            errors.append(
                f"Redis 不可达：redis-cli PING -> {proc.stdout.strip() or proc.stderr.strip()[:200]}\n"
                f"   起容器：docker compose up -d redis；"
                f"或设 E2E_REDIS / E2E_REDIS_CONTAINER / E2E_REDIS_PASSWORD"
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Redis 检查失败：{type(exc).__name__}: {exc}")

    return errors


# ── 指标 ─────────────────────────────────────────────────────────────


def _parse_metrics(text: str) -> dict[str, Any]:
    """把 Prometheus 文本解析成我们需要的三族计数。"""
    snapshot: dict[str, Any] = {
        "requests_total": 0.0,
        "requests": {},
        "tool_decisions": {},
        "approvals": {},
    }
    for line in text.splitlines():
        match = _COUNTER_RE.match(line.strip())
        if not match:
            continue
        name = match.group("name")
        value = float(match.group("value"))
        labels = tuple(sorted((k, v) for k, v in _LABEL_RE.findall(match.group("labels"))))
        if name == "agent_requests_total":
            snapshot["requests_total"] += value
            snapshot["requests"][labels] = value
        elif name == "agent_tool_decisions_total":
            snapshot["tool_decisions"][labels] = value
        elif name == "agent_approvals_total":
            snapshot["approvals"][labels] = value
    return snapshot


def metrics_text() -> str:
    """取 `/metrics` 原文；未启用时抛可读异常（让场景判 FAIL 而不是假装通过）。"""
    resp = httpx.get(f"{_API}/metrics", timeout=10.0)
    if resp.status_code != 200:
        raise RuntimeError(
            f"GET /metrics -> HTTP {resp.status_code}（需要 ENABLE_METRICS=true，默认开启）"
        )
    return resp.text


def snapshot_metrics() -> dict[str, Any]:
    """解析 `/metrics`，取 requests / tool_decisions / approvals 计数快照。"""
    return _parse_metrics(metrics_text())


def _approval_count(snapshot: dict[str, Any], decision: str, outcome: str) -> float:
    """取 `agent_approvals_total{decision=..., outcome=...}` 的计数（缺失算 0）。"""
    for labels, value in snapshot["approvals"].items():
        mapping = dict(labels)
        if mapping.get("decision") == decision and mapping.get("outcome") == outcome:
            return float(value)
    return 0.0


# ── PostgreSQL ───────────────────────────────────────────────────────


def _require_thread_id(thread_id: str) -> str:
    """thread_id 只允许安全字符（SQL 里用 psql 变量绑定，仍先做白名单校验）。"""
    if not THREAD_ID_RE.match(thread_id):
        raise ValueError(f"非法 thread_id：{thread_id!r}")
    return thread_id


def _psql(sql: str, variables: dict[str, str] | None = None) -> str:
    """执行一条 psql 查询（数值/状态取 -tA 的裸输出；SQL 走 stdin 以支持 `:'var'`）。"""
    cmd = [*_PSQL_PREFIX]
    for key, value in (variables or {}).items():
        cmd += ["-v", f"{key}={value}"]
    proc = subprocess.run(cmd, input=sql, capture_output=True, text=True, timeout=20.0, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"psql 失败（{proc.returncode}）：{proc.stderr.strip()[:200]}")
    return proc.stdout.strip()


def query_state(thread_id: str) -> str | None:
    """查 `task_states.state`；没有记录返回 None。"""
    out = _psql(
        "SELECT state FROM task_states WHERE thread_id = :'tid'",
        {"tid": _require_thread_id(thread_id)},
    )
    return out or None


def query_checkpoint_count(thread_id: str) -> int:
    """查该 thread 在 LangGraph `checkpoints` 表里的行数。"""
    out = _psql(
        "SELECT count(*) FROM checkpoints WHERE thread_id = :'tid'",
        {"tid": _require_thread_id(thread_id)},
    )
    return int(out or 0)


# ── Redis ────────────────────────────────────────────────────────────


def redis_scan(pattern: str) -> list[str]:
    """`redis-cli --scan --pattern`，返回命中的 key 列表（不可用时返回空）。"""
    proc = _run([*_REDIS_PREFIX, "--scan", "--pattern", pattern], timeout=20.0)
    if proc.returncode != 0:
        raise RuntimeError(f"redis-cli --scan 失败：{proc.stderr.strip()[:200]}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def redis_get(key: str) -> str | None:
    """`redis-cli GET <key>`；key 不存在（空输出）返回 None。"""
    proc = _run([*_REDIS_PREFIX, "GET", key], timeout=20.0)
    if proc.returncode != 0:
        raise RuntimeError(f"redis-cli GET 失败：{proc.stderr.strip()[:200]}")
    return proc.stdout.strip() or None


# ── trace ────────────────────────────────────────────────────────────


def _trace_path() -> Path:
    """今天（本地日期）的 trace 文件路径，与 `TRACE_DIR/<YYYY-MM-DD>.jsonl` 一致。"""
    return TRACE_DIR / f"{datetime.now().astimezone():%Y-%m-%d}.jsonl"


def latest_trace_spans(since_ts: datetime) -> list[dict[str, Any]]:
    """读今天的 trace，只取 `start >= since_ts` 的行；坏行跳过。"""
    path = _trace_path()
    if not path.exists():
        return []
    spans: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                span = json.loads(line)
                start = datetime.fromisoformat(str(span.get("start")))
            except (ValueError, TypeError):
                continue
            if start >= since_ts:
                spans.append(span)
    return spans


# ── HTTP / SSE ───────────────────────────────────────────────────────


def _client() -> httpx.Client:
    """建一个读超时较长的客户端（深调研可能跑几分钟）。"""
    return httpx.Client(timeout=httpx.Timeout(READ_TIMEOUT, connect=5.0))


def _parse_frame(line: str) -> dict[str, Any] | None:
    """把一行 SSE 解成 dict；非 `data:` 行 / 坏 JSON 返回 None。"""
    payload = line.strip()
    if not payload.startswith("data: "):
        return None
    try:
        event = json.loads(payload[len("data: ") :])
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _read_sse(response: httpx.Response) -> list[dict[str, Any]]:
    """把一条 SSE 响应读完（读尽 = 服务端生成器跑完，状态机才落库）。"""
    frames: list[dict[str, Any]] = []
    for line in response.iter_lines():
        frame = _parse_frame(line)
        if frame is not None:
            frames.append(frame)
    return frames


def post_chat(client: httpx.Client, thread_id: str, message: str) -> list[dict[str, Any]]:
    """POST /chat/conversation 并读完整个 SSE 流。"""
    with client.stream(
        "POST",
        f"{_API}/chat/conversation",
        json={"thread_id": thread_id, "message": message},
    ) as response:
        if response.status_code != 200:
            body = response.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"POST /chat/conversation -> HTTP {response.status_code}：{body[:200]}"
            )
        return _read_sse(response)


def post_resume(
    client: httpx.Client,
    thread_id: str,
    approval_id: str,
    decision: str,
    request_id: str,
) -> tuple[int, list[dict[str, Any]], str]:
    """POST resume；返回 (状态码, SSE 帧, 非 200 时的响应体)。"""
    with client.stream(
        "POST",
        f"{_API}/chat/conversation/{thread_id}/resume",
        json={"approval_id": approval_id, "decision": decision, "request_id": request_id},
    ) as response:
        if response.status_code != 200:
            body = response.read().decode("utf-8", "replace")
            return response.status_code, [], body
        return 200, _read_sse(response), ""


# ── 场景 ─────────────────────────────────────────────────────────────


def _frame_types(frames: list[dict[str, Any]]) -> set[str]:
    return {str(frame.get("type")) for frame in frames}


def scenario_s1(ctx: dict[str, Any], run_start: datetime) -> tuple[str, str]:
    """S1 普通会话：帧集合 / final / state / checkpoints / requests +1。"""
    del run_start
    thread_id = ctx["s1_thread"]
    before = snapshot_metrics()

    frames = post_chat(ctx["client"], thread_id, "PostgreSQL 的写入吞吐量是多少？")
    types = _frame_types(frames)
    problems: list[str] = []

    illegal = types - CHAT_FRAME_TYPES
    if illegal:
        problems.append(f"非法帧类型={sorted(illegal)}")
    if "final" not in types:
        problems.append("缺 final 帧")
    if "interrupt" in types:
        problems.append("出现 interrupt 帧")

    state = query_state(thread_id)
    if state != "completed":
        problems.append(f"task_states.state={state}")

    checkpoints = query_checkpoint_count(thread_id)
    if checkpoints <= 0:
        problems.append("checkpoints 行数为 0")

    after = snapshot_metrics()
    delta = after["requests_total"] - before["requests_total"]
    if delta != 1:
        problems.append(f"agent_requests_total 增量={delta:+g}")

    if problems:
        return FAIL, "；".join(problems)
    return PASS, (
        f"{len(frames)} 帧（{','.join(sorted(types))}）；final 有；state=completed；"
        f"checkpoints={checkpoints}；requests +1"
    )


def scenario_s2(  # noqa: C901 - 八条断言按 (a)-(h) 顺序展开，拆函数反而看不出契约
    ctx: dict[str, Any], run_start: datetime
) -> tuple[str, str]:
    """S2 审批全链路：八字段 -> Redis -> interrupted -> resume -> 消费 -> completed -> trace -> metrics。"""
    thread_id = ctx["s2_thread"]
    before = snapshot_metrics()

    frames = post_chat(
        ctx["client"],
        thread_id,
        "请调用 deep_research 工具，对「PostgreSQL 与 MySQL 的写入吞吐量对比」做一次深调研",
    )
    types = _frame_types(frames)
    interrupts = [frame for frame in frames if frame.get("type") == "interrupt"]
    if not interrupts:
        return FAIL, (
            f"未收到 interrupt 帧（帧类型={sorted(types)}）；"
            f"需 ENABLE_TOOL_POLICY=true + ENABLE_HITL=true"
        )

    data = interrupts[0].get("data") or {}
    if not isinstance(data, dict):
        return FAIL, f"interrupt.data 不是 dict：{type(data).__name__}"

    problems: list[str] = []
    if set(data) != set(APPROVAL_FIELDS):
        problems.append(f"data 字段集={sorted(data)}（契约八字段）")

    approval_id = str(data.get("approval_id") or "")
    if not APPROVAL_ID_RE.match(approval_id):
        return FAIL, f"approval_id 不是 32 位 hex：{approval_id!r}"

    ctx["approval_id"] = approval_id
    key = f"approval:{approval_id}"
    scanned = redis_scan(key)
    payload = redis_get(key)
    if key not in scanned:
        problems.append("Redis 无 approval:<id>（scan 未命中）")

    state_before = query_state(thread_id)
    if state_before != "interrupted":
        problems.append(f"挂起时 state={state_before}")

    status, resume_frames, body = post_resume(
        ctx["client"], thread_id, approval_id, "approve", str(uuid.uuid4())
    )
    resume_types = _frame_types(resume_frames)
    if status != 200:
        problems.append(f"resume HTTP {status}：{body[:120]}")
    if "final" not in resume_types:
        problems.append("resume 流里没有 final")

    state_after = query_state(thread_id)
    if state_after != "completed":
        problems.append(f"resume 后 state={state_after}")

    if redis_get(key) is not None:
        problems.append("resume 后 approval:<id> 仍存在（凭据未消费）")

    tool_name = str(data.get("tool") or "")
    spans = latest_trace_spans(run_start)
    tool_span_ok = any(
        span.get("session_id") == thread_id
        and span.get("kind") == "tool"
        and span.get("name") == tool_name
        and (span.get("attrs") or {}).get("error_kind") == "approval_required"
        for span in spans
    )
    if not tool_span_ok:
        problems.append(f"trace 缺 {tool_name} 的 error_kind=approval_required")

    after = snapshot_metrics()
    delta = _approval_count(after, "approve", "approved") - _approval_count(
        before, "approve", "approved"
    )
    if delta != 1:
        problems.append(f"agent_approvals_total approve +{delta:g}")

    if problems:
        return FAIL, "；".join(problems)

    finals = [frame for frame in resume_frames if frame.get("type") == "final"]
    final_len = len(str(finals[-1].get("content") or "")) if finals else 0
    bound = thread_id in (payload or "")
    return PASS, (
        f"八字段精确；Redis 命中且绑定 thread={bound}；state=interrupted→completed；"
        f"resume 200 + final（{final_len} 字）；凭据已消费；"
        f"tool span error_kind=approval_required；approvals approve +1"
    )


def scenario_s3(ctx: dict[str, Any], run_start: datetime) -> tuple[str, str]:
    """S3 重放：同一 approval_id 再 resume -> 409，且凭据仍不存在。"""
    del run_start
    approval_id = ctx.get("approval_id")
    if not approval_id:
        return SKIP, "S2 未产出 approval_id（无中断帧），重放场景无从构造"

    thread_id = ctx["s2_thread"]
    status, _, body = post_resume(
        ctx["client"], thread_id, approval_id, "approve", str(uuid.uuid4())
    )
    leftover = redis_get(f"approval:{approval_id}")
    problems: list[str] = []
    if status != 409:
        problems.append(f"重放 HTTP {status}（期望 409）")
    if leftover is not None:
        problems.append("重放后凭据仍存在")
    if problems:
        return FAIL, "；".join(problems)
    return PASS, (
        f"409 重放（{body.strip()[:60]}）；approval:<id> 仍不存在；"
        f"410 过期场景 SKIP（需把 TTL 调短并重启服务，本卡不测）"
    )


def scenario_s4(ctx: dict[str, Any], run_start: datetime) -> tuple[str, str]:
    """S4 技能（模型依赖）：trace 的 tool 名集合含 list_skills + load_skill。"""
    thread_id = ctx["s4_thread"]
    frames = post_chat(ctx["client"], thread_id, "你能用哪些技能？挑一个和引用来源有关的读给我看")
    types = _frame_types(frames)
    spans = latest_trace_spans(run_start)
    names = [
        str(span.get("name"))
        for span in spans
        if span.get("session_id") == thread_id and span.get("kind") == "tool"
    ]
    if not names:
        return SKIP, (
            f"模型未触发工具调用（帧类型={sorted(types)}）；"
            f"若 ENABLE_SKILLS 未开启，技能工具本就不在注册表"
        )
    missing = {"list_skills", "load_skill"} - set(names)
    if missing:
        return FAIL, f"tool spans={sorted(set(names))}，缺 {sorted(missing)}"
    return PASS, f"tool spans={sorted(set(names))}"


def scenario_s5(ctx: dict[str, Any], run_start: datetime) -> tuple[str, str]:
    """S5 指标与 trace 自洽：标签基数、kind 枚举、父子可达、turn attrs 白名单。"""
    del ctx
    text = metrics_text()
    forbidden = len(_FORBIDDEN_LABEL_RE.findall(text))
    spans = latest_trace_spans(run_start)
    problems: list[str] = []

    if forbidden:
        problems.append(f"/metrics 出现 {forbidden} 处禁用标签")

    if not spans:
        problems.append("本次无 trace 行（ENABLE_TRACE 是否开启？）")

    span_ids = {str(span.get("span_id")) for span in spans}
    bad_kind = sorted({str(span.get("kind")) for span in spans} - SPAN_KINDS)
    if bad_kind:
        problems.append(f"非法 kind={bad_kind}")

    bad_parent = [
        span
        for span in spans
        if span.get("parent_span_id") and str(span["parent_span_id"]) not in span_ids
    ]
    if bad_parent:
        problems.append(f"{len(bad_parent)} 条 span 的 parent_span_id 不可达")

    bad_turn = [
        span
        for span in spans
        if span.get("kind") == "turn" and not set(span.get("attrs") or {}) <= TURN_ATTRS
    ]
    if bad_turn:
        problems.append(f"{len(bad_turn)} 条 turn span 的 attrs 越界")

    if problems:
        return FAIL, "；".join(problems)
    kinds = sorted({str(span.get("kind")) for span in spans})
    return PASS, (
        f"0 处禁用标签；{len(spans)} 条 span（kind={','.join(kinds)}；父子可达；turn attrs 合规）"
    )


# ── 输出 ─────────────────────────────────────────────────────────────


def _display_width(text: str) -> int:
    """按终端显示宽度算长度（CJK 算 2）。"""
    return sum(2 if east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度左对齐补空格。"""
    return text + " " * max(width - _display_width(text), 0)


def _wrap(text: str, width: int) -> list[str]:
    """按显示宽度折行（优先在「；」处断句）。"""
    lines: list[str] = []
    current = ""
    for segment in text.split("；"):
        piece = segment + "；"
        while _display_width(piece) > width:
            if current:
                lines.append(current)
                current = ""
            cut = 0
            while cut < len(piece) and _display_width(piece[: cut + 1]) <= width:
                cut += 1
            lines.append(piece[:cut])
            piece = piece[cut:]
        if _display_width(current + piece) > width:
            lines.append(current)
            current = piece
        else:
            current += piece
    if current:
        lines.append(current)
    return lines or [""]


def print_table(rows: list[tuple[str, str, str]]) -> None:
    """打印 `场景 | 结论 | 关键证据` 表格（多行证据对齐）。"""
    scenario_width = max(_display_width("场景"), *(_display_width(row[0]) for row in rows))
    verdict_width = max(_display_width("结论"), *(_display_width(row[1]) for row in rows))
    evidence_width = 88

    header = f"{_pad('场景', scenario_width)} | {_pad('结论', verdict_width)} | 关键证据"
    divider = f"{'-' * scenario_width}-+-{'-' * verdict_width}-+-{'-' * evidence_width}"
    print(header)
    print(divider)
    for name, verdict, evidence in rows:
        chunks = _wrap(evidence, evidence_width)
        print(f"{_pad(name, scenario_width)} | {_pad(verdict, verdict_width)} | {chunks[0]}")
        for chunk in chunks[1:]:
            print(f"{_pad('', scenario_width)} | {_pad('', verdict_width)} | {chunk}")


def main(argv: list[str] | None = None) -> int:
    """跑完 5 个场景并返回退出码。"""
    parser = argparse.ArgumentParser(
        description="跨层 E2E：SSE / task_states / Redis / trace / metrics 五处一致",
    )
    parser.add_argument(
        "--api",
        default=None,
        help=f"后端地址（默认取 E2E_API，否则 {API_DEFAULT}）",
    )
    args = parser.parse_args(argv)
    configure(args.api)

    print(f"E2E 跨层校验 · API={_API}")

    errors = probe()
    if errors:
        print("\n前置不满足（退出码 2）：")
        for error in errors:
            print(f"  - {error}")
        return 2

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    ctx: dict[str, Any] = {
        "client": _client(),
        "s1_thread": f"e2e-norm-{stamp}",
        "s2_thread": f"e2e-hitl-{stamp}",
        "s4_thread": f"e2e-skill-{stamp}",
    }
    print(f"线程：S1={ctx['s1_thread']}  S2={ctx['s2_thread']}  S4={ctx['s4_thread']}")

    run_start = datetime.now().astimezone()
    rows: list[tuple[str, str, str]] = []
    scenarios = (
        ("S1 普通会话", scenario_s1),
        ("S2 审批全链路", scenario_s2),
        ("S3 重放与过期", scenario_s3),
        ("S4 技能（模型依赖）", scenario_s4),
        ("S5 指标与 trace 自洽", scenario_s5),
    )
    try:
        for index, (name, scenario) in enumerate(scenarios):
            started = time.monotonic()
            try:
                verdict, evidence = scenario(ctx, run_start)
            except Exception as exc:  # noqa: BLE001 - 单场景异常不拖垮整轮
                verdict, evidence = FAIL, f"异常：{type(exc).__name__}: {exc}"
            rows.append((name, verdict, evidence))
            print(
                f"[{index + 1}/{len(scenarios)}] {name} -> {verdict} "
                f"({time.monotonic() - started:.1f}s)"
            )
    finally:
        ctx["client"].close()

    passed = sum(1 for _, verdict, _ in rows if verdict == PASS)
    failed = sum(1 for _, verdict, _ in rows if verdict == FAIL)
    skipped = sum(1 for _, verdict, _ in rows if verdict == SKIP)

    print()
    print_table(rows)
    print()
    print(f"E2E: {passed} passed, {failed} failed, {skipped} skipped")
    print(
        "复跑：poetry run python scripts/e2e_flow.py"
        "（换环境：--api http://host:port 或 E2E_API=...）"
    )

    if failed:
        return 1
    if skipped:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
