"""可观测性指标（B 卡：指标 + 告警）。

只做埋点与暴露，不参与任何业务判定；调用方一律通过本模块的薄封装上报。

三条硬约束（改动前先读）：

- **C1 标签基数纪律**：标签值只能是枚举值或工具名这类有限集合。
  **禁止**把 ``thread_id`` / ``session_id`` / ``approval_id`` / 用户输入放进标签——
  每个不同取值都会生成一条新时间序列，是 Prometheus 基数爆炸的头号原因。
  会话级信息走 span（trace），不进 metrics。
- **C2 开关关闭时零副作用**：``ENABLE_METRICS=False`` 时下面每个薄封装都直接
  ``return``，不累加、不观测；``/metrics`` 端点返回 404。指标对象本身在进程内
  注册（常量体积），但不产生任何观测数据。
- **C3 热路径轻量**：埋点只做 ``inc()`` / ``observe()``，不做序列化、不打日志、
  不碰网络与文件。

多进程注意：这里用的是 prometheus_client 的默认 ``REGISTRY``。uvicorn
``--workers > 1`` 时每个 worker 是独立进程，各自注册同名指标会互相覆盖、
聚合结果也不完整；多 worker 部署需改用 ``prometheus_client.multiprocess``
模式（``PROMETHEUS_MULTIPROC_DIR`` + ``MultiProcessCollector``）。本卡只覆盖
单进程部署。
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)

from config import settings

# ── 指标定义（标签集合保持有限：endpoint / outcome / tool / decision / kind / direction）──

REQUESTS: Counter = Counter(
    "agent_requests_total",
    "Agent 请求数（endpoint=conversation；outcome=completed/interrupted/failed/rejected）",
    ("endpoint", "outcome"),
)
TOOL_DECISIONS: Counter = Counter(
    "agent_tool_decisions_total",
    "工具策略判定数（decision=allow/deny/require_approval/rejected_by_user）",
    ("tool", "decision"),
)
APPROVALS: Counter = Counter(
    "agent_approvals_total",
    "人工审批结果数（outcome=approved/rejected/expired/replayed/conflict）",
    ("decision", "outcome"),
)
ERRORS: Counter = Counter(
    "agent_errors_total",
    "Agent 错误数（kind=transient/llm_recoverable/user_fixable/fatal）",
    ("kind",),
)
TOKENS: Counter = Counter(
    "agent_tokens_total",
    "累计 token 数（direction=in/out）",
    ("direction",),
)
COST: Counter = Counter(
    "agent_cost_yuan_total",
    "累计费用（元）；按 turn 增量累加",
)
REQUEST_DURATION: Histogram = Histogram(
    "agent_request_duration_seconds",
    "Agent 请求耗时（秒）",
    ("endpoint",),
    buckets=(1, 5, 15, 30, 60, 120, 300),
)


# ── 薄封装：每个都先判开关，关则直接 return（C2 / C3）──


def observe_request(endpoint: str, outcome: str, seconds: float) -> None:
    """记一次请求的结果与耗时；``endpoint`` / ``outcome`` 必须是有限枚举。"""
    if not settings.ENABLE_METRICS:
        return
    REQUESTS.labels(endpoint=endpoint, outcome=outcome).inc()
    REQUEST_DURATION.labels(endpoint=endpoint).observe(seconds)


def observe_tool_decision(tool: str, decision: str) -> None:
    """记一次工具策略判定；``tool`` 是注册表里的工具名（有限集合）。"""
    if not settings.ENABLE_METRICS:
        return
    TOOL_DECISIONS.labels(tool=tool, decision=decision).inc()


def observe_approval(decision: str, outcome: str) -> None:
    """记一次人工审批结果；``decision``=approve/reject，``outcome`` 见指标说明。"""
    if not settings.ENABLE_METRICS:
        return
    APPROVALS.labels(decision=decision, outcome=outcome).inc()


def observe_usage(tokens_in: int, tokens_out: int, cost: float) -> None:
    """记一轮的 token 与费用增量；取不到的字段由调用方补 0。"""
    if not settings.ENABLE_METRICS:
        return
    TOKENS.labels(direction="in").inc(tokens_in)
    TOKENS.labels(direction="out").inc(tokens_out)
    if cost:
        COST.inc(cost)


def observe_error(kind: str) -> None:
    """记一次错误；``kind`` 用四类错误枚举（transient/llm_recoverable/user_fixable/fatal）。"""
    if not settings.ENABLE_METRICS:
        return
    ERRORS.labels(kind=kind).inc()


def render() -> tuple[bytes, str]:
    """返回 Prometheus 文本格式的指标快照与 Content-Type。"""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


__all__ = [
    "APPROVALS",
    "COST",
    "ERRORS",
    "REQUEST_DURATION",
    "REQUESTS",
    "TOKENS",
    "TOOL_DECISIONS",
    "observe_approval",
    "observe_error",
    "observe_request",
    "observe_tool_decision",
    "observe_usage",
    "render",
]
