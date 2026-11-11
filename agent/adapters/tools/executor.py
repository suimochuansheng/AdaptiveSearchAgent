"""工具执行器（T6-3）：策略判定 + 超时 + 异常归一 + 记账的唯一入口。

`wrap_tool(fn, spec, ledger=None)` 返回的函数与 `fn` **签名 / schema 完全一致**：

- `functools.wraps` 让 `inspect.signature` 跟随 `__wrapped__`，`__name__` / `__doc__`
  也被复制，所以 `convert_to_openai_tool` 取到的 schema 与包装前逐字节一致；
- 包装函数恒为 `async def`（5 个内置工具都是协程函数），`await` 语义不变；
- **开关 `ENABLE_TOOL_POLICY` 关闭时原样返回 `fn`（同一个对象）**：零包装、零 IO、
  零 contextvar，代码路径与改造前一致（schema 哈希仍是基线 `c48c9d48…`）。

作用域（scope）通过 contextvar 传入：graph 的 `tools_node` 用
`with tool_scope(thread_id):` 包住 ToolNode 调用。拿不到 scope 时兜底
`"unknown"` 并只 warning 一次——宁可记账记错地方，也不能让工具调用崩掉。

执行口径：

| 决策 | 行为 | 记账 |
|---|---|---|
| `allow` | `asyncio.wait_for(fn(...), spec.timeout_seconds)` | 记 1 次 + 本次成本 + 估算 token |
| 超时 | 返回 `工具执行超时（>Xs），已终止`，不抛栈 | **照记 1 次**（钱已经花了） |
| 异常 | 归一为可读错误字符串，不抛栈 | **照记 1 次** |
| `deny` | 直接返回可读理由，**不调用 fn** | 不记 |
| `require_approval`（HITL 关） | 返回“需人工审批…本次已拒绝”，**不调用 fn** | 不记 |
| `require_approval`（HITL 开） | 签发 `ApprovalRequest` -> `interrupt()` 挂起；恢复后 approve 执行原工具 / reject 返回可读理由 | 仅 approve 执行后记 |

`require_approval` 的挂起由 G7/T9-2 接入：`ENABLE_HITL=False` 时保持 G5 原行为（同一个
拒绝字符串，零 IO）；`ENABLE_HITL=True` 且工具策略开启时，在工具函数内部调用
`langgraph.types.interrupt(审批载荷)`。LangGraph 的 `ToolNode` 会重新抛出
`GraphInterrupt`（见 `tool_node.py` 对 `GraphBubbleUp` 的特判），因此可以冒泡成中断。

**结果码**：拒绝 / 审批两条路径都通过 `agent.observability.tool_outcome` 暴露本次调用的
结果码（`approval_required` / `denied_timeout` / `denied_calls` / `denied_cost` /
`rejected_by_user`），由 LangChain handler 在工具结束时读进 `span.attrs.error_kind`。
executor 不直接调 tracer（分层：执行器只管执行，追踪是旁路）。

超时收尾：`asyncio.wait_for` 内部会取消并 await 目标任务；这里额外
`cancel()` + `gather(..., return_exceptions=True)` 一次，确保 Python 3.11 下
不留下 “Task was destroyed but it is pending” 的泄漏告警。
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import logging
from contextlib import contextmanager, suppress
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from langgraph.types import interrupt

from agent.domain.budget import BudgetLedger, call_cost
from agent.domain.tool_policy import PolicyContext, decide
from agent.observability.tool_outcome import (
    clear_tool_outcome,
    mark_outcome,
    remember_outcome,
    set_tool_outcome,
)
from config import settings
from src.context_manager import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from agent.domain.budget import BudgetSnapshot
    from agent.domain.roles import ApprovalRequest
    from agent.domain.tool import ToolSpec

logger = logging.getLogger(__name__)

_UNKNOWN_SCOPE = "unknown"
_REJECTED_MESSAGE = "用户拒绝执行该工具，本次未调用。"
_APPROVAL_UNAVAILABLE_MESSAGE = "审批服务不可用，已拒绝执行该工具（fail-closed）。"

_scope_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("tool_scope", default=None)
#: 本次图推进是否是 `Command(resume=...)` 触发的恢复（由 resume 路由设置）
_resume_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tool_approval_resume", default=False
)
_scope_warned = False
_default_ledger: BudgetLedger | None = None

__all__ = [
    "approval_resume_scope",
    "current_scope",
    "get_default_ledger",
    "set_default_ledger",
    "set_scope",
    "tool_scope",
    "wrap_tool",
]


# ── 作用域（contextvar） ──────────────────────────────────────────────


def set_scope(scope: str) -> contextvars.Token[str | None]:
    """显式设置当前作用域；调用方负责在结束时 `_scope_var.reset(token)`。"""
    return _scope_var.set(scope or _UNKNOWN_SCOPE)


def current_scope() -> str:
    """读当前作用域；没有时兜底 `"unknown"` 并只 warning 一次（不崩）。"""
    scope = _scope_var.get()
    if scope:
        return scope
    global _scope_warned
    if not _scope_warned:
        _scope_warned = True
        logger.warning("工具作用域未设置，预算记账退化为 scope=%s", _UNKNOWN_SCOPE)
    return _UNKNOWN_SCOPE


@contextmanager
def tool_scope(scope: str) -> Iterator[None]:
    """在 `with` 块内把作用域设为 `scope`。

    **开关关闭时什么都不设置**（不碰 contextvar、不产生 token），
    这样 graph 里那一行 `with` 在默认配置下是纯空转，不会有任何行为变化。
    """
    if not settings.ENABLE_TOOL_POLICY:
        yield
        return
    token = _scope_var.set(scope or _UNKNOWN_SCOPE)
    try:
        yield
    finally:
        _scope_var.reset(token)


@contextmanager
def approval_resume_scope() -> Iterator[None]:
    """在 `with` 块内标记“本次推进是 resume”。

    恢复时 LangGraph 会**从头重跑**被中断的工具函数；执行器据此跳过“再次签发凭据”，
    只调用 `interrupt()` 取回 resume 值（否则每恢复一次就多签一张没人用的凭据）。
    contextvar 由 resume 路由在驱动 `astream(Command(resume=...))` 前设置。
    """
    token = _resume_var.set(True)
    try:
        yield
    finally:
        _resume_var.reset(token)


def _args_digest(kwargs: dict[str, Any]) -> str:
    """参数摘要：`query=…&max=5` 形态，供人工核对“批的是什么”。

    只做稳定序列化，不解析参数语义；超长截断，避免把大对象塞进审批卡片。
    """
    parts = [f"{key}={kwargs[key]}" for key in sorted(kwargs)]
    digest = "&".join(parts)
    return digest[:300]


def _current_tool_call_id(tool_name: str, kwargs: dict[str, Any]) -> str:
    """尽力取当前 tool_call 的 id；取不到返回空串（字段仍存在，不影响续跑）。

    LangGraph 不通过工具参数暴露 tool_call_id，但当前 task 的 config 里有
    `__pregel_read`（读本 superstep 的通道值）。读最新一条 AIMessage 的 tool_calls，
    按“工具名 + 参数”匹配；这是内部接口，任何异常都静默兜底（id 只用于审计）。
    """
    try:
        from langgraph.config import get_config

        configurable = (get_config() or {}).get("configurable") or {}
        read = configurable.get("__pregel_read")
        if not callable(read):
            return ""
        messages = read("messages") or []
        for message in reversed(messages):
            for call in reversed(getattr(message, "tool_calls", None) or []):
                if call.get("name") == tool_name and dict(call.get("args") or {}) == dict(kwargs):
                    return str(call.get("id") or "")
    except Exception:  # noqa: BLE001 - 审批 id 只用于审计，取不到不能影响执行
        return ""
    return ""


# ── 账本（默认实现惰性创建，避免默认关闭时也去碰 Redis） ─────────────


def get_default_ledger() -> BudgetLedger:
    """默认账本：Redis 版；它自己处理“连不上就降级 Noop”，这里不兜异常。"""
    global _default_ledger
    if _default_ledger is None:
        from agent.adapters.tools.budget_redis import RedisBudgetLedger

        _default_ledger = RedisBudgetLedger()
    return _default_ledger


def set_default_ledger(ledger: BudgetLedger | None) -> None:
    """替换默认账本（组合根 / 测试注入用）；传 `None` 恢复惰性创建。"""
    global _default_ledger
    _default_ledger = ledger


# ── 包装 ─────────────────────────────────────────────────────────────


def wrap_tool(
    fn: Callable[..., Any],
    spec: ToolSpec,
    ledger: BudgetLedger | None = None,
) -> Callable[..., Any]:
    """把工具包装成“策略前置 + 超时 + 记账”的可调用体。

    Args:
        fn: 工具原本的可调用体（协程函数）。
        spec: 该工具的元数据（超时、成本、是否需要审批）。
        ledger: 账本；`None` 时用进程级默认账本。

    Returns:
        开关关闭时是 `fn` 本身（`is` 同一对象）；开启时是异步包装函数，
        签名与 `fn` 一致。
    """
    if not settings.ENABLE_TOOL_POLICY:
        return fn

    original_signature = inspect.signature(fn)

    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        scope = current_scope()
        active = ledger if ledger is not None else get_default_ledger()
        used: BudgetSnapshot = await _maybe_await(active.snapshot(scope))
        ctx = PolicyContext(
            scope=scope,
            max_calls=settings.TOOL_MAX_CALLS_PER_TURN,
            cost_limit=settings.TOOL_SESSION_COST_LIMIT,
            args=kwargs,
        )

        decision = decide(spec, ctx, used)
        if decision.kind == "require_approval":
            if not settings.ENABLE_HITL:
                # G5 原行为：返回可读拒绝理由，不调用 fn、不记账；只补结果码给追踪
                logger.info(
                    "工具调用被策略拦截：%s -> %s（%s）",
                    spec.name,
                    decision.kind,
                    decision.code,
                )
                return _mark_outcome(decision.reason, "approval_required", spec.name, kwargs)
            return await _run_with_approval(
                fn, spec, args, kwargs, active=active, scope=scope, used=used
            )

        if decision.kind != "allow":
            # deny：不调用 fn，也不记账
            logger.info(
                "工具调用被策略拦截：%s -> %s（%s）", spec.name, decision.kind, decision.code
            )
            return _mark_outcome(decision.reason, decision.code, spec.name, kwargs)

        return await _execute_and_record(fn, spec, args, kwargs, active=active, scope=scope)

    # 包装不能改变 schema：签名必须仍然跟随 `__wrapped__`
    if inspect.signature(wrapped) != original_signature:
        raise RuntimeError(f"工具包装后签名不一致：{spec.name}")
    return wrapped


def _mark_outcome(value: Any, code: str, tool_name: str, kwargs: dict[str, Any]) -> Any:
    """把结果码挂到返回值 + tool_call_id 索引上（回调两条路都能读到）。"""
    remember_outcome(_current_tool_call_id(tool_name, kwargs), code)
    return mark_outcome(value, code)


def _interrupt_with_outcome(payload: dict[str, Any]) -> Any:
    """调用 `interrupt()`；挂起时把 `approval_required` 挂到异常对象上。

    LangGraph 用 `GraphInterrupt` 冒泡，ToolNode 会重新抛出；LangChain 的 `on_tool_error`
    在回调线程里执行（contextvar 过不去），异常对象是同一个引用，所以把结果码挂在
    `exc.error_kind` 上，handler 能稳定读到（见 `tool_outcome.ToolOutcome` 的说明）。
    """
    set_tool_outcome("approval_required")
    try:
        return interrupt(payload)
    except BaseException as exc:
        # 挂不上也不能改变异常本身
        with suppress(Exception):
            exc.error_kind = "approval_required"  # type: ignore[attr-defined]
        raise


async def _run_with_approval(
    fn: Callable[..., Any],
    spec: ToolSpec,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    active: BudgetLedger,
    scope: str,
    used: BudgetSnapshot,
) -> Any:
    """HITL：签发凭据 -> `interrupt()` 挂起 -> 恢复后按决定执行或拒绝。

    - `approve`：继续执行原工具（走与 allow 相同的超时 / 记账路径）；
    - `reject`：返回可读理由字符串，**不抛异常**、不执行、不记账；
    - 凭据签发失败（Redis 不可用）：fail-closed，返回可读理由并拒绝执行。
    """
    if not _resume_var.get():
        from agent.adapters.stores import approval_store

        request: ApprovalRequest | None = await approval_store.create(
            thread_id=scope,
            tool=spec.name,
            args_digest=_args_digest(kwargs),
            cost_estimate=call_cost(spec),
            budget_remaining=max(0.0, settings.TOOL_SESSION_COST_LIMIT - used.cost),
            permission=spec.permission,
            tool_call_id=_current_tool_call_id(spec.name, kwargs),
        )
        if request is None:
            logger.warning("审批凭据签发失败，拒绝执行工具 %s（fail-closed）", spec.name)
            return _mark_outcome(
                _APPROVAL_UNAVAILABLE_MESSAGE, "approval_required", spec.name, kwargs
            )
        # 挂起期间先打标：恢复后若用户拒绝会改写为 rejected_by_user
        decision = _interrupt_with_outcome(asdict(request))
    else:
        # 恢复执行：凭据已在 resume 路由里原子消费，这里只取回 resume 值
        decision = _interrupt_with_outcome({"_resume": True})

    if decision != "approve":
        # 未知值一律按拒绝处理（契约：不得把未知 decision 当 approve）
        logger.info("用户拒绝执行工具 %s（decision=%r）", spec.name, decision)
        return _mark_outcome(_REJECTED_MESSAGE, "rejected_by_user", spec.name, kwargs)

    clear_tool_outcome()
    return await _execute_and_record(fn, spec, args, kwargs, active=active, scope=scope)


async def _execute_and_record(
    fn: Callable[..., Any],
    spec: ToolSpec,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    active: BudgetLedger,
    scope: str,
) -> Any:
    """执行原工具：超时 / 异常归一 + 记账；超时补 `denied_timeout` 结果码。"""
    try:
        result = await _run_with_timeout(fn, spec.timeout_seconds, args, kwargs)
    except TimeoutError:
        await _record(active, scope, spec)
        logger.warning("工具 %s 执行超时（>%ss）", spec.name, f"{spec.timeout_seconds:g}")
        return _mark_outcome(
            f"工具执行超时（>{spec.timeout_seconds:g}s），已终止",
            "denied_timeout",
            spec.name,
            kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - 工具异常归一为可读文本，不打断 Agent 循环
        await _record(active, scope, spec)
        logger.warning("工具 %s 执行失败：%s: %s", spec.name, type(exc).__name__, exc)
        return f"工具 {spec.name} 执行出错：{type(exc).__name__}: {exc}"

    clear_tool_outcome()
    await _record(active, scope, spec, result)
    return result


async def _run_with_timeout(
    fn: Callable[..., Any],
    timeout: float,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """执行 `fn` 并加超时；超时后取消任务并收尾，避免任务泄漏。"""
    task = asyncio.ensure_future(_invoke(fn, args, kwargs))
    try:
        return await asyncio.wait_for(task, timeout=timeout)
    except TimeoutError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


async def _invoke(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """兼容同步工具：同步返回值直接透传，可等待对象才 await。"""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def _maybe_await(value: Any) -> Any:
    """账本方法可能是同步（内存）或异步（Redis），统一收口。"""
    if inspect.isawaitable(value):
        return await value
    return value


async def _record(
    ledger: BudgetLedger,
    scope: str,
    spec: ToolSpec,
    result: Any = None,
) -> None:
    """记一次调用：次数 +1、成本按 v1 计价、token 按输出估算。

    记账本身失败（Redis 抖动等）只 warning，不能影响工具结果。
    """
    try:
        tokens = estimate_tokens(str(result)) if result is not None else 0
        await _maybe_await(ledger.add(scope, calls=1, cost=call_cost(spec), tokens=tokens))
    except Exception as exc:  # noqa: BLE001 - 记账失败不影响本次调用结果
        logger.warning("工具预算记账失败（%s）：%s", spec.name, exc)
