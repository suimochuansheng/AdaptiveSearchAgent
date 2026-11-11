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
| `require_approval` | v1 返回“需人工审批…本次已拒绝”，**不调用 fn** | 不记 |

`require_approval` 的打断（interrupt）与审批 UI 属 G7/T9-2，本卡只留这一个返回分支。

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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from agent.domain.budget import BudgetLedger, call_cost
from agent.domain.tool_policy import PolicyContext, decide
from config import settings
from src.context_manager import estimate_tokens

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from agent.domain.budget import BudgetSnapshot
    from agent.domain.tool import ToolSpec

logger = logging.getLogger(__name__)

_UNKNOWN_SCOPE = "unknown"

_scope_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("tool_scope", default=None)
_scope_warned = False
_default_ledger: BudgetLedger | None = None

__all__ = [
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
        if decision.kind != "allow":
            # deny / require_approval：不调用 fn，也不记账
            logger.info(
                "工具调用被策略拦截：%s -> %s（%s）", spec.name, decision.kind, decision.code
            )
            return decision.reason

        try:
            result = await _run_with_timeout(fn, spec.timeout_seconds, args, kwargs)
        except TimeoutError:
            await _record(active, scope, spec)
            logger.warning("工具 %s 执行超时（>%ss）", spec.name, f"{spec.timeout_seconds:g}")
            return f"工具执行超时（>{spec.timeout_seconds:g}s），已终止"
        except Exception as exc:  # noqa: BLE001 - 工具异常归一为可读文本，不打断 Agent 循环
            await _record(active, scope, spec)
            logger.warning("工具 %s 执行失败：%s: %s", spec.name, type(exc).__name__, exc)
            return f"工具 {spec.name} 执行出错：{type(exc).__name__}: {exc}"

        await _record(active, scope, spec, result)
        return result

    # 包装不能改变 schema：签名必须仍然跟随 `__wrapped__`
    if inspect.signature(wrapped) != original_signature:
        raise RuntimeError(f"工具包装后签名不一致：{spec.name}")
    return wrapped


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
