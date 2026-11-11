"""本次工具调用的结果码（T9-2，修 OI-INT-002）。

执行器（`agent/adapters/tools/executor.py`）在工具返回前写入一个稳定错误码，
LangChain 回调（`agent/observability/langchain_handler.py`）在工具结束时读出并填进
`span.attrs["error_kind"]`，**读完即清空**，避免串到下一次调用。

为什么用 contextvar 而不是在 executor 里直接调 tracer：

- executor 不依赖 tracer（分层：执行器只管执行，追踪是旁路）；
- ToolNode 并发执行同一轮的多个 tool_call 时，每个工具在独立的 asyncio Task 里，
  contextvars 随 Task 复制，互不污染；
- 开关关闭 / 正常执行时不写，handler 读到 `None` 就什么都不填，G6 行为不变。

**实测补充（LangChain 1.6 / LangGraph 1.2）**：同步回调由
`run_in_executor` + `copy_context()` 派发（`callbacks/manager.py`），contextvar 到不了
回调线程；且 `on_tool_end` 拿到的不是工具原始返回值，而是 `ToolMessage`。所以结果码
同时走三条读取源（handler 按顺序读，读到即清）：

1. contextvar（同任务 / 异步回调时最直接）；
2. `tool_call_id -> code` 索引（ToolMessage 上带 `tool_call_id`；并发安全）；
3. 返回值 / 异常对象上的 `error_kind` 属性（审批挂起走异常对象）。

取值集合（与 `docs/contracts/tool-policy` 的决策码一致）：

| 值 | 含义 |
|---|---|
| `approval_required` | 需人工审批：HITL 关闭时直接拒绝，或 HITL 开启时已挂起 |
| `denied_timeout` | 工具执行超时（钱已花，照记账） |
| `denied_calls` | 本轮工具调用次数超限 |
| `denied_cost` | 会话工具费用将超限 |
| `rejected_by_user` | 审批恢复后用户选择拒绝 |
"""

from __future__ import annotations

from contextvars import ContextVar
from threading import Lock
from typing import Any

__all__ = [
    "ToolOutcome",
    "clear_tool_outcome",
    "current_tool_outcome",
    "mark_outcome",
    "remember_outcome",
    "set_tool_outcome",
    "take_outcome_for",
    "take_tool_outcome",
]

_outcome_var: ContextVar[str | None] = ContextVar("tool_outcome", default=None)

#: tool_call_id -> 结果码。LangGraph 1.x 的 `on_tool_end` 收到的是 ToolMessage（不是工具
#: 原始返回值），字符串属性会被丢掉；tool_call_id 在 ToolMessage 上，用它做第三读取源。
_pending: dict[str, str] = {}
_pending_lock = Lock()


def remember_outcome(key: str, code: str) -> None:
    """按 tool_call_id 记下结果码；空 key 直接忽略。"""
    if not key:
        return
    with _pending_lock:
        _pending[key] = code


def take_outcome_for(key: str) -> str | None:
    """按 tool_call_id 取出并删除结果码。"""
    if not key:
        return None
    with _pending_lock:
        return _pending.pop(key, None)


class ToolOutcome(str):
    """带结果码的字符串：值仍是给模型看的可读文案，`error_kind` 供回调读取。

    为什么需要它：LangChain 的**同步**回调由 `run_in_executor` + `copy_context()` 派发
    （`callbacks/manager.py::_ahandle_event_for_handler`），执行器所在任务里设置的
    contextvar 到不了回调线程。字符串 / 异常对象是同一个对象引用，跨线程仍然可读，
    所以结果码同时挂在这里，作为 contextvar 之外的第二读取源（见 handler 的
    `_outcome_of`）。正常执行不包装，仍是普通 str。
    """

    error_kind: str

    def __new__(cls, value: str, error_kind: str) -> ToolOutcome:
        obj = super().__new__(cls, value)
        obj.error_kind = error_kind
        return obj


def mark_outcome(value: Any, code: str) -> Any:
    """把结果码挂到返回值上；非字符串原样返回（异常路径由调用方设置属性）。"""
    set_tool_outcome(code)
    if isinstance(value, str) and not isinstance(value, ToolOutcome):
        return ToolOutcome(str(value), code)
    return value


def set_tool_outcome(code: str) -> None:
    """标记本次工具调用的结果码；由执行器在返回前调用。"""
    _outcome_var.set(code)


def current_tool_outcome() -> str | None:
    """只读当前结果码（不清空），调试 / 测试用。"""
    return _outcome_var.get()


def take_tool_outcome() -> str | None:
    """读出并清空结果码；handler 在 `on_tool_end` / `on_tool_error` 里调用。"""
    code = _outcome_var.get()
    _outcome_var.set(None)
    return code


def clear_tool_outcome() -> None:
    """清空结果码；handler 在 `on_tool_start` 里调用，防止上一轮的码残留。"""
    _outcome_var.set(None)
