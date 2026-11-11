"""LangChain 回调 -> span（T7-0）。

只认两类工作单元：

| 回调 | span | 说明 |
|---|---|---|
| `on_chat_model_start` + `on_llm_end` / `on_chat_model_end` | `kind="llm"` | LangChain 1.x 的 chat model 完成事件走 `on_llm_end`，这里两个都接，按 run_id 幂等 |
| `on_tool_start` + `on_tool_end` / `on_tool_error` | `kind="tool"` | 工具边界 |
| `on_chain_start` / `on_chain_end` | 无 | 只做透传，不产 span |

`attrs` 严格只填契约白名单：`tool / tokens_in / tokens_out / cost / error_kind / approval_id`。
`cost` 不在回调里算（工具计价走 G5 账本，一轮汇总时由 `turn.py` 读快照）。

回调绝不能影响主流程：每个入口都兜住异常，只 warning；拿不到 turn 上下文时跳过 span。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

from agent.observability.tracer import SPAN_ATTRS, SpanHandle, Tracer, current_turn
from config import settings

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

_warned_no_langfuse = False


def _run_key(run_id: Any) -> str:
    return str(run_id)


def _serialized_name(serialized: Any, metadata: Any, default: str) -> str:
    """从 serialized / metadata 里取模型或工具名。"""
    if isinstance(serialized, dict):
        name = serialized.get("name")
        if isinstance(name, str) and name:
            return name
        ident = serialized.get("id")
        if isinstance(ident, list) and ident and isinstance(ident[-1], str):
            return ident[-1]
    if isinstance(metadata, dict):
        for key in ("ls_model_name", "name", "tool_name"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return default


def _extract_tokens(response: Any) -> dict[str, int]:
    """从 LLMResult 里取 token 用量；取不到就返回空 dict（不写 0 占位）。"""
    tokens_in: int | None = None
    tokens_out: int | None = None

    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(usage, dict):
            tokens_in = usage.get("prompt_tokens", usage.get("input_tokens"))
            tokens_out = usage.get("completion_tokens", usage.get("output_tokens"))

    if tokens_in is None and tokens_out is None:
        try:
            message = response.generations[0][0].message
            usage_metadata = getattr(message, "usage_metadata", None) or {}
            tokens_in = usage_metadata.get("input_tokens")
            tokens_out = usage_metadata.get("output_tokens")
        except (AttributeError, IndexError, KeyError, TypeError):
            pass

    out: dict[str, int] = {}
    if isinstance(tokens_in, int | float):
        out["tokens_in"] = int(tokens_in)
    if isinstance(tokens_out, int | float):
        out["tokens_out"] = int(tokens_out)
    return out


def _error_kind(error: BaseException) -> str:
    """失败分类（主文档 2.4.3）：超时视为 transient，其余 fatal。"""
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return "transient"
    return "fatal"


class SpanCollectorHandler(BaseCallbackHandler):
    """把 LangChain 的模型 / 工具回调映射成 JSONL span。"""

    #: 回调抛错不要向上冒（追踪不是业务依赖）
    raise_error = False

    def __init__(self, tracer: Tracer) -> None:
        super().__init__()
        self._tracer = tracer
        self._spans: dict[str, SpanHandle] = {}
        self._warned = False

    # ── 模型调用 ──────────────────────────────────────────────────

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_llm(run_id, serialized, metadata)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # 兼容非 chat 的 BaseLLM（chat model 不会走这里）
        self._start_llm(run_id, serialized, metadata)

    def on_chat_model_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._end_llm(run_id, response)

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        # langchain-core 1.x：chat model 的完成事件实际派发到 on_llm_end
        self._end_llm(run_id, response)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._fail(run_id, error)

    # ── 工具调用 ──────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_tool(run_id, serialized, metadata)

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._end(run_id, "ok", {})

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._fail(run_id, error)

    # ── chain：只透传，不产 span ──────────────────────────────────

    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        return None

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        return None

    # ── 内部 ──────────────────────────────────────────────────────

    def _start_llm(
        self,
        run_id: Any,
        serialized: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> None:
        self._safe_start(run_id, _serialized_name(serialized, metadata, "llm"), "llm")

    def _start_tool(
        self,
        run_id: Any,
        serialized: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> None:
        name = _serialized_name(serialized, metadata, "tool")
        self._safe_start(run_id, name, "tool", attrs={"tool": name})

    def _safe_start(
        self,
        run_id: Any,
        name: str,
        kind: str,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        try:
            if current_turn() is None:
                self._warn_once("回调找不到 turn 上下文，跳过 span")
                return
            span = self._tracer.start_span(name, kind, attrs=attrs)
            if not span.skipped:
                self._spans[_run_key(run_id)] = span
        except Exception as exc:  # noqa: BLE001 - 追踪失败不影响模型/工具
            self._warn_once(f"span 开启失败：{exc}")

    def _end_llm(self, run_id: Any, response: Any) -> None:
        try:
            tokens = _extract_tokens(response)
        except Exception as exc:  # noqa: BLE001
            self._warn_once(f"token 用量解析失败：{exc}")
            tokens = {}
        self._end(run_id, "ok", tokens)

    def _end(self, run_id: Any, status: str, attrs: dict[str, Any]) -> None:
        span = self._spans.pop(_run_key(run_id), None)
        if span is None:
            return
        try:
            self._tracer.end_span(span, status=status, attrs=attrs)
        except Exception as exc:  # noqa: BLE001
            self._warn_once(f"span 结束失败：{exc}")

    def _fail(self, run_id: Any, error: BaseException) -> None:
        self._end(run_id, "error", {"error_kind": _error_kind(error)})

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        logger.warning(message)


def build_callbacks(tracer: Tracer, langfuse_handler: Any = None) -> list[Any]:
    """组装 `/chat/conversation` 的 callbacks：Langfuse（可选）+ JSONL collector。

    - Langfuse 不可用（未启用/缺密钥/初始化失败）时只挂 collector；
    - `LANGFUSE_ENABLED=1` 但没有 handler（缺密钥）时只 warning 一次。
    """
    global _warned_no_langfuse

    handlers: list[Any] = []
    if langfuse_handler is not None:
        handlers.append(langfuse_handler)
    elif settings.LANGFUSE_ENABLED and not _warned_no_langfuse:
        _warned_no_langfuse = True
        logger.warning("LANGFUSE_ENABLED=1 但未拿到 handler（缺密钥），Langfuse 追踪降级为仅 JSONL")
    handlers.append(SpanCollectorHandler(tracer))
    return handlers


__all__ = ["SPAN_ATTRS", "SpanCollectorHandler", "build_callbacks"]
