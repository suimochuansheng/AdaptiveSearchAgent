"""链路追踪端口与 JSONL 实现（T7-0）。

口径以 `docs/contracts/trace-span.md` v1 为准：

- 一个 span 落一行 JSON，字段恒为十字段
  `trace_id / span_id / parent_span_id / session_id / turn_index / name / kind / start / end / attrs`；
- `kind` 只允许六值（`SPAN_KINDS`）；`attrs` 只允许契约 §3 的六个键（`SPAN_ATTRS`），
  **例外**：`turn` 根 span 额外承载本轮汇总 `tools` / `span_count`（T7-0 卡面要求，
  与 v1 §3 的差异见交接单，待计划窗口升 v2）；
- 开关关闭时调用方根本不会构造本模块的 tracer（`get_tracer()` 也不会被调用），
  因此默认路径零文件 IO；
- 写入失败（磁盘/权限）只打一条 warning，本进程降级为 Noop，不影响会话。

并发口径：`open(path, "a")` + **一次** `write()`，单行 JSON 不超过 PIPE_BUF 量级，
多协程/多进程追加不会交错；不加锁文件。
"""

from __future__ import annotations

import json
import logging
import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

#: 契约 §2：kind 六值（冻结）
SPAN_KINDS: frozenset[str] = frozenset({"turn", "llm", "tool", "compress", "offload", "approval"})

#: 契约 §3：通用 attrs 白名单（冻结六键）
SPAN_ATTRS: frozenset[str] = frozenset(
    {"tool", "tokens_in", "tokens_out", "cost", "error_kind", "approval_id"}
)

#: 契约 §3 v2：只对 kind="turn" 放行的汇总键（其余五 kind 仍严格六键）
TURN_ONLY_ATTRS: frozenset[str] = frozenset({"tools", "span_count"})

#: turn 根 span 可用的全部 attrs 键
TURN_ATTRS: frozenset[str] = SPAN_ATTRS | TURN_ONLY_ATTRS

#: 契约 §1：十字段（顺序即落盘顺序）
TRACE_FIELDS: tuple[str, ...] = (
    "trace_id",
    "span_id",
    "parent_span_id",
    "session_id",
    "turn_index",
    "name",
    "kind",
    "start",
    "end",
    "attrs",
)


def new_id() -> str:
    """32 位 hex id（契约：`secrets.token_hex(16)`）。"""
    return secrets.token_hex(16)


def _now() -> str:
    """ISO 8601 带本地偏移、毫秒精度（契约 §1 示例格式）。"""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


@dataclass
class SpanHandle:
    """一个进行中 / 已结束的 span。

    契约要求的最小字段是 `span_id / trace_id / start / turn_index / session_id`；
    其余字段是父子关联与落盘所需，不额外暴露给调用方。
    """

    span_id: str
    trace_id: str
    start: str
    turn_index: int
    session_id: str
    name: str = ""
    kind: str = ""
    parent_span_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    end: str = ""
    #: 没有 turn 上下文等兜底场景：不落盘（避免产生找不到父节点的孤儿 span）
    skipped: bool = field(default=False, repr=False)


class Tracer(Protocol):
    """追踪端口：只认开始 / 结束两个动作。"""

    def start_span(
        self,
        name: str,
        kind: str,
        *,
        parent: SpanHandle | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> SpanHandle:
        """开一个 span；`parent` 为空时挂到当前 turn 根 span 下。"""
        ...

    def end_span(
        self,
        h: SpanHandle,
        *,
        status: str = "ok",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """结束 span 并落盘（恰好一行）。"""
        ...


@dataclass
class TurnContext:
    """一轮的追踪上下文：同一轮的 span 共用 trace_id / session / turn_index。"""

    session_id: str
    turn_index: int
    trace_id: str
    root: SpanHandle | None = None
    spans: list[SpanHandle] = field(default_factory=list)


_TURN: ContextVar[TurnContext | None] = ContextVar("trace_turn", default=None)
_ACTIVE: list[TurnContext] = []


def current_turn() -> TurnContext | None:
    """读当前 turn 上下文。

    优先 contextvar；某些框架会在新任务里丢上下文，此时若进程内只有一个活跃轮次
    就复用它（单会话常见），否则返回 None（调用方据此跳过 span）。
    """
    ctx = _TURN.get()
    if ctx is not None:
        return ctx
    if len(_ACTIVE) == 1:
        return _ACTIVE[0]
    return None


@contextmanager
def bind_turn(
    tracer: Tracer,
    *,
    session_id: str,
    turn_index: int,
    trace_id: str | None = None,
) -> Iterator[TurnContext]:
    """绑定一轮的上下文（由 `turn.turn_span` 使用，也可单独用于测试）。"""
    del tracer  # 端口本身无状态，保留参数是为了显式绑定到某个 tracer
    ctx = TurnContext(
        session_id=session_id,
        turn_index=turn_index,
        trace_id=trace_id or new_id(),
    )
    token = _TURN.set(ctx)
    _ACTIVE.append(ctx)
    try:
        yield ctx
    finally:
        _ACTIVE.remove(ctx)
        _TURN.reset(token)


class JsonlTracer:
    """把 span 追加写到 `TRACE_DIR/<YYYY-MM-DD>.jsonl`。"""

    def __init__(self, trace_dir: str | Path) -> None:
        self._dir = Path(trace_dir)
        self._broken = False
        self._warned = False

    # ── Tracer 端口 ────────────────────────────────────────────────

    def start_span(
        self,
        name: str,
        kind: str,
        *,
        parent: SpanHandle | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> SpanHandle:
        if kind not in SPAN_KINDS:
            raise ValueError(f"非法 span kind: {kind!r}（只允许六值）")
        ctx = current_turn()

        if kind == "turn":
            trace_id = (
                parent.trace_id if parent is not None else (ctx.trace_id if ctx else new_id())
            )
            return self._new_handle(
                ctx,
                name=name,
                kind=kind,
                trace_id=trace_id,
                parent_span_id=None,
                attrs=attrs,
            )

        if parent is None and (ctx is None or ctx.root is None):
            self._warn_once("没有 turn 上下文，跳过 span（避免产生孤儿节点）")
            return SpanHandle(
                span_id="",
                trace_id="",
                start="",
                turn_index=0,
                session_id="",
                name=name,
                kind=kind,
                skipped=True,
            )

        if parent is not None:
            trace_id = parent.trace_id
            parent_span_id = parent.span_id
            session_id = parent.session_id
            turn_index = parent.turn_index
        else:
            assert ctx is not None and ctx.root is not None
            trace_id = ctx.trace_id
            parent_span_id = ctx.root.span_id
            session_id = ctx.session_id
            turn_index = ctx.turn_index

        handle = SpanHandle(
            span_id=new_id(),
            trace_id=trace_id,
            start=_now(),
            turn_index=turn_index,
            session_id=session_id,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            attrs=dict(attrs or {}),
        )
        if ctx is not None:
            ctx.spans.append(handle)
        return handle

    def end_span(
        self,
        h: SpanHandle,
        *,
        status: str = "ok",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        if h.skipped:
            return
        h.status = status
        if attrs:
            h.attrs.update(attrs)
        if status == "error" and not h.attrs.get("error_kind"):
            # 契约没有 status 字段：错误只通过 attrs.error_kind 表达（非空即失败）
            h.attrs["error_kind"] = "fatal"
        h.end = _now()
        self._drop_illegal_attrs(h)
        self._append(h)

    # ── 内部 ──────────────────────────────────────────────────────

    def _new_handle(
        self,
        ctx: TurnContext | None,
        *,
        name: str,
        kind: str,
        trace_id: str,
        parent_span_id: str | None,
        attrs: dict[str, Any] | None,
    ) -> SpanHandle:
        handle = SpanHandle(
            span_id=new_id(),
            trace_id=trace_id,
            start=_now(),
            turn_index=ctx.turn_index if ctx else 0,
            session_id=ctx.session_id if ctx else "",
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            attrs=dict(attrs or {}),
        )
        if ctx is not None:
            ctx.spans.append(handle)
            if kind == "turn" and ctx.root is None:
                ctx.root = handle
        return handle

    def _drop_illegal_attrs(self, h: SpanHandle) -> None:
        allowed = TURN_ATTRS if h.kind == "turn" else SPAN_ATTRS
        extra = set(h.attrs) - allowed
        if extra:
            for key in extra:
                h.attrs.pop(key, None)
            self._warn_once(f"span {h.name!r} attrs 越界已丢弃: {sorted(extra)}")
        if h.kind == "turn":
            self._drop_bad_turn_types(h)

    def _drop_bad_turn_types(self, h: SpanHandle) -> None:
        """v2 只对 turn 开口，但类型必须对：`tools: string[]` / `span_count: int`。"""
        bad: list[str] = []
        if "tools" in h.attrs:
            tools = h.attrs["tools"]
            if not (isinstance(tools, list) and all(isinstance(item, str) for item in tools)):
                h.attrs.pop("tools", None)
                bad.append("tools")
        if "span_count" in h.attrs:
            count = h.attrs["span_count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                h.attrs.pop("span_count", None)
                bad.append("span_count")
        if bad:
            self._warn_once(f"span {h.name!r} turn 汇总键类型不合法已丢弃: {bad}")

    def _append(self, h: SpanHandle) -> None:
        if self._broken:
            return
        record = {
            "trace_id": h.trace_id,
            "span_id": h.span_id,
            "parent_span_id": h.parent_span_id,
            "session_id": h.session_id,
            "turn_index": h.turn_index,
            "name": h.name,
            "kind": h.kind,
            "start": h.start,
            "end": h.end,
            "attrs": h.attrs,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        path = self._dir / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)  # 单次 write：追加不交错
        except Exception as exc:  # noqa: BLE001 - 追踪不是业务依赖，写失败不能打断会话
            self._broken = True
            logger.warning("链路追踪写入失败，本进程降级为 Noop：%s", exc)

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        logger.warning(message)


class NoopTracer:
    """全空实现：不分配 id、不产文件。"""

    def start_span(
        self,
        name: str,
        kind: str,
        *,
        parent: SpanHandle | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> SpanHandle:
        return SpanHandle(
            span_id="",
            trace_id="",
            start="",
            turn_index=0,
            session_id="",
            name=name,
            kind=kind,
            skipped=True,
        )

    def end_span(
        self,
        h: SpanHandle,
        *,
        status: str = "ok",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        return None


_default_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """进程级默认 tracer（惰性创建：开关关闭时没人调用它，因此零 IO）。"""
    global _default_tracer
    if _default_tracer is None:
        from config import settings

        _default_tracer = JsonlTracer(settings.TRACE_DIR)
    return _default_tracer
