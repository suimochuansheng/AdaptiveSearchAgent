"""子代理工具（T9-4）：主 Agent 把“脏活”交给一个独立上下文的子 Agent，只收回摘要。

v1 刻意做小（不改图结构、不做子代理专用工具集）：

- 复用 `build_conversation_graph()`，子代理用**独立 `thread_id` + 内存 checkpointer
  （`MemorySaver`）**，跑完即弃，不污染主会话的 checkpoint；
- 初始消息只放 `goal + constraints`，**不塞主会话历史**——上下文隔离就是这一条；
- 子代理与主 Agent 共用同一套工具（含子代理自己），所以必须有**递归守卫**：
  `contextvars` 记深度，同一调用栈里已有子代理时直接返回可读错误（深度上限 1 层）；
- 只回最后一条 AI 消息，并截断到**生效上限** `SUBAGENT_EFFECTIVE_MAX_OUTPUT_TOKENS`
  （启动时按 `tool_output_token_budget` 夹紧，避免进主会话被二次压缩）；
- 超时 / 异常一律返回可读错误字符串（不抛栈），主会话不会因为子任务失败而挂掉。

追踪（T7-0）关联：子代理内部的 span 必须落在**本轮同一个 trace_id** 下，且父节点是
“调用它的那条 tool span”。tool span 是回调旁路创建的、工具函数拿不到句柄，所以这里
从 LangChain 的回调管理器（`var_child_runnable_config`）反查正在执行的 `subagent_task`
span，再用一个“影子 turn 上下文”（`ctx.root = tool_span`）包住子图调用：子图里所有
`start_span(parent=None)` 都会挂到这条 tool span 下，trace_id / session / turn_index 复用。
找不到 tool span（没开追踪 / 被直接调用）时退化为不特殊处理，不报错。

预算：子代理的 tools 节点本来就把 `tool_scope(thread_id)` 设成子代理自己的 thread_id，
所以子代理内部工具调用记在**子 scope**，主 scope 只记 `subagent_task` 这一次调用。

开关：`ENABLE_SUBAGENT=False`（默认）时本模块不被登记，注册表 / schema 哈希保持基线。
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage

from agent.domain.tool import CostModel, ToolSpec
from config import settings
from src.context_manager import estimate_tokens
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from agent.observability.tracer import SpanHandle

logger = get_logger()

#: 工具名（注册表键 / 模型看到的函数名 / tool span 名，三处必须一致）
SUBAGENT_TOOL_NAME = "subagent_task"

#: 递归守卫：当前调用栈里已有几层子代理（v1 只允许 0 或 1）
_DEPTH: ContextVar[int] = ContextVar("subagent_depth", default=0)

#: 截断标记（保留首尾时插在中间）
_TRUNCATED_MARK = "…（已截断）"

#: 子代理的初始提示词前缀：冒烟脚本 / 调试靠它区分“这是子代理的会话”
_GOAL_PREFIX = "子任务目标："


# ── 对外：工具可调用体 ────────────────────────────────────────────────


async def subagent_task(goal: str, constraints: str = "") -> str:
    """把一个独立子任务交给子代理执行，只返回结论摘要（不会回传中间消息）。

    适用：需要多步工具调用、过程很长、主会话只关心结论的“脏活”。
    子代理**看不到**当前会话历史，所以 `goal` 必须自包含（背景、要查什么、要什么格式）。

    Args:
        goal: 子任务目标（自包含；写全必要背景）。
        constraints: 可选约束（输出格式 / 边界 / 不要做什么）。

    Returns:
        结论摘要（<= 生效上限 `SUBAGENT_EFFECTIVE_MAX_OUTPUT_TOKENS`）；
        失败 / 超时 / 嵌套时返回可读错误文案。
    """
    if _DEPTH.get() >= 1:
        logger.warning("子代理递归调用被拒绝：%s", str(goal)[:80])
        return "子代理不可嵌套：当前已经在一个子代理内部，请直接基于已有的工具结果作答。"

    text = str(goal or "").strip()
    if not text:
        return "子代理需要明确的目标：请把 goal 写成自包含的一句话或一段话。"

    max_turns = max(1, settings.SUBAGENT_MAX_TURNS)
    token = _DEPTH.set(_DEPTH.get() + 1)
    try:
        summary = await asyncio.wait_for(
            _run_subagent(text, str(constraints or "").strip(), max_turns),
            timeout=settings.SUBAGENT_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning("子代理执行超时（%ss）", settings.SUBAGENT_TIMEOUT_SECONDS)
        return (
            f"子代理执行超时（超过 {settings.SUBAGENT_TIMEOUT_SECONDS:g} 秒），已放弃本次子任务。"
            "可以缩小目标范围，或把任务拆成更小的子任务后重试。"
        )
    except Exception as exc:  # noqa: BLE001 - 子代理失败不能把主会话拖垮
        logger.exception("子代理执行失败")
        return f"子代理执行失败：{exc}。可以换个更具体的目标，或直接用现有工具回答。"
    finally:
        _DEPTH.reset(token)

    # 用生效值（启动时按 tool_output_token_budget 夹紧）：摘要进主会话会过
    # shrink_tool_message，超预算会被二次压缩，截断标记会丢。
    return _truncate_summary(summary, settings.SUBAGENT_EFFECTIVE_MAX_OUTPUT_TOKENS)


# ── 对外：登记（只在 ENABLE_SUBAGENT=True 时生效）────────────────────


def register_subagent_tool() -> bool:
    """把 `subagent_task` 登记进注册表；开关关闭或已登记时返回 False。

    与 MCP 的“启动时登记 + reset 图”是同一套做法：登记后由组合根重建会话图，
    模型下一轮就能看到这个工具。
    """
    if not settings.ENABLE_SUBAGENT:
        return False

    from agent.adapters.tools import registry

    if SUBAGENT_TOOL_NAME in registry.TOOL_SPECS:
        return False
    registry.register(SUBAGENT_TOOL_NAME, subagent_task, _build_spec())
    return True


def _build_spec() -> ToolSpec:
    """构造 ToolSpec（取值见 tool-spec 契约：external / 按次计费 / 不审批）。"""
    return ToolSpec(
        name=SUBAGENT_TOOL_NAME,
        description=(
            "把一个独立子任务交给子代理执行（独立上下文，只回传结论摘要，看不到当前会话历史）。\n"
            "适用：需要多步工具调用、过程很长、主会话只关心最终结论的复杂子任务。\n"
            "调用时请在 goal 里写全必要背景，并在 constraints 里写明输出格式或边界。"
        ),
        input_schema={
            "properties": {
                "goal": {
                    "description": "子任务目标（自包含：背景、要查什么、要什么结果）",
                    "type": "string",
                },
                "constraints": {
                    "default": "",
                    "description": "可选约束（输出格式 / 边界 / 不要做什么）",
                    "type": "string",
                },
            },
            "required": ["goal"],
            "type": "object",
        },
        permission="external",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=settings.SUBAGENT_TIMEOUT_SECONDS,
        max_output_tokens=settings.SUBAGENT_EFFECTIVE_MAX_OUTPUT_TOKENS,
        requires_approval=False,
        feature_flag="ENABLE_SUBAGENT",
    )


# ── 内部：跑子图 ─────────────────────────────────────────────────────


def _build_prompt(goal: str, constraints: str) -> str:
    """子代理的初始用户消息：只放本子任务的 goal / constraints，**不放**主会话历史。"""
    prompt = f"{_GOAL_PREFIX}{goal}"
    if constraints:
        prompt += f"\n约束：{constraints}"
    prompt += "\n请独立完成这个子任务，只输出最终结论（关键事实与结果），不要复述过程。"
    return prompt


async def _run_subagent(goal: str, constraints: str, max_turns: int) -> str:
    """建子图（独立 thread_id + 一次性 MemorySaver）并跑到结束，返回最后一条 AI 文本。"""
    from langgraph.checkpoint.memory import MemorySaver

    from src.conversation_graph import build_conversation_graph

    graph = build_conversation_graph(checkpointer=MemorySaver())
    thread_id = f"subagent-{uuid.uuid4().hex[:8]}"
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        # 每轮 = agent + tools 两步，再多给一步收尾（最终回答）
        "recursion_limit": 2 * max_turns + 1,
    }
    # 不显式传 callbacks：子图会继承外层工具调用的回调（LangChain 的
    # `var_child_runnable_config`），span 自然落在同一轮 trace 里；显式再传一份会**重复**
    # 产生 span（回调是追加而不是替换，冒烟脚本已抓到过），父节点由下面的影子上下文负责。

    with _trace_under_tool_span():
        try:
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content=_build_prompt(goal, constraints))]},
                config=config,
            )
        except Exception as exc:
            # 只吞“轮次上限”：其它异常（模型不可用等）交给调用方统一转可读错误
            from langgraph.errors import GraphRecursionError

            if not isinstance(exc, GraphRecursionError):
                raise
            logger.warning("子代理达到轮次上限（%d 轮），回传最后一次结论", max_turns)
            snapshot = await graph.aget_state(config)
            state = dict(snapshot.values or {}) if snapshot is not None else {}

    logger.info("子代理完成", extra={"subagent_thread_id": thread_id})
    return _extract_summary(state)


def _extract_summary(state: Any) -> str:
    """取最后一条**有文本**的 AI 消息；没有就返回可读兜底文案。"""
    messages = list((state or {}).get("messages") or [])
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = _content_to_text(message.content)
        if text:
            return text
    return "（子代理没有产出可回传的结论：可能达到轮次上限，或工具结果里没有可用信息）"


def _content_to_text(content: Any) -> str:
    """AIMessage.content 可能是 str 或多模态列表，统一成文本。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
        ]
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


# ── 内部：摘要截断（保留首尾 + 中间标记）──────────────────────────────


def _truncate_summary(text: str, max_tokens: int) -> str:
    """把摘要截到 `max_tokens` 以内：超限时保留首尾、中间插 `…（已截断）`。"""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    budget = max_tokens - estimate_tokens(_TRUNCATED_MARK)
    if budget <= 0:
        return _TRUNCATED_MARK

    head_budget = budget // 2
    head = _fit_head(text, head_budget)
    tail = _fit_tail(text, budget - head_budget)
    return f"{head}\n{_TRUNCATED_MARK}\n{tail}"


def _fit_head(text: str, budget: int) -> str:
    """最长前缀，使其估算 token ≤ budget（`estimate_tokens` 对前缀长度单调）。"""
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip()


def _fit_tail(text: str, budget: int) -> str:
    """最长后缀，使其估算 token ≤ budget。"""
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[-mid:]) <= budget:
            low = mid
        else:
            high = mid - 1
    return text[-low:].lstrip() if low else ""


# ── 内部：追踪关联（同一 trace_id + 父节点 = 调用方的 tool span）──────


def _active_tool_span(name: str) -> SpanHandle | None:
    """从 LangChain 回调管理器里反查“正在执行的那条 tool span”。

    tool span 由 `SpanCollectorHandler` 在 `on_tool_start` 创建，工具函数本身拿不到句柄；
    但工具执行时 `var_child_runnable_config` 里能拿到回调管理器，它持有 handler 的
    进行中 span 表。按名字 + 未结束匹配；找不到返回 None（调用方退化为不特殊处理）。
    """
    try:
        from langchain_core.runnables.config import var_child_runnable_config

        config = var_child_runnable_config.get()
    except Exception:  # noqa: BLE001 - 追踪是旁路，失败不能影响子代理执行
        return None

    manager = (config or {}).get("callbacks")
    for handler in getattr(manager, "handlers", None) or []:
        spans = getattr(handler, "_spans", None)
        if not isinstance(spans, dict):
            continue
        for span in reversed(list(spans.values())):
            if (
                getattr(span, "kind", "") == "tool"
                and getattr(span, "name", "") == name
                and not getattr(span, "end", "")
            ):
                return span
    return None


@contextmanager
def _trace_under_tool_span() -> Iterator[None]:
    """把子图里的 span 挂到调用它的 tool span 下（同一 trace_id + 父节点）。

    做法：临时绑定一个“影子 turn 上下文”，`root` 指向 tool span。子图的回调
    （显式传入的 `SpanCollectorHandler`）建 span 时读 `current_turn().root`，
    于是 parent_span_id = tool span、trace_id / session / turn_index 全部复用。
    退出时把子代理产生的 span 并入外层 turn 的 span 列表，保证本轮汇总（span_count）完整。
    """
    if not settings.ENABLE_TRACE:
        yield
        return

    parent = _active_tool_span(SUBAGENT_TOOL_NAME)
    if parent is None or getattr(parent, "skipped", False):
        yield
        return

    from agent.observability.tracer import bind_turn, current_turn, get_tracer

    outer = current_turn()
    with bind_turn(
        get_tracer(),
        session_id=parent.session_id,
        turn_index=parent.turn_index,
        trace_id=parent.trace_id,
    ) as ctx:
        ctx.root = parent
        try:
            yield
        finally:
            if outer is not None:
                outer.spans.extend(ctx.spans)
