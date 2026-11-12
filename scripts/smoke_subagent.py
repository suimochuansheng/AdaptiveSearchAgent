#!/usr/bin/env python
"""T9-4 冒烟：子代理工具（独立上下文 + 摘要回传 + 递归守卫 + 超时 + 预算 + 追踪）。

用**假模型 + 假工具**（不连 Ollama / 不连 RAG / 不连 Redis）演示五件事：

1. 开关关闭：注册表仍 5 个内置工具、schema 哈希 == 基线 `c48c9d48…`、没有 `subagent_task`；
2. 开关打开：注册表 +1，`to_lc_tools()` 名字列表末尾多一个 `subagent_task`；
3. 调用一次：主会话只新增"AI 工具调用 + 一条工具结果 + 最终回答"，
   子代理内部的 echo 工具结果**不出现在**主会话（上下文隔离）；
4. 递归守卫：子代理里再调 `subagent_task` -> 返回“子代理不可嵌套”，不递归；
5. 超时 / 预算 / 追踪：`SUBAGENT_TIMEOUT_SECONDS=1` 跑长任务 -> 可读超时错误；
   子代理的工具调用记在子 scope（独立 thread_id），主 scope 只记 1 次；
   子代理 span 落在同一 trace_id 下，父节点 = 调用它的那条 tool span。

跑法（仓库根目录，任意 cwd 均可）：

    poetry run python scripts/smoke_subagent.py

退出码 0 = 全部断言通过；1 = 有失败（输出里逐条 [FAIL]）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import 项目模块
    sys.path.insert(0, str(REPO))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from agent.adapters.tools import executor, registry
from agent.adapters.tools.budget_memory import MemoryBudgetLedger
from agent.adapters.tools.subagent import (
    SUBAGENT_TOOL_NAME,
    _truncate_summary,
    register_subagent_tool,
    subagent_task,
)
from agent.domain.tool import CostModel, ToolSpec
from config import settings

BASELINE_NAMES = [
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
]
BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
GOAL_PREFIX = "子任务目标："

#: 假模型的剧本开关：normal / guard / timeout
SCRIPT: dict[str, str] = {"mode": "normal"}
#: 观测：子代理内部最多看到多少条消息（上下文隔离的反向证据）
OBSERVED: dict[str, int] = {"sub_messages": 0}

_PASS = 0
_FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """打印一条断言结果并累计计数。"""
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        _FAIL += 1
        print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ── 假模型 ─────────────────────────────────────────────────────────────


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _last_tool_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            return str(message.content)
    return ""


class FakeChatModel(BaseChatModel):
    """脚本化假模型：按“是否子代理会话 + 已有几条工具结果”决定下一步。"""

    model_config = ConfigDict(extra="ignore")

    @property
    def _llm_type(self) -> str:
        return "fake-subagent-smoke-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:
        # 与真实模型接口一致：只记绑定动作，不影响剧本
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=self._script(messages))])

    def _script(self, messages: list[Any]) -> AIMessage:
        is_sub = any(
            isinstance(message, HumanMessage) and str(message.content).startswith(GOAL_PREFIX)
            for message in messages
        )
        tool_count = sum(1 for message in messages if isinstance(message, ToolMessage))
        mode = SCRIPT["mode"]

        if is_sub:
            OBSERVED["sub_messages"] = max(OBSERVED["sub_messages"], len(messages))
            if mode == "guard":
                if tool_count == 0:
                    return _tool_call(
                        SUBAGENT_TOOL_NAME, {"goal": "在子代理里再套一层子代理"}, "sub-rec-1"
                    )
                return AIMessage(content=f"GUARD_SAW::{_last_tool_text(messages)}")
            if mode == "timeout":
                if tool_count == 0:
                    return _tool_call("slow_note", {"text": "慢任务", "delay": 3}, "sub-slow-1")
                return AIMessage(content="慢任务完成")
            # normal：跑两步工具（证明“多步”），然后只回一条结论
            if tool_count == 0:
                return _tool_call("echo_note", {"text": "子代理第一步"}, "sub-echo-1")
            if tool_count == 1:
                return _tool_call("echo_note", {"text": "子代理第二步"}, "sub-echo-2")
            return AIMessage(
                content=(
                    f"子代理结论：两步工具结果分别是 {_last_tool_text(messages)}；关键事实 X=42。"
                )
            )

        # 主会话
        if tool_count == 0:
            return _tool_call(
                SUBAGENT_TOOL_NAME,
                {"goal": "整理三件事实", "constraints": "只回结论"},
                "main-sub-1",
            )
        return AIMessage(content="主会话最终回答：子代理已返回摘要。")


# ── 假工具 ─────────────────────────────────────────────────────────────


async def echo_note(text: str) -> str:
    """回显一段文本（T9-4 冒烟假工具）。"""
    return f"echo:{text}"


async def slow_note(text: str, delay: float = 3.0) -> str:
    """延迟后回显（T9-4 冒烟假工具：验证子代理超时）。"""
    await asyncio.sleep(delay)
    return f"slow:{text}"


def _fake_spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="T9-4 冒烟假工具",
        input_schema={"properties": {}, "type": "object"},
        permission="read",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=30.0,
        max_output_tokens=500,
        requires_approval=False,
        feature_flag=None,
    )


# ── 工具：schema 指纹 / 主调用 ────────────────────────────────────────


def _schema_sha(tools: list[Any]) -> str:
    """与 scripts/smoke_tools_schema.py 同口径的 schema 指纹。"""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schemas = [convert_to_openai_tool(tool) for tool in tools]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest()


async def _run_main_call(thread_id: str) -> dict[str, Any]:
    """跑一次主会话（假模型 -> subagent_task -> 最终回答），返回 checkpoint 里的 state。"""
    from langgraph.checkpoint.memory import MemorySaver

    from src.conversation_graph import build_conversation_graph

    graph = build_conversation_graph(checkpointer=MemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    if settings.ENABLE_TRACE:
        from agent.observability.langchain_handler import build_callbacks
        from agent.observability.tracer import get_tracer
        from agent.observability.turn import turn_span

        tracer = get_tracer()
        config["callbacks"] = build_callbacks(tracer)
        trace_cm: Any = turn_span(tracer, thread_id, 0)
    else:
        trace_cm = nullcontext()

    async with trace_cm:
        await graph.ainvoke(
            {"messages": [HumanMessage(content="主会话问题：请让子代理整理事实")]},
            config=config,
        )

    snapshot = await graph.aget_state(config)
    return dict(snapshot.values or {})


def _sub_scopes(ledger: MemoryBudgetLedger) -> set[str]:
    """账本里子代理 scope（thread_id 以 subagent- 开头）的快照。"""
    data = getattr(ledger, "_data", {})
    return {str(scope) for scope in data if str(scope).startswith("subagent-")}


# ── 主流程 ─────────────────────────────────────────────────────────────


async def main() -> int:
    from src import conversation_graph as cg

    original_model = cg.ChatOllama
    setattr(cg, "ChatOllama", FakeChatModel)  # noqa: B010 - 冒烟里临时替换模型类
    try:
        _section("1. 开关关闭（基线）")
        names_before = [getattr(tool, "__name__", "?") for tool in registry.to_lc_tools()]
        sha_before = _schema_sha(registry.to_lc_tools())
        check(
            "注册表仍是 5 个内置工具",
            names_before == BASELINE_NAMES,
            ", ".join(names_before),
        )
        check("schema 哈希 == 基线 c48c9d48…", sha_before == BASELINE_SHA256, sha_before)
        check("注册表里没有 subagent_task", SUBAGENT_TOOL_NAME not in registry.TOOL_SPECS)

        _section("2. 开关打开（登记 + 模型可见）")
        settings.ENABLE_SUBAGENT = True
        registered = register_subagent_tool()
        names_after = [getattr(tool, "__name__", "?") for tool in registry.to_lc_tools()]
        check("register_subagent_tool() 登记成功", registered)
        check(
            "注册表 +1",
            len(names_after) == len(names_before) + 1,
            f"{len(names_before)} -> {len(names_after)}",
        )
        check(
            "to_lc_tools() 末尾是 subagent_task",
            names_after[-1] == SUBAGENT_TOOL_NAME,
            ", ".join(names_after),
        )
        # 假工具只服务冒烟：登记在 subagent 之后，避免干扰“注册表 +1”的计数
        registry.register("echo_note", echo_note, _fake_spec("echo_note"))
        registry.register("slow_note", slow_note, _fake_spec("slow_note"))
        check(
            "schema 指纹已变化（新工具真的进了模型可见列表）",
            _schema_sha(registry.to_lc_tools()) != sha_before,
        )

        _section("3. 调用一次：只回摘要 + 主会话消息数")
        settings.ENABLE_TOOL_POLICY = True
        ledger = MemoryBudgetLedger()
        executor.set_default_ledger(ledger)
        SCRIPT["mode"] = "normal"

        main_thread = "smoke-main-thread"
        values = await _run_main_call(main_thread)
        messages = list(values.get("messages") or [])
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        summary = str(tool_messages[0].content) if tool_messages else ""

        from src.context_manager import estimate_tokens

        check(
            "摘要 ≤ SUBAGENT_MAX_OUTPUT_TOKENS",
            estimate_tokens(summary) <= settings.SUBAGENT_MAX_OUTPUT_TOKENS,
            f"{estimate_tokens(summary)} <= {settings.SUBAGENT_MAX_OUTPUT_TOKENS} tokens",
        )
        check("摘要来自子代理的最后一条 AI 消息", "子代理结论" in summary, summary[:60])
        check(
            "主会话消息数 = 4（用户 1 + AI 工具调用 1 + 工具结果 1 + 最终回答 1）",
            len(messages) == 4,
            f"{len(messages)} 条",
        )
        check(
            "主会话只有 1 条工具结果（子代理的中间消息没有回传）",
            len(tool_messages) == 1,
            f"{len(tool_messages)} 条",
        )
        main_text = "\n".join(str(m.content) for m in messages)
        check(
            "上下文隔离：子代理内部 echo 结果没有作为消息进入主会话",
            "echo:子代理第一步" not in main_text
            and all(str(m.content) != "echo:子代理第二步" for m in messages),
            f"子代理内部最多看到 {OBSERVED['sub_messages']} 条消息，主会话只有摘要",
        )
        check(
            "子代理确实跑了多步（内部消息 > 主会话贡献的 2 条）",
            OBSERVED["sub_messages"] >= 6,
            f"子代理内部 {OBSERVED['sub_messages']} 条 vs 主会话新增 3 条",
        )

        long_text = "开头结论" + "中间的调查过程。" * 400 + "结尾结论"
        truncated = _truncate_summary(long_text, 120)
        check(
            "超长摘要被截断到上限内（保留首尾 + 中间标记）",
            estimate_tokens(truncated) <= 120
            and truncated.startswith("开头结论")
            and truncated.endswith("结尾结论")
            and "…（已截断）" in truncated,
            f"{estimate_tokens(truncated)} <= 120 tokens",
        )

        _section("4. 预算：子 scope vs 主 scope")
        main_snapshot = ledger.snapshot(main_thread)
        sub_scopes = sorted(_sub_scopes(ledger))
        sub_calls = sum(ledger.snapshot(scope).calls for scope in sub_scopes)
        check(
            "主 scope 只记 1 次调用（subagent_task 本身）",
            main_snapshot.calls == 1,
            f"calls={main_snapshot.calls}",
        )
        check(
            "子代理有独立 scope（thread_id=subagent-…）",
            len(sub_scopes) == 1,
            ", ".join(sub_scopes) or "(无)",
        )
        check("子代理的两步工具调用记在子 scope", sub_calls == 2, f"calls={sub_calls}")

        _section("5. 递归守卫：子代理里再调 subagent_task")
        SCRIPT["mode"] = "guard"
        guard_result = await subagent_task("在子代理里再套一层子代理")
        check(
            "返回“子代理不可嵌套”",
            "子代理不可嵌套" in guard_result,
            guard_result[:80],
        )
        check("没有递归下去（能正常拿到最终结论）", "GUARD_SAW" in guard_result)

        _section("6. 超时：SUBAGENT_TIMEOUT_SECONDS=1 跑 3 秒任务")
        SCRIPT["mode"] = "timeout"
        original_timeout = settings.SUBAGENT_TIMEOUT_SECONDS
        settings.SUBAGENT_TIMEOUT_SECONDS = 1
        started = time.monotonic()
        timeout_result = await subagent_task("跑一个慢任务")
        elapsed = time.monotonic() - started
        settings.SUBAGENT_TIMEOUT_SECONDS = original_timeout
        check("返回可读超时错误（不抛栈）", "超时" in timeout_result, timeout_result[:80])
        check("1 秒左右就返回（没有等满 3 秒）", elapsed < 2.5, f"{elapsed:.2f}s")

        _section("7. 追踪：同一 trace_id + 父节点 = 调用它的 tool span")
        with tempfile.TemporaryDirectory(prefix="smoke-subagent-trace-") as trace_dir:
            import agent.observability.tracer as tracer_module

            tracer_module._default_tracer = None  # 换 TRACE_DIR 后重新建 tracer
            settings.ENABLE_TRACE = True
            settings.TRACE_DIR = trace_dir
            SCRIPT["mode"] = "normal"
            await _run_main_call("smoke-trace-thread")

            files = sorted(Path(trace_dir).glob("*.jsonl"))
            spans = [
                json.loads(line)
                for path in files
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for span in spans:
                print(
                    f"  span {span['kind']:5} {span['name']:18} "
                    f"id={span['span_id'][:8]} parent={(span['parent_span_id'] or '')[:8]}"
                )
            trace_ids = {span["trace_id"] for span in spans}
            check("所有 span 在同一 trace_id 下", len(trace_ids) == 1, ", ".join(trace_ids))
            sub_tool = [
                span
                for span in spans
                if span["kind"] == "tool" and span["name"] == SUBAGENT_TOOL_NAME
            ]
            check("存在调用 subagent_task 的 tool span", len(sub_tool) == 1)
            inner_tools = [
                span for span in spans if span["kind"] == "tool" and span["name"] == "echo_note"
            ]
            check("子代理内部工具 span 存在", len(inner_tools) == 2, f"{len(inner_tools)} 条")
            check(
                "子代理内部 span 的 parent_span_id = 调用它的 tool span",
                bool(sub_tool)
                and all(span["parent_span_id"] == sub_tool[0]["span_id"] for span in inner_tools),
                f"tool span={sub_tool[0]['span_id'][:8] if sub_tool else '?'}",
            )
            inner_llm = [
                span
                for span in spans
                if span["kind"] == "llm"
                and sub_tool
                and span["parent_span_id"] == sub_tool[0]["span_id"]
            ]
            check(
                "子代理的 llm span 同样挂在这条 tool span 下",
                len(inner_llm) >= 1,
                f"{len(inner_llm)} 条",
            )
            settings.ENABLE_TRACE = False
    finally:
        setattr(cg, "ChatOllama", original_model)  # noqa: B010
        for name in ("echo_note", "slow_note", SUBAGENT_TOOL_NAME):
            registry.TOOL_SPECS.pop(name, None)
            registry._TOOL_FNS.pop(name, None)
        settings.ENABLE_SUBAGENT = False
        settings.ENABLE_TOOL_POLICY = False
        settings.ENABLE_TRACE = False
        executor.set_default_ledger(None)

    _section("结果")
    print(f"PASS={_PASS} FAIL={_FAIL}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
