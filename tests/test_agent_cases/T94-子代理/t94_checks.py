#!/usr/bin/env python
"""G12 / T9-4 子代理工具 —— 测试窗口（test-G12）**独立验收**。

用途：
    不读开发窗口的冒烟脚本结论、不复用它的断言；本脚本自己搭“假模型 + 假工具”，
    从当前代码独立取证，验证子代理工具的语义与不变量。只读，不改业务代码。

覆盖（判分标准 A/B 必过，C 只记 open_issue）：
    A1  开关关：注册表 5 个、无 subagent_task、schema 哈希仍为 c48c9d48…2a99
    A2  开关开：注册表 6 个、to_lc_tools() 末尾是 subagent_task
    A3  上下文隔离：主会话只增 2 条（AI 工具调用 + 工具结果），
        子代理内部多轮消息逐条比对后确认没有一条进入主会话
    A4  摘要上限：长文场景返回值 <= SUBAGENT_MAX_OUTPUT_TOKENS，且带“已截断”标记
    A5  递归守卫：子代理内再调 subagent_task -> “子代理不可嵌套”，不递归、不超时
    A6  超时：SUBAGENT_TIMEOUT_SECONDS=1 + 慢任务 -> 1~2 秒内返回可读错误
    A7  预算 scope：主 scope calls==1；子 scope（独立 thread_id）记录内部工具调用次数
    A8  主会话历史不进子代理：主会话暗号在子代理侧完全不可见
    B1  pytest / collect / nodeids 与 docs/baseline.md §2 一致
    B2  lint-imports 3 kept
    B3  src/conversation_graph.py 的 add_node / add_edge 零改动
    B4  契约文档（docs/contracts/）与前端（frontend/）零改动
    B5  台账里没有 T94 行（开发窗口不该写台账）
    C1  trace 树：子代理内部 span 与本轮同 trace_id，父节点是 subagent_task tool span
    C2  并发两个 subagent_task：父 span 是否串（交接单声明的 v1 边界，确认现象并记 OI）
    C3  子代理内工具抛异常：主会话拿到可读摘要/错误，不是栈
    C4  连续两次子代理调用：子 scope 隔离还是复用
    C5  子代理产出空字符串：主会话是否出现“空工具结果”

退出码：
    0 = A1~A8 / B1~B5 全部通过（C 类只打 [C-PASS]/[C-ISSUE]，不影响退出码）
    1 = A/B 至少一项失败，或脚本自身异常中止

为什么不是 `test_*.py`：
    会被 pytest 收集，`--collect-only` 的收集数改变，`docs/baseline.md` §2 的
    nodeids 哈希立即失效（见 tests/test_agent_cases/README.md）。

只读承诺：
    不写仓库内业务文件；不 import 开发窗口的 scripts/smoke_subagent.py，也不读它的断言。
    副作用只有：pytest 的 `.pytest_cache/`（B1 取证用，已 gitignore）、临时目录里的
    trace JSONL（tempfile，跑完即删）。台账由测试窗口在本脚本全绿后另行追加（GB18030 / CRLF）。
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
for _p in (REPO, CASE_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _baseline  # noqa: E402  （tests/test_agent_cases/_baseline.py，非 dev 冒烟产物）
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import ConfigDict  # noqa: E402

from agent.adapters.tools import executor, registry  # noqa: E402
from agent.adapters.tools import subagent as subagent_module  # noqa: E402
from agent.adapters.tools.budget_memory import MemoryBudgetLedger  # noqa: E402
from agent.adapters.tools.subagent import (  # noqa: E402
    SUBAGENT_TOOL_NAME,
    register_subagent_tool,
    subagent_task,
)
from agent.domain.tool import CostModel, ToolSpec  # noqa: E402
from config import settings  # noqa: E402
from src.context_manager import estimate_tokens  # noqa: E402

# ── 冻结基线（docs/baseline.md §2 / T6-0 契约）───────────────────────────
BASELINE_NAMES = [
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
]
BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
#: 本卡开工 HEAD（交接单第 1 节）与验收 HEAD；B3/B4 的“零改动”以这段区间为准
RANGE_START = "9f2f1a5"
HEAD_COMMIT = "cbe3726"
LEDGER = REPO / "tests" / "test_agent_cases" / "台账用例.csv"
TRUNCATED_MARK = "…（已截断）"
#: 子代理初始消息前缀（实现契约，见交接单 §2；取模块常量，避免硬编码漂移）
GOAL_PREFIX = getattr(subagent_module, "_GOAL_PREFIX", "子任务目标：")

#: 子代理剧本：normal / long / guard / slow / boom / empty / amnesia
SUB_MODE: dict[str, str] = {"mode": "normal"}
#: 主会话剧本：batch=一次 AI 消息里并发几个 subagent_task；goal=子任务目标
MAIN_SCRIPT: dict[str, Any] = {"batch": 1, "goal": "统计探针观测结果，只回结论"}
#: 观测：子代理每次模型调用时看到的消息（角色名, 文本），用于“逐条比对”
SUB_SEEN: list[list[tuple[str, str]]] = []
#: 观测：主会话每次模型调用时看到的消息（角色名, 文本），用于 A8 反证
MAIN_SEEN: list[list[tuple[str, str]]] = []

#: 长文场景：约 8000 个 CJK 字符，远超默认 1500 token 上限
LONG_TEXT = "长摘要开头。" + "调查过程记录" * 1600 + "长摘要结尾。"

RESULTS: list[tuple[str, str, bool, str]] = []
C_PROBES: list[tuple[str, bool, str]] = []


# ── 打印与记录 ────────────────────────────────────────────────────────


def record(case_id: str, title: str, ok: bool, evidence: str) -> bool:
    """记录一条 A/B 判定并按统一格式打印。"""
    RESULTS.append((case_id, title, ok, evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {case_id} {title} — {evidence}", flush=True)
    return ok


def record_c(probe: str, ok: bool, evidence: str) -> None:
    """记录一条 C 类探针（不参与 RESULT 计数，也不影响退出码）。"""
    C_PROBES.append((probe, ok, evidence))
    print(f"[{'C-PASS' if ok else 'C-ISSUE'}] {probe} — {evidence}", flush=True)


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}", flush=True)


# ── 假模型 / 假工具 ───────────────────────────────────────────────────


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _tool_texts(messages: list[Any]) -> list[str]:
    return [str(message.content) for message in messages if isinstance(message, ToolMessage)]


class ScriptedChatModel(BaseChatModel):
    """按“主会话 / 子代理会话”分流剧本的假模型（不连 Ollama）。"""

    model_config = ConfigDict(extra="ignore")

    @property
    def _llm_type(self) -> str:
        return "t94-scripted-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
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
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    def _respond(self, messages: list[Any]) -> AIMessage:
        texts = [(type(message).__name__, str(message.content)) for message in messages]
        is_sub = any(
            isinstance(message, HumanMessage) and str(message.content).startswith(GOAL_PREFIX)
            for message in messages
        )
        if is_sub:
            SUB_SEEN.append(list(texts))
            return self._sub(messages)
        MAIN_SEEN.append(list(texts))
        return self._main(messages)

    def _sub(self, messages: list[Any]) -> AIMessage:
        handlers = {
            "long": self._sub_long,
            "guard": self._sub_guard,
            "slow": self._sub_slow,
            "boom": self._sub_boom,
            "empty": self._sub_empty,
            "amnesia": self._sub_amnesia,
        }
        return handlers.get(SUB_MODE["mode"], self._sub_normal)(messages)

    def _sub_long(self, messages: list[Any]) -> AIMessage:
        del messages
        return AIMessage(content=LONG_TEXT)

    def _sub_guard(self, messages: list[Any]) -> AIMessage:
        done = _tool_texts(messages)
        if not done:
            return _tool_call(SUBAGENT_TOOL_NAME, {"goal": "子代理里再委托一层"}, "t94-g-1")
        return AIMessage(content=f"GUARD_RESULT::{done[-1]}")

    def _sub_slow(self, messages: list[Any]) -> AIMessage:
        if not _tool_texts(messages):
            return _tool_call("slow_probe", {"delay": 5.0}, "t94-s-1")
        return AIMessage(content="慢任务完成")

    def _sub_boom(self, messages: list[Any]) -> AIMessage:
        done = _tool_texts(messages)
        if not done:
            return _tool_call("exploding_probe", {}, "t94-b-1")
        return AIMessage(content=f"子代理结论：{done[-1]}")

    def _sub_empty(self, messages: list[Any]) -> AIMessage:
        del messages
        return AIMessage(content="")

    def _sub_amnesia(self, messages: list[Any]) -> AIMessage:
        joined = "\n".join(str(message.content) for message in messages)
        if "香蕉" in joined:
            return AIMessage(content="子代理回答：暗号是香蕉价格 7 元。")
        return AIMessage(content="子代理回答：不知道暗号（没有收到主会话历史）。")

    def _sub_normal(self, messages: list[Any]) -> AIMessage:
        # normal：跑两步探针，证明子代理内部确实是多轮（非单次回显）
        done = _tool_texts(messages)
        if not done:
            return _tool_call("probe_note", {"tag": "SEED-A1"}, "t94-p-1")
        if len(done) == 1:
            return _tool_call("probe_note", {"tag": "SEED-B2"}, "t94-p-2")
        return AIMessage(content="子代理摘要：两次探针观测完成；关键事实 K=314。")

    def _main(self, messages: list[Any]) -> AIMessage:
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="主会话最终回答：已收到子代理摘要。")
        batch = int(MAIN_SCRIPT["batch"])
        goal = str(MAIN_SCRIPT["goal"])
        calls = [
            {
                "name": SUBAGENT_TOOL_NAME,
                "args": {"goal": goal, "constraints": "只回结论"},
                "id": f"t94-main-{index}",
            }
            for index in range(batch)
        ]
        return AIMessage(content="", tool_calls=calls)


async def probe_note(tag: str) -> str:
    """子代理内部假工具：返回带唯一种子标记的探针结果。"""
    return f"PROBE::{tag}::done"


async def slow_probe(delay: float = 5.0) -> str:
    """子代理内部慢工具（A6 超时用）。"""
    await asyncio.sleep(delay)
    return "SLOW::done"


async def exploding_probe() -> str:
    """子代理内部会抛异常的工具（C3 用）。"""
    raise RuntimeError("BOOM: 子代理内部工具异常")


def _probe_spec(name: str, timeout: float = 30.0) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="T94 验收假工具",
        input_schema={"properties": {}, "type": "object"},
        permission="read",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=timeout,
        max_output_tokens=800,
        requires_approval=False,
        feature_flag=None,
    )


# ── 小工具 ────────────────────────────────────────────────────────────


def tool_names(tools: list[Any]) -> list[str]:
    return [getattr(tool, "__name__", "?") for tool in tools]


def schema_hash(tools: list[Any]) -> str:
    """与基线同口径：OpenAI function schema 规范化后的 sha256。"""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schemas = [convert_to_openai_tool(tool) for tool in tools]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=120, check=False
    )
    return (proc.stdout + proc.stderr).strip()


def read_spans(directory: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                spans.append(json.loads(line))
    return spans


def set_main(goal: str, batch: int = 1) -> None:
    MAIN_SCRIPT["goal"] = goal
    MAIN_SCRIPT["batch"] = batch


async def run_graph(
    thread_id: str,
    user_text: str,
    *,
    collect_states: bool = False,
    trace: bool = False,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """跑一次主会话图，返回 (最终 messages, astream 的 values 快照列表)。"""
    from langgraph.checkpoint.memory import MemorySaver

    from src.conversation_graph import build_conversation_graph

    graph = build_conversation_graph(checkpointer=MemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    states: list[dict[str, Any]] = []

    if trace:
        from agent.observability.langchain_handler import build_callbacks
        from agent.observability.tracer import get_tracer
        from agent.observability.turn import turn_span

        tracer = get_tracer()
        config["callbacks"] = build_callbacks(tracer)
        trace_cm: Any = turn_span(tracer, thread_id, 0)
    else:
        trace_cm = nullcontext()

    payload = {"messages": [HumanMessage(content=user_text)]}
    async with trace_cm:
        if collect_states:
            async for values in graph.astream(payload, config=config, stream_mode="values"):
                states.append(dict(values))
        else:
            await graph.ainvoke(payload, config=config)

    snapshot = await graph.aget_state(config)
    messages = list((snapshot.values or {}).get("messages") or [])
    return messages, states


# ── A 类 ──────────────────────────────────────────────────────────────


def check_a1() -> None:
    """A1：开关关 -> 注册表 5 个、无 subagent_task、schema 哈希不变。"""
    tools = registry.to_lc_tools()
    names = tool_names(tools)
    digest = schema_hash(tools)
    ok = (
        names == BASELINE_NAMES
        and SUBAGENT_TOOL_NAME not in registry.TOOL_SPECS
        and SUBAGENT_TOOL_NAME not in names
        and digest == BASELINE_SHA256
        and settings.ENABLE_SUBAGENT is False
    )
    record(
        "T94-001",
        "A1 开关关：注册表 5 个 / 无 subagent_task / schema 哈希不变",
        ok,
        f"names={names}；flag={settings.ENABLE_SUBAGENT}；"
        f"sha256={digest[:8]}…（基线 {BASELINE_SHA256[:8]}…）",
    )


def check_a2() -> None:
    """A2：开关开 -> 注册表 6 个、to_lc_tools() 末尾是 subagent_task。"""
    settings.ENABLE_SUBAGENT = True
    registered = register_subagent_tool()
    tools = registry.to_lc_tools()
    names = tool_names(tools)
    ok = (
        registered
        and len(names) == len(BASELINE_NAMES) + 1
        and names[-1] == SUBAGENT_TOOL_NAME
        and SUBAGENT_TOOL_NAME in registry.TOOL_SPECS
    )
    record(
        "T94-002",
        "A2 开关开：注册表 6 个 / to_lc_tools() 末尾是 subagent_task",
        ok,
        f"register={registered}；{len(BASELINE_NAMES)}->{len(names)}；last={names[-1] if names else '?'}",
    )
    # 假工具在 A2 判定之后登记，避免干扰“+1”计数
    registry.register("probe_note", probe_note, _probe_spec("probe_note"))
    registry.register("slow_probe", slow_probe, _probe_spec("slow_probe"))
    registry.register("exploding_probe", exploding_probe, _probe_spec("exploding_probe"))


async def check_a3() -> None:
    """A3：主会话只增 2 条；子代理内部消息一条都不进主会话（逐条比对内容）。"""
    SUB_SEEN.clear()
    MAIN_SEEN.clear()
    SUB_MODE["mode"] = "normal"
    set_main("统计探针观测结果，只回结论")

    messages, states = await run_graph(
        "t94-a3-isolation", "请让子代理整理两条探针观测。", collect_states=True
    )
    sub_snapshot = [list(seen) for seen in SUB_SEEN]

    # 找“工具结果刚落地”的那个状态：相对初始用户消息只应多 2 条
    added: list[Any] | None = None
    for values in states:
        current = values.get("messages") or []
        if (
            added is None
            and len(current) >= 2
            and isinstance(current[-1], ToolMessage)
            and getattr(current[-1], "name", "") == SUBAGENT_TOOL_NAME
        ):
            added = list(current[1:])
    added = added or []

    shape_ok = (
        len(added) == 2
        and isinstance(added[0], AIMessage)
        and bool(added[0].tool_calls)
        and added[0].tool_calls[0].get("name") == SUBAGENT_TOOL_NAME
        and isinstance(added[1], ToolMessage)
        and getattr(added[1], "name", "") == SUBAGENT_TOOL_NAME
    )

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    main_contents = [str(message.content) for message in messages]
    summary = str(tool_messages[0].content) if tool_messages else ""

    # 逐条比对：子代理内部看过的每条非空消息，除“被判为摘要的那条结论”外，
    # 都不允许与主会话任何一条消息内容完全相同。
    sub_texts = [text for seen in sub_snapshot for _role, text in seen]
    leaked = [text for text in sub_texts if text and text != summary and text in main_contents]
    seed_leaked = any(
        token in content
        for content in main_contents
        for token in ("SEED-A1", "SEED-B2", "PROBE::", "probe_note")
    )
    multi_round = max((len(seen) for seen in sub_snapshot), default=0) >= 5
    isolation_ok = not leaked and not seed_leaked and multi_round
    ok = shape_ok and isolation_ok and len(messages) == 4 and len(tool_messages) == 1

    record(
        "T94-003",
        "A3 上下文隔离：主会话只增 2 条 / 子代理内部消息逐条比对不泄漏",
        ok,
        f"工具结果落地时新增={len(added)} 条"
        f"（{type(added[0]).__name__ if added else '-'}+{type(added[1]).__name__ if len(added) > 1 else '-'}）；"
        f"最终 {len(messages)} 条/工具结果 {len(tool_messages)} 条；子代理内部最多 {max((len(s) for s in sub_snapshot), default=0)} 条；"
        f"泄漏={leaked or '无'}；种子泄漏={seed_leaked}",
    )


async def check_a4() -> None:
    """A4：长文场景，子代理返回值 <= SUBAGENT_MAX_OUTPUT_TOKENS 且带截断标记。

    口径：`subagent_task()` 的**返回值**就是摘要契约（工具层）。主会话里还会经过
    T6-3 的 `tool_output_token_budget`（默认 1200）二次压缩，那是另一层机制，
    这里单独取证并记 open_issue，不计入本项失败。
    """
    SUB_MODE["mode"] = "long"
    set_main("让子代理产出一篇很长的调查报告")
    direct = await subagent_task("产出一篇很长的调查报告")
    messages, _ = await run_graph("t94-a4-summary", "请让子代理产出一篇长文。")
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    in_main = str(tool_messages[0].content) if tool_messages else ""
    cap = int(settings.SUBAGENT_MAX_OUTPUT_TOKENS)
    tokens = estimate_tokens(direct)
    ok = (
        tokens <= cap
        and TRUNCATED_MARK in direct
        and direct.startswith("长摘要开头")
        and direct.endswith("长摘要结尾。")
    )
    main_tokens = estimate_tokens(in_main)
    main_lost_marker = bool(in_main) and TRUNCATED_MARK not in in_main
    if main_lost_marker:
        print(
            "OPEN_ISSUE: OI-T94-001 子代理摘要上限 1500 > 主会话工具输出预算 1200，"
            f"进主会话时被二次压缩（{tokens} -> {main_tokens} tokens），截断标记与首尾保留丢失",
            flush=True,
        )
    record(
        "T94-004",
        "A4 摘要上限：返回值 <= SUBAGENT_MAX_OUTPUT_TOKENS 且有截断标记",
        ok,
        f"返回值 tokens={tokens} <= cap={cap}；标记={TRUNCATED_MARK in direct}；"
        f"首尾保留={direct.startswith('长摘要开头')}/{direct.endswith('长摘要结尾。')}；"
        f"原始长度={len(LONG_TEXT)} 字；进主会话后 tokens={main_tokens}、标记保留={not main_lost_marker}",
    )


async def check_a5() -> None:
    """A5：子代理内再调 subagent_task -> 可读拒绝，不递归、不超时；深度可复位。"""
    SUB_MODE["mode"] = "guard"
    started = time.monotonic()
    result = await subagent_task("在子代理里再套一层子代理")
    elapsed = time.monotonic() - started

    SUB_MODE["mode"] = "normal"
    after = await subagent_task("守卫之后跑一个正常子任务")
    depth_reset = "子代理摘要" in after

    ok = "子代理不可嵌套" in result and "GUARD_RESULT" in result and elapsed < 8 and depth_reset
    record(
        "T94-005",
        "A5 递归守卫：再调返回“子代理不可嵌套”且不递归",
        ok,
        f"返回「{result[:40]}…」；守卫后回归正常={depth_reset}；耗时 {elapsed:.2f}s",
    )


async def check_a6() -> None:
    """A6：SUBAGENT_TIMEOUT_SECONDS=1 + 慢任务 -> 1~2 秒内返回可读错误。"""
    SUB_MODE["mode"] = "slow"
    original = settings.SUBAGENT_TIMEOUT_SECONDS
    settings.SUBAGENT_TIMEOUT_SECONDS = 1
    try:
        started = time.monotonic()
        result = await subagent_task("跑一个 5 秒的慢任务")
        elapsed = time.monotonic() - started
    finally:
        settings.SUBAGENT_TIMEOUT_SECONDS = original
        SUB_MODE["mode"] = "normal"

    ok = "超时" in result and "Traceback" not in result and 0.8 <= elapsed <= 2.5
    record(
        "T94-006",
        "A6 超时：1 秒超时返回可读错误（不抛栈）",
        ok,
        f"返回「{result[:46]}…」；耗时 {elapsed:.2f}s（要求 1~2s 内）",
    )


async def check_a7() -> None:
    """A7：主 scope calls==1；子 scope（独立 thread_id）记录内部工具调用。"""
    ledger = MemoryBudgetLedger()
    executor.set_default_ledger(ledger)
    settings.ENABLE_TOOL_POLICY = True
    SUB_MODE["mode"] = "normal"
    set_main("统计探针观测结果，只回结论")

    await run_graph("t94-a7-budget", "请让子代理整理两条探针观测。")

    main_calls = ledger.snapshot("t94-a7-budget").calls
    data = getattr(ledger, "_data", {})
    sub_scopes = sorted(str(scope) for scope in data if str(scope).startswith("subagent-"))
    sub_calls = sum(ledger.snapshot(scope).calls for scope in sub_scopes)

    ok = main_calls == 1 and len(sub_scopes) == 1 and sub_calls == 2
    record(
        "T94-007",
        "A7 预算 scope：主 scope 1 次 / 子 scope 记内部 2 次",
        ok,
        f"main calls={main_calls}；sub_scope={sub_scopes or '无'}；sub calls={sub_calls}",
    )


async def check_a8() -> None:
    """A8：主会话暗号不进子代理（子代理侧消息里完全看不到暗号）。"""
    SUB_SEEN.clear()
    MAIN_SEEN.clear()
    SUB_MODE["mode"] = "amnesia"
    set_main("请复述用户刚才告诉你的暗号；如果不知道就明确说不知道。")

    messages, _ = await run_graph(
        "t94-a8-history",
        "请记住这个暗号：香蕉价格是 7 元。之后我会考你，先让子代理去记一下。",
    )
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    summary = str(tool_messages[0].content) if tool_messages else ""
    sub_texts = [text for seen in SUB_SEEN for _role, text in seen]
    main_texts = [text for seen in MAIN_SEEN for _role, text in seen]

    main_saw = any("香蕉" in text for text in main_texts)
    sub_saw = any("香蕉" in text for text in sub_texts)
    first_roles = SUB_SEEN[0] if SUB_SEEN else []
    first_humans = [text for role, text in first_roles if role == "HumanMessage"]
    first_is_goal_only = (
        len(first_humans) == 1
        and first_humans[0].startswith(GOAL_PREFIX)
        and all(role in {"SystemMessage", "HumanMessage"} for role, _ in first_roles)
    )

    ok = main_saw and not sub_saw and "不知道" in summary and first_is_goal_only
    record(
        "T94-008",
        "A8 主会话历史不进子代理：暗号在子代理侧不可见",
        ok,
        f"主会话侧看到暗号={main_saw}；子代理侧看到暗号={sub_saw}；"
        f"子代理首轮={[role for role, _ in first_roles]}（应只有 system+goal）；"
        f"摘要「{summary[:34]}…」",
    )


# ── C 类 ──────────────────────────────────────────────────────────────


async def check_c1() -> bool:
    """C1：trace 树同 trace_id，子代理 span 的父节点是 subagent_task tool span。"""
    with tempfile.TemporaryDirectory(prefix="t94-trace-") as trace_dir:
        import agent.observability.tracer as tracer_module

        settings.ENABLE_TRACE = True
        settings.TRACE_DIR = trace_dir
        tracer_module._default_tracer = None
        SUB_MODE["mode"] = "normal"
        set_main("统计探针观测结果，只回结论")
        try:
            await run_graph("t94-c1-trace", "请让子代理整理两条探针观测。", trace=True)
            spans = read_spans(trace_dir)
        finally:
            settings.ENABLE_TRACE = False
            tracer_module._default_tracer = None

    trace_ids = {span.get("trace_id") for span in spans}
    sub_tools = [
        span
        for span in spans
        if span.get("kind") == "tool" and span.get("name") == SUBAGENT_TOOL_NAME
    ]
    parent_id = sub_tools[0].get("span_id") if len(sub_tools) == 1 else None
    inner = [span for span in spans if parent_id and span.get("parent_span_id") == parent_id]
    inner_llm = [span for span in inner if span.get("kind") == "llm"]
    inner_tools = [span for span in inner if span.get("kind") == "tool"]
    ok = (
        len(spans) > 0
        and len(trace_ids) == 1
        and len(sub_tools) == 1
        and len(inner_llm) >= 1
        and any(span.get("name") == "probe_note" for span in inner_tools)
    )
    record_c(
        "T94-C1 trace 树",
        ok,
        f"spans={len(spans)}；trace_id 数={len(trace_ids)}；subagent tool span={len(sub_tools)}；"
        f"其下 llm={len(inner_llm)} / tool={len(inner_tools)}"
        f"（{sorted({str(span.get('name')) for span in inner_tools})}）",
    )
    return ok


async def check_c2() -> bool:
    """C2：并发两个 subagent_task 时父 span 是否串（v1 已知边界）。"""
    with tempfile.TemporaryDirectory(prefix="t94-trace-dual-") as trace_dir:
        import agent.observability.tracer as tracer_module

        settings.ENABLE_TRACE = True
        settings.TRACE_DIR = trace_dir
        tracer_module._default_tracer = None
        SUB_MODE["mode"] = "normal"
        set_main("统计探针观测结果，只回结论", batch=2)
        try:
            await run_graph("t94-c2-concurrent", "请并发让两个子代理各整理一份探针。", trace=True)
            spans = read_spans(trace_dir)
        finally:
            settings.ENABLE_TRACE = False
            tracer_module._default_tracer = None

    sub_tools = [
        span
        for span in spans
        if span.get("kind") == "tool" and span.get("name") == SUBAGENT_TOOL_NAME
    ]
    parents = {span.get("span_id") for span in sub_tools}
    inner = [span for span in spans if span.get("parent_span_id") in parents]
    inner_parents = {span.get("parent_span_id") for span in inner}
    crossed = len(sub_tools) == 2 and len(inner_parents) == 1
    ok = not crossed
    detail = (
        f"并发 subagent tool span={len(sub_tools)}；子代理内部 span={len(inner)}；"
        f"挂在 {len(inner_parents)} 条父 span 下"
        + ("（两个子代理的 span 串到同一条父节点 -> v1 边界复现）" if crossed else "（父节点分开）")
    )
    if crossed:
        print(
            "OPEN_ISSUE: OI-T94-002 同一轮并发两个 subagent_task 时，两个子代理的内部 span "
            "全部挂到同一条父 tool span（_active_tool_span 按“最后一条未结束”匹配）；"
            "交接单已声明为 v1 边界，单调用不受影响",
            flush=True,
        )
    record_c("T94-C2 并发父 span", ok, detail)
    return ok


async def check_c3() -> bool:
    """C3：子代理内工具抛异常 -> 主会话拿到可读摘要/错误，不是栈。"""
    SUB_MODE["mode"] = "boom"
    set_main("让子代理调用会失败的工具并汇报")
    settings.ENABLE_TOOL_POLICY = True
    messages, _ = await run_graph("t94-c3-boom", "请让子代理跑一次会失败的工具。")
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    text = str(tool_messages[0].content) if tool_messages else ""
    readable = ("出错" in text or "失败" in text) and "Traceback" not in text
    ok = bool(text.strip()) and readable
    record_c(
        "T94-C3 工具异常",
        ok,
        f"主会话工具结果「{text[:56]}…」；含栈={'Traceback' in text}；可读={'出错' in text or '失败' in text}",
    )
    return ok


async def check_c4() -> bool:
    """C4：连续两次子代理调用，子 scope 是否隔离。"""
    ledger = MemoryBudgetLedger()
    executor.set_default_ledger(ledger)
    settings.ENABLE_TOOL_POLICY = True
    SUB_MODE["mode"] = "normal"
    set_main("统计探针观测结果，只回结论")
    await run_graph("t94-c4-first", "第一次请让子代理整理探针。")
    await run_graph("t94-c4-second", "第二次请让子代理整理探针。")

    data = getattr(ledger, "_data", {})
    sub_scopes = sorted(str(scope) for scope in data if str(scope).startswith("subagent-"))
    calls = [ledger.snapshot(scope).calls for scope in sub_scopes]
    isolated = len(sub_scopes) == 2 and calls == [2, 2]
    record_c(
        "T94-C4 连续调用隔离",
        isolated,
        f"子 scope 数={len(sub_scopes)}；各自 calls={calls}（应为 2 个独立 scope、各 2 次）",
    )
    return isolated


async def check_c5() -> bool:
    """C5：子代理产出空字符串时，主会话是否拿到空工具结果。"""
    SUB_MODE["mode"] = "empty"
    set_main("让子代理给出一个空结论")
    messages, _ = await run_graph("t94-c5-empty", "请让子代理给结论。")
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    text = str(tool_messages[0].content) if tool_messages else ""
    final = str(messages[-1].content) if messages else ""
    ok = bool(text.strip()) and bool(final.strip())
    record_c(
        "T94-C5 空产出退化",
        ok,
        f"工具结果长度={len(text)}（{'非空兜底' if text.strip() else '空！'}）；"
        f"主会话最终回答={'有' if final.strip() else '无'}；内容「{text[:40]}…」",
    )
    return ok


# ── B 类 ──────────────────────────────────────────────────────────────


def check_b1() -> None:
    """B1：pytest 通过数 / 收集数 / nodeids 与 docs/baseline.md §2 一致。"""
    exp = _baseline.expected()
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    collect_line = ""
    for line in reversed(collect.stdout.splitlines()):
        if "collected" in line:
            collect_line = line.strip()
            break

    nodeids = REPO / ".pytest_cache" / "v" / "cache" / "nodeids"
    node_hash = "missing"
    node_lines = -1
    if nodeids.exists():
        raw = nodeids.read_bytes()
        node_hash = hashlib.sha256(raw).hexdigest()
        node_lines = raw.count(b"\n")

    full = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    tail = ""
    for line in reversed(full.stdout.splitlines()):
        if re.search(r"\d+ passed", line):
            tail = line.strip()
            break
    match = re.search(r"(\d+) passed", tail)
    passed = int(match.group(1)) if match else -1

    ok = (
        full.returncode == 0
        and passed == exp["pytest_passed"]
        and collect.returncode == 0
        and exp["collect"] in collect_line
        and node_hash == exp["nodeids_sha256"]
        and node_lines == exp["nodeids_lines"]
    )
    record(
        "T94-009",
        "B1 pytest / collect / nodeids 与 baseline.md 一致",
        ok,
        f"pytest 末行「{tail}」；collect「{collect_line}」；"
        f"nodeids {node_lines} 行 sha256={node_hash[:8]}…"
        f"（baseline: {exp['pytest_passed']} passed / {exp['collect']} / "
        f"{exp['nodeids_lines']} 行 / {exp['nodeids_sha256'][:8]}…）",
    )


def check_b2() -> None:
    """B2：lint-imports 3 kept。"""
    exe = shutil.which("lint-imports")
    if exe is None:
        record("T94-010", "B2 lint-imports 3 kept", False, "找不到 lint-imports")
        return
    proc = subprocess.run([exe], cwd=REPO, capture_output=True, text=True, timeout=180, check=False)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
    match = re.search(r"Contracts:\s*(\d+) kept,\s*(\d+) broken", plain)
    kept, broken = (match.group(1), match.group(2)) if match else ("?", "?")
    ok = proc.returncode == 0 and (kept, broken) == ("3", "0")
    record(
        "T94-010",
        "B2 lint-imports 3 kept",
        ok,
        f"exit={proc.returncode}；Contracts: {kept} kept, {broken} broken",
    )


def check_b3() -> None:
    """B3：src/conversation_graph.py 的 add_node / add_edge 零改动。"""
    path = REPO / "src" / "conversation_graph.py"
    text = path.read_text(encoding="utf-8")
    structure_ok = all(
        marker in text
        for marker in (
            'graph.add_node("agent", agent_node)',
            'graph.add_node("tools", tools_node)',
            'graph.add_edge(START, "agent")',
            'graph.add_edge("tools", "agent")',
        )
    )
    range_diff = git(
        "diff", "--name-only", f"{RANGE_START}..HEAD", "--", "src/conversation_graph.py"
    )
    work_diff = git("diff", "--name-only", "--", "src/conversation_graph.py")
    status = git("status", "--porcelain", "--", "src/conversation_graph.py")
    ok = structure_ok and not range_diff and not work_diff and not status
    record(
        "T94-011",
        "B3 图结构（add_node / add_edge）零改动",
        ok,
        f"节点/边原样={structure_ok}；区间 diff='{range_diff}'；工作树 diff='{work_diff}'；"
        f"status='{status}'",
    )


def check_b4() -> None:
    """B4：契约文档与前端零改动。"""
    changed = [
        line for line in git("diff", "--name-only", f"{RANGE_START}..HEAD").splitlines() if line
    ]
    forbidden = [
        path
        for path in changed
        if path.startswith("docs/contracts/") or path.startswith("frontend/")
    ]
    dirty = git("status", "--porcelain", "--", "docs/contracts", "frontend")
    ok = not forbidden and not dirty
    record(
        "T94-012",
        "B4 契约文档 / 前端零改动",
        ok,
        f"区间改动={changed or '无'}；其中契约/前端={forbidden or '无'}；工作树='{dirty}'",
    )


def check_b5() -> None:
    """B5：台账里没有**开发窗口**写的 T94 行（测试窗口自己的行不算）。

    口径与 T93 一致：T94 行可以存在，但 `run_by` 必须是 `test-G12`（测试窗口），
    且开工区间内动过台账的提交只能是 `test(cases)`。这样脚本在台账追加后仍可复跑。
    """
    if not LEDGER.exists():
        record("T94-013", "B5 台账无开发窗口的 T94 行", False, f"台账不存在：{LEDGER}")
        return
    with LEDGER.open(encoding="gb18030", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    t94 = [row for row in rows if (row.get("case_id") or "").startswith("T94")]
    foreign = [row.get("case_id") for row in t94 if (row.get("run_by") or "").strip() != "test-G12"]
    commits = [
        line
        for line in git(
            "log", "--format=%s", f"{RANGE_START}..HEAD", "--", str(LEDGER.relative_to(REPO))
        ).splitlines()
        if line
    ]
    foreign_commit = [subject for subject in commits if not subject.startswith("test(cases)")]
    ok = not foreign and not foreign_commit
    record(
        "T94-013",
        "B5 台账无开发窗口的 T94 行",
        ok,
        f"台账共 {len(rows)} 行；T94 行={len(t94)}；非 test-G12 写的={foreign or '无'}；"
        f"开工后动过台账的提交={commits or '无'}",
    )


# ── 主流程 ────────────────────────────────────────────────────────────


async def guarded(case_id: str, title: str, fn: Any) -> None:
    """跑一个 A/B 检查：异常转成 [FAIL]，不让一项问题吞掉整份报告。"""
    try:
        result = fn()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - 验收脚本要指出“哪项失败”，不是打栈
        record(case_id, title, False, f"脚本异常：{type(exc).__name__}: {exc}")


async def guarded_c(probe: str, fn: Any) -> bool:
    """跑一个 C 类探针：异常记 [C-ISSUE]，不判失败。"""
    try:
        return bool(await fn())
    except Exception as exc:  # noqa: BLE001
        record_c(probe, False, f"脚本异常：{type(exc).__name__}: {exc}")
        return False


def cleanup(cg: Any, original_model: Any) -> None:
    """还原进程内状态：模型类、开关、账本、假工具。"""
    setattr(cg, "ChatOllama", original_model)  # noqa: B010
    settings.ENABLE_SUBAGENT = False
    settings.ENABLE_TOOL_POLICY = False
    settings.ENABLE_TRACE = False
    executor.set_default_ledger(None)
    for name in ("probe_note", "slow_probe", "exploding_probe", SUBAGENT_TOOL_NAME):
        registry.TOOL_SPECS.pop(name, None)
        registry._TOOL_FNS.pop(name, None)


async def main() -> int:
    from src import conversation_graph as cg

    started = time.monotonic()
    original_model = cg.ChatOllama
    setattr(cg, "ChatOllama", ScriptedChatModel)  # noqa: B010 - 验收里临时替换模型类
    try:
        section("A1 / A2：开关与注册表")
        check_a1()
        check_a2()

        section("A3：上下文隔离")
        await guarded("T94-003", "A3 上下文隔离", check_a3)

        section("A4 / A5 / A6：摘要上限 / 递归守卫 / 超时")
        await guarded("T94-004", "A4 摘要上限", check_a4)
        await guarded("T94-005", "A5 递归守卫", check_a5)
        await guarded("T94-006", "A6 超时", check_a6)

        section("A7 / A8：预算 scope / 历史隔离")
        await guarded("T94-007", "A7 预算 scope", check_a7)
        await guarded("T94-008", "A8 历史不进子代理", check_a8)

        section("C1 / C2：trace 树与并发父 span")
        c1_ok = await guarded_c("T94-C1 trace 树", check_c1)
        c2_ok = await guarded_c("T94-C2 并发父 span", check_c2)
        record(
            "T94-014",
            "C1+C2 trace 树 / 并发父 span（边界汇总，不判失败）",
            True,
            f"C1={'同 trace 且父节点正确' if c1_ok else '见 OPEN_ISSUE'}；"
            f"C2={'父节点分开' if c2_ok else '两个子代理 span 串父节点（v1 边界复现）'}",
        )

        section("C3 / C4 / C5：异常 / 连续调用 / 空产出")
        c3_ok = await guarded_c("T94-C3 工具异常", check_c3)
        c4_ok = await guarded_c("T94-C4 连续调用隔离", check_c4)
        c5_ok = await guarded_c("T94-C5 空产出退化", check_c5)
        record(
            "T94-015",
            "C3+C4+C5 异常 / 连续调用 / 空产出（边界汇总，不判失败）",
            True,
            f"C3={'可读错误' if c3_ok else '见 OPEN_ISSUE'}；"
            f"C4={'隔离' if c4_ok else '见 OPEN_ISSUE'}；"
            f"C5={'非空兜底' if c5_ok else '见 OPEN_ISSUE'}",
        )

        section("B1 ~ B5：不变量")
        await guarded("T94-009", "B1 pytest/collect/nodeids", check_b1)
        await guarded("T94-010", "B2 lint-imports", check_b2)
        await guarded("T94-011", "B3 图结构零改动", check_b3)
        await guarded("T94-012", "B4 契约/前端零改动", check_b4)
        await guarded("T94-013", "B5 台账无开发窗口的 T94 行", check_b5)
    finally:
        cleanup(cg, original_model)

    failed = [case_id for case_id, _, ok, _ in RESULTS if not ok]
    passed = len(RESULTS) - len(failed)
    issues = [probe for probe, ok, _ in C_PROBES if not ok]

    print("-" * 78)
    print(
        f"耗时 {time.monotonic() - started:.1f}s；A/B {len(RESULTS)} 项；C 类探针 {len(C_PROBES)} 项"
    )
    if failed:
        print(f"失败：{', '.join(failed)}")
    print(f"C 类 open_issue：{', '.join(issues) if issues else '无'}")
    print(f"RESULT: {passed} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
