"""历史压缩测试（阶段 B2）——用假模型，不依赖真实 LLM。

覆盖：短历史不动、长历史缩窗 + 摘要、增量摘要不重复归纳、摘要失败退化、
预算内裁剪、关闭开关回退、至少保留最后一条消息。
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.context_manager import (
    build_context,
    estimate_tokens,
    fallback_summary,
    message_tokens,
    plan_window,
    summarize_history,
)


class _FakeLLM:
    """记录收到的 prompt，返回固定摘要。"""

    def __init__(
        self, reply: str = "决定：用 A 方案；未解决：无；关键数字：1200 万元；用户要求：要中文"
    ) -> None:
        self.prompts: list[list[Any]] = []
        self._reply = reply
        self.error: Exception | None = None

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.prompts.append(messages)
        if self.error is not None:
            raise self.error
        return AIMessage(content=self._reply)


def _history(count: int, *, chars: int = 120) -> list[Any]:
    """构造交替的对话历史，每条约 chars 个字符。"""
    messages: list[Any] = []
    for i in range(count):
        text = f"第{i}条：这是用于测试上下文窗口的对话内容，" + "填充" * (chars // 2)
        messages.append(HumanMessage(content=text) if i % 2 == 0 else AIMessage(content=text))
    return messages


async def test_short_history_is_passed_through() -> None:
    llm = _FakeLLM()
    messages = _history(3, chars=20)

    built = await build_context(
        llm=llm,
        messages=messages,
        system_prompt="SYSTEM",
        history_budget=3000,
        summary_budget=800,
    )

    assert built.summarized is False
    assert built.summary_upto == 0
    assert len(built.messages) == 1 + len(messages)
    assert isinstance(built.messages[0], SystemMessage)
    assert built.messages[0].content == "SYSTEM"
    assert llm.prompts == []  # 没有触发摘要调用


async def test_long_history_shrinks_window_and_summarizes() -> None:
    llm = _FakeLLM()
    messages = _history(40, chars=120)  # 约 4800 tokens，超 3000 预算

    built = await build_context(
        llm=llm,
        messages=messages,
        system_prompt="SYSTEM",
        history_budget=3000,
        summary_budget=800,
    )

    assert built.summarized is True
    assert 0 < built.summary_upto < len(messages)
    assert len(llm.prompts) == 1
    # 稳定区 + 摘要 + 最近原文
    assert isinstance(built.messages[0], SystemMessage)
    assert "早期摘要" in str(built.messages[1].content)
    kept = built.messages[2:]
    assert kept == messages[built.summary_upto :]
    assert estimate_tokens(str(built.summary)) <= 800 + 80


async def test_incremental_summary_reuses_previous() -> None:
    llm = _FakeLLM()
    messages = _history(40, chars=120)

    built = await build_context(
        llm=llm,
        messages=messages,
        system_prompt="SYSTEM",
        summary="旧摘要：用户决定用 B 方案",
        summary_upto=10,
        history_budget=3000,
        summary_budget=800,
    )

    prompt_text = str(llm.prompts[0][1].content)
    assert "旧摘要：用户决定用 B 方案" in prompt_text  # 旧摘要参与合并
    assert "【新增对话】" in prompt_text
    assert "第10条" in prompt_text  # 只从 summary_upto 之后开始
    assert "第0条" not in prompt_text  # 已归纳过的旧消息不重复送
    assert built.summary_upto > 10


async def test_summary_failure_falls_back_to_rules() -> None:
    llm = _FakeLLM()
    llm.error = RuntimeError("model timeout")
    messages = _history(40, chars=120)

    built = await build_context(
        llm=llm,
        messages=messages,
        system_prompt="SYSTEM",
        history_budget=3000,
        summary_budget=800,
    )

    # 不抛异常，且有兜底摘要（规则摘要会带上原文片段）
    assert built.summarized is True
    assert built.summary
    assert "第0条" in built.summary


async def test_no_llm_still_compresses_with_rules() -> None:
    messages = _history(20, chars=120)

    built = await build_context(
        llm=None,
        messages=messages,
        system_prompt="SYSTEM",
        history_budget=1000,
        summary_budget=800,
    )

    assert built.summarized is True
    assert built.summary
    assert built.messages[-1] == messages[-1]


async def test_disabled_keeps_full_history() -> None:
    llm = _FakeLLM()
    messages = _history(40, chars=120)

    built = await build_context(
        llm=llm,
        messages=messages,
        system_prompt="SYSTEM",
        history_budget=3000,
        summary_budget=800,
        enabled=False,
    )

    assert built.summarized is False
    assert built.summary_upto == 0
    assert len(built.messages) == 1 + len(messages)
    assert llm.prompts == []


def test_plan_window_keeps_at_least_last_message() -> None:
    messages = _history(5, chars=200)

    assert plan_window(messages, token_budget=0) == len(messages) - 1
    assert plan_window([], token_budget=100) == 0


def test_fallback_summary_keeps_numbers() -> None:
    summary = fallback_summary([HumanMessage(content="预算 1200 万元，A 方案胜出，交付周期 45 天")])

    assert "1200" in summary
    assert "45" in summary


def test_message_tokens_counts_tool_calls() -> None:
    plain = AIMessage(content="ok")
    with_calls = AIMessage(
        content="ok",
        tool_calls=[{"name": "deep_research", "args": {"query": "对比 A 和 B 的功耗"}, "id": "1"}],
    )

    assert message_tokens(with_calls) > message_tokens(plain)


async def test_summary_is_trimmed_to_budget() -> None:
    llm = _FakeLLM(reply="很长的摘要。" * 800)
    messages = _history(40, chars=120)

    built = await build_context(
        llm=llm,
        messages=messages,
        system_prompt="SYSTEM",
        history_budget=3000,
        summary_budget=200,
    )

    assert estimate_tokens(built.summary) <= 200 + 120


async def test_summarize_history_without_messages_returns_previous() -> None:
    llm = _FakeLLM()

    assert await summarize_history(llm, [], previous_summary="旧") == "旧"
    assert llm.prompts == []
