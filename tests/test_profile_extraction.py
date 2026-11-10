"""画像抽取测试（阶段 C3b）——注入假模型与假存储，不依赖 LLM / DB。

覆盖：prompt 组装、JSON 解析容错、写库、失败降级、每 N 轮触发一次（且同一轮不重复入队）。
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.memory.profile_extraction import (
    build_extraction_prompt,
    extract_and_store,
    parse_profile_payload,
)


class _FakeStore:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []

    async def upsert_profile(
        self,
        user_id: str,
        preferred_language: str | None = None,
        answer_style: str | None = None,
        recent_topics: list[str] | None = None,
    ) -> None:
        self.upserts.append(
            {
                "user_id": user_id,
                "preferred_language": preferred_language,
                "answer_style": answer_style,
                "recent_topics": recent_topics,
            }
        )


def _llm(reply: str):
    async def _call(prompt: str) -> str:
        _call.prompt = prompt  # type: ignore[attr-defined]
        return reply

    return _call


def test_parse_handles_fenced_json() -> None:
    payload = parse_profile_payload(
        '```json\n{"preferred_language": "en", "answer_style": "table", '
        '"recent_topics": ["上下文", "评测"]}\n```'
    )

    assert payload == {
        "preferred_language": "en",
        "answer_style": "table",
        "recent_topics": ["上下文", "评测"],
    }


def test_parse_drops_invalid_values() -> None:
    payload = parse_profile_payload(
        '{"preferred_language": "fr", "answer_style": "verbose", "recent_topics": 3}'
    )

    assert payload == {}


def test_parse_returns_empty_on_garbage() -> None:
    assert parse_profile_payload("这不是 JSON") == {}


def test_parse_caps_topics() -> None:
    payload = parse_profile_payload('{"recent_topics": ["a","b","c","d","e","f","g"]}')

    assert len(payload["recent_topics"]) == 5


def test_prompt_includes_summary_and_recent_messages() -> None:
    messages = [HumanMessage(f"第{i}轮问题") for i in range(30)]

    prompt = build_extraction_prompt(messages, summary="之前讨论了成本", max_messages=5)

    assert "之前讨论了成本" in prompt
    assert "第29轮问题" in prompt
    assert "第0轮问题" not in prompt  # 只取尾部，控制成本


async def test_extract_and_store_writes_parsed_fields() -> None:
    store = _FakeStore()
    llm = _llm('{"preferred_language": "zh", "answer_style": "table", "recent_topics": ["成本"]}')

    payload = await extract_and_store(
        llm_fn=llm, store=store, user_id="u-1", messages=[HumanMessage("用表格对比")]
    )

    assert payload["answer_style"] == "table"
    assert store.upserts == [
        {
            "user_id": "u-1",
            "preferred_language": "zh",
            "answer_style": "table",
            "recent_topics": ["成本"],
        }
    ]


async def test_extract_failure_is_silent() -> None:
    """模型挂了：不写库、不抛异常（抽取是增强，不是依赖）。"""
    store = _FakeStore()

    async def _boom(prompt: str) -> str:
        raise RuntimeError("model down")

    payload = await extract_and_store(
        llm_fn=_boom, store=store, user_id="u-1", messages=[HumanMessage("hi")]
    )

    assert payload == {}
    assert store.upserts == []


async def test_extract_skips_empty_conversation() -> None:
    store = _FakeStore()
    payload = await extract_and_store(llm_fn=_llm("{}"), store=store, user_id="u-1", messages=[])

    assert payload == {}
    assert store.upserts == []


# ── 触发逻辑：每 N 轮入队一次，同一轮不重复 ──────────────────────────────


async def test_extraction_is_enqueued_every_n_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.conversation_graph as cg
    from config import settings

    monkeypatch.setattr(settings, "MEMORY_ENABLED", True)
    monkeypatch.setattr(settings, "MEMORY_EXTRACT_EVERY_TURNS", 10)

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "src.tasks_memory.enqueue_profile_extraction",
        lambda thread_id, user_id: calls.append((thread_id, user_id)),
    )

    class _Model:
        def __init__(self, **_: Any) -> None:
            pass

        def bind_tools(self, _tools: Any) -> "_Model":
            return self

        async def ainvoke(self, _messages: Any) -> Any:
            return AIMessage("好的")

    monkeypatch.setattr(cg, "ChatOllama", _Model)
    graph = cg.build_conversation_graph()
    config = {"configurable": {"thread_id": "t-9"}}

    history = [HumanMessage(f"第{i}轮") for i in range(9)]
    result = await graph.ainvoke(
        {"messages": [*history, HumanMessage("第10轮")], "user_id": "u-9"}, config=config
    )
    assert calls == [("t-9", "u-9")]
    assert result["profile_extract_upto"] == 10

    # 同一轮再来一次 agent 执行（模拟工具循环）→ 不应重复入队
    await graph.ainvoke({"messages": [HumanMessage("第10轮")], "user_id": "u-9"}, config=None)
    assert len(calls) == 1
