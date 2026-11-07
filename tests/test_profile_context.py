"""跨会话画像注入测试（阶段 C3a）——只测"读"，不依赖 PG / LLM。

回归点：
- 画像进稳定区（system）而不是历史；
- 一轮只读一次 DB，之后用 state 缓存；
- 关闭开关 / 读失败 / 默认画像 都不改变行为（可回滚）。
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.memory.profile_context import format_profile


class _FakeStore:
    def __init__(
        self, profile: dict[str, Any] | None = None, error: Exception | None = None
    ) -> None:
        self.profile = profile or {}
        self.error = error
        self.calls = 0

    async def get_profile(self, user_id: str) -> dict[str, Any]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.profile


def test_format_profile_renders_preferences() -> None:
    text = format_profile(
        {"preferred_language": "en", "answer_style": "concise", "recent_topics": ["上下文", "评测"]}
    )

    assert "English" in text
    assert "精简" in text
    assert "上下文" in text and "评测" in text


def test_default_profile_is_empty() -> None:
    """全是默认值 = 没有画像，不占稳定区预算。"""
    assert format_profile({"preferred_language": "zh", "answer_style": "detailed"}) == ""


def test_format_profile_respects_budget() -> None:
    profile = {
        "recent_topics": [f"话题{i}" for i in range(200)],
    }

    text = format_profile(profile, token_budget=80)

    from src.context_manager import estimate_tokens

    assert estimate_tokens(text) <= 80 + 120


async def test_load_profile_context_degrades_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.memory import profile_context as pc

    monkeypatch.setattr(pc.settings, "MEMORY_ENABLED", True)
    monkeypatch.setattr(
        "src.memory.user_profile_store.get_user_profile_store",
        _fake_store_factory(_FakeStore(error=RuntimeError("db down"))),
    )

    assert await pc.load_profile_context("u-1") == ""


async def test_load_profile_context_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.memory import profile_context as pc

    monkeypatch.setattr(pc.settings, "MEMORY_ENABLED", False)
    store = _FakeStore({"recent_topics": ["x"]})
    monkeypatch.setattr(
        "src.memory.user_profile_store.get_user_profile_store", _fake_store_factory(store)
    )

    assert await pc.load_profile_context("u-1") == ""
    assert store.calls == 0


async def test_agent_node_injects_profile_once_into_system(monkeypatch: pytest.MonkeyPatch) -> None:
    """画像进 system（稳定区），且第二轮不再读 DB。"""
    import src.conversation_graph as cg
    from config import settings

    monkeypatch.setattr(settings, "MEMORY_ENABLED", True)
    store = _FakeStore({"recent_topics": ["上下文压缩"], "answer_style": "table"})
    monkeypatch.setattr(
        "src.memory.user_profile_store.get_user_profile_store", _fake_store_factory(store)
    )

    seen_systems: list[str] = []

    class _Model:
        def __init__(self, **_: Any) -> None:
            pass

        def bind_tools(self, _tools: Any) -> "_Model":
            return self

        async def ainvoke(self, messages: Any) -> Any:
            seen_systems.append(str(messages[0].content))
            return AIMessage("好的")

    monkeypatch.setattr(cg, "ChatOllama", _Model)
    from langgraph.checkpoint.memory import InMemorySaver

    graph = cg.build_conversation_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-1"}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage("第一轮")], "user_id": "u-1"}, config=config
    )
    assert "上下文压缩" in seen_systems[0]
    assert result["profile_context"], "画像应写回 state 供后续轮次复用"
    assert store.calls == 1

    await graph.ainvoke({"messages": [HumanMessage("第二轮")]}, config=config)
    assert store.calls == 1, "第二轮不应再查 DB"


def _fake_store_factory(store: _FakeStore) -> Any:
    async def _factory() -> _FakeStore:
        return store

    return _factory
