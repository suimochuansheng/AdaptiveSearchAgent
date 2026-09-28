"""Planner 节点单元测试。"""

import pytest

from src.agents.planner import planner
from tests.conftest import create_test_state


@pytest.mark.asyncio
async def test_planner_basic(monkeypatch) -> None:
    async def fake_llm(prompt, config):
        content = '{"plan": ["GPU 性能", "GPU 架构"]}'
        return (content, 10, 20, 30)

    monkeypatch.setattr("src.agents.planner.llm_call_with_fallback", fake_llm)
    state = create_test_state({"user_query": "GPU 性能对比", "iteration": 0})
    result = await planner(state)
    assert "plan" in result
    assert len(result["plan"]) > 0
