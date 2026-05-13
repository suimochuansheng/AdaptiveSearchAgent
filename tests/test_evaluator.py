"""Evaluator 节点单元测试。"""

import pytest

from src.agents.evaluator import evaluator
from tests.conftest import create_test_state


@pytest.mark.asyncio
async def test_evaluator() -> None:
    state = create_test_state(
        {
            "user_query": "LangGraph 是什么",
            "search_results": [{"keyword": "LangGraph", "content": "LangGraph 是..."}],
        }
    )
    result = await evaluator(state)
    assert "confidence_score" in result
    assert 0.0 <= result["confidence_score"] <= 1.0
