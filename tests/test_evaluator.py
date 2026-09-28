"""Evaluator 节点单元测试。"""

import pytest

from src.agents.evaluator import evaluator
from tests.conftest import create_test_state

# @pytest.mark.asyncio
# async def test_evaluator() -> None:
#     state = create_test_state(
#         {
#             "user_query": "LangGraph 是什么",
#             "search_results": [{"keyword": "LangGraph", "content": "LangGraph 是..."}],
#         }
#     )
#     result = await evaluator(state)
#     assert "confidence_score" in result
#     assert 0.0 <= result["confidence_score"] <= 1.0


@pytest.mark.asyncio
async def test_evaluator(monkeypatch):
    async def fake_llm(prompt, config):
        content = '{"confidence_score": 0.85, "missing_info": "", "retry_keywords": []}'
        return (content, 10, 20, 30)

    monkeypatch.setattr("src.agents.evaluator.llm_call_with_fallback", fake_llm)
    state = create_test_state(
        {
            "user_query": "LangGraph 是什么",
            "search_results": [{"keyword": "LangGraph", "content": "LangGraph 是..."}],
        }
    )
    result = await evaluator(state)
    assert "confidence_score" in result
    assert 0.0 <= result["confidence_score"] <= 1.0
