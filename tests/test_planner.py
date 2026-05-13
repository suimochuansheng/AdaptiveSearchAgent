"""Planner 节点单元测试。"""

import pytest

from src.agents.planner import planner
from tests.conftest import create_test_state


@pytest.mark.asyncio
async def test_planner_basic() -> None:
    state = create_test_state({"user_query": "GPU 性能对比", "iteration": 0})
    result = await planner(state)
    assert "plan" in result
    assert len(result["plan"]) > 0
