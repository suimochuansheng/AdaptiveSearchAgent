"""Search Worker 节点单元测试。"""

import pytest

from src.agents.search_worker import search_worker
from tests.conftest import create_test_state


@pytest.mark.asyncio
async def test_search_worker() -> None:
    state = create_test_state({"plan": ["Python 异步编程"]})
    result = await search_worker(state)
    assert "search_results" in result
    assert len(result["search_results"]) == 1
    assert "content" in result["search_results"][0]
