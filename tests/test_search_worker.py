"""Search Worker 及相关并行搜索节点单元测试。

覆盖模块：
- search_worker         — 单关键词搜索节点（Send API 驱动，返回 _search_accum）
- parallel_searcher     — 并行搜索分发节点（写入 _batch_keywords，返回 dict）
- route_to_search_workers — 条件边路由函数（生成 Send 扇出）
- get_llm_semaphore     — 全局限流信号量（单例模式）
- limited_llm_call      — 带并发限制的 LLM 调用包装器

注：接口已于重构后演进 —— parallel_searcher 不再直接返回 list[Send]，
而是写入 state["_batch_keywords"]，由 route_to_search_workers 生成 Send；
search_worker 返回 {"_search_accum": [...]}（经自定义 reducer 累积），
最终由 evaluator 合并进 search_results。
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langgraph.types import Send

from config import settings
from src.agents.parallel_searcher import parallel_searcher, route_to_search_workers
from src.agents.search_worker import search_worker
from src.utils.llm_factory import get_llm_semaphore, limited_llm_call
from tests.conftest import create_test_state

# ============================================================================
# search_worker 测试 — 单关键词搜索节点
# ============================================================================


class TestSearchWorker:
    """测试重构后的 search_worker 节点：接收单个 keyword，返回 _search_accum。"""

    @pytest.mark.asyncio
    async def test_搜索单个关键词返回正确结构(self) -> None:
        """验证 search_worker 对单个关键词的正确搜索流程。

        前置条件：state 包含 "keyword" 字段（由 route_to_search_workers 通过 Send 传入）。
        预期结果：返回的 _search_accum 是单元素列表，含 keyword 和 content。
        """
        state: dict = {
            "user_query": "",
            "iteration": 0,
            "plan": [],
            "search_results": [],
            "confidence_score": 0.0,
            "missing_info": "",
            "retry_keywords": [],
            "final_report": "",
            "human_approved": False,
            "task_id": "test-001",
            "total_tokens": 0,
            "pending_keywords": [],
            "keyword": "Python 异步编程",  # Send API 注入的字段，不在 TypedDict 定义中
        }

        # Mock Tavily 搜索，避免真实网络调用
        with patch(
            "src.agents.search_worker.search_tavily",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = "Python 异步编程是一种并发编程范式..."

            result = await search_worker(state)

        # 验证返回结构：新接口使用 _search_accum
        assert "_search_accum" in result
        assert len(result["_search_accum"]) == 1
        assert result["_search_accum"][0]["keyword"] == "Python 异步编程"
        assert "Python 异步编程是一种并发编程范式" in result["_search_accum"][0]["content"]

    @pytest.mark.asyncio
    async def test_搜索节点缺少keyword字段时报错(self) -> None:
        """验证 state 中缺少必需的 "keyword" 字段时抛出 KeyError。

        前置条件：state 不包含 "keyword" 键。
        预期结果：search_worker 在访问 state["keyword"] 时抛出 KeyError。
        """
        state = create_test_state({"task_id": "test-no-keyword"})
        # create_test_state 不包含 "keyword" 字段

        with pytest.raises(KeyError):
            await search_worker(state)

    @pytest.mark.asyncio
    async def test_搜索结果为空字符串时正常返回(self) -> None:
        """验证 Tavily 返回空内容时，search_worker 仍能正常处理。

        边界场景：搜索不存在的概念时 Tavily 可能返回空字符串。
        预期结果：返回正常的结构化结果，content 存在。
        """
        state: dict = {
            "user_query": "",
            "iteration": 0,
            "plan": [],
            "search_results": [],
            "confidence_score": 0.0,
            "missing_info": "",
            "retry_keywords": [],
            "final_report": "",
            "human_approved": False,
            "task_id": "test-empty",
            "total_tokens": 0,
            "pending_keywords": [],
            "keyword": "nonexistent_xyz_123",
        }

        with patch(
            "src.agents.search_worker.search_tavily",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = "未找到关于 'nonexistent_xyz_123' 的相关信息。"

            result = await search_worker(state)

        assert len(result["_search_accum"]) == 1
        assert result["_search_accum"][0]["keyword"] == "nonexistent_xyz_123"
        # content 存在即可，不要求非空
        assert "content" in result["_search_accum"][0]


# ============================================================================
# parallel_searcher 测试 — 并行搜索分发节点（返回 dict）
# ============================================================================


class TestParallelSearcher:
    """测试 parallel_searcher 节点的关键词分发、去重和分批逻辑。

    重构后该节点返回 {"_batch_keywords": [...], "pending_keywords": [...]}，
    不再直接返回 list[Send]；Send 扇出由 route_to_search_workers 完成。
    """

    @pytest.mark.asyncio
    async def test_从plan中取关键词并分发(self) -> None:
        """验证 parallel_searcher 在无 pending/retry 时使用 plan 作为关键词来源。

        前置条件：state 包含 plan 但没有 pending_keywords 和 retry_keywords。
        预期结果：_batch_keywords 包含 plan 中的全部关键词。
        """
        state = create_test_state(
            {
                "plan": ["Qwen2.5 性能", "Qwen2.5 架构"],
                "pending_keywords": [],
                "retry_keywords": [],
            }
        )

        result = await parallel_searcher(state)

        assert isinstance(result, dict)
        assert result["_batch_keywords"] == ["Qwen2.5 性能", "Qwen2.5 架构"]
        assert result["pending_keywords"] == []

    @pytest.mark.asyncio
    async def test_优先使用pending_keywords(self) -> None:
        """验证 pending_keywords 优先级高于 retry_keywords 和 plan。

        前置条件：state 同时包含 pending_keywords、retry_keywords 和 plan。
        预期结果：仅使用 pending_keywords 分发，忽略 retry 和 plan。
        """
        state = create_test_state(
            {
                "pending_keywords": ["待处理关键词"],
                "retry_keywords": ["重试关键词"],
                "plan": ["初始关键词"],
            }
        )

        result = await parallel_searcher(state)

        assert result["_batch_keywords"] == ["待处理关键词"]

    @pytest.mark.asyncio
    async def test_次优先使用retry_keywords(self) -> None:
        """验证当 pending_keywords 为空时，使用 retry_keywords 分发。

        前置条件：state 包含 retry_keywords 但没有 pending_keywords。
        预期结果：使用 retry_keywords 作为分发来源。
        """
        state = create_test_state(
            {
                "pending_keywords": [],
                "retry_keywords": ["RTX 5090 功耗", "RTX 5090 TDP"],
                "plan": ["RTX 5090"],
            }
        )

        result = await parallel_searcher(state)

        keywords = result["_batch_keywords"]
        assert "RTX 5090 功耗" in keywords
        assert "RTX 5090 TDP" in keywords

    @pytest.mark.asyncio
    async def test_去重已搜索过的关键词(self) -> None:
        """验证 parallel_searcher 自动排除 search_results 中已有的关键词。

        前置条件：plan 包含关键词 A、B，但 A 已出现在 search_results 中。
        预期结果：只分发关键词 B，A 被过滤掉。
        """
        state = create_test_state(
            {
                "plan": ["Python", "Golang"],
                "search_results": [{"keyword": "Python", "content": "Python 是..."}],
            }
        )

        result = await parallel_searcher(state)

        # Python 应被去重过滤，只保留 Golang
        keywords = result["_batch_keywords"]
        assert "Golang" in keywords
        assert "Python" not in keywords

    @pytest.mark.asyncio
    async def test_所有关键词已搜索时返回空批次(self) -> None:
        """验证当所有关键词都已被搜索过时，返回空 _batch_keywords。

        前置条件：plan 中的所有关键词都已存在于 search_results 中。
        预期结果：_batch_keywords 为空，且 pending_keywords 被清空。
        """
        state = create_test_state(
            {
                "plan": ["Python"],
                "search_results": [{"keyword": "Python", "content": "Python 是..."}],
            }
        )

        result = await parallel_searcher(state)

        assert result["_batch_keywords"] == []
        # pending_keywords 被清空，防止后续死循环
        assert result["pending_keywords"] == []

    @pytest.mark.asyncio
    async def test_分批机制超量关键词存入pending(self) -> None:
        """验证当关键词数量超过 max_concurrent_searches 时，溢出的存入 pending_keywords。

        前置条件：待搜索关键词数（limit+2）> settings.max_concurrent_searches（limit）。
        预期结果：
        - 当前批次 _batch_keywords 包含 limit 个
        - 剩余 2 个写入 pending_keywords
        """
        limit = settings.max_concurrent_searches
        keywords = [f"K{i}" for i in range(limit + 2)]
        state = create_test_state(
            {
                "plan": keywords,
                "pending_keywords": [],
            }
        )

        result = await parallel_searcher(state)

        batch_keywords = result["_batch_keywords"]
        assert len(batch_keywords) == limit
        assert batch_keywords == keywords[:limit]
        assert result["pending_keywords"] == keywords[limit:]

    @pytest.mark.asyncio
    async def test_空关键词列表返回空(self) -> None:
        """验证当所有来源的关键词都为空时，返回空 _batch_keywords。

        前置条件：plan、retry_keywords、pending_keywords 均为空列表。
        预期结果：_batch_keywords 为空，pending_keywords 保持为空。
        """
        state = create_test_state(
            {
                "plan": [],
                "retry_keywords": [],
                "pending_keywords": [],
            }
        )

        result = await parallel_searcher(state)

        assert result["_batch_keywords"] == []
        assert result["pending_keywords"] == []


# ============================================================================
# route_to_search_workers 测试 — Send 扇出路由函数
# ============================================================================


class TestRouteToSearchWorkers:
    """测试 route_to_search_workers 路由函数（重构后新增的 Send 生成器）。"""

    def test_有批次关键词时返回Send对象列表(self) -> None:
        """验证 _batch_keywords 非空时返回合法的 Send 列表。"""
        state = create_test_state({"_batch_keywords": ["关键词A", "关键词B"]})

        result = route_to_search_workers(state)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Send) for s in result)
        assert result[0].node == "search_worker"
        assert result[0].arg == {"keyword": "关键词A"}
        assert result[1].arg == {"keyword": "关键词B"}

    def test_空批次关键词时返回evaluator(self) -> None:
        """验证 _batch_keywords 为空时返回 "evaluator" 字符串（跳过搜索）。"""
        state = create_test_state({"_batch_keywords": []})

        result = route_to_search_workers(state)

        assert result == "evaluator"


# ============================================================================
# get_llm_semaphore / limited_llm_call 测试 — LLM 限流机制
# ============================================================================


class TestLLMSemaphore:
    """测试全局限流信号量的创建、复用和并发限制行为。"""

    def test_获取信号量返回单例(self) -> None:
        """验证 get_llm_semaphore 多次调用返回同一个 Semaphore 实例。"""
        sem1 = get_llm_semaphore()
        sem2 = get_llm_semaphore()

        assert sem1 is sem2

    def test_信号量默认限制为2(self) -> None:
        """验证 Semaphore 的默认并发限制为 2。"""
        sem = get_llm_semaphore()
        assert sem._value == 2

    @pytest.mark.asyncio
    async def test_limited_llm_call传递正确的prompt(self) -> None:
        """验证 limited_llm_call 将 prompt 正确传递给 LLM 并返回纯文本内容。"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "这是 LLM 的回复文本"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await limited_llm_call(mock_llm, "请生成搜索关键词")

        mock_llm.ainvoke.assert_awaited_once_with("请生成搜索关键词")
        assert result == "这是 LLM 的回复文本"

    @pytest.mark.asyncio
    async def test_limited_llm_call处理非字符串content(self) -> None:
        """验证当 LLM 返回的 content 不是字符串时，自动转为字符串。"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = [{"type": "text", "text": "结构化回复"}]
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await limited_llm_call(mock_llm, "测试")

        assert isinstance(result, str)
        assert "结构化回复" in result

    @pytest.mark.asyncio
    async def test_并发调用受信号量限制(self) -> None:
        """验证多个并发的 limited_llm_call 受 Semaphore 限制。"""
        import src.utils.llm_factory as llm_fac

        llm_fac._llm_semaphore = asyncio.Semaphore(2)

        concurrent_count = 0
        max_concurrent_observed = 0

        async def mock_invoke(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent_observed
            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            mock_resp = Mock()
            mock_resp.content = "ok"
            return mock_resp

        mock_llm = Mock()
        mock_llm.ainvoke = mock_invoke

        tasks = [limited_llm_call(mock_llm, f"prompt_{i}") for i in range(3)]
        await asyncio.gather(*tasks)

        assert max_concurrent_observed <= 2
        assert max_concurrent_observed >= 2

        llm_fac._llm_semaphore = None


# ============================================================================
# search_worker 与 parallel_searcher 集成场景测试
# ============================================================================


class TestParallelSearchIntegration:
    """测试 parallel_searcher → route_to_search_workers → search_worker 的端到端流程。"""

    @pytest.mark.asyncio
    async def test_分发后搜索节点正常执行(self) -> None:
        """模拟 LangGraph Send 机制：分发的 Send 载荷能被 search_worker 正确消费。

        流程：
        1. parallel_searcher 写入 _batch_keywords
        2. route_to_search_workers 生成 Send 列表
        3. 每个 Send 的 arg 作为 state 传给 search_worker
        4. search_worker 返回 _search_accum
        """
        state = create_test_state(
            {
                "plan": ["LangGraph 并行搜索"],
                "task_id": "integration-test",
            }
        )

        # Step 1: 分发（写入 _batch_keywords）
        update = await parallel_searcher(state)
        state.update(update)  # type: ignore[typeddict-item]

        # Step 2: 路由生成 Send
        sends = route_to_search_workers(state)
        assert isinstance(sends, list)
        assert len(sends) == 1

        # Step 3: 模拟 LangGraph 将 Send.arg 作为 state 传给 search_worker
        worker_state = {**state, **sends[0].arg}

        with patch(
            "src.agents.search_worker.search_tavily",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = "LangGraph 支持通过 Send API 实现并行搜索..."

            search_result = await search_worker(worker_state)

        # Step 4: 验证结果能被 LangGraph 的自定义 reducer 累积
        assert len(search_result["_search_accum"]) == 1
        assert search_result["_search_accum"][0]["keyword"] == "LangGraph 并行搜索"
        assert "Send API" in search_result["_search_accum"][0]["content"]


# ============================================================================
# AgentState 新增 pending_keywords 字段验证
# ============================================================================


class TestPendingKeywordsState:
    """验证 state.py 新增的 pending_keywords 字段在各组件中正确工作。"""

    def test_create_test_state_包含pending_keywords(self) -> None:
        """验证测试夹具 create_test_state 默认包含 pending_keywords 字段。"""
        state = create_test_state()
        assert "pending_keywords" in state
        assert state["pending_keywords"] == []

    def test_pending_keywords可通过overrides覆盖(self) -> None:
        """验证 create_test_state 的 overrides 参数可覆盖 pending_keywords。"""
        state = create_test_state({"pending_keywords": ["待处理1", "待处理2"]})
        assert state["pending_keywords"] == ["待处理1", "待处理2"]
