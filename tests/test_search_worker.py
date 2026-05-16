"""Search Worker 及相关并行搜索节点单元测试。

覆盖模块：
- search_worker         — 单关键词搜索节点（重构后由 Send API 驱动）
- parallel_searcher     — 并行搜索分发节点（关键词分批 + 去重）
- get_llm_semaphore     — 全局限流信号量（单例模式）
- limited_llm_call      — 带并发限制的 LLM 调用包装器
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langgraph.types import Send

from config import settings
from src.agents.parallel_searcher import parallel_searcher
from src.agents.search_worker import search_worker
from src.utils.llm_factory import get_llm_semaphore, limited_llm_call
from tests.conftest import create_test_state

# ============================================================================
# search_worker 测试 — 单关键词搜索节点
# ============================================================================


class TestSearchWorker:
    """测试重构后的 search_worker 节点：接收单个 keyword，返回单条搜索结果。"""

    @pytest.mark.asyncio
    async def test_搜索单个关键词返回正确结构(self) -> None:
        """验证 search_worker 对单个关键词的正确搜索流程。

        前置条件：state 包含 "keyword" 字段（由 parallel_searcher 通过 Send 传入）。
        预期结果：返回的 search_results 是单元素列表，含 keyword 和 content。
        """
        # 构造包含 keyword 的 state（模拟 Send API 传入的载荷）
        # 注意：Send API 会合并全局 state + Send.arg，因此 keyword 不在 AgentState TypedDict 定义中
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

        # 验证返回结构
        assert "search_results" in result
        assert len(result["search_results"]) == 1
        assert result["search_results"][0]["keyword"] == "Python 异步编程"
        assert "Python 异步编程是一种并发编程范式" in result["search_results"][0]["content"]

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
        预期结果：返回正常的结构化结果，content 为空字符串。
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

        assert len(result["search_results"]) == 1
        assert result["search_results"][0]["keyword"] == "nonexistent_xyz_123"
        # content 存在即可，不要求非空
        assert "content" in result["search_results"][0]


# ============================================================================
# parallel_searcher 测试 — 并行搜索分发节点
# ============================================================================


class TestParallelSearcher:
    """测试 parallel_searcher 节点的关键词分发、去重和分批逻辑。"""

    @pytest.mark.asyncio
    async def test_从plan中取关键词并分发(self) -> None:
        """验证 parallel_searcher 在无 pending/retry 时使用 plan 作为关键词来源。

        前置条件：state 包含 plan 但没有 pending_keywords 和 retry_keywords。
        预期结果：为 plan 中的每个关键词生成对应的 Send 任务。
        """
        state = create_test_state(
            {
                "plan": ["Qwen2.5 性能", "Qwen2.5 架构"],
                "pending_keywords": [],
                "retry_keywords": [],
            }
        )

        result = await parallel_searcher(state)

        # 验证返回 Send 列表
        assert isinstance(result, list)
        assert len(result) == 2
        # 验证每个元素是 Send 对象
        assert all(isinstance(s, Send) for s in result)
        # 验证 Send 的目标节点和载荷
        assert result[0].node == "search_worker"
        assert result[0].arg == {"keyword": "Qwen2.5 性能"}
        assert result[1].arg == {"keyword": "Qwen2.5 架构"}

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

        assert len(result) == 1
        assert result[0].arg["keyword"] == "待处理关键词"

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

        assert len(result) == 2
        keywords = [s.arg["keyword"] for s in result]
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
        keywords = [s.arg["keyword"] for s in result]
        assert "Golang" in keywords
        assert "Python" not in keywords

    @pytest.mark.asyncio
    async def test_所有关键词已搜索时返回空列表(self) -> None:
        """验证当所有关键词都已被搜索过时，返回空 Send 列表。

        前置条件：plan 中的所有关键词都已存在于 search_results 中。
        预期结果：返回空列表 []，且 pending_keywords 被清空。
        """
        state = create_test_state(
            {
                "plan": ["Python"],
                "search_results": [{"keyword": "Python", "content": "Python 是..."}],
            }
        )

        result = await parallel_searcher(state)

        assert result == []
        # 验证 pending_keywords 被清空，防止后续死循环
        assert state["pending_keywords"] == []

    @pytest.mark.asyncio
    async def test_分批机制超量关键词存入pending(self) -> None:
        """验证当关键词数量超过 max_concurrent_searches 时，溢出的存入 pending_keywords。

        前置条件：待搜索关键词数（5个） > settings.max_concurrent_searches（3个）。
        预期结果：
        - 当前批次分发 3 个 Send 任务
        - 剩余 2 个关键词写入 state["pending_keywords"]
        """
        limit = settings.max_concurrent_searches
        # 生成 limit + 2 个关键词，确保必然超出限制
        keywords = [f"K{i}" for i in range(limit + 2)]
        state = create_test_state(
            {
                "plan": keywords,
                "pending_keywords": [],
            }
        )

        result = await parallel_searcher(state)

        # 当前批次应包含 limit 个（= max_concurrent_searches 实际值）
        batch_keywords = [s.arg["keyword"] for s in result]
        assert len(batch_keywords) == limit
        assert batch_keywords == keywords[:limit]

        # 剩余 2 个应存入 pending_keywords
        assert state["pending_keywords"] == keywords[limit:]

    @pytest.mark.asyncio
    async def test_空关键词列表返回空(self) -> None:
        """验证当所有来源的关键词都为空时，返回空 Send 列表。

        前置条件：plan、retry_keywords、pending_keywords 均为空列表。
        预期结果：返回 []，pending_keywords 保持为空。
        """
        state = create_test_state(
            {
                "plan": [],
                "retry_keywords": [],
                "pending_keywords": [],
            }
        )

        result = await parallel_searcher(state)

        assert result == []
        assert state["pending_keywords"] == []

    @pytest.mark.asyncio
    async def test_返回的是Send对象列表(self) -> None:
        """验证 parallel_searcher 返回的每个元素都是合法的 LangGraph Send 对象。

        验证 Send 对象的两个关键属性：node（目标节点名）和 arg（载荷字典）。
        """
        state = create_test_state({"plan": ["关键词A"]})

        result = await parallel_searcher(state)

        assert len(result) == 1
        send_obj = result[0]
        # Send 是 LangGraph 内置类型，用于并行分发任务
        assert isinstance(send_obj, Send)
        assert send_obj.node == "search_worker"
        assert isinstance(send_obj.arg, dict)
        assert "keyword" in send_obj.arg


# ============================================================================
# get_llm_semaphore / limited_llm_call 测试 — LLM 限流机制
# ============================================================================


class TestLLMSemaphore:
    """测试全局限流信号量的创建、复用和并发限制行为。"""

    def test_获取信号量返回单例(self) -> None:
        """验证 get_llm_semaphore 多次调用返回同一个 Semaphore 实例。

        全局唯一的限流闸机：整个程序只应存在一个 Semaphore，
        所有 LLM 调用共用它来协调并发。
        """
        sem1 = get_llm_semaphore()
        sem2 = get_llm_semaphore()

        # 两次调用应返回同一个对象（单例模式）
        assert sem1 is sem2

    def test_信号量默认限制为2(self) -> None:
        """验证 Semaphore 的默认并发限制为 2。

        asyncio.Semaphore(2) 表示最多允许 2 个协程同时持有锁，
        多余的调用会被阻塞等待。
        """
        sem = get_llm_semaphore()
        # Semaphore._value 表示当前可用的许可数
        assert sem._value == 2

    @pytest.mark.asyncio
    async def test_limited_llm_call传递正确的prompt(self) -> None:
        """验证 limited_llm_call 将 prompt 正确传递给 LLM 并返回纯文本内容。

        使用 Mock LLM 模拟 ainvoke 响应，验证：
        1. LLM 收到正确的 prompt 参数
        2. 返回的内容被正确提取
        """
        # 构造一个 Mock LLM
        mock_llm = Mock()
        # 模拟 ainvoke 返回一个 AIMessage 对象
        mock_response = Mock()
        mock_response.content = "这是 LLM 的回复文本"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await limited_llm_call(mock_llm, "请生成搜索关键词")

        # 验证 prompt 被正确传递
        mock_llm.ainvoke.assert_awaited_once_with("请生成搜索关键词")
        # 验证返回纯文本内容
        assert result == "这是 LLM 的回复文本"

    @pytest.mark.asyncio
    async def test_limited_llm_call处理非字符串content(self) -> None:
        """验证当 LLM 返回的 content 不是字符串时，自动转为字符串。

        某些 LLM provider 可能返回 list[dict] 等复杂类型。
        limited_llm_call 内有 isinstance 守卫来兜底。
        """
        mock_llm = Mock()
        mock_response = Mock()
        # 模拟返回 list 类型的 content（某些多模态模型的返回格式）
        mock_response.content = [{"type": "text", "text": "结构化回复"}]
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await limited_llm_call(mock_llm, "测试")

        # 应被 str() 转换为字符串
        assert isinstance(result, str)
        assert "结构化回复" in result

    @pytest.mark.asyncio
    async def test_并发调用受信号量限制(self) -> None:
        """验证多个并发的 limited_llm_call 受 Semaphore 限制。

        启动 3 个并发任务，Semaphore limit=2：
        - 同时最多 2 个在执行中
        - 第 3 个必须等待前序完成才能获得锁
        """
        # 重置信号量以确保测试隔离（直接操作全局变量，测试用）
        import src.utils.llm_factory as llm_fac

        llm_fac._llm_semaphore = asyncio.Semaphore(2)

        # 用一个计数器跟踪同时执行的协程数
        concurrent_count = 0
        max_concurrent_observed = 0

        async def mock_invoke(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent_observed
            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
            # 短暂等待模拟真实 LLM 调用延迟
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            mock_resp = Mock()
            mock_resp.content = "ok"
            return mock_resp

        mock_llm = Mock()
        mock_llm.ainvoke = mock_invoke

        # 同时启动 3 个 limited_llm_call 任务
        tasks = [limited_llm_call(mock_llm, f"prompt_{i}") for i in range(3)]
        await asyncio.gather(*tasks)

        # 最大并发数应不超过 Semaphore 限制 2
        assert max_concurrent_observed <= 2
        # 至少有一次并发（因为有 3 个任务，limit=2，必然有并发）
        assert max_concurrent_observed >= 2

        # 清理：重置信号量，避免影响其他测试
        llm_fac._llm_semaphore = None


# ============================================================================
# search_worker 与 parallel_searcher 集成场景测试
# ============================================================================


class TestParallelSearchIntegration:
    """测试 parallel_searcher 分发后 search_worker 能正确执行的端到端流程。"""

    @pytest.mark.asyncio
    async def test_分发后搜索节点正常执行(self) -> None:
        """模拟 LangGraph Send 机制：parallel_searcher 分发的 Send 载荷能被 search_worker 正确消费。

        流程：
        1. parallel_searcher 生成 Send 列表
        2. 每个 Send 的 arg 作为 state 传给 search_worker
        3. search_worker 返回单条搜索结果
        """
        state = create_test_state(
            {
                "plan": ["LangGraph 并行搜索"],
                "task_id": "integration-test",
            }
        )

        # Step 1: 分发
        sends = await parallel_searcher(state)
        assert len(sends) == 1

        # Step 2: 模拟 LangGraph 将 Send.arg 作为 state 传给 search_worker
        worker_state = {**state, **sends[0].arg}

        with patch(
            "src.agents.search_worker.search_tavily",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = "LangGraph 支持通过 Send API 实现并行搜索..."

            search_result = await search_worker(worker_state)

        # Step 3: 验证结果能被 LangGraph 的 operator.add 累积
        assert len(search_result["search_results"]) == 1
        assert search_result["search_results"][0]["keyword"] == "LangGraph 并行搜索"
        assert "Send API" in search_result["search_results"][0]["content"]


# ============================================================================
# AgentState 新增 pending_keywords 字段验证
# ============================================================================


class TestPendingKeywordsState:
    """验证 state.py 新增的 pending_keywords 字段在各组件中正确工作。"""

    def test_create_test_state_包含pending_keywords(self) -> None:
        """验证测试夹具 create_test_state 默认包含 pending_keywords 字段。

        这是支撑所有 parallel_searcher 测试的基础。
        """
        state = create_test_state()
        assert "pending_keywords" in state
        assert state["pending_keywords"] == []

    def test_pending_keywords可通过overrides覆盖(self) -> None:
        """验证 create_test_state 的 overrides 参数可覆盖 pending_keywords。"""
        state = create_test_state({"pending_keywords": ["待处理1", "待处理2"]})
        assert state["pending_keywords"] == ["待处理1", "待处理2"]
