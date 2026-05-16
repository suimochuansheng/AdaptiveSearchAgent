"""M2 里程碑测试：并行搜索分发 + 多 Wave 循环 + 去重 + 并发控制。

覆盖 M2 核心功能：
- should_continue_wave  — 波次/重试/结束决策
- build_graph           — 并行图编译验证
- 多 Wave 端到端流程    — 分批 → 波次循环 → 重试 → 结束
- 去重跨波验证          — 已搜关键词不在后续波次中重复
- 迭代计数语义          — 仅 planner 重试时递增，波次间不变
"""

from typing import TYPE_CHECKING

# 仅用于类型检查的导入，运行时不会执行
if TYPE_CHECKING:
    from src.state import AgentState

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.graph.state import CompiledStateGraph

from config import settings
from main import build_graph, should_continue_wave
from src.agents.parallel_searcher import parallel_searcher
from src.agents.search_worker import search_worker
from src.utils.metrics import metrics
from tests.conftest import create_test_state

# ============================================================================
# should_continue_wave 单元测试 — 波次/重试/结束决策
# ============================================================================


class TestShouldContinueWave:
    """测试 should_continue_wave 条件边决策函数的三个分支。"""

    def test_pending非空返回parallel_searcher(self) -> None:
        """验证当 pending_keywords 非空时，优先返回 parallel_searcher 继续下一波。

        前置条件：state 包含待处理的关键词。
        预期结果：返回 "parallel_searcher"，不触发 iteration 递增和 metrics 记录。
        """
        state = create_test_state(
            {
                "iteration": 0,
                "pending_keywords": ["K4", "K5"],
                "confidence_score": 0.5,
            }
        )
        # 保存初始 iteration 值用于验证不变
        initial_iteration = state["iteration"]

        result = should_continue_wave(state)

        assert result == "parallel_searcher"
        # pending 路径不应改变 iteration
        assert state["iteration"] == initial_iteration

    def test_pending空且置信度达标返回writer(self) -> None:
        """验证当 pending 为空且置信度达标时，返回 writer 并记录 metrics。

        前置条件：pending 为空，confidence >= threshold。
        预期结果：返回 "writer"，调用 metrics.record_iterations()。
        """
        state = create_test_state(
            {
                "iteration": 2,
                "pending_keywords": [],
                "confidence_score": 0.85,  # >= 0.8 threshold
            }
        )

        original_metrics_len = len(metrics.iterations_per_query)

        result = should_continue_wave(state)

        assert result == "writer"
        # 验证 metrics 记录被调用（iterations_per_query 增加了）
        assert len(metrics.iterations_per_query) == original_metrics_len + 1
        assert metrics.iterations_per_query[-1] == 2  # 记录了 iteration=2

    def test_pending空且达最大迭代返回writer(self) -> None:
        """验证当 pending 为空且 iteration 达到上限时，即使置信度低也返回 writer。

        前置条件：pending 为空，iteration >= max_iterations，但 confidence 很低。
        预期结果：返回 "writer"（强制结束）。
        """
        state = create_test_state(
            {
                "iteration": settings.max_iterations,
                "pending_keywords": [],
                "confidence_score": 0.1,  # 远低于阈值
            }
        )

        result = should_continue_wave(state)

        assert result == "writer"

    def test_pending空且低置信度返回planner并递增iteration(self) -> None:
        """验证当 pending 为空且不满足结束条件时，返回 planner 并递增 iteration。

        前置条件：pending 为空，confidence < threshold，iteration < max。
        预期结果：返回 "planner"，iteration 从 1 递增到 2。
        """
        state = create_test_state(
            {
                "iteration": 1,
                "pending_keywords": [],
                "confidence_score": 0.3,
            }
        )

        result = should_continue_wave(state)

        assert result == "planner"
        # iteration 应在原值基础上 +1
        assert state["iteration"] == 2

    def test_波次循环不触发iteration递增(self) -> None:
        """验证多次 pending 波次循环过程中 iteration 保持不变。

        模拟场景：第一波搜索后 evaluator 评估不达标，但 pending 仍有剩余，
        应回到 parallel_searcher 而非 planner，iteration 不应增加。

        前置条件：pending 非空（模拟波间状态）。
        预期结果：返回 "parallel_searcher"，iteration 不变。
        """
        state = create_test_state(
            {
                "iteration": 0,
                "pending_keywords": ["K4", "K5"],
                "confidence_score": 0.4,
            }
        )

        result = should_continue_wave(state)
        assert result == "parallel_searcher"
        assert state["iteration"] == 0  # 波次不递增

        # 模拟第二波结束：pending 已清空
        state["pending_keywords"] = []
        state["confidence_score"] = 0.6

        result = should_continue_wave(state)
        assert result == "planner"
        assert state["iteration"] == 1  # 只在回 planner 时递增


# ============================================================================
# build_graph 编译验证
# ============================================================================


class TestBuildGraph:
    """验证 build_graph() 输出的图结构正确性。"""

    def test_图编译成功(self) -> None:
        """验证 build_graph() 能正常编译，不抛出异常。"""
        graph = build_graph()
        assert graph is not None
        assert isinstance(graph, CompiledStateGraph)

    def test_图包含全部5个节点(self) -> None:
        """验证编译后的图包含 M2 所需的全部节点。

        M2 节点：planner, parallel_searcher, search_worker, evaluator, writer。
        """
        graph = build_graph()
        # nodes 是 {名称: Node对象} 字典，需用 .values() 获取 Node 对象
        node_dict = graph.get_graph().nodes

        node_names = {n.name for n in node_dict.values()}
        assert "planner" in node_names
        assert "parallel_searcher" in node_names
        assert "search_worker" in node_names
        assert "evaluator" in node_names
        assert "writer" in node_names

    def test_图入口为planner(self) -> None:
        """验证图的入口节点是 planner。"""
        graph = build_graph()
        node_dict = graph.get_graph().nodes
        # 找到入口节点（__start__）
        entry_node = next(n for n in node_dict.values() if n.name == "__start__")
        # 从 __start__ 出发的边指向的目标节点 id
        target_ids = [e.target for e in graph.get_graph().edges if e.source == entry_node.id]
        # 验证 planner 在目标中
        planner_nodes = [n for n in node_dict.values() if n.name == "planner"]
        assert len(planner_nodes) == 1
        planner_id = planner_nodes[0].id
        assert planner_id in target_ids


# ============================================================================
# 多 Wave 端到端集成测试
# ============================================================================


class TestMultiWaveIntegration:
    """测试 parallel_searcher → search_worker → evaluator → should_continue_wave
    的完整多波循环流程。"""

    @pytest.mark.asyncio
    async def test_两波消费全部关键词(self) -> None:
        """端到端验证两波搜索完整流程。

        场景：planner 生成 5 个关键词，max_concurrent=2（.env 中配置），
        需要 3 波才能全部消费完（2 + 2 + 1）。

        验证：
        1. 第一波分发 2 个 → pending 剩 3 个
        2. 第二波分发 2 个 → pending 剩 1 个
        3. 第三波分发 1 个 → pending 清空
        """
        limit = settings.max_concurrent_searches
        all_keywords = [f"KW{i}" for i in range(limit * 2 + 1)]  # 5 个关键词
        state = create_test_state({"plan": all_keywords})

        # ---- 第一波 ----
        sends1 = await parallel_searcher(state)
        batch1 = [s.arg["keyword"] for s in sends1]
        assert len(batch1) == limit
        assert batch1 == all_keywords[:limit]
        assert state["pending_keywords"] == all_keywords[limit:]

        # 模拟 should_continue_wave 判断：pending 非空 → 回到 parallel_searcher
        wave_decision = should_continue_wave(state)
        assert wave_decision == "parallel_searcher"

        # ---- 第二波 ----
        sends2 = await parallel_searcher(state)
        batch2 = [s.arg["keyword"] for s in sends2]
        assert len(batch2) == limit
        assert batch2 == all_keywords[limit : limit * 2]
        assert state["pending_keywords"] == all_keywords[limit * 2 :]  # 剩下 1 个

        # ---- 第三波 ----
        sends3 = await parallel_searcher(state)
        batch3 = [s.arg["keyword"] for s in sends3]
        assert len(batch3) == 1
        assert batch3 == all_keywords[limit * 2 :]
        assert state["pending_keywords"] == []  # 全部消费完毕

        # pending 为空后，should_continue_wave 才会判断置信度
        # 这里设置低置信度 → 应返回 planner
        state["confidence_score"] = 0.4
        final_decision = should_continue_wave(state)
        assert final_decision == "planner"

    @pytest.mark.asyncio
    async def test_搜索节点并行执行不互相阻塞(self) -> None:
        """验证多个 search_worker（通过 Send 分发）能并行执行而非串行。

        通过 mock search_tavily 并记录并发调用时间来验证：
        如果 N 个 search_worker 是并行的，总耗时 ≈ 单次搜索耗时；
        如果串行的，总耗时 ≈ N × 单次搜索耗时。
        """

        # 使用事件来同步并发任务的启动
        start_event = asyncio.Event()
        max_concurrent = 0
        current_concurrent = 0

        async def mock_search_with_tracking(*args, **kwargs):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            # 等待所有任务启动后再释放
            await start_event.wait()
            await asyncio.sleep(0.02)  # 模拟搜索延迟
            current_concurrent -= 1
            return "搜索结果内容"

        keywords = [f"K{i}" for i in range(settings.max_concurrent_searches)]
        state = create_test_state({"plan": keywords})

        with patch(
            "src.agents.search_worker.search_tavily",
            new=mock_search_with_tracking,
        ):
            # 分发任务
            sends = await parallel_searcher(state)

            # 并行执行所有 search_worker（模拟 LangGraph Send 机制）
            async def run_worker(send):
                worker_state = {**state, **send.arg}
                return await search_worker(worker_state)

            # 创建任务但不等待
            tasks = [asyncio.create_task(run_worker(s)) for s in sends]
            # 短暂等待让所有任务都进入 mock_search_with_tracking
            await asyncio.sleep(0.01)
            # 释放所有任务
            start_event.set()
            # 等待全部完成
            results = await asyncio.gather(*tasks)

        # 验证：最大并发数应等于任务数（说明它们真正并行执行了）
        assert max_concurrent == len(keywords)
        # 每个 worker 返回了正确结果
        for i, result in enumerate(results):
            assert "search_results" in result
            assert result["search_results"][0]["keyword"] == keywords[i]

    @pytest.mark.asyncio
    async def test_search_results跨波累积(self) -> None:
        """验证 search_results 通过 operator.add 在多波之间正确累加。

        模拟两波搜索后 search_results 应包含所有波次的结果，而非覆盖。
        """
        limit = settings.max_concurrent_searches

        # 构造已有一波结果的 state
        first_wave_results = [{"keyword": f"K{i}", "content": f"第{i}个结果"} for i in range(limit)]
        state = create_test_state(
            {
                # pending_keywords 剩余关键词触发第二波
                "pending_keywords": [f"K{limit}", f"K{limit + 1}"],
                "search_results": first_wave_results,
            }
        )

        # Mock 第二波搜索
        with patch(
            "src.agents.search_worker.search_tavily",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = "第二波搜索结果"

            # 分发第二波
            sends = await parallel_searcher(state)
            assert len(sends) == 2

            # 模拟 LangGraph 并行执行所有 Send
            all_results = list(first_wave_results)  # 模拟 operator.add 的起始值
            for send in sends:
                worker_state = {**state, **send.arg}
                result = await search_worker(worker_state)
                all_results.extend(result["search_results"])  # operator.add 行为

        # 验证累积结果数量 = 第一波 + 第二波
        assert len(all_results) == limit + 2
        # 验证第一波结果仍在
        for item in first_wave_results:
            assert item in all_results
        # 验证第二波结果存在
        second_wave_keywords = {r["keyword"] for r in all_results[limit:]}
        assert f"K{limit}" in second_wave_keywords
        assert f"K{limit + 1}" in second_wave_keywords

    @pytest.mark.asyncio
    async def test_去重在多波中持续生效(self) -> None:
        """验证去重逻辑在多波循环中不会失效。

        场景：第一波搜索了 K1，第二波 pending 中的 K1 应被过滤不重复搜索。
        """
        state = create_test_state(
            {
                "pending_keywords": ["K1", "K2"],
                "search_results": [
                    {"keyword": "K1", "content": "K1 已搜"}  # K1 已存在
                ],
            }
        )

        sends = await parallel_searcher(state)

        # K1 应被去重过滤，只保留 K2
        batch_keywords = [s.arg["keyword"] for s in sends]
        assert "K2" in batch_keywords
        assert "K1" not in batch_keywords
        assert len(batch_keywords) == 1

    @pytest.mark.asyncio
    async def test_波间iteration不变重试时递增(self) -> None:
        """验证 iteration 只在 planner 重试路径递增，波次循环中不变。

        完整模拟：
        1. 初始 iteration=0，第一波搜索
        2. evaluator 不递增（已移除此职责）
        3. should_continue_wave: pending 非空 → parallel_searcher，iteration 仍为 0
        4. 第二波搜索
        5. should_continue_wave: pending 空 + 不达标 → planner，iteration 变为 1
        """
        state = create_test_state(
            {
                "iteration": 0,
                "pending_keywords": ["K3", "K4"],
                "confidence_score": 0.3,
            }
        )

        # 第一波结束 → pending 非空
        result1 = should_continue_wave(state)
        assert result1 == "parallel_searcher"
        assert state["iteration"] == 0  # 波次不递增

        # 模拟第二波后 pending 清空
        state["pending_keywords"] = []
        state["confidence_score"] = 0.4  # 仍不达标

        # 第二波结束 → pending 空 + 不达标
        result2 = should_continue_wave(state)
        assert result2 == "planner"
        assert state["iteration"] == 1  # 回 planner 才递增

    @pytest.mark.asyncio
    async def test_M2完整流程手动编排(self) -> None:
        """手动编排 M2 全部节点协作流程，验证端到端行为。

        流程：planner → parallel_searcher → search_worker(×N) → evaluator →
              should_continue_wave(→ parallel_searcher → ...) → writer

        使用 mock 替代 LLM 和 Tavily，按 LangGraph 图结构手动串联所有节点。
        """
        from src.agents.evaluator import evaluator
        from src.agents.planner import planner

        # === 初始状态 ===
        state: AgentState = {
            "user_query": "测试问题",
            "iteration": 0,
            "plan": [],
            "search_results": [],
            "confidence_score": 0.0,
            "missing_info": "",
            "retry_keywords": [],
            "final_report": "",
            "human_approved": False,
            "task_id": "m2-manual-test",
            "total_tokens": 0,
            "pending_keywords": [],
        }

        with (
            patch(
                "src.agents.planner.limited_llm_call",
                new_callable=AsyncMock,
            ) as mock_planner_llm,
            patch(
                "src.agents.search_worker.search_tavily",
                new_callable=AsyncMock,
            ) as mock_search,
            patch(
                "src.agents.evaluator.limited_llm_call",
                new_callable=AsyncMock,
            ) as mock_eval_llm,
        ):
            # Planner: 生成 4 个关键词（正好触发两波，每波 max_concurrent 个）
            limit = settings.max_concurrent_searches
            all_kw = [f"M2_KW{i}" for i in range(limit * 2)]
            mock_planner_llm.return_value = f'{{"plan": {all_kw}}}'

            mock_search.return_value = "Mock 搜索结果"
            mock_eval_llm.return_value = (
                '{"confidence_score": 0.95, "missing_info": "", "retry_keywords": []}'
            )

            # Step 1: Planner → 生成 plan
            planner_result = await planner(state)
            state.update(planner_result)  # type: ignore[typeddict-item]
            assert state["plan"] == all_kw

            # Step 2: parallel_searcher → 第一波分发
            sends_wave1 = await parallel_searcher(state)
            assert len(sends_wave1) == limit  # 第一波 = limit 个
            assert all(s.node == "search_worker" for s in sends_wave1)
            # 验证 pending 有剩余
            assert len(state["pending_keywords"]) == limit

            # Step 3: 模拟 search_worker 并行消费第一波
            for send in sends_wave1:
                worker_state = {**state, **send.arg}
                worker_result = await search_worker(worker_state)
                state["search_results"].extend(worker_result["search_results"])
            assert len(state["search_results"]) == limit

            # Step 4: evaluator → 评估（iteration 不变，由 should_continue_wave 管理）
            evaluator_result = await evaluator(state)
            state.update(evaluator_result)  # type: ignore[typeddict-item]

            # Step 5: should_continue_wave → pending 非空，回 parallel_searcher
            decision1 = should_continue_wave(state)
            assert decision1 == "parallel_searcher"
            assert state["iteration"] == 0  # 波次不递增

            # Step 6: parallel_searcher → 第二波分发
            sends_wave2 = await parallel_searcher(state)
            assert len(sends_wave2) == limit
            # 验证第二波关键词与第一波不重复（去重生效）
            wave1_kw = set()
            for r in state["search_results"]:
                wave1_kw.add(r["keyword"])
            wave2_kw = {s.arg["keyword"] for s in sends_wave2}
            assert wave1_kw.isdisjoint(wave2_kw)  # 两波无交集
            assert state["pending_keywords"] == []  # 全部消费完毕

            # Step 7: 模拟 search_worker 并行消费第二波
            for send in sends_wave2:
                worker_state = {**state, **send.arg}
                worker_result = await search_worker(worker_state)
                state["search_results"].extend(worker_result["search_results"])
            assert len(state["search_results"]) == limit * 2  # 两波累计

            # Step 8: evaluator → 评估
            state.update(await evaluator(state))  # type: ignore[typeddict-item]

            # Step 9: should_continue_wave → pending 空 + 高置信度 → writer
            decision2 = should_continue_wave(state)
            assert decision2 == "writer"

            # Step 10: Writer → 生成最终报告
            from src.agents.writer import writer

            writer_result = await writer(state)
            state.update(writer_result)  # type: ignore[typeddict-item]

            assert "final_report" in state
            assert len(state["final_report"]) > 0
            assert "测试问题" in state["final_report"]
            # 报告应包含关键词
            for kw in all_kw:
                assert kw in state["final_report"]
