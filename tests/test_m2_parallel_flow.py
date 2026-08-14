"""M2 里程碑测试：并行搜索分发 + 多 Wave 循环 + 去重 + 并发控制。

覆盖 M2 核心功能：
- should_continue_wave  — 波次/重试/结束决策（纯决策函数）
- build_graph           — 并行图编译验证
- route_to_search_workers — Send 扇出路由函数
- 多 Wave 端到端流程    — 分批 → 波次循环 → 重试 → 结束
- 去重跨波验证          — 已搜关键词不在后续波次中重复
- 迭代计数语义          — 由 evaluator 节点递增，波次循环中不变

注：接口重构后 parallel_searcher 返回 {"_batch_keywords", "pending_keywords"}，
Send 扇出由 route_to_search_workers 完成；search_worker 返回 _search_accum。
"""

from typing import TYPE_CHECKING

# 仅用于类型检查的导入，运行时不会执行
if TYPE_CHECKING:
    from src.state import AgentState
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.graph.state import CompiledStateGraph

from config import settings
from src.agents.parallel_searcher import parallel_searcher, route_to_search_workers
from src.agents.search_worker import search_worker
from src.graph_factory import build_graph, should_continue_wave
from tests.conftest import create_test_state

# ============================================================================
# should_continue_wave 单元测试 — 波次/重试/结束决策
# ============================================================================


class TestShouldContinueWave:
    """测试 should_continue_wave 条件边决策函数。

    重构后该函数是纯决策函数：只根据 state 返回路由目标，
    不递增 iteration、不记录 metrics（这两项职责已移交 evaluator 节点）。
    """

    def test_pending非空返回parallel_searcher(self) -> None:
        """验证当 pending_keywords 非空时，优先返回 parallel_searcher 继续下一波。"""
        state = create_test_state(
            {
                "iteration": 0,
                "pending_keywords": ["K4", "K5"],
                "confidence_score": 0.5,
            }
        )

        result = should_continue_wave(state)

        assert result == "parallel_searcher"
        assert state["iteration"] == 0  # 纯函数不修改 iteration

    def test_pending空且置信度达标返回writer(self) -> None:
        """验证当 pending 为空且置信度达标时，返回 writer。

        重构后 should_continue_wave 不再记录 metrics（该职责已移交 evaluator），
        因此此处只断言路由结果。
        """
        state = create_test_state(
            {
                "iteration": 2,
                "pending_keywords": [],
                "confidence_score": 0.85,  # >= 0.8 threshold
            }
        )

        result = should_continue_wave(state)

        assert result == "writer"

    def test_pending空且达最大迭代返回writer(self) -> None:
        """验证当 iteration 达到上限时，即使置信度低也强制返回 writer。"""
        state = create_test_state(
            {
                "iteration": settings.max_iterations,
                "pending_keywords": [],
                "confidence_score": 0.1,  # 远低于阈值
            }
        )

        result = should_continue_wave(state)

        assert result == "writer"

    def test_pending空且低置信度返回planner(self) -> None:
        """验证当 pending 为空且不满足结束条件时，返回 planner。

        注意：iteration 递增已移交 evaluator 节点，should_continue_wave 是纯函数，
        不会修改 state 的 iteration。
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
        assert state["iteration"] == 1  # 纯函数不递增

    def test_是纯函数不修改iteration(self) -> None:
        """验证 should_continue_wave 在多次调用中均不修改 iteration。"""
        state = create_test_state(
            {
                "iteration": 0,
                "pending_keywords": ["K4", "K5"],
                "confidence_score": 0.4,
            }
        )

        assert should_continue_wave(state) == "parallel_searcher"
        assert state["iteration"] == 0

        # 模拟第二波结束：pending 清空、仍不达标
        state["pending_keywords"] = []
        state["confidence_score"] = 0.6

        assert should_continue_wave(state) == "planner"
        assert state["iteration"] == 0  # 递增由 evaluator 负责，此处仍为 0


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
        """验证编译后的图包含 M2 所需的全部节点。"""
        graph = build_graph()
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
        entry_node = next(n for n in node_dict.values() if n.name == "__start__")
        target_ids = [e.target for e in graph.get_graph().edges if e.source == entry_node.id]
        planner_nodes = [n for n in node_dict.values() if n.name == "planner"]
        assert len(planner_nodes) == 1
        planner_id = planner_nodes[0].id
        assert planner_id in target_ids


# ============================================================================
# 多 Wave 端到端集成测试
# ============================================================================


class TestMultiWaveIntegration:
    """测试 parallel_searcher → route_to_search_workers → search_worker →
    evaluator → should_continue_wave 的完整多波循环流程。"""

    @pytest.mark.asyncio
    async def test_两波消费全部关键词(self) -> None:
        """端到端验证分批消费：plan 5 个关键词，max_concurrent 个一批，分 3 波消费完。"""
        limit = settings.max_concurrent_searches
        all_keywords = [f"KW{i}" for i in range(limit * 2 + 1)]  # limit*2+1 个
        state = create_test_state({"plan": all_keywords})

        # ---- 第一波 ----
        update1 = await parallel_searcher(state)
        state.update(update1)  # type: ignore[typeddict-item]
        assert update1["_batch_keywords"] == all_keywords[:limit]
        assert update1["pending_keywords"] == all_keywords[limit:]

        # pending 非空 → 回到 parallel_searcher
        assert should_continue_wave(state) == "parallel_searcher"

        # ---- 第二波 ----
        update2 = await parallel_searcher(state)
        state.update(update2)  # type: ignore[typeddict-item]
        assert update2["_batch_keywords"] == all_keywords[limit : limit * 2]
        assert update2["pending_keywords"] == all_keywords[limit * 2 :]

        # ---- 第三波 ----
        update3 = await parallel_searcher(state)
        state.update(update3)  # type: ignore[typeddict-item]
        assert update3["_batch_keywords"] == all_keywords[limit * 2 :]
        assert update3["pending_keywords"] == []

        # pending 清空后，should_continue_wave 才判断置信度
        state["confidence_score"] = 0.4
        assert should_continue_wave(state) == "planner"

    @pytest.mark.asyncio
    async def test_搜索节点并行执行不互相阻塞(self) -> None:
        """验证多个 search_worker 能并行执行而非串行。

        通过 mock search_tavily 并记录并发调用时间：若 N 个 worker 并行，
        最大并发数应等于 N。
        """
        start_event = asyncio.Event()
        max_concurrent = 0
        current_concurrent = 0

        async def mock_search_with_tracking(*args, **kwargs):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await start_event.wait()
            await asyncio.sleep(0.02)
            current_concurrent -= 1
            return "搜索结果内容"

        keywords = [f"K{i}" for i in range(settings.max_concurrent_searches)]
        state = create_test_state({"plan": keywords})

        with patch(
            "src.agents.search_worker.search_tavily",
            new=mock_search_with_tracking,
        ):
            # 分发任务（写入 _batch_keywords）
            update = await parallel_searcher(state)
            state.update(update)  # type: ignore[typeddict-item]

            # 路由生成 Send
            sends = route_to_search_workers(state)
            assert isinstance(sends, list)

            # 并行执行所有 search_worker（模拟 LangGraph Send 机制）
            async def run_worker(send):
                worker_state = {**state, **send.arg}
                return await search_worker(worker_state)

            tasks = [asyncio.create_task(run_worker(s)) for s in sends]
            await asyncio.sleep(0.01)
            start_event.set()
            results = await asyncio.gather(*tasks)

        # 验证：最大并发数应等于任务数（说明它们真正并行执行了）
        assert max_concurrent == len(keywords)
        for i, result in enumerate(results):
            assert "_search_accum" in result
            assert result["_search_accum"][0]["keyword"] == keywords[i]

    @pytest.mark.asyncio
    async def test_search_results跨波累积(self) -> None:
        """验证多波搜索结果能正确累积（第一波结果 + 第二波结果）。"""
        limit = settings.max_concurrent_searches

        first_wave_results = [{"keyword": f"K{i}", "content": f"第{i}个结果"} for i in range(limit)]
        state = create_test_state(
            {
                "pending_keywords": [f"K{limit}", f"K{limit + 1}"],
                "search_results": first_wave_results,
            }
        )

        with patch(
            "src.agents.search_worker.search_tavily",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = "第二波搜索结果"

            update = await parallel_searcher(state)
            state.update(update)  # type: ignore[typeddict-item]
            sends = route_to_search_workers(state)
            assert isinstance(sends, list)
            assert len(sends) == 2

            # 模拟 LangGraph 并行执行所有 Send，累积到 _search_accum
            all_results = list(first_wave_results)
            for send in sends:
                worker_state = {**state, **send.arg}
                result = await search_worker(worker_state)
                all_results.extend(result["_search_accum"])

        # 验证累积结果数量 = 第一波 + 第二波
        assert len(all_results) == limit + 2
        for item in first_wave_results:
            assert item in all_results
        second_wave_keywords = {r["keyword"] for r in all_results[limit:]}
        assert f"K{limit}" in second_wave_keywords
        assert f"K{limit + 1}" in second_wave_keywords

    @pytest.mark.asyncio
    async def test_去重在多波中持续生效(self) -> None:
        """验证去重逻辑在多波循环中不会失效：已搜的 K1 不重复分发。"""
        state = create_test_state(
            {
                "pending_keywords": ["K1", "K2"],
                "search_results": [
                    {"keyword": "K1", "content": "K1 已搜"}  # K1 已存在
                ],
            }
        )

        update = await parallel_searcher(state)

        # K1 应被去重过滤，只保留 K2
        batch_keywords = update["_batch_keywords"]
        assert "K2" in batch_keywords
        assert "K1" not in batch_keywords
        assert len(batch_keywords) == 1

    @pytest.mark.asyncio
    async def test_evaluator负责递增iteration(self) -> None:
        """验证 iteration 递增由 evaluator 节点负责（should_continue_wave 是纯函数）。"""
        from src.agents.evaluator import evaluator

        state = create_test_state(
            {
                "iteration": 0,
                "search_results": [],
                "user_query": "测试",
                "retry_keywords": [],
            }
        )

        with patch(
            "src.agents.evaluator.llm_call_with_fallback",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = (
                '{"confidence_score": 0.4, "missing_info": "缺数据", "retry_keywords": []}',
                0,
                0,
                0,
            )
            result = await evaluator(state)

        assert result["iteration"] == 1  # evaluator 递增轮次

    @pytest.mark.asyncio
    async def test_M2完整流程手动编排(self) -> None:
        """手动编排 M2 全部节点协作流程，验证端到端行为。

        流程：planner → parallel_searcher → route_to_search_workers →
              search_worker(×N) → evaluator → should_continue_wave → writer
        """
        from src.agents.evaluator import evaluator
        from src.agents.planner import planner

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
                "src.agents.planner.llm_call_with_fallback",
                new_callable=AsyncMock,
            ) as mock_planner_llm,
            patch(
                "src.agents.search_worker.search_tavily",
                new_callable=AsyncMock,
            ) as mock_search,
            patch(
                "src.agents.evaluator.llm_call_with_fallback",
                new_callable=AsyncMock,
            ) as mock_eval_llm,
            patch(
                "src.agents.writer.llm_call_with_fallback",
                new_callable=AsyncMock,
            ) as mock_writer_llm,
        ):
            limit = settings.max_concurrent_searches
            all_kw = [f"M2_KW{i}" for i in range(limit * 2)]

            # Planner 生成 limit*2 个关键词（正好触发两波）
            mock_planner_llm.return_value = (json.dumps({"plan": all_kw}), 0, 0, 0)
            mock_search.return_value = "Mock 搜索结果"
            # 第一波评估：低置信度 → 继续
            mock_eval_llm.return_value = (
                '{"confidence_score": 0.4, "missing_info": "缺数据", "retry_keywords": []}',
                0,
                0,
                0,
            )

            # Step 1: Planner → 生成 plan
            planner_result = await planner(state)
            state.update(planner_result)  # type: ignore[typeddict-item]
            assert state["plan"] == all_kw

            # Step 2: parallel_searcher → 第一波分发（写入 _batch_keywords）
            update1 = await parallel_searcher(state)
            state.update(update1)  # type: ignore[typeddict-item]
            sends_wave1 = route_to_search_workers(state)
            assert isinstance(sends_wave1, list)
            assert len(sends_wave1) == limit
            assert len(state["pending_keywords"]) == limit

            # Step 3: search_worker 消费第一波（累积到 _search_accum）
            accum: list[dict] = []
            for send in sends_wave1:
                worker_state = {**state, **send.arg}
                worker_result = await search_worker(worker_state)
                accum.extend(worker_result["_search_accum"])
            state["_search_accum"] = accum  # 模拟 reduce_accum 累积

            # Step 4: evaluator → 评估（合并 _search_accum、递增 iteration）
            evaluator_result = await evaluator(state)
            state.update(evaluator_result)  # type: ignore[typeddict-item]
            assert len(state["search_results"]) == limit
            assert state["iteration"] == 1  # evaluator 递增

            # Step 5: should_continue_wave → pending 非空，回 parallel_searcher
            decision1 = should_continue_wave(state)
            assert decision1 == "parallel_searcher"

            # Step 6: 第二波分发
            update2 = await parallel_searcher(state)
            state.update(update2)  # type: ignore[typeddict-item]
            sends_wave2 = route_to_search_workers(state)
            assert isinstance(sends_wave2, list)
            assert len(sends_wave2) == limit
            assert state["pending_keywords"] == []  # 全部消费完毕

            # Step 7: search_worker 消费第二波
            accum2: list[dict] = []
            for send in sends_wave2:
                worker_state = {**state, **send.arg}
                worker_result = await search_worker(worker_state)
                accum2.extend(worker_result["_search_accum"])
            state["_search_accum"] = accum2

            # Step 8: evaluator → 第二波后高置信度
            mock_eval_llm.return_value = (
                '{"confidence_score": 0.95, "missing_info": "", "retry_keywords": []}',
                0,
                0,
                0,
            )
            evaluator_result2 = await evaluator(state)
            state.update(evaluator_result2)  # type: ignore[typeddict-item]
            assert len(state["search_results"]) == limit * 2

            # Step 9: should_continue_wave → 高置信度 → writer
            decision2 = should_continue_wave(state)
            assert decision2 == "writer"

            # Step 10: Writer → 生成最终报告
            from src.agents.writer import writer

            mock_writer_llm.return_value = (
                f"这是关于测试问题的调研报告，覆盖关键词：{' '.join(all_kw)}",
                0,
                0,
                0,
            )
            writer_result = await writer(state)
            state.update(writer_result)  # type: ignore[typeddict-item]

            assert "final_report" in state
            assert len(state["final_report"]) > 0
            assert "测试问题" in state["final_report"]
            for kw in all_kw:
                assert kw in state["final_report"]
