"""会话式 Agent 图结构与 deep_research 工具行为测试（阶段 A 修复）。

- 工具版图必须跳过 hitl_confirm（interrupt 不能嵌套在工具调用内），旧入口必须保留；
- deep_research 的并发保护、异常回灌、报告截断、独立 thread_id / 无 checkpointer。
都不需要真实 LLM：deep_research 内部调用的旧图用假图替换。
"""

from typing import Any

from src import conversation_graph as cg
from src.graph_factory import build_graph


class _FakeWorkflow:
    """假的旧 workflow 图：记录调用参数，返回预置 final_report。"""

    def __init__(self, *, report: str = "报告", error: Exception | None = None) -> None:
        self._report = report
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None):
        self.calls.append({"state": state, "config": config})
        if self._error is not None:
            raise self._error
        return {"final_report": self._report}


def test_tool_mode_graph_skips_hitl() -> None:
    """工具版图没有 hitl_confirm 节点。"""
    nodes = set(build_graph(checkpointer=None, include_hitl=False).get_graph().nodes)

    assert "hitl_confirm" not in nodes
    assert {"planner", "parallel_searcher", "evaluator", "writer"} <= nodes


def test_legacy_graph_keeps_hitl() -> None:
    """旧入口默认仍带人工确认，两个入口并存。"""
    nodes = set(build_graph().get_graph().nodes)

    assert "hitl_confirm" in nodes


async def test_deep_research_rejects_concurrent_call(monkeypatch) -> None:
    """同一时刻只跑一个调研，第二次直接回灌提示而不是排队。"""
    monkeypatch.setattr(cg, "_deep_research_in_flight", True)

    result = await cg.deep_research("对比 A 和 B")

    assert "正在进行" in result


async def test_deep_research_calls_tool_mode_graph_with_isolated_thread(monkeypatch) -> None:
    """调研内部用独立 thread_id、不接 checkpointer、跳过 HITL。"""
    fake = _FakeWorkflow(report="报告正文")
    captured: dict[str, Any] = {}

    def _fake_build_graph(checkpointer=None, include_hitl=True):  # type: ignore[no-untyped-def]
        captured["checkpointer"] = checkpointer
        captured["include_hitl"] = include_hitl
        return fake

    monkeypatch.setattr("src.graph_factory.build_graph", _fake_build_graph)
    monkeypatch.setattr(cg, "_deep_research_in_flight", False)

    result = await cg.deep_research("对比 A 和 B")

    assert result == "报告正文"
    assert captured == {"checkpointer": None, "include_hitl": False}
    thread_id = fake.calls[0]["config"]["configurable"]["thread_id"]  # type: ignore[index]
    assert thread_id.startswith("deep-research-")
    assert cg._deep_research_in_flight is False


async def test_deep_research_truncates_long_report(monkeypatch) -> None:
    """超长报告按字符上限截断，并标注已截断（段落重要度压缩是阶段 B）。"""
    monkeypatch.setattr(
        "src.graph_factory.build_graph", lambda **_: _FakeWorkflow(report="x" * 5000)
    )
    monkeypatch.setattr(cg, "_deep_research_in_flight", False)

    result = await cg.deep_research("对比 A 和 B")

    assert result.endswith("已截断）")
    assert len(result) < 5000


async def test_deep_research_turns_exception_into_tool_text(monkeypatch) -> None:
    """调研失败不把整个会话打断：异常转成 ToolMessage 文本回灌给模型。"""
    monkeypatch.setattr(
        "src.graph_factory.build_graph",
        lambda **_: _FakeWorkflow(error=RuntimeError("tavily 429")),
    )
    monkeypatch.setattr(cg, "_deep_research_in_flight", False)

    result = await cg.deep_research("对比 A 和 B")

    assert "深度调研执行失败" in result
    assert "tavily 429" in result
    # 失败后标志必须复位，否则这个工具会被永久锁死
    assert cg._deep_research_in_flight is False


async def test_deep_research_empty_report_falls_back(monkeypatch) -> None:
    """没产出报告时给模型一句可行动的话，而不是空字符串。"""
    monkeypatch.setattr("src.graph_factory.build_graph", lambda **_: _FakeWorkflow(report=""))
    monkeypatch.setattr(cg, "_deep_research_in_flight", False)

    result = await cg.deep_research("对比 A 和 B")

    assert "未能生成报告" in result
