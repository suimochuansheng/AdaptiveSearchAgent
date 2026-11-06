"""deep_research 的 stub 后端测试（开发模式）。

目的：搭主循环时不依赖旧 workflow / RAG / Ollama 子服务，但要保证
"假图与真图接口一致"，否则切换后主循环行为会变。
"""

from typing import Any

from src import conversation_graph as cg


class _FakeWorkflow:
    def __init__(self, report: str = "真报告") -> None:
        self.report = report
        self.calls = 0

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None):
        self.calls += 1
        return {"final_report": self.report}


async def test_stub_backend_returns_report_without_old_graph(monkeypatch) -> None:
    from config import settings

    monkeypatch.setattr(settings, "DEEP_RESEARCH_BACKEND", "stub")
    monkeypatch.setattr(cg, "_deep_research_in_flight", False)

    def _explode(**_: Any) -> Any:
        raise AssertionError("stub 模式不应构建旧 workflow 图")

    monkeypatch.setattr("src.graph_factory.build_graph", _explode)

    result = await cg.deep_research("对比 A 和 B")

    assert "[stub]" in result
    assert "对比 A 和 B" in result
    assert "结论" in result


async def test_stub_backend_still_respects_concurrency_guard(monkeypatch) -> None:
    from config import settings

    monkeypatch.setattr(settings, "DEEP_RESEARCH_BACKEND", "stub")
    monkeypatch.setattr(cg, "_deep_research_in_flight", True)

    result = await cg.deep_research("对比 A 和 B")

    assert "正在进行" in result


async def test_workflow_backend_is_default(monkeypatch) -> None:
    """默认必须还是真图：验收/生产行为不能因为开发模式改变。"""
    from config import settings

    assert settings.DEEP_RESEARCH_BACKEND == "workflow"

    fake = _FakeWorkflow("真图产出")
    monkeypatch.setattr("src.graph_factory.build_graph", lambda **_: fake)
    monkeypatch.setattr(cg, "_deep_research_in_flight", False)

    result = await cg.deep_research("对比 A 和 B")

    assert result == "真图产出"
    assert fake.calls == 1


async def test_stub_report_is_deterministic_and_compressible(monkeypatch) -> None:
    """假报告要稳定（可断言）且结构完整（有标题/数字/结论，能走压缩链路）。"""
    from src.tools.stub_research import stub_report

    first = stub_report("同一个问题")
    second = stub_report("同一个问题")

    assert first == second
    assert "1200" in first and "结论" in first and "##" in first


async def test_stub_backend_resets_in_flight_flag(monkeypatch) -> None:
    from config import settings

    monkeypatch.setattr(settings, "DEEP_RESEARCH_BACKEND", "stub")
    monkeypatch.setattr(cg, "_deep_research_in_flight", False)

    await cg.deep_research("一次调研")

    assert cg._deep_research_in_flight is False


def test_needle_report_is_long_and_hides_needle_in_middle() -> None:
    """压测用长报告：足够长（会触发 offload），针眼在压缩必丢的中段。"""
    from src.context_manager import compress_tool_output, estimate_tokens
    from src.tools.stub_research import NEEDLE, stub_report

    report = stub_report("请给我一份长报告，对比 A 和 B")

    assert estimate_tokens(report) > 1200  # 会触发 Offload
    assert NEEDLE in report

    compressed = compress_tool_output(report, 1200)
    assert NEEDLE not in compressed  # 压缩版确实丢了针眼 → 只能回读
