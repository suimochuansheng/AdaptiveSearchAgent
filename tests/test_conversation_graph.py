"""会话式 Agent 图结构与 deep_research 工具行为测试（阶段 A 修复）。

- 工具版图必须跳过 hitl_confirm（interrupt 不能嵌套在工具调用内），旧入口必须保留；
- deep_research 的并发保护、异常回灌、报告压缩（保末段结论）、独立 thread_id / 无 checkpointer。
都不需要真实 LLM：deep_research 内部调用的旧图用假图替换。
"""

from typing import Any, cast

from src import conversation_graph as cg
from src.context_manager import estimate_tokens
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


async def test_deep_research_returns_raw_report(monkeypatch) -> None:
    """调研工具原样返回报告：压缩与落盘统一由 tools 节点收口（阶段 B3）。

    这样原始报告先进 Offload 文件，prompt 里才拿压缩版——细节仍可回读。
    """
    filler = "检索到一段与问题关系不大的叙述，用于填充篇幅。" * 8
    report = "\n\n".join(
        [f"第 {i} 段：{filler}" for i in range(30)]
        + ["结论：A 总成本 1200 万元，优于 B 的 1500 万元。"]
    )
    monkeypatch.setattr("src.graph_factory.build_graph", lambda **_: _FakeWorkflow(report=report))
    monkeypatch.setattr(cg, "_deep_research_in_flight", False)

    result = await cg.deep_research("对比 A 和 B")

    assert result == report
    assert estimate_tokens(result) > 3000  # 确实超了落盘阈值，交给 tools 节点处理


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


def test_build_graph_pins_num_ctx(monkeypatch) -> None:
    """窗口必须显式钉住：不设时由 Ollama 默认值决定，预算分层就失去锚点。"""
    from config import settings

    captured: dict[str, Any] = {}

    class _DummyModel:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def bind_tools(self, tools: Any) -> "_DummyModel":
            captured["tools"] = [t.__name__ for t in tools]
            return self

        async def ainvoke(self, messages: Any) -> Any:
            from langchain_core.messages import AIMessage

            return AIMessage("ok")

    monkeypatch.setattr(cg, "ChatOllama", _DummyModel)

    cg.build_conversation_graph()

    assert captured["num_ctx"] == settings.ollama_num_ctx >= 8192
    assert captured["tools"] == [
        "search_knowledge",
        "search_tavily",
        "calculator",
        "deep_research",
        "read_offload",
    ]


def test_conversation_state_tracks_summary_ledger() -> None:
    """压缩账本字段必须在 state schema 里，否则摘要每轮重算、旧会话也无法续跑。"""
    assert "messages" in cg.ConversationState.__annotations__
    assert "context_summary" in cg.ConversationState.__annotations__
    assert "summary_upto" in cg.ConversationState.__annotations__


async def test_agent_node_summarizes_long_history(monkeypatch) -> None:
    """端到端（无真实 LLM）：长历史触发摘要并写回账本，旧消息不再原样进 prompt。"""
    from langchain_core.messages import AIMessage, HumanMessage

    from config import settings

    monkeypatch.setattr(settings, "context_history_budget", 200)
    monkeypatch.setattr(settings, "context_summary_budget", 200)

    summary_prompts: list[str] = []

    class _DummyModel:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def bind_tools(self, tools: Any) -> "_DummyModel":
            return self

        async def ainvoke(self, messages: Any) -> Any:
            system_text = str(messages[0].content)
            if "对话压缩器" in system_text:
                summary_prompts.append(system_text)
                return AIMessage("决定：用 A 方案；关键数字：1200")
            return AIMessage("最终回答")

    monkeypatch.setattr(cg, "ChatOllama", _DummyModel)
    graph = cg.build_conversation_graph()

    history = [HumanMessage(content=f"第 {i} 条：" + "内容" * 40) for i in range(30)]
    result = await graph.ainvoke({"messages": history})

    assert summary_prompts, "长历史应该触发一次摘要调用"
    assert result["context_summary"].startswith("决定：用 A 方案")
    assert result["summary_upto"] > 0
    assert result["messages"][-1].content == "最终回答"


async def test_tools_node_runs_and_shrinks(monkeypatch) -> None:
    """工具节点跑通：叫一次 calculator → 真实执行 → 结果回到会话（不需要 LLM）。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class _ToolCallingModel:
        def __init__(self, **kwargs: Any) -> None:
            self.calls = 0

        def bind_tools(self, tools: Any) -> "_ToolCallingModel":
            return self

        async def ainvoke(self, messages: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"}],
                )
            return AIMessage("答案是 2")

    monkeypatch.setattr(cg, "ChatOllama", _ToolCallingModel)
    graph = cg.build_conversation_graph()

    result = await graph.ainvoke({"messages": [HumanMessage("1+1 等于几")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "calculator 应该被执行"
    assert "2" in str(tool_messages[0].content)
    assert result["messages"][-1].content == "答案是 2"


# ── reject 分档语义 v2（2026-11-24）：只有 dangerous 被拒才终止本轮 ──────


def _state_with_rejected_tool(
    tool: str, outcome: str | None = "rejected_by_user"
) -> "cg.ConversationState":
    """构造"最后一条消息是被拒的工具结果"的会话状态。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    call = {"name": tool, "args": {}, "id": "call-x"}
    return cast(
        "cg.ConversationState",
        {
            "messages": [
                HumanMessage("q"),
                AIMessage(content="", tool_calls=[call]),
                ToolMessage(
                    content="用户拒绝执行该工具，本次未调用。", tool_call_id="call-x", name=tool
                ),
            ],
            "last_tool_outcome": outcome,
        },
    )


def test_route_after_tools_end_only_for_dangerous_rejection() -> None:
    """分档判据是 spec.permission：dangerous→END，external/read→agent，未拒→agent。"""
    # dangerous 被拒 -> 终止
    assert cg._route_after_tools(_state_with_rejected_tool("deep_research")) is cg.END
    # 低风险被拒 -> 维持 v1 回灌
    assert cg._route_after_tools(_state_with_rejected_tool("search_knowledge")) == "agent"
    assert cg._route_after_tools(_state_with_rejected_tool("calculator")) == "agent"
    # 没有拒绝 -> 永远回模型
    assert (
        cg._route_after_tools(_state_with_rejected_tool("deep_research", outcome=None)) == "agent"
    )
    # 未注册工具按低风险处理（注册表缺项不得升级为终止）
    assert cg._route_after_tools(_state_with_rejected_tool("mcp_gone_tool")) == "agent"


async def test_dangerous_reject_ends_turn_without_calling_model_again(monkeypatch) -> None:
    """端到端（无真实 LLM）：dangerous 被拒后图在 tools 之后 END，模型不再被叫第二次。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    class _ToolCallingModel:
        def __init__(self, **kwargs: Any) -> None:
            self.calls = 0

        def bind_tools(self, tools: Any) -> "_ToolCallingModel":
            return self

        async def ainvoke(self, messages: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "deep_research", "args": {"query": "x"}, "id": "c1"}],
                )
            return AIMessage("模型不该被再次调用（否则就是自造内容）")

    monkeypatch.setattr(cg, "ChatOllama", _ToolCallingModel)
    monkeypatch.setattr(cg, "_deep_research_in_flight", False)
    monkeypatch.setattr(
        "src.graph_factory.build_graph",
        lambda **kwargs: _FakeWorkflow(report="调研报告正文"),
    )
    # 模拟"用户点了拒绝"：执行器拒绝分支判定为真（图侧只依赖这一个信号）
    monkeypatch.setattr(cg.executor, "is_user_rejection", lambda message: True)

    graph = cg.build_conversation_graph()
    result = await graph.ainvoke({"messages": [HumanMessage("帮我调研一下")]})

    assert result["last_tool_outcome"] == "rejected_by_user"
    last = result["messages"][-1]
    assert isinstance(last, ToolMessage), f"本轮应终止在工具结果上: {type(last)}"
    contents = [str(m.content) for m in result["messages"] if isinstance(m, AIMessage)]
    assert "模型不该被再次调用（否则就是自造内容）" not in contents

    # ── 对照：低风险（external）被拒仍回灌继续，本轮继续产生内容 ──────────
    calls: list[int] = []

    class _LowRiskModel:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def bind_tools(self, tools: Any) -> "_LowRiskModel":
            return self

        async def ainvoke(self, messages: Any) -> Any:
            calls.append(1)
            if len(calls) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "search_knowledge", "args": {"query": "q"}, "id": "c2"}],
                )
            return AIMessage("tool 被拒了，我换个说法继续答。")

    # 不真的去打 RAG 网关：这里只关心"被拒之后还回不回模型"
    async def _fake_search_knowledge(query: str, **kwargs: Any) -> str:
        """假的 RAG 检索：本用例只关心"被拒之后还回不回模型"，不真的打网关。"""
        return "[stub] 检索结果"

    monkeypatch.setattr("src.tools.rag.search_knowledge", _fake_search_knowledge)
    monkeypatch.setattr(cg, "ChatOllama", _LowRiskModel)
    graph2 = cg.build_conversation_graph()
    result2 = await graph2.ainvoke({"messages": [HumanMessage("查一下内部知识库")]})

    assert len(calls) == 2, f"external 被拒后应回灌继续（模型应被调用 2 次）: {len(calls)}"
    assert result2["messages"][-1].content == "tool 被拒了，我换个说法继续答。"
    # 结局字段在回到模型时被清掉，语义固定为"本轮"
    assert result2["last_tool_outcome"] is None
