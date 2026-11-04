"""会话式 Agent 主循环 —— 阶段 A 改造（feature/conversation-agent）。

把 ReAct 演示图（tool_demo.py）升级为产品主干：

- **MessagesState**（对话历史为核心）替代任务进度板式的 AgentState；
- agent 节点：`ChatOllama.bind_tools` 绑定 4 个工具，每一步由模型自主决策；
- tools 节点：`ToolNode` 执行工具调用，结果作为 ToolMessage 回灌；
- 条件边：最后一条消息含 `tool_calls` → 继续执行工具；否则 → 结束（最终回答）；
- **deep_research 工具**：把旧的 workflow 主图（规划→并行搜索→评估→重试→写报告）
  包装成普通工具——"workflow 降级为工具"，对应 Codex/Claude Code 里 skill 的模式。

对应面试概念（面试题驱动）：
- Agentic Loop / Function Calling：T1/T2/S3
- workflow 与 Agent 的取舍：Q22/T5
- 为什么工具版图要跳过 HITL：Q3（interrupt 无法嵌套在工具调用里）
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from config import settings
from src.context_manager import compress_tool_output
from src.tools.calculator import calculate_math
from src.tools.rag import search_knowledge
from src.tools.search import search_tavily
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from src.state import AgentState

logger = get_logger()

# deep_research 回灌报告的 token 预算（阶段 B1：段落重要度压缩替代从头硬截断）
_DEEP_RESEARCH_TOKEN_BUDGET = 2000


async def calculator(expression: str) -> str:
    """计算数学表达式（如 "2 + 3 * sin(0.5)"）。只接受具体数字与白名单数学函数。"""
    return await calculate_math(expression)


# deep_research 同一时刻只允许一个在跑。
# ToolNode 会并发执行同一轮的多个 tool_call：两个调研同时进入会撞 Semaphore(2) 的 LLM
# 并发额度与 Tavily 限速，外层会话锁只管会话、不会拦第二次调用。
# 这里用模块级布尔而不是 Semaphore：检查与置位之间没有 await，单线程 asyncio 下天然原子。
_deep_research_in_flight = False


async def deep_research(query: str) -> str:
    """深度调研：对复杂问题执行「规划→并行搜索→评估→重试→生成报告」的完整闭环。

    适用场景：需要多角度、多来源交叉验证的对比/调研类问题。
    内部运行旧版 workflow 主图（include_hitl=False 跳过人工确认——工具调用内
    无法挂起等人工，HITL 职责由外层会话循环承担）。
    """
    global _deep_research_in_flight
    if _deep_research_in_flight:
        return "已有另一项深度调研正在进行，请先基于它的结果回答；需要补充请把问题拆得更具体。"

    _deep_research_in_flight = True
    try:
        return await _run_deep_research(query)
    except Exception as exc:  # noqa: BLE001 - 工具异常回灌给模型，不让整个会话挂掉
        logger.exception("deep_research 执行失败")
        return f"深度调研执行失败：{exc}。可以换一种问法，或拆成更具体的小问题再试。"
    finally:
        _deep_research_in_flight = False


async def _run_deep_research(query: str) -> str:
    """执行一次调研闭环并把报告压到预算内（并发与异常处理在调用方）。"""
    from src.graph_factory import build_graph

    # 工具内使用独立 thread_id，不污染外层会话的 checkpoint
    thread_id = f"deep-research-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id, "llm_provider": "ollama"}}

    initial_state: AgentState = {
        "user_query": query,
        "plan": [],
        "search_results": [],
        "_search_accum": [],
        "confidence_score": 0.0,
        "missing_info": "",
        "user_feedback": "",
        "retry_keywords": [],
        "iteration": 0,
        "final_report": "",
        "task_id": thread_id,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "current_llm": "ollama",
        "pending_keywords": [],
        "_batch_keywords": [],
        "hitl_decision": "confirmed",
    }

    # 工具版图：不接 checkpointer（无 interrupt、无需恢复），跳过 HITL 节点
    graph = build_graph(checkpointer=None, include_hitl=False)
    final_state = await graph.ainvoke(initial_state, config=config)

    report = final_state.get("final_report", "")
    if not report:
        return "深度调研未能生成报告，请换一种问法或拆成更小的问题。"
    # 阶段 B1：按段落重要度压缩（末段/数字/结论优先），不再从头硬截断
    return compress_tool_output(report, _DEEP_RESEARCH_TOKEN_BUDGET)


# ── 工具列表：模型每轮只能从这 4 个工具里选 ──────────────────────────────
TOOLS = [search_knowledge, search_tavily, calculator, deep_research]

# ── System Prompt（稳定区：角色 + 工具索引 + 判断规则，阶段 B 会做预算管理）──
SYSTEM_PROMPT = """你是内部知识助手，可以自主选择工具完成任务：

search_knowledge(query): 查询内部知识库（文档/FAQ）
search_tavily(keyword): 搜索互联网实时信息
calculator(expression): 计算数学表达式（如 "2 + 3 * sin(0.5)"）
deep_research(query): 对复杂问题做多角度深度调研，生成带来源的调研报告

判断原则：
- 简单事实 / 内部文档问题 → 用 search_knowledge，根据结果回答
- 实时新闻 / 公开信息 → 用 search_tavily
- 纯数学计算 → 仅用 calculator
- 需要多角度对比、综合调研的复杂问题 → 用 deep_research
- 回答只基于工具返回的结果；工具没有的信息，明确说不知道，不要编造
"""


def build_conversation_graph(checkpointer: BaseCheckpointSaver | None = None):
    """构建会话式 Agent 主循环（agent ⇄ tools 循环 + 条件边终止）。

    Args:
        checkpointer: 传入后支持多轮会话（同一 thread_id 的消息自动追加）
                     与中断恢复；不传则为无状态单轮模式。
    """
    model = ChatOllama(
        model=settings.ollama_model_name,
        base_url=settings.ollama_base_url,
        temperature=0,
        # 显式钉住窗口：不设时由 Ollama 模型默认值决定，阶段 B 的预算分层就没有锚点
        num_ctx=settings.ollama_num_ctx,
    ).bind_tools(TOOLS)

    async def agent_node(state: MessagesState) -> dict:
        # 每轮在消息头部注入 SystemMessage（保持无状态、可缓存前缀）
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        response = await model.ainvoke(full_messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        # 唯一的循环判断点：最后一条消息是否包含工具调用请求
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
    )
    graph.add_edge("tools", "agent")

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "Conversation agent graph built",
        extra={"tools": [t.__name__ for t in TOOLS], "model": settings.ollama_model_name},
    )
    return compiled


# ── 本地测试入口 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    async def _main() -> None:
        graph = build_conversation_graph()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="请对比 RTX 5090 和 RTX 4090 的功耗")]}
        )
        final_message = result["messages"][-1]
        print(f"\n{'=' * 60}")
        print(f"最终回复: {final_message.content}")
        print(f"{'=' * 60}")

    asyncio.run(_main())
