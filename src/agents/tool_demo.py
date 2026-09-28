"""
工具动态路由 Demo —— 展示 LangChain bind_tools + ToolNode 标准协议。

验证 3 个工具的动态调用能力：search_knowledge / search_tavily / calculator。
此模块仅作架构验证，主流程不受影响，由 ENABLE_DYNAMIC_TOOLS 开关控制。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from config import settings
from src.tools.calculator import calculate_math
from src.tools.rag import search_knowledge
from src.tools.search import search_tavily
from src.utils.logger import get_logger

logger = get_logger()

# ── calculator 别名（工具函数名为 calculate_math，bind_tools 以实际函数名为准）──


async def calculator(expression: str) -> str:
    """计算数学表达式（如 "2 + 3 * sin(0.5)"）。

    Args:
        expression: 数学表达式字符串。

    Returns:
        计算结果字符串。
    """
    return await calculate_math(expression)


# ── 工具列表 ──────────────────────────────────────────────────

TOOLS = [search_knowledge, search_tavily, calculator]

# ── System Prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """你是智能助手，可根据问题自主选择工具：

search_knowledge(query): 查询内部知识库（文档/FAQ）
search_tavily(keyword): 搜索互联网实时信息
calculator(expression): 计算数学表达式（如 "2 + 3 * sin(0.5)"）

判断原则：
- 纯数学计算 → 仅用 calculator
- 内部文档/产品问题 → 仅用 search_knowledge
- 实时新闻/公开信息 → 仅用 search_tavily
- 复杂问题可组合调用多个工具"""


def build_dynamic_tool_graph() -> StateGraph:
    """构建并返回绑定了 3 个工具的 LangGraph 状态图。

    Returns:
        编译后的 StateGraph，入口节点为 agent，工具结果回到 agent 继续推理。
    """
    model = ChatOllama(
        model=settings.ollama_model_name,
        base_url=settings.ollama_base_url,
        temperature=0,
    ).bind_tools(TOOLS)

    async def agent_node(state: MessagesState) -> dict:
        """Agent 节点：调用绑定工具的模型进行推理。"""
        messages = state["messages"]
        # 在消息列表头部注入 SystemMessage（每次调用以保持 stateless）
        from langchain_core.messages import SystemMessage

        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = await model.ainvoke(full_messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
    )
    graph.add_edge("tools", "agent")

    compiled = graph.compile()

    logger.info(
        "Dynamic tool graph built",
        extra={
            "tools": [t.__name__ for t in TOOLS],
            "model": settings.ollama_model_name,
        },
    )

    return compiled


# ── 测试入口 ───────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def _main() -> None:
        graph = build_dynamic_tool_graph()
        result = await graph.ainvoke({"messages": [HumanMessage(content="3.14 的平方根是多少？")]})
        # 提取最后一条 AI 消息的内容并打印
        final_message = result["messages"][-1]
        print(f"\n{'='*60}")
        print(f"最终回复: {final_message.content}")
        print(f"{'='*60}")

    asyncio.run(_main())
