"""搜索计划生成节点（Planner 节点）模块。

LangGraph 工作流中的第一个处理节点，负责将用户的自然语言查询
转化为一组精确的搜索关键词，驱动后续的并行搜索节点。
"""

from langchain_ollama import ChatOllama

from config import settings
from src.state import AgentState
from src.utils.json_parser import robust_json_parse

# 初始化 Ollama 本地 LLM 实例
# temperature=0 确保每次输出稳定可复现，适合关键词提取任务
llm = ChatOllama(
    model=settings.ollama_model_name,
    base_url=settings.ollama_base_url,
    temperature=0,
    # api_key=settings.OLLAMA_API_KEY,
)


async def planner(state: AgentState) -> dict:
    """LangGraph 节点函数：根据用户查询（以及缺失信息）生成搜索关键词列表。

    作为 LangGraph 有向图中的 Planner 节点，此函数接收当前全局状态，
    通过 LLM 分析用户查询并生成 3-5 个高价值搜索关键词。

    工作流程：
    1. 从 state 中提取用户查询和上一轮缺失信息（如有）
    2. 构造提示词，告知 LLM 搜索意图并指定 JSON 输出格式
    3. 异步调用 Ollama LLM 生成关键词列表
    4. 使用鲁棒 JSON 解析器提取结构化结果
    5. 返回部分状态更新，LangGraph 自动合并到全局状态

    Args:
        state: LangGraph 传递的当前全局状态（AgentState 字典）。
               包含 user_query、iteration、missing_info 等字段。

    Returns:
        dict: 部分状态更新字典，仅包含 {"plan": [...]}。
              LangGraph 自动将此更新合并到全局 AgentState 中，
              不会影响其他节点维护的字段。
    """
    query = state["user_query"]
    missing = state.get("missing_info", "")
    retry_keywords = state.get("retry_keywords", [])

    # 构造提示词：整合上一轮的缺失信息和建议关键词
    hints: list[str] = []
    if missing:
        hints.append(f"上一轮缺少的信息：{missing}")
    if retry_keywords:
        hints.append(f"建议补充搜索以下关键词：{', '.join(retry_keywords)}")
    hint_text = "\n".join(hints)

    prompt = (
        f"用户问题：{query}\n"
        f"{hint_text}\n"
        '请生成3-5个搜索关键词，输出JSON格式：{"plan": ["关键词1", "关键词2"]}'
    )

    # 异步调用 LLM 生成搜索计划
    # ainvoke 是 LangChain 的异步调用方法，不阻塞事件循环
    response = await llm.ainvoke(prompt)

    # 从 LLM 响应中提取 JSON，降级时使用原始查询作为唯一关键词
    # response.content 类型为 str | list，此处确保传入字符串
    content = response.content if isinstance(response.content, str) else str(response.content)
    data = await robust_json_parse(content)
    plan = data.get("plan", [query])

    return {"plan": plan}
