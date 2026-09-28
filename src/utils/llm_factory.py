"""LLM 模型工厂模块。

通过 LangGraph RunnableConfig 实现请求级模型动态切换（DeepSeek / Ollama）。
切换优先级：
1. llm_provider 参数（仅供测试直接调用）
2. config["configurable"]["llm_provider"]（生产路径，唯一外部入口）
"""

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from pydantic import SecretStr

from config import settings

# 全局信号量（在模块加载时创建）
# 这是一个全局唯一的闸机
# 整个程序从头到尾只创建一个
# 所有协程都共用它来限流
_llm_semaphore = None


def _resolve_provider(config: RunnableConfig | None = None) -> str | None:
    """从请求级 config 中解析 LLM provider，不读取全局 env。"""
    if config and "configurable" in config:
        provider: str | None = config["configurable"].get("llm_provider")
        if provider:
            return provider
    return None


def get_llm(
    temperature: float = 0,
    llm_provider: str | None = None,
    config: RunnableConfig | None = None,
) -> ChatDeepSeek | ChatOllama:
    """获取 LLM 大模型实例（工厂方法）。

    生产路径通过 graph.ainvoke 的 config 参数传入 provider：
        graph.ainvoke(input, config={"configurable": {"llm_provider": "deepseek"}})
    测试路径可显式传 llm_provider 参数，不暴露给外部调用方。

    Args:
        temperature: 温度系数（0 = 最稳定）。
        llm_provider: 直接指定 provider，仅供测试使用。
        config: LangGraph RunnableConfig，生产路径的唯一 provider 来源。

    Returns:
        对应的 LangChain LLM 聊天模型实例。
    """
    provider = llm_provider or _resolve_provider(config) or "ollama"

    if provider == "deepseek":
        return ChatDeepSeek(
            model=settings.DEEPSEEK_MODEL_NAME or "deepseek-chat",
            api_key=SecretStr(settings.deepseek_api_key),
            temperature=temperature,
        )
    else:
        return ChatOllama(
            model=settings.ollama_model_name,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )


# 就是个获取闸机的函数
# 第一次调用 → 创建闸机（最多放 2 个）
# 之后调用 → 直接返回已经创建好的闸机
# 保证全局唯一、不重复创建，限流才有效
def get_llm_semaphore(limit: int = 2):
    """获取全局 LLM 调用信号量，控制并发调用数。"""
    global _llm_semaphore
    if _llm_semaphore is None:
        # 第一次调用时创建信号量，后续调用复用同一个实例
        # Semaphore理解为一个限流闸机，允许同时通过的最大数量为 limit，多余的调用会被阻塞等待
        _llm_semaphore = asyncio.Semaphore(limit)
    return _llm_semaphore


async def limited_llm_call(llm: BaseChatModel, prompt: str) -> str:
    """带并发限制的 LLM 调用。"""
    sem = get_llm_semaphore()  # 获取全局信号量实例-拿到全局闸机
    # async with 是Python 固定异步语法，同一时刻，最多只有 sem 限定数量的异步任务在并发执行
    async with sem:  # 👇 这行 = 排队进闸机，最多2个同时进
        response = await llm.ainvoke(prompt)
        # 显式标注类型，消除 mypy 的 Any 污染
        raw: object = response.content
        content: str = raw if isinstance(raw, str) else str(raw)
        return content
