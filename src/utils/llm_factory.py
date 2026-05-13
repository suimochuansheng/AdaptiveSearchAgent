"""LLM 模型工厂模块。

通过 LangGraph RunnableConfig 实现请求级模型动态切换（DeepSeek / Ollama）。
切换优先级：
1. llm_provider 参数（仅供测试直接调用）
2. config["configurable"]["llm_provider"]（生产路径，唯一外部入口）
"""

from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from pydantic import SecretStr

from config import settings


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
