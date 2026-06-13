"""双模型备援 + Token 计数模块。

提供统一的 LLM 调用入口 llm_call_with_fallback，实现：
1. 主模型 ollama → 重试 N 次 → 自动切换备援 deepseek
2. 切换后全局 config 被修改，后续节点自动使用新模型
3. 每次调用返回 (响应文本, 输入token数, 输出token数, 总token数)
4. 信号量限流（保留原有的并发控制）
"""

import asyncio
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from pydantic import SecretStr

from config import settings

logger = logging.getLogger(__name__)


def _resolve_provider(config: RunnableConfig | None = None) -> str:
    """从请求级 config 中解析当前 LLM provider，默认 ollama。"""
    if config and "configurable" in config:
        provider: str | None = config["configurable"].get("llm_provider")
        if provider:
            return provider
    return "ollama"


def _get_llm_instance(provider: str, temperature: float = 0) -> ChatOllama | ChatDeepSeek:
    """根据 provider 名称创建对应的 LLM 实例（同步工厂，不含调用）。"""
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


def _extract_tokens(response: AIMessage) -> tuple[int, int, int]:
    """从 AIMessage 响应中提取 (input_tokens, output_tokens, total_tokens)。

    ChatOllama 和 ChatDeepSeek 均返回 AIMessage，其 usage_metadata 字典
    包含 input_tokens / output_tokens / total_tokens 三个字段。
    """
    # getattr获取response中usage_metadata的值，如果没有给{}，如果为None通过or {} 返回空字典，
    # 双重保险，防止程序崩溃
    usage: dict[str, int] = getattr(response, "usage_metadata", {}) or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("total_tokens", 0)


# ── 全局信号量（保留原有的并发限流机制） ──────────────────────────────
_llm_semaphore: asyncio.Semaphore | None = None

# ── 全局模型切换计数器（供 KPI 报告读取） ────────────────────────────
# 每次 run_agent 调用前通过 reset_model_switch_count() 归零，
# 备援切换时 llm_call_with_fallback 自动 +1。
_model_switch_count: int = 0


def reset_model_switch_count() -> None:
    """重置模型切换计数器（每个新会话开始时调用）。"""
    global _model_switch_count
    _model_switch_count = 0


def get_model_switch_count() -> int:
    """读取当前会话的模型切换次数。"""
    return _model_switch_count


def _get_semaphore() -> asyncio.Semaphore:
    """获取全局 LLM 调用信号量（单例，控制并发数）。"""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
    return _llm_semaphore


async def _invoke_with_semaphore(llm: BaseChatModel, prompt: str) -> AIMessage:
    """带信号量限流的单次 LLM 调用。"""
    # sem 是管这 5 个通行证的管理员
    sem = _get_semaphore()
    # 先领一张通行证，才能进去执行 LLM 调用，调用结束后自动释放通行证
    async with sem:
        response: Any = await llm.ainvoke(prompt)
        # langchain 的 ainvoke 返回 AIMessage
        if not isinstance(response, AIMessage):
            # 防御：极端情况下返回非 AIMessage，包装一下
            return AIMessage(content=str(response))
        return response


async def llm_call_with_fallback(
    prompt: str,
    config: RunnableConfig | None = None,
) -> tuple[str, int, int, int]:
    """统一的 LLM 调用入口：主模型重试失败后自动切换备援。

    执行流程：
    1. 从 config 读取当前 provider（主模型 ollama，切换后变 deepseek）
    2. 调用主模型，最多重试 llm_max_retries 次
    3. 全部失败后切换到备援模型（deepseek），修改 config 使后续节点生效
    4. 备援模型也失败则抛出异常

    Args:
        prompt: 发送给 LLM 的提示词文本。
        config: LangGraph RunnableConfig，含 configurable.llm_provider。
                切换后该字典会被原地修改，后续节点自动读取新 provider。

    Returns:
        (response_text, input_tokens, output_tokens, total_tokens) 四元组。
    """
    provider = _resolve_provider(config)
    last_error: Exception | None = None

    # ── 阶段 1：尝试当前 provider，带重试 ──────────────────────────
    # +1 是因为 range 是左闭右开，重试次数是实际尝试次数，不是重试次数
    for attempt in range(settings.llm_max_retries + 1):
        try:
            # 获取模型实例-可能是 ollama 也可能是 deepseek
            llm = _get_llm_instance(provider)
            # 将模型实例和提示词发给带信号机制限制的调用函数，等待响应
            response = await _invoke_with_semaphore(llm, prompt)
            #
            content: str = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            in_tok, out_tok, total_tok = _extract_tokens(response)
            return content, in_tok, out_tok, total_tok
        except Exception as e:
            # 记录最后一次错误，继续重试或进入备援
            last_error = e
            #
            if attempt < settings.llm_max_retries:
                # 还有重试额度，短暂等待后继续
                import random

                # 添加Jitter避免所有请求同时重试，造成雪崩
                sleep_time = min((2**attempt) * random.uniform(0.5, 1.0), 10)
                logger.warning(
                    "LLM 调用失败(provider=%s, attempt=%d/%d): %s，%.1fs 后重试",
                    provider,
                    attempt + 1,
                    settings.llm_max_retries + 1,
                    e,
                    sleep_time,
                )
                await asyncio.sleep(sleep_time)
                continue
            # 重试耗尽，进入备援阶段
            break

    # ── 阶段 2：备援切换到 deepseek ───────────────────────────────
    if not settings.llm_fallback_enabled or provider == "deepseek":
        # 已经是 deepseek 仍失败，或无备援 → 直接抛出
        raise last_error  # type: ignore[misc]

    logger.warning(
        "主模型 %s 重试 %d 次全部失败(最后错误: %s)，切换到备援 DeepSeek",
        provider,
        settings.llm_max_retries + 1,
        last_error,
    )

    # 修改 config 中的 provider，后续节点自动使用 deepseek
    if config and "configurable" in config:
        config["configurable"]["llm_provider"] = "deepseek"

    # 记录一次模型切换（供 KPI 报告统计）
    global _model_switch_count
    _model_switch_count += 1

    # 备援模型仅尝试一次（不重试）
    llm = _get_llm_instance("deepseek")
    response = await _invoke_with_semaphore(llm, prompt)
    content = str(response.content) if not isinstance(response.content, str) else response.content
    # 获取token计数相关字段
    in_tok, out_tok, total_tok = _extract_tokens(response)
    return content, in_tok, out_tok, total_tok
