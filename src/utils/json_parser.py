"""鲁棒 JSON 解析工具模块。

提供四层回退机制的 JSON 解析：直接解析 → 正则提取 → 宽松匹配 → DeepSeek LLM 修复。
"""

import json
import re
from typing import Any

import httpx

from config import settings
from src.utils.metrics import metrics

# DeepSeek API 配置（OpenAI 兼容接口）
# 通过 pydantic-settings 自动加载 .env 中的 DEEPSEEK_API_KEY
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = settings.deepseek_api_key


def _parse_direct(text: str) -> dict[str, Any] | None:
    """策略1：直接 JSON 解析。成功返回字典，失败返回 None。"""
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_regex_nested(text: str) -> dict[str, Any] | None:
    """
    策略2：嵌套感知正则提取
    作用：从文本里提取 **最外层 { ... }**，支持内部嵌套一层 { }
    例如：好的，这是结果 {"a": {"b": 1}} 谢谢 → 提取 {"a": {"b": 1}}
    """
    # 核心：正则匹配 最外层 { ... }，允许内部嵌套一层 { }
    match = re.search(
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # 嵌套感知正则
        text,
        re.DOTALL,  # 让 . 可以匹配换行符
    )

    # 如果没匹配到任何 { ... } → 返回 None
    if not match:
        return None

    try:
        # 把匹配到的字符串 → 转成 JSON 字典
        result = json.loads(match.group())

        # 确保返回的是字典，不是列表/字符串等其他类型
        return result if isinstance(result, dict) else None

    # 解析失败 → 返回 None
    except json.JSONDecodeError:
        return None


def _parse_regex_greedy(text: str) -> dict[str, Any] | None:
    """策略3：贪婪正则匹配，匹配最外层大括号，容忍换行和特殊字符。"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group())
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


async def _repair_with_deepseek(malformed_text: str) -> str | None:
    """策略4：调用 DeepSeek API 修复损坏的 JSON 文本。

    使用 httpx.AsyncClient 原生异步 HTTP 调用，不阻塞事件循环。

    Args:
        malformed_text: 三层正则策略均无法解析的原始文本。

    Returns:
        修复后的 JSON 字符串（必须为合法的字典 JSON），失败或内容无效时返回 None。
    """
    if not DEEPSEEK_API_KEY:
        return None

    system_prompt = (
        "你是一个 JSON 修复助手。用户会提供一段无法解析的文本，"
        "请从中提取或修复出一个合法的 JSON 对象（必须是字典格式，不能是数组），"
        "只输出修复后的 JSON，不要添加任何解释、markdown 标记或额外文本。"
    )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                DEEPSEEK_API_URL,
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"请修复以下文本中的 JSON：\n{malformed_text}"},
                    ],
                    "temperature": 0,
                    "max_tokens": 2000,
                },
                timeout=20.0,
            )
            if response.status_code != 200:
                return None
            # ["choices"]：DeepSeek 的返回格式固定有一个 choices 数组里面存放 AI 给出的回答列表
            # fixed_text： 提取 DeepSeek 修复好的 JSON 字符串
            fixed_text: str = response.json()["choices"][0]["message"]["content"].strip()
            # 验证修复后的文本是否为合法 JSON 字典（且不包含误导性的 error 字段）
            try:
                parsed = json.loads(fixed_text)
                if isinstance(parsed, dict) and "error" not in parsed:
                    return fixed_text
            except json.JSONDecodeError:
                pass
            return None
        # 捕获 httpx 请求错误、JSON 解析错误、以及可能的键错误或索引错误，统一返回 None
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError):
            return None


async def robust_json_parse(text: str) -> dict[str, Any]:
    """从 LLM 输出的文本中鲁棒地提取 JSON 对象。

    四层回退机制（从严格到宽松）：
    1. 直接 JSON 解析（_parse_direct）
    2. 嵌套感知的正则提取（_parse_regex_nested）
    3. 贪婪正则匹配（_parse_regex_greedy）
    4. DeepSeek LLM 修复（_repair_with_deepseek，使用 httpx 原生异步调用）

    Args:
        text: LLM 的原始输出字符串，可能包含非 JSON 内容。

    Returns:
        dict: 解析成功返回字典，所有策略均失败返回空字典。
    """
    if not text or not text.strip():
        await metrics.record_parse(success=False)
        return {}

    # 策略1：直接 JSON 解析
    result = _parse_direct(text)
    if result is not None:
        await metrics.record_parse(success=True)
        return result

    # 策略2：嵌套感知正则提取
    result = _parse_regex_nested(text)
    if result is not None:
        await metrics.record_parse(success=True, fallback=True)
        return result

    # 策略3：贪婪正则匹配
    result = _parse_regex_greedy(text)
    if result is not None:
        await metrics.record_parse(success=True, fallback=True)
        return result

    # 策略4：DeepSeek LLM 模型降级修复（httpx 异步调用）
    fixed_text = await _repair_with_deepseek(text)
    if fixed_text:
        result = _parse_direct(fixed_text)
        if result is not None:
            await metrics.record_parse(success=True, deepseek_fallback=True)
            return result

    # 所有策略均失败，最终降级返回空字典
    await metrics.record_parse(success=False)
    return {}
