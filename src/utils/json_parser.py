"""鲁棒 JSON 解析工具模块。

提供四层回退机制的 JSON 解析：直接解析 → 正则提取 → 宽松匹配 → DeepSeek LLM 修复。
"""

import json
import re
import urllib.request
from typing import Any

from config import settings
from src.utils.metrics import metrics

# DeepSeek API 配置（OpenAI 兼容接口）
# 通过 pydantic-settings 自动加载 .env 中的 DEEPSEEK_API_KEY
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = settings.DEEPSEEK_API_KEY


def _parse_direct(text: str) -> dict[str, Any] | None:
    """策略1：直接 JSON 解析。成功返回字典，失败返回 None。"""
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_regex_nested(text: str) -> dict[str, Any] | None:
    """策略2：嵌套感知正则提取，匹配最外层 { ... } 允许内部嵌套一层。"""
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group())
        return result if isinstance(result, dict) else None
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


def _repair_with_deepseek(malformed_text: str) -> str | None:
    """策略4：调用 DeepSeek API 修复损坏的 JSON 文本。

    向 DeepSeek 发送损坏的文本，要求其修复为标准 JSON 格式后返回。
    使用同步 HTTP 调用以保持与 robust_json_parse 的兼容性。

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

    payload = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请修复以下文本中的 JSON：\n{malformed_text}"},
            ],
            "temperature": 0,
            "max_tokens": 2000,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        fixed_text = body["choices"][0]["message"]["content"].strip()
        # 验证修复后的文本是否为合法 JSON 字典（且不包含误导性的 error 字段）
        try:
            parsed = json.loads(fixed_text)
            if isinstance(parsed, dict) and "error" not in parsed:
                return str(fixed_text)
            else:
                # 解析出的是非字典，或者包含 error 字段，视为无效修复
                return None
        except json.JSONDecodeError:
            return None
    except (urllib.error.URLError, json.JSONDecodeError, OSError, KeyError, IndexError):
        return None


def robust_json_parse(text: str) -> dict[str, Any]:
    """从 LLM 输出的文本中鲁棒地提取 JSON 对象。

    四层回退机制（从严格到宽松）：
    1. 直接 JSON 解析（_parse_direct）
    2. 嵌套感知的正则提取（_parse_regex_nested）
    3. 贪婪正则匹配（_parse_regex_greedy）
    4. DeepSeek LLM 修复（_repair_with_deepseek）

    Args:
        text: LLM 的原始输出字符串，可能包含非 JSON 内容。

    Returns:
        dict: 解析成功返回字典，所有策略均失败返回空字典。
    """
    if not text or not text.strip():
        metrics.record_parse(success=False)
        return {}

    # 策略1：直接 JSON 解析
    result = _parse_direct(text)
    if result is not None:
        metrics.record_parse(success=True)
        return result

    # 策略2：嵌套感知正则提取
    result = _parse_regex_nested(text)
    if result is not None:
        metrics.record_parse(success=True, fallback=True)
        return result

    # 策略3：贪婪正则匹配
    result = _parse_regex_greedy(text)
    if result is not None:
        metrics.record_parse(success=True, fallback=True)
        return result

    # 策略4：DeepSeek LLM 模型降级修复
    fixed_text = _repair_with_deepseek(text)
    if fixed_text:
        result = _parse_direct(fixed_text)
        if result is not None:
            metrics.record_parse(success=True, deepseek_fallback=True)
            return result

    # 所有策略均失败，最终降级返回空字典
    metrics.record_parse(success=False)
    return {}
