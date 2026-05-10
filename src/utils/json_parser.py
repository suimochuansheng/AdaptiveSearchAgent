"""鲁棒 JSON 解析工具模块。

处理 LLM 输出的不完美 JSON 文本，支持从混杂内容中提取结构化数据。
"""

import json
import re
from typing import Any


def robust_json_parse(text: str) -> dict[str, Any]:
    """从 LLM 输出的文本中鲁棒地提取 JSON 对象。

    处理 LLM 常见输出问题：首尾多余文本、markdown 代码块包裹、
    不含 JSON 的纯文本等。

    Args:
        text: LLM 的原始输出字符串，可能包含非 JSON 内容。

    Returns:
        dict: 解析成功返回字典，解析失败返回空字典（降级处理）。
    """
    if not text or not text.strip():
        return {}

    # 策略1：直接尝试 JSON 解析（最优情况）
    try:
        result = json.loads(text)
        # 确保返回的是字典类型，列表等非字典结果降级为包装字典
        if isinstance(result, dict):
            return result
        return {}
    except json.JSONDecodeError:
        pass

    # 策略2：从 markdown 代码块或混杂文本中提取第一个 JSON 对象
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 策略3：更宽松的提取 —— 匹配嵌套 JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 所有策略均失败，返回空字典作为降级结果
    return {}
