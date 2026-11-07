"""画像抽取（阶段 C3b：写侧）。

思路：会话每满 N 轮，异步跑一次抽取，把"语言偏好 / 回答风格 / 最近关注"写回
`user_profiles`。抽取失败的代价是"这次没更新"，绝不能影响会话本身——
所以这里所有失败路径都返回空结果，不抛异常。

可测性：模型调用（llm_fn）与存储（store）全部注入，单测不需要 LLM / DB。
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.utils.logger import get_logger

logger = get_logger()

PROFILE_PROMPT = """你是用户画像提取器。阅读下面的会话记录与摘要，输出一个严格 JSON 对象：

{
  "preferred_language": "zh" 或 "en" 或 null,
  "answer_style": "concise" 或 "detailed" 或 "table" 或 null,
  "recent_topics": ["主题1", "主题2"]  （最多 5 个，没有就给空数组）
}

规则：
- 只根据对话内容判断，不要编造；
- 判断不出来就给 null / 空数组；
- 只输出 JSON，不要解释、不要代码块标记。
"""

_ALLOWED_LANGS = {"zh", "en"}
_ALLOWED_STYLES = {"concise", "detailed", "table"}
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMFn(Protocol):
    """给一段 prompt，返回文本。"""

    async def __call__(self, prompt: str) -> str: ...


class ProfileStore(Protocol):
    async def upsert_profile(
        self,
        user_id: str,
        preferred_language: str | None = None,
        answer_style: str | None = None,
        recent_topics: list[str] | None = None,
    ) -> None: ...


def build_extraction_prompt(
    messages: list[Any], summary: str = "", max_messages: int = 20
) -> str:
    """把最近若干条消息拼成抽取用的 prompt（只取尾部，控制成本）。"""
    recent = messages[-max_messages:]
    lines: list[str] = []
    for message in recent:
        role = getattr(message, "type", "unknown")
        content = str(getattr(message, "content", "")).strip().replace("\n", " ")[:300]
        if content:
            lines.append(f"- {role}: {content}")

    parts = [PROFILE_PROMPT]
    if summary:
        parts.append(f"\n【会话摘要】\n{summary}\n")
    parts.append("\n【最近对话】\n" + "\n".join(lines))
    return "".join(parts)


def parse_profile_payload(text: str) -> dict[str, Any]:
    """解析模型输出；容忍 ```json 代码块；字段非法就丢弃（不报错）。"""
    raw = text.strip()
    fenced = _FENCE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("画像抽取结果不是合法 JSON，本次跳过")
        return {}
    if not isinstance(data, dict):
        return {}

    payload: dict[str, Any] = {}

    language = data.get("preferred_language")
    if language in _ALLOWED_LANGS:
        payload["preferred_language"] = language

    style = data.get("answer_style")
    if style in _ALLOWED_STYLES:
        payload["answer_style"] = style

    topics = data.get("recent_topics")
    if isinstance(topics, list):
        cleaned = [str(topic).strip() for topic in topics if str(topic).strip()]
        if cleaned:
            payload["recent_topics"] = cleaned[:5]

    return payload


async def extract_and_store(
    *,
    llm_fn: LLMFn,
    store: ProfileStore,
    user_id: str,
    messages: list[Any],
    summary: str = "",
    max_messages: int = 20,
) -> dict[str, Any]:
    """跑一次抽取并落库；任何失败都返回空 dict（不影响会话）。"""
    if not user_id or not messages:
        return {}

    prompt = build_extraction_prompt(messages, summary, max_messages)
    try:
        text = await llm_fn(prompt)
    except Exception:  # noqa: BLE001 - 抽取是增强能力，失败只记日志
        logger.warning("画像抽取调用失败", extra={"user_id": user_id}, exc_info=True)
        return {}

    payload = parse_profile_payload(text)
    if not payload:
        return {}

    try:
        await store.upsert_profile(user_id, **payload)
    except Exception:  # noqa: BLE001
        logger.warning("画像写库失败", extra={"user_id": user_id}, exc_info=True)
        return {}

    logger.info("用户画像已更新", extra={"user_id": user_id, "fields": sorted(payload)})
    return payload
