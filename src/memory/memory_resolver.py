"""记忆解析器：将用户画像注入 Planner 的系统提示词。

从 user_profiles 表读取用户偏好（语言、回答风格、历史话题），
构造一个包含个性化上下文的 System Prompt 字符串。
"""

from __future__ import annotations

from src.memory.user_profile_store import get_user_profile_store
from src.utils.logger import get_logger

logger = get_logger()

# ── 回答风格描述映射 ──────────────────────────────────────────

_STYLE_MAP: dict[str, str] = {
    "concise": "用最精简的语言回答，不超过 3 段",
    "detailed": "提供详细的解释和背景，包含具体数据",
    "table": "优先使用表格对比形式呈现",
}
_STYLE_DEFAULT = "detailed"

# ── 语言名称映射 ──────────────────────────────────────────────

_LANG_MAP: dict[str, str] = {
    "zh": "中文",
    "en": "English",
}
_LANG_DEFAULT = "中文"


async def build_planner_prompt_with_memory(user_query: str, thread_id: str) -> str:
    """根据用户画像构建带个性化偏好的 Planner System Prompt。

    流程：
    1. 从 user_profiles 读取 thread_id 对应的用户画像
    2. 映射风格/语言为人类可读描述
    3. 提取最近 3 个话题，拼接历史上下文提示
    4. 组装完整的 System Prompt 字符串

    Args:
        user_query: 用户原始查询（当前未直接嵌入 system prompt，预留扩展）。
        thread_id: 用户/会话标识，对应 user_profiles.user_id。

    Returns:
        可直接作为 SystemMessage.content 的提示词字符串。
    """
    store = await get_user_profile_store()
    profile = await store.get_profile(thread_id)

    style: str = profile.get("answer_style", _STYLE_DEFAULT)
    style_desc: str = _STYLE_MAP.get(style, _STYLE_MAP[_STYLE_DEFAULT])

    lang: str = profile.get("preferred_language", "zh")
    lang_name: str = _LANG_MAP.get(lang, _LANG_DEFAULT)

    topics: list[str] = profile.get("recent_topics", [])[:3]
    topic_hint = ""
    if topics:
        topic_hint = f"用户最近关注的主题：{', '.join(topics)}，可优先补充相关信息。\n"

    prompt = (
        "你是专业的自适应搜索助手，根据用户偏好生成精准搜索关键词。\n"
        "用户偏好：\n"
        f"- 语言：{lang_name}\n"
        f"- 回答风格：{style_desc}\n"
        f"{topic_hint}"
        "请基于以上偏好，生成 3-5 个精准的搜索关键词（仅输出关键词列表，JSON 格式）。"
    )

    logger.info(
        "Planner prompt built with memory",
        extra={"thread_id": thread_id, "style": style},
    )

    return prompt
