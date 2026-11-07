"""跨会话画像注入（阶段 C3a：只做"读"）。

设计要点：
- 画像属于**稳定区**：进 system prompt，每轮重注入，不参与历史压缩；
- 只在会话第一轮读一次 DB，结果缓存在 state 的 `profile_context` 里，之后不再查；
- 读失败 / 无画像 / 值是默认值 → 返回空串，行为与不启用记忆完全一致（可回滚）；
- 硬预算 200 tokens，超了用段落压缩兜底。
"""

from __future__ import annotations

from typing import Any

from config import settings
from src.checkpointer import get_business_pool
from src.context_manager import compress_tool_output, estimate_tokens
from src.utils.logger import get_logger

logger = get_logger()

# 与 src/memory/schema.sql 对齐（幂等建表，旧库也能用）
_DDL_TABLE = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id            TEXT PRIMARY KEY,
    preferred_language TEXT DEFAULT 'zh',
    answer_style       TEXT DEFAULT 'detailed',
    recent_topics      TEXT[] DEFAULT '{}',
    updated_at         TIMESTAMP DEFAULT NOW()
)
"""

_DDL_INDEX = "CREATE INDEX IF NOT EXISTS idx_user_profiles_updated ON user_profiles (updated_at)"

_LANG_NAMES = {"zh": "中文", "en": "English"}
_STYLE_NAMES = {
    "concise": "精简（不超过 3 段）",
    "detailed": "详细（含数据与背景）",
    "table": "优先用表格对比",
}


async def ensure_user_profiles_table() -> None:
    """建画像表（幂等）。schema.sql 不会自动执行，这里补上。"""
    async with get_business_pool().connection() as conn:
        await conn.execute(_DDL_TABLE)
        await conn.execute(_DDL_INDEX)


def format_profile(profile: dict[str, Any], token_budget: int | None = None) -> str:
    """把画像字典格式化成一段提示词文本；没有有效内容时返回空串。"""
    budget = token_budget or settings.MEMORY_PROFILE_TOKEN_BUDGET

    language = str(profile.get("preferred_language") or "zh")
    style = str(profile.get("answer_style") or "detailed")
    topics = [str(topic) for topic in (profile.get("recent_topics") or []) if str(topic).strip()]

    # 全是默认值 = 没有有效画像，别占用预算
    if language == "zh" and style == "detailed" and not topics:
        return ""

    lines = ["用户画像（跨会话记忆，供参考）："]
    lines.append(f"- 偏好语言：{_LANG_NAMES.get(language, language)}")
    lines.append(f"- 回答风格：{_STYLE_NAMES.get(style, style)}")
    if topics:
        lines.append(f"- 最近关注：{'、'.join(topics[:5])}")

    text = "\n".join(lines)
    if estimate_tokens(text) > budget:
        text = compress_tool_output(text, budget)
    return text


async def load_profile_context(user_id: str, token_budget: int | None = None) -> str:
    """读一次画像并格式化；任何失败都降级为空串（记忆是增强，不是依赖）。"""
    if not user_id or not settings.MEMORY_ENABLED:
        return ""

    try:
        from src.memory.user_profile_store import get_user_profile_store

        store = await get_user_profile_store()
        profile = await store.get_profile(user_id)
    except Exception:  # noqa: BLE001 - 读画像失败不能影响本轮回答
        logger.warning("读取用户画像失败，按无画像处理", extra={"user_id": user_id}, exc_info=True)
        return ""

    context = format_profile(profile, token_budget)
    if context:
        logger.info(
            "用户画像已注入稳定区",
            extra={"user_id": user_id, "tokens": estimate_tokens(context)},
        )
    return context
