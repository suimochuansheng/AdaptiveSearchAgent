"""用户画像持久化存储层。

复用 checkpointer 的 AsyncConnectionPool，零额外连接配置。
所有数据库操作通过连接池执行，连接池由 init_checkpointer() 统一管理生命周期。
"""

from __future__ import annotations

from src.checkpointer import get_checkpointer
from src.utils.logger import get_logger

logger = get_logger()


class UserProfileStore:
    """用户偏好与历史话题的 CRUD 封装。

    不持有独立数据库连接 —— 通过 get_checkpointer().conn 复用
    checkpointer 模块的 AsyncConnectionPool，无需额外配置。
    """

    def __init__(self) -> None:
        self._pool = None
        self._checkpointer = None

    async def _ensure_pool(self) -> None:
        """延迟初始化：从全局 checkpointer 提取连接池引用。

        get_checkpointer() 为同步函数（返回 AsyncPostgresSaver 实例），
        因此不添加 await。若 checkpointer 尚未初始化，RuntimeError 会自然上抛。
        """
        if self._pool is not None:
            return
        saver = get_checkpointer()
        # saver.conn 即 checkpointer 模块中创建的 AsyncConnectionPool
        self._pool = saver.conn
        self._checkpointer = saver

    async def get_profile(self, user_id: str) -> dict:
        """获取用户画像，不存在时返回安全默认值。

        Args:
            user_id: 用户/线程唯一标识。

        Returns:
            字典永远包含 3 个键：preferred_language, answer_style, recent_topics。
        """
        await self._ensure_pool()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT preferred_language, answer_style, recent_topics "
                    "FROM user_profiles WHERE user_id = %s",
                    (user_id,),
                )
                row = await cur.fetchone()

        if row is None:
            return {
                "preferred_language": "zh",
                "answer_style": "detailed",
                "recent_topics": [],
            }

        topics = row["recent_topics"]
        return {
            "preferred_language": row["preferred_language"],
            "answer_style": row["answer_style"],
            "recent_topics": list(topics) if topics else [],
        }

    async def upsert_profile(
        self,
        user_id: str,
        preferred_language: str | None = None,
        answer_style: str | None = None,
        recent_topics: list[str] | None = None,
    ) -> None:
        """插入或合并更新用户画像。

        COALESCE(EXCLUDED.field, user_profiles.field) 策略：
        - 传入非 None 值时更新对应字段
        - 传入 None 时保留数据库中原值

        Args:
            user_id: 用户唯一标识。
            preferred_language: 偏好语言（如 'zh', 'en'），None 保留原值。
            answer_style: 回答风格（如 'detailed', 'concise', 'table'），None 保留原值。
            recent_topics: 最近话题列表，None 保留原值。
        """
        await self._ensure_pool()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO user_profiles
                        (user_id, preferred_language, answer_style, recent_topics, updated_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        preferred_language = COALESCE(EXCLUDED.preferred_language, user_profiles.preferred_language),
                        answer_style      = COALESCE(EXCLUDED.answer_style,      user_profiles.answer_style),
                        recent_topics     = COALESCE(EXCLUDED.recent_topics,     user_profiles.recent_topics),
                        updated_at        = NOW()
                    """,
                    (user_id, preferred_language, answer_style, recent_topics),
                )

        logger.info(
            "User profile upserted",
            extra={
                "user_id": user_id,
                "preferred_language": preferred_language,
                "answer_style": answer_style,
            },
        )

    async def push_topic(self, user_id: str, topic: str, max_topics: int = 5) -> None:
        """将新话题推入最近话题列表头部，自动去重并截断。

        Args:
            user_id: 用户唯一标识。
            topic: 新话题字符串。
            max_topics: 最多保留的话题数量（默认 5）。
        """
        profile = await self.get_profile(user_id)
        old_topics: list[str] = profile.get("recent_topics", [])

        # 去重：移除旧列表中与 topic 相同的项，再插入到头部
        new_topics = [topic] + [t for t in old_topics if t != topic]
        new_topics = new_topics[:max_topics]

        await self.upsert_profile(user_id, recent_topics=new_topics)

        logger.info(
            "Topic pushed to user profile",
            extra={"user_id": user_id, "topic": topic, "recent_topics": new_topics},
        )


# ── 全局单例 ──────────────────────────────────────────────────

_store: UserProfileStore | None = None


async def get_user_profile_store() -> UserProfileStore:
    """获取全局唯一的 UserProfileStore 实例（延迟初始化）。

    首次调用时创建实例并提取连接池引用；
    后续调用直接返回已缓存的实例。

    Returns:
        全局单例 UserProfileStore。
    """
    global _store
    if _store is None:
        _store = UserProfileStore()
        await _store._ensure_pool()
    return _store
