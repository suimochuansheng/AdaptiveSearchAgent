-- ============================================================
-- AdaptiveSearchAgent 长期记忆模块 — 数据库建表脚本
-- 目标数据库：PostgreSQL
-- ============================================================

-- 用户画像表：持久化用户偏好、语言与历史话题
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id         TEXT PRIMARY KEY,
    preferred_language TEXT DEFAULT 'zh',
    answer_style    TEXT DEFAULT 'detailed',
    recent_topics   TEXT[] DEFAULT '{}',
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- 加速按更新时间排序的查询（如"最近活跃用户"）
CREATE INDEX IF NOT EXISTS idx_user_profiles_updated
    ON user_profiles (updated_at);
