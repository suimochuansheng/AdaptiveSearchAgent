"""
基础设施增强功能集成测试（第 1~4 阶段）
覆盖：配置开关、长期记忆存储/解析器、动态工具图结构、异步任务 API 端点

运行方式：
    poetry run pytest tests/test_infra_enhancements.py -v

注意：所有外部依赖（数据库、Ollama、Celery）均被 mock，测试可离线运行。
Celery 未安装时，相关测试会自动跳过。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# ═══════════════════════════════════════════════════════════════
# 全局夹具：API 测试客户端
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def client():
    """创建 FastAPI TestClient，并 mock 掉 api_main 中的数据库/Redis 依赖"""
    # 关键：api_main.py 中使用了 `from src import checkpointer`，
    # 因此我们需要 mock `api_main.checkpointer` 模块及其 `_global_pool`
    with (
        patch("api_main.checkpointer") as mock_checkpointer_mod,
        patch("api_main.redis_client") as mock_redis,
    ):
        # Mock Redis 锁（避免实际连接）
        mock_redis.lock.return_value = AsyncMock()
        mock_redis.lock.return_value.acquire = AsyncMock(return_value=True)
        mock_redis.lock.return_value.release = AsyncMock()

        # Mock 数据库连接池（_global_pool）
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()

        # 配置连接池的上下文管理器链
        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
        mock_conn.execute = AsyncMock()  # 新增：模拟直接 execute 调用

        # 将 mock 池赋值给 checkpointer 模块的 _global_pool
        mock_checkpointer_mod._global_pool = mock_pool

        # 延迟导入 api_main，确保所有 patch 生效
        from api_main import app

        return TestClient(app)


# ═══════════════════════════════════════════════════════════════
# 第 1 阶段：配置开关测试
# ═══════════════════════════════════════════════════════════════


class TestPhase1Config:
    """验证 config.py 新增的 4 个配置项"""

    def test_new_config_defaults(self):
        from config import settings

        assert hasattr(settings, "ENABLE_DYNAMIC_TOOLS")
        assert hasattr(settings, "ENABLE_MEMORY")
        assert hasattr(settings, "ENABLE_ASYNC_TASK")
        assert hasattr(settings, "ASYNC_TASK_THRESHOLD_SECONDS")

        # 验证默认值（零破坏性）
        assert settings.ENABLE_DYNAMIC_TOOLS is False
        assert settings.ENABLE_MEMORY is False
        assert settings.ENABLE_ASYNC_TASK is False
        assert settings.ASYNC_TASK_THRESHOLD_SECONDS == 120


# ═══════════════════════════════════════════════════════════════
# 第 2 阶段：长期记忆模块测试
# ═══════════════════════════════════════════════════════════════


class TestPhase2MemoryStore:
    """测试 UserProfileStore 的 CRUD 和话题推送逻辑"""

    @pytest_asyncio.fixture
    async def store_with_mock_db(self):
        """返回一个 UserProfileStore，其内部数据库连接被 mock"""
        from src.memory.user_profile_store import UserProfileStore

        store = UserProfileStore()

        # Mock 连接池和游标
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = AsyncMock()

        mock_pool.connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor

        store._pool = mock_pool
        store._checkpointer = MagicMock()
        return store, mock_cursor

    @pytest.mark.asyncio
    async def test_get_profile_returns_default_when_not_exists(self, store_with_mock_db):
        store, mock_cursor = store_with_mock_db
        mock_cursor.fetchone.return_value = None

        profile = await store.get_profile("non_existent_user")

        assert profile["preferred_language"] == "zh"
        assert profile["answer_style"] == "detailed"
        assert profile["recent_topics"] == []

    @pytest.mark.asyncio
    async def test_get_profile_returns_existing_record(self, store_with_mock_db):
        store, mock_cursor = store_with_mock_db
        mock_cursor.fetchone.return_value = {
            "preferred_language": "en",
            "answer_style": "concise",
            "recent_topics": ["AI", "RAG"],
        }

        profile = await store.get_profile("existing_user")

        assert profile["preferred_language"] == "en"
        assert profile["answer_style"] == "concise"
        assert profile["recent_topics"] == ["AI", "RAG"]

    @pytest.mark.asyncio
    async def test_push_topic_inserts_at_front_and_truncates(self, store_with_mock_db):
        store, mock_cursor = store_with_mock_db
        mock_cursor.fetchone.return_value = {
            "preferred_language": "zh",
            "answer_style": "detailed",
            "recent_topics": ["old1", "old2", "old3", "old4", "old5"],
        }

        with patch.object(store, "upsert_profile", new=AsyncMock()) as mock_upsert:
            await store.push_topic("test_user", "new_topic", max_topics=3)
            mock_upsert.assert_called_once()
            args = mock_upsert.call_args[1]
            assert args["recent_topics"] == ["new_topic", "old1", "old2"]


class TestPhase2MemoryResolver:
    """测试 MemoryResolver 构建带偏好的 System Prompt"""

    @pytest.mark.asyncio
    async def test_build_prompt_injects_preferences(self):
        from src.memory.memory_resolver import build_planner_prompt_with_memory

        mock_profile = {
            "preferred_language": "zh",
            "answer_style": "table",
            "recent_topics": ["数据库优化", "索引设计"],
        }
        mock_store = AsyncMock()
        mock_store.get_profile.return_value = mock_profile

        with patch("src.memory.memory_resolver.get_user_profile_store", return_value=mock_store):
            prompt = await build_planner_prompt_with_memory("test query", "thread_123")

        assert "中文" in prompt
        assert "表格对比" in prompt
        assert "数据库优化" in prompt
        assert "索引设计" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_uses_defaults_when_profile_empty(self):
        from src.memory.memory_resolver import build_planner_prompt_with_memory

        mock_profile = {
            "preferred_language": "zh",
            "answer_style": "detailed",
            "recent_topics": [],
        }
        mock_store = AsyncMock()
        mock_store.get_profile.return_value = mock_profile

        with patch("src.memory.memory_resolver.get_user_profile_store", return_value=mock_store):
            prompt = await build_planner_prompt_with_memory("test query", "thread_123")

        assert "中文" in prompt
        assert "详细的解释" in prompt
        assert "用户最近关注" not in prompt


# ═══════════════════════════════════════════════════════════════
# 第 3 阶段：动态工具路由图结构测试
# ═══════════════════════════════════════════════════════════════


class TestPhase3ToolDemo:
    """验证 tool_demo.py 的图结构正确性（不实际调用 LLM）"""

    def test_build_graph_has_correct_structure(self):
        from src.agents.tool_demo import TOOLS, build_dynamic_tool_graph

        with patch("src.agents.tool_demo.ChatOllama") as MockOllama:
            mock_model = MagicMock()
            mock_model.bind_tools.return_value = mock_model
            MockOllama.return_value = mock_model

            graph = build_dynamic_tool_graph()

        # 验证工具列表包含 3 个工具
        assert len(TOOLS) == 3
        tool_names = [t.__name__ for t in TOOLS]
        assert "search_knowledge" in tool_names
        assert "search_tavily" in tool_names
        assert "calculator" in tool_names

        # 验证 bind_tools 被调用
        mock_model.bind_tools.assert_called_once()

        # 验证图已编译（有 invoke 方法）
        assert hasattr(graph, "invoke")


# ═══════════════════════════════════════════════════════════════
# 第 4 阶段：异步任务 API 端点测试
# ═══════════════════════════════════════════════════════════════


class TestPhase4AsyncAPI:
    """测试新增的 GET /api/task/{thread_id}/status 和 /api/task/{thread_id}/result"""

    def test_status_endpoint_returns_404_when_task_not_found(self, client):
        """任务不存在时返回 404"""
        # 模拟数据库查询返回 None
        with patch("api_main.checkpointer") as mock_cp:
            mock_cp._global_pool = MagicMock()
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()
            mock_cursor.fetchone.return_value = None

            mock_cp._global_pool.connection.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_cp._global_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute.return_value = mock_cursor
            # 注意：这里 mock_conn.execute 返回 cursor，cursor.fetchone 返回 None

            response = client.get("/api/task/non_existent/status")
            assert response.status_code == 404
            assert "Task not found" in response.json()["detail"]

    def test_status_endpoint_returns_task_state_when_exists(self, client):
        """任务存在时返回状态信息"""
        with patch("api_main.checkpointer") as mock_cp:
            mock_cp._global_pool = MagicMock()
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()
            mock_cursor.fetchone.return_value = {
                "thread_id": "task_123",
                "state": "running",
                "interrupt_time": None,
            }
            mock_cp._global_pool.connection.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_cp._global_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute.return_value = mock_cursor

            response = client.get("/api/task/task_123/status")
            assert response.status_code == 200
            data = response.json()
            assert data["state"] == "running"
            assert data["thread_id"] == "task_123"

    def test_result_endpoint_returns_202_when_task_not_completed(self, client):
        """任务未完成时返回 202（仍在处理）"""
        with patch("api_main.checkpointer") as mock_cp:
            mock_cp._global_pool = MagicMock()
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()
            mock_cursor.fetchone.return_value = {
                "thread_id": "task_123",
                "state": "running",
            }
            mock_cp._global_pool.connection.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_cp._global_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute.return_value = mock_cursor

            response = client.get("/api/task/task_123/result")
            assert response.status_code == 202
            assert "still processing" in response.json()["message"]

    def test_result_endpoint_returns_final_report_when_completed(self, client):
        """任务完成时返回最终报告"""
        with (
            patch("api_main.checkpointer") as mock_cp,
            patch("api_main.get_graph") as mock_get_graph,
        ):
            # Mock checkpointer 返回 completed 状态
            mock_cp._global_pool = MagicMock()
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()
            mock_cursor.fetchone.return_value = {
                "thread_id": "task_123",
                "state": "completed",
            }
            mock_cp._global_pool.connection.return_value.__aenter__ = AsyncMock(
                return_value=mock_conn
            )
            mock_cp._global_pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_conn.execute.return_value = mock_cursor

            # Mock graph.aget_state 返回最终报告
            mock_state = AsyncMock()
            mock_state.values = {
                "final_report": "这是测试生成的最终报告。",
                "total_tokens": 1234,
            }
            mock_graph = AsyncMock()
            mock_graph.aget_state.return_value = mock_state
            mock_get_graph.return_value = mock_graph

            response = client.get("/api/task/task_123/result")
            assert response.status_code == 200
            data = response.json()
            assert data["final_report"] == "这是测试生成的最终报告。"
            assert data["total_tokens"] == 1234


# ═══════════════════════════════════════════════════════════════
# 额外验证：Celery 任务模块可导入
# ═══════════════════════════════════════════════════════════════


class TestPhase4CeleryTask:
    """验证 src/tasks.py 模块可正常导入且 Celery 配置正确"""

    def test_tasks_module_imports_and_config(self):
        """验证 Celery 配置（若 Celery 已安装）"""
        try:
            from src.tasks import celery_app

            # 验证 broker_url 使用了 db=1
            assert "redis://" in celery_app.conf.broker_url
            assert "/1" in celery_app.conf.broker_url  # 确保使用 db=1
            assert celery_app.conf.task_soft_time_limit == 580
            assert celery_app.conf.task_time_limit == 600
            assert celery_app.conf.task_acks_late is True
        except ImportError as e:
            pytest.skip(f"Celery not installed, skipping import test: {e}")
