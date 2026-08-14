"""
测试 RAG 无结果退化处理功能。

验证 search_knowledge() 在子服务返回 "未找到相关内容" 时正确返回空字符串。
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from src.tools.rag import search_knowledge


class TestRAGFallback:
    """RAG 无结果退化处理测试套件。"""

    @pytest.mark.asyncio
    @patch("src.tools.rag.httpx.AsyncClient")
    async def test_search_knowledge_returns_empty_for_no_result(self, mock_client):
        """验证子服务返回 '未找到相关内容' 时，返回空字符串。"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={"result": "未找到相关内容"})

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)

        result = await search_knowledge("不存在的查询", top_k=3)
        assert result == ""

    @pytest.mark.asyncio
    @patch("src.tools.rag.httpx.AsyncClient")
    async def test_search_knowledge_returns_empty_for_empty_result(self, mock_client):
        """验证子服务返回空 result 时，返回空字符串。"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={"result": ""})

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)

        result = await search_knowledge("测试查询", top_k=3)
        assert result == ""

    @pytest.mark.asyncio
    @patch("src.tools.rag.httpx.AsyncClient")
    async def test_search_knowledge_returns_result_for_valid_result(self, mock_client):
        """验证子服务返回有效结果时，正确透传。"""
        expected_result = "这是有效的检索结果"
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(return_value={"result": expected_result})

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)

        result = await search_knowledge("有效查询", top_k=3)
        assert result == expected_result

    @pytest.mark.asyncio
    @patch("src.tools.rag.httpx.AsyncClient")
    async def test_search_knowledge_handles_timeout_gracefully(self, mock_client):
        """验证超时时返回空字符串，不崩溃。"""
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(side_effect=httpx.TimeoutException("超时"))
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)

        result = await search_knowledge("测试查询", top_k=3)
        assert result == ""

    @pytest.mark.asyncio
    @patch("src.tools.rag.httpx.AsyncClient")
    async def test_search_knowledge_handles_http_error_gracefully(self, mock_client):
        """验证 HTTP 错误时返回空字符串，不崩溃。"""
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "404",
                request=Mock(),
                response=Mock(status_code=404, text="Not Found"),
            )
        )
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)

        result = await search_knowledge("测试查询", top_k=3)
        assert result == ""
