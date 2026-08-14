"""前端流控逻辑集成测试 —— mock HTTP 覆盖所有执行路径。

测试目标：验证 _stream_agent 在各种后端响应场景下的行为：
  - 正常首次执行（status → final → kpi）
  - 思考事件分流（status/search_result/kpi → thinking_step）
  - 输出事件分流（final/error → output_msg）
  - 错误路径：HTTP 429 / 500 / JSON 解析失败

通过 mock httpx.AsyncClient 模拟后端 SSE 流，无需真实后端服务。

注：前端重构为"全自动模式"后，原 _stream_agent_with_interrupt（带中断审批）
已被 _stream_agent 取代，签名改为 keyword-only：
_stream_agent(*, thread_id, message, thinking_step, output_msg, client)。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# 单下划线 _xxx 是模块私有标识，测试中直接调用以验证其行为。
from frontend.app import _stream_agent  # type: ignore[attr-defined]

# =============================================================================
# 辅助工具
# =============================================================================


def _make_sse_chunks(lines: list[str]) -> list[str]:
    """将字符串列表包装为 SSE 格式（每行加 "data: " 前缀）。"""
    return [f"data: {line}" if line else "" for line in lines]


def _make_step_and_msg() -> tuple[AsyncMock, AsyncMock, list[str]]:
    """创建 mock thinking_step 与 output_msg，并收集 output_msg 的 stream_token 文本。"""
    collected: list[str] = []

    thinking_step = AsyncMock()
    thinking_step.stream_token = AsyncMock()

    output_msg = AsyncMock()
    output_msg.update = AsyncMock()

    async def _collect(text: str) -> None:
        collected.append(text)

    output_msg.stream_token = _collect
    return thinking_step, output_msg, collected


def _make_mock_stream_response(sse_lines: list[str], status_code: int = 200):
    """创建一个满足异步上下文管理器协议的 mock SSE 响应对象。"""

    class MockStreamResponse:
        def __init__(self) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("error", request=AsyncMock(), response=AsyncMock())

        async def aiter_lines(self) -> AsyncIterator[str]:
            for line in sse_lines:
                yield line

        async def __aenter__(self) -> MockStreamResponse:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    return MockStreamResponse()


def _make_mock_client(sse_lines: list[str], status_code: int = 200) -> AsyncMock:
    """创建 mock httpx.AsyncClient，其 stream 方法返回异步上下文管理器。

    关键：httpx.AsyncClient.stream() 是同步方法（返回 async cm），
    必须用 MagicMock 而非 AsyncMock，否则调用后会返回 coroutine。
    """
    client = AsyncMock()
    client.stream = MagicMock(return_value=_make_mock_stream_response(sse_lines, status_code))
    return client


# =============================================================================
# 1. _stream_agent 正常流程
# =============================================================================
class TestStreamAgentNormalFlow:
    """测试 _stream_agent 在正常 SSE 流下的各种序列组合。"""

    @pytest.mark.asyncio
    async def test_完整流程_status_final_kpi(self) -> None:
        """模拟 status → final → kpi 完整序列，验证事件分流消费。"""
        thinking_step, output_msg, collected = _make_step_and_msg()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "status", "confidence": 0.6, "iteration": 1}),
                json.dumps({"type": "final", "content": "最终答案"}),
                json.dumps({"type": "kpi", "data": {"elapsed_seconds": 5.0}}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent(
            thread_id="test-001",
            message="测试问题",
            thinking_step=thinking_step,
            output_msg=output_msg,
            client=client,
        )

        client.stream.assert_called_once()
        call_kwargs = client.stream.call_args
        assert call_kwargs[0][0] == "POST"
        assert "/chat/stream" in call_kwargs[0][1]
        payload = call_kwargs[1]["json"]
        assert payload["thread_id"] == "test-001"
        assert payload["message"] == "测试问题"
        # status/kpi 写入思考 Step，final 写入主 Message
        assert thinking_step.stream_token.called
        assert "最终答案" in collected
        assert output_msg.update.called

    @pytest.mark.asyncio
    async def test_仅kpi事件无final_流正常结束(self) -> None:
        """只有 status + kpi 事件，没有 final → 流消费完毕后正常结束。"""
        thinking_step, output_msg, _ = _make_step_and_msg()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "status", "confidence": 0.9, "iteration": 3}),
                json.dumps({"type": "kpi", "data": {"elapsed_seconds": 10.0}}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent(
            thread_id="test-001",
            message="问题",
            thinking_step=thinking_step,
            output_msg=output_msg,
            client=client,
        )

        assert thinking_step.stream_token.called
        assert output_msg.update.called

    @pytest.mark.asyncio
    async def test_final事件终止流后不再消费后续事件(self) -> None:
        """final 事件后应调用 output_msg.update 并终止，不再消费后续事件。"""
        thinking_step, output_msg, collected = _make_step_and_msg()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "final", "content": "完成"}),
                json.dumps({"type": "kpi", "data": {"elapsed_seconds": 1.0}}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent(
            thread_id="test-001",
            message="问题",
            thinking_step=thinking_step,
            output_msg=output_msg,
            client=client,
        )

        assert "完成" in collected
        # final 后 return，kpi 不应写入思考 Step
        assert not thinking_step.stream_token.called


# =============================================================================
# 2. _stream_agent 错误路径
# =============================================================================
class TestStreamAgentErrorPaths:
    """测试 _stream_agent 的各种错误和边界场景。"""

    @pytest.mark.asyncio
    async def test_http_429_错误传播(self) -> None:
        """后端返回 429 → HTTPStatusError 应向上传播给调用方。"""
        thinking_step, output_msg, _ = _make_step_and_msg()
        client = _make_mock_client([], status_code=429)

        with pytest.raises(httpx.HTTPStatusError):
            await _stream_agent(
                thread_id="test-001",
                message="问题",
                thinking_step=thinking_step,
                output_msg=output_msg,
                client=client,
            )

    @pytest.mark.asyncio
    async def test_http_500_错误传播(self) -> None:
        """后端返回 500 → HTTPStatusError 应向上传播。"""
        thinking_step, output_msg, _ = _make_step_and_msg()
        client = _make_mock_client([], status_code=500)

        with pytest.raises(httpx.HTTPStatusError):
            await _stream_agent(
                thread_id="test-001",
                message="问题",
                thinking_step=thinking_step,
                output_msg=output_msg,
                client=client,
            )

    @pytest.mark.asyncio
    async def test_invalid_json_行被跳过不崩溃(self) -> None:
        """SSE 流中包含非法 JSON → 跳过该行，继续处理后续有效事件。"""
        thinking_step, output_msg, collected = _make_step_and_msg()
        sse_lines = _make_sse_chunks(
            [
                "这不是合法JSON{{{",
                json.dumps({"type": "status", "confidence": 0.5, "iteration": 1}),
                json.dumps({"type": "final", "content": "忽略非法行后仍正常"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent(
            thread_id="test-001",
            message="问题",
            thinking_step=thinking_step,
            output_msg=output_msg,
            client=client,
        )

        assert "忽略非法行后仍正常" in collected

    @pytest.mark.asyncio
    async def test_空字符串行被跳过(self) -> None:
        """SSE 流中混入空行 → 不影响正常事件处理。"""
        thinking_step, output_msg, collected = _make_step_and_msg()
        sse_lines = [
            "",
            f"data: {json.dumps({'type': 'final', 'content': '正确'})}",
        ]

        client = _make_mock_client(sse_lines)

        await _stream_agent(
            thread_id="test-001",
            message="问题",
            thinking_step=thinking_step,
            output_msg=output_msg,
            client=client,
        )

        assert "正确" in collected

    @pytest.mark.asyncio
    async def test_非data前缀行被跳过(self) -> None:
        """不以 "data: " 开头的行 → 被跳过，不崩溃。"""
        thinking_step, output_msg, collected = _make_step_and_msg()
        sse_lines = [
            "event: ping",
            f"data: {json.dumps({'type': 'final', 'content': 'ok'})}",
        ]

        client = _make_mock_client(sse_lines)

        await _stream_agent(
            thread_id="test-001",
            message="问题",
            thinking_step=thinking_step,
            output_msg=output_msg,
            client=client,
        )

        assert "ok" in collected


# =============================================================================
# 3. 请求负载构造验证
# =============================================================================
class TestPayloadConstruction:
    """验证请求体构造是否正确。"""

    @pytest.mark.asyncio
    async def test_请求payload只含thread_id和message(self) -> None:
        """payload 应只含 thread_id + message，无其他字段。"""
        thinking_step, output_msg, _ = _make_step_and_msg()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "final", "content": "ok"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent(
            thread_id="th-001",
            message="hello world",
            thinking_step=thinking_step,
            output_msg=output_msg,
            client=client,
        )

        payload = client.stream.call_args[1]["json"]
        assert payload == {"thread_id": "th-001", "message": "hello world"}
