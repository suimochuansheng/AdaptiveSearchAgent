"""前端流控逻辑集成测试 —— mock HTTP 覆盖所有执行路径。

测试目标：验证 _stream_agent_with_interrupt 在各种
后端响应场景下的行为，覆盖请求流图中的所有分支：
  - 路径A：正常首次执行（status → final → kpi）
  - 路径B：中断恢复（interrupt → 审批 → 恢复执行）
  - 路径B2：拒绝旧会话（resume_value="no"）
  - 错误路径：HTTP 429 / 500 / 超时 / JSON 解析失败

通过 mock httpx.AsyncClient 模拟后端 SSE 流，无需真实后端服务。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# 单下划线 _xxx 是模块私有标识，mypy 默认禁止外部跨文件导入，判定该属性不存在
# 但在测试中我们需要直接调用这些函数来验证其行为，因此使用 type: ignore 来绕过 mypy 的检查。
from frontend.app import (
    _stream_agent_with_interrupt,  # type: ignore[attr-defined]
)

# =============================================================================
# 辅助工具
# =============================================================================


def _make_sse_chunks(lines: list[str]) -> list[str]:
    """将字符串列表包装为 SSE 格式（每行加 "data: " 前缀）。"""
    return [f"data: {line}" if line else "" for line in lines]


def _mock_msg_with_collector() -> tuple[AsyncMock, list[str]]:
    """创建一个 mock Message，同时收集所有 stream_token 调用的文本。"""
    collected: list[str] = []
    msg = AsyncMock()
    msg.update = AsyncMock()

    async def _collect(text: str) -> None:
        collected.append(text)

    msg.stream_token = _collect
    return msg, collected


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
    # stream 是同步方法 → MagicMock
    client.stream = MagicMock(return_value=_make_mock_stream_response(sse_lines, status_code))
    return client


# =============================================================================
# 1. _stream_agent_with_interrupt 正常流程
# =============================================================================
class TestStreamAgentWithInterruptNormalFlow:
    """测试 _stream_agent_with_interrupt 在正常 SSE 流下的各种序列组合。"""

    @pytest.mark.asyncio
    async def test_完整流程_status_final_kpi(self) -> None:
        """模拟 status → final → kpi 完整序列，验证所有事件都被消费。"""
        msg, collected = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "status", "confidence": 0.6, "iteration": 1}),
                json.dumps({"type": "final", "content": "最终答案"}),
                json.dumps({"type": "kpi", "data": {"elapsed_seconds": 5.0}}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent_with_interrupt(
            thread_id="test-001",
            message="测试问题",
            resume_value=None,
            msg=msg,
            client=client,
        )

        client.stream.assert_called_once()
        call_kwargs = client.stream.call_args
        assert call_kwargs[0][0] == "POST"  # 第一个位置参数是 HTTP 方法
        assert "/chat/stream" in call_kwargs[0][1]  # 第二个位置参数是 URL
        payload = call_kwargs[1]["json"]
        assert payload["thread_id"] == "test-001"
        assert payload["message"] == "测试问题"
        assert "resume_value" not in payload
        assert len(collected) >= 2
        assert msg.update.called

    @pytest.mark.asyncio
    async def test_首次执行_message为None时使用空字符串(self) -> None:
        """message=None → payload 中 message 应为 ""。"""
        msg, _ = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "final", "content": "done"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent_with_interrupt(
            thread_id="test-001",
            message=None,
            resume_value=None,
            msg=msg,
            client=client,
        )

        call_kwargs = client.stream.call_args
        assert call_kwargs[1]["json"]["message"] == ""

    @pytest.mark.asyncio
    async def test_恢复执行_resume_value传yes(self) -> None:
        """resume_value="yes" → payload 含 resume_value，message=None。"""
        msg, _ = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "final", "content": "恢复后完成"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent_with_interrupt(
            thread_id="test-001",
            message="会被覆盖",
            resume_value="yes",
            msg=msg,
            client=client,
        )

        call_kwargs = client.stream.call_args
        payload = call_kwargs[1]["json"]
        assert payload["resume_value"] == "yes"
        assert payload["message"] is None

    @pytest.mark.asyncio
    async def test_仅kpi事件无final_流正常结束(self) -> None:
        """只有 status + kpi 事件，没有 final → 流消费完毕后正常结束。"""
        msg, _ = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "status", "confidence": 0.9, "iteration": 3}),
                json.dumps({"type": "kpi", "data": {"elapsed_seconds": 10.0}}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent_with_interrupt(
            thread_id="test-001",
            message="问题",
            resume_value=None,
            msg=msg,
            client=client,
        )

        assert msg.update.called


# =============================================================================
# 2. _stream_agent_with_interrupt 错误路径
# =============================================================================
class TestStreamAgentWithInterruptErrorPaths:
    """测试 _stream_agent_with_interrupt 的各种错误和边界场景。"""

    @pytest.mark.asyncio
    async def test_http_429_错误传播(self) -> None:
        """后端返回 429 → HTTPStatusError 应向上传播给调用方。"""
        msg, _ = _mock_msg_with_collector()
        client = _make_mock_client([], status_code=429)

        with pytest.raises(httpx.HTTPStatusError):
            await _stream_agent_with_interrupt(
                thread_id="test-001",
                message="问题",
                resume_value=None,
                msg=msg,
                client=client,
            )

    @pytest.mark.asyncio
    async def test_http_500_错误传播(self) -> None:
        """后端返回 500 → HTTPStatusError 应向上传播。"""
        msg, _ = _mock_msg_with_collector()
        client = _make_mock_client([], status_code=500)

        with pytest.raises(httpx.HTTPStatusError):
            await _stream_agent_with_interrupt(
                thread_id="test-001",
                message="问题",
                resume_value=None,
                msg=msg,
                client=client,
            )

    @pytest.mark.asyncio
    async def test_invalid_json_行被跳过不崩溃(self) -> None:
        """SSE 流中包含非法 JSON → 跳过该行，继续处理后续有效事件。"""
        msg, collected = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                "这不是合法JSON{{{",
                json.dumps({"type": "status", "confidence": 0.5, "iteration": 1}),
                json.dumps({"type": "final", "content": "忽略非法行后仍正常"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent_with_interrupt(
            thread_id="test-001",
            message="问题",
            resume_value=None,
            msg=msg,
            client=client,
        )

        assert len(collected) >= 2

    @pytest.mark.asyncio
    async def test_空字符串行被跳过(self) -> None:
        """SSE 流中混入空行 → 不影响正常事件处理。"""
        msg, collected = _mock_msg_with_collector()
        sse_lines = [
            "",
            f"data: {json.dumps({'type': 'final', 'content': '正确'})}",
        ]

        client = _make_mock_client(sse_lines)

        await _stream_agent_with_interrupt(
            thread_id="test-001",
            message="问题",
            resume_value=None,
            msg=msg,
            client=client,
        )

        assert len(collected) >= 1

    @pytest.mark.asyncio
    async def test_非data前缀行被跳过(self) -> None:
        """不以 "data: " 开头的行 → 被跳过，不崩溃。"""
        msg, collected = _mock_msg_with_collector()
        sse_lines = [
            "event: ping",
            f"data: {json.dumps({'type': 'final', 'content': 'ok'})}",
        ]

        client = _make_mock_client(sse_lines)

        await _stream_agent_with_interrupt(
            thread_id="test-001",
            message="问题",
            resume_value=None,
            msg=msg,
            client=client,
        )

        assert len(collected) >= 1


# =============================================================================
# 3. _stream_agent_with_interrupt 中断处理
# =============================================================================
class TestStreamAgentWithInterrupt:
    """测试带中断审批的流式请求。"""

    @pytest.mark.asyncio
    async def test_正常流程无中断返回None(self) -> None:
        """无 interrupt 事件 → 正常结束，返回 None。"""
        msg, _ = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "status", "confidence": 0.5, "iteration": 1}),
                json.dumps({"type": "final", "content": "完成"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        result = await _stream_agent_with_interrupt(
            thread_id="test-001",
            message="问题",
            resume_value=None,
            msg=msg,
            client=client,
        )

        assert result is None
        assert msg.update.called

    @pytest.mark.asyncio
    async def test_interrupt事件_弹窗并返回用户选择(self) -> None:
        """收到 interrupt 事件 → 弹出审批框 → 返回用户选择（mock 为 "yes"）。"""
        msg, collected = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps(
                    {
                        "type": "interrupt",
                        "question": "是否批准继续？",
                        "data": {},
                    }
                ),
            ]
        )

        client = _make_mock_client(sse_lines)

        mock_ask_response = MagicMock()
        mock_ask_response.get.return_value = "yes"
        mock_ask_msg_instance = AsyncMock()
        mock_ask_msg_instance.send = AsyncMock(return_value=mock_ask_response)

        mock_wait_msg_instance = AsyncMock()
        mock_wait_msg_instance.send = AsyncMock()
        mock_wait_msg_instance.update = AsyncMock()

        with (
            patch(
                "frontend.app.cl.AskActionMessage",
                return_value=mock_ask_msg_instance,
            ),
            patch(
                "frontend.app.cl.Message",
                return_value=mock_wait_msg_instance,
            ),
            patch("frontend.app.cl.Action"),
        ):
            result = await _stream_agent_with_interrupt(
                thread_id="test-001",
                message="问题",
                resume_value=None,
                msg=msg,
                client=client,
            )

        assert result == "yes"

    @pytest.mark.asyncio
    async def test_interrupt拒绝返回no(self) -> None:
        """用户选择拒绝 → 返回 "no"。"""
        msg, _ = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "interrupt", "question": "请审批"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        mock_ask_response = MagicMock()
        mock_ask_response.get.return_value = "no"
        mock_ask_msg_instance = AsyncMock()
        mock_ask_msg_instance.send = AsyncMock(return_value=mock_ask_response)

        mock_wait_msg_instance = AsyncMock()
        mock_wait_msg_instance.send = AsyncMock()
        mock_wait_msg_instance.update = AsyncMock()

        with (
            patch(
                "frontend.app.cl.AskActionMessage",
                return_value=mock_ask_msg_instance,
            ),
            patch(
                "frontend.app.cl.Message",
                return_value=mock_wait_msg_instance,
            ),
            patch("frontend.app.cl.Action"),
        ):
            result = await _stream_agent_with_interrupt(
                thread_id="test-001",
                message="问题",
                resume_value=None,
                msg=msg,
                client=client,
            )

        assert result == "no"

    @pytest.mark.asyncio
    async def test_interrupt_res为None时默认返回no(self) -> None:
        """AskActionMessage.send() 返回 None → 默认 choice="no"。"""
        msg, _ = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "interrupt", "question": "审批"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        mock_ask_msg_instance = AsyncMock()
        mock_ask_msg_instance.send = AsyncMock(return_value=None)

        mock_wait_msg_instance = AsyncMock()
        mock_wait_msg_instance.send = AsyncMock()
        mock_wait_msg_instance.update = AsyncMock()

        with (
            patch(
                "frontend.app.cl.AskActionMessage",
                return_value=mock_ask_msg_instance,
            ),
            patch(
                "frontend.app.cl.Message",
                return_value=mock_wait_msg_instance,
            ),
            patch("frontend.app.cl.Action"),
        ):
            result = await _stream_agent_with_interrupt(
                thread_id="test-001",
                message="问题",
                resume_value=None,
                msg=msg,
                client=client,
            )

        assert result == "no"


# =============================================================================
# 4. _stream_agent_with_interrupt 错误路径
# =============================================================================
class TestStreamAgentWithInterruptErrors:
    """测试带中断流式请求的错误处理。"""

    @pytest.mark.asyncio
    async def test_error事件返回None(self) -> None:
        """收到 error 事件 → _handle_sse_event 返回 True → 整体返回 None。"""
        msg, collected = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "error", "message": "服务异常"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        result = await _stream_agent_with_interrupt(
            thread_id="test-001",
            message="问题",
            resume_value=None,
            msg=msg,
            client=client,
        )

        assert result is None
        assert any("服务异常" in t for t in collected)


# =============================================================================
# 5. 请求负载构造验证
# =============================================================================
class TestPayloadConstruction:
    """验证不同参数组合下构造的 HTTP 请求体是否正确。"""

    @pytest.mark.asyncio
    async def test_首次执行payload(self) -> None:
        """resume_value=None, message="hello" → payload 只含 thread_id + message。"""
        msg, _ = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "final", "content": "ok"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent_with_interrupt(
            thread_id="th-001",
            message="hello world",
            resume_value=None,
            msg=msg,
            client=client,
        )

        payload = client.stream.call_args[1]["json"]
        assert payload["thread_id"] == "th-001"
        assert payload["message"] == "hello world"
        assert "resume_value" not in payload

    @pytest.mark.asyncio
    async def test_恢复执行payload(self) -> None:
        """resume_value="yes" → payload 含 resume_value，message=None。"""
        msg, _ = _mock_msg_with_collector()
        sse_lines = _make_sse_chunks(
            [
                json.dumps({"type": "final", "content": "ok"}),
            ]
        )

        client = _make_mock_client(sse_lines)

        await _stream_agent_with_interrupt(
            thread_id="th-002",
            message="ignored",
            resume_value="yes",
            msg=msg,
            client=client,
        )

        payload = client.stream.call_args[1]["json"]
        assert payload["thread_id"] == "th-002"
        assert payload["resume_value"] == "yes"
        assert payload["message"] is None
