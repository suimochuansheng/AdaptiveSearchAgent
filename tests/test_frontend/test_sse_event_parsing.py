"""SSE 事件解析单元测试 —— 覆盖 _handle_sse_event 所有事件类型。

测试目标：验证前端对后端 SSE 事件的解析逻辑在各种输入下的正确性。
无需网络，无需 Chainlit 运行时，纯逻辑测试。
"""

from unittest.mock import AsyncMock

import pytest

# 从 frontend 模块导入待测函数
from frontend.app import _handle_sse_event


# =============================================================================
# 辅助函数：创建 mock Message 对象
# =============================================================================
def _mock_msg() -> AsyncMock:
    """创建一个带 stream_token 方法的 mock Message。"""
    msg = AsyncMock()
    msg.stream_token = AsyncMock()
    return msg


# =============================================================================
# 1. status 事件 — 中间状态推送
# =============================================================================
class TestStatusEvent:
    """验证 status 事件的格式化输出（置信度、迭代轮数、缺失信息）。"""

    @pytest.mark.asyncio
    async def test_完整状态事件(self) -> None:
        """status 事件含 confidence、iteration、missing_info → 应输出所有三项。"""
        msg = _mock_msg()
        data = {"confidence": 0.75, "iteration": 2, "missing_info": "缺少NLP最新进展"}

        stopped = await _handle_sse_event("status", data, msg)

        assert stopped is False  # status 不终止流
        msg.stream_token.assert_called_once()
        token_text: str = msg.stream_token.call_args[0][0]
        assert "第 2 轮评估" in token_text
        assert "75%" in token_text
        assert "缺少NLP最新进展" in token_text

    @pytest.mark.asyncio
    async def test_status_无缺失信息(self) -> None:
        """status 事件 missing_info 为空 → 不输出缺失信息行。"""
        msg = _mock_msg()
        data = {"confidence": 0.5, "iteration": 1, "missing_info": ""}

        stopped = await _handle_sse_event("status", data, msg)

        assert stopped is False
        token_text: str = msg.stream_token.call_args[0][0]
        assert "第 1 轮评估" in token_text
        assert "50%" in token_text
        assert "缺失信息" not in token_text

    @pytest.mark.asyncio
    async def test_status_缺失字段使用默认值(self) -> None:
        """status 事件字段缺失 → 使用默认值 0 / 空字符串。"""
        msg = _mock_msg()
        data: dict = {}

        stopped = await _handle_sse_event("status", data, msg)

        assert stopped is False
        token_text: str = msg.stream_token.call_args[0][0]
        assert "第 0 轮评估" in token_text
        assert "0%" in token_text


# =============================================================================
# 2. final 事件 — 最终报告
# =============================================================================
class TestFinalEvent:
    """验证 final 事件：有内容输出、空内容输出、返回 True 终止流。"""

    @pytest.mark.asyncio
    async def test_final_有内容应输出并终止(self) -> None:
        """final 事件含 content → 输出 content 并返回 True（终止流）。"""
        msg = _mock_msg()
        data = {"content": "## 最终报告\n这是测试报告内容。"}

        stopped = await _handle_sse_event("final", data, msg)

        assert stopped is True
        msg.stream_token.assert_called_once_with("## 最终报告\n这是测试报告内容。")

    @pytest.mark.asyncio
    async def test_final_空内容只终止不输出(self) -> None:
        """final 事件 content 为空 → 不调用 stream_token，但仍返回 True。"""
        msg = _mock_msg()
        data = {"content": ""}

        stopped = await _handle_sse_event("final", data, msg)

        assert stopped is True
        msg.stream_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_final_缺少content字段(self) -> None:
        """final 事件没有 content 字段 → 不崩溃，返回 True。"""
        msg = _mock_msg()
        data: dict = {}

        stopped = await _handle_sse_event("final", data, msg)

        assert stopped is True
        msg.stream_token.assert_not_called()


# =============================================================================
# 3. kpi 事件 — 执行统计
# =============================================================================
class TestKpiEvent:
    """验证 KPI 事件的统计数据格式化。"""

    @pytest.mark.asyncio
    async def test_kpi_完整数据(self) -> None:
        """kpi 事件包含完整 KPI 数据 → 应格式化输出所有统计项。"""
        msg = _mock_msg()
        data = {
            "data": {
                "current_llm": "deepseek",
                "input_tokens": 1500,
                "output_tokens": 800,
                "total_tokens": 2300,
                "confidence_score": 0.85,
                "model_switches": 2,
                "elapsed_seconds": 12.5,
            }
        }

        stopped = await _handle_sse_event("kpi", data, msg)

        assert stopped is False  # kpi 不终止流
        token_text: str = msg.stream_token.call_args[0][0]
        assert "执行统计" in token_text
        assert "deepseek" in token_text
        assert "1,500" in token_text
        assert "800" in token_text
        assert "2,300" in token_text
        assert "85%" in token_text
        assert "2 次" in token_text
        assert "12.5s" in token_text

    @pytest.mark.asyncio
    async def test_kpi_空data使用默认值(self) -> None:
        """kpi 事件 data 为空 → 使用默认值，不崩溃。"""
        msg = _mock_msg()
        data: dict = {"data": {}}

        stopped = await _handle_sse_event("kpi", data, msg)

        assert stopped is False
        token_text: str = msg.stream_token.call_args[0][0]
        assert "0" in token_text  # 各项默认值为 0

    @pytest.mark.asyncio
    async def test_kpi_缺少data字段(self) -> None:
        """kpi 事件完全缺少 data 字段 → 不崩溃，使用默认值。"""
        msg = _mock_msg()
        data: dict = {}

        stopped = await _handle_sse_event("kpi", data, msg)

        assert stopped is False
        token_text: str = msg.stream_token.call_args[0][0]
        assert "执行统计" in token_text


# =============================================================================
# 4. error 事件 — 错误处理
# =============================================================================
class TestErrorEvent:
    """验证 error 事件：输出错误信息并终止流。"""

    @pytest.mark.asyncio
    async def test_error_有消息应输出并终止(self) -> None:
        """error 事件含 message → 输出错误信息并返回 True。"""
        msg = _mock_msg()
        data = {"message": "模型调用超时，请重试"}

        stopped = await _handle_sse_event("error", data, msg)

        assert stopped is True
        token_text: str = msg.stream_token.call_args[0][0]
        assert "错误" in token_text
        assert "模型调用超时，请重试" in token_text

    @pytest.mark.asyncio
    async def test_error_无消息使用默认文本(self) -> None:
        """error 事件缺少 message → 使用"未知错误"。"""
        msg = _mock_msg()
        data: dict = {}

        stopped = await _handle_sse_event("error", data, msg)

        assert stopped is True
        token_text: str = msg.stream_token.call_args[0][0]
        assert "未知错误" in token_text


# =============================================================================
# 5. 未知事件类型 / 边界情况
# =============================================================================
class TestUnknownEvent:
    """验证未知事件类型和边界输入的处理。"""

    @pytest.mark.asyncio
    async def test_未知事件类型不终止流(self) -> None:
        """未知事件类型 → 返回 False，不调用 stream_token。"""
        msg = _mock_msg()
        data: dict = {}

        stopped = await _handle_sse_event("unknown_type", data, msg)

        assert stopped is False
        msg.stream_token.assert_not_called()

    @pytest.mark.asyncio
    async def test_空事件类型不终止流(self) -> None:
        """空字符串事件类型 → 返回 False，不崩溃。"""
        msg = _mock_msg()
        data: dict = {}

        stopped = await _handle_sse_event("", data, msg)

        assert stopped is False

    @pytest.mark.asyncio
    async def test_interrupt事件类型不在此处理(self) -> None:
        """interrupt 类型应由上层 _stream_agent_with_interrupt 处理，
        _handle_sse_event 中应作为未知事件返回 False。"""
        msg = _mock_msg()
        data = {"question": "需要批准", "data": {}}

        stopped = await _handle_sse_event("interrupt", data, msg)

        assert stopped is False
