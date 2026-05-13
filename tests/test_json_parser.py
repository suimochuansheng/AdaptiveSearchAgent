"""json_parser 模块单元测试 — 覆盖四层回退策略和异步 DeepSeek 修复。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.utils.json_parser as jp

# =============================================================================
# 策略1：_parse_direct — 直接 JSON 解析
# =============================================================================


def test_parse_direct_valid() -> None:
    """合法 JSON 字典应直接返回。"""
    result = jp._parse_direct('{"plan": ["a", "b"]}')
    assert result == {"plan": ["a", "b"]}


def test_parse_direct_array_not_dict() -> None:
    """合法 JSON 但不是字典应返回 None。"""
    result = jp._parse_direct('["a", "b"]')
    assert result is None


def test_parse_direct_invalid() -> None:
    """非法 JSON 应返回 None。"""
    result = jp._parse_direct("not json at all")
    assert result is None


# =============================================================================
# 策略2：_parse_regex_nested — 嵌套感知正则提取（最多一层嵌套）
# =============================================================================


def test_parse_regex_nested_extracts_from_text() -> None:
    """从带前缀/后缀的文本中提取 JSON 对象。"""
    result = jp._parse_regex_nested('前缀 {"key": "val"} 后缀')
    assert result == {"key": "val"}


def test_parse_regex_nested_one_level() -> None:
    """一层嵌套可以正常提取。"""
    result = jp._parse_regex_nested('{"outer": {"inner": 1}}')
    assert result == {"outer": {"inner": 1}}


def test_parse_regex_nested_vs_greedy_outermost() -> None:
    """两层嵌套时策略2匹配内层子对象，策略3匹配完整最外层。

    re.search 会从任意位置匹配，因此策略2在两层嵌套文本中
    可能匹配到内层的合法子 JSON（如 {"deep": 1}），而非完整结构。
    策略3的贪婪 .* 则始终匹配最外层大括号，拿到完整 JSON。
    """
    text = '{"outer": {"inner": {"deep": 1}}}'

    nested_result = jp._parse_regex_nested(text)
    # 策略2 至少能匹配到内层子 JSON
    assert nested_result is not None

    greedy_result = jp._parse_regex_greedy(text)
    # 策略3 贪婪匹配到完整的最外层 JSON
    assert greedy_result == {"outer": {"inner": {"deep": 1}}}


def test_parse_regex_nested_no_braces() -> None:
    """文本中无大括号应返回 None。"""
    result = jp._parse_regex_nested("纯文本无括号")
    assert result is None


# =============================================================================
# 策略3：_parse_regex_greedy — 贪婪正则匹配
# =============================================================================


def test_parse_regex_greedy_handles_deep_nesting() -> None:
    """两层嵌套在策略2失败后，策略3应成功。"""
    result = jp._parse_regex_greedy('{"a": {"b": {"c": 1}}}')
    assert result == {"a": {"b": {"c": 1}}}


def test_parse_regex_greedy_no_braces() -> None:
    """无大括号返回 None。"""
    result = jp._parse_regex_greedy("无括号")
    assert result is None


# =============================================================================
# robust_json_parse — 端到端四层回退
# =============================================================================


@pytest.mark.asyncio
async def test_robust_empty_text_returns_empty_dict() -> None:
    """空字符串直接降级返回 {}。"""
    result = await jp.robust_json_parse("")
    assert result == {}


@pytest.mark.asyncio
async def test_robust_direct_parse() -> None:
    """标准 JSON 走策略1成功。"""
    result = await jp.robust_json_parse('{"k": "v"}')
    assert result == {"k": "v"}


@pytest.mark.asyncio
async def test_robust_markdown_wrapped() -> None:
    """LLM 常见输出：```json ... ``` 包裹，策略1失败→策略2提取成功。"""
    text = '好的，这是结果：\n```json\n{"key": "value"}\n```'
    result = await jp.robust_json_parse(text)
    assert result == {"key": "value"}


@pytest.mark.asyncio
async def test_robust_all_strategies_fail() -> None:
    """所有策略均失败，返回 {}。"""
    result = await jp.robust_json_parse("今天天气真好")
    assert result == {}


# =============================================================================
# 策略4：_repair_with_deepseek — 异步 DeepSeek 修复
# =============================================================================


@pytest.mark.asyncio
async def test_repair_no_api_key_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 API Key 时直接返回 None，不发起 HTTP 请求。"""
    monkeypatch.setattr(jp, "DEEPSEEK_API_KEY", "")
    result = await jp._repair_with_deepseek("broken json")
    assert result is None


@pytest.mark.asyncio
async def test_repair_deepseek_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock DeepSeek 返回有效修复文本。"""
    monkeypatch.setattr(jp, "DEEPSEEK_API_KEY", "test-key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": '{"fixed": true}'}}]}

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await jp._repair_with_deepseek('{"broken": ')

    assert result == '{"fixed": true}'


@pytest.mark.asyncio
async def test_repair_deepseek_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock DeepSeek 返回非 200，返回 None。"""
    monkeypatch.setattr(jp, "DEEPSEEK_API_KEY", "test-key")

    mock_response = MagicMock()
    mock_response.status_code = 500

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await jp._repair_with_deepseek("broken")

    assert result is None


@pytest.mark.asyncio
async def test_repair_deepseek_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock httpx 抛出网络异常，返回 None。"""
    monkeypatch.setattr(jp, "DEEPSEEK_API_KEY", "test-key")

    import httpx

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post.side_effect = httpx.ConnectError("connection refused")

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await jp._repair_with_deepseek("broken")

    assert result is None


@pytest.mark.asyncio
async def test_repair_response_not_valid_json_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek 修复后内容不是合法 JSON 字典，返回 None。"""
    monkeypatch.setattr(jp, "DEEPSEEK_API_KEY", "test-key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    # 返回的是数组而非字典
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '["not", "a", "dict"]'}}]
    }

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await jp._repair_with_deepseek("broken")

    assert result is None
