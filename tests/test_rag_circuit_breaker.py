"""RAG 熔断测试：子服务不可用时不要每次都撞一遍超时。

背景：一次深调研里 `search_worker` 会被调用多次，RAG 没起时每次都等连接失败，
整轮被拖到 10 分钟。熔断后冷却期内直接返回空，由 Tavily 兜底。
"""

from typing import Any

import httpx
import pytest

from src.tools import rag


class _Resp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FailingClient:
    """连接级失败（模拟子服务没起）。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "_FailingClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(self, *args: Any, **kwargs: Any) -> Any:
        raise httpx.ConnectError("connection refused")


class _OkClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls = 0

    async def __aenter__(self) -> "_OkClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(self, *args: Any, **kwargs: Any) -> _Resp:
        self.calls += 1
        return _Resp({"result": "本地知识库片段"})


class _ForbiddenClient:
    """熔断期内如果还有代码去建客户端，测试就该失败。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("熔断期内不应发起 HTTP 请求")


@pytest.fixture(autouse=True)
def _reset_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个用例从"可用"状态开始。"""
    rag._unavailable_until = 0.0
    monkeypatch.setattr(rag, "_clock", lambda: 1000.0)


async def test_connection_failure_opens_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag.httpx, "AsyncClient", _FailingClient)

    assert await rag.search_knowledge("query") == ""
    assert rag._rag_available() is False


async def test_breaker_short_circuits_without_http(monkeypatch: pytest.MonkeyPatch) -> None:
    rag._unavailable_until = 1000.0 + 60  # 处于冷却期
    monkeypatch.setattr(rag.httpx, "AsyncClient", _ForbiddenClient)

    assert await rag.search_knowledge("query") == ""


async def test_breaker_closes_after_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    rag._unavailable_until = 1000.0 + 60
    monkeypatch.setattr(rag, "_clock", lambda: 1000.0 + 61)
    monkeypatch.setattr(rag.httpx, "AsyncClient", _OkClient)

    assert await rag.search_knowledge("query") == "本地知识库片段"


async def test_success_resets_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    rag._unavailable_until = 1000.0 + 60  # 先假装熔断中
    monkeypatch.setattr(rag, "_clock", lambda: 1000.0 + 61)
    monkeypatch.setattr(rag.httpx, "AsyncClient", _OkClient)

    await rag.search_knowledge("query")

    assert rag._rag_available() is True


async def test_timeout_also_opens_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TimeoutClient(_FailingClient):
        async def post(self, *args: Any, **kwargs: Any) -> Any:
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(rag.httpx, "AsyncClient", _TimeoutClient)

    assert await rag.search_knowledge("query") == ""
    assert rag._rag_available() is False


async def test_http_status_error_does_not_open_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """子服务活着但这次请求报错（4xx/5xx）不该熔断，下次还要正常试。"""

    request = httpx.Request("POST", "http://rag.local/api/v1/search")
    response = httpx.Response(500, request=request, text="internal error")

    class _StatusErrorClient(_FailingClient):
        async def post(self, *args: Any, **kwargs: Any) -> Any:
            raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr(rag.httpx, "AsyncClient", _StatusErrorClient)

    assert await rag.search_knowledge("query") == ""
    assert rag._rag_available() is True
