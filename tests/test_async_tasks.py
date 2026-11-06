"""异步长任务测试（阶段 C1）——不依赖 PG / Redis / Celery / LLM。

覆盖：状态机流转、进度更新、取消只在 superstep 之间生效、失败落状态、
幂等（已完成不重跑）、提交接口的开关与入队失败处理。
"""

import json
from typing import Any

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage

from src.tasks_conversation import RESULT_KEY_TEMPLATE, run_conversation_task


class _FakeStore:
    """内存版任务存储，接口与 async_tasks_store 对齐。"""

    def __init__(self, task: dict[str, Any] | None, *, claim_ok: bool = True) -> None:
        self.task = task
        self.states: list[tuple[str, int | None, str | None]] = []
        self.cancel = False
        self.claim_ok = claim_ok
        self.claim_calls = 0

    async def claim(self, task_id: str) -> bool:
        self.claim_calls += 1
        if self.claim_ok and self.task is not None:
            self.task["state"] = "running"
        return self.claim_ok

    async def get(self, task_id: str) -> dict[str, Any] | None:
        return self.task

    async def set_state(
        self,
        task_id: str,
        state: str,
        *,
        progress: int | None = None,
        error: str | None = None,
        result_ref: str | None = None,
    ) -> None:
        self.states.append((state, progress, error))
        assert self.task is not None
        self.task["state"] = state
        if progress is not None:
            self.task["progress"] = progress
        if error:
            self.task["error"] = error

    async def is_cancel_requested(self, task_id: str) -> bool:
        return self.cancel


class _FakeResultStore:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttl: int | None = None

    async def set(self, key: str, value: str, ttl: int) -> None:
        self.data[key] = value
        self.ttl = ttl

    async def get(self, key: str) -> str | None:
        return self.data.get(key)


class _FakeGraph:
    """按顺序回放 superstep 快照。"""

    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        on_event: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.events = events
        self.consumed = 0
        self.on_event = on_event
        self.error = error

    async def astream(self, inputs: Any, config: Any = None, stream_mode: Any = None):
        for event in self.events:
            self.consumed += 1
            if self.on_event is not None:
                self.on_event(self.consumed)
            yield event
        if self.error is not None:
            raise self.error


def _task(state: str = "queued", message: str = "3.14 的平方根") -> dict[str, Any]:
    return {
        "task_id": "t-1",
        "thread_id": "thread-1",
        "user_id": "user-1",
        "state": state,
        "progress": 0,
        "payload": {"message": message},
    }


async def test_happy_path_marks_completed_and_stores_result() -> None:
    store = _FakeStore(_task())
    results = _FakeResultStore()
    graph = _FakeGraph(
        [
            {"messages": [AIMessage("思考中")]},
            {"messages": [AIMessage("思考中"), AIMessage("1.77")]},
        ]
    )

    async def _factory() -> _FakeGraph:
        return graph

    outcome = await run_conversation_task(
        "t-1", graph_factory=_factory, store=store, result_store=results, ttl=60
    )

    assert outcome["state"] == "completed"
    assert outcome["answer"] == "1.77"
    assert store.states[0][0] == "running"  # 开跑即置 running
    assert store.states[-1][0] == "completed"
    assert store.states[-1][1] == 100  # 完成时进度 100

    key = RESULT_KEY_TEMPLATE.format(task_id="t-1")
    assert json.loads(results.data[key])["answer"] == "1.77"
    assert results.ttl == 60


async def test_cancel_takes_effect_between_supersteps() -> None:
    store = _FakeStore(_task())
    results = _FakeResultStore()

    def _cancel_after_first(step: int) -> None:
        store.cancel = True

    graph = _FakeGraph(
        [
            {"messages": [AIMessage("一")]},
            {"messages": [AIMessage("二")]},
            {"messages": [AIMessage("三")]},
        ],
        on_event=_cancel_after_first,
    )

    async def _factory() -> _FakeGraph:
        return graph

    outcome = await run_conversation_task(
        "t-1", graph_factory=_factory, store=store, result_store=results
    )

    assert outcome["state"] == "cancelled"
    assert graph.consumed == 1  # 第一个 superstep 后立刻停，不再消耗模型调用
    assert store.states[-1][0] == "cancelled"
    assert results.data == {}  # 取消的任务不写结果


async def test_exception_marks_failed_with_reason() -> None:
    store = _FakeStore(_task())
    results = _FakeResultStore()
    graph = _FakeGraph([], error=RuntimeError("ollama 超时"))

    async def _factory() -> _FakeGraph:
        return graph

    outcome = await run_conversation_task(
        "t-1", graph_factory=_factory, store=store, result_store=results
    )

    assert outcome["state"] == "failed"
    assert store.states[-1][0] == "failed"
    assert "RuntimeError" in (store.states[-1][2] or "")
    assert "ollama 超时" in (store.states[-1][2] or "")


async def test_completed_task_is_idempotent() -> None:
    """Worker 重跑已完成任务：不再建图、不再调模型。"""
    store = _FakeStore(_task(state="completed"))
    results = _FakeResultStore()
    called = False

    async def _factory() -> _FakeGraph:
        nonlocal called
        called = True
        return _FakeGraph([])

    outcome = await run_conversation_task(
        "t-1", graph_factory=_factory, store=store, result_store=results
    )

    assert outcome == {"task_id": "t-1", "state": "completed", "idempotent": True}
    assert called is False
    assert store.states == []


async def test_missing_task_returns_missing() -> None:
    store = _FakeStore(None)
    outcome = await run_conversation_task(
        "nope", graph_factory=lambda: None, store=store, result_store=_FakeResultStore()
    )

    assert outcome["state"] == "missing"


async def test_submit_endpoint_rejects_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import api_main
    from config import settings

    monkeypatch.setattr(settings, "ENABLE_ASYNC_TASK", False)

    with pytest.raises(HTTPException) as exc:
        await api_main.submit_conversation_task(
            api_main.AsyncConversationRequest(thread_id="t", message="hi")
        )

    assert exc.value.status_code == 403


async def test_submit_endpoint_enqueues(monkeypatch: pytest.MonkeyPatch) -> None:
    import api_main
    import src.async_tasks_store as store_module
    import src.tasks_conversation as task_module
    from config import settings

    monkeypatch.setattr(settings, "ENABLE_ASYNC_TASK", True)
    monkeypatch.setattr(store_module, "create_task", _fake_create_task)
    monkeypatch.setattr(task_module, "enqueue_conversation_task", lambda task_id: None)

    response = await api_main.submit_conversation_task(
        api_main.AsyncConversationRequest(thread_id="t-1", message="hi")
    )

    assert response["task_id"] == "generated-id"
    assert response["state"] == "queued"
    assert response["status_url"].endswith("/status")


async def test_submit_endpoint_reports_broker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import api_main
    import src.async_tasks_store as store_module
    import src.tasks_conversation as task_module
    from config import settings

    monkeypatch.setattr(settings, "ENABLE_ASYNC_TASK", True)
    monkeypatch.setattr(store_module, "create_task", _fake_create_task)
    monkeypatch.setattr(store_module, "set_state", _fake_set_state)

    def _boom(task_id: str) -> None:
        raise ConnectionError("redis down")

    monkeypatch.setattr(task_module, "enqueue_conversation_task", _boom)

    with pytest.raises(HTTPException) as exc:
        await api_main.submit_conversation_task(
            api_main.AsyncConversationRequest(thread_id="t-1", message="hi")
        )

    assert exc.value.status_code == 503
    assert "Celery Worker" in str(exc.value.detail)


async def _fake_create_task(
    *, thread_id: str, user_id: str, message: str, kind: str = "conversation"
) -> str:
    return "generated-id"


async def _fake_set_state(task_id: str, state: str, **kwargs: Any) -> None:
    return None


async def test_cancel_while_queued_skips_graph() -> None:
    """排队期间取消：不建图、不调模型，直接置 cancelled。"""
    store = _FakeStore(_task(state="queued"))
    store.cancel = True
    results = _FakeResultStore()
    built = False

    async def _factory() -> _FakeGraph:
        nonlocal built
        built = True
        return _FakeGraph([])

    outcome = await run_conversation_task(
        "t-1", graph_factory=_factory, store=store, result_store=results
    )

    assert outcome["state"] == "cancelled"
    assert built is False
    assert store.states[-1][0] == "cancelled"


async def test_second_worker_cannot_claim_same_task() -> None:
    """两个 Worker 同时拿到同一任务：认领失败的那个直接退出，不重复调模型。"""
    store = _FakeStore(_task(state="queued"), claim_ok=False)
    results = _FakeResultStore()
    built = False

    async def _factory() -> _FakeGraph:
        nonlocal built
        built = True
        return _FakeGraph([])

    outcome = await run_conversation_task(
        "t-1", graph_factory=_factory, store=store, result_store=results
    )

    assert outcome["skipped"] is True
    assert built is False
    assert results.data == {}


async def test_failed_task_is_not_rerun() -> None:
    """已失败的任务是终态：Worker 重投时直接返回，不再重跑。"""
    store = _FakeStore({**_task(state="failed"), "error": "RuntimeError: boom"})
    results = _FakeResultStore()

    outcome = await run_conversation_task(
        "t-1", graph_factory=lambda: None, store=store, result_store=results
    )

    assert outcome["state"] == "failed"
    assert outcome["skipped"] is True
    assert store.claim_calls == 0
