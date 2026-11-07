"""异步任务维护循环测试（阶段 C2 修复）。

回归点：验收时发现"异步任务维护完成"这条 INFO 日志从来没出现——
因为 api_main 用了 `logging.getLogger(__name__)`，冒泡到 root 的 WARNING 级别被吞。
现在统一用项目 logger，并且把"一轮维护"抽成可测函数，用 caplog 锁住。
"""

import logging
from typing import Any

import api_main


def _fakes(
    *,
    requeued: list[str] | None = None,
    failed: int = 0,
    purged: int = 0,
) -> dict[str, Any]:
    calls: dict[str, Any] = {"enqueued": []}

    async def _requeue(_stale: int, _max: int) -> list[str]:
        return requeued or []

    async def _fail(_stale: int, _max: int) -> int:
        return failed

    async def _purge(_days: int) -> int:
        return purged

    def _enqueue(task_id: str) -> None:
        calls["enqueued"].append(task_id)

    return {
        "requeue_stale": _requeue,
        "fail_stale_exhausted": _fail,
        "purge_finished": _purge,
        "enqueue": _enqueue,
        "_calls": calls,
    }


async def test_maintenance_requeues_and_logs_summary(caplog: Any) -> None:
    fakes = _fakes(requeued=["t-1", "t-2"], failed=1, purged=3)
    caplog.set_level(logging.INFO, logger="adaptive_agent")

    stats = await api_main.maintain_async_tasks_once(
        requeue_stale=fakes["requeue_stale"],
        fail_stale_exhausted=fakes["fail_stale_exhausted"],
        purge_finished=fakes["purge_finished"],
        enqueue=fakes["enqueue"],
    )

    assert stats == {"requeued": 2, "failed": 1, "purged": 3}
    assert fakes["_calls"]["enqueued"] == ["t-1", "t-2"]
    # 核心回归：汇总日志必须真的打出来（之前被 root 的 WARNING 级别吞掉）
    messages = [record.getMessage() for record in caplog.records]
    assert any("异步任务维护完成" in message for message in messages), messages


async def test_maintenance_is_silent_when_nothing_to_do(caplog: Any) -> None:
    fakes = _fakes()
    caplog.set_level(logging.INFO, logger="adaptive_agent")

    stats = await api_main.maintain_async_tasks_once(
        requeue_stale=fakes["requeue_stale"],
        fail_stale_exhausted=fakes["fail_stale_exhausted"],
        purge_finished=fakes["purge_finished"],
        enqueue=fakes["enqueue"],
    )

    assert stats == {"requeued": 0, "failed": 0, "purged": 0}
    assert all("异步任务维护完成" not in record.getMessage() for record in caplog.records)


async def test_maintenance_enqueues_only_requeued_tasks() -> None:
    fakes = _fakes(requeued=["only-one"])

    await api_main.maintain_async_tasks_once(
        requeue_stale=fakes["requeue_stale"],
        fail_stale_exhausted=fakes["fail_stale_exhausted"],
        purge_finished=fakes["purge_finished"],
        enqueue=fakes["enqueue"],
    )

    assert fakes["_calls"]["enqueued"] == ["only-one"]
