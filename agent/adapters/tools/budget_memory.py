"""内存预算账本（T6-3）：测试 / 单进程脚本用，进程退出即丢。

与 Redis 实现的区别只有持久化：累加语义、快照形状、`scope` 键完全一致，
所以执行器与策略对两种实现无感知（执行器用 `inspect.isawaitable` 兼容同步返回）。

端口 `BudgetLedger` 声明在 `agent/app/ports.py`；本文件同时提供两个实现：

- `MemoryBudgetLedger`：进程内容器，测试 / 单进程脚本用；
- `NoopLedger`：不记账的降级实现（Redis 不可用或未配置时兜底），快照恒为全 0。

线程说明：`add` / `snapshot` 内部没有 `await`，在 asyncio 单线程下天然原子；
这里再包一把 `threading.Lock`，只是为了将来在线程池里跑测试时不丢计数。
"""

from __future__ import annotations

import threading

from agent.app.ports import BudgetLedger
from agent.domain.budget import BudgetSnapshot

__all__ = ["MemoryBudgetLedger", "NoopLedger"]


class MemoryBudgetLedger(BudgetLedger):
    """`dict` 实现的内存账本：`scope -> [calls, cost, tokens]`。"""

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def add(
        self,
        scope: str,
        *,
        calls: int = 0,
        cost: float = 0.0,
        tokens: int = 0,
    ) -> None:
        with self._lock:
            row = self._data.setdefault(scope, [0, 0.0, 0])
            row[0] += calls
            row[1] += cost
            row[2] += tokens

    def snapshot(self, scope: str) -> BudgetSnapshot:
        with self._lock:
            calls, cost, tokens = self._data.get(scope, [0, 0.0, 0])
        return BudgetSnapshot(calls=int(calls), cost=float(cost), tokens=int(tokens))

    def reset(self, scope: str) -> None:
        with self._lock:
            self._data.pop(scope, None)


class NoopLedger(BudgetLedger):
    """不记账的账本：Redis 不可用或未配置时退化使用，保证会话不被预算模块拖垮。

    快照恒为全 0，因此策略永远看到“预算充足”——这是**有意的降级**：
    预算模块坏掉时宁可少拦几次，也不能让用户会话直接挂掉。
    """

    def add(
        self,
        scope: str,
        *,
        calls: int = 0,
        cost: float = 0.0,
        tokens: int = 0,
    ) -> None:
        return None

    def snapshot(self, scope: str) -> BudgetSnapshot:
        return BudgetSnapshot()

    def reset(self, scope: str) -> None:
        return None
