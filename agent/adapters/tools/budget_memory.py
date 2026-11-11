"""内存预算账本（T6-3）：测试 / 单进程脚本用，进程退出即丢。

与 Redis 实现的区别只有持久化：累加语义、快照形状、`scope` 键完全一致，
所以执行器与策略对两种实现无感知（执行器用 `inspect.isawaitable` 兼容同步返回）。

线程说明：`add` / `snapshot` 内部没有 `await`，在 asyncio 单线程下天然原子；
这里再包一把 `threading.Lock`，只是为了将来在线程池里跑测试时不丢计数。
"""

from __future__ import annotations

import threading

from agent.domain.budget import BudgetSnapshot


class MemoryBudgetLedger:
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
