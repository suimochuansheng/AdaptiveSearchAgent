"""工具预算领域模型（T6-3）：调用 / 费用 / token 的累计快照与账本协议。

纯定义，**零第三方依赖**（同 `tool.py`：不得 import redis / langchain）；
`pyproject.toml` 的 `core-no-framework` contract 会拦框架依赖，Redis 实现放在
`agent/adapters/tools/budget_redis.py`。

计价口径 v1（**近似**）：一次调用 = 1 个 unit，成本恒等于 `spec.cost_model.unit_cost`。
`unit="keyword"` 的“每个关键词计一次”、`unit="token"` 的按量计价细分，留到确有真实
计价需求时再做——策略层要保持纯函数、零 IO，不去解析工具参数（否则策略会被参数
解析错误带偏，也没法单测）。因此 v1 对 `search_tavily` 这类按词计价的工具会**少记**
费用，属已知近似，不是 bug。

账本实现允许同一 `scope` 并发调用：`add` 的累加与 `snapshot` 的读取不是原子的，
允许 **±1 次调用**的误差（见 `budget_redis.py` docstring）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from agent.domain.tool import ToolSpec


@dataclass(frozen=True)
class BudgetSnapshot:
    """某个作用域（通常是 `thread_id`）已消耗的预算快照。"""

    calls: int = 0
    cost: float = 0.0
    tokens: int = 0


def call_cost(spec: ToolSpec) -> float:
    """本次调用的计费成本（元）。v1：一次调用 = 1 个 unit。"""
    return float(spec.cost_model.unit_cost)


class BudgetLedger(Protocol):
    """预算账本端口：策略只读 `snapshot`，执行器只写 `add`。

    方法可以是**同步**（内存实现）或**异步**（Redis 实现），执行器用
    `inspect.isawaitable` 统一收口；返回值注解写成 `T | Awaitable[T]` 就是为了让
    两种实现都满足端口类型。
    """

    def add(
        self,
        scope: str,
        *,
        calls: int = 0,
        cost: float = 0.0,
        tokens: int = 0,
    ) -> None | Awaitable[None]:
        """把一次调用的消耗累加进 `scope`。"""
        ...

    def snapshot(self, scope: str) -> BudgetSnapshot | Awaitable[BudgetSnapshot]:
        """读 `scope` 当前已消耗的预算。"""
        ...

    def reset(self, scope: str) -> None | Awaitable[None]:
        """清空 `scope` 的累计值（换轮 / 换会话时用）。"""
        ...


class NoopLedger:
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
