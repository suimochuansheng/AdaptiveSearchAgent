"""工具预算领域模型（T6-3）：调用 / 费用 / token 的累计快照与计价纯函数。

纯定义，**零第三方依赖**（同 `tool.py`：不得 import redis / langchain）；
`pyproject.toml` 的 `core-no-framework` contract 会拦框架依赖，Redis 实现放在
`agent/adapters/tools/budget_redis.py`。

**端口口径（OI-TAR-002）**：账本接口 `BudgetLedger` 是"需要实现方"的端口，已迁到
`agent/app/ports.py`；本模块只留 `BudgetSnapshot` / `call_cost` 这些纯数据与纯函数，
**不再出现 `Protocol`**。降级用的 `NoopLedger` 也随之落到适配器层
（`agent/adapters/tools/budget_memory.py`），由 `budget_redis.py` 继续再导出以兼容旧导入。
账本实现的并发误差（允许 ±1 次调用）写在该端口的 docstring 里。

计价口径 v1（**近似**）：一次调用 = 1 个 unit，成本恒等于 `spec.cost_model.unit_cost`。
`unit="keyword"` 的“每个关键词计一次”、`unit="token"` 的按量计价细分，留到确有真实
计价需求时再做——策略层要保持纯函数、零 IO，不去解析工具参数（否则策略会被参数
解析错误带偏，也没法单测）。因此 v1 对 `search_tavily` 这类按词计价的工具会**少记**
费用，属已知近似，不是 bug。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
