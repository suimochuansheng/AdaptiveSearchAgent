"""工具策略（T6-3）：`decide()` 纯函数，零 IO、零框架依赖。

规则最小集（**按顺序短路**，顺序本身是契约）：

1. `spec.requires_approval is True` -> `require_approval`（code=`approval_required`）
2. `used.calls >= ctx.max_calls` -> `deny`（code=`denied_calls`）
3. `used.cost + 本次成本 > ctx.cost_limit` -> `deny`（code=`denied_cost`）
4. 否则 -> `allow`（code=`allowed`）

`timeout` **不进 decide**：超时是执行期事实（工具真的跑慢了），由 executor 判定并
返回可读错误；放进策略会被误判成“策略拒绝”，也无法留痕到账本。

`requires_approval` 目前只认 `bool` 形态（`is True`）；`ApprovalThreshold` 的阈值判定
属 G7/T9-2 的 HITL 范围，本卡不实现，遇到对象形态按“不强制审批”继续往下走。

依赖注入：上限 / 费用上限放在 `PolicyContext` 里（默认值与 `config.py` 一致），
这样本模块不用 import `config`，测试也能直接构造边界值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agent.domain.budget import call_cost

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agent.domain.budget import BudgetSnapshot
    from agent.domain.tool import ToolSpec

PolicyDecision = Literal["allow", "deny", "require_approval"]
"""策略结论三值。"""

DecisionCode = Literal[
    "denied_timeout",
    "denied_calls",
    "denied_cost",
    "approval_required",
    "allowed",
]
"""给日志 / 审计用的稳定错误码（可读，不随文案变化）。

`denied_timeout` 不由本模块产出，保留在枚举里是因为它和其余四个码同属
“一次工具调用没跑成”的原因分类；执行器超时分支按同一分类给出可读文案，
不产出 `Decision`（超时不是策略判断）。
"""


@dataclass(frozen=True)
class PolicyContext:
    """决策上下文：作用域 + 本卡的两个上限 + 本次调用参数。

    `args` 目前不参与判定，保留给 `ApprovalThreshold`（G7）用；不要在这里做
    参数校验——工具参数校验是 LangChain schema 的职责。
    """

    scope: str = "unknown"
    max_calls: int = 8
    cost_limit: float = 2.0
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """一次调用的策略结论。`reason` 是给模型看的可读中文，`code` 是给机器看的。"""

    kind: PolicyDecision
    reason: str
    code: DecisionCode


def decide(spec: ToolSpec, ctx: PolicyContext, used: BudgetSnapshot) -> Decision:
    """按规则最小集给出结论；纯函数，同样的入参永远同样的出参。"""
    if spec.requires_approval is True:
        return Decision(
            kind="require_approval",
            reason="该工具需要人工审批后才能执行（HITL 未启用，本次已拒绝）",
            code="approval_required",
        )

    if used.calls >= ctx.max_calls:
        return Decision(
            kind="deny",
            reason=f"本轮工具调用次数已达上限（{ctx.max_calls} 次），本次调用被拒绝。",
            code="denied_calls",
        )

    if used.cost + call_cost(spec) > ctx.cost_limit:
        return Decision(
            kind="deny",
            reason=f"会话工具费用将超过上限（{ctx.cost_limit:g} 元），本次调用被拒绝。",
            code="denied_cost",
        )

    return Decision(
        kind="allow",
        reason="预算内，允许执行。",
        code="allowed",
    )
