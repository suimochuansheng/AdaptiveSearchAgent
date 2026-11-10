"""角色与审批决策（T1-2 契约冻结）：`Decision` / `ApprovalRequest`。

纯数据定义，**零第三方依赖**：不得 import langchain / langgraph
（`pyproject.toml` 的 `core-no-framework` contract 会拦）。
字段与 `docs/contracts/approval.md` 逐项一致，由
`tests/test_agent_cases/T12-契约冻结/t12_checks.py` 机读比对。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agent.domain.tool import Permission

Decision = Literal["approve", "reject"]
"""审批决定只有两个值：不做 `approve_and_remember`（隐式放行会绕过审计）。"""


@dataclass(frozen=True)
class ApprovalRequest:
    """一次待人工审批的工具调用：由服务端生成，经 `interrupt` 帧下发。

    `approval_id` 由服务端 `secrets.token_hex(16)` 生成（32 位 hex）、一次性、
    绑定 `thread_id`、TTL 10 分钟；`expires_at` 为带偏移的 ISO 8601 字符串。
    """

    approval_id: str
    tool_call_id: str
    tool: str
    args_digest: str
    cost_estimate: float
    budget_remaining: float
    permission: Permission
    expires_at: str
