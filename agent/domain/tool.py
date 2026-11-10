"""工具领域模型（T1-2 契约冻结）：`ToolSpec` / `CostModel` / `ApprovalThreshold`。

纯数据定义，**零第三方依赖**：不得 import langchain / langgraph
（`pyproject.toml` 的 `core-no-framework` contract 会拦）。
字段与 `docs/contracts/tool-spec.md` 逐项一致，由
`tests/test_agent_cases/T12-契约冻结/t12_checks.py` 机读比对。

`ToolResult` 只占位、不实现：契约见《Agent 架构设计定稿主文档》2.4.3，
落地在后续卡（工具返回类型改造不属于 T1-2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Permission = Literal["read", "external", "write", "dangerous"]
"""工具权限四值：`read` / `external` / `write` / `dangerous`（见 tool-spec.md §2）。"""

CostUnit = Literal["call", "keyword", "token"]
"""计价单位三值（见 tool-spec.md §3）。"""


@dataclass(frozen=True)
class CostModel:
    """计价模型：费用 = 实际 unit 数 × `unit_cost`（元 / unit）。"""

    unit: CostUnit
    unit_cost: float


@dataclass(frozen=True)
class ApprovalThreshold:
    """`ToolSpec.requires_approval` 的阈值形态：参数值 `>= threshold` 时需人工审批。"""

    threshold_field: str
    threshold: float


@dataclass(frozen=True)
class ToolSpec:
    """注册表里的一条工具元数据：只有元数据，没有可调用体，也没有执行逻辑。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    permission: Permission
    cost_model: CostModel
    timeout_seconds: float
    max_output_tokens: int
    requires_approval: bool | ApprovalThreshold
    feature_flag: str | None


# ToolResult：占位，不在 T1-2 实现。
# 契约（主文档 2.4.3）= {status, content, artifact_ref, error_kind, task_id}，
# 其中 status 取 "ok" | "error" | "pending"；具体落地属后续卡（P2 范围）。
