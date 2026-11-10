# 契约 · ToolSpec（工具注册表条目）

> 卡：**T1-2（G3 契约冻结）** ｜ 状态：**v1 已冻结（2026-11-11）** ｜ 本文件：`docs/contracts/tool-spec.md`
> 上游依据：《Agent 架构设计定稿主文档》2.4.3 / 2.9；修订文件第 6 节 T1-2 卡面
> 消费方：`agent/domain/tool.py`（本卡）、`agent/adapters/tools/registry.py`（T6-0）、
> `agent/domain/tool_policy.py` + `agent/adapters/tools/executor.py`（T6-3）、
> `agent/adapters/tools/mcp_client.py`（T9-3）
>
> **变更规则**：字段与枚举冻结；注册表只能**填值**，不能加字段。确需改动必须新开
> `docs/revisions/YYYY-MM-DD-<主题>.md`，由计划窗口升 v2 并同步代码与 `t12_checks.py`。
>
> **机读锚点**：`<!-- t12:... -->` 注释块由 `tests/test_agent_cases/T12-契约冻结/t12_checks.py`
> 与 `agent/domain/tool.py` 的 dataclass 字段逐项比对。

## 1. 定义

```python
Permission = Literal["read", "external", "write", "dangerous"]

@dataclass(frozen=True)
class CostModel:
    unit: Literal["call", "keyword", "token"]
    unit_cost: float

@dataclass(frozen=True)
class ApprovalThreshold:
    threshold_field: str
    threshold: float

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    permission: Permission
    cost_model: CostModel
    timeout_seconds: float
    max_output_tokens: int
    requires_approval: bool | ApprovalThreshold
    feature_flag: str | None
```

`ToolSpec` 是**注册表里的元数据**，不承载执行逻辑：`fn` 不在契约里（工具栏是 adapter 的事）。

<!-- t12:fields ToolSpec -->
| 字段 | 类型 | 必填 | 说明 | 谁消费 |
|---|---|---|---|---|
| `name` | `str` | 是 | 工具名，与 LangChain `tool.name` 一致，全局唯一 | 注册表、模型 bind_tools、审计 |
| `description` | `str` | 是 | 给模型看的用途说明（决定模型何时选它） | 模型绑定（`bind_tools`） |
| `input_schema` | `dict` | 是 | JSON Schema（对象），`properties` 即参数名 | 模型绑定、参数校验、MCP 转出 |
| `permission` | `Permission` | 是 | 四值枚举，见 §2 | T6-3 策略（`write`/`dangerous` 走阈值判定） |
| `cost_model` | `CostModel` | 是 | 计价模型，见 §3 | T6-3 预算记账、审批卡片成本行 |
| `timeout_seconds` | `float` | 是 | 单次调用超时（秒），超时由执行器 `deny` | T6-3 执行器 |
| `max_output_tokens` | `int` | 是 | 输出 token 上限，超限截断并交 offload | T6-3 执行器 / T5-1 offload |
| `requires_approval` | `bool \| ApprovalThreshold` | 是 | `True` 恒审批；对象形态按阈值判定（§4） | T6-3 策略 -> `ApprovalRequest` |
| `feature_flag` | `str \| None` | 是 | 绑定的开关名（`config.py` 的 `ENABLE_*`）；`None` = 常开 | 注册表（T6-0）、组合根 |
<!-- /t12:fields ToolSpec -->

## 2. 枚举：`permission` 四值

<!-- t12:enum ToolSpec.permission -->
| 值 | 含义 | 典型工具 |
|---|---|---|
| `read` | 只读本地数据/内存，无外部副作用 | `calculator`、`read_offload` |
| `external` | 出网调用外部服务（有网络与费用风险） | `search_knowledge`、`search_tavily` |
| `write` | 写文件 / 写库 / 发消息等有副作用操作 | （v1 暂无） |
| `dangerous` | 高副作用或高成本，需按阈值强制人工裁决 | `deep_research` |
<!-- /t12:enum ToolSpec.permission -->

## 3. 计价模型 `CostModel`

<!-- t12:fields CostModel -->
| 字段 | 类型 | 必填 | 说明 | 谁消费 |
|---|---|---|---|---|
| `unit` | `Literal["call", "keyword", "token"]` | 是 | 计价单位，见下表 | T6-3 记账 |
| `unit_cost` | `float` | 是 | **每个 unit 的单价（元）**；费用 = 实际 unit 数 × `unit_cost` | T6-3 记账、审批卡片 |
<!-- /t12:fields CostModel -->

<!-- t12:enum CostModel.unit -->
| 值 | unit 的含义 | 示例 |
|---|---|---|
| `call` | 每次调用计一次 | `calculator`：`unit="call"`, `unit_cost=0.0` |
| `keyword` | 每个关键词计一次 | `search_tavily`：`unit="keyword"`, `unit_cost=0.02`（0.02 元/个） |
| `token` | 每 1 token 计一次；按"元/千 token"报价时换算成单价（例：2 元/千 token -> `unit_cost=0.002`） | `deep_research`：按消耗 token 计价 |
<!-- /t12:enum CostModel.unit -->

## 4. `requires_approval` 的两种形态

| 形态 | 语义 | 例子 |
|---|---|---|
| `bool` | `True` = 每次都审批；`False` = 不审批 | `requires_approval=True` |
| `ApprovalThreshold` | 取参数里的数值字段与阈值比较，**达到或超过（>=）** 即审批 | `{"threshold_field": "max_results", "threshold": 5}` |

<!-- t12:fields ApprovalThreshold -->
| 字段 | 类型 | 必填 | 说明 | 谁消费 |
|---|---|---|---|---|
| `threshold_field` | `str` | 是 | `input_schema.properties` 中的数值参数名 | T6-3 策略 |
| `threshold` | `float` | 是 | 阈值；实际值 `>= threshold` 时产出 `ApprovalRequest` | T6-3 策略 |
<!-- /t12:fields ApprovalThreshold -->

## 5. 示例（虚构工具，仅演示形状，不是注册表内容）

```json
{
  "name": "example_tool",
  "description": "演示用；真实工具取值由 T6-0 注册表定稿",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "max_results": {"type": "integer"}
    },
    "required": ["query"]
  },
  "permission": "external",
  "cost_model": {"unit": "keyword", "unit_cost": 0.02},
  "timeout_seconds": 20.0,
  "max_output_tokens": 2000,
  "requires_approval": {"threshold_field": "max_results", "threshold": 5},
  "feature_flag": null
}
```

本契约**不规定**现有 5 个工具的具体取值（`search_knowledge` / `search_tavily` / `calculator` /
`deep_research` / `read_offload`）：那是 T6-0 注册表的产出，且必须保持"同名同序、schema 哈希不变"。

## 6. 不变量（冻结期禁止）

1. `ToolSpec` **恰好九个字段**，顺序与名称冻结；不加 `fn`、不加 `enabled`、不加 `tags`。
2. `permission` 恒为四值；不得自造 `admin` / `network` / `unsafe` 等值。
3. `cost_model.unit` 恒为三值；`unit_cost` 是"元 / unit"，不是"元 / 千次"。
4. `requires_approval` 只有 `bool` 与 `{"threshold_field","threshold"}` 两形态，阈值为**闭区间下界**（`>=`）。
5. 注册表只能填值，**不能靠加字段**扩展能力；扩展字段须走 v2 修订。
6. `ToolResult` 不在本卡落地（占位注释），`Tool.execute` 的返回类型改造属后续卡。

## 7. 谁消费它

| 消费方 | 用法 | 卡 |
|---|---|---|
| `agent/domain/tool.py` | 数据类本体（本卡落地） | T1-2 |
| `agent/adapters/tools/registry.py` | 5 个工具按 `ToolSpec` 登记，`to_lc_tools()` 供 `bind_tools` | T6-0 |
| `agent/domain/tool_policy.py` | `decide(spec, ctx, budget)` 读 `permission` / `requires_approval` / `timeout_seconds` | T6-3 |
| `agent/adapters/tools/executor.py` | 读 `timeout_seconds` / `max_output_tokens` 做超时与截断 | T6-3 |
| `agent/adapters/tools/mcp_client.py` | 把 MCP tool 转成 `ToolSpec` | T9-3 |

## 8. 变更记录

| 日期 | 版本 | 变更 | 依据 |
|---|---|---|---|
| 2026-11-11 | v1 | 冻结 `ToolSpec` 九字段、`CostModel` 二字段、`Permission` 四值、`requires_approval` 两形态 | T1-2；主文档 2.4.3 / 2.9 |
