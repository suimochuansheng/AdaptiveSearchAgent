# 契约 · Approval（工具级人工审批）

> 卡：**T1-2（G3 契约冻结）** ｜ 状态：**v1 已冻结（2026-11-11）** ｜ 本文件：`docs/contracts/approval.md`
> 上游依据：《Agent 架构设计定稿主文档》2.4.2 / 2.9；2026-11-11 裁决 P4（`docs/pending.md`）
> 消费方：`agent/domain/roles.py`（本卡）、`agent/app`（T9-2 策略与状态机）、
> `agent/adapters/runtime`（T3-2b 挂起/恢复）、`frontend/app.py`（G9 审批卡片）
>
> **变更规则**：冻结后不改字段、不改 `Decision` 取值。确需改动必须新开
> `docs/revisions/YYYY-MM-DD-<主题>.md`，由计划窗口升 v2 并同步代码与 `t12_checks.py`。
>
> **机读锚点**：`<!-- t12:... -->` 注释块由 `tests/test_agent_cases/T12-契约冻结/t12_checks.py`
> 与 `agent/domain/roles.py` 的 dataclass 字段逐项比对。

## 1. 定义

```python
@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    tool_call_id: str
    tool: str
    args_digest: str
    cost_estimate: float
    budget_remaining: float
    permission: Permission
    expires_at: str

Decision = Literal["approve", "reject"]
```

`Permission` 来自 `agent/domain/tool.py`（四值枚举，见 `tool-spec.md` §2）。

审批是**挂起 + 恢复**：策略判定 `require_approval` -> 服务端生成 `ApprovalRequest` 并挂起本轮 ->
`RuntimeEvent(kind="interrupt", payload={"approval": …})` 下发 -> 前端弹卡 -> `POST …/resume`
（见 [resume-http.md](resume-http.md)）-> 服务端校验后继续或终止。

<!-- t12:fields ApprovalRequest -->
| 字段 | 类型 | 必填 | 说明 | 谁消费 |
|---|---|---|---|---|
| `approval_id` | `str` | 是 | 服务端签发的一次性凭证，32 位小写 hex（§3） | 前端原样回传；服务端做一次性校验 |
| `tool_call_id` | `str` | 是 | 被挂起的工具调用 id，审批通过后按它续跑 | T9-2 恢复执行器 |
| `tool` | `str` | 是 | 工具名，必须在 `ToolSpec.name` 中存在（见 `tool-spec.md`） | 审批卡片标题；审计 |
| `args_digest` | `str` | 是 | 参数摘要（默认 `query=…&max=5` 形态），用于人工核对"批的是什么" | 审批卡片正文 |
| `cost_estimate` | `float` | 是 | 本次预估费用（元），由 `ToolSpec.cost_model` 计算 | 卡片成本行；审计 |
| `budget_remaining` | `float` | 是 | 本会话剩余预算（元），审批时快照 | 卡片预算行；T6-3 账本 |
| `permission` | `Permission` | 是 | `read` / `external` / `write` / `dangerous`（枚举见 `tool-spec.md`） | 卡片风险提示；`dangerous` 必审批 |
| `expires_at` | `str` | 是 | ISO 8601 带时区偏移（如 `2026-11-10T19:00:00+08:00`），= 生成时刻 + TTL | 卡片到期行；服务端 410 判定 |
<!-- /t12:fields ApprovalRequest -->

## 2. 枚举：`Decision` 只有两个值

<!-- t12:enum Decision -->
| 值 | 含义 | 服务端行为 |
|---|---|---|
| `approve` | 同意执行 | 消费 `approval_id`，按 `tool_call_id` 续跑该次工具调用，继续本轮的 SSE |
| `reject` | 拒绝执行 | 消费 `approval_id`，不执行工具，把"已拒绝"作为工具结果回灌模型，本轮继续 |
<!-- /t12:enum Decision -->

**不做 `approve_and_remember`**（2026-11-10 审批结论）：任何"记住这个选择 / 本会话放行"
都是隐式放行，会绕过审计与逐次成本核对，v1 不提供。

## 3. `approval_id` 规则

| 项 | 规则 |
|---|---|
| 生成方 | **仅服务端**（`secrets.token_hex(16)` -> 32 位小写 hex）；客户端不得生成、不得推导 |
| 一次性 | 成功消费后立即失效；再次携带同一 `approval_id` -> `409` |
| 绑定 | 绑定 `thread_id`；换 thread 使用同一 `approval_id` 视为重放 -> `409` |
| TTL | **10 分钟**（`expires_at` = 生成时刻 + 600s）；超时 -> `410`，前端提示重新发起 |
| 传输 | 只经 `interrupt` 帧下行、经 `resume` body 上行；服务端侧的临时存储由 T9-2 决定（建议不进 LangGraph checkpoint，避免 state schema 迁移） |

## 4. 卡片状态机（前端，G9 已实现）

```text
pending ──approve──> approved ──200──> resumed
        └─reject───> rejected
        └─timeout──> timeout（默认 120s，等价 reject，不调用 resume）
approved/rejected/timeout 之后收到 409/410 -> aborted（提示重新发起）
```

卡片必须显示：工具名、参数摘要、预估成本、剩余预算、到期时间；按钮只有
`同意执行` / `拒绝`（不存在"记住选择"）。

## 5. 不变量（冻结期禁止）

1. `ApprovalRequest` **恰好八个字段**，字段名与类型冻结；不多加 `reason`、`user_id` 之类的实现细节（历史删除项见 §7）。
2. `Decision` 恒为 `approve` / `reject` 二值；不新增 `approve_and_remember` / `always` / `deny`。
3. `approval_id` 只由服务端生成；`ApprovalRequest` 不携带任何第二凭证（`resume` body 亦然）。
4. TTL 固定 10 分钟；`expires_at` 必须是带偏移的 ISO 8601 字符串。
5. 审批信息只出现在 `RuntimeEvent(kind="interrupt")` 的 `payload.approval`，不得新开 kind。
6. 前端超时按 `reject` 处理，**不调用** `/resume`（避免"超时即放行"）。

## 6. 谁消费它

| 消费方 | 用法 | 卡 |
|---|---|---|
| `agent/domain/roles.py` | 数据类本体（本卡落地） | T1-2 |
| 工具策略 `decide(...)` | 命中审批规则时产出 `ApprovalRequest` | T6-3 / T9-2 |
| `LangGraphRuntime` | 挂起本轮、生成 interrupt 事件、恢复续跑 | T3-2b / T9-2 |
| `resume` HTTP 路由 | 校验 `approval_id` 的一次性/TTL/归属，消费 `Decision` | T9-2 |
| `frontend/app.py` | 渲染五要素卡片，回传 `decision` | G9 |

## 7. 变更记录

<!-- t12:history -->
| 日期 | 版本 | 变更 | 依据 |
|---|---|---|---|
| 2026-11-11 | v1 | 冻结 `ApprovalRequest` 八字段与 `Decision` 二值 | T1-2；主文档 2.4.2 / 2.9 |
| 2026-11-11 | v1 | **删除 `token` 字段**：只留 `approval_id` / `decision` / `request_id` 三字段；理由 = 第二凭证与 `approval_id` 同通道下发，不增加实质安全，一次性 `approval_id` + `request_id` 幂等 + 会话锁已覆盖重放 | OI-TDM-003 裁决；`docs/pending.md` P4 |
