# 契约 · Approval（工具级人工审批）

> 卡：**T1-2（G3 契约冻结）** ｜ 状态：**v1 已冻结（2026-11-11）→ v2（2026-11-24，`reject` 分档语义）** ｜ 本文件：`docs/contracts/approval.md`
> 上游依据：《Agent 架构设计定稿主文档》2.4.2 / 2.9；2026-11-11 裁决 P4（`docs/pending.md`）；
> 2026-11-24 计划窗口 D16 裁决（`docs/revisions/2026-11-24-reject-terminates-turn.md`）
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
| `reject` | 拒绝执行 | 消费 `approval_id`，**不执行工具、不记账、不扣预算**；随后按 `ToolSpec.permission` 分档（§2.1）：`dangerous` → **终止本轮**（不再进模型推理）；其余 → 把"已拒绝"作为工具结果回灌模型，本轮继续 |
<!-- /t12:enum Decision -->

**不做 `approve_and_remember`**（2026-11-10 审批结论）：任何"记住这个选择 / 本会话放行"
都是隐式放行，会绕过审计与逐次成本核对，v1 不提供。

### 2.1 `reject` 的分档语义（v2，2026-11-24 裁决）

`reject` **不是全局**"本轮到此为止"，也**不是** v1 那样一律回灌——**按被批工具的风险档位分流**。
判据只用 `ToolSpec.permission`（见 `tool-spec.md` §2），**不按工具名硬编码**，不新增枚举值。

| `permission` | 拒绝后行为 | 是否再进模型推理 | 本轮是否推送增量文本帧（`token`） | 典型工具 |
|---|---|---|---|---|
| `dangerous` | **终止本轮**：图在 tools 之后直接 `END`，本轮 SSE 以**恰好一条** `final` 收尾 | 否 | 否（0 条） | `deep_research` |
| `external` / `read` / `write` | **回灌继续**（= v1 语义）：把"已拒绝"作为工具结果交给模型，模型可用别的路径作答 | 是 | 是（正常流式） | `search_knowledge` / `search_tavily` / `calculator` / `read_offload` |

终止分支的 `final` 文案固定为：`已拒绝执行 {tool_name}，本轮未产生内容`。

条文细则：

1. **终止只对 `dangerous` 成立**。`dangerous` 是"能力门"（有成本模型、单次可跑数百秒）；
   用户拒绝的潜台词是"这一轮我不批这笔钱 / 这个长任务"。若让模型绕过审批自造一份内容
   （实测：拒绝 `deep_research` 后自造 3095 字带假数字的"调研报告"），审批就只拦住了工具、
   没拦住产出，比直接报错更有害。
2. **低风险维持回灌**。`external` / `read` / `write` 的拒绝只意味着"这条路径不通"，
   模型换个安全做法是期望行为；把它们也终止，等于把"拒绝一次"放大成"这一轮什么都别答"。
3. **未注册的工具**按低风险处理（维持回灌）：审批链路不得因注册表缺项而升级为终止。
4. **超时与 TTL 过期同义**：前端 120s 超时按 `reject` 处理、**不调用** `/resume`（§4）；
   凭据 TTL 到期返回的 `410` 仍是纯状态码分支，不参与"本轮终止与否"的判定。
5. **状态码一律不变**：`409` / `410` / `404` / `200` 的语义与本条无关，见
   [resume-http.md](resume-http.md) §4。终止路径结束后会话回到 `completed`（或 `idle`）并释放锁。
6. **预算与审计不变**：终止分支不执行工具、不记账、不扣预算；trace 的 tool span 仍以
   `error_kind=rejected_by_user` 记录"被拒"。
7. **不做"拒绝并说明"**：带 feedback 的第三按钮属 **v2.1**，需要新增 `Decision` 枚举值，
   本版不加——必须改工具字段或枚举时，由计划窗口另升 v3 并同步 `t12_checks.py`。

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
7. `reject` 的分流只由 `ToolSpec.permission` 决定：`dangerous` 终止、其余回灌；
   不得按工具名硬编码，也不得把低风险拒绝升级为终止（§2.1）。

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
| 2026-11-24 | **v2** | `reject` 升为**分档语义**（新增 §2.1，§2 表 `reject` 行改写，§5 加第 7 条不变量）：`dangerous` 被拒 → **终止本轮**（不回模型推理、零 token、恰好一条 `final`）；`external`/`read`/`write` → 维持回灌继续。字段与 `Decision` 枚举**未改** | 计划窗口 D16 裁决；`docs/revisions/2026-11-24-reject-terminates-turn.md` |
