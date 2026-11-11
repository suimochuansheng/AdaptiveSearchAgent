# 契约 · resume HTTP（审批恢复端点）

> 卡：**T1-2（G3 契约冻结）** ｜ 状态：**v1 已冻结（2026-11-11）** ｜ 本文件：`docs/contracts/resume-http.md`
> 上游依据：《Agent 架构设计定稿主文档》2.4.2；2026-11-11 裁决 P4（`docs/pending.md`）
> 消费方：`agent/adapters/api/web.py`（T9-2 实现路由）、`frontend/app.py::_resume_conversation`（G9 已对接）
>
> **变更规则**：路径、body 键、状态码语义冻结。确需改动必须新开
> `docs/revisions/YYYY-MM-DD-<主题>.md`，由计划窗口升 v2 并同步 `t12_checks.py` 与前端。
>
> **机读锚点**：`<!-- t12:... -->` 注释块由 `tests/test_agent_cases/T12-契约冻结/t12_checks.py` 比对。

## 1. 端点

```text
POST /chat/conversation/{thread_id}/resume
Content-Type: application/json
```

- `{thread_id}`：与发起本轮 `/chat/conversation` 时**完全相同**的 thread id。
- 请求体是 JSON 对象，**恰好三个键**（§2），不接受多余键（多传即视为协议漂移，服务端忽略未知键但不改语义）。
- 响应：成功是 **SSE 流**，失败是 **JSON 错误体 + 状态码**（§4）。

## 2. 请求体（三字段，冻结）

<!-- t12:keys resume.body -->
| 键 | 类型 | 说明 | 谁消费 |
|---|---|---|---|
| `approval_id` | `str` | 32 位小写 hex，来自 `interrupt` 帧的 `payload.approval.approval_id`；一次性、绑定 thread_id、TTL 10 分钟 | 服务端校验一次性/归属/TTL |
| `decision` | `str` | `"approve"` 或 `"reject"`（枚举见 `approval.md` §2） | 服务端决定执行还是拒绝 |
| `request_id` | `str` | 客户端生成的 **uuid4**，每次重试都要新值；服务端 24h 幂等去重 | 服务端幂等键（防重放） |
<!-- /t12:keys resume.body -->

```json
{
  "approval_id": "9f2c1a4e7b3d5c8f0a1b2c3d4e5f6071",
  "decision": "approve",
  "request_id": "3d1c9b6e-8f2a-4c7d-9e01-5b6a7c8d9e0f"
}
```

字段来源关系：`approval_id` 是 `ApprovalRequest.approval_id` 的回传；`decision` 是 `Decision` 的取值；
`request_id` **不属于** `ApprovalRequest`，由客户端在本端生成。

## 3. 幂等：`request_id` 24 小时窗口

| 项 | 规则 |
|---|---|
| 生成方 | 客户端（前端 `uuid.uuid4()`），每次"点按钮"生成一个新值 |
| 去重键 | `(thread_id, request_id)`，服务端保留 **24 小时** |
| 重复 | 窗口内同键再次到达 -> `409`，**不重复执行**工具、不重放 SSE |
| 超窗 | 超过 24h 后同键到达按新请求处理（`approval_id` 早已因 10 分钟 TTL 失效 -> `410`） |

## 4. 状态码语义（冻结）

| 状态码 | 含义 | 响应体 | 前端行为 |
|---|---|---|---|
| `200` | 恢复成功 | `text/event-stream`，继续推送与 `/chat/conversation` **同一套** SSE 帧（六个 `type` 见 `runtime-event.md` §4，外加心跳 `ping`） | 继续渲染；若再收到 `interrupt` 则再弹一张卡 |
| `409` | 重放 / 已消费 / `request_id` 重复 / `approval_id` 与 thread 不匹配 | `{"detail": "…"}` | 提示"该审批已被处理或重复提交"，结束本轮卡片 |
| `410` | 过期（`approval_id` 超过 TTL 10 分钟） | `{"detail": "…"}` | 提示"审批已过期"，结束本轮卡片 |
| `404` | 未知 thread；**或该路由尚未实现（审批未启用）** | `{"detail": "…"}` | 提示"审批功能未启用（等待 T9-2）"，**不抛异常、不卡界面** |

约定：`404` 同时覆盖"thread 不存在"与"后端还没实现 resume"两种情况——前端只能给出
"未启用/找不到会话"这类可读提示，绝不因此崩溃；这保证了 T9-2 落地前后前端行为一致。

## 5. 时序

```text
前端                         服务端
 │  POST /chat/conversation    │
 │ ──────────────────────────> │  工具策略判定 require_approval
 │  data: {"type":"interrupt", │  生成 ApprovalRequest（TTL 10min）并挂起
 │         "data":{…}}         │
 │ <────────────────────────── │
 │  弹卡：同意 / 拒绝 / 120s 超时（超时=reject，不发 resume）
 │  POST …/{thread_id}/resume  │
 │  {approval_id,decision,request_id}
 │ ──────────────────────────> │  校验：一次性 / 归属 / TTL / request_id 24h
 │                             │  通过 -> 续跑（approve）或回灌"已拒绝"（reject）
 │  200 + SSE 继续             │
 │ <────────────────────────── │
 │  409 / 410 / 404 走提示分支  │
```

## 6. 不变量（冻结期禁止）

1. body **恰好三个键**：`approval_id` / `decision` / `request_id`；不加凭证类字段（历史见 §8）。
2. `decision` 二值，不做 `approve_and_remember`；服务端不得把未知 decision 当 `approve`。
3. `approval_id` 由服务端生成、一次性、绑定 `thread_id`、TTL 10 分钟；客户端不得自造。
4. `request_id` 由客户端生成（uuid4），幂等窗口固定 24 小时。
5. 状态码语义见 §4；`200` 必须是 SSE，不得返回普通 JSON。
6. 路由不存在时保持 `404`（不得用 500/401 代替），前端据此显示"审批未启用"。

## 7. 兼容说明

- 现网前端（G9）对 `404` **与** `405` 都按"审批未启用"处理：`405` 只是"路由存在但方法不对"的
  历史兜底，属实现细节，**不写进契约**；T9-2 落地后只应出现 `404`。
- 前端已实现"同一 `approval_id` 只弹一张卡"的去重护栏（G9 `_register_approval`），
  它不是服务端幂等的替代：服务端仍须做 `approval_id` 一次性 + `request_id` 24h 去重。

## 8. 变更记录

<!-- t12:history -->
| 日期 | 版本 | 变更 | 依据 |
|---|---|---|---|
| 2026-11-11 | v1 | 冻结路径、三字段 body 与 `200/409/410/404` 语义 | T1-2 |
| 2026-11-11 | v1 | **删除 `token` 字段**：body 由四字段收为三字段；`interrupt` 帧也不再下发该字段，前端不会出现"空串占位" | OI-TDM-003 裁决；`docs/pending.md` P4 |
| 2026-11-11 | v1 | 409 亦用于审批凭据存储不可用（本次未执行、可重试）；返回码集合不变 | OI-T92-001 |
