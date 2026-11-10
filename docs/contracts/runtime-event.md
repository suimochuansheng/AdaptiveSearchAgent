# 契约 · RuntimeEvent（运行时事件）

> 卡：**T1-2（G3 契约冻结）** ｜ 状态：**v1 已冻结（2026-11-11）** ｜ 本文件：`docs/contracts/runtime-event.md`
> 上游依据：《Agent 架构设计定稿主文档》2.4.2 端口接口（`RuntimeEvent` 定义）
> 消费方：`agent/app/ports.py`（T3-2a `AgentRuntime`）、`agent/adapters/runtime`（T3-2b）、
> `agent/adapters/api/web.py`（T2-1/T3-2a）、`agent/observability/event_tap.py`（T7-0）、`frontend/app.py`（G9）
>
> **变更规则**：冻结后**不新增 `kind`**。确需改动必须新开 `docs/revisions/YYYY-MM-DD-<主题>.md`，
> 由计划窗口升 v2，并同步改 `agent/domain/`、五份契约与本目录的 `t12_checks.py`。
> 实现卡（T3-2a / T7-0 / T9-2 …）只消费本契约，**不得自行加字段或加 kind**。
>
> **机读锚点**：下面 `<!-- t12:... -->` 注释块由 `tests/test_agent_cases/T12-契约冻结/t12_checks.py`
> 逐项比对（字段名/枚举值/键名）。删改这类注释或表行即视为改契约，脚本会红。

## 1. 定义

```python
@dataclass(frozen=True)
class RuntimeEvent:
    kind: Literal["token", "tool_call", "tool_result", "interrupt", "final", "error"]
    payload: dict
```

一个 `RuntimeEvent` = 主循环向外暴露的**一个语义事件**；SSE 帧是它在 HTTP 上的**编码**，两者不是一回事
（见 §4 对照表）。`payload` 的形状由 `kind` 决定，见 §3。

<!-- t12:fields RuntimeEvent -->
| 字段 | 类型 | 必填 | 说明 | 谁消费 |
|---|---|---|---|---|
| `kind` | `Literal[6 值]` | 是 | 事件类型，取值见 §2 | 所有消费方首先按它分流 |
| `payload` | `dict` | 是 | 该 kind 的载荷，形状见 §3；**没有额外顶层字段** | adapter 编码、前端渲染、tracer |
<!-- /t12:fields RuntimeEvent -->

## 2. 枚举：`kind` 只有六个值

<!-- t12:enum RuntimeEvent.kind -->
| 值 | 含义 | 现网 `/chat/conversation` 帧 | 状态 |
|---|---|---|---|
| `token` | 增量文本（一段） | `data: {"type":"token","content":"…"}` | 已实现 |
| `tool_call` | 模型决定调用工具（`args` 已定稿） | `data: {"type":"tool_call","tools":[{"name":…,"args":{…}}]}` | 已实现 |
| `tool_result` | 工具返回（结果已就绪） | 无对应帧（前端按未知 type 静默忽略） | **未实现**（T3-2a/T7-0 前不产生） |
| `interrupt` | 需人工审批，本轮挂起 | 无对应帧 | **待 T9-2**（前端卡片已就绪，见 `approval.md`） |
| `final` | 本轮最终回答（收尾） | `data: {"type":"final","content":"…"}` | 已实现 |
| `error` | 可读错误（给人看的一句话） | `data: {"type":"error","message":"服务异常：…"}` | 已实现 |
<!-- /t12:enum RuntimeEvent.kind -->

**`ping` 不是 RuntimeEvent**：它是 SSE 心跳（`data: {"type":"ping"}`，静默
`heartbeat_seconds`（默认 10.0）秒无事件时发一次），用于保活与探测断连，
**不进 `RuntimeEvent`，也不进枚举**（见 §4 末行）。

## 3. `payload` 形状（按 kind，键名冻结）

<!-- t12:keys RuntimeEvent.payload.token -->
| 键 | 类型 | 说明 |
|---|---|---|
| `content` | `str` | 本段增量文本，可为空串以外的任意文本 |
<!-- /t12:keys RuntimeEvent.payload.token -->

<!-- t12:keys RuntimeEvent.payload.tool_call -->
| 键 | 类型 | 说明 |
|---|---|---|
| `tools` | `list[dict]` | 本轮决定调用的工具列表，元素形状见下表 |
<!-- /t12:keys RuntimeEvent.payload.tool_call -->

`payload.tools[]` 的元素：

<!-- t12:keys RuntimeEvent.payload.tool_call.tools[] -->
| 键 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | 工具名，必须在 `ToolSpec.name` 中存在（见 `tool-spec.md`） |
| `args` | `dict` | 模型给出的参数，已通过 `input_schema` 校验 |
<!-- /t12:keys RuntimeEvent.payload.tool_call.tools[] -->

<!-- t12:keys RuntimeEvent.payload.tool_result -->
| 键 | 类型 | 说明 |
|---|---|---|
| `tool_call_id` | `str` | 与模型给出的调用 id 对应，用于把结果配回 `tool_call` |
| `tool` | `str` | 工具名 |
| `content` | `str` | 工具输出（超大时由 ArtifactStore offload，正文放指针，不在此塞全文） |
<!-- /t12:keys RuntimeEvent.payload.tool_result -->

<!-- t12:keys RuntimeEvent.payload.interrupt -->
| 键 | 类型 | 说明 |
|---|---|---|
| `approval` | `dict` | 审批载荷，字段**恰好**等于 `ApprovalRequest`（见 `approval.md`） |
<!-- /t12:keys RuntimeEvent.payload.interrupt -->

<!-- t12:keys RuntimeEvent.payload.final -->
| 键 | 类型 | 说明 |
|---|---|---|
| `content` | `str` | 本轮完整回答（与已流出的 token 拼接一致） |
<!-- /t12:keys RuntimeEvent.payload.final -->

<!-- t12:keys RuntimeEvent.payload.error -->
| 键 | 类型 | 说明 |
|---|---|---|
| `message` | `str` | 给人看的可读错误，不含栈、不含密钥 |
<!-- /t12:keys RuntimeEvent.payload.error -->

## 4. 与现网 `/chat/conversation` 的逐项对照

现网 SSE 每帧都是 `data: {json}\n\n`，**没有 `event:` 行**，靠 JSON 的 `type` 分流
（`src/conversation_stream.py` + `api_main.py::chat_conversation`）。对照如下：

| RuntimeEvent.kind | 现网帧 `type` | 帧样例 | 现网实现 | 迁移落点 |
|---|---|---|---|---|
| `token` | `token` | `data: {"type":"token","content":"你"}` | ✅ 已实现 | 原样映射 |
| `tool_call` | `tool_call` | `data: {"type":"tool_call","tools":[{"name":"search_knowledge","args":{…}}]}` | ✅ 已实现 | 原样映射 |
| `tool_result` | （无） | — | ⛔ 未实现 | T3-2a/T7-0 起产出；前端按未知 `type` 忽略即可 |
| `interrupt` | `interrupt` | `data: {"type":"interrupt","data":{…ApprovalRequest…}}` | ⛔ 待 T9-2 | T9-2 起，`payload.approval` 编到帧的 `data` |
| `final` | `final` | `data: {"type":"final","content":"你好！"}` | ✅ 已实现 | 原样映射 |
| `error` | `error` | `data: {"type":"error","message":"服务异常：模型调用超时"}` | ✅ 已实现 | 原样映射 |
| （不属于 RuntimeEvent） | `ping` | `data: {"type":"ping"}` | ✅ 已实现 | **心跳**，不进 RuntimeEvent |

编码约定（adapter 侧，T3-2a 落地）：

| RuntimeEvent | SSE 帧 |
|---|---|
| `token` / `final` / `error` / `tool_call` | `payload` 的键**平铺**进帧（`kind` -> `type`） |
| `tool_result` | 平铺，`type="tool_result"` |
| `interrupt` | `{"type":"interrupt","data": payload["approval"]}`（`data` 即审批对象本体） |

## 5. 不变量（冻结期禁止）

1. **kind 恒为六值**，不新增、不改名、不删除；`approve` / `confirm` / `approval` 这类"新 kind"一律不许。
2. 审批信息**只**放 `payload.approval`；不得为审批单开新 kind（审批 ≠ 事件类型）。
3. `ping` 永远不是 RuntimeEvent；它只存在于传输层。
4. 前端遇到**未知 `type` / 未知 `kind`** 必须静默忽略（不抛异常、不渲染、不中断本轮）。
5. `payload` 内不得出现第二凭证字段（历史与裁决见 `approval.md` 变更记录）。
6. `RuntimeEvent` 是纯数据：不加方法、不加执行逻辑，`agent/domain` 与 `agent/app` 不得因此 import 框架。

## 6. 谁消费它

| 消费方 | 用法 | 卡 |
|---|---|---|
| `AgentRuntime` 端口 | `AsyncIterator[RuntimeEvent]` 的产出类型 | T3-2a |
| `LangGraphRuntime` | 把 `astream_events` 映射成 RuntimeEvent | T3-2b |
| `agent/adapters/api/web.py` | RuntimeEvent -> SSE 帧（§4 编码约定） | T2-1 / T3-2a |
| `event_tap` / `JsonlTracer` | 按 kind 建 span：`turn/llm/tool/approval`（见 `trace-span.md`） | T7-0 |
| `frontend/app.py` | 按 `kind` 渲染文本/工具卡/审批卡/错误 | G9 已实现 |

## 7. 变更记录

| 日期 | 版本 | 变更 | 依据 |
|---|---|---|---|
| 2026-11-11 | v1 | 冻结六 kind 与各 payload 键；与主文档 2.4.2 逐项 diff，无新增 kind、无新增顶层字段 | T1-2；主文档 2.4.2 |
