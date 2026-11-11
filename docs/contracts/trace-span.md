# 契约 · TraceSpan（链路追踪 span）

> 卡：**T1-2（G3 契约冻结）** ｜ 状态：**v2 已冻结（2026-11-11）** ｜ 本文件：`docs/contracts/trace-span.md`
> 上游依据：《Agent 架构设计定稿主文档》2.4.2；修订文件 T7-0 卡面
> 消费方：`agent/observability/tracer.py`（T7-0 `JsonlTracer`）、`agent/observability/event_tap.py`（T7-0）、
> `scripts/smoke_trace.py`（T7-0）、`docs/baseline.md §7`（T9-5 费用归因）
>
> **变更规则**：字段、`kind` 枚举、`attrs` 白名单冻结。v1 的十字段 / 六 kind / 六键白名单保持不变；
> **v2 只对 `kind="turn"` 放行两个汇总键**（`tools` / `span_count`），其余五个 kind 仍严格六键。
> 确需再改必须新开 `docs/revisions/YYYY-MM-DD-<主题>.md`，由计划窗口升 v3 并同步代码与 `t12_checks.py`。
>
> **机读锚点**：`<!-- t12:... -->` 注释块由 `tests/test_agent_cases/T12-契约冻结/t12_checks.py` 比对。

## 1. 定义

一个 span = 一次可计量的工作单元（一轮、一次 LLM 调用、一次工具调用、一次压缩/offload、一次审批等待）。
一行 JSONL = 一个 span。

<!-- t12:fields TraceSpan -->
| 字段 | 类型 | 必填 | 说明 | 谁消费 |
|---|---|---|---|---|
| `trace_id` | `str` | 是 | 本轮追踪 id，32 位 hex（`secrets.token_hex(16)`）；同一轮的 turn/llm/tool span 共用 | 关联查询、`smoke_trace.py` |
| `span_id` | `str` | 是 | span 唯一 id（32 位 hex）；同一行内不复用 | 父子关联 |
| `parent_span_id` | `str \| None` | 是 | 父 span id；`kind="turn"` 为 `null`（本轮根） | 重建 span 树 |
| `session_id` | `str` | 是 | 会话 id（= `thread_id`） | 按会话聚合费用 |
| `turn_index` | `int` | 是 | 会话内第几轮，**从 1 开始（1 = 会话第一轮）**，同会话单调递增 | 对齐 SSE 轮次 |
| `name` | `str` | 是 | 工作单元名：`turn` / 模型名 / 工具名 / `compress` / `offload` / 工具名（审批） | 展示、按名称聚合 |
| `kind` | `str` | 是 | 六值枚举，见 §2 | 分类统计 |
| `start` | `str` | 是 | 开始时刻，ISO 8601 带偏移、毫秒精度（`2026-11-11T00:31:02.114+08:00`） | 耗时计算 |
| `end` | `str` | 是 | 结束时刻，同上；span 结束时才写行（不写"未结束"行） | 耗时计算 |
| `attrs` | `dict` | 是 | 白名单键，见 §3；**只放摘要与计数，不放正文/参数原文** | 费用与错误归因 |
<!-- /t12:fields TraceSpan -->

## 2. 枚举：`kind` 六值

<!-- t12:enum TraceSpan.kind -->
| 值 | 含义 | `name` 取什么 | 典型父 span |
|---|---|---|---|
| `turn` | 一轮对话（根 span） | `turn` | 无（`parent_span_id=null`） |
| `llm` | 一次模型调用（含流式） | 模型名 | `turn` |
| `tool` | 一次工具调用 | 工具名 | `turn` |
| `compress` | 一次上下文压缩 | `compress` | `turn` |
| `offload` | 一次大结果 offload / 回读 | `offload` | `turn` 或 `tool` |
| `approval` | 一次审批等待（从弹卡到恢复/超时） | 工具名 | `turn` |
<!-- /t12:enum TraceSpan.kind -->

## 3. `attrs` 白名单（按 kind 分两支）

白名单按 `kind` 分两支（v2）：

1. **非 turn**（`llm` / `tool` / `compress` / `offload` / `approval`）：严格六键
   `tool / tokens_in / tokens_out / cost / error_kind / approval_id`，多一个键都丢弃。
2. **turn**（根 span）：六键之外**额外允许** `tools: string[]`、`span_count: int`
   （一轮汇总专用），其余键仍不许加。

通用六键：

<!-- t12:keys TraceSpan.attrs -->
| 键 | 类型 | 说明 |
|---|---|---|
| `tool` | `str` | 工具名（`kind="tool"` 时与 `name` 一致；其余 kind 可省略） |
| `tokens_in` | `int` | 输入 token 数（`llm` / `compress`） |
| `tokens_out` | `int` | 输出 token 数（`llm` / `tool`） |
| `cost` | `float` | 本次费用（元），口径同 `tool-spec.md` 的 `CostModel` |
| `error_kind` | `str \| None` | 失败分类，取值沿用主文档 2.4.3：`transient` / `llm_recoverable` / `user_fixable` / `fatal` |
| `approval_id` | `str` | 审批 span 关联的 `ApprovalRequest.approval_id` |
<!-- /t12:keys TraceSpan.attrs -->

**`kind="turn"` 专用汇总键（v2 新增，只对根 span 开口）**

<!-- t12:keys TraceSpan.attrs.turn -->
| 键 | 类型 | 说明 |
|---|---|---|
| `tools` | `string[]` | 本轮工具调用名列表（按调用发生顺序；无工具为 `[]`） |
| `span_count` | `int` | 本轮 span 总数（含 `turn` 根自身） |
<!-- /t12:keys TraceSpan.attrs.turn -->

- **只有 `kind="turn"` 可以带这两个键**；`llm` / `tool` / `compress` / `offload` / `approval`
  仍严格六键白名单，多一个键都必须丢弃（`JsonlTracer` 已按此实现）。
- `turn` 的 `attrs` 里同样可以有通用六键（如 `tokens_in` / `tokens_out` / `cost` / `error_kind`）。

**错误语义**：本契约**不设 `status` 字段**。失败由 `attrs.error_kind` 表达：
为空 = 成功；非空取四类之一 `transient` / `llm_recoverable` / `user_fixable` / `fatal`。
同一语义不两处表达（内存句柄的 `status` 仅供运行时使用，不落盘）。

**白名单之外不许加键**：`kind="turn"` 的 `tools` / `span_count` 是 v2 的唯一例外；想记新东西
（如 `model`、`cached`、`candidates`）必须先升 v3——否则 G6/T7-0 与 G11/T9-5 的字段核对会失配。

## 4. 落盘

| 项 | 规则 |
|---|---|
| 路径 | `logs/trace/<date>.jsonl`（`<date>` = 本地日期 `YYYY-MM-DD`，即 `TRACE_DIR/<date>.jsonl`） |
| 格式 | UTF-8，一行一个 JSON 对象（无 BOM、无尾随逗号）；追加写（`a`），不加锁文件 |
| 写入时机 | span 结束时整行写入；不写"进行中"的占位行 |
| 不写哪里 | **不写业务表**（Postgres 会话/任务表）、**不写 checkpoint**（LangGraph state） |
| 开关 | `config.ENABLE_TRACE`（T7-0 新增，默认 `False`）；关闭时零额外 IO、零行为差异 |
| 失败降级 | 目录不可写 / Langfuse 密钥为空 -> 只打一条告警，**不影响主流程**（追踪不是业务依赖） |

## 5. 示例（一行 JSONL）

```json
{"trace_id":"9f2c1a4e7b3d5c8f0a1b2c3d4e5f6071","span_id":"1a2b3c4d5e6f708192a3b4c5d6e7f809","parent_span_id":"00112233445566778899aabbccddeeff","session_id":"tid-6","turn_index":1,"name":"search_knowledge","kind":"tool","start":"2026-11-11T00:31:02.114+08:00","end":"2026-11-11T00:31:04.902+08:00","attrs":{"tool":"search_knowledge","tokens_out":812,"cost":0.0,"error_kind":null}}
```

对照：同一轮里应能看到 `turn` 根 span、至少一个 `llm` span；含工具调用时 `tool` span 的
`parent_span_id` 指向 `turn` 的 `span_id`（`smoke_trace.py` 按此断言父子关系）。

## 6. 不变量（冻结期禁止）

1. 字段恒为十个：`trace_id / span_id / parent_span_id / session_id / turn_index / name / kind / start / end / attrs`。
2. `kind` 恒为六值；不得新增 `retrieval` / `rerank` / `memory` 之类。
3. `attrs` 只允许 §3 六个通用键；**唯一例外**：`kind="turn"` 可另带 `tools` / `span_count`。
   不得塞入 `messages`、`args`、`answer` 等正文。
4. 只写 `logs/trace/<date>.jsonl`；**不写业务表、不写 checkpoint**。
5. `turn` span 的 `parent_span_id` 必须为 `null`；其余 span 的父 id 必须能在同一文件里找到。
6. 追踪是旁路：关 flag 行为零差异，写入失败不得让本轮失败。
7. 本契约**不设 `status` 字段**：成功 = `attrs.error_kind` 为空；失败 = 非空（四类之一）。
8. `turn_index` 从 **1** 开始，`1` = 会话第一轮；仅取不到状态时允许兜底 `0`（不写正常路径）。

## 7. 谁消费它

| 消费方 | 用法 | 卡 |
|---|---|---|
| `JsonlTracer` | 按本契约序列化并追加写 | T7-0 |
| `event_tap` | 把 `astream_events` 的 on_* 事件映射成 span | T7-0 |
| `scripts/smoke_trace.py` | 断言 `turn -> tool -> llm` 父子关系与字段齐全 | T7-0 |
| `docs/baseline.md §7` / T9-5 | 用 `cost` / `tokens_*` 归因开/关 HITL 的差异 | T9-5 |

## 8. 变更记录

| 日期 | 版本 | 变更 | 依据 |
|---|---|---|---|
| 2026-11-11 | v1 | 冻结十字段、六 kind、六 attrs 白名单与 `logs/trace/<date>.jsonl` 落盘口径 | T1-2；主文档 2.4.2；T7-0 卡面 |
| 2026-11-11 | v2 | 仅 `kind="turn"` 增加汇总键 `tools` / `span_count`（其余五 kind 仍严格六键）；明确不设 `status` 字段、错误由 `attrs.error_kind` 表达；明确 `turn_index=1` = 会话第一轮 | 触发卡 T7-0；裁决人 计划窗口 |
