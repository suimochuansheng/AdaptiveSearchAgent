# 架构一览（六边形 / 端口 / 依赖规则）

> 状态：**T-arch 样板切片已落地（2026-11-11）**。`agent/app/ports.py` 声明 8 个端口
> 与 `RuntimeEvent`；`agent/adapters/runtime/langgraph_runtime.py` 与两个预算账本适配器
> （`agent/adapters/tools/budget_memory.py` / `budget_redis.py`，G11 起归入端口口径）是
> 已实现的适配器；其余适配器仍是空目录。`src/` **一行未搬**，现网主循环原样复用。
>
> 目标不是全量迁移，而是用一层薄样板把"六边形 / 端口 / 依赖方向"讲清楚、可执行、
> 可被契约测试打到。全量迁移见文末「被砍项与取舍」。

## 一句话结论

**主循环留在 LangGraph（成熟、可中断、可 checkpoint），业务策略（权限、预算、上下文、错误分类）上移到 `app`/`domain`；框架关在 `adapters/runtime` 之后，换框架只换一个适配器。**

## 六边形示意

```
                ┌──────────────────────────────┐
                │  main.py（组合根，T2-2 才出现）│
                └──────────────┬───────────────┘
                    实例化 + 注入│
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
┌────────────────────┐   实现端口    ┌────────────────────────┐
│ app（用例 + 策略）  │◄─────────────│ adapters（框架 + IO）   │
│ ports.py（7 端口）  │              │ runtime / models / ... │
└─────────┬──────────┘              └───────────┬────────────┘
          │ 使用                                 │ 使用
          └──────────────► domain ◄──────────────┘
                       （零依赖，纯定义）
```

箭头方向 = **依赖方向**：`adapters -> app.ports -> domain`。`app` 不知道谁实现端口；
`domain` 谁都不依赖。

## 依赖方向

| 从 | 可依赖 | 禁止 |
|---|---|---|
| `agent.domain` | 无（仅标准库） | `app`、`adapters`、`langchain*`、`langgraph*` |
| `agent.app` | `agent.domain`、`agent.app.ports` | `agent.adapters`、`langchain*`、`langgraph*` |
| `agent.adapters` | `agent.app.ports`、`agent.domain`、过渡期可 import `src` | `agent.app` 的其他模块（service 等） |
| `main.py`（组合根） | `app`、`adapters`、`domain` | —（唯一例外） |

三条硬约束由 `pyproject.toml` 的 import-linter 执行（见下节）：

1. `core-no-framework`：`agent.domain` / `agent.app` 不得 import langchain / langgraph 全家；
2. `app-independent`：`agent.app` 不得 import `agent.adapters`；
3. `adapters-only-ports`：`agent.adapters` 只能从 `agent.app` 里 import `ports`
   （`ignore_imports` 放行这一条），看不到 app 的具体实现。

**过渡期例外**：`adapters` 允许 import `src`（现网实现），因为本卡只做"包一层"，
不搬循环。这是防腐层的**边界**，不是终点；`src` 迁完后这些 import 会消失，届时
`src` 也应纳入 import-linter 的约束范围。

**组合根例外**：`main.py`（T2-2）是唯一允许同时 import `app` 与 `adapters` 的文件；
它负责实例化适配器并注入。当前文件尚不存在，`composition-root-only` 规则见
`docs/pending.md` P2。

## 已知缺陷

- `AgentRuntime.resume` 端口不含审批凭据，P0 只能保证"经 HTTP 路由进入"这一条路径安全（绕过路由直接调端口时，凭据校验不在端口契约内）。
- `domain` 里不再允许出现 `Protocol`（本轮已把 `BudgetLedger` 迁到端口层，`NoopLedger` 随之下沉到适配器），后续新增接口一律进 `app/ports.py`。

## 端口清单与实现映射

端口只在 `agent/app/ports.py` 声明（`Protocol` + `@runtime_checkable`，只声明不实现）。
每个端口的 docstring 都写死三段契约：**幂等性 / 错误语义 / 并发安全**。

| 端口 | 方法（契约） | 现实现 / 落点 | 状态 | 换实现要动什么 |
|---|---|---|---|---|
| `AgentRuntime` | `run_turn` / `resume` | `LangGraphRuntime`（`agent/adapters/runtime/langgraph_runtime.py`，内部包 `build_conversation_graph()` + `stream_conversation()`） | **已实现（样板）** | 只换 runtime 适配器；app 看到的仍是 `AsyncIterator[RuntimeEvent]` |
| `ModelClient` | `stream_chat` / `complete` / `count_tokens` | 现网 `src/utils/llm_factory` + `ChatOllama`（间接） | 未适配 | 新增 `adapters/models/`，app 的用例不改签名 |
| `SessionStore` | `load` / `save` / `mark_ended` | 现网 checkpointer / Redis（`src/checkpointer.py`） | 未适配 | 换存储实现（Redis / Postgres / 内存）不动 app |
| `ArtifactStore` | `put` / `get` / `search` | 现网 `src/tools/offload` 的落盘逻辑 | 未适配 | 换对象存储 / 本地盘只换适配器 |
| `TaskStore` | `enqueue` / `get` / `update` / `cancel` | 现网 API 内长任务 + Celery worker | 未适配 | 换队列（Celery / Redis Stream）不动用例 |
| `MemoryStore` | `append` / `recall` / `get_profile` / `upsert_profile` | 现网 `src/memory/` + 画像存储 | 未适配 | 记忆与画像同端口，换库不拆事务 |
| `EventBus` | `publish` / `subscribe` | 现网 SSE 编码 + tracer 直连 | 未适配 | 观测/前端/审计的扇出方式可换，主循环不改 |
| `BudgetLedger` | `add` / `snapshot` / `reset` | `MemoryBudgetLedger`（`agent/adapters/tools/budget_memory.py`）/ `RedisBudgetLedger`（`agent/adapters/tools/budget_redis.py`，失败降级 `NoopLedger`） | **已实现** | 换持久化（Redis / 内存 / DB）不动执行器与策略；原声明在 `agent/domain/budget.py`，OI-TAR-002 后迁入端口层 |

**不单独建的两个端口**（有意合并，避免同一模型/用户出现两套口径）：

- `TokenCounter` -> 归并进 `ModelClient.count_tokens`：token 计数是模型能力的属性；
- `UserProfileStore` -> 归并进 `MemoryStore.get_profile` / `upsert_profile`：画像与记忆
  共享用户分区与淘汰策略，拆开会把一次写入拆成两个事务。

**谁能换实现**：应用层只认端口。换 `AgentRuntime` = 换一个适配器（LangGraph 换成
别的框架，`interrupt`/`resume`/`final` 语义由 `RuntimeEvent` 契约保证）；换 store /
model / bus = 换对应适配器并对齐端口 docstring 的三段契约；应用层一行不改。

**契约测试**：`tests/contracts/test_agent_runtime_contract.py` 一份，参数化挂
`LangGraphRuntime`（假图，不连 Ollama）与一个内存 `Fake`。断言 `isinstance(impl,
AgentRuntime)`、`kind` 只出现六值、`ping` 不出现。新增实现只需往参数表加一行。

## 为什么"循环留在适配器、策略上移到应用层"

**循环留**：中断/恢复、checkpoint、`astream` 的 superstep、`ToolNode` 这些轮子
LangGraph 已经成熟；把它们搬进 app 等于重写框架，收益极低、风险极高。本卡的做法是
**包一层**：`LangGraphRuntime` 只把 `session` 翻译成 `inputs`/`config`，把
`stream_conversation` 的 SSE 帧翻译成 `RuntimeEvent`；`ping` 在适配器丢弃，
`interrupt` 的 `approval` 载荷原样进 `payload.approval`。

**策略上移**：权限、预算、上下文压缩、错误分类是纯规则，放在 `domain`/`app` 可以
单测、可以换策略而不碰框架；它们只依赖端口抽象，不被 `AIMessage` / `Command` 这类
框架类型污染。`hooks` 只透传（挂到 `configurable["runtime_hooks"]`），压缩/校验/预算
不在适配器里做——这就是"循环留在适配器、策略上移到应用层"的分工。

**代价（诚实记录）**：样板期 `LangGraphRuntime` 仍然 import `src`，防腐层只做了
一半；真正的 ACL（`DomainMessage` 等）是 T3-1 的活，不在本卡。

## 被砍项与取舍

| 项 | 为什么没做（一句话） | 何时做 |
|---|---|---|
| `T2-1` 拆 `api_main` -> `adapters/api` | HTTP/SSE 路由重组是大改造，样板期拆会与前端联调互相踩，先把端口/适配器边界立住更划算。 | 前端联调稳定后 |
| `T2-2` 组合根 `main.py` | 目前只有端口 + 一个适配器，没有可装配的 service 用例，过早抽组合根只会多一层转发。 | 第二个适配器（至少 store/model 之一）落地后 |
| `T3-1` ACL 全量 | 领域消息与框架消息的防腐层要逐节点迁移，属于全量迁移；样板期只在 `RuntimeEvent` 边界做映射。 | 主循环迁移时 |
| `T5-x` 搬 `context_manager` / `service` | 上下文预算与用例服务仍依赖 `src` 的既有实现和测试，搬过来会把"样板"变成"全量迁移"。 | `ModelClient` / `SessionStore` 适配后 |

## 组合根

`main.py` 负责实例化适配器并注入 `app`，是**唯一允许同时 import `app` 与 `adapters` 的文件**。它计划在 T2-2 出现；T-arch 不建。

**待启用规则（T2-2 后）**：`composition-root-only`。实现方式二选一：用 `independence` contract 声明 `app` 与 `adapters` 独立，再用 `ignore_imports` 放行 `main`；或对 `main` 之外的模块加 `forbidden` 检查。当前无法配置是因为目标文件尚不存在。

## 可执行门禁（`poetry run lint-imports`）

配置在 `pyproject.toml` 的 `[tool.importlinter]`，三条 contract：

| contract | source | forbidden | 说明 |
|---|---|---|---|
| `core-no-framework` | `agent.domain`, `agent.app` | `langchain*`, `langgraph*` | 核心零框架 |
| `app-independent` | `agent.app` | `agent.adapters` | 应用层不反向依赖 |
| `adapters-only-ports` | `agent.adapters` | `agent.app` | 唯一例外 `agent.app.ports`（`ignore_imports`） |

T-arch 之后 `adapters-only-ports` 的 "No matches for ignored import" 警告会消失：
`LangGraphRuntime` 真实 import 了 `agent.app.ports`，例外规则被命中。验收输出应为
`Contracts: 3 kept, 0 broken.` 且无 warning。
