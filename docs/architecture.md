# 架构一览（六边形 / 端口 / 依赖规则）

> 状态：T1-1 只建目录与门禁，`agent/` 下没有任何可执行实现；`src/` 一行未搬。
> 端口签名与契约测试在 T1-2 / T-arch 定义，本文只固定方向与名字。

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

## 依赖方向

| 从 | 可依赖 | 禁止 |
|---|---|---|
| `agent.domain` | 无 | `app`、`adapters`、`langchain*`、`langgraph*` |
| `agent.app` | `agent.domain`、`agent.app.ports` | `agent.adapters`、`langchain*`、`langgraph*` |
| `agent.adapters` | `agent.domain`、`agent.app.ports` | `agent.app` 的其他模块 |
| `main.py`（组合根） | `app`、`adapters`、`domain` | —（唯一例外） |

硬约束：`app` 只能看到端口抽象；`adapters` 只能看到 `app.ports`，看不到 `app.service` 等具体业务；`domain` 谁都不依赖。

## 端口清单

| 端口 | 职责 | 适配器目录 | 签名 |
|---|---|---|---|
| `AgentRuntime` | 承载主循环（LangGraph） | `agent/adapters/runtime` | T-arch |
| `ModelClient` | 模型调用（流式 + structured） | `agent/adapters/models` | T-arch |
| `SessionStore` | 会话持久化 | `agent/adapters/stores` | T-arch |
| `ArtifactStore` | 大结果 offload | `agent/adapters/stores` | T-arch |
| `TaskStore` | 长任务队列 + 状态 | `agent/adapters/tasks` | T-arch |
| `MemoryStore` | 长时记忆 + 画像 | `agent/adapters/stores` | T-arch |
| `EventBus` | 事件发布/订阅 | `agent/adapters/events` | T-arch |

非端口适配器（组合根直接装配，不经过端口）：

| 目录 | 用途 |
|---|---|
| `agent/adapters/locks` | 分布式锁薄封装 |
| `agent/adapters/api` | 入站 HTTP / SSE 路由 |
| `agent/adapters/tools` | 工具实现与注册表 |

## 组合根

`main.py` 负责实例化适配器并注入 `app`，是**唯一允许同时 import `app` 与 `adapters` 的文件**。它计划在 T2-2 出现；T1-1 不建。

**待启用规则（T2-2 后）**：`composition-root-only`。实现方式二选一：用 `independence` contract 声明 `app` 与 `adapters` 独立，再用 `ignore_imports` 放行 `main`；或对 `main` 之外的模块加 `forbidden` 检查。当前无法配置是因为目标文件尚不存在。

## 可执行门禁（`poetry run lint-imports`）

配置在 `pyproject.toml` 的 `[tool.importlinter]`，三条 contract：

| contract | source | forbidden | 说明 |
|---|---|---|---|
| `core-no-framework` | `agent.domain`, `agent.app` | `langchain*`, `langgraph*` | 核心零框架 |
| `app-independent` | `agent.app` | `agent.adapters` | 应用层不反向依赖 |
| `adapters-only-ports` | `agent.adapters` | `agent.app` | 唯一例外 `agent.app.ports`（`ignore_imports`） |

骨架期 `adapters-only-ports` 会有一条 "No matches for ignored import" 警告：例外规则尚未被真实 import 命中，属预期；一旦 adapter import 了 `app.ports` 警告消失。

## 为什么循环留、策略上移

主循环留在 LangGraph，是因为中断/恢复、checkpoint、流式事件这些轮子已经成熟；把权限、预算、上下文、错误分类上移到 `app`/`domain`，是为了让这些策略可单测、可替换，并且不被框架类型污染。
