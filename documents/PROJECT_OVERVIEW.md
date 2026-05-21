# AdaptiveSearchAgent — 项目简介

> **目标读者**：新接入的 AI Agent。读完本文档即可理解项目全貌，无需翻阅代码。

---

## 1. 项目身份

| 项目名 | AdaptiveSearchAgent |
|--------|---------------------|
| 简称 | adsearchagent |
| 定位 | 基于 LangGraph 的自适应搜索代理（多轮规划→搜索→评估→重试循环） |
| 作者 | suimo (huazhu7k7k@163.com) |
| Python 版本 | 3.11.15 |
| 包管理 | Poetry（虚拟环境激活: `eval $(poetry env activate)`） |

---

## 2. 技术栈一览

### 核心框架

| 技术 | 用途 |
|------|------|
| **LangGraph** ^1.1.10 | 图状态机编排引擎，整个 Agent 的"骨架" |
| **LangChain** ^1.2.17 | LLM 应用框架，ChatModel 抽象 + Prompt 管理 |
| **langchain-ollama** ^1.1.0 | Ollama 本地模型桥接 |
| **langchain-deepseek** ^1.0.1 | DeepSeek 云端模型桥接 |
| **langgraph-checkpoint-sqlite** ^2.0.0 | 状态持久化（SQLite），支持中断恢复 |

### 搜索 & 数据

| 技术 | 用途 |
|------|------|
| **Tavily** ^0.7.24 | 在线搜索 API（唯一搜索后端） |
| **tenacity** ^9.1.4 | 指数退避重试（搜索/LLM 调用容错） |

### Web 服务

| 技术 | 用途 |
|------|------|
| **FastAPI** ^0.136.1 | Web API 框架（预留，待接入） |
| **uvicorn** ^0.46.0 | ASGI 服务器 |

### 配置 & 可观测

| 技术 | 用途 |
|------|------|
| **pydantic-settings** ^2.14.0 | 配置管理，自动读取 `.env` |
| **python-json-logger** ^4.1.0 | JSON 格式结构化日志 |
| **rich** ^13.0.0 | CLI 终端美观表格（KPI 仪表盘） |
| **httpx** ^0.28.1 | 原生异步 HTTP 客户端（JSON 修复调用 DeepSeek） |

### 开发工具链

| 技术 | 用途 |
|------|------|
| **ruff** | Lint + 自动格式化 |
| **mypy** | 静态类型检查 |
| **pytest + pytest-asyncio** | 测试框架 |
| **pre-commit** | Git 提交前自动检查 |
| **bandit** | 安全漏洞扫描 |

---

## 3. 项目架构

### 3.1 整体工作流（LangGraph StateGraph）

```
用户查询
   │
   ▼
┌──────────┐     ┌───────────────────┐     ┌──────────────┐
│ Planner  │────▶│ ParallelSearcher  │────▶│ SearchWorker │
│ 生成搜索  │     │ 关键词分批+去重    │     │   × N 并行   │
│ 关键词列表 │     │ 生成 Send 扇出    │     │ Tavily 搜索  │
└──────────┘     └───────────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                    ┌─────────────────────────────────────┐
                    │            Evaluator                │
                    │  评估搜索结果置信度，判断是否充足      │
                    └──────────────────┬──────────────────┘
                                       │
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                    还有剩余批次   置信度不足     置信度达标
                    → Searcher    → Planner     → HumanApproval
                         │             │             │
                         └──── 循环 ───┘        批准 │ 拒绝
                                                   ▼    ▼
                                               Writer  END
                                             生成报告
```

### 3.2 关键节点的作用

| 节点 | 文件 | 职责 |
|------|------|------|
| **Planner** | `src/agents/planner.py` | 接收用户查询 + 上一轮缺失信息，调用 LLM 生成 3~5 个搜索关键词 |
| **ParallelSearcher** | `src/agents/parallel_searcher.py` | 关键词去重 + 分批（每批 `max_concurrent_searches` 个），通过 `route_to_search_workers` 生成 Send 扇出 |
| **SearchWorker** | `src/agents/search_worker.py` | 执行单个关键词的 Tavily 搜索，返回结果字典 |
| **Evaluator** | `src/agents/evaluator.py` | 评估搜索结果置信度（0~1），输出缺失信息和补充搜索关键词 |
| **HumanApproval** | `main.py` 内嵌 | 使用 LangGraph `interrupt()` 挂起图，等待人工输入 `yes/no` |
| **Writer** | `src/agents/writer.py` | 汇总所有搜索结果，生成 Markdown 格式报告 + Token 成本统计 |

### 3.3 状态定义

`src/state.py` — `AgentState(TypedDict)`，核心字段：

| 字段 | 类型 | 用途 |
|------|------|------|
| `user_query` | `str` | 原始用户查询，全程只读 |
| `plan` | `list[str]` | Planner 生成的关键词列表 |
| `search_results` | `Annotated[list[dict], operator.add]` | 多轮累积的搜索结果（自动拼接） |
| `confidence_score` | `float` | 当前置信度，≥0.8 达标 |
| `iteration` | `int` | 当前迭代轮次，上限 `max_iterations=3` |
| `missing_info` | `str` | 当前结果缺失的信息说明 |
| `retry_keywords` | `list[str]` | 基于缺失信息生成的补充关键词 |
| `final_report` | `str` | 最终 Markdown 报告 |
| `total_tokens` / `input_tokens` / `output_tokens` | `int` (operator.add) | 全流程 Token 累计 |
| `current_llm` | `str` | 当前实际使用的模型（ollama / deepseek） |
| `pending_keywords` | `list[str]` | 批次搜索中尚未处理的关键词 |
| `_batch_keywords` | `list[str]` | 当前批次关键词（paralel_searcher → 路由函数内部通道） |
| `thread_id` | `str` | 会话唯一标识 |

---

## 4. 已实现的核心功能

### 4.1 自适应多轮搜索循环

- Planner 首轮生成关键词 → 并行搜索 → Evaluator 评估
- 置信度不足且未达最大迭代次数 → 返回 Planner 生成补充关键词再搜
- 置信度达标或达到最大迭代 → 进入 HumanApproval → Writer

### 4.2 并行搜索（LangGraph Send API 扇出）

- `parallel_searcher` 将关键词分批（默认每批 2 个）
- `route_to_search_workers` 返回 `list[Send]` 实现并行扇出
- 所有 `search_worker` 完成后自动汇聚到 `evaluator`
- 支持多批次：剩余关键词写入 `pending_keywords`，evaluator 后优先处理

### 4.3 双模型备援（Ollama → DeepSeek）

- 默认使用 Ollama 本地模型
- 失败后指数退避重试（最多 `llm_max_retries=2` 次）
- 重试耗尽自动切换到 DeepSeek 云端模型
- 切换后修改 `config["configurable"]["llm_provider"]`，后续节点自动生效
- 全局计数器追踪切换次数（KPI 报告展示）

### 4.4 人工审批（Human-in-the-Loop）

- 搜索结果达到置信度后，通过 LangGraph `interrupt()` 挂起图
- 用户输入 `yes` → 进入 Writer 生成报告；`no` → 直接结束
- 状态通过 SQLite checkpoint 持久化，支持中断恢复

### 4.5 鲁棒 JSON 解析（四层回退）

`src/utils/json_parser.py` — 应对 LLM 输出非标准 JSON：

1. **直接解析** — `json.loads()`
2. **嵌套感知正则** — 提取最外层 `{...}`，容忍一层内部嵌套
3. **贪婪正则** — `r"\{.*\}"` 匹配
4. **DeepSeek LLM 修复** — 调用 DeepSeek API 修复损坏 JSON

### 4.6 安全数学计算器

`src/utils/math_safe_calc.py` — 基于 AST 白名单的表达式求值：
- 绝对禁止 `eval` / `exec`
- 白名单：仅允许 `+ - * / **` 和 `sin/cos/tan/sqrt/log/exp/abs`
- 任何不在白名单的 AST 节点立即抛出 `SecurityError`

### 4.7 可观测性体系

| 组件 | 文件 | 能力 |
|------|------|------|
| JSON 日志 | `src/utils/logger.py` | `@log_node` 装饰器自动记录节点耗时、task_id、iteration |
| 指标统计 | `src/utils/metrics.py` | JSON 解析命中率、迭代次数、Token 总量，异步持久化到 `data/metrics.json` |
| KPI 仪表盘 | `src/utils/cli_report.py` | Rich 表格展示：会话ID、模型切换、Token 消耗、费用估算、置信度、耗时 |
| Token 追踪 | `src/utils/llm_utils.py` | 每次 LLM 调用返回 `(content, input_tokens, output_tokens, total_tokens)`，自动累加到 state |

### 4.8 并发控制

- LLM 调用：全局 `asyncio.Semaphore`（默认 max 2 并发）
- 搜索任务：通过 `max_concurrent_searches` 控制每批并行数（默认 2）
- 可在 `config.py` 中调整或通过 `.env` 覆盖

### 4.9 配置管理

`config.py` — 基于 `pydantic-settings`，`.env` 自动加载，关键配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `tavily_api_key` | 必填 | Tavily 搜索 API 密钥 |
| `ollama_model_name` | `""` | Ollama 本地模型名 |
| `deepseek_api_key` | `""` | DeepSeek API 密钥 |
| `confidence_threshold` | `0.8` | 置信度达标阈值 |
| `max_iterations` | `3` | 最大迭代轮次 |
| `max_concurrent_searches` | `2` | 并行搜索任务数 |
| `max_concurrent_llm_calls` | `2` | 并行 LLM 调用数 |
| `llm_max_retries` | `2` | 主模型重试次数 |
| `llm_fallback_enabled` | `True` | 是否启用备援切换 |

---

## 5. 项目文件结构

```
AdaptiveSearchAgent/
├── main.py                     # 主入口：build_graph() + run_agent() + CLI 交互
├── config.py                   # pydantic-settings 配置类（.env 自动加载）
├── .env                        # 环境变量（API Key 等，不提交）
├── .pre-commit-config.yaml     # Git 提交前检查
├── AGENTS.md                   # AI Agent 入口文档（开发规范、工作流）
├── PROJECT_OVERVIEW.md         # ← 本文件
├── data/                       # 数据目录（metrics.json, checkpoints.db）
├── logs/                       # 日志目录
├── src/
│   ├── state.py                # AgentState TypedDict 定义
│   ├── agents/
│   │   ├── planner.py          # 搜索计划生成
│   │   ├── parallel_searcher.py # 并行搜索分发 + 路由函数
│   │   ├── search_worker.py    # 单关键词搜索执行
│   │   ├── evaluator.py        # 结果评估
│   │   └── writer.py           # 报告生成
│   ├── tools/
│   │   └── search.py           # Tavily 搜索封装（含重试）
│   └── utils/
│       ├── llm_factory.py      # LLM 工厂 + 信号量限流
│       ├── llm_utils.py        # 双模型备援 + Token 计数
│       ├── json_parser.py      # 四层鲁棒 JSON 解析
│       ├── logger.py           # JSON 日志 + @log_node 装饰器
│       ├── metrics.py          # 指标统计（异步持久化）
│       ├── cli_report.py       # Rich KPI 仪表盘
│       └── math_safe_calc.py   # AST 白名单安全计算器
└── tests/                      # 单元测试
```

---

## 6. 快速启动

```bash
# 1. 进入项目
cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent

# 2. 激活虚拟环境
eval $(poetry env activate)

# 3. 运行 Agent（CLI 交互模式）
python main.py

# 4. 运行测试
pytest

# 5. 代码质量检查
ruff check --fix . && ruff format . && mypy .
```

---

## 7. 当前开发状态

| 模块 | 状态 |
|------|------|
| 核心工作流（Planner → Search → Evaluate → Retry 循环） | ✅ 已实现 |
| 并行搜索（Send 扇出 + 批次管理） | ✅ 已实现 |
| 双模型备援（Ollama → DeepSeek） | ✅ 已实现 |
| 人工审批（HIL interrupt） | ✅ 已实现 |
| 鲁棒 JSON 解析 | ✅ 已实现 |
| 安全数学计算器 | ✅ 已实现 |
| 可观测性（日志 + 指标 + KPI 仪表盘） | ✅ 已实现 |
| SQLite 持久化（checkpoint） | ✅ 已实现 |
| FastAPI Web 接口 | ❌ 预留，待实现 |
| 多搜索后端支持（目前仅 Tavily） | ❌ 待扩展 |

---

> **给新 Agent 的建议**：先读 `main.py` 的 `build_graph()` 和 `run_agent()` 理解整体编排，再按需深入各个 `src/agents/` 和 `src/utils/` 模块。
