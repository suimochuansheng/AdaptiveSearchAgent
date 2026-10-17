# AdaptiveSearchAgent

基于 LangGraph 的自适应搜索 Agent，实现「规划 → 并行搜索 → 评估 → 重试 → 生成报告」闭环，支持本地知识库（RAG）与在线搜索（Tavily）双路并行。

---

AdaptiveSearchAgent搜索对话过程GIF（GIF加载慢，亦可进docs/screenshots/adaptive.gif 查看）
![对话演示](docs/screenshots/adaptive.gif)

---

## 解决什么问题

当你提一个需要调研的问题（比如"对比几款 GPU 的功耗和性能"）时，系统自动拆解成多个搜索关键词，并行地从本地知识库和互联网同时搜索，然后评估找到的信息够不够——不够就自动补搜，够了就生成一份带来源和成本统计的 Markdown 调研报告。全程无需人工干预，只在可能产生高额搜索费用时主动向你确认。

---

## 核心亮点

### 1. Send 扇出 + 多 Wave 自适应循环
关键词级并行搜索：一个节点分批关键词，扇出多个 worker 并行检索，四分支路由实现「搜到满意为止、但绝不超限」。**最多 5 个关键词并行、3 轮迭代上限**。
代码：`src/agents/parallel_searcher.py:75`、`src/graph_factory.py:33`

### 2. HITL 高成本前置确认
首轮搜索前估算成本，达到阈值时用 `interrupt()` 暂停图执行，用户确认后才继续，取消则零费用。**阈值 5 个关键词、单价 0.02 元/词**。（后续可按模型厂商实际费率动态估算）
代码：`src/agents/hitl_confirm.py:38`、`src/graph_factory.py:87`

### 3. RRF + FlagReranker 多级检索（调用子服务）
主服务通过 HTTP 调用 RAG 子服务的混合检索——向量 + BM25 用 RRF 融合（`k=60`）、候选池放大 3 倍、FlagReranker 交叉编码器精排。细节见 [rag-ingest-gateway](https://github.com/suimochuansheng/rag-ingest-gateway)。
主服务调用（消费方）：`src/tools/rag.py:13`


### 4. 双模型备援 + 四层 JSON 容错
LLM 主模型 Ollama 失败后指数退避重试、自动切换 DeepSeek；LLM 输出 JSON 用四层解析兜底（直接解析 → 正则提取 → 贪婪匹配 → DeepSeek 修复）。**重试 2 次、并发限流 2**。
代码：`src/utils/llm_utils.py:105`、`src/utils/json_parser.py:125`

### 5. Redis 分布式锁 + 双连接池
会话级 Redis 锁防止同一会话并发重复执行（含僵尸锁自愈）；数据库连接池拆成 Checkpointer 专用池 + 业务池隔离。**锁 TTL 600 秒、Saver 池 max10 / 业务池 max5**。
代码：`api_main.py:238`、`src/checkpointer.py:29`

### 6. ReAct 工具动态路由（bind_tools + ToolNode）
基于 LangGraph 标准工具调用协议，模型自主决策调用 `search_knowledge`（内部 RAG）/ `search_tavily`（互联网）/ `calculator`（数学计算），支持单工具直答与「先搜后算」多工具链式组合，并返回完整执行轨迹。由 `ENABLE_DYNAMIC_TOOLS` 开关零破坏性启用。
代码：`src/agents/tool_demo.py:57`、`api_main.py:607`

---

## 量化结果

| 指标 | 值 |
|---|---|
| 单次请求 LLM 调用上限 | 5 次（1 planner + 3 evaluator + 1 writer，max_iterations=3） |
| 搜索并发 | 5 个关键词 × 2 路检索 = 10 并发 I/O |
| 置信度阈值 | 0.8（达到即停止搜索） |
| Redis 锁 TTL | 600 秒 |
| 代码规模 | 主服务约 3.5k 行（Python） |
| 测试覆盖 | 14 个测试文件 |

---

## 架构图

```mermaid
flowchart TD
    A[用户请求] --> B[planner 拆词]
    B --> C[hitl_confirm 成本确认]
    C --> D[parallel_searcher 分批]
    D --> E[search_worker ×N 并行]
    E --> F[evaluator 评估]
    F --> G{路由判断}
    G -->|继续| D
    G -->|重规划| B
    G -->|完成| H[writer 生成报告]
    H --> I[SSE 流式返回]
```

> search_worker 通过 HTTP 调用 RAG 子服务（rag-ingest-gateway）做本地知识库检索。

---

## 工具动态路由（ReAct Demo）

主流程之外提供一套独立的 ReAct 工具调用演示模块，验证 LangGraph `bind_tools` + `ToolNode` 标准协议下模型「自主选工具 → 工具执行 → 结果回灌 → 继续推理」的完整闭环。默认关闭，开启后新增 `POST /demo/tools` 演示端点（`src/agents/tool_demo.py`）。

```mermaid
flowchart TD
    U[用户输入] --> AG[agent 节点<br/>ChatOllama.bind_tools 绑定 3 个工具]

    subgraph TN[ToolNode 工具执行器]
        T1[search_knowledge<br/>内部 RAG 知识库]
        T2[search_tavily<br/>Tavily 互联网搜索]
        T3[calculator<br/>安全数学计算]
    end

    AG --> RC{最后一条消息<br/>含 tool_calls?}
    RC -->|否| OUT[返回最终答案]
    RC -->|是| TN
    TN -->|ToolMessage 结果回灌| AG
```

**自主决策原则**（System Prompt 注入）：

| 问题类型 | 路由到 |
|---|---|
| 内部文档 / 产品问题 | `search_knowledge` |
| 实时新闻 / 公开信息 | `search_tavily` |
| 纯数学计算 | `calculator` |
| 复杂问题 | 多工具链式组合 |

**面试演示：先搜后算的多步 ReAct 轨迹**（Prompt 见 `api_main.py:588`）：

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant API as POST /demo/tools
    participant AG as agent 节点<br/>bind_tools 模型
    participant TN as ToolNode
    participant TAV as search_tavily
    participant CAL as calculator

    U->>API: 请先搜索 OpenAI 2025 年各季度营收，<br/>再用 calculator 计算总和
    API->>AG: graph.ainvoke(messages)
    AG->>AG: 规划：先搜 4 个季度，再求和
    AG->>TN: 发起 tool_call：search_tavily(OpenAI 2025 Q1 revenue)
    TN->>TAV: HTTP 搜索
    TAV-->>TN: Q1 营收文本
    TN-->>AG: ToolMessage 回灌
    loop Q2 ~ Q4（循环 3 次）
        AG->>TN: 发起 tool_call：search_tavily(OpenAI 2025 Qn revenue)
        TN->>TAV: HTTP 搜索
        TAV-->>AG: ToolMessage 回灌
    end
    AG->>TN: 发起 tool_call：calculator(Q1+Q2+Q3+Q4)
    TN->>CAL: 安全计算
    CAL-->>TN: 总和结果
    TN-->>AG: ToolMessage 回灌
    AG-->>API: 最终中文答案（无 tool_calls，路由到 END）
    API-->>U: 最终回复 + 完整执行轨迹 JSON
```

**启用方式**：

```bash
# .env 中开启（默认 False，零破坏性）
ENABLE_DYNAMIC_TOOLS=True

# 演示请求
curl -X POST http://localhost:8000/demo/tools \
  -H "Content-Type: application/json" \
  -d '{"query": "请先搜索 OpenAI 2025 年各季度营收，然后把四个季度的数字加起来，用 calculator 计算总和，最后用中文回答。"}'
```

> 演示图懒加载为单例缓存（`api_main.py:597`），端点返回每条消息的执行轨迹（用户输入 / AI 工具调用 / 工具结果 / 最终回复），便于面试现场直观展示 ReAct 推理过程。

---

## 技术栈

- **核心**：Python 3.11 / LangGraph / FastAPI / PostgreSQL（pgvector + Checkpointer）/ Redis / Ollama
- **工程**：SSE 流式 / Docker Compose / pytest / Langfuse 追踪 / Chainlit 前端 / DeepSeek 备援 / Tavily 在线搜索

---

## 快速开始

**前置依赖**：

- Docker（用于 PostgreSQL + Redis）
- Poetry（Python 依赖管理）
- Ollama + 已下载模型 `qwen2.5:7b`
- RAG 子服务已独立部署（见 [rag-ingest-gateway](https://github.com/suimochuansheng/rag-ingest-gateway)），与主服务放在同一父目录

```bash
# 0. 启动依赖服务（PostgreSQL + Redis）
docker compose up -d db_dev db_test redis

# 1. 安装 Python 依赖
poetry install

# 2. 启动主服务后端（端口 8000）
poetry run uvicorn api_main:app --reload --port 8000

# 3. 启动对话前端（可选，端口 8001）
poetry run chainlit run frontend/app.py --port 8001
```

> 健康检查：`GET http://localhost:8000/health` 应返回 `{"status":"ok"}`

---

## 项目结构

```bash
AdaptiveSearchAgent/
├── api_main.py              # FastAPI SSE 流式入口（含分布式锁、任务状态机）
├── config.py                # Pydantic Settings 全局配置
├── src/
│   ├── state.py             # AgentState + 自定义 reducer（哨兵清零）
│   ├── graph_factory.py     # LangGraph 图构建 + 条件路由
│   ├── checkpointer.py      # AsyncPostgresSaver + 双连接池
│   ├── agents/              # 6 个核心节点（planner/hitl_confirm/parallel_searcher/search_worker/evaluator/writer）+ tool_demo ReAct 演示
│   ├── tools/               # RAG(HTTP)/Tavily/calculator
│   └── utils/               # LLM 备援、JSON 容错、指标、日志
├── tests/                   # 14 个测试文件
├── frontend/                # Chainlit 对话 UI
└── scripts/                 # Ragas 离线评估等
```

---

## 演进方向

当前版本聚焦 MVP 端到端跑通，以下是有明确方案、待落地的优化项：

- **异步任务接线**：Celery 骨架已就绪，当前用 FastAPI BackgroundTasks 处理请求。下一步在"请求超过阈值"场景接入 Celery，释放请求线程。
- **结果缓存**：Redis 客户端已就绪。下一步对热点查询的最终报告做缓存（query 归一化 + 哈希做 key + TTL），砍掉重复 LLM 成本。
- **LLM 调用超时保护**：当前依赖重试 + 备援，无显式超时。下一步在 `llm_call_with_fallback` 外层包 `asyncio.wait_for`，防止请求假死占用并发槽位。
