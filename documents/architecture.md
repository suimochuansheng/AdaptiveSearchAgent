# AdaptiveSearchAgent 架构文档

> 生成时间：2025-01  
> 基于当前代码自动分析生成

---

## 1. 系统全景

```mermaid
graph TB
    subgraph 入口["入口层"]
        USER[用户输入]
        MAIN[main.py<br/>run_agent / build_graph]
    end

    subgraph Agent核心["LangGraph Agent 核心"]
        direction TB
        PLANNER[Planner 节点<br/>生成搜索关键词]
        SEARCH[Search Worker 节点<br/>并行执行搜索]
        EVALUATOR[Evaluator 节点<br/>评估结果充分性]
        WRITER[Writer 节点<br/>生成最终报告]
    end

    subgraph 工具层["工具与基础设施"]
        TAVILY[Tavily 搜索 API<br/>带指数退避重试]
        OLLAMA[Ollama 本地 LLM<br/>qwen2.5:7b]
        DEEPSEEK[DeepSeek API<br/>JSON 修复备援]
        METRICS[Metrics 指标统计<br/>自动持久化]
    end

    subgraph 数据["数据 & 配置"]
        STATE[AgentState<br/>LangGraph 共享状态]
        CONFIG[Settings<br/>pydantic-settings<br/>.env 自动加载]
    end

    USER --> MAIN
    MAIN --> PLANNER
    PLANNER -->|plan: 关键词列表| SEARCH
    SEARCH -->|search_results: 搜索结果| EVALUATOR
    EVALUATOR -->|条件判断| PLANNER
    EVALUATOR -->|置信度达标| WRITER
    WRITER -->|final_report| USER

    PLANNER -.->|调用| OLLAMA
    EVALUATOR -.->|调用| OLLAMA
    SEARCH -.->|调用| TAVILY

    PLANNER -.->|JSON解析回退| DEEPSEEK
    EVALUATOR -.->|JSON解析回退| DEEPSEEK

    STATE -.-> PLANNER
    STATE -.-> SEARCH
    STATE -.-> EVALUATOR
    STATE -.-> WRITER

    CONFIG -.-> PLANNER
    CONFIG -.-> SEARCH
    CONFIG -.-> EVALUATOR
    CONFIG -.-> DEEPSEEK
```

---

## 2. LangGraph 工作流（状态机）

```mermaid
stateDiagram-v2
    [*] --> Planner

    state Planner {
        [*] --> 读取user_query
        读取user_query --> 拼接missing_info+retry_keywords
        拼接missing_info+retry_keywords --> 调用Ollama_LLM
        调用Ollama_LLM --> robust_json_parse
        robust_json_parse --> 输出plan列表
        输出plan列表 --> [*]
    }

    Planner --> SearchWorker

    state SearchWorker {
        [*] --> 遍历plan关键词
        遍历plan关键词 --> search_tavily
        search_tavily --> 累加到search_results
        累加到search_results --> 遍历plan关键词 : 下一个关键词
        累加到search_results --> 输出search_results
        输出search_results --> [*]
    }

    SearchWorker --> Evaluator

    state Evaluator {
        [*] --> 汇总search_results摘要
        汇总search_results摘要 --> 调用Ollama_LLM评估
        调用Ollama_LLM评估 --> robust_json_parse
        robust_json_parse --> 输出confidence_score
        输出confidence_score --> 递增iteration
        递增iteration --> [*]
    }

    Evaluator --> Writer : confidence ≥ 0.8<br/>或 iteration ≥ 3
    Evaluator --> Planner : confidence < 0.8<br/>且 iteration < 3

    Writer --> [*]
```

---

## 3. AgentState — 状态累积机制

```mermaid
graph LR
    subgraph 初始状态["初始状态 (run_agent)"]
        IQ["user_query: 用户问题"]
        IZ["iteration: 0"]
        CP["confidence_score: 0.0"]
        MI["missing_info: ''"]
        RK["retry_keywords: []"]
    end

    subgraph Planner输出["Planner 返回"]
        PL["plan: 关键词列表"]
    end

    subgraph Search输出["Search Worker 返回"]
        SR["search_results: 搜索结果<br/>operator.add 累加"]
    end

    subgraph Evaluator输出["Evaluator 返回"]
        CS["confidence_score: 0.0-1.0"]
        NM["missing_info: 缺失描述"]
        NK["retry_keywords: 建议关键词"]
        IT["iteration: +1"]
    end

    subgraph Writer输出["Writer 返回"]
        FR["final_report: Markdown报告"]
    end

    subgraph 合并后状态["LangGraph 自动合并"]
        ALL["完整 AgentState"]
    end

    IQ --> ALL
    IZ --> ALL
    PL --> ALL
    SR --> ALL
    CS --> ALL
    NM --> ALL
    NK --> ALL
    IT --> ALL
    FR --> ALL
```

**关键机制**：`search_results` 使用 `Annotated[List, operator.add]`，LangGraph 在多个并行搜索节点返回结果时自动拼接，而非覆盖。

---

## 4. JSON 解析 — 四层回退策略

```mermaid
flowchart TD
    INPUT["LLM 原始输出文本"]

    INPUT --> S1{"策略1<br/>_parse_direct"}
    S1 -->|"json.loads 成功<br/>且为 dict"| DONE["✅ 返回 dict"]
    S1 -->|"失败"| S2{"策略2<br/>_parse_regex_nested"}

    S2 -->|"正则匹配<br/>单层嵌套 JSON"| DONE
    S2 -->|"失败"| S3{"策略3<br/>_parse_regex_greedy"}

    S3 -->|"贪婪匹配<br/>最外层大括号"| DONE
    S3 -->|"失败"| S4{"策略4<br/>_repair_with_deepseek"}

    S4 -->|"httpx 异步调用<br/>DeepSeek 修复"| S4CHECK{"修复结果<br/>是合法 dict？"}
    S4CHECK -->|"是"| DONE
    S4CHECK -->|"否"| FAIL["❌ 返回 {}"]
    S4 -->|"无 Key / 网络异常"| FAIL

    style S1 fill:#90EE90,stroke:#333
    style S2 fill:#FFD700,stroke:#333
    style S3 fill:#FFA500,stroke:#333
    style S4 fill:#FF6347,stroke:#333
    style DONE fill:#90EE90,stroke:#333,stroke-width:3px
    style FAIL fill:#DC143C,stroke:#333,color:#fff
```

| 策略 | 函数 | 方式 | 耗时 | 可靠性 |
|------|------|------|------|--------|
| 1 | `_parse_direct` | `json.loads()` 直接解析 | ~0ms | 低（LLM输出常含多余文本） |
| 2 | `_parse_regex_nested` | 正则提取 `{...}` 最多一层嵌套 | ~0ms | 中 |
| 3 | `_parse_regex_greedy` | 正则贪婪匹配 `{.*}` | ~0ms | 中高 |
| 4 | `_repair_with_deepseek` | DeepSeek LLM 修复（httpx 异步） | ~2000ms | 高 |

---

## 5. 重试循环机制

```mermaid
sequenceDiagram
    participant P as Planner
    participant S as Search Worker
    participant E as Evaluator
    participant W as Writer

    Note over P,W: 第 1 轮

    P->>P: 生成 plan = [关键词1, 关键词2, 关键词3]
    P->>S: plan
    S->>S: 并行搜索 3 个关键词
    S->>E: search_results
    E->>E: 评估 → confidence = 0.4
    E-->>P: missing_info + retry_keywords

    Note over P,W: 第 2 轮（重试）

    P->>P: 拼接 missing_info + retry_keywords<br/>生成新 plan = [补充关键词1, 补充关键词2]
    P->>S: 新 plan
    S->>S: 搜索新关键词
    S->>E: 累积的 search_results
    E->>E: 评估 → confidence = 0.9 ≥ 0.8 ✅

    Note over P,W: 收敛 → Writer

    E->>W: 全部 search_results
    W->>W: 生成 Markdown 报告
    W-->>P: final_report（返回用户）
```

**停止条件**：
- `confidence_score >= settings.confidence_threshold (0.8)` → 进入 Writer
- `iteration >= settings.max_iterations (3)` → 强制进入 Writer（即使信心不足）

---

## 6. Tavily 搜索 — 指数退避重试

```mermaid
flowchart LR
    CALL[search_tavily 调用]
    CALL --> EXEC[TavilyClient.search<br/>search_depth=basic<br/>max_results=3]
    EXEC -->|成功| TEXT[拼接 content 字段<br/>截断至 3000 字符]
    EXEC -->|异常| RETRY[⏳ 指数退避重试<br/>最多3次<br/>等待: 2s → 4s → 8s]
    RETRY -->|第N次| EXEC
    RETRY -->|3次全部失败| ERR[抛出异常]
    TEXT --> DONE[返回摘要文本]
```

**重试配置**（`tenacity` 库）：
- 最大次数：3
- 退避策略：指数（`wait_exponential`），乘数1，最小2s，最大10s
- 触发条件：任意 `Exception`

---

## 7. 节点执行日志（可观测性）

```mermaid
flowchart TD
    ENTER[节点函数被调用] --> LOG_START[log_node 装饰器]
    LOG_START --> RECORD[记录开始时间<br/>提取 task_id / iteration]
    RECORD --> EXEC[执行节点函数]
    EXEC -->|成功| LOG_OK[记录 INFO 日志<br/>node_name + duration_ms<br/>+ task_id + iteration]
    EXEC -->|异常| LOG_ERR[记录 ERROR 日志<br/>node_name + duration_ms<br/>+ error 信息]
    LOG_ERR --> RAISE[重新抛出异常]
    LOG_OK --> RETURN[返回部分状态更新]
```

---

## 8. Metrics 指标统计

```mermaid
flowchart LR
    subgraph 计数["解析策略计数"]
        PT[parse_total<br/>总调用次数]
        PS[parse_success<br/>成功次数]
        PF[parse_fallback<br/>正则回退成功]
        PD[parse_deepseek_fallback<br/>DeepSeek修复成功]
    end

    subgraph 派生["派生指标"]
        RATE[get_parse_success_rate<br/>= parse_success / parse_total]
    end

    subgraph 持久化["自动持久化"]
        SAVE[每次 record_* 调用<br/>自动 save 到<br/>data/metrics.json]
    end

    subgraph 其他["其他指标"]
        ITERS[iterations_per_query<br/>每次查询的迭代轮次]
        TOKENS[total_tokens<br/>累计 Token 消耗]
    end

    PT --> RATE
    PS --> RATE
    PT --> SAVE
    PS --> SAVE
    ITERS --> SAVE
```

---

## 9. 配置管理

```mermaid
flowchart TD
    ENV[.env 文件] -->|pydantic-settings 自动加载| SETTINGS[Settings 单例]
    ENV_VARS[环境变量] -->|case_sensitive=False| SETTINGS

    SETTINGS -->|tavily_api_key| TAVILY[Tavily Client]
    SETTINGS -->|deepseek_api_key| DEEPSEEK[JSON 修复备援]
    SETTINGS -->|ollama_model_name| OLLAMA1[Planner LLM]
    SETTINGS -->|ollama_base_url| OLLAMA1
    SETTINGS -->|ollama_model_name| OLLAMA2[Evaluator LLM]
    SETTINGS -->|confidence_threshold| EVAL[Evaluator 阈值判断]
    SETTINGS -->|max_iterations| MAIN[main.py 停止条件]
```

---

## 10. 项目文件全景

```
AdaptiveSearchAgent/
├── main.py                         ← Agent 入口 / LangGraph 图编排
├── config.py                       ← pydantic-settings 配置
├── .env                            ← 环境变量（gitignore）
│
├── src/
│   ├── state.py                    ← AgentState TypedDict 定义
│   ├── agents/
│   │   ├── planner.py              ← 节点1：生成搜索关键词
│   │   ├── search_worker.py        ← 节点2：执行 Tavily 搜索
│   │   ├── evaluator.py            ← 节点3：评估结果充分性
│   │   └── writer.py               ← 节点4：生成 Markdown 报告
│   ├── tools/
│   │   └── search.py               ← Tavily 搜索封装 + 重试
│   └── utils/
│       ├── json_parser.py          ← 四层回退 JSON 解析
│       ├── logger.py               ← log_node 装饰器
│       └── metrics.py              ← 指标统计 + 持久化
│
├── tests/
│   ├── conftest.py                 ← create_test_state 工厂函数
│   ├── test_config.py              ← 配置挡板测试
│   ├── test_planner.py             ← Planner 集成测试
│   ├── test_search_worker.py       ← Search 集成测试
│   ├── test_evaluator.py           ← Evaluator 集成测试
│   ├── test_json_parser.py         ← JSON 解析单元测试
│   └── test_metrics.py             ← Metrics 单元测试
│
└── documents/
    └── architecture.md             ← 本文档
```
