# LLM Provider 请求级动态切换设计文档

> 关联文件：`src/utils/llm_factory.py`、`src/agents/planner.py`、`src/agents/evaluator.py`

---

## 1. 一句话概括

**将 LLM 模型选择从"进程启动时全局固定"改为"每次请求动态指定"，调用方通过 `graph.ainvoke` 的 `config` 参数传入 provider，节点不再依赖 `.env` 全局状态。**

---

## 2. 改造前 vs 改造后对比

### 改造前（静态全局）

```mermaid
flowchart LR
    subgraph 启动时["进程启动时（一次性）"]
        ENV[".env<br/>LLM_PROVIDER=deepseek"]
    end

    subgraph 运行时["每次请求"]
        G1["graph.ainvoke(input_1)"]
        G2["graph.ainvoke(input_2)"]
    end

    subgraph 模块["模块导入时执行"]
        FAC["llm = get_llm()  ← 读 .env"]
    end

    ENV -->|一次性读取| FAC
    FAC -->|"同一个 llm 实例"| G1
    FAC -->|"同一个 llm 实例"| G2

    style ENV fill:#FF6347,stroke:#333,color:#fff
    style FAC fill:#FF6347,stroke:#333,color:#fff
```

**问题**：所有请求共用同一个 LLM 实例，改模型需要重启进程。

### 改造后（请求级动态）

```mermaid
flowchart LR
    subgraph 请求1["请求 1"]
        C1["config={'configurable':<br/>{'llm_provider':'deepseek'}}"]
    end

    subgraph 请求2["请求 2"]
        C2["config={'configurable':<br/>{'llm_provider':'ollama'}}"]
    end

    subgraph 节点["节点函数内"]
        P1["planner(state, config)"]
        P2["planner(state, config)"]
    end

    subgraph 工厂["llm_factory.py"]
        F1["get_llm(config=config)"]
        F2["get_llm(config=config)"]
    end

    subgraph 实例["动态创建的 LLM"]
        L1["ChatDeepSeek<br/>（在线 API）"]
        L2["ChatOllama<br/>（本地模型）"]
    end

    C1 --> P1 --> F1 --> L1
    C2 --> P2 --> F2 --> L2

    style C1 fill:#4169E1,stroke:#333,color:#fff
    style C2 fill:#228B22,stroke:#333,color:#fff
    style L1 fill:#4169E1,stroke:#333,color:#fff
    style L2 fill:#228B22,stroke:#333,color:#fff
```

**效果**：请求 A 走 DeepSeek，请求 B 走 Ollama，互不干扰，无需重启。

---

## 3. 完整数据流（带文件名）

```mermaid
sequenceDiagram
    participant Caller as 调用方（服务层/FastAPI）
    participant Graph as LangGraph 引擎
    participant Planner as src/agents/planner.py
    participant Evaluator as src/agents/evaluator.py
    participant Factory as src/utils/llm_factory.py
    participant DeepSeek as ChatDeepSeek<br/>api.deepseek.com
    participant Ollama as ChatOllama<br/>localhost:11434

    Note over Caller,Ollama: 请求指定 provider = "deepseek"

    Caller->>Graph: graph.ainvoke(input, config={"configurable":<br/>{"llm_provider": "deepseek"}})
    Graph->>Planner: planner(state, config)
    Planner->>Factory: get_llm(temperature=0, config=config)
    Factory->>Factory: _resolve_provider(config)<br/>→ 提取 config["configurable"]["llm_provider"]<br/>→ 返回 "deepseek"
    Factory->>DeepSeek: ChatDeepSeek(api_key, model)
    DeepSeek-->>Factory: LLM 实例
    Factory-->>Planner: ChatDeepSeek 实例
    Planner->>DeepSeek: llm.ainvoke(prompt)
    DeepSeek-->>Planner: 生成的 plan 关键词
    Planner-->>Graph: {"plan": [...]}

    Graph->>Evaluator: evaluator(state, config)
    Evaluator->>Factory: get_llm(temperature=0, config=config)
    Factory->>Factory: _resolve_provider(config)<br/>→ 同一个 config，返回 "deepseek"
    Factory->>DeepSeek: ChatDeepSeek(api_key, model)
    Factory-->>Evaluator: ChatDeepSeek 实例
    Evaluator->>DeepSeek: llm.ainvoke(prompt)
    DeepSeek-->>Evaluator: 评估结果
    Evaluator-->>Graph: {"confidence_score": ...}
```

---

## 4. 改造的 3 个文件 —— 改动内容与设计意图

### 4.1 `src/utils/llm_factory.py`（核心）

```mermaid
flowchart TD
    INPUT["get_llm(temperature, llm_provider, config)"]
    INPUT --> CHECK1{"llm_provider<br/>参数被传入？"}
    CHECK1 -->|是| USE1["使用 llm_provider<br/>（测试路径）"]
    CHECK1 -->|否| CHECK2{"config 中有<br/>configurable.llm_provider？"}
    CHECK2 -->|是| USE2["使用 config 中的值<br/>（生产路径）"]
    CHECK2 -->|否| USE3["硬编码默认 'ollama'"]

    USE1 --> BRANCH{"provider ==<br/>'deepseek'？"}
    USE2 --> BRANCH
    USE3 --> BRANCH

    BRANCH -->|是| DEEPSEEK["ChatDeepSeek(<br/>model, api_key, temperature)"]
    BRANCH -->|否| OLLAMA["ChatOllama(<br/>model, base_url, temperature)"]

    style CHECK1 fill:#FFD700,stroke:#333
    style CHECK2 fill:#4169E1,stroke:#333,color:#fff
    style USE1 fill:#FFD700,stroke:#333
    style USE2 fill:#4169E1,stroke:#333,color:#fff
    style USE3 fill:#808080,stroke:#333,color:#fff
```

#### 改动了什么

| 位置 | 改动前 | 改动后 |
|------|--------|--------|
| 新增 `_resolve_provider()` 函数 | 不存在 | 从 `config["configurable"]["llm_provider"]` 提取值，**绝不读取 `settings.llm_provider`** |
| `get_llm()` 签名 | `get_llm(temperature)` 只接受温度 | 新增 `llm_provider` 和 `config` 两个可选参数 |
| `get_llm()` provider 来源 | `settings.llm_provider`（读 .env） | `llm_provider 参数 → config → "ollama"` 三级回退 |
| 模块级 docstring | 描述 env 切换 | 明确标注"生产路径 / 测试路径" |

#### 为什么这样设计

1. **分离关注点**：生产路径（config）和测试路径（llm_provider 参数）泾渭分明，测试可以绕过 config 直接指定 provider
2. **不读 .env**：`.env` 是进程级全局状态，取消读取意味着 provider 的决策权完全交给调用方，服务层可以按用户、按 API Key、按租户任意策略选择模型
3. **"ollama" 硬编码回退**：当所有途径都未指定时，安全回退到本地模型，保证系统不会因缺失配置而崩溃

---

### 4.2 `src/agents/planner.py`（节点函数）

```mermaid
flowchart TD
    subgraph 改造前["改造前"]
        A1["模块导入时执行:<br/>llm = get_llm()  ← 一次性固化"]
        A2["async def planner(state):<br/>    response = await llm.ainvoke(prompt)"]
        A1 -.->|"llm 是模块级变量"| A2
    end

    subgraph 改造后["改造后"]
        B1["async def planner(state, config):<br/>    llm = get_llm(temperature=0, config=config)<br/>    response = await llm.ainvoke(prompt)"]
    end

    style A1 fill:#FF6347,stroke:#333,color:#fff
    style B1 fill:#4169E1,stroke:#333,color:#fff
```

#### 改动了什么

| 位置 | 改动前 | 改动后 |
|------|--------|--------|
| 模块级变量 | `llm = get_llm(temperature=0)` 在导入时创建 | **已删除** |
| 函数签名 | `async def planner(state: AgentState) -> dict` | `async def planner(state: AgentState, config: RunnableConfig \| None = None) -> dict` |
| LLM 创建位置 | 模块顶层（导入时执行一次） | 函数体内部（每次调用时动态创建） |
| 新增 import | 无 | `from langchain_core.runnables import RunnableConfig` |

#### 为什么这样设计

1. **每次请求新建 LLM 实例**：`ChatOllama` / `ChatDeepSeek` 本质是纯参数封装（不含连接池），创建开销微秒级，换来 provider 的动态切换能力
2. **`config` 参数可选**：设为 `None` 默认值，测试代码 `await planner(state)` 仍然可以不加 config 直接调用（此时走硬编码 ollama 回退）
3. **LangGraph 自动注入**：当通过 `graph.ainvoke(input, config=...)` 调用时，LangGraph 引擎自动将 config 作为第二个位置参数传入节点函数，无需手动传递

---

### 4.3 `src/agents/evaluator.py`（节点函数）

与 `planner.py` 改动完全一致：删除模块级 `llm`，新增 `config` 参数，函数内动态创建。

额外注意：`@log_node("evaluator")` 装饰器内部使用 `*args, **kwargs` 透传参数，新增的 `config` 参数不受影响。

---

## 5. 调用方式全景

```mermaid
flowchart TB
    subgraph 生产路径["生产路径（服务层/FastAPI）"]
        PROD["graph.ainvoke(input,<br/>config={'configurable':<br/>{'llm_provider': 'deepseek'}})"]
    end

    subgraph 测试路径["测试路径（pytest 内部）"]
        TEST["get_llm(llm_provider='ollama')<br/>不暴露给外部调用方"]
    end

    subgraph 节点["LangGraph 节点"]
        NODE["planner(state, config)<br/>evaluator(state, config)"]
    end

    subgraph 工厂["src/utils/llm_factory.py"]
        RESOLVER["_resolve_provider(config)<br/>↓<br/>提取 config['configurable']['llm_provider']"]
        FACTORY["get_llm(...)<br/>↓<br/>provider 决定 ChatDeepSeek / ChatOllama"]
    end

    PROD -->|"config 自动注入"| NODE
    NODE -->|"get_llm(config=config)"| RESOLVER
    RESOLVER --> FACTORY

    TEST -.->|"仅供测试"| FACTORY

    style PROD fill:#4169E1,stroke:#333,color:#fff
    style TEST fill:#FFD700,stroke:#333
```

---

## 6. 设计原则总结

| 原则 | 体现 |
|------|------|
| **无状态节点** | 节点函数不从全局变量读取 provider，仅依赖入参 `config`，可安全水平扩展 |
| **决策权上移** | provider 选择由调用方（服务层）决定，而非下沉到基础设施代码 |
| **测试友好** | `llm_provider` 参数为测试保留直接注入通道，不影响生产路径 |
| **安全回退** | 所有路径都未指定时硬编码默认 `"ollama"`，不发生 `NoneType` 崩溃 |
| **最小改动** | 仅修改 1 个工厂 + 2 个节点函数，`search_worker.py` / `writer.py` 等零改动 |
