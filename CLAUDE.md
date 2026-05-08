# AdaptiveSearchAgent 开发环境

## 项目概述
Adaptive Search Agent - 基于 LangGraph/LangChain 的自适应搜索代理项目。

## 环境配置

### Python 与包管理
- **Python 版本**: 3.11.15 (CPython)
- **包管理器**: Poetry
- **虚拟环境路径**: `/home/ubhuazhu/.cache/pypoetry/virtualenvs/adsearchagent-w97WTAdM-py3.11`
- **package-mode**: `false`（仅依赖管理，不构建发布包）

### 激活虚拟环境
```bash
cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent
eval $(poetry env activate)
```

### 运行命令
所有 Python 命令需在 poetry 虚拟环境下执行，两种等价方式：
```bash
# 方式1：先激活环境，再直接运行
eval $(poetry env activate)
python xxx.py

# 方式2：通过 poetry run 运行
poetry run python xxx.py
```

## 主要依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| langgraph | ^1.1.10 | 图状态机编排（Agent 核心框架） |
| langchain | ^1.2.17 | LLM 应用框架 |
| fastapi | ^0.136.1 | Web API 框架 |
| uvicorn | ^0.46.0 | ASGI 服务器 |
| tavily-python | ^0.7.24 | Tavily 搜索 API 客户端 |
| ollama | ^0.6.2 | 本地 Ollama 模型调用 |
| pydantic-settings | ^2.14.0 | 配置管理（自动加载 .env） |
| tenacity | ^9.1.4 | 重试/容错机制 |
| python-json-logger | ^4.1.0 | JSON 格式日志 |

### 开发依赖
| 依赖 | 版本 | 用途 |
|------|------|------|
| pytest | >=9.0.3 | 测试框架 |
| pytest-asyncio | >=1.3.0 | 异步测试支持 |
| black | >=26.3.1 | 代码格式化 |
| isort | >=8.0.1 | import 排序 |
| mypy | >=2.0.0 | 静态类型检查 |

## 配置文件

- **pyproject.toml** — Poetry 项目定义，包含全部依赖声明
- **config.py** — 基于 pydantic-settings 的 Settings 类，自动读取 .env 文件
- **.env** — 环境变量（API Key 等敏感信息，勿提交到版本控制）

### Settings 类可配置项
- `TAVILY_API_KEY` — Tavily 搜索 API 密钥
- `DEEPSEEK_API_KEY` — DeepSeek API 密钥（默认空）
- `OLLAMA_BASE_URL` — Ollama 服务地址（默认空）
- `OPENAI_BASE_URL` — OpenAI 兼容 API 地址（默认空）
- `MODEL_NAME` — 模型名称（默认空）
- `OPENAI_API_KEY` — OpenAI API 密钥（默认空）

## 项目文件结构
```
AdaptiveSearchAgent/
├── pyproject.toml      # Poetry 项目配置
├── poetry.lock         # 依赖锁定文件
├── config.py           # 配置类（pydantic-settings）
├── .env                # 环境变量（敏感信息，勿提交）
└── CLAUDE.md           # 本文件 — 开发环境说明
```

## 注意事项
- 项目位于 WSL2 环境中（Linux 6.6.87.2-microsoft-standard-WSL2）
- 使用 `eval $(poetry env activate)` 激活环境后，shell 中的 python/pip 即指向虚拟环境
- 不要提交 `.env` 文件到版本控制
- 格式化代码前运行 `black . && isort .`
- 提交前运行 `mypy .` 做类型检查，`pytest` 跑测试
