# AdaptiveSearchAgent 开发环境

## 项目概述
Adaptive Search Agent — 基于 LangGraph/LangChain 的自适应搜索代理项目，当前处于初期搭建阶段。

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
```bash
# 方式1：先激活环境，再直接运行
eval $(poetry env activate)
python xxx.py

# 方式2：通过 poetry run 运行
poetry run python xxx.py
```

## 生产依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| langgraph | ^1.1.10 | 图状态机编排（Agent 核心框架） |
| langchain | ^1.2.17 | LLM 应用框架 |
| langchain-community | ^0.4.1 | LangChain 社区集成 |
| langchain-ollama | ^1.1.0 | LangChain Ollama 模型集成 |
| fastapi | ^0.136.1 | Web API 框架 |
| uvicorn | ^0.46.0 | ASGI 服务器 |
| tavily-python | ^0.7.24 | Tavily 搜索 API 客户端 |
| ollama | ^0.6.2 | 本地 Ollama 模型调用 |
| pydantic-settings | ^2.14.0 | 配置管理（自动加载 .env） |
| tenacity | ^9.1.4 | 重试/容错机制 |
| python-json-logger | ^4.1.0 | JSON 格式日志 |

## 开发依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| pytest | >=9.0.3 | 测试框架 |
| pytest-asyncio | >=1.3.0 | 异步测试支持 |
| black | >=26.3.1 | 代码格式化 |
| isort | >=8.0.1 | import 排序 |
| mypy | >=2.0.0 | 静态类型检查 |
| ruff | >=0.15.12 | 快速 Python linter + 格式化 |
| bandit | >=1.9.4 | 安全漏洞扫描 |
| pre-commit | >=4.6.0 | Git pre-commit 钩子管理 |

## 代码质量工具链

### Ruff 配置（pyproject.toml）
- **行长度**: 100 字符
- **目标版本**: py311
- **启用的 lint 规则**: E, F, W, I, C, B, UP, SIM, TCH
- **忽略**: E501（行长度交给格式化器处理）
- **引号风格**: 双引号

### Pre-commit 钩子（.pre-commit-config.yaml）
提交前自动运行以下检查：
1. **ruff** — lint 检查 + 自动修复（`--fix --exit-non-zero-on-fix`）
2. **ruff-format** — 代码格式化
3. **mypy** — 静态类型检查（`--ignore-missing-imports`）

```bash
# 安装 pre-commit 钩子（首次使用）
pre-commit install

# 手动对所有文件运行
pre-commit run --all-files

# 单独运行各工具
ruff check --fix .
ruff format .
mypy .
bandit -r src/
```

## 配置文件

| 文件 | 用途 |
|------|------|
| pyproject.toml | Poetry 项目定义、依赖声明、ruff 配置 |
| poetry.lock | 依赖版本锁定 |
| config.py | pydantic-settings 配置类，自动读取 .env |
| .env | 环境变量（敏感信息，勿提交） |
| .gitignore | Git 忽略规则 |
| .pre-commit-config.yaml | Pre-commit 钩子定义 |

### Settings 类可配置项（config.py）
- `TAVILY_API_KEY` — Tavily 搜索 API 密钥
- `DEEPSEEK_API_KEY` — DeepSeek API 密钥（默认空）
- `OLLAMA_BASE_URL` — Ollama 服务地址（默认空）
- `OPENAI_BASE_URL` — OpenAI 兼容 API 地址（默认空）
- `MODEL_NAME` — 模型名称（默认空）
- `OPENAI_API_KEY` — OpenAI API 密钥（默认空）

## 项目文件结构

```
AdaptiveSearchAgent/
├── pyproject.toml              # Poetry 项目配置 + ruff 配置
├── poetry.lock                 # 依赖锁定文件
├── config.py                   # 配置类（pydantic-settings）
├── .env                        # 环境变量（勿提交）
├── .gitignore                  # Git 忽略规则
├── .pre-commit-config.yaml     # Pre-commit 钩子（ruff + mypy）
├── data/                       # 数据目录（预留）
├── logs/                       # 日志目录（预留）
├── src/
│   ├── __init__.py
│   ├── state.py                # Agent 状态定义
│   ├── agents/
│   │   └── __init__.py
│   ├── tools/
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py
│       └── json_parser.py      # JSON 解析工具
└── tests/
    └── __init__.py
```

> `.mypy_cache/` 和 `.ruff_cache/` 为工具自动生成的缓存目录，已在 .gitignore 中排除。

## 注意事项

- 项目位于 WSL2 环境中（Linux 6.6.87.2-microsoft-standard-WSL2）
- 使用 `eval $(poetry env activate)` 激活环境后，python/pip 即指向虚拟环境
- **不要提交 `.env` 文件到版本控制**
- 首次克隆项目后需运行 `pre-commit install` 安装 Git 钩子
- 提交前 pre-commit 会自动运行 ruff + mypy，无需手动执行
- 提交前也可手动运行 `pytest` 确保测试通过
