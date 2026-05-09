# AGENTS.md — AdaptiveSearchAgent 项目入口文档

> **目标读者**：AI 编程 Agent。阅读本文档后即可直接开始开发，无需额外查阅其他文件。

---

## 1. 项目身份

| 项目名 | AdaptiveSearchAgent |
|--------|---------------------|
| 简称 | adsearchagent |
| 描述 | 基于 LangGraph / LangChain 的自适应搜索代理 |
| 当前阶段 | 初期搭建 — 基础设施已就绪，核心逻辑待实现 |
| 运行环境 | WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2) |
| 作者 | suimo (huazhu7k7k@163.com) |

---

## 2. 环境启动（新 Agent 接入第一件事）

```bash
# 进入项目
cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent

# 激活 Poetry 虚拟环境
eval $(poetry env activate)

# （首次）安装依赖
poetry install

# （首次）安装 pre-commit Git 钩子
pre-commit install
```

- **Python 版本**: 3.11.15 (CPython)
- **虚拟环境路径**: `/home/ubhuazhu/.cache/pypoetry/virtualenvs/adsearchagent-w97WTAdM-py3.11`
- **运行命令**: 激活环境后直接 `python xxx.py`，或使用 `poetry run python xxx.py`

---

## 3. 技术栈

### 生产依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| langgraph | ^1.1.10 | 图状态机编排（Agent 核心） |
| langchain | ^1.2.17 | LLM 应用框架 |
| langchain-community | ^0.4.1 | LangChain 社区集成 |
| langchain-ollama | ^1.1.0 | LangChain ↔ Ollama 桥接 |
| fastapi | ^0.136.1 | Web API 框架 |
| uvicorn | ^0.46.0 | ASGI 服务器 |
| tavily-python | ^0.7.24 | Tavily 搜索 API |
| ollama | ^0.6.2 | 本地 Ollama 模型调用 |
| pydantic-settings | ^2.14.0 | 配置管理（自动读 .env） |
| tenacity | ^9.1.4 | 重试与容错 |
| python-json-logger | ^4.1.0 | JSON 格式日志 |

### 开发依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| pytest | >=9.0.3 | 测试框架 |
| pytest-asyncio | >=1.3.0 | 异步测试 |
| ruff | >=0.15.12 | Lint + 格式化 |
| mypy | >=2.0.0 | 静态类型检查 |
| black | >=26.3.1 | 代码格式化（备用） |
| isort | >=8.0.1 | import 排序（备用） |
| bandit | >=1.9.4 | 安全漏洞扫描 |
| pre-commit | >=4.6.0 | Git 提交前自动检查 |

---

## 4. 项目文件结构

```
AdaptiveSearchAgent/
├── AGENTS.md                     # ← 本文件：AI Agent 入口文档
├── CLAUDE.md                     # 开发环境详细说明（辅助参考）
├── .clinerules/development.md    # 项目开发规范（工作流程/编码/Git/语言）
├── pyproject.toml                # Poetry 项目配置 + ruff 配置
├── poetry.lock                   # 依赖版本锁定
├── config.py                     # pydantic-settings 配置类
├── .env                          # 环境变量（敏感信息，gitignore 已排除）
├── .gitignore                    # Git 忽略规则
├── .pre-commit-config.yaml       # Pre-commit 钩子（ruff + mypy）
├── data/                         # 数据目录（预留，空）
├── logs/                         # 日志目录（预留，空）
├── src/
│   ├── __init__.py               # 空
│   ├── state.py                  # Agent 状态定义（空，待实现）
│   ├── agents/
│   │   └── __init__.py           # 空
│   ├── tools/
│   │   └── __init__.py           # 空
│   └── utils/
│       ├── __init__.py           # 空
│       └── json_parser.py        # JSON 解析工具（空，待实现）
└── tests/
    └── __init__.py               # 空
```

### 已排除（gitignore）

| 路径 | 原因 |
|------|------|
| `.env` | 含 API Key |
| `.deepcode/` `.claude/` | AI 工具配置，含 API Key / 自动生成内容 |
| `.mypy_cache/` `.ruff_cache/` | 工具缓存 |
| `test_skills.py` | 临时脚本 |
| `.vscode/` `.idea/` | IDE 配置 |

---

## 5. 当前实现状态

| 模块 | 状态 | 说明 |
|------|------|------|
| `src/state.py` | 空 | Agent 状态定义待实现 |
| `src/agents/` | 空 | Agent 节点逻辑待实现 |
| `src/tools/` | 空 | 工具函数待实现 |
| `src/utils/json_parser.py` | 空 | JSON 解析待实现 |
| `config.py` | 已完成 | Settings 类，支持 .env 自动加载 |
| `tests/` | 空 | 测试待编写 |

> **总结**：项目骨架已搭好，所有包管理、代码质量工具链、配置管理已就绪，可直接开始编写核心业务代码。

---

## 6. 开发工作流

Agent 在接到开发任务时，按以下步骤执行：

```
1. 需求理解
   └→ 分析请求，确认目标

2. 技术设计（复杂功能）
   └→ 输出简要技术方案，获得确认后再编码

3. 编码实现
   ├→ 遵循 Python 编码规范（见下文）
   ├→ 必要时补充单元测试
   └→ 确保 import 顺序和格式正确

4. 自动化验证（编码后立即执行）
   ├→ ruff check --fix .     # 自动修复 lint 问题
   ├→ ruff format .           # 格式化代码
   ├→ mypy .                  # 静态类型检查
   └→ pytest                  # 运行测试

5. 提交代码
   └→ 遵循 Git 提交规范（见下文）
```

---

## 7. Python 编码规范

| 规则 | 要求 |
|------|------|
| 行长度 | ≤ 100 字符 |
| 引号风格 | 双引号 |
| 文档字符串 | 三重双引号 `"""` |
| 类型注解 | 所有函数参数和返回值必须添加 |
| 导入顺序 | 标准库 → 第三方 → 本地模块，组间空一行 |
| 错误处理 | EAFP 风格（`try/except KeyError`），避免 `if 'key' in dict` |
| 配置读取 | API Key / URL 等必须从 `config.py` 导入，**禁止硬编码** |
| 禁止项 | `eval()`、`exec()`、动态构造 SQL |

### Ruff 配置（pyproject.toml）

- 目标版本: `py311`
- 启用规则: `E, F, W, I, C, B, UP, SIM, TCH`
- 忽略: `E501`（行长度由格式化器处理）

---

## 8. 配置管理

所有配置通过 `config.py` 的 `Settings` 类集中管理，自动读取 `.env` 文件：

```python
from config import settings

# 使用示例
settings.tavily_api_key
settings.deepseek_api_key
settings.MODEL_NAME
```

### 可配置项

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `tavily_api_key` | Tavily 搜索 API 密钥 | 必填 |
| `deepseek_api_key` | DeepSeek API 密钥 | `""` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `""` |
| `OPENAI_BASE_URL` | OpenAI 兼容 API 地址 | `""` |
| `MODEL_NAME` | 模型名称 | `""` |
| `OPENAI_API_KEY` | OpenAI API 密钥 | `""` |

---

## 9. Git 提交规范

```
格式：<类型>(<作用域>): <描述>

类型：
  feat     — 新功能
  fix      — 修复
  docs     — 文档
  style    — 格式
  refactor — 重构
  test     — 测试
  chore    — 杂项

示例：
  feat(planner): 添加搜索计划生成节点
  fix(state): 修复 Agent 状态序列化错误
```

---

## 10. 沟通语言规范

| 规则 | 要求 |
|------|------|
| 对话语言 | 始终使用**简体中文**（解释、思考、回复） |
| 技术术语 | 可保留英文或加注中文 |
| 代码注释 | 使用中文 |
| 口语词 | ok、todo、fix、wip 等可保留原文 |

---

## 11. 相关文档

| 文件 | 用途 |
|------|------|
| `CLAUDE.md` | 开发环境详细说明（环境配置、依赖表、工具链） |
| `.clinerules/development.md` | 开发规范原文（工作流程、编码、Git、语言） |
| `pyproject.toml` | Poetry 依赖声明 + ruff 配置 |
| `.pre-commit-config.yaml` | Pre-commit 钩子定义 |

---

> **Agent 快速启动清单**：
> 1. `eval $(poetry env activate)` — 激活环境
> 2. 阅读上文"当前实现状态"了解哪些模块待开发
> 3. 编码后运行 `ruff check --fix . && ruff format . && mypy . && pytest`
> 4. 提交时遵循 `类型(作用域): 描述` 格式
> 5. 始终用简体中文交流
