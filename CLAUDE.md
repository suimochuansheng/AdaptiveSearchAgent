# AdaptiveSearchAgent – 项目认知基座

## 项目定位
基于 LangGraph 的自适应搜索 Agent，具备：
- 自适应搜索闭环（评估→重试关键词）
- 并行搜索 + 扇入同步
- 双模型备援（Ollama Qwen2.5 本地 + DeepSeek API）
- 生产级可观测性（结构化日志、Semaphore 并发控制、安全数学计算）

## 技术栈摘要
- Python 3.11 + Poetry
- 核心：LangGraph, LangChain, Tavily, Ollama
- 质量：ruff, mypy, pytest, pre-commit

## 架构核心
- 状态定义：`AgentState`，`search_results` 使用 `operator.add` 实现并行合并
- 节点：Planner → 并行搜索 → Evaluator → (重试) → Human-in-the-loop → Writer
- 并发控制：`asyncio.Semaphore(2)` 限制 LLM 并发

## 环境快速激活
```bash
cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent
eval $(poetry env activate)
```

## 关键文件位置
- 配置：config.py（读取 .env）
- 状态：src/state.py
- 工具：src/utils/json_parser.py
- 测试：tests/

