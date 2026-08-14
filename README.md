# 项目组成

AdaptiveSearchAgent —— 自适应搜索 Agent 主服务（FastAPI + LangGraph + pgvector RAG）。

## 🚀 快速启动（Docker Compose）

在父目录 `dev_pros` 执行：

```bash
docker compose up -d
```

服务启动后访问：

- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 本地开发（非容器）

```bash
poetry install
poetry run uvicorn api_main:app --reload --port 8000
```
