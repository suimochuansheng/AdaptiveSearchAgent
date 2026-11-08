# evals/ — 离线评测

> 评测代码和被测代码**同版本**放在一起：基线按 `git commit` 对比，产物写到仓库外。

## 入口

| 脚本 | 评什么 | 需要 |
|---|---|---|
| `rag_quality.py` | 检索层（context_recall / faithfulness / answer_relevancy） | RAG 网关、Ollama、Ragas 依赖 |
| `agent_quality.py` | Agent 端到端答案质量（LLM judge 1~10 分） | 后端、裁判 API |
| `run_baseline.sh` | 一条龙：起服务、跑 Ragas、归档报告 | Docker、Ollama、后端 |

## 常用命令

```bash
# 检索层（约 12 分钟）
poetry run python evals/rag_quality.py --model bge-m3

# Agent 端到端（先 3 题冒烟）
poetry run python evals/agent_quality.py --limit 3 --label smoke
poetry run python evals/agent_quality.py --label current

# 与基线对比（平均分下降 > 2 分退出码 1）
poetry run python evals/agent_quality.py --compare <baseline.json> --output <current.json>
```

## 文件约定

- **数据集**：`evals/datasets/*.yaml`（进 git，保证可复现）；
- **产物**：`../文档/项目/AdaptiveSearchAgent/评测报告/`（由 `EVAL_REPORT_DIR` 控制，不进 git）；
- **基线**：报告里记 `git_commit` + `dataset_sha256` + `judge_model`，三者必须一致才可比。

## 三条可比性规则

1. 同一批题（加新题不重跑基线，按**共同题**算均值）；
2. 同一裁判模型（见报告里的 `judge_model`）；
3. 同一被测模型与环境（同一份 `.env`、同样起 `ingest_dev`）。
