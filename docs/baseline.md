# T0-1 基线快照

> 生成时间：2026-11-08 23:52（Asia/Shanghai）
> 分支：`card/T0-1`（基于 `d51650b`）
> 用途：后续所有改造的回归对比基准。只修环境、不动业务逻辑。
>
> **更新记录**
> - 2026-11-09：裁判 `qwen3.8-flash` 额度到期，换成 `qwen3.8-max-0902` 并重跑基线；
>   同时移除评测脚本里的 WSL 原生网关兜底，RAG 统一走 `ingest_dev` 容器（8100）。
>   → **第 4 节的数字与旧 flash 版本不可横比，旧的已作废。**

---

## 1. 环境修复（本卡完成）

| 问题 | 现象 | 修复 |
|---|---|---|
| `requirements.txt` 与 `pyproject.toml` 解析结果错位 | `requirements` 里 `langchain-core==1.3.3`、`langchain==1.2.17`，实际安装 `1.6.3` / `1.4.0`；langgraph 家族同样落后 | 把 `langchain*` / `langgraph*` 共 **14 个** pin 对齐到已验证的安装版本（见下表） |
| RAG 检索稳定超时 | `src/tools/rag.py` 客户端超时 **10s**，而 BGE 精排单次 **10.9s**；结果：20/20 题检索为空、Ragas 三项指标接近 0 | 新增 `RAG_TIMEOUT_SECONDS=30`（`config.py`），客户端改用该配置；熔断逻辑不变 |

**对齐后的关键版本**

| 包 | 版本 |
|---|---|
| langchain | 1.4.0 |
| langchain-core | 1.6.3 |
| langchain-ollama | 1.1.0 |
| langchain-community | 0.4.1 |
| langgraph | 1.2.11 |

**RAG 实测延迟**：`POST /api/v1/search`（kb_id=eval_v3, bge-m3, BGE 精排）≈ **10.9s/次**（冷热一致）。
这就是超时必须 ≥30s 的原因；低于 11s 会稳定超时。

## 2. 测试基线

| 项 | 值 |
|---|---|
| `pytest -q` | **223 passed, 4 skipped, 4 deselected**（后端未启动时） |
| `pytest --collect-only -q` | **227/231 collected（4 deselected）**，无 import 失败 |
| 测试文件数 | 27 |
| `.pytest_cache/v/cache/nodeids` | 228 行，sha256 `a57008dd1a22cb16ff5096dda448dcee641f266ce63786dc96eb7bfe25657334` |

> 回归时先比"总数与 nodeids 哈希"：数量变化不一定是回归，但**nodeids 哈希变化必须能解释**。
> 重新生成：`pytest --collect-only -q` 后取其 sha256。

## 3. 代码规模

| 项 | 行数 |
|---|---|
| `api_main.py` | 1063 |
| `src/` 全部 `.py`（40 个文件） | 4704 |
| `tests/` 全部 `.py` | 4936 |

## 4. Ragas 检索层基线（20 题黄金集）

配置：`evals/datasets/qa_dataset_v3.yaml`（sha256 `367f35c1ec11…`）、`kb_id=eval_v3`、
embedding `bge-m3`、reranker `bge`、生成 `qwen35-9b-official`、裁判 **`qwen3.8-max-0902`**、
top_k=3、workers=**2**、RAG 网关 **`http://localhost:8100`（ingest_dev 容器）**。

| 指标 | mean | median | min |
|---|---|---|---|
| context_recall | **1.0000** | 1.0000 | 1.0 |
| faithfulness | **0.8030** | 1.0000 | 0.0 |
| answer_relevancy | **0.8339** | 0.8412 | 0.6204 |

- 检索空结果：**0 次**；生成降级：**0 次**；
- 归因参考：hit@1 **95%**、hit@3 **100%**、hit@3（精排后）**100%**；
- 上下文切片：min/max/avg = 3/3/3；
- 耗时：检索 212.5s + 评估 467.1s = **679.6s**；
- 被测 commit：`5893a2a`（dirty files: 0）；
- 报告目录：`文档/项目/AdaptiveSearchAgent/评测报告/baseline_8100/`
  （`run_meta.txt` 记录了裁判/网关/并发，可核对）。

**回归判定**

- `context_recall` 必须保持 **1.0**（检索掉链子会直接反映在这里）；
- `faithfulness` 优先看**中位数（当前 1.0）与 0 分题数（当前 3 题）**，均值只作参考：
  实测 0 分题多来自裁判输出截断（`IncompleteOutputException`），单题塌陷就能让均值掉 ~0.05，
  比指标本身的波动还大；
- `answer_relevancy` 均值下降 **> 0.04** 再判退化（同环境下两次跑的实测差约 0.02）。

> 追查 0 分题时先看 `run.log` 有没有 `IncompleteOutputException`——那是裁判问题，不是系统变差。
> 旧 flash 基线（faith 0.9205 / relevancy 0.8983）已作废，仅作历史记录，**不要拿来对比**。

## 5. 固定环境信息

| 项 | 值 |
|---|---|
| Python | 3.11.15 |
| 虚拟环境 | `/home/ubhuazhu/.cache/pypoetry/virtualenvs/adsearchagent-w97WTAdM-py3.11` |
| RAG 服务 | `dev_pros-ingest_dev-1`（容器），8100 端口，healthy；**基线只认这条路径**——评测脚本已移除 WSL 原生网关兜底，8100 不可用时直接报错 |
| Redis / Postgres | `redis_cache` / `postgres_dev`，healthy |
| Ollama | `qwen35-9b-official:latest`（11434） |
| 裁判 | `qwen3.8-max-0902`（阿里云百炼，`.env` 的 `ALIYUN_QWEN_MODEL_NAME`） |

## 6. 复跑命令（照抄）

```bash
cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent
eval $(poetry env activate)

pytest -q                      # 期望 223 passed, 4 skipped, 4 deselected
pytest --collect-only -q       # 期望 227/231 collected

# Ragas（约 11 分钟）
# 前置：RAG 网关必须在跑，否则脚本直接失败（不再兜底）
cd /home/ubhuazhu/dev_pros && docker compose up -d ingest_dev
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8100/health   # 期望 200

cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent
./evals/run_baseline.sh --tag baseline_check
```
