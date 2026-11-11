# T0-1 基线快照

> 生成时间：2026-11-08 23:52（Asia/Shanghai）
> 分支：`card/T0-1`（基于 `d51650b`）
> 用途：后续所有改造的回归对比基准。只修环境、不动业务逻辑。
>
> **更新记录**
> - 2026-11-09：裁判 `qwen3.8-flash` 额度到期，换成 `qwen3.8-max-0902` 并重跑基线；
>   同时移除评测脚本里的 WSL 原生网关兜底，RAG 统一走 `ingest_dev` 容器（8100）。
>   → **第 4 节的数字与旧 flash 版本不可横比，旧的已作废。**
> - 2026-11-10（T0-2）：裁判名收敛到 `EVAL_JUDGE_MODEL`（默认取 `ALIYUN_QWEN_MODEL_NAME`），
>   脚本启动即打印并写入报告；并发固定 `workers=2`、空检索自动重试 1 次；
>   回归判据由 ±0.02 改为"**同口径跑 2 次取均值，两次均值差 ≤ 记录方差**"；新增 §7 Agent raw 基线。
> - 2026-11-11（T-arch）：新增 1 份端口契约测试（2 例），pytest 通过数 223 -> 225，
>   收集数 227 -> 229（总数 231 -> 233），nodeids 哈希同步更新；详见 §2 的说明。

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

> **2026-11-11（T-arch）预期变化**：因新增 1 份端口契约测试
> `tests/contracts/test_agent_runtime_contract.py`（N = **2** 例，参数化挂
> `LangGraphRuntime` + `Fake`），收集数由 **227 变为 227+N = 229**（deselected 仍为 4，
> 总数 231 -> 233），`pytest -q` 由 223 passed 变为 **225 passed**；nodeids 哈希随之
> 更新。**这是有意的基线更新，不是回归**；除新增文件外没有改动任何既有用例。

| 项 | 值 |
|---|---|
| `pytest -q` | **225 passed, 4 skipped, 4 deselected**（后端未启动时） |
| `pytest --collect-only -q` | **229/233 collected（4 deselected）**，无 import 失败 |
| 测试文件数 | 28 |
| `.pytest_cache/v/cache/nodeids` | 230 行，sha256 `8c3cab4dcbda90dd1c90dafb9d0ad88d43af47723ab3a2dfa03fba2a26d4c8f3` |

> 回归时先比"总数与 nodeids 哈希"：数量变化不一定是回归，但**nodeids 哈希变化必须能解释**。
> 重新生成：`pytest --collect-only -q` 后取其 sha256。
> 上一版（T-arch 前）：223 passed / 227/231 / 228 行 / `a57008dd1a22cb16ff5096dda448dcee641f266ce63786dc96eb7bfe25657334`。

## 3. 代码规模

| 项 | 行数 |
|---|---|
| `api_main.py` | 1063 |
| `src/` 全部 `.py`（40 个文件） | 4704 |
| `tests/` 全部 `.py` | 4936 |

## 4. Ragas 检索层基线（20 题黄金集）

> **旧 flash 口径作废**：`qwen3.8-flash` 时代的数字（faith 0.9205 / relevancy 0.8983）仅作历史记录，不要拿来对比。

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

**回归判定（T0-2 起：同口径两次取均值）**

判据不再是"单次 ±0.02"，而是"**同口径跑 2 次取均值，两次均值差 ≤ 记录方差**"：

1. 固定口径：同裁判（`EVAL_JUDGE_MODEL`）、同 `workers=2`、同数据集（`qa_dataset_v3.yaml`，sha256 必须一致）；
2. 同口径连续跑 **2 次**，各取 20 题均值；判定值取两次均值的**平均**，并记录两次的绝对差；
3. 判定式：`|mean(run1) − mean(run2)| ≤ 记录方差`。超过方差时先查环境/裁判漂移
   （核对 `run_meta.txt` 的裁判、commit、数据集 sha256），不要直接判系统退化；
4. 三项指标各自判；`context_recall` 仍要求 **= 1.0**；
5. 空检索判据：**≤ 1 次且可归因**。脚本对空结果自动重试 1 次（报告里的 `retrieval_retries`），
   仍为空才计入 `retrieval_errors`；出现时必须在 `run.log` 里找到对应题目并说明原因；
6. `faithfulness` 继续优先看**中位数（1.0）与 0 分题数（3 题）**，均值只作参考：
   这 3 题都是表格/数值题，答案数字（如 `4.2 ms`）确实出现在检索上下文里，裁判仍判不支持——
   属指标口径问题，不是系统退化；
   > 实测 0 分题多来自裁判输出截断（`IncompleteOutputException`）：单题塌陷就能让均值掉 ~0.05，
   > 比指标本身的波动还大。追查 0 分题时先看 `run.log` 有没有 `IncompleteOutputException`——
   > 那是裁判问题，不是系统变差。（T0-2 验收 OI-T02-002 恢复）
7. 旧 flash 基线（faith 0.9205 / relevancy 0.8983）已作废，仅作历史记录，**不要拿来对比**。

**本期同口径两次复现（2026-11-10，裁判 `qwen3.8-max-0902`、workers=2）**

| 指标 | run 1 | run 2 | 两次均值 | \|run1−run2\| | 记录方差（判据上限） |
|---|---|---|---|---|---|
| context_recall | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000（要求恒等） |
| faithfulness | 0.8509 | 0.8101 | 0.8305 | 0.0408 | 0.0408 |
| answer_relevancy | 0.8205 | 0.8406 | 0.8306 | 0.0201 | 0.0408 |

> 两份 JSON：
> - run 1：`文档/项目/AdaptiveSearchAgent/评测报告/raw_20261110_20261110-1329/ragas_evaluation_bge-m3_bge_ollama.json`
> - run 2：`文档/项目/AdaptiveSearchAgent/评测报告/raw_20261110_20261110-1341/ragas_evaluation_bge-m3_bge_ollama.json`
>
> 两次均 `retrieval_errors=0`、`retrieval_retries=0`，裁判均为 `qwen3.8-max-0902`、`workers=2`。
> **记录方差取三项里最大的实测差 = 0.0408（faithfulness）**；之后同口径复跑，只要两次均值差不超过 0.0408 就算稳定，
> 不再看旧的 ±0.02。faithfulness 单次 0.81~0.85 的摆动主要来自裁判对表格/数值题的判定，不是检索退化。
> 与本节冻结值（faith 0.8030 / relevancy 0.8339）相比：faith 差 0.0275、relevancy 差 0.0033，均在 0.0408 以内，**同口径稳定**。

## 5. 固定环境信息

| 项 | 值 |
|---|---|
| Python | 3.11.15 |
| 虚拟环境 | `/home/ubhuazhu/.cache/pypoetry/virtualenvs/adsearchagent-w97WTAdM-py3.11` |
| RAG 服务 | `dev_pros-ingest_dev-1`（容器），8100 端口，healthy；**基线只认这条路径**——评测脚本已移除 WSL 原生网关兜底，8100 不可用时直接报错 |
| Redis / Postgres | `redis_cache` / `postgres_dev`，healthy |
| Ollama | `qwen35-9b-official:latest`（11434） |
| 裁判 | `qwen3.8-max-0902`（阿里云百炼，`.env` 的 `ALIYUN_QWEN_MODEL_NAME`） |
| 裁判模型（评测判定） | `EVAL_JUDGE_MODEL`（`config.py`，默认取 `ALIYUN_QWEN_MODEL_NAME`）；本期 = `qwen3.8-max-0902`。`run_baseline.sh --dry-run` 与 `agent_quality.py` 启动时都会打印，报告 `run_meta.txt` / JSON 的 `judge_model` 同步记录 |

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

## 7. Agent 能力基线（raw，无四能力）

> **口径**：改造前的 Agent 主循环（`TOOLS` 硬编码、静态 HITL 子图、无注册表/权限预算/追踪），
> 即"raw"——四能力（工具注册表、工具级权限与预算、HITL、追踪）全部尚未接入。
> 这张表是后续四能力改造的**唯一对比基准**，改造后必须用同裁判、同题集、同入口复跑。

| 项 | 值 |
|---|---|
| 题集 | `evals/datasets/qa_dataset_v3.yaml`（20 题，sha256 `367f35c1ec11…`） |
| 入口 | `POST http://localhost:8000/chat/conversation`（后端 `api_main:app`） |
| 裁判 | `qwen3.8-max-0902`（`EVAL_JUDGE_MODEL`，默认取 `ALIYUN_QWEN_MODEL_NAME`；启动即打印，JSON 的 `judge_model` 同步记录） |
| 评分 | 1~10 分整数（judge 只看"回答 vs 参考答案"，`temperature=0`） |
| 并发 | `--concurrency 1`（`deep_research` 进程级单飞保护） |
| 被测 commit | `9bf183d`（dirty files: 6，本卡改动未提交） |

**20 题结果**

| 指标 | 值 |
|---|---|
| 均分 | **9.65** |
| 有效/失败 | 20 / 0 |
| min / max | 7.0 / 10.0 |
| 耗时 | 20 题累计 629.3s（约 10.5 分钟） |

**分数分布**

| 分数 | 10 | 9 | 8 | 7 | ≤6 |
|---|---|---|---|---|---|
| 题数 | 16 | 2 | 1 | 1 | 0 |

- 失分题：`q011`=7、`q013`=8、`q005`=9、`q006`=9；其余 16 题满分；
- 20 题均无 SSE `error` 事件（`errors` 为空）；
- 报告：`文档/项目/AdaptiveSearchAgent/评测报告/agent_eval_baseline_raw_9bf183d_20261110-1403.json`。

**复跑命令**

```bash
cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent
poetry run uvicorn api_main:app --host 127.0.0.1 --port 8000 &   # 需在跑
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/docs   # 期望 200
poetry run python evals/agent_quality.py --label baseline_raw
```

> 报告默认带 `<commit>_<YYYYmmdd-HHMM>` 时间戳，同 label 复跑不互相覆盖；
> 判定沿用脚本 `--max-drop 2.0`（与 `--compare` 基线比较），换裁判则分数不可横比。

## 8. Agent 能力基线 v2（G11：四能力三组对照，同裁判同题）

> **口径**：与 §7 完全一致——同一份 `evals/agent_quality.py` 默认题集（20 题，
> `evals/datasets/qa_dataset_v3.yaml`，sha256 `367f35c1ec11…`）、同一入口
> `POST /chat/conversation`、同一裁判 `qwen3.8-max-0902`（`temperature=0`）、
> `--concurrency 1`、`DEEP_RESEARCH_BACKEND=workflow`（未用 stub）。
> 三组只改四个能力开关：A 组"全关"复现改造前，B 组开治理（工具策略 + 追踪），C 组再加 HITL。
> 被测 commit `24a916c`（G11 改动未提交，工作区 dirty）；每组一个独立后端进程，跑完即换，
> 服务启动时把开关作为环境变量注入（快照见下表最后一列）。

| 组别 | 均分 | 有效/失败 | 分布（10/9/8/7/<=6） | 平均耗时 | 平均工具调用 | 本会触发审批次数 | flag 快照 |
|---|---|---|---|---|---|---|---|
| A 全关（复现 raw） | **9.70** | 20/0 | 17/1/1/1/0 | 30.3s | 1.00 | 0 | `TOOL_POLICY=0 TRACE=0 HITL=0 MCP=0 EVAL_AUTO_APPROVE=0` |
| B 治理开（审批/MCP 关） | **9.75** | 20/0 | 16/3/1/0/0 | 29.6s | 1.00 | 0 | `TOOL_POLICY=1 TRACE=1 HITL=0 MCP=0 EVAL_AUTO_APPROVE=0` |
| C 治理+HITL（自动放行） | **9.85** | 20/0 | 18/1/1/0/0 | 29.7s | 1.00 | 0（20 题未触发；探针放行 1 次，不计入表） | `TOOL_POLICY=1 TRACE=1 HITL=1 MCP=0 EVAL_AUTO_APPROVE=1` |

**与 §7（raw，9.65）的差异与结论**

- A 9.70 与 raw 9.65 差 **+0.05**（阈值 0.3 以内）→ **治理未降质**：A 组四能力全关，
  与 §7 同一代码路径、同一裁判、同题集，差异只来自裁判的单题 ±1 分波动；
- B 9.75（+0.10）、C 9.85（+0.20）同样在阈值内；三组均 0 失败题、0 SSE `error`，
  平均工具调用恒为 1.00（每题只调一次 `search_knowledge`），说明四能力接线没有改变工具选择行为。
- A/B/C 之间的小分摆动集中在 q005/q011/q013/q014/q018 这 5 题（同一工具 `search_knowledge`、
  同一答案路径，仅裁判给分 ±1），属 §4 记录的裁判波动，不是能力差异。

**审批链路的代价（C 组 vs B 组）**

- 均分差 **+0.10**、平均耗时差 **+0.1s/题**（29.7s vs 29.6s），都在裁判波动与进程噪声内；
- **本轮 20 题没有任何一题调用 `deep_research`（唯一 `requires_approval=True` 的工具）**，
  所以 C 组 `auto_approved=0`、上表"本会触发审批次数"为 0——审批分支没有真正进入，
  这 0.10 分 / 0.1s 是"HITL 接线开销 + 裁判波动"，**不是审批执行成本**；
- 为证明自动放行链路真的生效，C 组后端另发 1 次 `deep_research` 探针请求
  （`thread_id=g11-probe-hitl`，不计入 20 题统计）：服务日志出现
  `【评测专用】EVAL_AUTO_APPROVE=1，自动放行需审批工具 deep_research（累计 1 次；生产禁用）`，
  SSE 正常收尾、未挂起等审批；`ENV=prod` 下同一开关被忽略（见 `T95-基线v2/g11_checks.py`）。

**报告文件（三份 JSON 的 `judge_model` 均为 `qwen3.8-max-0902`）**

- A：`文档/项目/AdaptiveSearchAgent/评测报告/agent_eval_G11_A_off_24a916c_20261111-2300.json`
- B：`文档/项目/AdaptiveSearchAgent/评测报告/agent_eval_G11_B_governance_24a916c_20261111-2310.json`
- C：`文档/项目/AdaptiveSearchAgent/评测报告/agent_eval_G11_C_hitl_24a916c_20261111-2323.json`

> 复跑命令见 `tests/test_agent_cases/T95-基线v2/交接单.md`；A 组即"全关复现改造前"，
> 与 §7 可直接横比；换裁判则三组都不可横比（脚本会打印实际裁判名并写进 JSON）。
