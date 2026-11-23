# 轻量评测门禁（evals/）

> 把"跑一次评测、手写一段话进 baseline.md"换成：结果落库 -> 配置矩阵 -> 自动对比 -> 门禁判定 -> 坏 case 回流。
> 界面只有 CLI + markdown 报告；不起服务、不加前端、不新增依赖。

## 1. 边界：CI 验证门禁逻辑，真实评测在本地/发版时跑

| 层 | 文件 | 是否调模型 | 在哪跑 |
|---|---|---|---|
| 真实评测 | `evals/run.py` | 是（后端 + Ollama / RAG 网关 + 裁判） | 本地 / 发版前 |
| 门禁判定 | `evals/gate.py` | 否（纯函数） | 本地、CI |
| 落库 | `evals/store.py` | 否 | 本地、CI |
| 报告 | `evals/report.py` | 否 | 本地、CI |
| 坏 case 回流 | `evals/backfill.py` | 否 | 本地、CI |
| 离线冒烟 | `scripts/smoke_eval_gate.py` | 否（fixture） | CI |

CI 没有模型凭据，**不跑 `evals/run.py`**；`.github/workflows/eval-gate.yml` 只跑离线冒烟，
证明"门禁逻辑 + 报告生成 + 边界错误"没坏。

## 2. 判据原文与来历

> 同口径两次均值差 <= 记录方差（默认 **0.0408**）；单题差值只做归因，不单独判失败。

- `0.0408` 来自 `docs/baseline.md` §4：Ragas `faithfulness` 同口径两次实测
  `0.8509` / `0.8101`，差 `0.0408`，取三项指标里最大的实测差作为记录方差；
- 门禁**不发明新阈值**，也**不改成"分数变低就失败"**：单次比较会被裁判噪声误判；
- 判据是**双侧**的：`|candidate_mean - baseline_mean| <= 方差`。提升幅度超过方差同样判 FAIL，
  表示"变化超出记录噪声、不可用旧基线解释"，不等于"变好了"；
- 少于 2 道共同有效题、方差 <= 0、per_case 缺字段都直接给可读错误（退出码 2），不猜。

## 3. 怎么跑

```bash
cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent
eval $(poetry env activate)

# 1) 真实评测：单配置 1 题冒烟（产物三件套）
python -m evals.run --configs A --limit 1

# 2) 配置矩阵：A 全关 / B 治理 / C +审批
python -m evals.run --configs A,B,C --set golden20

# 3) 跑完与同数据集上一次 run 对比并写进报告
python -m evals.run --configs A,B --compare last

# 4) 只看门禁（退出码 0=PASS / 1=FAIL / 2=输入不合法）
python -m evals.gate --baseline <run_id|path> --candidate <run_id|path>
python -m evals.gate --compare last

# 5) 离线冒烟（不调模型，CI 同款）
python scripts/smoke_eval_gate.py

# 6) 台账坏 case -> 回归集候选（只生成，不自动入集）
python -m evals.backfill \
  --ledger tests/test_agent_cases/台账用例.csv \
  --out evals/datasets/regression/candidates.json
```

产物：`evals/results/<run_id>/meta.json` + `per_case.json` + `report.md`（默认 gitignore）。
每个 config 一个独立后端子进程，启动时注入 `evals/configs.py` 的 env，跑完即停。

## 4. 配置矩阵

| id | 开关 | 用途 |
|---|---|---|
| A | `TOOL_POLICY=0 TRACE=0 HITL=0 SKILLS=0` | 复现改造前 raw，与 `baseline.md` §7 可横比 |
| B | `TOOL_POLICY=1 TRACE=1 HITL=0 SKILLS=0` | 治理（权限预算 + 追踪） |
| C | `TOOL_POLICY=1 TRACE=1 HITL=1 EVAL_AUTO_APPROVE=1` | 再加审批；自动放行仅评测用（与 §8 C 组同口径） |

## 5. 怎么加 case

- **黄金集**：`evals/datasets/qa_dataset_v3.yaml`（20 题，sha256 写进 meta）；
  加题后旧 run 仍可比较，门禁按**共同题**算均值；
- **回归集**：先跑 `evals.backfill` 生成 `evals/datasets/regression/candidates.json`，
  人工确认后把候选转成正式 YAML 题（带 `question` / `ground_truth`），
  再用 `python -m evals.run --dataset <yaml>` 跑；
- 新 run 只写 `evals/results/`，不改 `docs/baseline.md` 的既有数字。

## 6. 已知边界

- **单裁判**：所有分数来自 `EVAL_JUDGE_MODEL`，换裁判则不可横比（meta 记录 judge_model）；
- **20 题样本小**：单题 ±1 分就能让均值动 0.05，因此门禁只看均值差，单题只归因；
- **CI 不跑真模型**：门禁只能证明逻辑正确，不能证明本次线上质量；
- **0.0408 来自 Ragas（0~1 指标）**，迁到 1~10 分 judge 上非常紧；
  超出阈值先看裁判波动 / 环境，再谈退化；
- `latency_ms` / `tokens` 来自现有 SSE；`tokens` 在会话端点没有 kpi 帧时为 `null`，
  报告不据此判定；
- 门禁只比较**共同有效题**：某题候选失败（无分数）不计入均值，但会在失败清单里归因。
