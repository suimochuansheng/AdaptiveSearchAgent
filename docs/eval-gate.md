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

> 有效阈值 = max(记录方差，单题当量)；
> 单题当量 = score_step / n_cases；默认 score_step=1.0，n_cases 取 meta.dataset_size。
> `candidate - baseline` 落在 ±有效阈值内为 `PASS`；
> 低于 `-有效阈值` 为 `FAIL_DEGRADED`；高于 `+有效阈值` 为 `WARN_IMPROVED`。

- 记录方差 `0.0408` 来自 `docs/baseline.md` §4：Ragas `faithfulness` 同口径两次实测
  `0.8509` / `0.8101`，差 `0.0408`；
- **为什么阈值不能小于单题当量**：20 题时单题当量 = 1/20 = **0.05**，而 0.0408 < 0.05。
  若直接拿 0.0408 当阈值，“净降 1 题”（delta = -0.05）必然 FAIL，这是算术上的假阳性。
  所以有效阈值取 `max(0.0408, 0.05) = 0.05`；净降 1 题 PASS，净降 3 题（-0.15）仍 FAIL；
- 今天实测 A 同配置两次：均值差 0.0000，逐题 2 升 / 2 降 / 16 不变；
  B/C 相对 A 的 -0.05 正是“净 1 题”，在旧阈值下会被误判为 `FAIL_DEGRADED`；
- `score_step` 从 per_case 分数差推最小正步长，推不出用 1.0；`n_cases` 优先取
  `meta.dataset_size`（全量题数），缺省用共同题数；
- **逐题抖动只进报告，不进判定**：`flips={up,down,unchanged}` + 双侧符号检验
  `p=2*sum_{i<=min(up,down)} C(n,i)/2^n`；p 文本用于判断“变化是否像随机抖动”，
  不改变三态结论；
- 三态含义：
  - `PASS`：变化在有效阈值内；措辞随 delta 变化（见下）；
  - `WARN_IMPROVED`：提升超出有效阈值，**不直接放行**，需人工复核（可能是工具未触发 /
    裁判漂移 / 答案变短），CLI 退出码 0 但报告保留 WARN；
  - `FAIL_DEGRADED`：退化超出有效阈值，CLI 退出码 1；
- CLI：`PASS` / `WARN_IMPROVED` -> 0，`FAIL_DEGRADED` -> 1，输入不合法 -> 2；
- **有效性守卫**：任一侧 `valid_for_quality=false` 默认拒绝（退出码 2）；
  确需对比加 `--allow-invalid-baseline` / `--allow-invalid-candidate`，结果带 `quality_warning`；
- **题集守卫**：两侧题号不一致默认拒绝，错误列出“候选缺 N 题：q003”；确需按共同题比较加
  `--allow-partial`，结果带 `compared_cases` / `skipped`，报告写“结论不可跨版本比较”；
- **零差值**：`|delta| < 1e-9` 时 reason 写“无差异（两均值完全相同）”；
- **代码版本**：两 run 的 `git_commit` 不同时 `code_version_changed=true`；退出码不变，
  但报告顶部提示“分数差异不能归因于配置开关”，且 |delta| 超方差时 reason 改为
  “含代码版本差异，无法归因，需同版本复跑”；
- 少于 2 道共同有效题、方差为负、步长非正、per_case 缺字段都直接给可读错误（退出码 2），不猜。

### 2.1 措辞口径

- **归因标签**：`FAIL_DEGRADED` 写「失败归因：…」；`PASS` / `WARN_IMPROVED` 写
  「变化归因：…」——PASS 时逐题下降只是变化，不叫失败；报告第 3/5 节的表头同步切换；
- **delta 措辞**：
  - `delta == 0` -> “无差异（两均值完全相同）”；
  - `0 < |delta| <= 有效阈值` -> “差异在单题当量内（{delta_in_cases:.1f} 题），不显著”；
  - 超出阈值且为正 -> “提升超出阈值，需人工复核”；
  - 超出阈值且为负 -> “退化超出阈值”；
- **步长留痕**：reason 的当量处写 `单题当量 0.0500（score_step=1, n=20）`；
- **样本过小**：`n_cases < 5` 时 `sample_warning` 非空，reason 末尾追加
  “样本过小（n=N），单题当量 W，判据判别力弱”；状态仍按三态，不阻断，报告顶部提示；
- **数据集不一致**：两次 run 的 `dataset_sha256` 不同默认退出 2，
  提示“两次 run 数据集不同（a… vs b…），不可直接比较”；
  `--allow-mixed-dataset` 才继续，结果 `dataset_mismatch=true`，报告单独一行提示
  （与 `--allow-partial` 独立，可同时出现）。

## 3. 怎么跑

```bash
cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent
eval $(poetry env activate)

# 1) 真实评测：单配置 1 题冒烟（产物三件套）
python -m evals.run --configs A --limit 1

# 1b) 环境未就绪时只验链路（meta.valid_for_quality=false，报告顶部带警告）
python -m evals.run --configs A --limit 1 --allow-degraded

# 1c) --limit 小于题数 => meta.partial=true / publishable=false，报告提示“非全量”
#     对比时默认过滤 valid_for_quality=false 的历史 run；要纳入加 --allow-invalid
python -m evals.run --configs A --limit 1
python -m evals.run --configs A,B --compare last --allow-invalid

# 2) 配置矩阵：A 全关 / B 治理 / C +审批
python -m evals.run --configs A,B,C --set golden20

# 3) 跑完与同数据集上一次 run 对比并写进报告
python -m evals.run --configs A,B --compare last

# 4) 只看门禁（退出码 0=PASS/WARN / 1=FAIL_DEGRADED / 2=输入不合法）
python -m evals.gate --baseline <run_id|path> --candidate <run_id|path>
python -m evals.gate --compare last
# 自动找同 config/同 commit/同数据集的重复 run，用均值差绝对值当方差
python -m evals.gate --baseline <run> --candidate <run> --variance auto
# 明确放行无效 run / 题集不一致（结果会带 quality_warning / compared_cases / skipped）
python -m evals.gate --baseline <run> --candidate <run> \
  --allow-invalid-baseline --allow-invalid-candidate --allow-partial --allow-mixed-dataset

# 5) 离线冒烟（不调模型，CI 同款）
python scripts/smoke_eval_gate.py

# 6) 台账坏 case -> 回归集候选（只生成，不自动入集）
python -m evals.backfill \
  --ledger tests/test_agent_cases/台账用例.csv \
  --out evals/datasets/regression/candidates.json
```

产物：`evals/results/<run_id>/meta.json` + `per_case.json` + `report.md`（默认 gitignore）。
每个 config 一个独立后端子进程，启动时注入 `evals/configs.py` 的 env，跑完即停。

开跑前探活 RAG `/health`、模型一次极短调用、PG / Redis TCP 可达；任一失败默认拒绝开跑
（退出码 2，打印 `环境未就绪：rag` 这类可读原因）。`--allow-degraded` 可继续跑，
但 `meta.valid_for_quality=false`、`invalid_reason` 记录原因、报告顶部显示链路验证警告。

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
- **0.0408 来自 Ragas（0~1 指标）**；但 20 题时单题当量 0.05 更大，所以有效阈值是 0.05。
  低于单题当量的阈值会把“净 1 题”误判为退化，已被本补丁修正；
- **逐题抖动是证据不是判据**：2 升 2 降的符号检验 p=1.00，说明变化更像噪声；
  但报告仍会列出逐题下降项，供人工归因；
- `--variance 0` 合法：记录方差为 0 时有效阈值退化为单题当量；负数方差仍然拒绝；
- `score_step` 默认 1.0；整数 1~10 分 judge 推出来就是 1.0，换 0~1 指标时建议显式传步长；
- `latency_ms` / `tokens` 来自现有 SSE；`tokens` 在会话端点没有 kpi 帧时为 `null`，
  报告不据此判定；
- 门禁只比较**共同有效题**：某题候选失败（无分数）不计入均值，但会在变化/失败清单里归因；
- `n_cases < 5` 的 run 会有 `sample_warning`，但**不阻断**；样本过小的结论只能作链路参考；
- `dataset_sha256` 不同默认拒绝；`--allow-mixed-dataset` 只用于诊断，结论不可跨数据集比较；
- **非全量 run**（`--limit N < dataset_size`）分数不可用于发布或写简历：
  meta 标 `partial=true / publishable=false`，报告顶部有醒目提示；
- **代码版本不同**的两次 run 不做配置归因：报告顶部提示，`|delta|` 超方差时 reason
  明确写“无法归因，需同版本复跑”；
- `valid_for_quality=false` 的 run（如 2026-11-23 的 3 个链路验证 run）分数不可用于质量结论；
  历史 report.md 是补丁前生成的，可能没有顶部警告，以 meta 标注为准。
