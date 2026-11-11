# tests/test_agent_cases — 测试窗口用例目录

> 本目录是**测试窗口**的验收台账与检查脚本，不属于 pytest 测试套件。
> 目的：卡级验收可复跑、可追溯，同时**不污染** `docs/baseline.md §2` 冻结的 pytest 快照。

## 1. 编码（两套，别混）

| 文件 | 编码 | 说明 |
|---|---|---|
| `台账用例.csv` | **GB18030** | 读写都必须显式 `encoding="gb18030"`。单元格只允许**中文标点 + ASCII**，禁 emoji / 箭头（→）/ 希腊字母（Δ）/ 数学符号（≤）——它们不在 GB18030 里或会写成乱码。需要表达时写 `->`、`<=`、`delta`。 |
| `*.py` / `README.md` / 报告 JSON | **UTF-8** | 全部按 UTF-8 读写，禁止 `encoding` 省略成平台默认。 |

行尾：CSV 为 **CRLF**（沿用已有文件），用 `csv.writer(..., lineterminator="\r\n")` 追加。

## 2. 命名规则：为什么检查脚本不叫 `test_*.py`

`pytest --collect-only` 会把 `test_*.py` / `*_test.py` 全部收进来。一旦本目录的检查脚本被收集：

- `pytest --collect-only -q` 的收集数从 **227/231** 变成 `227+N`；
- `docs/baseline.md §2` 冻结的 **nodeids 哈希**（228 行）立刻失效，每张卡"开工前三件事"第 3 条就过不了；
- 验收脚本会带着真实评测目录、后端端口等**环境依赖**混进单测，CI 上必挂。

所以统一命名 **`<卡号小写>_checks.py`**（如 `t01_checks.py`、`t02_checks.py`），
用独立进程跑、自己打印 `[PASS]/[FAIL]` 与 `RESULT:` 末行、用退出码表达结论。
**永不**叫 `test_*.py`。

目录名格式：`<卡号>-<主题>/`，例如 `T01-环境搭建+ragas基线快照/`、`T02-评测口径+raw基线/`。

## 3. 运行方式

```bash
cd /home/ubhuazhu/dev_pros/AdaptiveSearchAgent

# 单卡检查（打印 [PASS]/[FAIL] 明细，末行 RESULT: n passed, m failed）
poetry run python "tests/test_agent_cases/T01-环境搭建+ragas基线快照/t01_checks.py"
poetry run python "tests/test_agent_cases/T02-评测口径+raw基线/t02_checks.py"

# T02 的离线模式：跳过唯一的那次 dry-run，其余按已归档证据判定
poetry run python "tests/test_agent_cases/T02-评测口径+raw基线/t02_checks.py" --no-dry-run

# 确认没被 pytest 收集（必须仍是 227/231）
poetry run pytest --collect-only -q | tail -2
```

退出码：`0` = 全部通过；`1` = 有失败项。脚本本身**只读**（T02 唯一写动作是它自己触发的 dry-run，
该命令内部 `mkdir` 后立即 `rmdir`，不留残物）。

## 4. case_id 区间表

| 前缀 | 归属 | 状态 |
|---|---|---|
| `T01-` | 环境搭建 + Ragas 基线快照 | 已用（001~018） |
| `T02-` | 评测口径修复 + Agent raw 基线 | 已用（001~010） |
| `T11-` | 预留（后续卡） | 未登记，占用前先在本表补一行 |
| `T12-` | 契约冻结（G3） | 已用（台账 001~010；脚本自检 001~015） |
| `T60-` | 预留 | 同上 |
| `T63-` | 预留 | 同上 |
| `T70-` | 预留 | 同上 |
| `T92-` | 预留 | 同上 |
| `T93-` | 预留 | 同上 |
| `TDM-` | 预留（数据/模型类） | 同上 |
| `TAR-` | 预留（归档/回归类） | 同上 |
| `T95-` | 预留 | 同上 |
| `INT-` | 跨卡集成用例（I1 架构基线、I2 治理闭环） | 已用（001~016） |
| `UI-` | 前端/交互用例 | 同上 |

规则：**区间单调、不回收、不复用**；新增前缀必须先在表中登记，再往 CSV 追加行。

## 5. 台账列约定（严格 15 列）

```
case_id,module,sub_feature,title,level,auto,priority,precondition,steps,expected,status,last_run,run_by,notes,open_issue
```

| 列 | 允许值 / 格式 |
|---|---|
| `level` | `functional` / `E2E` |
| `auto` | `auto` / `manual` |
| `priority` | `P0` / `P1` / `P2` |
| `status` | `pass` / `fail` / `block` |
| `last_run` | `YYYY/MM/DD`（如 `2026/11/10`） |
| `run_by` | 执行者标识（T01 为 `agent`，T02 起为 `test-G1`） |
| `notes` | 证据**路径**或命令输出末行，不写长文本 |
| `open_issue` | 一行一条，以 `OI-<卡号>-<序号>` 开头；无则留空 |

**只追加自己的行，不改别人的行**；追加后回读校验：总列数恒为 15、枚举值在允许集合内。
