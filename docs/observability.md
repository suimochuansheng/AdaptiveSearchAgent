# 可观测性：指标与告警（B 卡）

指标由 `agent/observability/metrics.py` 用 `prometheus_client` 定义，FastAPI 通过
`GET /metrics` 暴露 Prometheus 文本格式（单进程默认 `REGISTRY`）。
`ENABLE_METRICS` 默认开启；关闭时埋点直接 return（零累加），`/metrics` 返回 404。

## 1. 指标清单

| 指标 | 类型 | 标签 | 回答什么问题 |
|---|---|---|---|
| `agent_requests_total` | Counter | `endpoint`, `outcome` | 会话端点成功/中断/失败各多少？错误率的分母是什么？ |
| `agent_request_duration_seconds` | Histogram | `endpoint` | 请求耗时分布与 p95 是多少？有没有长尾？ |
| `agent_tool_decisions_total` | Counter | `tool`, `decision` | 哪些工具在 allow/deny/需审批/被用户拒绝？策略是否误杀？ |
| `agent_approvals_total` | Counter | `decision`, `outcome` | 审批通过、拒绝、过期、重放、冲突各多少？ |
| `agent_errors_total` | Counter | `kind` | 错误集中在哪一类（transient / llm_recoverable / user_fixable / fatal）？ |
| `agent_tokens_total` | Counter | `direction` | 输入/输出 token 各消耗多少？ |
| `agent_cost_yuan_total` | Counter | 无 | 累计花了多少钱？（按 turn 增量累加） |

`endpoint` 目前只有 `conversation`；`outcome` 取 `completed / interrupted / failed / rejected`；
`decision`（工具）取 `allow / deny / require_approval / rejected_by_user`；
审批 `outcome` 取 `approved / rejected / expired / replayed / conflict`。

## 2. 标签纪律

**禁止**把 `thread_id` / `session_id` / `approval_id` / 用户输入放进标签：每个不同取值都会
生成一条新时间序列，是 Prometheus 基数爆炸的头号原因；会话级信息走 span（trace），不进 metrics。

## 3. 三条告警的阈值理由

| 告警 | 表达式 | 理由 |
|---|---|---|
| `AgentErrorRateHigh` | 错误率 > 5%，持续 10m | 20 题一轮里 1 题失败就是 5%，`for: 10m` 过滤偶发抖动；持续 10 分钟说明是模型/检索/网络的系统性问题。 |
| `AgentLatencyP95High` | p95 > 60s，持续 10m | 单题评测约 30s（RAG 精排单次约 11s），p95 超 60s 已是基线两倍，说明出现长尾（重试/大检索/工具串行等待）。 |
| `AgentCostHourlyHigh` | 1h 费用增量 > 20 元，持续 5m | 单题成本远低于 1 元，20 元/小时相当于连续 20 题以上全价请求，超出日常评测规模；持续 5 分钟排除一波尖峰，先查是否被刷或重试死循环。 |

规则文件：`monitoring/alerts.yml`（Prometheus rule_files 直接加载）。

## 4. 接入 Prometheus

```yaml
scrape_configs:
  - job_name: adaptive-search-agent
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

## 5. 已知边界

- **单进程限定**：默认 `REGISTRY` 在 `uvicorn --workers > 1` 下会多进程重复注册/聚合缺失；
  多 worker 需改用 `prometheus_client.multiprocess` 模式（`PROMETHEUS_MULTIPROC_DIR`）。
- **token / cost 依赖 turn span**：两项从 `agent/observability/turn.py` 的 turn 收尾取，
  因此只在 `ENABLE_TRACE=True` 时有值；trace 关闭时 `observe_usage` 不会被调用。
- **费用口径**：账本给的是会话累计费用，埋点按 session 记住上次值后只上报**增量**，
  避免多轮会话重复计数；游标只在进程内、上限 1024 个会话。
