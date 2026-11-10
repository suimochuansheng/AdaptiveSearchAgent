# 待办与未决项（等后续组件完成后回来处理）

> 维护：本文件只记「现在还做不了、要等其他卡完成或需要人裁决」的事项。
> 已完成的历史事项不写在这里（看 git log 与各卡交接单）。
> 最后更新：2026-11-11

## 一览

| 编号 | 事项 | 依赖 / 触发条件 | 状态 | 谁能做 |
|---|---|---|---|---|
| P2 | 启用 `composition-root-only` 规则 | T2-2 组合根 `main.py` 出现 | 未开始 | 开发窗口 |
| P4 | 落地 resume 三字段契约（删 `token`） | G3 前置小修 0a | 已裁决，待落地 | 开发窗口 |

---

## 已关闭

| 编号 | 关闭日期 | 结论 |
|---|---|---|
| P1 | 2026-11-11 | 已 squash 合并为 `67a0190`，`1990712` 修正 TDM-B2 比对；`tdm_checks.py` 21/21 |
| P3 | 2026-11-11 | "0b" 为计划窗口笔误；实为台账 `T02-005` 由 `pass` 改 `block`（`OI-T02-001`），执行者为测试窗口，并入 G3-0b |
| P5 | 2026-11-11 | 远端 `dev_setup` 已删除 |

## P2 composition-root-only 规则

- **现状**：`docs/architecture.md`「待启用规则」已写设计，但 `main.py` 尚不存在，import-linter
  无法配置；`adapters-only-ports` 目前有 1 条 `unmatched ignore` 警告也属同一原因。
- **完成后动作**：在 `pyproject.toml` 增加 `forbidden` contract（只有 `main.py` 可同时 import
  `agent.app` 与 `agent.adapters`）。
- **验收**：`poetry run lint-imports` 输出 4 条 contract 全 KEPT、无 warning。

## P4 resume 三字段契约（已裁决，待落地）

- **裁决（2026-11-11，计划窗口）**：**删除 `token` 字段**。理由：token 与 `approval_id` 由同一通道下发，
  第二凭证不带来实质安全增益；一次性 `approval_id` + `request_id` 幂等 + 会话锁已覆盖重放；
  保留它只会导致前端发空串占位、测试只能断言"字段存在"。
- **新契约**（以 `文档/docs/windows/车道包-G9.md` 第 3 节为准）：

  ```text
  POST /chat/conversation/{thread_id}/resume
  body {"approval_id":"<32 hex>","decision":"approve"|"reject","request_id":"<uuid4>"}
  200 继续 SSE / 409 重放或 request_id 重复 / 410 过期 / 404 未知 thread
  interrupt payload 不再包含 token
  ```
