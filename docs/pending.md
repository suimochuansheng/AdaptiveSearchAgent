# 待办与未决项（等后续组件完成后回来处理）

> 维护：本文件只记「现在还做不了、要等其他卡完成或需要人裁决」的事项。
> 已完成的历史事项不写在这里（看 git log 与各卡交接单）。
> 最后更新：2026-11-10

## 一览

| 编号 | 事项 | 依赖 / 触发条件 | 谁能做 |
|---|---|---|---|
| P1 | 合并 `feature/frontend-tdm` | 后端 `/chat/conversation`、`/resume` 联调通过 | 开发窗口 |
| P2 | 启用 `composition-root-only` 规则 | T2-2 组合根 `main.py` 出现 | 开发窗口 |
| P3 | 补 T11 的 0b 卡面与 OI 编号 | 计划窗口补发 0b 定义 | 计划窗口 |
| P4 | 裁决 OI-TDM-003（resume `token` 无来源） | 前后端契约决定 token 去留/来源 | 计划窗口 |
| P5 | 删除远端 `dev_setup` | 确认无人在用（本地已删） | 任何人 |

---

## P1 合并前端分支

- **现状**：`feature/frontend-tdm` 领先 `main` 5 个提交，已推远端；`tdm_checks.py` 21/21。
- **为什么现在不能合**：后端会话端点未跑通，前端只能在假 SSE 下自测；`main` 上的
  `tdm_checks.py` 也无法独立运行（依赖分支里的 `frontend/app.py`）。
- **完成后动作**：

  ```bash
  git switch feature/frontend-tdm
  git rebase main            # 把 5 个 WIP 提交叠到最新 main
  git rebase -i main         # squash 成 1~2 个正式提交（feat + test）
  git push -f origin feature/frontend-tdm
  # 走 PR 或直接 merge 回 main
  ```

- **验收**：合并后在 `main` 上跑 `python "tests/test_agent_cases/TDM-会话前端+审批卡片/tdm_checks.py"`，末行
  `RESULT: 21 passed, 0 failed`；`pytest -q` 仍 `223 passed / 4 skipped`。

## P2 composition-root-only 规则

- **现状**：`docs/architecture.md`「待启用规则」已写设计，但 `main.py` 尚不存在，import-linter
  无法配置；`adapters-only-ports` 目前有 1 条 `unmatched ignore` 警告也属同一原因。
- **完成后动作**：在 `pyproject.toml` 增加 `forbidden` contract（只有 `main.py` 可同时 import
  `agent.app` 与 `agent.adapters`）。
- **验收**：`poetry run lint-imports` 输出 4 条 contract 全 KEPT、无 warning。

## P3 T11 的 0b 卡面

- **现状**：`tests/test_agent_cases/T11-三层骨架+依赖规则/交接单.md` 第 6 节记录：计划区只登记了
  0a（`docs/baseline.md` 恢复 `IncompleteOutputException` 归因提示，OI-T02-002），**0b 的卡面与
  OI 编号缺失**，开发窗口未擅自新增。
- **需要你提供**：0b 到底是什么改动、对应哪个 OI 编号。
- **补完后动作**：在交接单第 6 节补一行，并视情况进 `台账用例.csv`（GB18030）。

## P4 OI-TDM-003：resume 的 `token` 契约无来源

- **现象**：`POST /chat/conversation/{thread_id}/resume` 的请求体要求 4 个字段
  （`approval_id` / `decision` / `request_id` / `token`），但 Chainlit 侧没有任何 token 来源，
  前端只能发 `token=""` 占位；`tdm_checks.py` 的 A6 也只断言「字段存在且是 str」。
- **两种收口方式**（择一，需计划窗口定）：
  1. 契约保留 `token`：由后端在 interrupt 数据里下发一次性 token，前端原样回传，
     A6 改为断言非空且与下发值一致；
  2. 契约删掉 `token`：前端不再发送，A6 断言字段不存在。
- **改动面**：`frontend/app.py` 的 resume 请求体、`tdm_checks.py` A6、台账 `TDM-016`
  （现为 `pass` + `open_issue=OI-TDM-003`，属「已知未决」，收口后清空）。

## P5 远端 dev_setup 分支

- **现状**：本地 `dev_setup` 与 `feature/conversation-agent` 已确认是 `main` 的祖先并删除；
  远端 `origin/dev_setup` 仍在。
- **完成后动作**（确认无人使用后）：

  ```bash
  git push origin --delete dev_setup
  ```
