# 修订：`reject` 分档语义（v1 → v2）

> 变更对象：`docs/contracts/approval.md`（§2 `Decision` 表 `reject` 行 + 新增 §2.1 + §5 第 7 条）
> 裁决：计划窗口 **D16**（2026-11-24）｜ 实施：开发窗口，同日
> 触发：前端 Bug 修复窗口（S3）移交单《审批「拒绝」应当终止本轮》
> （`tests/test_agent_cases/TDM-会话前端+审批卡片/移交主窗口-拒绝语义.md`）

---

## 1. v1 语义（2026-11-11 冻结）

`docs/contracts/approval.md` v1 §2 原文：

> `reject` — 消费 `approval_id`，不执行工具，把"已拒绝"作为工具结果**回灌模型**，**本轮继续**。

实现上就是 `src/conversation_graph.py` 的 `graph.add_edge("tools", "agent")`——**无条件**回到模型。
工具确实没执行、没记账（这点一直是对的），但"拒绝"只拦住了**这一次工具调用**，
没有拦住**本轮的产出**。

## 2. 实测问题

真实链路（前端 :8202 + 后端 :8201，2026-11-24）：

| 观测 | v1 实测值 |
|---|---|
| 点「拒绝」→ resume 首个 `token` | **25.3s** |
| 拒绝后的续写长度 | **3095 字**，一整篇「深度调研报告」 |
| 拒绝 → 本轮结束 | **48.4s** |
| 报告内容质量 | 含「1,200 QPS」「内部知识库」等**没有任何检索支撑**的数字（本轮零次工具执行） |
| trace | `error_kind=rejected_by_user`（工具未执行、未记账，这部分是对的） |
| 拒绝后 1s 发新消息 | `429`「该会话正在执行中」，且用户那句话被 Chainlit 输入框静默吞掉 |

三条产品口径都因此受损：

1. **绕过审批**：用户以为"我拦住了"，实际只拦住了工具；模型自己把答案编出来了，
   而且看起来"有数据"。这比直接报错更有害。
2. **成本语义被破坏**：拒绝的潜台词之一是"别花这笔钱"，但本轮仍继续烧 25~48s 的 token。
3. **交互不可判定**：用户无法区分"这是它真答的"和"这是它被拒绝后自己编的"。

## 3. 业界两派

判据不是"哪个框架怎么做的"，而是**批的是什么**。

**流派 A：拒绝 = 反馈给模型，本轮继续**

- [LangChain HITL 文档](https://docs.langchain.org.cn/oss/javascript/langchain/human-in-the-loop)：
  `reject` = 跳过该工具调用，并向智能体返回拒绝反馈。
- [OpenAI Agents SDK HITL](https://openai.github.io/openai-agents-js/guides/human-in-the-loop/)：
  interrupt 的 approve/reject 决议，reject 后 run 继续（还有 sticky `alwaysReject`）。
- 典型：Claude Code / Codex 的权限拒绝——工具返回 denied，模型换个安全做法。
- 适用：审批粒度是"这一步能不能做"，且模型有**安全的替代路径**。

**流派 B：拒绝 = 本轮/本次运行终止**

- [LangGraph HITL 概念](https://github.com/vksx/langgraph/blob/d0bf7837bd52f9f1b1887c33629a67b83fa2d44d/docs/docs/concepts/human_in_the_loop.md)：
  动作被拒绝时可以阻止图执行该步，走哪条边（含 `END`）由你决定。
- 生产系统对**不可逆 / 付费**动作（付款、部署、发邮件、调有成本的第三方）一律 fail-closed。
- [aegra HITL 示例](https://github.com/aegra/aegra/blob/2693066413dfafce3fd21bf581f10bd2495b8dd3/docs/guides/human-in-the-loop.mdx)
  有专节「Ignore the tool call → Skip the tool and end the turn」。

## 4. 为什么**分档**而不是全局 `reject = END`

两派各自都对，因为它们批的东西不同：

- `deep_research` 是 `permission="dangerous"`、有 `CostModel`、单次可跑 600s 的**能力门**。
  拒绝它 = "这一轮我不批这笔钱 / 这个长任务" → 属**流派 B**，必须终止，否则就是本节
  §2 那个"自造 3095 字报告"。
- `search_knowledge` / `search_tavily`（`external`）、`calculator` / `read_offload`（`read`）
  是**无副作用或低成本**的取数动作。拒绝它 = "这条路不通，换个办法" → 属**流派 A**，
  终止反而是过度反应：用户只想拦一次检索，不该把整轮回答也一起否掉。

所以 v2 的条文是**按风险档位分流**，判据只用 `ToolSpec.permission`：

| `permission` | `reject` 后 |
|---|---|
| `dangerous` | 终止本轮：tools 之后直接 `END`，不回模型推理；SSE 恰好一条 `final`（`已拒绝执行 {tool}，本轮未产生内容`），全程零 `token` |
| `external` / `read` / `write` | 维持 v1：回灌"已拒绝"，本轮继续正常出字 |

**不按工具名硬编码**（否则新增 `dangerous` 工具就静默漏掉分档），**不加新 `Decision` 枚举值**
（"拒绝并说明"是 v2.1 的事）。未注册工具按低风险处理，审批链路不得因注册表缺项而升级为终止。

## 5. 影响面

| 面 | 变化 |
|---|---|
| 契约 | `approval.md` 升 v2：§2 `reject` 行改写、新增 §2.1、§5 加第 7 条不变量；**字段与 `Decision` 枚举未改**，`t12_checks.py` 的机读锚点（`t12:fields` / `t12:enum` / `t12:history`）仍逐项匹配 |
| 图结构 | `src/conversation_graph.py`：`add_edge("tools","agent")` → `add_conditional_edges("tools", _route_after_tools)`；`ConversationState` 增 `last_tool_outcome` |
| SSE | `src/conversation_stream.py`：终止分支显式推一条 `final`；"interrupt 过不发 final"只对挂起态保留 |
| 状态机 | resume 路由收尾把 `task_states` 放回 `completed`（断连时 `idle`）并释放 Redis 锁；**`409`/`410`/`404`/`200` 语义一字未改** |
| 工具执行 | 不变：不执行、不记账、不扣预算；trace 仍 `error_kind=rejected_by_user` |
| 前端 | **零改动**（前端已单独处理输入框锁）。旧前端收到这一条 `final` 不渲染，行为向后兼容 |
| 低风险路径 | 不变（`external`/`read` 拒绝仍回灌继续），并由回归用例守住 |
| 已知未做 | **v2.1「拒绝并说明」**：审批卡片第三按钮 + `Decision` 新枚举 + 一句 feedback 才走回灌。需要计划窗口另开修订（v3） |

## 6. 迁移与兼容

- 旧 checkpoint 缺 `last_tool_outcome` 字段：代码一律 `.get` 读，缺字段按"无终止"处理，旧会话可续跑。
- 旧前端：拒绝后仍会收到 `resume` 的 200 流；现在流里只有一条 `final`（不再有 25~48s 的续写），
  前端既有的"后台消费、不渲染"逻辑继续有效，无需回滚。
- 回滚方式：把 `_route_after_tools` 换回 `add_edge("tools","agent")` 并把 §2.1 标记作废；
  字段可保留（只写不读），不需要 checkpoint 迁移。
