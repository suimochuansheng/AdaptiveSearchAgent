"""端口（Port）契约：六边形架构的"里侧接口"。

本文件**只声明、不实现**：8 个 `Protocol` + 1 个纯数据 `RuntimeEvent`。
应用层（`agent/app`）只依赖这些抽象；谁提供实现由适配器决定，换实现只换适配器
（组合根负责装配），应用层一行不改。

依赖方向（硬约束，`poetry run lint-imports` 的 `core-no-framework` 会拦）：

    adapters  ──►  app.ports  ──►  domain
    （框架/IO）     （本文件）      （零依赖纯定义）

因此本文件**不得** import `langchain*` / `langgraph*` / 任何 IO 库，只允许标准库。
`RuntimeEvent` 同理：它是纯数据，不加方法、不加执行逻辑。

契约冻结依据：`docs/contracts/runtime-event.md`（`kind` 只有六值，`payload` 键名冻结）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence

    from agent.domain.budget import BudgetSnapshot

__all__ = [
    "AgentRuntime",
    "ArtifactStore",
    "BudgetLedger",
    "EventBus",
    "MemoryStore",
    "ModelClient",
    "RuntimeEvent",
    "SessionStore",
    "TaskStore",
]

RuntimeKind = Literal["token", "tool_call", "tool_result", "interrupt", "final", "error"]
"""`RuntimeEvent.kind` 的六值（见 `docs/contracts/runtime-event.md` §2）。

- `token` / `final` / `error`：文本增量、本轮收尾、可读错误；
- `tool_call` / `tool_result`：模型决定调用工具、工具返回结果；
- `interrupt`：外层 HITL 审批挂起（本轮不产生 `final`）。

SSE 的 `ping` **不是** RuntimeEvent：它只存在于传输层，由适配器在编码时补发。
"""


@dataclass(frozen=True)
class RuntimeEvent:
    """主循环向外暴露的一个语义事件（纯数据，无方法）。

    Attributes:
        kind: 事件类型，只能取 `RuntimeKind` 的六值之一。
        payload: 该 kind 的载荷，形状由 `docs/contracts/runtime-event.md` §3 冻结：
            `token`/`final` -> `{"content": str}`；`tool_call` -> `{"tools": list}`；
            `tool_result` -> `{"tool_call_id", "tool", "content"}`；
            `interrupt` -> `{"approval": dict}`；`error` -> `{"message": str}`。
    """

    kind: RuntimeKind
    payload: dict[str, Any]


@runtime_checkable
class AgentRuntime(Protocol):
    """Agent 运行时端口：承载"一轮对话"的主循环（现由 LangGraph 实现）。

    契约要点：

    - **幂等性**：`run_turn` 按 `session` 的 `thread_id` 续接同一会话，重复投递同一轮
      输入不保证去重（去重属于网关/会话锁的职责）；`resume` 必须消费一次性审批凭据，
      同一凭据重复恢复由审批存储拒绝（幂等边界在 store，不在这里）。
    - **错误语义**：流内错误以 `RuntimeEvent(kind="error", payload={"message": ...})`
      给出，**不抛栈给调用方**；实现自身不可恢复的故障才允许抛异常，且不得吞掉
      已经产出的 `final`/`interrupt`。`interrupt` 之后本轮结束，不补 `final`。
    - **并发安全**：同一 `thread_id` 的 `run_turn` / `resume` **不得并发**（由调用方
      的会话锁保证）；实现可以持有非线程安全的图/客户端缓存，但跨 `thread_id`
      的调用应互不影响。
    """

    def run_turn(
        self,
        session: Any,
        hooks: Any | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        """跑一轮对话，按语义事件流式产出。

        Args:
            session: 会话引用，至少含 `thread_id` 与本轮输入（`input`/`messages`）；
                具体形状由适配器约定，应用层只做透明传递。
            hooks: 运行时钩子（压缩 / 校验 / 预算等的挂载点）；适配器**只透传**，
                不解释。实现可忽略。

        Returns:
            异步事件流；`ping` 等传输层心跳不得出现在其中。
        """
        ...

    def resume(
        self,
        session: Any,
        command: Any,
    ) -> AsyncIterator[RuntimeEvent]:
        """从挂起态恢复一轮对话（HITL 审批后的续跑）。

        Args:
            session: 同 `run_turn`，`thread_id` 必须指向已挂起的会话。
            command: 恢复指令（审批结论等）；由适配器编码成框架的恢复原语。

        Returns:
            与 `run_turn` 同构的异步事件流；再次命中审批时以新的 `interrupt` 收尾。
        """
        ...


@runtime_checkable
class ModelClient(Protocol):
    """模型调用端口：流式 / 一次性 / 计数，三者都归这一个端口。

    `TokenCounter` **不单独建端口**：token 计数是模型能力的属性，拆出去会出现
    "同一模型两套口径"；需要精确计数时由实现内部调用模型自带 tokenizer。

    契约要点：

    - **幂等性**：只读调用，重复调用不改变任何持久状态；`count_tokens` 对同一输入
      必须稳定（允许模型切换导致跨版本差异，但同一实现内不得随机）。
    - **错误语义**：网络/限流等可重试错误由实现内部重试或抛领域异常；流式中途失败
      必须以可读错误终止迭代，**不得**产出半截后静默结束。`count_tokens` 不得抛
      "不支持"给调用方——估算也要给一个数。
    - **并发安全**：同一实例可被多个 `thread_id` 并发调用；实现若持有可变缓冲，
      必须按调用隔离（不得把两轮的 token 流混在一起）。
    """

    def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式对话：逐段产出文本增量（不含工具调用帧）。"""
        ...

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """一次性对话：返回完整文本。"""
        ...

    def count_tokens(self, text: str) -> int:
        """估算/精确计算一段文本的 token 数（不得抛"不支持"）。"""
        ...


@runtime_checkable
class SessionStore(Protocol):
    """会话持久化端口：会话元数据与消息历史的唯一落点。

    契约要点：

    - **幂等性**：`save` 按会话 id 覆盖写（upsert），重复保存同一快照结果一致；
      `mark_ended` 对已结束会话是 no-op（可重复调用）。
    - **错误语义**：存储不可用时抛 `StoreUnavailable` 类领域异常，**不得**返回空对象
      冒充"会话不存在"（空会话与存储故障必须可区分）；`load` 未命中返回 `None`。
    - **并发安全**：同一会话的读写在存储层串行（或乐观锁）；`mark_ended` 与 `save`
      并发时，"已结束"状态优先，不得被后续 save 复活。
    """

    async def load(self, session_id: str) -> Any | None:
        """读取会话快照；不存在返回 `None`。"""
        ...

    async def save(self, session: Any) -> None:
        """保存/覆盖会话快照。"""
        ...

    async def mark_ended(self, session_id: str) -> None:
        """把会话标记为已结束（幂等）。"""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """大结果落地端口：工具输出过大时先落盘，正文只回传指针。

    契约要点：

    - **幂等性**：`put` 对相同内容允许生成不同指针（内容寻址实现可相同），但同一指针
      的 `get` 结果必须稳定；`search` 只读。
    - **错误语义**：`get` 未命中/过期返回 `None`（调用方据此提示"回读失败"）；
      写入失败抛领域异常，**不得**返回假指针。`put` 返回的指针必须能直接喂给 `get`。
    - **并发安全**：并发 `put` 不得互相覆盖（指针唯一）；并发 `get` 与 `put` 同一指针
      时，读旧值或新值都可接受，但不得读到半截内容（写入原子）。
    """

    async def put(
        self,
        data: str,
        *,
        content_type: str = "text/plain",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """写入内容并返回可回读的指针。"""
        ...

    async def get(self, pointer: str) -> str | None:
        """按指针回读内容；不存在/过期返回 `None`。"""
        ...

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> Sequence[Any]:
        """按内容检索已落地的制品（返回条目列表，形状由实现约定）。"""
        ...


@runtime_checkable
class TaskStore(Protocol):
    """长任务端口：入队、查询、状态更新与取消。

    契约要点：

    - **幂等性**：`enqueue` 每次调用产生新任务；`cancel` 对已取消/已完成任务是 no-op；
      `update` 只做幂等赋值（同一变更重复应用结果一致）。
    - **错误语义**：`get` 未命中返回 `None`；`update` 的任务不存在时抛领域异常；
      `cancel` 不得把已完成任务改回取消（终态不可逆）。
    - **并发安全**：同一任务的状态迁移必须原子（CAS 或存储层事务）；worker 与
      `cancel` 并发时以先落地的终态为准，不得出现"取消后又被 worker 写回 running"。
    """

    async def enqueue(self, task: Any) -> str:
        """提交任务，返回任务 id。"""
        ...

    async def get(self, task_id: str) -> Any | None:
        """查询任务快照；不存在返回 `None`。"""
        ...

    async def update(self, task_id: str, **changes: Any) -> Any:
        """更新任务字段并返回最新快照。"""
        ...

    async def cancel(self, task_id: str) -> None:
        """请求取消任务（幂等；终态不可逆）。"""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """长时记忆与用户画像端口（画像**不单独建端口**）。

    `UserProfileStore` 不拆分：画像与记忆共享同一用户分区、同一 TTL/淘汰策略，
    拆成两个端口会让"写记忆、读画像"跨两个事务。故合并为 `get_profile` /
    `upsert_profile` 两个方法。

    契约要点：

    - **幂等性**：`append` 每次产生新记忆条目（不去重）；`upsert_profile` 按用户覆盖，
      同一画像重复写入结果一致；`recall` 只读。
    - **错误语义**：存储不可用抛领域异常，**不得**返回空列表冒充"没有记忆"；
      `get_profile` 未命中返回 `None`（新用户），不是错误。
    - **并发安全**：同一用户的 `upsert_profile` 以最后一次写入为准（last-write-wins）；
      `append` 与 `recall` 可并发，`recall` 允许看不到刚写入的最新条目（最终一致），
      但不得读到损坏条目。
    """

    async def append(self, user_id: str, item: Any) -> None:
        """追加一条长时记忆。"""
        ...

    async def recall(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 10,
    ) -> Sequence[Any]:
        """按语义/关键词召回记忆条目。"""
        ...

    async def get_profile(self, user_id: str) -> Any | None:
        """读取用户画像；新用户返回 `None`。"""
        ...

    async def upsert_profile(self, user_id: str, profile: Any) -> None:
        """写入/覆盖用户画像（last-write-wins）。"""
        ...


@runtime_checkable
class EventBus(Protocol):
    """事件总线端口：把 `RuntimeEvent` 扇出给观测 / 前端 / 审计等订阅方。

    契约要点：

    - **幂等性**：`publish` 至少一次投递；订阅方必须按事件自身 id/位点去重
      （总线不承诺 exactly-once）。`subscribe` 只建立订阅，不重放历史。
    - **错误语义**：单个订阅方失败**不得**阻断发布与其它订阅方（隔离故障）；
      `publish` 不得把订阅方的异常抛给主循环。
    - **并发安全**：多生产者并发 `publish` 时，单个订阅者看到的事件顺序必须与
      发布顺序一致（至少按会话内 FIFO）；`subscribe` 与 `publish` 可并发。
    """

    async def publish(self, event: RuntimeEvent) -> None:
        """发布一个运行时事件（订阅方失败被隔离）。"""
        ...

    def subscribe(self, *kinds: str) -> AsyncIterator[RuntimeEvent]:
        """订阅事件流；`kinds` 为空表示全部六值。"""
        ...


@runtime_checkable
class BudgetLedger(Protocol):
    """工具预算账本端口：策略只读 `snapshot`，执行器只写 `add`。

    原声明在 `agent/domain/budget.py`（OI-TAR-002）；`domain` 只放纯数据 / 纯规则，
    凡是"需要实现方"的接口一律进本文件，`domain` 里不再允许出现 `Protocol`。

    方法可以是**同步**（内存实现）或**异步**（Redis 实现），执行器用
    `inspect.isawaitable` 统一收口；返回值注解写成 `T | Awaitable[T]` 就是为了让
    两种实现都满足端口类型。

    契约要点：

    - **幂等性**：`add` **不幂等**（每次调用都累加，重复投递会多记），去重是调用方
      的职责；`reset` 幂等（清空不存在的作用域不报错）。
    - **错误语义**：存储不可用**不得**把异常抛给工具调用（记账失败只降级/告警，
      工具结果照常返回）；读 `snapshot` 失败时允许退化为全 0 快照——预算模块坏掉
      不能拖垮会话。
    - **并发安全**：同一 `scope` 的并发 `add` 允许 **±1 次调用**误差（读快照 -> 决策
      -> 记账整体不是原子的）；`reset` 与 `add` 并发时结果不保证，调用方需串行化。
    """

    def add(
        self,
        scope: str,
        *,
        calls: int = 0,
        cost: float = 0.0,
        tokens: int = 0,
    ) -> None | Awaitable[None]:
        """把一次调用的消耗累加进 `scope`。"""
        ...

    def snapshot(self, scope: str) -> BudgetSnapshot | Awaitable[BudgetSnapshot]:
        """读 `scope` 当前已消耗的预算。"""
        ...

    def reset(self, scope: str) -> None | Awaitable[None]:
        """清空 `scope` 的累计值（换轮 / 换会话时用）。"""
        ...
