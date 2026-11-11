"""Redis 预算账本（T6-3）：`tool_budget:<scope>` hash，字段 `calls` / `cost` / `tokens`。

写入用 `HINCRBY` / `HINCRBYFLOAT`（多客户端原子），每次写入顺带 `EXPIRE` 24h，
避免长期不活跃会话的键无限堆积。

**并发误差（已知，允许）**：单条命令是原子的，但“读快照 -> `decide()` -> 记账”整体
不是；同一 `scope` 并发调用时可能都读到旧快照，于是多数放 1 次。v1 的预算是**软上限**，
多放一次比把会话卡住更可接受，因此不做 Lua 原子化，允许 **±1 次调用**误差。

**降级**：未配置 `REDIS_URL`、连不上、命令报错，都只 `warning` 一次，然后退化为
Noop 行为（记账丢弃、快照全 0）。预算模块坏掉不能拖垮会话——这与
`src/tasks_conversation.py` 的“队列不可用不影响回答”是同一原则。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from agent.adapters.tools.budget_memory import NoopLedger
from agent.app.ports import BudgetLedger
from agent.domain.budget import BudgetSnapshot
from config import settings

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_KEY_PREFIX = "tool_budget"
_TTL_SECONDS = 24 * 60 * 60

__all__ = ["NoopLedger", "RedisBudgetLedger"]


class RedisBudgetLedger(BudgetLedger):
    """Redis 版账本；所有方法都是异步的，失败自动降级为空操作。"""

    def __init__(
        self,
        client: Redis | None = None,
        *,
        url: str | None = None,
        ttl_seconds: int = _TTL_SECONDS,
    ) -> None:
        self._client = client
        self._url = url if url is not None else settings.REDIS_URL
        self._ttl_seconds = ttl_seconds
        # 不在构造期判死：延迟到首次使用时检查，这样 warning 只出现一次且带原因
        self._degraded = False
        self._warned = False

    # ── 内部：连接与降级 ─────────────────────────────────────────────

    def _degrade(self, reason: str) -> None:
        """只 warning 一次，然后永久退化为 Noop。"""
        if not self._warned:
            self._warned = True
            logger.warning("工具预算账本降级为 Noop（%s）", reason)
        self._degraded = True

    async def _redis(self) -> Redis | None:
        """惰性建连接并 ping 一次；不可用则降级并返回 None。"""
        if self._degraded:
            return None
        if self._client is None:
            if not self._url:
                self._degrade("未配置 REDIS_URL")
                return None
            try:
                from redis import asyncio as aioredis

                self._client = aioredis.from_url(
                    self._url,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
                # redis-py 的类型标注是 `Awaitable[T] | T`（兼容同步/异步客户端），
                # 这里用的就是异步客户端，cast 只为过 mypy。
                await cast("Awaitable[bool]", self._client.ping())
            except Exception as exc:  # noqa: BLE001 - 任何连接问题都只降级，不阻塞
                self._degrade(f"{type(exc).__name__}: {exc}")
                return None
        return self._client

    @staticmethod
    def _key(scope: str) -> str:
        return f"{_KEY_PREFIX}:{scope or 'unknown'}"

    # ── 端口实现 ─────────────────────────────────────────────────────

    async def add(
        self,
        scope: str,
        *,
        calls: int = 0,
        cost: float = 0.0,
        tokens: int = 0,
    ) -> None:
        client = await self._redis()
        if client is None:
            return
        key = self._key(scope)
        try:
            pipe = client.pipeline(transaction=False)
            pipe.hincrby(key, "calls", calls)
            pipe.hincrbyfloat(key, "cost", cost)
            pipe.hincrby(key, "tokens", tokens)
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()
        except Exception as exc:  # noqa: BLE001 - 记账失败不能影响工具结果
            self._degrade(f"{type(exc).__name__}: {exc}")

    async def snapshot(self, scope: str) -> BudgetSnapshot:
        client = await self._redis()
        if client is None:
            return BudgetSnapshot()
        try:
            data: dict[str, Any] = await cast(
                "Awaitable[dict[str, Any]]", client.hgetall(self._key(scope))
            )
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"{type(exc).__name__}: {exc}")
            return BudgetSnapshot()
        return BudgetSnapshot(
            calls=int(float(data.get("calls", 0) or 0)),
            cost=float(data.get("cost", 0.0) or 0.0),
            tokens=int(float(data.get("tokens", 0) or 0)),
        )

    async def reset(self, scope: str) -> None:
        client = await self._redis()
        if client is None:
            return
        try:
            await client.delete(self._key(scope))
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"{type(exc).__name__}: {exc}")
