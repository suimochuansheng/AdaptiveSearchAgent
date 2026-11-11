"""审批凭据存储（T9-2）：Redis 一次性凭据 + 墓碑 + `request_id` 幂等。

三把键（都在 Redis，`decode_responses=True`）：

| 键 | 值 | TTL | 用途 |
|---|---|---|---|
| `approval:<approval_id>` | JSON（八字段 + `thread_id`） | `APPROVAL_TTL_SECONDS`（默认 600s） | 一次性凭据本体 |
| `approval_gone:<approval_id>` | `issued` / `consumed` | `APPROVAL_TOMBSTONE_TTL_SECONDS`（默认 24h） | 区分 409（已消费）/ 410（过期）/ 404（从未存在） |
| `approval_req:<thread_id>:<request_id>` | `1` | 24h | resume 幂等（`SET NX`） |

**为什么墓碑有两个值**：

- 契约 `resume-http.md` §4：`409` = 重放 / 已消费；`410` = 过期。二者在 Redis 里都是
  “`approval:<id>` 不存在”，只靠一个“存在过”的墓碑**无法区分**。所以创建时就写
  `issued`，成功消费后改写成 `consumed`；查询时按值分流，严格满足契约与验收第 5/7 条。
- 墓碑 TTL 24h 覆盖契约的幂等窗口；10 分钟后凭据过期，墓碑仍在，所以 `410` 可判。

**fail-closed**：Redis 未配置 / 连不上 / 命令报错时，`create` 返回 `None`（执行器据此
拒绝执行），`consume` / `get` 抛 `ApprovalStoreUnavailable`（路由据此不执行）。
只 warning 一次，绝不降级成“放行”。这与 `budget_redis.py` 的“记账坏了不影响会话”
方向相反：预算可以少拦，审批不能少拦。

**原子性**：消费用 Lua 脚本“比对 thread_id -> DEL -> 把墓碑改成 consumed”一次完成，
二次消费 / 并发消费只有一个能成功（验收第 9 条）。
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from agent.domain.roles import ApprovalRequest
from config import settings

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

APPROVAL_KEY = "approval:{approval_id}"
TOMBSTONE_KEY = "approval_gone:{approval_id}"
REQUEST_KEY = "approval_req:{key}"

_ISSUED = "issued"
_CONSUMED = "consumed"

# 返回值约定：nil -> 键不存在；'mismatch' -> thread 不匹配（不删）；raw -> 成功消费
_CONSUME_LUA = """
local raw = redis.call('GET', KEYS[1])
if not raw then
  return nil
end
local ok, obj = pcall(cjson.decode, raw)
if not ok or obj['thread_id'] ~= ARGV[1] then
  return 'mismatch'
end
redis.call('DEL', KEYS[1])
redis.call('SET', KEYS[2], 'consumed', 'EX', tonumber(ARGV[2]))
return raw
"""


class ApprovalStoreUnavailable(RuntimeError):
    """审批存储不可用；调用方必须 fail-closed（不执行、不放行）。"""


@dataclass(frozen=True)
class StoredApproval:
    """Redis 里的一条待审批记录：冻结的八字段 + 归属会话（不入 ApprovalRequest）。"""

    thread_id: str
    request: ApprovalRequest


def _now() -> datetime:
    """带本地时区的当前时刻（`expires_at` 必须是带偏移的 ISO 8601）。"""
    return datetime.now().astimezone()


def _parse(raw: str) -> StoredApproval | None:
    """把 Redis 里的 JSON 解析回记录；任何字段缺失 / 类型错误都返回 None。"""
    try:
        data: dict[str, Any] = json.loads(raw)
        request = ApprovalRequest(
            approval_id=str(data["approval_id"]),
            tool_call_id=str(data.get("tool_call_id") or ""),
            tool=str(data["tool"]),
            args_digest=str(data.get("args_digest") or ""),
            cost_estimate=float(data["cost_estimate"]),
            budget_remaining=float(data["budget_remaining"]),
            permission=cast("Any", data["permission"]),
            expires_at=str(data["expires_at"]),
        )
        thread_id = str(data["thread_id"])
    except (KeyError, TypeError, ValueError):
        return None
    return StoredApproval(thread_id=thread_id, request=request)


class ApprovalStore:
    """Redis 版审批存储；所有方法都是异步的。"""

    def __init__(
        self,
        client: Redis | None = None,
        *,
        url: str | None = None,
        ttl_seconds: int | None = None,
        tombstone_ttl_seconds: int | None = None,
    ) -> None:
        self._client = client
        self._url = url if url is not None else settings.REDIS_URL
        # 显式传参时钉死；否则每次调用读 settings（测试可临时调 1 秒验证 410）
        self._ttl_override = ttl_seconds
        self._tombstone_ttl_override = tombstone_ttl_seconds
        self._degraded = False
        self._warned = False

    def _ttl(self) -> int:
        """凭据 TTL（秒）：未显式传参时跟随 `settings.APPROVAL_TTL_SECONDS`。"""
        return (
            self._ttl_override if self._ttl_override is not None else settings.APPROVAL_TTL_SECONDS
        )

    def _tombstone_ttl(self) -> int:
        """墓碑 TTL（秒）：未显式传参时跟随 settings。"""
        return (
            self._tombstone_ttl_override
            if self._tombstone_ttl_override is not None
            else settings.APPROVAL_TOMBSTONE_TTL_SECONDS
        )

    # ── 内部：连接与降级 ─────────────────────────────────────────────

    def _degrade(self, reason: str) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("审批存储不可用，所有审批 fail-closed（%s）", reason)
        self._degraded = True

    async def _redis(self) -> Redis:
        """惰性建连接；不可用直接抛 `ApprovalStoreUnavailable`。"""
        if self._degraded:
            raise ApprovalStoreUnavailable("审批存储已降级（fail-closed）")
        if self._client is not None:
            return self._client
        if not self._url:
            self._degrade("未配置 REDIS_URL")
            raise ApprovalStoreUnavailable("未配置 REDIS_URL")
        try:
            from redis import asyncio as aioredis

            self._client = aioredis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            return self._client
        except Exception as exc:  # noqa: BLE001 - 任何连接失败都 fail-closed
            self._degrade(f"连接失败：{exc}")
            raise ApprovalStoreUnavailable(str(exc)) from exc

    async def _run(self, coro: Any) -> Any:
        """执行一条 Redis 命令；异常统一转成 `ApprovalStoreUnavailable`。"""
        try:
            return await coro
        except ApprovalStoreUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - Redis 抖动 / 超时都算不可用
            self._degrade(f"命令失败：{exc}")
            raise ApprovalStoreUnavailable(str(exc)) from exc

    # ── 写：创建凭据 ─────────────────────────────────────────────────

    async def create(
        self,
        thread_id: str,
        tool: str,
        args_digest: str,
        cost_estimate: float,
        budget_remaining: float,
        permission: str,
        *,
        tool_call_id: str = "",
    ) -> ApprovalRequest | None:
        """签发一条审批凭据；Redis 不可用时返回 `None`（调用方必须拒绝执行）。

        `approval_id` 只由服务端生成：`secrets.token_hex(16)` -> 32 位小写 hex。
        """
        approval_id = secrets.token_hex(16)
        ttl = self._ttl()
        expires_at = (_now() + timedelta(seconds=ttl)).isoformat(timespec="seconds")
        request = ApprovalRequest(
            approval_id=approval_id,
            tool_call_id=tool_call_id,
            tool=tool,
            args_digest=args_digest,
            cost_estimate=float(cost_estimate),
            budget_remaining=float(budget_remaining),
            permission=cast("Any", permission),
            expires_at=expires_at,
        )
        payload = json.dumps({**asdict(request), "thread_id": thread_id}, ensure_ascii=False)
        try:
            client = await self._redis()
            pipe = client.pipeline(transaction=True)
            pipe.set(APPROVAL_KEY.format(approval_id=approval_id), payload, ex=ttl)
            pipe.set(
                TOMBSTONE_KEY.format(approval_id=approval_id),
                _ISSUED,
                ex=self._tombstone_ttl(),
            )
            await self._run(pipe.execute())
        except ApprovalStoreUnavailable:
            return None
        return request

    # ── 读：查询 / 消费 / 墓碑 ───────────────────────────────────────

    async def get(self, approval_id: str) -> StoredApproval | None:
        """读取凭据（不消费）；不存在返回 None，Redis 不可用抛异常。"""
        client = await self._redis()
        raw = await self._run(client.get(APPROVAL_KEY.format(approval_id=approval_id)))
        return _parse(raw) if isinstance(raw, str) else None

    async def consume(self, approval_id: str, thread_id: str) -> StoredApproval | None:
        """原子消费：thread_id 匹配才 DEL 并把墓碑改成 `consumed`。

        Returns:
            成功返回记录；键不存在 / thread 不匹配 / 已被并发消费都返回 None。
        """
        client = await self._redis()
        raw = await self._run(
            client.eval(
                _CONSUME_LUA,
                2,
                APPROVAL_KEY.format(approval_id=approval_id),
                TOMBSTONE_KEY.format(approval_id=approval_id),
                thread_id,
                str(self._tombstone_ttl()),
            )
        )
        if not isinstance(raw, str) or raw == "mismatch":
            return None
        return _parse(raw)

    async def tombstone_state(self, approval_id: str) -> str | None:
        """墓碑状态：`issued`（过期）/ `consumed`（已消费）/ None（从未存在）。"""
        client = await self._redis()
        raw = await self._run(client.get(TOMBSTONE_KEY.format(approval_id=approval_id)))
        return raw if isinstance(raw, str) else None

    # ── 幂等：request_id ─────────────────────────────────────────────

    async def remember_request(self, request_id: str, thread_id: str | None = None) -> bool:
        """`SET NX approval_req:<id>` TTL 24h；返回是否首次。

        Redis 不可用时**抛 `ApprovalStoreUnavailable`**（fail-closed），而不是伪装成
        "重复请求"：调用方（resume 路由）据此返回 409 并给出真实原因
        "审批凭据存储暂不可用，本次未执行，请稍后重试"（OI-T92-001）。
        """
        key = REQUEST_KEY.format(key=f"{thread_id}:{request_id}" if thread_id else request_id)
        client = await self._redis()
        created = await self._run(client.set(key, "1", nx=True, ex=24 * 60 * 60))
        return bool(created)


# ── 进程级单例 + 便捷函数（执行器 / 路由用） ─────────────────────────

_default_store: ApprovalStore | None = None


def get_default_store() -> ApprovalStore:
    """返回进程级审批存储（惰性创建，不碰 Redis）。"""
    global _default_store
    if _default_store is None:
        _default_store = ApprovalStore()
    return _default_store


def set_default_store(store: ApprovalStore | None) -> None:
    """替换默认存储（测试 / 组合根注入）；传 None 恢复惰性创建。"""
    global _default_store
    _default_store = store


async def create(*args: Any, **kwargs: Any) -> ApprovalRequest | None:
    """`ApprovalStore.create` 的模块级便捷入口。"""
    return await get_default_store().create(*args, **kwargs)


async def consume(approval_id: str, thread_id: str) -> StoredApproval | None:
    """`ApprovalStore.consume` 的模块级便捷入口。"""
    return await get_default_store().consume(approval_id, thread_id)


async def get(approval_id: str) -> StoredApproval | None:
    """`ApprovalStore.get` 的模块级便捷入口。"""
    return await get_default_store().get(approval_id)


async def tombstone_state(approval_id: str) -> str | None:
    """`ApprovalStore.tombstone_state` 的模块级便捷入口。"""
    return await get_default_store().tombstone_state(approval_id)


async def remember_request(request_id: str, thread_id: str | None = None) -> bool:
    """`ApprovalStore.remember_request` 的模块级便捷入口。"""
    return await get_default_store().remember_request(request_id, thread_id)


__all__ = [
    "APPROVAL_KEY",
    "ApprovalStore",
    "ApprovalStoreUnavailable",
    "REQUEST_KEY",
    "StoredApproval",
    "TOMBSTONE_KEY",
    "consume",
    "create",
    "get",
    "get_default_store",
    "remember_request",
    "set_default_store",
    "tombstone_state",
]
