"""MCP 工具适配器（T9-3）：stdio 子进程 → `ToolSpec` + 异步可调用体。

职责边界（只做 tools，不做 resources / prompts）：

- `load_mcp_tools(cmd, args, timeout)`：起子进程 → `initialize` → `tools/list`，
  为每个远端工具构造 `(ToolSpec, 异步可调用体)`；
- 可调用体内部做 `tools/call`，**失败 / 超时只返回可读错误字符串，不抛栈**；
- `mark_unavailable(name)`：Server 断开后把该工具标成不可用，后续调用直接返回降级文案；
- `close_mcp_session()`：应用关闭阶段释放子进程与会话。

契约取值（见 `docs/contracts/tool-spec.md`）：`permission="external"`、
`cost_model=CostModel("call", 0.0)`、`requires_approval=False`（审批是危险工具的语义，
MCP 工具一律不审批）、`feature_flag="ENABLE_MCP"`。

**热插拔**：本模块只负责“加载 + 调用”，登记与重建图由组合根（`api_main`）做：
`load_mcp_tools()` → 逐个 `registry.register()` → `reset_conversation_graph()`。

协议实现用官方 `mcp` SDK（本仓库已装 1.30.0）：`mcp.client.stdio.stdio_client` 起子进程，
`mcp.ClientSession` 说话术。会话常驻（不按次重启子进程），关闭时按**进入顺序倒序**逐个释放：
`ClientSession` 在退出时会 cancel 自己的 task group 并向上抛 `CancelledError`，所以每个上下文
单独 try/except——否则前一个的取消会让后面的子进程清理被跳过。
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.domain.tool import CostModel, ToolSpec
from config import settings
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = get_logger()

#: 仓库根：`agent/adapters/tools/mcp_client.py` → parents[3]；子进程 cwd 固定在这里，
#: 这样 `MCP_SERVER_ARGS=scripts/mcp_server.py` 这类相对路径在任意 cwd 下都能起来。
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Server 断开 / 调用失败后的统一降级文案（模型和安全网都靠它判断“别重试”）
UNAVAILABLE_TEMPLATE = (
    "该 MCP 工具当前不可用（{name}）：MCP Server 未连接或已断开。可以改用内置工具，或稍后重试。"
)

#: JSON Schema 的 `type` → Python 注解（只用于把远端 schema 还原成函数签名）
_PY_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

# 会话状态：模块级单例（一个进程连一个 MCP Server）
_SESSION: Any | None = None
#: 已进入的异步上下文，按进入顺序存（关闭时倒序）：
#: [stdio_client(...), ClientSession(...)]。不用 AsyncExitStack 是因为它的 aclose()
#: 遇到第一个异常就停，而 ClientSession 退出必然抛 CancelledError，会跳过子进程清理。
_OPEN: list[Any] = []
_TIMEOUT: float = 10.0
_MAX_OUTPUT_TOKENS: int = 2000
_UNAVAILABLE: set[str] = set()


# ── 对外：不可用标记 ──────────────────────────────────────────────────


def mark_unavailable(name: str) -> None:
    """把某个 MCP 工具标记为不可用：后续调用直接返回降级文案，不再打 Server。

    Args:
        name: 工具名（重复标记幂等，只在第一次 warning）。
    """
    if name not in _UNAVAILABLE:
        logger.warning("MCP 工具标记为不可用：%s", name)
    _UNAVAILABLE.add(name)


def is_unavailable(name: str) -> bool:
    """该工具是否已被标记为不可用（排障 / 冒烟用）。"""
    return name in _UNAVAILABLE


def _unavailable_text(name: str) -> str:
    return UNAVAILABLE_TEMPLATE.format(name=name)


# ── 对外：加载 / 调用 / 关闭 ──────────────────────────────────────────


async def load_mcp_tools(
    cmd: str,
    args: Sequence[str] | None = None,
    timeout: float | None = None,
) -> list[tuple[ToolSpec, Callable[..., Any]]]:
    """起 stdio MCP Server 并取回它的工具列表。

    Args:
        cmd: Server 命令；传 `"python"` 时换成当前解释器（保证子进程能用本仓库 venv 里的 mcp）。
        args: 命令参数（相对路径按仓库根解析）。
        timeout: 握手 / `tools/list` / 后续单次 `tools/call` 的超时秒数；
            缺省取 `settings.MCP_TIMEOUT_SECONDS`。

    Returns:
        `[(ToolSpec, 异步可调用体), ...]`，顺序与 Server 的 `tools/list` 一致。

    Raises:
        Exception: 起不来 / 握手失败 / `tools/list` 失败（调用方决定是否降级）。
            失败时不会留下半开的子进程。
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    global _MAX_OUTPUT_TOKENS, _OPEN, _SESSION, _TIMEOUT

    timeout = float(timeout if timeout is not None else settings.MCP_TIMEOUT_SECONDS)
    max_output_tokens = int(settings.MCP_MAX_OUTPUT_TOKENS)

    # 允许重复加载：先把上一次的连接收掉，避免子进程泄漏
    await close_mcp_session()
    _UNAVAILABLE.clear()

    params = StdioServerParameters(
        command=_resolve_cmd(cmd),
        args=[str(a) for a in (args or ())],
        cwd=str(REPO_ROOT),
    )

    opened: list[Any] = []
    try:
        stdio_cm = stdio_client(params)
        read_stream, write_stream = await stdio_cm.__aenter__()
        opened.append(stdio_cm)

        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        opened.append(session_cm)

        await asyncio.wait_for(session.initialize(), timeout=timeout)
        listed = await asyncio.wait_for(session.list_tools(), timeout=timeout)
    except asyncio.CancelledError as exc:
        # 子进程起不来时，官方 SDK 的 task group 取消会以 CancelledError 冒到调用方；
        # 这里转成普通异常，避免“启动失败”被当成任务取消而穿透 lifespan。
        await _close_opened(opened, quiet=True)
        raise RuntimeError(
            "MCP Server 握手未完成：子进程未启动或提前退出（检查命令 / 参数 / 依赖）"
        ) from exc
    except BaseException:
        await _close_opened(opened, quiet=True)
        raise

    _OPEN = opened
    _SESSION = session
    _TIMEOUT = timeout
    _MAX_OUTPUT_TOKENS = max_output_tokens

    tools: list[tuple[ToolSpec, Callable[..., Any]]] = []
    for tool in listed.tools:
        spec = _build_spec(tool, timeout, max_output_tokens)
        tools.append(
            (
                spec,
                _build_callable(tool.name, tool.description or "", tool.inputSchema or {}, timeout),
            )
        )

    logger.info(
        "MCP 工具加载完成",
        extra={"count": len(tools), "names": [spec.name for spec, _ in tools]},
    )
    return tools


async def call_mcp_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> str:
    """调用一个 MCP 工具，返回文本结果或**可读错误字符串**（不抛栈）。

    Args:
        name: 工具名。
        arguments: 工具参数（JSON 对象）。
        timeout: 单次调用超时，缺省用加载时的值。

    Returns:
        工具输出的纯文本；失败时是给人/给模型看的中文错误，不会抛异常。
    """
    if name in _UNAVAILABLE:
        return _unavailable_text(name)

    session = _SESSION
    if session is None:
        mark_unavailable(name)
        return _unavailable_text(name)

    wait_seconds = float(timeout if timeout is not None else _TIMEOUT)
    try:
        result = await asyncio.wait_for(
            session.call_tool(name, arguments or {}), timeout=wait_seconds
        )
    except TimeoutError:
        mark_unavailable(name)
        return f"MCP 工具调用超时（{name}，>{wait_seconds:g}s）。{_unavailable_text(name)}"
    except Exception as exc:  # noqa: BLE001 - 工具失败只能回灌给模型，不能让会话崩
        mark_unavailable(name)
        logger.warning("MCP 工具调用失败：%s：%s", name, exc)
        return f"MCP 工具调用失败（{name}）：{exc}。{_unavailable_text(name)}"

    return _render_result(name, result)


async def close_mcp_session() -> None:
    """关闭 MCP 会话与子进程（幂等；应用关闭阶段 / 重新加载前调用）。"""
    global _OPEN, _SESSION

    opened, _OPEN = _OPEN, []
    _SESSION = None
    await _close_opened(opened)


def reset_unavailable() -> None:
    """清掉“不可用”标记（重新连接后调用；冒烟 / 测试用）。"""
    _UNAVAILABLE.clear()


# ── 内部：子进程命令、spec、可调用体 ─────────────────────────────────


def _resolve_cmd(cmd: str) -> str:
    """`python` 换成当前解释器：子进程必须能用本仓库 venv 里的 `mcp`。"""
    return sys.executable if cmd.strip() in {"python", "python3"} else cmd


async def _close_opened(opened: list[Any], *, quiet: bool = False) -> None:
    """倒序释放已进入的异步上下文；每个单独 try/except（见模块 docstring 的解释）。

    Args:
        opened: 按进入顺序排的异步上下文。
        quiet: 握手已经失败时用——清理阶段的报错不再刷第二条 warning。
    """
    for cm in reversed(opened):
        try:
            await cm.__aexit__(None, None, None)
        except BaseException as exc:  # noqa: BLE001 - 关闭阶段的异常一律吞掉
            if not quiet:
                logger.warning("关闭 MCP 会话组件时出错（已忽略）：%s", exc)


def _build_spec(tool: Any, timeout: float, max_output_tokens: int) -> ToolSpec:
    """把 MCP 的 tool 描述转成本仓库的 `ToolSpec`。"""
    return ToolSpec(
        name=tool.name,
        description=tool.description or f"MCP 工具 {tool.name}",
        input_schema=dict(tool.inputSchema or {"type": "object", "properties": {}}),
        permission="external",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=timeout,
        max_output_tokens=max_output_tokens,
        requires_approval=False,
        feature_flag="ENABLE_MCP",
    )


def _build_callable(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    timeout: float,
) -> Callable[..., Any]:
    """用远端 schema 生成一个“看起来就像本地工具”的异步函数。

    `bind_tools` / `convert_to_openai_tool` 读的是 `__signature__` + `__annotations__`，
    所以这里按 JSON Schema 还原签名与类型注解：注册进来的 MCP 工具和内置工具
    在模型眼里完全同构（只是名字不同）。
    """

    signature, annotations = _signature_from_schema(input_schema)

    async def _mcp_tool(*args: Any, **kwargs: Any) -> str:
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError as exc:
            return f"MCP 工具参数不合法（{name}）：{exc}"
        return await call_mcp_tool(name, dict(bound.arguments), timeout)

    _mcp_tool.__name__ = name
    _mcp_tool.__qualname__ = name
    _mcp_tool.__doc__ = description
    _mcp_tool.__signature__ = signature  # type: ignore[attr-defined]
    _mcp_tool.__annotations__ = annotations
    return _mcp_tool


def _signature_from_schema(
    input_schema: dict[str, Any],
) -> tuple[inspect.Signature, dict[str, Any]]:
    """JSON Schema → (`inspect.Signature`, 注解字典)；必填参数排在可选参数前面。"""
    properties: dict[str, Any] = input_schema.get("properties") or {}
    required = set(input_schema.get("required") or ())

    params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for prop_name, prop in properties.items():
        prop = prop or {}
        annotation = _PY_TYPES.get(str(prop.get("type")), Any)
        default = inspect.Parameter.empty if prop_name in required else prop.get("default", None)
        params.append(
            inspect.Parameter(
                prop_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )
        annotations[prop_name] = annotation

    # Python 语法要求：无默认值的参数必须在前，否则 inspect.Signature 直接报错
    params.sort(key=lambda p: p.default is not inspect.Parameter.empty)
    return inspect.Signature(params), annotations


def _render_result(name: str, result: Any) -> str:
    """把 MCP 的 CallToolResult 压成纯文本（自定义内容类型退化成 str）。"""
    texts: list[str] = []
    for item in getattr(result, "content", None) or ():
        text = getattr(item, "text", None)
        texts.append(str(text) if text is not None else str(item))

    if getattr(result, "isError", False):
        detail = "\n".join(texts) or "Server 未返回错误详情"
        return f"MCP 工具执行出错（{name}）：{detail}"

    if texts:
        return "\n".join(texts)
    structured = getattr(result, "structuredContent", None)
    if structured:
        return str(structured)
    return f"MCP 工具（{name}）执行完成，但没有返回文本内容。"
