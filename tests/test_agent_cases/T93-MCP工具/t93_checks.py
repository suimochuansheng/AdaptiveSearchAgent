"""G8/T9-3 MCP 工具适配器自测：8 条验收（开关关基线 / 热插拔 / 降级 / 失败不阻塞）。

运行（仓库根目录，任意 cwd 均可）：

    poetry run python "tests/test_agent_cases/T93-MCP工具/t93_checks.py"

覆盖（对应交接单第 4 节）：

    A1  开关关：5 个内置工具、顺序与 schema 哈希 = c48c9d48…2a99、TOOLS 与 to_lc_tools 同一对象
    A2  开关开 + Server 正常：注册表 5 → 7，to_lc_tools() 同步 7 个
    A3  **重建图之后**假模型 bind_tools 收到的工具名集合含 word_count / unit_convert
    A4  word_count("hello world") 计数正确（字符 11 / 词 2 / 行 1）
    A5  停掉 Server 再调：返回可读降级文案，不抛栈、不崩会话
    A6  Server 启动失败（命令改成 false）：一条 warning，不抛异常，5 个内置工具仍可用
    A7  关开关重启（新进程）：注册表回到 5 个，无 MCP 残留
    A8  新增工具不改旧工具签名：前 5 个 schema 哈希仍 = 基线

不依赖 Ollama / RAG / Postgres / Redis；MCP Server 是本地 stdio 子进程。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from agent.adapters.tools import mcp_client, registry  # noqa: E402
from config import settings  # noqa: E402
from src import conversation_graph as cg  # noqa: E402

BASELINE_NAMES = [
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
]
BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
NEW_TOOLS = {"word_count", "unit_convert"}

RESULTS: list[tuple[str, bool, str]] = []


def record(case_id: str, ok: bool, detail: str) -> None:
    RESULTS.append((case_id, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {case_id} {detail}")


def schema_hash(tools: list[Any]) -> str:
    """与 scripts/smoke_tools_schema.py 完全同一个哈希口径。"""
    schemas = [convert_to_openai_tool(tool) for tool in tools]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def drop_mcp_tools() -> None:
    """把动态注册的 MCP 工具撤回（模拟“关开关重启”）；只动本卡加进来的名字。"""
    import api_main

    for name in list(registry.TOOL_SPECS):
        if name not in BASELINE_NAMES:
            registry.TOOL_SPECS.pop(name, None)
            registry._TOOL_FNS.pop(name, None)  # noqa: SLF001 - 自测脚本，撤回动态注册用
    api_main.reset_conversation_graph()
    mcp_client.reset_unavailable()


def graph_visible_names() -> list[str]:
    """真的建一次图，用假模型接住 bind_tools 的实参（不是只看注册表）。"""
    captured: dict[str, list[str]] = {}

    class _CaptureModel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def bind_tools(self, tools: list[object]) -> _CaptureModel:
            captured["names"] = [getattr(tool, "__name__", "?") for tool in tools]
            return self

        async def ainvoke(self, _messages: object) -> object:  # pragma: no cover
            from langchain_core.messages import AIMessage

            return AIMessage("ok")

    original = cg.ChatOllama
    setattr(cg, "ChatOllama", _CaptureModel)  # noqa: B010 - 自测里临时替换模型类
    try:
        cg.build_conversation_graph()
    finally:
        setattr(cg, "ChatOllama", original)  # noqa: B010
    return captured.get("names", [])


def check_a1_baseline() -> None:
    """A1：开关关 → 基线工具集与 schema 指纹完全不变。"""
    names = [tool.__name__ for tool in cg.current_tools()]
    identity = all(a is b for a, b in zip(cg.TOOLS, cg.current_tools(), strict=False))
    digest = schema_hash(cg.current_tools())
    ok = names == BASELINE_NAMES and identity and digest == BASELINE_SHA256
    record(
        "A1",
        ok,
        f"names={names} identity={identity} sha256={digest[:12]}… ({digest == BASELINE_SHA256})",
    )


async def check_a2_registration() -> bool:
    """A2：Server 正常 → 5 → 7（注册表与 to_lc_tools 同步）。"""
    args = ["scripts/mcp_server.py"]
    try:
        items = await mcp_client.load_mcp_tools(
            "python", args, timeout=settings.MCP_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001 - 自测要报告原因而不是抛栈
        record("A2", False, f"加载失败：{exc}")
        return False

    for tool_spec, fn in items:
        registry.register(tool_spec.name, fn, tool_spec)

    count = len(registry.TOOL_SPECS)
    lc_count = len(registry.to_lc_tools())
    ok = count == 7 and lc_count == 7 and {spec.name for spec, _ in items} == NEW_TOOLS
    record(
        "A2",
        ok,
        f"注册表 {count} 个 / to_lc_tools {lc_count} 个 / 新增 {[s.name for s, _ in items]}",
    )
    return ok


def check_a3_graph_sees_tools() -> None:
    """A3：重建图之后，模型 bind_tools 收到的名字集合里有两个新工具。"""
    names = graph_visible_names()
    ok = NEW_TOOLS.issubset(set(names)) and len(names) == 7
    record("A3", ok, f"bind_tools 收到 {len(names)} 个：{names}")


def check_a8_old_schema_unchanged() -> None:
    """A8：新增工具不改旧工具签名——前 5 个的哈希仍是基线。"""
    digest = schema_hash(cg.current_tools()[:5])
    record("A8", digest == BASELINE_SHA256, f"前 5 个 sha256={digest[:12]}…")


async def check_a4_word_count() -> str | None:
    """A4：word_count("hello world") 返回正确计数。"""
    by_name = {tool.__name__: tool for tool in cg.current_tools()}
    result = await by_name["word_count"]("hello world")
    ok = result == "字符数：11，词数：2，行数：1"
    record("A4", ok, f'word_count("hello world") -> {result}')
    return result


async def check_a5_degraded() -> None:
    """A5：停掉 Server 再调 → 可读降级文案，不抛栈。"""
    by_name = {tool.__name__: tool for tool in cg.current_tools()}
    await mcp_client.close_mcp_session()
    try:
        result = await by_name["word_count"]("hello world")
    except Exception as exc:  # noqa: BLE001 - 抛异常就是这条验收失败
        record("A5", False, f"降级时抛了异常：{exc!r}")
        return
    ok = "当前不可用" in result and "word_count" in result
    record("A5", ok, f"降级文案：{result}")


class _CaptureHandler(logging.Handler):
    """收集项目 logger 的 warning，验证“启动失败只一条 warning”。"""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


async def check_a6_startup_failure() -> None:
    """A6：命令改成 false → 一条 warning、启动继续、5 个内置工具可用。"""
    import api_main

    drop_mcp_tools()
    handler = _CaptureHandler()
    logger = logging.getLogger("adaptive_agent")
    logger.addHandler(handler)

    old_enabled = settings.ENABLE_MCP
    old_cmd = settings.MCP_SERVER_CMD
    old_args = settings.MCP_SERVER_ARGS
    settings.ENABLE_MCP = True
    settings.MCP_SERVER_CMD = "false"
    settings.MCP_SERVER_ARGS = ""
    raised: Exception | None = None
    try:
        await api_main._register_mcp_tools()  # noqa: SLF001 - 启动钩子的验收
    except Exception as exc:  # noqa: BLE001
        raised = exc
    finally:
        settings.ENABLE_MCP = old_enabled
        settings.MCP_SERVER_CMD = old_cmd
        settings.MCP_SERVER_ARGS = old_args
        logger.removeHandler(handler)

    mcp_warnings = [m for m in handler.messages if "MCP 工具加载失败" in m]
    names = [tool.__name__ for tool in cg.current_tools()]
    ok = raised is None and len(mcp_warnings) == 1 and names == BASELINE_NAMES
    record(
        "A6",
        ok,
        f"raised={raised!r} warnings={len(mcp_warnings)} 内置工具={names}",
    )


def check_a7_restart_without_flag() -> None:
    """A7：关开关“重启”（新进程）→ 注册表回到 5 个，没有 MCP 残留。"""
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
        "from agent.adapters.tools import registry;"
        "print(len(registry.TOOL_SPECS));"
        "print(','.join(registry.TOOL_SPECS))"
    )
    env = {k: v for k, v in os.environ.items() if k != "ENABLE_MCP"}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )
    lines = proc.stdout.strip().splitlines()
    count = lines[0] if lines else "?"
    names = lines[1].split(",") if len(lines) > 1 else []
    ok = proc.returncode == 0 and count == "5" and names == BASELINE_NAMES
    record("A7", ok, f"新进程 count={count} names={names} rc={proc.returncode}")


def check_config_defaults() -> None:
    """配置项齐全且默认关闭（开关关 = 零子进程、零 mcp 导入）。"""
    fields = {
        "ENABLE_MCP": False,
        "MCP_SERVER_CMD": "python",
        "MCP_SERVER_ARGS": "scripts/mcp_server.py",
        "MCP_TIMEOUT_SECONDS": 10.0,
        "MCP_MAX_OUTPUT_TOKENS": 2000,
    }
    actual = {name: getattr(settings, name, "<missing>") for name in fields}
    ok = actual == fields
    record("A0", ok, f"settings={actual}")


async def main() -> int:
    print(f"仓库根：{REPO_ROOT}")
    check_config_defaults()
    check_a1_baseline()
    if not await check_a2_registration():
        print("\nA2 未通过，后续热插拔用例跳过。")
    else:
        check_a8_old_schema_unchanged()
        check_a3_graph_sees_tools()
        await check_a4_word_count()
        await check_a5_degraded()
        drop_mcp_tools()
    await check_a6_startup_failure()
    check_a7_restart_without_flag()
    drop_mcp_tools()

    failed = [case for case, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 70)
    print(f"合计 {len(RESULTS)} 条，通过 {len(RESULTS) - len(failed)} 条，失败 {len(failed)} 条")
    if failed:
        print("失败：" + ", ".join(failed))
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
