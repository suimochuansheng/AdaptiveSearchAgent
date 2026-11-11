#!/usr/bin/env python
"""T9-3 冒烟：MCP 工具「热插拔 + 降级」一条龙演示。

跑法（仓库根目录，任意 cwd 均可）：

    poetry run python scripts/smoke_mcp.py

依次打印：

1. 注册前：注册表里的工具数量与名字（5 个内置工具，基线）；
2. 起 stdio MCP Server、加载工具、逐个 `registry.register()`、`reset_conversation_graph()`；
3. 注册后：注册表 / `to_lc_tools()` / **重建图后模型 bind_tools 看到的**工具列表（应为 7 个）；
4. 调用新工具 `word_count("hello world")`；
5. 关掉 Server 后再调一次：打印降级文案（不抛栈、会话不崩）。

退出码 0 = 冒烟通过；非 0 = 哪一步没达到预期（输出里有原因）。
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import 项目模块
    sys.path.insert(0, str(REPO))

BASELINE_NAMES = [
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
]
EXPECTED_NEW = {"word_count", "unit_convert"}


def _section(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def _print_tools(prefix: str, names: list[str]) -> None:
    print(f"{prefix}: {len(names)} 个 -> {', '.join(names) or '(空)'}")


def _graph_visible_names() -> list[str]:
    """用假模型接住 `bind_tools` 的实参：证明**重建图之后模型真能看到**新工具。

    只看注册表不算数（图是编译期绑定工具的），所以这里真的调一次
    `build_conversation_graph()`，把 `ChatOllama` 换成只记录名字的假模型。
    """
    from src import conversation_graph as cg

    captured: dict[str, list[str]] = {}

    class _CaptureModel:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def bind_tools(self, tools: list[Any]) -> _CaptureModel:
            captured["names"] = [getattr(tool, "__name__", "?") for tool in tools]
            return self

        async def ainvoke(self, _messages: Any) -> Any:  # pragma: no cover - 冒烟不真跑
            from langchain_core.messages import AIMessage

            return AIMessage("smoke")

    original = cg.ChatOllama
    setattr(cg, "ChatOllama", _CaptureModel)  # noqa: B010 - 冒烟里临时替换模型类
    try:
        cg.build_conversation_graph()
    finally:
        setattr(cg, "ChatOllama", original)  # noqa: B010
    return captured.get("names", [])


async def _run() -> int:  # noqa: C901 - 按演示顺序线性展开，拆开反而难对照输出
    from agent.adapters.tools import mcp_client, registry
    from config import settings
    from src.conversation_graph import current_tools

    failures: list[str] = []

    before = [tool.__name__ for tool in current_tools()]
    _section("1. 注册前（基线）")
    _print_tools("注册表 to_lc_tools()", before)
    if before != BASELINE_NAMES:
        failures.append(f"注册前工具不是 5 个内置工具：{before}")

    _section("2. 加载 MCP Server 并热注册")
    args = shlex.split(settings.MCP_SERVER_ARGS)
    print(f"命令: {settings.MCP_SERVER_CMD} {' '.join(args)}")
    try:
        items = await mcp_client.load_mcp_tools(
            settings.MCP_SERVER_CMD, args, timeout=settings.MCP_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001 - 冒烟要把失败打出来而不是抛栈
        print(f"加载失败：{exc}")
        return 1
    print(f"加载到 {len(items)} 个工具：{', '.join(spec.name for spec, _ in items)}")

    for tool_spec, fn in items:
        registry.register(tool_spec.name, fn, tool_spec)

    if any(spec.name in BASELINE_NAMES for spec, _ in items):
        failures.append("新工具与内置工具重名（不应该注册成功）")

    # 图是懒加载单例：不重置的话下一轮模型还是只看到旧工具
    import api_main

    api_main.reset_conversation_graph()
    after = [tool.__name__ for tool in current_tools()]

    _section("3. 注册后（热插拔生效）")
    _print_tools("注册表 to_lc_tools()", after)
    graph_names = _graph_visible_names()
    _print_tools("重建图后模型 bind_tools 收到", graph_names)

    if len(after) != len(BASELINE_NAMES) + len(items):
        failures.append(f"注册后数量不对：{len(after)} != {len(BASELINE_NAMES) + len(items)}")
    if not EXPECTED_NEW.issubset(set(graph_names)):
        failures.append(f"重建图后模型看不到新工具：{graph_names}")

    by_name = {tool.__name__: tool for tool in current_tools()}
    _section("4. 调用新工具 word_count")
    count_result = await by_name["word_count"]("hello world")
    print(f'word_count("hello world") -> {count_result}')
    if "字符数：11" not in count_result or "词数：2" not in count_result:
        failures.append(f"word_count 计数不对：{count_result}")
    print(f"unit_convert(1, km, m) -> {await by_name['unit_convert'](1, 'km', 'm')}")

    _section("5. 关掉 Server 再调一次（降级）")
    await mcp_client.close_mcp_session()
    print("MCP Server 子进程已关闭")
    degraded = await by_name["word_count"]("hello world")
    print(f'word_count("hello world") -> {degraded}')
    if "当前不可用" not in degraded:
        failures.append(f"降级文案不对：{degraded}")

    _section("结论")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS：5 → 7 热插拔，重建图后模型可见，Server 关闭后降级为可读文案。")
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
