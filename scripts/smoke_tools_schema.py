#!/usr/bin/env python
"""T6-0 / T-arch 冒烟：打印工具 schema 的两个指纹（快照 vs 模型真正看到的）。

背景（OI-T93-003）：`src.conversation_graph.TOOLS` 是**模块导入时的快照**，MCP 工具
热注册后不会变；而 `current_tools()` 每次读注册表，才是 `bind_tools` / `ToolNode`
真正拿到的列表。只打印前者会把"模型看到的工具"测丢，所以本脚本同时打印两个指纹：

    snapshot_sha256：TOOLS（导入时快照，deprecated 兼容别名）
    current_sha256 ：current_tools()（模型当前真正看到的工具集）

两行都带名字列表与字节数；开关（ENABLE_MCP）关闭时两者**必须相等**，不相等直接
exit 1。开关打开时允许不等（热注册后快照本来就该落后），但会打印 `status: DIFF`
以及两侧名字差异，不静默。

哈希口径（与开工基线命令一致）：对
`json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()` 取 sha256，
其中 `schemas = [convert_to_openai_tool(tool) for tool in tools]`。

用法（任意 cwd 均可，建议仓库根目录）：

    poetry run python scripts/smoke_tools_schema.py          # 人读摘要
    poetry run python scripts/smoke_tools_schema.py --json   # 完整 schema JSON 到 stdout

基线（2026-11-11 冻结，T6-0；开关关闭时两个指纹都等于它）：

    names : search_knowledge, search_tavily, calculator, deep_research, read_offload
    sha256: c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99
    bytes : 2558
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import src
    sys.path.insert(0, str(REPO))

#: 开关关闭时，两个指纹都必须等于这个值（T6-0 冻结）。
BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"


@dataclass(frozen=True)
class Fingerprint:
    """一次 schema 指纹：哈希、名字列表、规范化字节数。"""

    sha256: str
    names: tuple[str, ...]
    bytes: int


def _fingerprint(tools: list[Any]) -> Fingerprint:
    """按开工基线口径计算 `tools` 的 schema 指纹。"""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schemas = [convert_to_openai_tool(tool) for tool in tools]
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    return Fingerprint(
        sha256=hashlib.sha256(canonical).hexdigest(),
        names=tuple(schema["function"]["name"] for schema in schemas),
        bytes=len(canonical),
    )


def _maybe_register_mcp_tools() -> None:
    """开关打开时，按组合根的同一路径热注册 MCP 工具（只为本进程指纹服务）。

    开关关闭时是 no-op，脚本行为与改造前一致。加载失败只 warning：指纹脚本不能
    因为外部 Server 起不来就红，快照本身仍然可打印。
    """
    from config import settings

    if not settings.ENABLE_MCP:
        return

    from agent.adapters.tools.mcp_client import close_mcp_session, load_mcp_tools
    from agent.adapters.tools.registry import register

    async def _load() -> int:
        items = await load_mcp_tools(
            settings.MCP_SERVER_CMD,
            shlex.split(settings.MCP_SERVER_ARGS),
            timeout=settings.MCP_TIMEOUT_SECONDS,
        )
        try:
            for spec, callable_ in items:
                register(spec.name, callable_, spec)
        finally:
            # 本脚本只取指纹，不调工具；注册完就释放子进程，避免孤儿进程
            await close_mcp_session()
        return len(items)

    try:
        count = asyncio.run(_load())
    except Exception as exc:  # noqa: BLE001 - 外部 Server 失败只降级，不中断指纹
        print(
            f"warn: ENABLE_MCP 打开但 MCP 工具加载失败（本次按内置工具计算）：{exc}",
            file=sys.stderr,
        )
        return
    print(f"info: ENABLE_MCP 打开，已热注册 {count} 个 MCP 工具（仅本进程指纹）", file=sys.stderr)


def _print_diff(snapshot: Fingerprint, current: Fingerprint) -> None:
    """开关打开且两指纹不等时，打印名字差异与明确标记（OI-T93-003）。"""
    snapshot_only = [name for name in snapshot.names if name not in current.names]
    current_only = [name for name in current.names if name not in snapshot.names]
    print("status: DIFF（ENABLE_MCP 打开时允许不等：快照是导入时旧列表，current 是模型可见）")
    print(f"  snapshot_only: {', '.join(snapshot_only) or '无'}")
    print(f"  current_only : {', '.join(current_only) or '无'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="打印工具 schema 的双指纹（快照 / 模型可见）")
    parser.add_argument(
        "--json",
        action="store_true",
        help="把 current_tools() 的完整 schema 以 JSON 打到 stdout（供人工比对，不参与哈希）",
    )
    args = parser.parse_args()

    # 先取快照：TOOLS 在 import 时已经求值；MCP 热注册不会改它
    from src.conversation_graph import TOOLS, current_tools

    snapshot = _fingerprint(list(TOOLS))

    # 兼容旧输出：--json 仍输出完整 schema 数组（现在是模型真正看到的那一份）
    if args.json:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        _maybe_register_mcp_tools()
        schemas = [convert_to_openai_tool(tool) for tool in current_tools()]
        print(json.dumps(schemas, sort_keys=True, ensure_ascii=False, indent=2))
        return 0

    _maybe_register_mcp_tools()
    current = _fingerprint(list(current_tools()))

    print(
        f"snapshot_sha256: {snapshot.sha256}  bytes={snapshot.bytes}  "
        f"names={', '.join(snapshot.names)}"
    )
    print(
        f"current_sha256 : {current.sha256}  bytes={current.bytes}  "
        f"names={', '.join(current.names)}"
    )

    if snapshot == current:
        print("status: MATCH（snapshot == current）")
        if snapshot.sha256 != BASELINE_SHA256:
            print(
                f"warn: 指纹与 T6-0 基线不一致：{snapshot.sha256} != {BASELINE_SHA256}",
                file=sys.stderr,
            )
        return 0

    _print_diff(snapshot, current)
    from config import settings

    if not settings.ENABLE_MCP:
        # 开关关着还漂移，说明注册表被旁路改了：这是回归，必须红
        print(
            "error: ENABLE_MCP 关闭时 snapshot 与 current 必须相等（注册表被旁路修改）",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
