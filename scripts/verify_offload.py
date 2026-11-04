#!/usr/bin/env python3
"""Offload 链路验收脚本（阶段 B3）。

用途：验收 3.1/3.2 时，真实深调研大多产出不到 3000 tokens 的旧阈值，导致落盘路径
从来没被跑到。本脚本用**真实代码、真实配置阈值**跑一遍完整链路，只把"工具输出"换成
一段确定性的长文本：

    ToolMessage(8000+ 字) → shrink_tool_message → 落盘 data/offload/... → read_offload 回读

它验证的是：阈值判定、压缩、指针回灌、文件落盘、指针解析与关键词回读。
**不 monkeypatch 任何配置**（阈值来自 config.py 的默认值）。

用法（在仓库根目录）：
    poetry run python scripts/verify_offload.py

退出码 0 = 全部通过；1 = 有失败项。
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

# 允许从仓库根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import ToolMessage  # noqa: E402

from config import settings  # noqa: E402
from src.context_manager import estimate_tokens  # noqa: E402
from src.tools.offload import OFFLOAD_ROOT, read_offload, shrink_tool_message  # noqa: E402


def _fake_report() -> str:
    """构造一份"结论在末尾"的长报告，长度确保超过落盘阈值。"""
    body = [
        f"第 {i} 段：这一段是对比调研的背景叙述，说明测试的必要性与数据来源。"
        f"检索结果 {i} 显示两款硬件在功耗与性能上的差异需要交叉验证。"
        for i in range(1, 61)
    ]
    conclusion = "结论：RTX 5090 功耗 575W、RTX 4090 功耗 450W；性能领先约 35%，能效比落后约 12%。"
    return "\n\n".join(body + [conclusion])


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


async def main() -> int:
    raw = _fake_report()
    print(f"原始输出: {len(raw)} 字 / 约 {estimate_tokens(raw)} tokens")
    print(
        f"配置阈值: tool_output_token_budget={settings.tool_output_token_budget}, "
        f"offload_token_threshold={settings.offload_token_threshold}"
    )

    message = ToolMessage(content=raw, tool_call_id="verify-1", name="deep_research")
    shrunk = shrink_tool_message(message, thread_id="verify-offload")
    content = str(shrunk.content)
    print(f"进 prompt 的版本: 约 {estimate_tokens(content)} tokens")

    ok = True
    ok &= _check("触发压缩（prompt 版本更短）", estimate_tokens(content) < estimate_tokens(raw))
    ok &= _check("回灌了卸载指针", "[已卸载]" in content and "read_offload" in content)
    ok &= _check(
        "prompt 版本守住预算",
        estimate_tokens(content) <= settings.tool_output_token_budget + 400,
        f"{estimate_tokens(content)} tokens",
    )

    files = sorted(OFFLOAD_ROOT.rglob("*.md")) if OFFLOAD_ROOT.exists() else []
    ok &= _check("已落盘", bool(files), str(files[-1]) if files else "未找到文件")
    if files:
        ok &= _check("磁盘上是未压缩原文", files[-1].read_text(encoding="utf-8") == raw)

    if "[已卸载]" in content:
        pointer = content.split("已存到 ")[1].split("。")[0]
        ok &= _check("指针是相对路径（不含 data/offload 前缀）", not pointer.startswith("data/"))

        recalled = await read_offload(pointer)
        ok &= _check(
            "回读比 prompt 版本信息更多",
            estimate_tokens(recalled) > estimate_tokens(content),
            f"{estimate_tokens(recalled)} vs {estimate_tokens(content)} tokens",
        )
        ok &= _check("回读保住了末尾结论", "575W" in recalled and "能效比" in recalled)

        by_keyword = await read_offload(pointer, keyword="能效比")
        ok &= _check("关键词回读命中", "能效比" in by_keyword)
        ok &= _check("关键词回读排除了无关段", "背景叙述" not in by_keyword)

    blocked = await read_offload("../../etc/passwd")
    ok &= _check("路径穿越被拦截", "读取失败" in blocked, blocked[:40])

    missing = await read_offload("verify-offload/not-exist.md")
    ok &= _check("缺失指针优雅降级", "读取失败" in missing and "重新执行" in missing)

    # 清掉本次验收产生的文件，避免污染 data/offload（不影响其他 thread 的记录）
    thread_dir = OFFLOAD_ROOT / "verify-offload"
    if thread_dir.exists():
        shutil.rmtree(thread_dir)
        print(f"已清理验收文件: {thread_dir}")

    print("\n结论：", "全部通过" if ok else "存在失败项，请回贴上面的输出")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
