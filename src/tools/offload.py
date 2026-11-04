"""工具输出的 Offload（阶段 B3）——文件系统当"第二大脑"。

设计意图（对照蓝图第 3 步第三级降级）：
- 单条工具输出太大时，**原始内容落盘**，prompt 里只留"段落重要度压缩后的版本 + 指针"；
- 模型需要细节时调用 `read_offload(pointer, keyword)` 按需回读，而不是把几万字塞进窗口；
- 指针是仓库内相对路径（`data/offload/<thread>/<tool>-<hash>.md`），读取时强制校验
  必须落在 OFFLOAD_ROOT 之内，防止路径穿越。

为什么不在工具内部压缩：压缩属于"进 prompt 之前"的编排，放在 tools 节点统一做，
所有工具（含 deep_research）行为一致，也便于单点回滚。
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from config import settings
from src.context_manager import compress_tool_output, estimate_tokens
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from langchain_core.messages import ToolMessage

logger = get_logger()

# 卸载文件根目录（data/ 已在 .gitignore 中，不会进版本库）
OFFLOAD_ROOT = Path("data/offload")

# 指针里展示的摘要长度
_POINTER_PREVIEW_CHARS = 120


def _offload_root() -> Path:
    """返回当前卸载根目录（测试里会被 monkeypatch 成临时目录）。"""
    return OFFLOAD_ROOT


def _safe_name(text: str) -> str:
    """工具名/线程 id 里的危险字符替换掉，避免拼出非法文件名。"""
    # 不允许 "."，避免拼出 ".." 这类会破坏路径语义的名字
    return re.sub(r"[^A-Za-z0-9_-]", "_", text)[:64] or "unknown"


def save_offload(thread_id: str, tool_name: str, content: str) -> tuple[str, Path]:
    """把原始工具输出写到磁盘。

    Args:
        thread_id: 会话 id，用于分目录，避免不同会话互相覆盖。
        tool_name: 工具名（用于文件名前缀，便于排查）。
        content: 原始输出。

    Returns:
        (相对路径, 绝对路径)。相对路径就是要回灌给模型的指针。
    """
    safe_thread = _safe_name(thread_id)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    directory = _offload_root() / safe_thread
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(tool_name)}-{stamp}-{digest}.md"
    path.write_text(content, encoding="utf-8")
    # 回灌给模型的是相对指针（可移植、且读取时会再校验根目录）
    rel_path = path.resolve().relative_to(_offload_root().resolve()).as_posix()
    return rel_path, path


def pointer_text(rel_path: str, content: str) -> str:
    """生成回灌给模型的指针说明（含一句话预览，便于判断要不要回读）。"""
    preview = " ".join(content.split())[:_POINTER_PREVIEW_CHARS]
    tokens = estimate_tokens(content)
    return (
        f"[已卸载] 完整输出约 {tokens} tokens，已存到 {rel_path}。"
        f"预览：{preview}…\n"
        f'需要细节时调用 read_offload(pointer="{rel_path}", keyword="…") 按需回读。'
    )


def shrink_tool_message(message: ToolMessage, *, thread_id: str) -> ToolMessage:
    """工具消息进 prompt 前的统一收口：超预算压缩，超大则落盘 + 回灌指针。

    Args:
        message: ToolNode 产出的原始工具消息。
        thread_id: 会话 id（卸载目录用）。

    Returns:
        可能被替换 content 的新消息；未超预算时原样返回。
    """
    text = str(message.content)
    tokens = estimate_tokens(text)
    if tokens <= settings.tool_output_token_budget:
        return message

    body = compress_tool_output(text, settings.tool_output_token_budget)
    if settings.offload_enabled and tokens > settings.offload_token_threshold:
        try:
            rel_path, _ = save_offload(thread_id, message.name or "tool", text)
            body = f"{body}\n\n{pointer_text(rel_path, text)}"
        except OSError:
            # 落盘失败不能让会话挂掉：退化成"只压缩"，并留下日志
            logger.warning("工具输出落盘失败，退化为纯压缩", exc_info=True)

    return message.model_copy(update={"content": body})


def _resolve_pointer(pointer: str) -> Path:
    """把指针解析成 OFFLOAD_ROOT 内的绝对路径，越界直接拒绝。"""
    root = _offload_root().resolve()
    candidate = (root / pointer).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("指针必须指向 data/offload 目录内部")
    if not candidate.is_file():
        raise FileNotFoundError(pointer)
    return candidate


def _select_by_keyword(text: str, keyword: str, token_budget: int) -> str:
    """只回读包含关键词的段落；一个都没命中时退回整篇压缩。"""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    hits = [p for p in paragraphs if keyword in p]
    if not hits:
        return compress_tool_output(text, token_budget)
    selected = "\n\n".join(hits)
    return compress_tool_output(selected, token_budget)


async def read_offload(pointer: str, keyword: str = "") -> str:
    """读取之前被卸载的完整工具输出（可按关键词只取相关段落）。

    Args:
        pointer: 卸载时回灌的指针，形如 data/offload/<thread>/<tool>-<hash>.md。
        keyword: 只返回包含该关键词的段落；留空则返回压缩后的全文。

    Returns:
        文本结果；出错时返回可行动的说明（不抛异常，避免打断 Agent 循环）。
    """
    try:
        path = _resolve_pointer(pointer)
    except ValueError as exc:
        return f"读取失败：{exc}。请使用工具输出里给出的 pointer 原样传入。"
    except FileNotFoundError:
        return f"读取失败：找不到 {pointer}。可能已过期清理，请重新执行原工具。"

    text = path.read_text(encoding="utf-8")
    budget = settings.offload_read_token_budget
    body = (
        _select_by_keyword(text, keyword, budget) if keyword else compress_tool_output(text, budget)
    )
    return f"（{pointer} 回读，原文约 {estimate_tokens(text)} tokens）\n\n{body}"
