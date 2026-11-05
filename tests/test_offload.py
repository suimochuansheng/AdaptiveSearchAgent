"""工具输出 Offload 测试（阶段 B3）——落盘、回灌指针、按需回读、防路径穿越。

全部不依赖 LLM；卸载根目录用临时目录（monkeypatch），不污染仓库 data/。
"""

from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from src.context_manager import estimate_tokens
from src.tools import offload as off


@pytest.fixture(autouse=True)
def _temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """每个用例独立的卸载目录。"""
    monkeypatch.setattr(off, "OFFLOAD_ROOT", tmp_path)
    return tmp_path


def _paragraphs(count: int, *, chars: int = 200) -> str:
    return "\n\n".join(f"第 {i} 段：" + "正文" * (chars // 2) for i in range(count))


def _message(content: str, name: str = "deep_research") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="call-1", name=name)


async def test_save_and_read_roundtrip(_temp_root: Path) -> None:
    raw = _paragraphs(30)

    rel_path, path = off.save_offload("thread-1", "deep_research", raw)

    assert path.is_file()
    assert path.read_text(encoding="utf-8") == raw
    assert rel_path.startswith("thread-1/deep_research-")

    recalled = await off.read_offload(rel_path)
    assert "回读" in recalled
    assert "第 0 段" in recalled


async def test_read_by_keyword_returns_only_matching_paragraphs() -> None:
    raw = "\n\n".join(
        [
            "第 1 段：与问题无关的背景介绍。",
            "第 2 段：成本对比显示 A 方案 1200 万元，B 方案 1500 万元。",
            "第 3 段：另一段无关的收尾说明。",
        ]
    )
    rel_path, _ = off.save_offload("thread-k", "deep_research", raw)

    recalled = await off.read_offload(rel_path, keyword="成本")

    assert "成本对比" in recalled
    assert "背景介绍" not in recalled
    assert "收尾说明" not in recalled


async def test_pointer_must_stay_inside_offload_root() -> None:
    result = await off.read_offload("../../etc/passwd")

    assert "读取失败" in result
    assert "data/offload" in result


async def test_missing_file_gives_actionable_message() -> None:
    result = await off.read_offload("thread-x/not-exist.md")

    assert "读取失败" in result
    assert "重新执行" in result


def test_small_output_is_untouched() -> None:
    message = _message("结论：1.77")

    shrunk = off.shrink_tool_message(message, thread_id="t")

    assert shrunk.content == "结论：1.77"
    assert "[已卸载]" not in str(shrunk.content)


def test_raised_threshold_keeps_medium_output_in_prompt_only(
    _temp_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """把阈值调大 = 只要压缩、不落盘（"只对超大结果落盘"模式）。"""
    from config import settings

    monkeypatch.setattr(settings, "offload_token_threshold", 3000)
    raw = _paragraphs(10, chars=200)
    assert 1200 < estimate_tokens(raw) <= 3000

    shrunk = off.shrink_tool_message(_message(raw), thread_id="t")

    assert estimate_tokens(str(shrunk.content)) <= 1200 + 200
    assert "[已卸载]" not in str(shrunk.content)
    assert list(_temp_root.iterdir()) == []  # 没有落盘


async def test_default_threshold_offloads_any_lossy_compression(_temp_root: Path) -> None:
    """默认阈值 = prompt 预算：只要发生有损压缩，就把原件留在磁盘上。"""
    raw = _paragraphs(8, chars=220)  # 约 1760 tokens，刚过 1200
    assert 1200 < estimate_tokens(raw) < 3000

    shrunk = off.shrink_tool_message(_message(raw), thread_id="t")

    assert "[已卸载]" in str(shrunk.content)
    files = list(_temp_root.rglob("*.md"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == raw
    # 文件名用本地时间戳（不校验数值，只校验格式段存在）
    assert len(files[0].stem.split("-")) >= 3


async def test_large_output_is_offloaded_with_pointer(_temp_root: Path) -> None:
    raw = _paragraphs(40, chars=200)
    assert estimate_tokens(raw) > 3000

    shrunk = off.shrink_tool_message(_message(raw), thread_id="t")

    content = str(shrunk.content)
    assert "[已卸载]" in content
    assert "read_offload" in content
    assert estimate_tokens(content) <= 1200 + 400  # 压缩版 + 指针

    files = list(_temp_root.rglob("*.md"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == raw  # 原始内容完整保留

    # 从指针能回读到比 prompt 版本更多的细节
    rel_path = content.split("已存到 ")[1].split("。")[0]
    recalled = await off.read_offload(rel_path)
    assert estimate_tokens(recalled) > estimate_tokens(content)
    assert "直接作答" in recalled  # 回读后要提示模型收尾，别再重复调研


def test_offload_failure_degrades_to_compression(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _paragraphs(40, chars=200)
    monkeypatch.setattr(
        off, "save_offload", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )

    shrunk = off.shrink_tool_message(_message(raw), thread_id="t")

    assert "[已卸载]" not in str(shrunk.content)
    assert str(shrunk.content).strip()


def test_offload_disabled_still_compresses(monkeypatch: pytest.MonkeyPatch) -> None:
    from config import settings

    raw = _paragraphs(40, chars=200)
    monkeypatch.setattr(settings, "offload_enabled", False)

    shrunk = off.shrink_tool_message(_message(raw), thread_id="t")

    assert "[已卸载]" not in str(shrunk.content)
    assert estimate_tokens(str(shrunk.content)) <= 1200 + 200


def test_save_offload_sanitizes_ids(tmp_path: Path) -> None:
    rel_path, path = off.save_offload("../evil/thread", "deep research", "x")

    assert path.is_file()
    assert ".." not in rel_path
    assert " " not in str(path.name)
