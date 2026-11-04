"""上下文管理基础件测试（阶段 B1）——纯函数、不依赖 LLM。

重点验证面试官点名的问题：**从头硬截断会丢末尾结论**，现在的段落重要度压缩必须保住末段、
保住数字，并且结果真的落在预算内。
"""

from src.context_manager import (
    compress_tool_output,
    estimate_tokens,
    score_paragraph,
    select_paragraphs,
    split_paragraphs,
)


def _report(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs)


def _long_report() -> str:
    """30 段中段啰嗦 + 末段结论，模拟一篇"关键信息在最后"的调研报告。"""
    body = [f"第 {i} 段：检索到一段与问题关系不大的叙述，用于填充篇幅。" for i in range(1, 30)]
    return _report(
        ["开头：本次调研对比了两个型号。", *body, "结论：A 总成本 1200 万元，优于 B 的 1500 万元。"]
    )


def test_estimate_tokens_counts_cjk_by_char() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好") == 2
    assert estimate_tokens("abcd") == 1  # 4 个非中日韩字符 ≈ 1 token
    assert estimate_tokens("你好 abcd") == 2 + 2  # 空格+abcd 共 5 个非中日韩字符，向上取整


def test_split_paragraphs_ignores_blank_runs() -> None:
    assert split_paragraphs("a\n\n\n b \n\nc") == ["a", "b", "c"]


def test_paragraph_scoring_prefers_conclusions_and_numbers() -> None:
    plain = ["叙述" * 50, "叙述" * 50, "叙述" * 50]
    conclusion = ["叙述" * 50, "叙述" * 50, "结论：总成本 1200 万元"]

    assert score_paragraph(conclusion[-1], 2, 3) > score_paragraph(plain[-1], 2, 3)
    assert score_paragraph("## 标题", 1, 3) > score_paragraph("叙述" * 50, 1, 3)


def test_select_keeps_last_paragraph_and_numbers() -> None:
    """核心回归：结论在末段时，压缩后必须还在，且带数字。"""
    text = _long_report()

    compressed = select_paragraphs(text, token_budget=600)

    assert "结论：A 总成本 1200 万元，优于 B 的 1500 万元。" in compressed
    assert "1500" in compressed
    assert "省略" in compressed  # 中段确实被裁掉了
    assert estimate_tokens(compressed) <= 600 + 120  # 允许省略标记的少量开销


def test_select_preserves_original_order() -> None:
    text = _report([f"第 {i} 段：数字 {i * 10}% 的结论。" for i in range(1, 20)])

    compressed = select_paragraphs(text, token_budget=200)
    kept = [line for line in compressed.split("\n\n") if line.startswith("第 ")]
    indices = [int(line.split(" 段")[0].removeprefix("第 ")) for line in kept]

    assert indices == sorted(indices)


def test_short_text_is_returned_unchanged() -> None:
    text = "结论：1.77"

    assert compress_tool_output(text, token_budget=100) == text


def test_single_huge_paragraph_keeps_tail() -> None:
    """单段超预算：保留头尾——尾部结论不能被切掉。"""
    text = "开头说明" + "填充内容。" * 2000 + "尾部结论：1.77 是最终答案。"

    compressed = select_paragraphs(text, token_budget=200)

    assert "尾部结论" in compressed
    assert "中段省略" in compressed
    assert estimate_tokens(compressed) <= 200 + 80


def test_tiny_budget_still_returns_something() -> None:
    """预算极端小时不能返回空串（空 ToolMessage 会让模型以为工具没结果）。"""
    text = _long_report()

    compressed = select_paragraphs(text, token_budget=30)

    assert compressed
    assert len(compressed) <= 200
