"""上下文管理的确定性基础件（阶段 B1）。

阶段 B 的四层预算里，最先要"看得见"的是工具输出这一层：原实现是从头硬截断
（`report[:4000]`），而结论、数据往往在末尾，等于把最重要的信息扔掉。

这里的策略是**段落重要度打分**：标题段、含数字的段、含结论词的段、首段与末段加权，
中段冗长叙述降权；超预算时按分数挑选、但**保持原文顺序**，被丢弃处写明省略了多少段。
末段永远优先保留——即使预算只够一段，也先保末段。

本模块只做确定性计算、不调 LLM：
- 语义压缩（第二级，要点 + 关键数字）→ B2
- Offload（第三级，落盘 + 回灌指针）→ B3
"""

from __future__ import annotations

import re

# 中日韩字符：按 1 token/字估算（保守，宁可高估）
_CJK = re.compile(r"[\u4e00-\u9fff]")

# 数字 + 常见单位（金额/百分比/性能指标），这类段落通常承载关键结论
_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|元|万元|亿元|亿|万|GB|MB|ms|s|倍|个|条|W|℃)?")

# 结论性词汇：结论、数据、建议通常出现在这些词附近
_CONCLUSION_WORDS = (
    "结论",
    "综上",
    "因此",
    "所以",
    "建议",
    "风险",
    "成本",
    "价格",
    "预算",
    "胜出",
    "最终",
    "总计",
    "对比",
)

# 标题形态：Markdown 标题、中文序号、数字序号
_TITLE = re.compile(
    r"^\s*(?:#{1,6}\s|第[一二三四五六七八九十百]+[章节部分、.]|\d+[.、)]\s|[一二三四五六七八九十]+、)"
)

_OMIT_TEMPLATE = "…（省略 {n} 段，约 {chars} 字）"

# 压缩标记本身也要占预算，留一点余量给它，避免"压完还是超"
_MARKER_RESERVE = 24


def estimate_tokens(text: str) -> int:
    """粗估 token 数：中日韩字符按 1 token/字，其余按 4 字符 ≈ 1 token。

    宁高估不低估——预算超了模型会静默截断 prompt 头部（system 先丢），比压缩更糟。
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk + 3) // 4


def split_paragraphs(text: str) -> list[str]:
    """按空行切段（报告类文本的天然边界），保留非空段落。"""
    return [part for part in (p.strip() for p in re.split(r"\n\s*\n", text)) if part]


def score_paragraph(paragraph: str, index: int, total: int) -> float:
    """给一段打分：标题 / 数字 / 结论词 / 首末段加权，长段降权。"""
    score = 0.0
    if _TITLE.match(paragraph):
        score += 3.0
    if _NUMBER.search(paragraph):
        score += 3.0
    if any(word in paragraph for word in _CONCLUSION_WORDS):
        score += 2.0
    if index == 0:
        score += 2.0
    if index == total - 1:
        score += 3.0
    # 中段冗长叙述降权（最多扣 2 分，避免长段必被淘汰）
    score -= min(len(paragraph) / 800.0, 2.0)
    return score


def _trim_block(text: str, token_budget: int) -> str:
    """单段本身超预算：保留头尾、省略中段（结论常在尾部）。"""
    if estimate_tokens(text) <= token_budget:
        return text
    keep = max(token_budget, 32)
    head_len = keep * 3 // 5
    tail_len = keep - head_len
    if len(text) <= head_len + tail_len:
        return text
    return f"{text[:head_len]}…（中段省略）…{text[-tail_len:]}"


def _omit_marker(dropped: list[str]) -> str:
    """被丢弃段落的占位标记（让模型知道"这里被压过"，而不是以为原文就这么短）。"""
    return _OMIT_TEMPLATE.format(n=len(dropped), chars=sum(len(p) for p in dropped))


def _pick_blocks(paragraphs: list[str], scores: list[float], token_budget: int) -> dict[int, str]:
    """按分数挑选段落并裁剪到预算内；末段与首段优先占预算。"""
    total = len(paragraphs)
    must = [total - 1] + ([0] if total > 1 else [])
    order = must + [
        idx for idx in sorted(range(total), key=lambda i: (-scores[i], i)) if idx not in must
    ]

    kept: dict[int, str] = {}
    used = _MARKER_RESERVE
    for idx in order:
        remaining = token_budget - used
        if remaining <= 0:
            break
        block = _trim_block(paragraphs[idx], remaining)
        cost = estimate_tokens(block)
        if cost > remaining:
            continue
        kept[idx] = block
        used += cost
    return kept


def _render(paragraphs: list[str], kept: dict[int, str]) -> str:
    """按原顺序拼回保留段，被丢弃的连续段合成一条省略标记。"""
    parts: list[str] = []
    dropped: list[str] = []
    for i, para in enumerate(paragraphs):
        if i in kept:
            if dropped:
                parts.append(_omit_marker(dropped))
                dropped = []
            parts.append(kept[i])
        else:
            dropped.append(para)
    if dropped:
        parts.append(_omit_marker(dropped))
    return "\n\n".join(parts)


def select_paragraphs(text: str, token_budget: int) -> str:
    """按重要度挑选段落，保持原顺序，丢弃处标注省略。

    Args:
        text: 原始文本。
        token_budget: 内容预算（tokens），不含省略标记的少量开销。

    Returns:
        压缩后的文本；原文已在预算内则原样返回。
    """
    if token_budget <= 0:
        return ""
    if estimate_tokens(text) <= token_budget:
        return text

    paras = split_paragraphs(text)
    if not paras:
        return ""

    scores = [score_paragraph(p, i, len(paras)) for i, p in enumerate(paras)]
    kept = _pick_blocks(paras, scores, token_budget)
    if not kept:
        # 预算小到一段都放不下：至少保住末段的尾巴
        return _trim_block(paras[-1], token_budget)
    return _render(paras, kept)


def compress_tool_output(text: str, token_budget: int) -> str:
    """工具输出压缩（第一级：段落重要度）。超预算时给出保留结论的更短版本。

    第二级（语义压缩）与第三级（Offload 落盘）分别在 B2/B3 接入，接口保持不变。
    """
    if estimate_tokens(text) <= token_budget:
        return text
    return select_paragraphs(text, token_budget)
