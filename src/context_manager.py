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

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.utils.logger import get_logger

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


# ── 历史压缩（阶段 B2）───────────────────────────────────────────────────

logger = get_logger()

SUMMARY_PROMPT = """你是对话压缩器。从给定的对话记录中提取四类信息，输出中文要点：
1. 已经做出的决定（用户确认过的结论、选择、参数）
2. 尚未解决的问题（还没答完、待补充的信息）
3. 关键数字与结论（金额、数量、比较结果、日期）
4. 用户明确提出的要求或偏好

规则：
- 只输出这四节，用"决定 / 未解决 / 关键数字 / 用户要求"作小标题；
- 不要寒暄、不要复述过程、不要评论；
- 没有内容的节写"无"；
- 若给出"已有摘要"，把它与本轮新增对话合并成一份新摘要，不要丢已有事实。"""


@dataclass
class ContextBuild:
    """一次 prompt 组装的产物。

    Attributes:
        messages: 真正发给模型的列表（稳定区 + 旧轮摘要 + 最近原文）。
        summary: 更新后的摘要（没有新增归纳时保持原值）。
        summary_upto: 摘要已经覆盖到第几条消息。
        summarized: 本轮是否发生了新的归纳（决定要不要写回 state）。
    """

    messages: list[BaseMessage]
    summary: str
    summary_upto: int
    summarized: bool


def _content_text(message: BaseMessage) -> str:
    """把消息内容统一取成字符串（多模态 content 是 list）。"""
    content: Any = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def message_tokens(message: BaseMessage) -> int:
    """单条消息的 token 估算（正文 + 工具调用参数）。"""
    total = estimate_tokens(_content_text(message))
    for call in getattr(message, "tool_calls", None) or []:
        total += estimate_tokens(json.dumps(call, ensure_ascii=False))
    return total


def plan_window(messages: list[BaseMessage], token_budget: int) -> int:
    """从末尾往前累计，返回"从这里开始保留原文"的下标。

    至少保留最后一条消息，避免窗口算成空。
    """
    if not messages:
        return 0
    used = 0
    keep_from = len(messages) - 1
    for idx in range(len(messages) - 1, -1, -1):
        cost = message_tokens(messages[idx])
        if used + cost > token_budget and idx < len(messages) - 1:
            break
        used += cost
        keep_from = idx
    return keep_from


def fallback_summary(messages: list[BaseMessage], *, max_chars: int = 600) -> str:
    """规则兜底摘要：每条取开头一句 + 命中的数字，防止摘要调用失败时信息全丢。"""
    parts: list[str] = []
    for message in messages:
        text = " ".join(_content_text(message).split())
        if not text:
            continue
        numbers = _NUMBER.findall(text)[:5]
        piece = text[:80]
        if numbers:
            piece += f"（数字：{', '.join(numbers)}）"
        parts.append(piece)
    return "；".join(parts)[:max_chars]


async def summarize_history(
    llm: Any,
    older: list[BaseMessage],
    *,
    previous_summary: str = "",
    max_tokens: int = 800,
) -> str:
    """把旧轮次压成摘要；失败时退化为规则摘要，绝不抛异常打断会话。"""
    if not older:
        return previous_summary

    transcript = "\n".join(f"- {m.type}: {_content_text(m)[:400]}" for m in older)
    previous = f"【已有摘要】\n{previous_summary}\n\n" if previous_summary else ""
    prompt = [
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=f"{previous}【新增对话】\n{transcript}"),
    ]

    summary = ""
    try:
        response = await llm.ainvoke(prompt)
        summary = str(response.content).strip()
    except Exception:  # noqa: BLE001 - 摘要失败不能影响主回答
        logger.warning("历史摘要调用失败，退化为规则摘要", exc_info=True)

    if not summary:
        summary = fallback_summary(older)
        if previous_summary:
            summary = f"{previous_summary}\n{summary}"
    # 摘要本身也要守预算，否则旧轮压缩完仍然超窗
    return compress_tool_output(summary, max_tokens)


async def build_context(
    *,
    llm: Any,
    messages: list[BaseMessage],
    system_prompt: str,
    summary: str = "",
    summary_upto: int = 0,
    history_budget: int,
    summary_budget: int,
    enabled: bool = True,
) -> ContextBuild:
    """组装发给模型的 prompt：稳定区 + 旧轮摘要 + 最近原文。

    Args:
        llm: 用于生成摘要的模型（传 None 则用规则兜底，且不会丢消息）。
        messages: 全量对话历史（state 里的 messages）。
        system_prompt: 稳定区内容，永不压缩。
        summary: 上一轮的摘要。
        summary_upto: 摘要已覆盖到的消息下标。
        history_budget: 最近原文的 token 预算。
        summary_budget: 摘要的 token 预算。
        enabled: 关掉则短路成"system + 全量历史"，回退阶段 A 行为。

    Returns:
        ContextBuild：可直接发给模型的 messages + 需要写回的摘要状态。
    """
    if not enabled:
        return ContextBuild([SystemMessage(content=system_prompt), *messages], summary, 0, False)

    keep_from = max(plan_window(messages, history_budget), summary_upto)
    older = messages[summary_upto:keep_from]

    summarized = False
    if older:
        if llm is not None:
            summary = await summarize_history(
                llm, older, previous_summary=summary, max_tokens=summary_budget
            )
        else:
            tail = fallback_summary(older)
            summary = f"{summary}\n{tail}".strip() if summary else tail
        summary_upto = keep_from
        summarized = True

    prompt_messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    if summary:
        prompt_messages.append(
            SystemMessage(content=f"会话早期摘要（压缩自前 {summary_upto} 条消息）：\n{summary}")
        )
    prompt_messages.extend(messages[keep_from:])
    return ContextBuild(prompt_messages, summary, summary_upto, summarized)
