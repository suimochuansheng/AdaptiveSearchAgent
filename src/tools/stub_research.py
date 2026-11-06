"""deep_research 的开发期假图（stub）。

用途：搭会话主循环（MessagesState + 工具调用）时，不依赖旧 workflow、不依赖
RAG 子服务、不花几分钟等真调研。切换方式：`.env` 里设

    DEEP_RESEARCH_BACKEND=stub

默认仍是 `workflow`（真跑旧图），所以生产/验收行为不变。

关键点：假图与真图**接口完全一致**——都是 `ainvoke(state, config)` 返回
`{"final_report": str}`。因为 `deep_research` 已经是一个普通工具，换后端只影响
工具内部，影响不到外层会话循环。等主循环稳定后，再回头动旧图。
"""

from __future__ import annotations

from typing import Any, Final

STUB_MARKER: Final = "[stub]"

# 假报告的段落结构：有标题、有数字、有结论，便于顺带验证段落压缩/Offload 链路
_STUB_TEMPLATE = """{marker} 深度调研（开发期假数据，未调用旧 workflow）

## 调研问题
{query}

## 对比维度
- 维度一：单位成本，样本 A 约 1200 元，样本 B 约 1500 元，差距 20%。
- 维度二：交付周期，样本 A 为 45 天，样本 B 为 60 天。
- 维度三：稳定性，样本 A 的故障率 1.2%，样本 B 为 0.8%。

## 结论
综合成本与周期，样本 A 更优；若以稳定性优先，则选样本 B。
{marker} 以上内容为占位数据，仅用于联调主循环，不代表真实调研结论。
"""


# 压测"回读"用的针眼：正文中段的一条无数字、无结论词的唯一串，
# 段落重要度压缩必然把它丢掉 → 模型只能靠 read_offload 才答得出
NEEDLE: Final = "NEEDLE-7f3a9c"


def _needle_report(query: str) -> str:
    """构造一份超长假报告，针眼藏在中段（正常压缩会被丢掉）。"""
    paragraphs = [f"{STUB_MARKER} 调研问题：{query}", "## 背景"]
    for i in range(1, 61):
        if i == 20:
            # 针眼段刻意写得又长又没营养：打分时被压到最低，压缩一定丢掉它
            paragraphs.append(
                f"第 {i} 段：这一段的唯一作用是把内部标识 {NEEDLE} 淹没在冗长叙述里，"
                + "补充说明。" * 40
            )
        elif i % 5 == 0:
            paragraphs.append(
                f"第 {i} 段：样本 A 的成本 {i * 100} 元，样本 B 的成本 {i * 120} 元。"
            )
        else:
            paragraphs.append(f"第 {i} 段：关于对比维度的补充说明，" + "用于填充篇幅。" * 6)
    paragraphs.append("## 结论")
    paragraphs.append("综合成本与周期，样本 A 更优；若以稳定性优先，则选样本 B。")
    return "\n\n".join(paragraphs)


def stub_report(query: str) -> str:
    """生成一份确定性的假报告（同样的 query 永远得到同样的文本）。

    查询里包含"长报告"时返回超长版本（约 2000+ tokens），用于压测压缩/Offload 链路。
    """
    if "长报告" in query:
        return _needle_report(query)
    return _STUB_TEMPLATE.format(marker=STUB_MARKER, query=query)


class StubResearchGraph:
    """假图：接口与编译后的 workflow 图一致（只需要 ainvoke）。"""

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        query = str(state.get("user_query", ""))
        return {"final_report": stub_report(query)}


def build_stub_graph() -> StubResearchGraph:
    """工厂函数：与 src.graph_factory.build_graph 的调用方式对齐。"""
    return StubResearchGraph()
