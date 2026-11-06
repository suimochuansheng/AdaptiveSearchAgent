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


def stub_report(query: str) -> str:
    """生成一份确定性的假报告（同样的 query 永远得到同样的文本）。"""
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
