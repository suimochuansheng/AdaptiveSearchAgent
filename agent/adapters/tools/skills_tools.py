"""Skills 工具（T6-4）：`list_skills` 列出索引，`load_skill` 按需读正文。

渐进披露的落地：模型先看索引（name + when_to_use），需要时再读正文。工具本身
只做“读磁盘上的技能包”，不触发任何远程加载；技能正文是否进 prompt 由模型决定。

开关 `ENABLE_SKILLS=True` 时由组合根（`api_main`）登记；关闭时注册表仍是 5 个
内置工具，schema 哈希不变。
"""

from __future__ import annotations

from typing import Any

from agent.adapters.skills import loader
from agent.domain.tool import CostModel, ToolSpec
from config import settings

LIST_SKILLS_TOOL_NAME = "list_skills"
LOAD_SKILL_TOOL_NAME = "load_skill"


def list_skills() -> str:
    """列出当前可用的技能包（每行：name — when_to_use，最多 SKILLS_INDEX_MAX 条）。

    当你不确定该用哪条能力规范时先调用它；拿到 name 后再调用 load_skill 读取正文。
    索引只占少量 token，技能正文不会自动进入上下文。
    """
    specs = loader.scan()
    if not specs:
        return "（暂无可用技能）"

    limit = max(0, int(settings.SKILLS_INDEX_MAX))
    shown = specs[:limit]
    return "\n".join(f"{item.name} — {' '.join(item.when_to_use.split())}" for item in shown)


def load_skill(name: str) -> str:
    """读取指定技能的正文（不含 frontmatter）；未知或不合法名字返回可读错误。

    Args:
        name: 技能名，必须来自 list_skills 的返回结果（只允许 [a-z0-9-]）。
    """
    return loader.load(name)


def register_skills_tools() -> bool:
    """把两个 Skills 工具登记进注册表；开关关闭或已登记时返回 False。"""
    if not settings.ENABLE_SKILLS:
        return False

    from agent.adapters.tools import registry

    if LIST_SKILLS_TOOL_NAME in registry.TOOL_SPECS or LOAD_SKILL_TOOL_NAME in registry.TOOL_SPECS:
        return False

    registry.register(LIST_SKILLS_TOOL_NAME, list_skills, _build_list_spec())
    registry.register(LOAD_SKILL_TOOL_NAME, load_skill, _build_load_spec())
    return True


def _build_list_spec() -> ToolSpec:
    schema = _function_schema(list_skills)
    return ToolSpec(
        name=LIST_SKILLS_TOOL_NAME,
        description=str(schema.get("description") or ""),
        input_schema=schema["parameters"],
        permission="read",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=5.0,
        max_output_tokens=settings.SKILL_MAX_BODY_TOKENS,
        requires_approval=False,
        feature_flag="ENABLE_SKILLS",
    )


def _function_schema(fn: Any) -> dict[str, Any]:
    """取 LangChain 实际交给模型的 function schema，保证 ToolSpec 与之一致。"""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return convert_to_openai_tool(fn)["function"]


def _build_load_spec() -> ToolSpec:
    schema = _function_schema(load_skill)
    return ToolSpec(
        name=LOAD_SKILL_TOOL_NAME,
        description=str(schema.get("description") or ""),
        input_schema=schema["parameters"],
        permission="read",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=5.0,
        max_output_tokens=settings.SKILL_MAX_BODY_TOKENS,
        requires_approval=False,
        feature_flag="ENABLE_SKILLS",
    )
