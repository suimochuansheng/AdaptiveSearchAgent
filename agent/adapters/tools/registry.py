"""工具注册表（T6-0）：`TOOLS` 的单一事实源。

- `TOOL_SPECS`：5 个内置工具按 `ToolSpec` 登记（取值见 `docs/contracts/tool-spec.md`）；
- `to_lc_tools()`：交给 `bind_tools` / `ToolNode` 的可调用体列表，**顺序即基线顺序**；
- `register()`：T9-3 的 MCP 工具动态登记入口（本卡只做入表 + 重名报错）。

`ToolSpec` 只承载元数据，不承载执行逻辑：执行 / 超时 / 预算在 T6-3。
内置的 `calculator` / `deep_research` 定义在 `src.conversation_graph`，而后者 import 本模块，
所以工具可调用体按需惰性导入（模块顶层导入会形成循环依赖）。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from agent.adapters.tools.executor import wrap_tool
from agent.domain.tool import CostModel, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Callable

# 内置工具名 -> 定义模块：`to_lc_tools()` 取不到 `_TOOL_FNS` 时按此惰性导入。
_TOOL_MODULE_BY_NAME: dict[str, str] = {
    "search_knowledge": "src.tools.rag",
    "search_tavily": "src.tools.search",
    "calculator": "src.conversation_graph",
    "deep_research": "src.conversation_graph",
    "read_offload": "src.tools.offload",
}

# 5 条 spec 的值来自 docs/contracts/tool-spec.md；`input_schema` 取自
# `convert_to_openai_tool(tool)["function"]["parameters"]`（2026-11-11 基线），
# 测试窗口用 schema 哈希回验，改工具签名/文档字符串必须同步这里。
TOOL_SPECS: dict[str, ToolSpec] = {
    "search_knowledge": ToolSpec(
        name="search_knowledge",
        description="调用子服务 /api/v1/search 进行向量检索。\n通过 HTTP 调用子服务的 /search 接口进行 RAG 检索。\n超时 RAG_TIMEOUT_SECONDS（默认 30s），若失败则返回空字符串（降级处理）。\nArgs:\n    query:  检索查询文本。\n    top_k:  返回的最相关结果数量。\n    kb_id:  目标知识库 ID。\n    model: Embedding 模型（仅 bge-m3）\n    search_mode: 检索模式 (hybrid 或 vector)",
        input_schema={
            "properties": {
                "query": {"type": "string"},
                "top_k": {"default": 3, "type": "integer"},
                "kb_id": {"default": "default", "type": "string"},
                "model": {"default": "bge-m3", "type": "string"},
                "search_mode": {"default": "hybrid", "type": "string"},
                "skip_reranker": {"default": False, "type": "boolean"},
            },
            "required": ["query"],
            "type": "object",
        },
        permission="external",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=35.0,
        max_output_tokens=3000,
        requires_approval=False,
        feature_flag=None,
    ),
    "search_tavily": ToolSpec(
        name="search_tavily",
        description="执行 Tavily 搜索，返回结果摘要文本。",
        input_schema={
            "properties": {"keyword": {"description": "搜索关键词", "type": "string"}},
            "required": ["keyword"],
            "type": "object",
        },
        permission="external",
        cost_model=CostModel(unit="keyword", unit_cost=0.02),
        timeout_seconds=20.0,
        max_output_tokens=3000,
        requires_approval=False,
        feature_flag=None,
    ),
    "calculator": ToolSpec(
        name="calculator",
        description='计算数学表达式（如 "2 + 3 * sin(0.5)"）。只接受具体数字与白名单数学函数。',
        input_schema={
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "type": "object",
        },
        permission="read",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=2.0,
        max_output_tokens=500,
        requires_approval=False,
        feature_flag=None,
    ),
    "deep_research": ToolSpec(
        name="deep_research",
        description="深度调研：对复杂问题执行「规划→并行搜索→评估→重试→生成报告」的完整闭环。 适用场景：需要多角度、多来源交叉验证的对比/调研类问题。\n内部运行旧版 workflow 主图（include_hitl=False 跳过人工确认——工具调用内\n无法挂起等人工，HITL 职责由外层会话循环承担）。",
        input_schema={
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "type": "object",
        },
        permission="dangerous",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=600.0,
        max_output_tokens=8000,
        requires_approval=True,
        feature_flag=None,
    ),
    "read_offload": ToolSpec(
        name="read_offload",
        description="读取之前被卸载的完整工具输出（可按关键词只取相关段落）。",
        input_schema={
            "properties": {
                "pointer": {
                    "description": "卸载时回灌的指针，形如 "
                    "`<thread>/<tool>-<时间戳>-<hash>.md` （相对 "
                    "data/offload 的路径，不含 `data/offload/` 前缀）。",
                    "type": "string",
                },
                "keyword": {
                    "default": "",
                    "description": "只返回包含该关键词的段落；留空则返回压缩后的全文。",
                    "type": "string",
                },
            },
            "required": ["pointer"],
            "type": "object",
        },
        permission="read",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=5.0,
        max_output_tokens=8000,
        requires_approval=False,
        feature_flag=None,
    ),
}

# `register()` 登记进来的可调用体；内置工具不落这里，见 `_resolve()`。
_TOOL_FNS: dict[str, Callable[..., Any]] = {}


def spec(name: str) -> ToolSpec:
    """按名字取一条 spec；未注册抛 `KeyError`。"""
    return TOOL_SPECS[name]


def all_specs() -> list[ToolSpec]:
    """按登记顺序返回全部 spec（5 个内置工具的顺序与基线一致）。"""
    return list(TOOL_SPECS.values())


def to_lc_tools() -> list[Callable[..., Any]]:
    """返回可直接交给 `bind_tools` / `ToolNode` 的工具列表。

    顺序 = `TOOL_SPECS` 的登记顺序；`register()` 追加的工具排在末尾。

    `ENABLE_TOOL_POLICY` 关闭时返回工具**原本的可调用体**（`is` 同一对象，
    不包装、不 IO）；开启时交给 `executor.wrap_tool` 做策略 / 超时 / 记账包装，
    签名与 schema 不变（schema 哈希仍是基线 `c48c9d48…`）。
    """
    return [wrap_tool(_resolve(name), TOOL_SPECS[name]) for name in TOOL_SPECS]


def register(name: str, fn: Callable[..., Any], tool_spec: ToolSpec) -> None:
    """登记一个动态工具（T9-3 MCP 用）；本卡只做入表与重名报错。

    Args:
        name: 工具名，全局唯一。
        fn: 工具可调用体。
        tool_spec: 该工具的元数据，`tool_spec.name` 必须与 `name` 一致。

    Raises:
        ValueError: 名字已存在，或 spec.name 与 name 不一致。
    """
    if name in TOOL_SPECS:
        raise ValueError(f"工具已注册，不能重复：{name}")
    if tool_spec.name != name:
        raise ValueError(f"spec.name（{tool_spec.name}）与注册名（{name}）不一致")
    TOOL_SPECS[name] = tool_spec
    _TOOL_FNS[name] = fn


def _resolve(name: str) -> Callable[..., Any]:
    """取工具可调用体：`register()` 进来的直接用，内置的按模块惰性导入。"""
    fn = _TOOL_FNS.get(name)
    if fn is not None:
        return fn
    module_name = _TOOL_MODULE_BY_NAME.get(name)
    if module_name is None:
        raise KeyError(f"工具没有实现体：{name}")
    return getattr(importlib.import_module(module_name), name)
