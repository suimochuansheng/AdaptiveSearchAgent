#!/usr/bin/env python
"""T60 工具注册表（T6-0 / G4）—— 卡级自检脚本。

用法（仓库根目录，任意 cwd 均可）：

    poetry run python "tests/test_agent_cases/T60-工具注册表/t60_checks.py"

退出码：
    0 = 全部检查通过（末行 `RESULT: n passed, 0 failed`）
    1 = 至少一项失败

为什么不是 `test_*.py`：
    `tests/test_agent_cases/README.md` 约定卡级自检统一命名 `<卡号小写>_checks.py`；
    叫 `test_*.py` 会被 pytest 收集，`--collect-only` 从 227/231 变成 227+N，
    `docs/baseline.md` §2 的 nodeids 哈希立即失效。

校验口径（T60-001 ~ T60-010）：
    1) `TOOLS` 5 个名字与顺序 = 基线；
    2) schema sha256 / 字节数 = 基线，且 5 个函数的返回注解仍是 `str`；
    3) `TOOL_SPECS` 字段与 `docs/contracts/tool-spec.md` 的 `t12:fields` 机读块逐项一致；
    4) 5 条 spec 的取值与本卡冻结表一致，枚举合法；
    5) 每条 `input_schema` = `convert_to_openai_tool(tool)["function"]["parameters"]`；
    6) `register()` 假工具后 `to_lc_tools()` 多一个、重名报错；
    7) `src/conversation_graph.py` 的 `TOOLS` 确实来自 `registry.to_lc_tools()`。

注意：脚本进程内会 `register()` 一个假工具（T60-008/009），
只影响本进程，不落盘、不影响 pytest。
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
import sys
import typing
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CONTRACT = REPO / "docs" / "contracts" / "tool-spec.md"
CONVERSATION_GRAPH = REPO / "src" / "conversation_graph.py"

if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import agent / src
    sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------------------------
# 冻结常量：这里的值 = T6-0 基线。改动必须同步 docs/contracts/tool-spec.md 与交接单。
# --------------------------------------------------------------------------------------

BASELINE_NAMES = [
    "search_knowledge",
    "search_tavily",
    "calculator",
    "deep_research",
    "read_offload",
]
BASELINE_SHA256 = "c48c9d4886caf77908fe850229bc04ae9b7ea60b3750bdba707ff815fcea2a99"
BASELINE_BYTES = 2558

TOOL_FIELDS = [
    "name",
    "description",
    "input_schema",
    "permission",
    "cost_model",
    "timeout_seconds",
    "max_output_tokens",
    "requires_approval",
    "feature_flag",
]
PERMISSIONS = {"read", "external", "write", "dangerous"}
COST_UNITS = {"call", "keyword", "token"}

# name -> 期望取值（permission / cost_model.unit / unit_cost / timeout / max_tokens / approval）
BASELINE_SPECS: dict[str, dict[str, Any]] = {
    "search_knowledge": {
        "permission": "external",
        "unit": "call",
        "unit_cost": 0.0,
        "timeout_seconds": 35.0,
        "max_output_tokens": 3000,
        "requires_approval": False,
        "feature_flag": None,
    },
    "search_tavily": {
        "permission": "external",
        "unit": "keyword",
        "unit_cost": 0.02,
        "timeout_seconds": 20.0,
        "max_output_tokens": 3000,
        "requires_approval": False,
        "feature_flag": None,
    },
    "calculator": {
        "permission": "read",
        "unit": "call",
        "unit_cost": 0.0,
        "timeout_seconds": 2.0,
        "max_output_tokens": 500,
        "requires_approval": False,
        "feature_flag": None,
    },
    "deep_research": {
        "permission": "dangerous",
        "unit": "call",
        "unit_cost": 0.0,
        "timeout_seconds": 600.0,
        "max_output_tokens": 8000,
        "requires_approval": True,
        "feature_flag": None,
    },
    "read_offload": {
        "permission": "read",
        "unit": "call",
        "unit_cost": 0.0,
        "timeout_seconds": 5.0,
        "max_output_tokens": 8000,
        "requires_approval": False,
        "feature_flag": None,
    },
}


class Checker:
    """极简 runner：每项打一行 `[PASS]/[FAIL]`，末尾汇总。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def run(self, case_id: str, title: str, fn: Any) -> None:
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - 卡级脚本要汇总所有失败项
            self.failed += 1
            print(f"[FAIL] {case_id} {title}: {type(exc).__name__}: {exc}")
        else:
            self.passed += 1
            print(f"[PASS] {case_id} {title}: {detail}")


def _converted() -> list[dict[str, Any]]:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from src.conversation_graph import TOOLS

    return [convert_to_openai_tool(tool) for tool in TOOLS]


def check_names() -> str:
    from src.conversation_graph import TOOLS

    names = [tool.__name__ for tool in TOOLS]
    assert names == BASELINE_NAMES, f"名字/顺序不一致：{names}"
    assert len(names) == 5, f"工具数应为 5，实际 {len(names)}"
    return "5/5 同名同序"


def check_schema_hash() -> str:
    schemas = _converted()
    canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    assert digest == BASELINE_SHA256, f"sha256 不一致：{digest}"
    assert len(canonical) == BASELINE_BYTES, f"字节数不一致：{len(canonical)}"
    return f"sha256={digest[:12]}… bytes={len(canonical)}"


def _contract_fields(marker: str) -> list[str]:
    text = CONTRACT.read_text(encoding="utf-8")
    block = re.search(
        rf"<!-- t12:fields {marker} -->(.*?)<!-- /t12:fields {marker} -->",
        text,
        re.S,
    )
    assert block is not None, f"契约缺少 t12:fields {marker} 机读块"
    return re.findall(r"^\| `([a-z_]+)` \|", block.group(1), re.M)


def check_contract_fields() -> str:
    from agent.domain.tool import ApprovalThreshold, CostModel, ToolSpec

    tool_fields = [field.name for field in dataclasses.fields(ToolSpec)]
    assert tool_fields == TOOL_FIELDS, f"ToolSpec 字段与冻结表不一致：{tool_fields}"
    assert _contract_fields("ToolSpec") == tool_fields, "契约 ToolSpec 字段与 dataclass 不一致"
    assert _contract_fields("CostModel") == [
        field.name for field in dataclasses.fields(CostModel)
    ], "契约 CostModel 字段与 dataclass 不一致"
    assert _contract_fields("ApprovalThreshold") == [
        field.name for field in dataclasses.fields(ApprovalThreshold)
    ], "契约 ApprovalThreshold 字段与 dataclass 不一致"
    return f"ToolSpec {len(tool_fields)} 字段 / CostModel 2 / ApprovalThreshold 2 逐项一致"


def check_spec_values() -> str:
    from agent.adapters.tools import registry
    from agent.domain.tool import CostModel, ToolSpec

    assert list(registry.TOOL_SPECS) == BASELINE_NAMES, (
        f"TOOL_SPECS 键/顺序不一致：{list(registry.TOOL_SPECS)}"
    )
    assert [item.name for item in registry.all_specs()] == BASELINE_NAMES, "all_specs() 顺序不一致"
    assert all(registry.spec(name) is registry.TOOL_SPECS[name] for name in BASELINE_NAMES), (
        "spec() 未返回同一对象"
    )

    for name, expected in BASELINE_SPECS.items():
        item = registry.spec(name)
        assert isinstance(item, ToolSpec), f"{name}: 不是 ToolSpec"
        assert item.name == name, f"{name}: spec.name={item.name}"
        assert isinstance(item.description, str) and item.description, f"{name}: description 为空"
        assert isinstance(item.input_schema, dict), f"{name}: input_schema 不是 dict"
        assert item.input_schema.get("type") == "object", f"{name}: input_schema 不是 object"

        assert item.permission in PERMISSIONS, f"{name}: permission={item.permission}"
        assert item.permission == expected["permission"], (
            f"{name}: permission={item.permission}，期望 {expected['permission']}"
        )

        assert isinstance(item.cost_model, CostModel), f"{name}: cost_model 类型错误"
        assert item.cost_model.unit in COST_UNITS, f"{name}: unit={item.cost_model.unit}"
        assert item.cost_model.unit == expected["unit"], f"{name}: unit 不一致"
        assert item.cost_model.unit_cost == expected["unit_cost"], f"{name}: unit_cost 不一致"

        assert isinstance(item.timeout_seconds, float), f"{name}: timeout 不是 float"
        assert item.timeout_seconds == expected["timeout_seconds"], f"{name}: timeout 不一致"
        assert isinstance(item.max_output_tokens, int), f"{name}: max_output_tokens 不是 int"
        assert item.max_output_tokens == expected["max_output_tokens"], (
            f"{name}: max_output_tokens 不一致"
        )
        assert isinstance(item.requires_approval, bool), (
            f"{name}: requires_approval 应为 bool（本卡不用阈值形态）"
        )
        assert item.requires_approval is expected["requires_approval"], (
            f"{name}: requires_approval 不一致"
        )
        assert item.feature_flag is expected["feature_flag"], f"{name}: feature_flag 不一致"

    assert registry.spec("deep_research").requires_approval is True, "deep_research 必须恒审批"
    return "5/5 spec 取值与冻结表一致，deep_research 恒审批"


def check_input_schema() -> str:
    from agent.adapters.tools import registry

    schemas = _converted()
    assert len(schemas) == 5, f"转换后应有 5 条 schema，实际 {len(schemas)}"
    for schema in schemas:
        name = schema["function"]["name"]
        assert schema["function"]["parameters"] == registry.spec(name).input_schema, (
            f"{name}: input_schema 与工具真实 schema 不一致"
        )
    return "5/5 input_schema = convert_to_openai_tool(...)[parameters]"


def check_return_types() -> str:
    from agent.adapters.tools import registry
    from src import conversation_graph as cg

    for name in BASELINE_NAMES:
        tool = getattr(cg, name)
        assert callable(tool), f"{name}: 不可调用"
        hints = typing.get_type_hints(tool)
        assert hints.get("return") is str, f"{name}: 返回注解不是 str（{hints.get('return')}）"
    assert [tool.__name__ for tool in registry.to_lc_tools()] == BASELINE_NAMES
    return "5/5 仍是可调用体且 return -> str"


def check_single_source() -> str:
    tree = ast.parse(CONVERSATION_GRAPH.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "TOOLS"
    ]
    assert len(assignments) == 1, f"TOOLS 赋值点应恰好 1 处，实际 {len(assignments)}"
    value = assignments[0].value
    assert isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute), (
        "TOOLS 不是 `registry.to_lc_tools()` 调用"
    )
    assert value.func.attr == "to_lc_tools", f"TOOLS 调用了 {value.func.attr}"
    assert isinstance(value.func.value, ast.Name) and value.func.value.id == "registry", (
        "TOOLS 调用者不是 registry"
    )
    return "TOOLS = registry.to_lc_tools()（唯一赋值点）"


def _fake_spec(name: str) -> Any:
    from agent.domain.tool import CostModel, ToolSpec

    return ToolSpec(
        name=name,
        description="T60 假工具：只用于验证 register",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        permission="read",
        cost_model=CostModel(unit="call", unit_cost=0.0),
        timeout_seconds=1.0,
        max_output_tokens=100,
        requires_approval=False,
        feature_flag=None,
    )


def check_register() -> str:
    from agent.adapters.tools import registry

    def fake_tool(text: str = "") -> str:
        """T60 假工具：回显入参。"""
        return text

    spec = _fake_spec("t60_fake_tool")
    before = len(registry.to_lc_tools())
    registry.register("t60_fake_tool", fake_tool, spec)

    assert registry.spec("t60_fake_tool") is spec, "spec() 取不到新登记的 spec"
    assert spec in registry.all_specs(), "all_specs() 不含新登记项"
    tools = registry.to_lc_tools()
    assert len(tools) == before + 1 == 6, f"to_lc_tools() 应多一个（5 -> 6），实际 {len(tools)}"
    assert tools[-1] is fake_tool, "新工具未排在末尾或不是原可调用体"
    assert [tool.__name__ for tool in tools[:-1]] == BASELINE_NAMES, "内置工具顺序被改动"
    return "假工具登记后 5 -> 6，顺序追加在末尾"


def check_register_duplicate() -> str:
    from agent.adapters.tools import registry

    def another_tool() -> str:
        return ""

    for name in ("search_knowledge", "t60_fake_tool"):
        try:
            registry.register(name, another_tool, _fake_spec(name))
        except ValueError as exc:
            assert "已注册" in str(exc), f"{name}: 异常信息不含'已注册'：{exc}"
        else:
            raise AssertionError(f"{name}: 重名登记未报错")
    return "重名登记均 ValueError（内置 + 动态）"


def check_specs_filled() -> str:
    """只填值、不加字段：5 条内置 spec 九字段齐全且非空（此时已追加 1 条假工具）。"""
    from agent.adapters.tools import registry
    from agent.domain.tool import ToolSpec

    assert len(dataclasses.fields(ToolSpec)) == 9, "ToolSpec 字段数必须仍是 9"
    assert list(registry.TOOL_SPECS)[:5] == BASELINE_NAMES, "内置 5 条顺序被改动"
    assert len(registry.TOOL_SPECS) == 6, f"应为 5 内置 + 1 假工具，实际 {len(registry.TOOL_SPECS)}"
    for item in registry.all_specs():
        assert item.description, f"{item.name}: description 为空"
        assert item.input_schema, f"{item.name}: input_schema 为空"
        assert item.cost_model is not None, f"{item.name}: cost_model 为空"
        assert {field.name for field in dataclasses.fields(item)} == set(TOOL_FIELDS)
    return "5 内置 + 1 假工具，均 9 字段且值非空"


def main() -> int:

    checker = Checker()
    checker.run("T60-001", "TOOLS 5 个名字与顺序", check_names)
    checker.run("T60-002", "schema sha256/字节数 = 基线", check_schema_hash)
    checker.run("T60-003", "契约字段与 dataclass 逐项一致", check_contract_fields)
    checker.run("T60-004", "5 条 spec 取值与冻结表一致", check_spec_values)
    checker.run("T60-005", "input_schema = 工具真实 schema", check_input_schema)
    checker.run("T60-006", "5 个函数返回注解仍为 str", check_return_types)
    checker.run("T60-007", "TOOLS 单一事实源 = registry", check_single_source)
    checker.run("T60-008", "register 假工具后 to_lc_tools 多一个", check_register)
    checker.run("T60-009", "重名 register 报错", check_register_duplicate)
    checker.run("T60-010", "只填值不加字段：9 字段且值非空", check_specs_filled)

    print(f"RESULT: {checker.passed} passed, {checker.failed} failed")
    return 0 if checker.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
