#!/usr/bin/env python
"""T12 契约冻结（T1-2 / G3）—— 卡级自检脚本。

用法（仓库根目录，任意 cwd 均可）：
  poetry run python "tests/test_agent_cases/T12-契约冻结/t12_checks.py"

退出码：
  0 = 全部检查通过（末行 `RESULT: n passed, 0 failed`）
  1 = 至少一项失败（末行 m > 0）

为什么不是 `test_*.py`：
  `pyproject.toml` 的 `testpaths=["tests"]` + pytest 默认收集规则会把 `test_*.py` 收进套件，
  收集数会从冻结的 227/231 变成 227+N，`docs/baseline.md` §2 的 nodeids 哈希立刻失效，
  每张卡"开工前三件事"第 3 条就过不了。按 `tests/test_agent_cases/README.md` 的约定，
  卡级自检统一命名 `<卡号小写>_checks.py`，独立进程跑、自己打印 `[PASS]/[FAIL]` 与
  `RESULT:` 末行、用退出码表达结论。**永不**叫 `test_*.py`。

编码：
  本脚本、`docs/contracts/*.md`、`agent/domain/*.py` 一律显式 UTF-8 读写；
  台账 `tests/test_agent_cases/台账用例.csv` 是 **GB18030**（只读回验 0a/0b 两行，
  读它必须 `encoding="gb18030"`，否则中文列直接 UnicodeDecodeError；本脚本不写台账）。

校验口径：
  1) 五份契约 md 的字段名/枚举值/键名，与 `agent/domain/*.py` 的 dataclass 字段**逐项**
     比对（顺序 + 名称），并与本脚本顶部的"冻结常量"比对；
  2) `kind` 六值、`Decision` 二值、`permission` 四值、`cost_model.unit` 三值；
  3) `approval.md` / `resume-http.md` 的**契约正文**不得出现第二凭证字段
     （历史裁决记录集中放在 `<!-- t12:history -->` 之后的变更记录段，允许提及）。
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import io
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

REPO = Path(__file__).resolve().parents[3]
CONTRACTS = REPO / "docs" / "contracts"
DOMAIN = REPO / "agent" / "domain"
LEDGER = REPO / "tests" / "test_agent_cases" / "台账用例.csv"

if str(REPO) not in sys.path:  # 让脚本在任意 cwd 下都能 import agent.domain
    sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------------------------
# 冻结常量：这里的值 = 契约本身。改这里就等于改契约，必须走 v2 修订文件。
# --------------------------------------------------------------------------------------

CONTRACT_FILES = [
    "runtime-event.md",
    "approval.md",
    "resume-http.md",
    "tool-spec.md",
    "trace-span.md",
]

RUNTIME_KINDS = ["token", "tool_call", "tool_result", "interrupt", "final", "error"]
RUNTIME_FIELDS = ["kind", "payload"]
PAYLOAD_KEYS: dict[str, list[str]] = {
    "token": ["content"],
    "tool_call": ["tools"],
    "tool_result": ["tool_call_id", "tool", "content"],
    "interrupt": ["approval"],
    "final": ["content"],
    "error": ["message"],
}
TOOL_CALL_ELEMENT_KEYS = ["name", "args"]
IMPLEMENTED_SSE_KINDS = ["token", "tool_call", "final", "error"]

APPROVAL_FIELDS = [
    "approval_id",
    "tool_call_id",
    "tool",
    "args_digest",
    "cost_estimate",
    "budget_remaining",
    "permission",
    "expires_at",
]
DECISIONS = ["approve", "reject"]

TOOL_SPEC_FIELDS = [
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
PERMISSIONS = ["read", "external", "write", "dangerous"]
COST_MODEL_FIELDS = ["unit", "unit_cost"]
COST_UNITS = ["call", "keyword", "token"]
THRESHOLD_FIELDS = ["threshold_field", "threshold"]

RESUME_BODY_KEYS = ["approval_id", "decision", "request_id"]
RESUME_STATUSES = ["200", "409", "410", "404"]

TRACE_FIELDS = [
    "trace_id",
    "span_id",
    "parent_span_id",
    "session_id",
    "turn_index",
    "name",
    "kind",
    "start",
    "end",
    "attrs",
]
SPAN_KINDS = ["turn", "llm", "tool", "compress", "offload", "approval"]
SPAN_ATTRS = ["tool", "tokens_in", "tokens_out", "cost", "error_kind", "approval_id"]

HISTORY_MARKER = "<!-- t12:history -->"
TOKEN_WORD = re.compile(r"(?<!\w)token(?!\w)")

# 台账 0a/0b 落位（GB18030）
LEDGER_ROWS = 48
LEDGER_COLS = 15
TDM_016_NOTES = "三字段；request_id=v4"


# --------------------------------------------------------------------------------------
# 解析工具
# --------------------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _md(name: str) -> str:
    return _read(CONTRACTS / name)


def _block(text: str, kind: str, block_id: str) -> list[str] | None:
    """取 `<!-- t12:<kind> <id> -->` 与闭合标记之间的表格首列（去反引号）。

    返回 None 表示锚点缺失（调用方据此 FAIL）。
    """
    pattern = re.compile(
        rf"<!-- t12:{kind} {re.escape(block_id)} -->\r?\n(.*?)"
        rf"<!-- /t12:{kind} {re.escape(block_id)} -->",
        re.S,
    )
    match = pattern.search(text)
    if match is None:
        return None
    names: list[str] = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        first = line.strip("|").split("|")[0].strip()
        cell = re.fullmatch(r"`([^`]+)`", first)
        if cell is not None:
            names.append(cell.group(1))
    return names


def _require_block(text: str, kind: str, block_id: str) -> list[str]:
    got = _block(text, kind, block_id)
    assert got is not None, f"缺机读锚点 `<!-- t12:{kind} {block_id} -->`"
    return got


def _json_blocks(text: str) -> list[Any]:
    """解析 md 里所有 ```json 代码块。"""
    out: list[Any] = []
    for body in re.findall(r"```json\r?\n(.*?)```", text, re.S):
        out.append(json.loads(body))
    return out


def _fields_of(cls: type) -> list[str]:
    return [f.name for f in dataclasses.fields(cls)]


def _literal_args(value: Any) -> list[str]:
    from typing import get_args

    return list(get_args(value))


# --------------------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------------------


def check_contracts_present() -> str:
    """五份契约文件存在、UTF-8 可读、带 v1 冻结标记与机读锚点。"""
    missing = [name for name in CONTRACT_FILES if not (CONTRACTS / name).is_file()]
    assert not missing, f"缺契约文件: {missing}"
    anchors = 0
    for name in CONTRACT_FILES:
        text = _md(name)
        assert "v1" in text and "冻结" in text, f"{name} 缺 v1 冻结标记"
        own = len(re.findall(r"<!-- t12:", text))
        assert own > 0, f"{name} 没有 t12 机读锚点"
        anchors += own
    return f"五份契约存在且冻结：{', '.join(CONTRACT_FILES)}；机读锚点 {anchors} 个"


def check_runtime_kinds() -> str:
    """`RuntimeEvent.kind` 恒为六值，且 md 枚举 == 冻结常量。"""
    text = _md("runtime-event.md")
    got = _require_block(text, "enum", "RuntimeEvent.kind")
    assert got == RUNTIME_KINDS, f"kind 枚举不一致: md={got} 冻结={RUNTIME_KINDS}"
    assert len(got) == len(set(got)) == 6, f"kind 不是 6 个唯一值: {got}"
    return f"kind = {got}（6/6，无新增）"


def check_runtime_fields_and_payload() -> str:
    """`RuntimeEvent` 二字段 + 六个 payload 形状（含 tool_call 元素）逐项一致。"""
    text = _md("runtime-event.md")
    got = _require_block(text, "fields", "RuntimeEvent")
    assert got == RUNTIME_FIELDS, f"RuntimeEvent 字段不一致: md={got}"
    for kind, keys in PAYLOAD_KEYS.items():
        real = _require_block(text, "keys", f"RuntimeEvent.payload.{kind}")
        assert real == keys, f"payload[{kind}] 键不一致: md={real} 冻结={keys}"
    elem = _require_block(text, "keys", "RuntimeEvent.payload.tool_call.tools[]")
    assert elem == TOOL_CALL_ELEMENT_KEYS, f"tool_call 元素键不一致: md={elem}"
    total = sum(len(v) for v in PAYLOAD_KEYS.values())
    return f"RuntimeEvent 字段 {got}；6 个 payload 共 {total} 键 + tools[] 元素 2 键，与冻结常量逐项一致"


def check_sse_mapping() -> str:
    """与现网 SSE 的逐项对照：四帧已实现、interrupt 待 T9-2、ping 是心跳不进 RuntimeEvent。"""
    text = _md("runtime-event.md")
    for kind in IMPLEMENTED_SSE_KINDS:
        row = next((ln for ln in text.splitlines() if ln.startswith(f"| `{kind}` |")), None)
        assert row is not None, f"对照表缺 `{kind}` 行"
        assert "已实现" in row, f"`{kind}` 行未标已实现: {row}"
    interrupt_row = next((ln for ln in text.splitlines() if ln.startswith("| `interrupt` |")), None)
    assert interrupt_row is not None, "对照表缺 `interrupt` 行"
    assert "T9-2" in interrupt_row, f"`interrupt` 行未标 T9-2: {interrupt_row}"
    assert "心跳" in text and "不进 RuntimeEvent" in text, "缺 ping 心跳、不进 RuntimeEvent 的说明"
    return "对照表：token/tool_call/final/error=已实现，interrupt=待 T9-2，ping=心跳且不进 RuntimeEvent"


def check_approval_fields() -> str:
    """`ApprovalRequest` 八字段：md == dataclass，逐项且顺序一致。"""
    from agent.domain.roles import ApprovalRequest

    md_fields = _require_block(_md("approval.md"), "fields", "ApprovalRequest")
    code_fields = _fields_of(ApprovalRequest)
    assert md_fields == APPROVAL_FIELDS, f"md 字段不一致: {md_fields}"
    assert code_fields == APPROVAL_FIELDS, f"dataclass 字段不一致: {code_fields}"
    assert md_fields == code_fields, f"md 与代码字段不一致: md={md_fields} code={code_fields}"
    return f"md 8 字段 == roles.ApprovalRequest 8 字段（顺序一致）：{code_fields}"


def check_decision_two_values() -> str:
    """`Decision` 二值：md 表 2 行 == Literal args == 冻结常量。"""
    from agent.domain.roles import Decision

    md_values = _require_block(_md("approval.md"), "enum", "Decision")
    code_values = _literal_args(Decision)
    assert md_values == DECISIONS, f"md Decision 不一致: {md_values}"
    assert code_values == DECISIONS, f"Literal 不一致: {code_values}"
    assert md_values == code_values, f"md 与代码不一致: md={md_values} code={code_values}"
    return f"Decision = {code_values}（2 值；无 approve_and_remember）"


def check_tool_spec_fields() -> str:
    """`ToolSpec` 九字段：md == dataclass == 冻结常量。"""
    from agent.domain.tool import ToolSpec

    md_fields = _require_block(_md("tool-spec.md"), "fields", "ToolSpec")
    code_fields = _fields_of(ToolSpec)
    assert md_fields == TOOL_SPEC_FIELDS, f"md 字段不一致: {md_fields}"
    assert code_fields == TOOL_SPEC_FIELDS, f"dataclass 字段不一致: {code_fields}"
    assert md_fields == code_fields, f"md 与代码字段不一致: md={md_fields} code={code_fields}"
    return f"md 9 字段 == tool.ToolSpec 9 字段（顺序一致）：{code_fields}"


def check_permission_four_values() -> str:
    """`permission` 四值：md 枚举 == Literal args == 冻结常量。"""
    from agent.domain.tool import Permission

    md_values = _require_block(_md("tool-spec.md"), "enum", "ToolSpec.permission")
    code_values = _literal_args(Permission)
    assert md_values == PERMISSIONS, f"md permission 不一致: {md_values}"
    assert code_values == PERMISSIONS, f"Literal 不一致: {code_values}"
    assert md_values == code_values, f"md 与代码不一致: md={md_values} code={code_values}"
    return f"permission = {code_values}（4 值）"


def check_cost_model() -> str:
    """`CostModel` 二字段 + `unit` 三值：md == dataclass == 冻结常量。"""
    from agent.domain.tool import CostModel, CostUnit

    text = _md("tool-spec.md")
    md_fields = _require_block(text, "fields", "CostModel")
    code_fields = _fields_of(CostModel)
    assert md_fields == COST_MODEL_FIELDS, f"md CostModel 字段不一致: {md_fields}"
    assert code_fields == COST_MODEL_FIELDS, f"dataclass 字段不一致: {code_fields}"
    md_units = _require_block(text, "enum", "CostModel.unit")
    code_units = _literal_args(CostUnit)
    assert md_units == COST_UNITS, f"md unit 不一致: {md_units}"
    assert code_units == COST_UNITS, f"Literal 不一致: {code_units}"
    return f"CostModel 字段 {code_fields}；unit = {code_units}（3 值）"


def check_requires_approval_forms() -> str:
    """`requires_approval` 两形态（bool / 阈值对象），阈值对象字段与 md 一致。"""
    from agent.domain.tool import ApprovalThreshold, ToolSpec

    text = _md("tool-spec.md")
    md_fields = _require_block(text, "fields", "ApprovalThreshold")
    code_fields = _fields_of(ApprovalThreshold)
    assert md_fields == THRESHOLD_FIELDS, f"md 阈值字段不一致: {md_fields}"
    assert code_fields == THRESHOLD_FIELDS, f"dataclass 字段不一致: {code_fields}"
    field_types = {f.name: str(f.type) for f in dataclasses.fields(ToolSpec)}
    annotation = field_types["requires_approval"]
    assert "bool" in annotation and "ApprovalThreshold" in annotation, (
        f"requires_approval 注解不是两形态: {annotation}"
    )
    return f"requires_approval: {annotation}；阈值对象字段 {code_fields}（>= 判定）"


def check_resume_http() -> str:
    """resume body 恰三键、状态码 200/409/410/404、幂等 24h；approval_id 来自 ApprovalRequest。"""
    text = _md("resume-http.md")
    md_keys = _require_block(text, "keys", "resume.body")
    assert md_keys == RESUME_BODY_KEYS, f"body 键不一致: md={md_keys}"
    assert set(md_keys) == {"approval_id", "decision", "request_id"}, f"body 不是三字段: {md_keys}"
    assert "approval_id" in APPROVAL_FIELDS, "body 的 approval_id 在 ApprovalRequest 里不存在"
    assert "request_id" not in APPROVAL_FIELDS, "request_id 应由客户端生成，不属于 ApprovalRequest"
    for status in RESUME_STATUSES:
        assert f"`{status}`" in text, f"缺状态码 {status}"
    assert "24 小时" in text, "缺 request_id 24h 幂等窗口"
    blocks = _json_blocks(text)
    assert blocks, "缺 body 示例 JSON"
    example = blocks[0]
    assert isinstance(example, dict) and set(example) == set(RESUME_BODY_KEYS), (
        f"示例 body 键不一致: {example!r}"
    )
    assert example["decision"] in DECISIONS, f"示例 decision 非法: {example['decision']!r}"
    return f"body 三键 {md_keys}；示例 JSON 键一致；状态码 {RESUME_STATUSES}；幂等 24h"


def check_no_token_in_contract_body() -> str:
    """`approval.md` / `resume-http.md` 契约正文无第二凭证字段；历史段须记录删除。"""
    import agent.domain.roles as roles
    import agent.domain.tool as tool

    for name in ("approval.md", "resume-http.md"):
        text = _md(name)
        body, marker, history = text.partition(HISTORY_MARKER)
        assert marker, f"{name} 缺 `{HISTORY_MARKER}` 变更记录分隔"
        hits = [
            f"L{i}: {line.strip()}"
            for i, line in enumerate(body.splitlines(), 1)
            if TOKEN_WORD.search(line)
        ]
        assert not hits, f"{name} 契约正文出现第二凭证字段: {hits}"
        assert "删除" in history and "2026-11-11" in history, f"{name} 历史段未记录删除裁决"
    for cls in (roles.ApprovalRequest, tool.ToolSpec):
        names = _fields_of(cls)
        assert "token" not in names, f"{cls.__name__} 出现 token 字段: {names}"
    assert "token" not in RESUME_BODY_KEYS, f"resume body 出现 token: {RESUME_BODY_KEYS}"
    return "approval/resume 契约正文 token 命中 0 处；两处历史段各有 2026-11-11 删除记录；dataclass 无 token"


def check_trace_span() -> str:
    """`TraceSpan` 十字段 / 六 kind / 六 attrs / 落盘口径；示例行可解析。"""
    text = _md("trace-span.md")
    md_fields = _require_block(text, "fields", "TraceSpan")
    assert md_fields == TRACE_FIELDS, f"TraceSpan 字段不一致: md={md_fields}"
    md_kinds = _require_block(text, "enum", "TraceSpan.kind")
    assert md_kinds == SPAN_KINDS, f"span kind 不一致: md={md_kinds}"
    md_attrs = _require_block(text, "keys", "TraceSpan.attrs")
    assert md_attrs == SPAN_ATTRS, f"attrs 白名单不一致: md={md_attrs}"
    assert "logs/trace" in text and "<date>.jsonl" in text, "缺 logs/trace/<date>.jsonl 落盘路径"
    assert "不写业务表" in text and "不写 checkpoint" in text, "缺不写业务表 / checkpoint 的约束"
    blocks = _json_blocks(text)
    assert blocks, "缺示例 JSONL 行"
    example = blocks[0]
    assert isinstance(example, dict) and list(example) == TRACE_FIELDS, (
        f"示例行字段与 TraceSpan 不一致: {list(example) if isinstance(example, dict) else example}"
    )
    assert example["kind"] in SPAN_KINDS, f"示例 kind 非法: {example['kind']!r}"
    extra = set(example["attrs"]) - set(SPAN_ATTRS)
    assert not extra, f"示例 attrs 越界: {extra}"
    return (
        f"TraceSpan {len(md_fields)} 字段；kind {len(md_kinds)} 值；attrs 白名单 {md_attrs}；"
        "落盘 logs/trace/<date>.jsonl（不写业务表/checkpoint）"
    )


def check_domain_no_framework() -> str:
    """`agent/domain/*.py` 零框架依赖（import-linter 的同口径前置验证）。"""
    forbidden = ("langchain", "langgraph")
    stdlib = set(sys.stdlib_module_names)
    external: list[str] = []
    count = 0
    for path in sorted(DOMAIN.glob("*.py")):
        tree = ast.parse(_read(path), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            for mod in modules:
                count += 1
                root = mod.split(".")[0]
                if mod.startswith(forbidden):
                    external.append(f"{path.name}: {mod}（框架依赖）")
                elif root == "agent":
                    if not mod.startswith("agent.domain"):
                        external.append(f"{path.name}: {mod}（domain 只能依赖 agent.domain 内部）")
                elif root not in stdlib:
                    external.append(f"{path.name}: {mod}（非 stdlib 的第三方依赖）")
    assert not external, f"domain 依赖越界: {external}"
    return f"agent/domain 共 {count} 条 import，0 条 langchain/langgraph、0 条 agent.app/adapters"


def check_ledger_rows() -> str:
    """台账不少于 48 行 / 15 列（GB18030），0a 的 TDM-016 与 0b 的 T02-005 已落位。"""
    raw = LEDGER.read_bytes()
    text = raw.decode("gb18030")  # 关键：台账是 GB18030，不是 UTF-8
    rows = list(csv.reader(io.StringIO(text)))
    # 2026-11-11 测试窗口裁定（OI-T12-003）：台账只允许追加，本卡自己就会 +10 行，
    # 后续窗口亦然，故改为“不少于基线行数”；0a/0b 两行仍逐项校验。
    assert len(rows) >= LEDGER_ROWS + 1, f"台账行数不足: {len(rows)}（基线表头 + {LEDGER_ROWS}）"
    bad = [i for i, row in enumerate(rows, 1) if len(row) != LEDGER_COLS]
    assert not bad, f"台账列数不是 {LEDGER_COLS}: 第 {bad} 行"
    by_id = {row[0]: row for row in rows[1:]}
    assert "TDM-016" in by_id, "台账缺 TDM-016"
    assert by_id["TDM-016"][13] == TDM_016_NOTES, f"TDM-016 notes 不对: {by_id['TDM-016'][13]!r}"
    assert by_id["TDM-016"][14] == "", f"TDM-016 open_issue 未清空: {by_id['TDM-016'][14]!r}"
    assert by_id["T02-005"][10] == "block", f"T02-005 status 不是 block: {by_id['T02-005'][10]!r}"
    assert "计划窗口裁定" in by_id["T02-005"][13], "T02-005 notes 缺裁定说明"
    return (
        f"台账 {len(rows) - 1} 行 × {LEDGER_COLS} 列（GB18030 解码通过，基线 {LEDGER_ROWS}）；"
        "TDM-016 notes=三字段/open_issue 空；T02-005 status=block"
    )


CHECKS: list[tuple[str, str, Callable[[], str]]] = [
    ("T12-001", "五份契约存在且冻结", check_contracts_present),
    ("T12-002", "RuntimeEvent.kind 六值", check_runtime_kinds),
    ("T12-003", "RuntimeEvent 字段与 payload 形状", check_runtime_fields_and_payload),
    ("T12-004", "与现网 SSE 逐项对照", check_sse_mapping),
    ("T12-005", "ApprovalRequest 八字段 md==代码", check_approval_fields),
    ("T12-006", "Decision 二值", check_decision_two_values),
    ("T12-007", "ToolSpec 九字段 md==代码", check_tool_spec_fields),
    ("T12-008", "permission 四值", check_permission_four_values),
    ("T12-009", "CostModel 二字段 + unit 三值", check_cost_model),
    ("T12-010", "requires_approval 两形态", check_requires_approval_forms),
    ("T12-011", "resume body 三键 + 状态码 + 24h 幂等", check_resume_http),
    ("T12-012", "approval/resume 正文无第二凭证字段", check_no_token_in_contract_body),
    ("T12-013", "TraceSpan 十字段/六 kind/六 attrs", check_trace_span),
    ("T12-014", "agent/domain 零框架依赖", check_domain_no_framework),
    ("T12-015", "台账 不少于 48 行 15 列 + 0a/0b 落位", check_ledger_rows),
]


def main() -> int:
    passed = 0
    failed = 0
    for case_id, title, func in CHECKS:
        try:
            detail = func()
        except Exception as exc:  # noqa: BLE001 —— 单条失败不影响其余用例
            failed += 1
            print(f"[FAIL] {case_id} {title} — {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"[PASS] {case_id} {title} — {detail}")
    print(f"RESULT: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
