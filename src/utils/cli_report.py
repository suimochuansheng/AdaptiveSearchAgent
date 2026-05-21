"""CLI 指标报告模块：使用 rich 库生成美观的终端 KPI 表格。

核心功能：
    1. 从 run_agent 完成后的状态数据中提取全部 KPI
    2. 使用 rich.Table 生成对齐、带颜色标记的指标表格
    3. 基于 DeepSeek 参考费率估算费用

DeepSeek 参考费率（2025）：
    - 输入：￥0.001 元 / 1K tokens
    - 输出：￥0.002 元 / 1K tokens

颜色规则：
    🟢 绿色 — 正常/优秀
    🟡 黄色 — 警告/临界
    🔴 红色 — 异常/需关注
"""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# DeepSeek 参考费率（每 1K tokens 的人民币价格）
_DEEPSEEK_INPUT_PRICE_PER_1K: float = 0.001
_DEEPSEEK_OUTPUT_PRICE_PER_1K: float = 0.002


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """根据 DeepSeek 费率估算费用（人民币，保留 4 位小数）。"""
    cost = (input_tokens / 1000) * _DEEPSEEK_INPUT_PRICE_PER_1K + (
        output_tokens / 1000
    ) * _DEEPSEEK_OUTPUT_PRICE_PER_1K
    return round(cost, 4)


def _status_text(value: str, color: str) -> Text:
    """创建带颜色的 rich Text 对象。"""
    return Text(value, style=color)


# ═══════════════════════════════════════════════════════════════════════
# 行填充子函数（按指标类别拆分为 3 个，每个圈复杂度 ≤ 10）
# ═══════════════════════════════════════════════════════════════════════


def _add_identity_rows(table: Table, thread_id: str, current_llm: str, model_switches: int) -> None:
    """添加：会话ID、最终模型、模型切换次数。"""
    # 会话 ID — 超过 20 字符截断显示
    tid = thread_id[:20] + "..." if len(thread_id) > 20 else thread_id
    table.add_row("会话 ID", tid, _status_text("✓ 正常", "green"))

    # 最终使用模型 — ollama=绿, deepseek=黄（表示切到了备援）
    if current_llm == "deepseek":
        ms = _status_text("⚡ DeepSeek (备援)", "yellow")
    elif current_llm == "ollama":
        ms = _status_text("🟢 Ollama (主)", "green")
    else:
        ms = _status_text(current_llm, "white")
    table.add_row("最终使用模型", current_llm, ms)

    # 模型切换次数 — 0=绿, 1=黄, ≥2=红
    if model_switches == 0:
        sw = _status_text("✓ 无切换", "green")
    elif model_switches == 1:
        sw = _status_text("⚠ 切换 1 次", "yellow")
    else:
        sw = _status_text(f"✗ 切换 {model_switches} 次", "red")
    table.add_row("模型切换次数", str(model_switches), sw)


def _add_token_rows(table: Table, in_tok: int, out_tok: int, total_tok: int, cost: float) -> None:
    """添加：输入/输出/总 Token、预估费用。"""
    # 输入 / 输出 Token — 中性色
    table.add_row("输入 Token", f"{in_tok:,}", _status_text("—", "white"))
    table.add_row("输出 Token", f"{out_tok:,}", _status_text("—", "white"))

    # 总 Token — <5000 绿, <20000 黄, ≥20000 红
    if total_tok < 5000:
        ts = _status_text("🟢 轻量", "green")
    elif total_tok < 20000:
        ts = _status_text("🟡 中等", "yellow")
    else:
        ts = _status_text("🔴 大量", "red")
    table.add_row("总 Token", f"{total_tok:,}", ts)

    # 预估费用 — <¥0.01 绿, <¥0.05 黄, ≥¥0.05 红
    cs = f"¥{cost:.4f}"
    if cost < 0.01:
        css = _status_text("🟢 极低", "green")
    elif cost < 0.05:
        css = _status_text("🟡 适中", "yellow")
    else:
        css = _status_text("🔴 较高", "red")
    table.add_row("预估费用 (DeepSeek)", cs, css)


def _add_result_rows(
    table: Table, confidence: float, approved: bool | None, elapsed: float
) -> None:
    """添加：置信度、人工审批、执行耗时。"""
    # 置信度 — ≥0.8 绿, ≥0.5 黄, <0.5 红
    cf = f"{confidence:.2%}"
    if confidence >= 0.8:
        cfs = _status_text("🟢 高置信", "green")
    elif confidence >= 0.5:
        cfs = _status_text("🟡 中等", "yellow")
    else:
        cfs = _status_text("🔴 低置信", "red")
    table.add_row("置信度得分", cf, cfs)

    # 人工审批 — True=绿, False=黄, None=N/A
    if approved is True:
        at, aps = "是", _status_text("✓ 已批准", "green")
    elif approved is False:
        at, aps = "否", _status_text("✗ 未批准", "yellow")
    else:
        at, aps = "N/A", _status_text("— N/A", "white")
    table.add_row("人工审批结果", at, aps)

    # 执行耗时 — <30s 绿, <60s 黄, ≥60s 红
    es = f"{elapsed:.2f}s"
    if elapsed < 30:
        ets = _status_text("🟢 快速", "green")
    elif elapsed < 60:
        ets = _status_text("🟡 中等", "yellow")
    else:
        ets = _status_text("🔴 较慢", "red")
    table.add_row("执行耗时", es, ets)


# ═══════════════════════════════════════════════════════════════════════
# 公开入口
# ═══════════════════════════════════════════════════════════════════════


def print_kpi_dashboard(cost_info: dict[str, Any]) -> None:
    """输出智能体执行 KPI 仪表盘（rich 表格）。

    在 run_agent 完成后调用，传入包含以下键的字典：
        thread_id, current_llm, model_switches,
        input_tokens, output_tokens, total_tokens,
        confidence_score, human_approved, elapsed_seconds
    """
    console = Console()

    # ── 标题面板 ─────────────────────────────────────────────────
    console.print(
        Panel(
            Text("智能体执行报告 & KPI 指标", style="bold cyan"),
            border_style="cyan",
            padding=(1, 4),
        )
    )

    # ── 表格定义（三列：指标 | 数值 | 状态） ─────────────────────
    table = Table(
        title="KPI 指标详情",
        show_header=True,
        header_style="bold white",
        border_style="bright_black",
        title_style="bold underline",
    )
    table.add_column("指标", style="cyan", no_wrap=True, width=20)
    table.add_column("数值", style="white", width=30)
    table.add_column("状态", width=16)

    # ── 分类填充（由 3 个子函数完成，每个圈复杂度 ≤ 10） ──────
    _add_identity_rows(
        table,
        str(cost_info.get("thread_id", "N/A")),
        str(cost_info.get("current_llm", "N/A")),
        int(cost_info.get("model_switches", 0)),
    )
    _add_token_rows(
        table,
        int(cost_info.get("input_tokens", 0)),
        int(cost_info.get("output_tokens", 0)),
        int(cost_info.get("total_tokens", 0)),
        _estimate_cost(
            int(cost_info.get("input_tokens", 0)),
            int(cost_info.get("output_tokens", 0)),
        ),
    )
    _add_result_rows(
        table,
        float(cost_info.get("confidence_score", 0.0)),
        cost_info.get("human_approved"),
        float(cost_info.get("elapsed_seconds", 0.0)),
    )

    # ── 输出 ─────────────────────────────────────────────────────
    console.print(table)
    console.print(
        Text(
            "※ 费用基于 DeepSeek 参考费率：输入 ¥0.001/1K tokens，输出 ¥0.002/1K tokens",
            style="dim italic",
        )
    )
