"""T9-3 演示用 MCP Server（stdio）：只暴露 tools 能力。

两个工具都是“本地不可能有副作用”的纯函数，用来证明**工具可以热插拔**：
不注册时模型只能看到 5 个内置工具；`ENABLE_MCP=True` 启动后变成 7 个。

启动（stdio，被 `agent/adapters/tools/mcp_client.py` 当子进程拉起）：

    poetry run python scripts/mcp_server.py

本文件**不 import 项目内模块**：MCP Server 是独立进程，只依赖 mcp SDK，
这样即使主应用配置炸了，工具进程也能单独跑起来排查。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("adsearchagent-demo-tools", log_level="WARNING")

#: 线性单位 → (量纲, 相对基准单位的倍率)
_LINEAR_UNITS: dict[str, tuple[str, float]] = {
    "km": ("length", 1000.0),
    "m": ("length", 1.0),
    "kg": ("mass", 1000.0),
    "g": ("mass", 1.0),
}

#: 温度单位（换算不是乘法，单独处理）
_TEMPERATURE_UNITS = {"c", "f"}

#: 单位别名（大小写已在入口统一成小写）
_UNIT_ALIASES: dict[str, str] = {
    "kilometer": "km",
    "kilometers": "km",
    "meter": "m",
    "meters": "m",
    "kilogram": "kg",
    "kilograms": "kg",
    "gram": "g",
    "grams": "g",
    "celsius": "c",
    "fahrenheit": "f",
}


@mcp.tool()
def word_count(text: str) -> str:
    """统计文本的字符数、词数与行数。

    Args:
        text: 待统计的文本（中英文均可）。

    Returns:
        形如「字符数：11，词数：2，行数：1」的一行结果；参数不是字符串时返回可读错误。
    """
    if not isinstance(text, str):
        return "参数 text 必须是字符串。"
    if not text:
        return "字符数：0，词数：0，行数：0"
    chars = len(text)
    words = len(text.split())
    lines = len(text.splitlines()) or 1
    return f"字符数：{chars}，词数：{words}，行数：{lines}"


@mcp.tool()
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算：支持 km/m、kg/g、c/f 三组。

    Args:
        value: 数值。
        from_unit: 源单位（km / m / kg / g / c / f，大小写不敏感）。
        to_unit: 目标单位（同源单位同一组）。

    Returns:
        形如「1 km = 1000 m」的结果；单位不支持或参数非法时返回可读错误，不抛异常。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "参数 value 必须是数字。"

    source = _normalize_unit(from_unit)
    target = _normalize_unit(to_unit)
    if source is None or target is None:
        return f"不支持的单位：{from_unit!r} → {to_unit!r}。支持 km/m、kg/g、c/f 三组。"

    if source == target:
        return f"{_fmt(value)} {source} = {_fmt(value)} {target}"

    if source in _TEMPERATURE_UNITS or target in _TEMPERATURE_UNITS:
        if source not in _TEMPERATURE_UNITS or target not in _TEMPERATURE_UNITS:
            return f"不支持跨量纲换算：{source} → {target}。温度只能在 c / f 之间换算。"
        converted = value * 9 / 5 + 32 if source == "c" else (value - 32) * 5 / 9
        return f"{_fmt(value)} {source} = {_fmt(converted)} {target}"

    source_dim, source_factor = _LINEAR_UNITS[source]
    target_dim, target_factor = _LINEAR_UNITS[target]
    if source_dim != target_dim:
        return f"不支持跨量纲换算：{source} → {target}。支持 km/m、kg/g、c/f 三组。"

    return f"{_fmt(value)} {source} = {_fmt(value * source_factor / target_factor)} {target}"


def _normalize_unit(unit: str) -> str | None:
    """单位归一：去空白、转小写、展开别名；未知单位返回 None。"""
    if not isinstance(unit, str):
        return None
    normalized = unit.strip().lower().replace("°", "")
    normalized = _UNIT_ALIASES.get(normalized, normalized)
    if normalized in _LINEAR_UNITS or normalized in _TEMPERATURE_UNITS:
        return normalized
    return None


def _fmt(value: float) -> str:
    """数值格式化：去掉无意义的尾零（1000.0 → 1000，1.5 → 1.5）。"""
    rounded = round(float(value), 4)
    return f"{rounded:g}"


def main() -> None:
    """stdio 传输（MCP 客户端按行读写 JSON-RPC，日志走 stderr）。"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
