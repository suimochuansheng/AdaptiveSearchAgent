#!/usr/bin/env python3
"""测试 json_parser 四层回退机制 + metrics 集成。

覆盖场景：
  策略1 - 直接 JSON 解析（正常 JSON）
  策略2 - 嵌套感知正则提取（markdown 代码块 + 嵌套 JSON）
  策略3 - 贪婪正则匹配（JSON 混在长文本中）
  策略4 - DeepSeek LLM 修复（损坏的 JSON，需配置 DEEPSEEK_API_KEY）
  全部失败 - 无法解析的文本
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.json_parser import (
    DEEPSEEK_API_KEY,
    _parse_direct,
    _parse_regex_greedy,
    _parse_regex_nested,
    _repair_with_deepseek,
    robust_json_parse,
)
from src.utils.metrics import metrics

# ==================== 辅助函数 ====================


def reset_metrics() -> None:
    """重置 metrics 计数器，确保每个测试场景独立统计。"""
    metrics.parse_success = 0
    metrics.parse_fallback = 0
    metrics.parse_deepseek_fallback = 0


def print_metrics(label: str) -> None:
    """打印当前 metrics 状态。"""
    print(
        f"  [{label}] success={metrics.parse_success}, "
        f"fallback={metrics.parse_fallback}, "
        f"deepseek={metrics.parse_deepseek_fallback}"
    )


def assert_metrics(
    test_name: str,
    expected_success: int,
    expected_fallback: int,
    expected_deepseek: int,
) -> None:
    """断言 metrics 计数器符合预期。"""
    ok = (
        metrics.parse_success == expected_success
        and metrics.parse_fallback == expected_fallback
        and metrics.parse_deepseek_fallback == expected_deepseek
    )
    status = "✅ PASS" if ok else "❌ FAIL"
    print(
        f"  {status}: {test_name} "
        f"(期望 success={expected_success} fallback={expected_fallback} "
        f"deepseek={expected_deepseek})"
    )
    if not ok:
        print_metrics("  实际值")


# ==================== 测试 1: 策略1 - 直接解析 ====================
print("=" * 60)
print("测试 1: 策略1 - 直接 JSON 解析")
print("=" * 60)
reset_metrics()

# 1.1 标准 JSON 对象
result = robust_json_parse('{"plan": ["搜索关键词1", "搜索关键词2"]}')
print(f"  1.1 标准 JSON → {result}")
assert_metrics("标准 JSON", expected_success=1, expected_fallback=0, expected_deepseek=0)

# 1.2 JSON 数组（非字典，应失败）
reset_metrics()
result = robust_json_parse('["a", "b", "c"]')
print(f"  1.2 JSON 数组（非字典）→ {result}")
assert_metrics("JSON 数组拒绝", expected_success=0, expected_fallback=0, expected_deepseek=0)

# 1.3 空字符串
reset_metrics()
result = robust_json_parse("")
print(f"  1.3 空字符串 → {result}")
assert_metrics("空字符串", expected_success=0, expected_fallback=0, expected_deepseek=0)

# ==================== 测试 2: 策略2 - 嵌套感知正则 ====================
print("\n" + "=" * 60)
print("测试 2: 策略2 - 嵌套感知正则提取")
print("=" * 60)
reset_metrics()

# 2.1 JSON 被 markdown 代码块包裹
result = robust_json_parse('```json\n{"score": 0.95, "items": ["a"]}\n```')
print(f"  2.1 markdown 代码块 → {result}")
assert_metrics("markdown 代码块", expected_success=1, expected_fallback=1, expected_deepseek=0)

# 2.2 JSON 有嵌套对象（最外层大括号内含一层嵌套）
reset_metrics()
result = robust_json_parse('前缀文本 {"outer": {"inner": "value"}, "list": [1,2]} 后缀文本')
print(f"  2.2 嵌套 JSON + 前后缀 → {result}")
assert_metrics("嵌套 JSON", expected_success=1, expected_fallback=1, expected_deepseek=0)

# 2.3 多个 JSON 候选，提取第一个
reset_metrics()
result = robust_json_parse('{"first": 1} 其他内容 {"second": 2}')
print(f"  2.3 多个 JSON 候选 → {result}")
assert_metrics("多候选 JSON", expected_success=1, expected_fallback=1, expected_deepseek=0)

# ==================== 测试 3: 策略3 - 贪婪正则 ====================
print("\n" + "=" * 60)
print("测试 3: 策略3 - 贪婪正则匹配")
print("=" * 60)
reset_metrics()

# 3.1 JSON 混在长文本中，且内部有换行
result = robust_json_parse(
    "这是一段很长的介绍文字，包含了各种描述。\n"
    '最终结论是：{"result": "成功", "confidence": 0.88}\n'
    "以上是分析结果。"
)
print(f"  3.1 JSON 混在长文本中 → {result}")
assert_metrics("长文本中提取", expected_success=1, expected_fallback=1, expected_deepseek=0)

# 3.2 JSON 跨多行且有特殊字符
reset_metrics()
result = robust_json_parse(
    "以下是输出：\n"
    "{\n"
    '  "title": "测试标题",\n'
    '  "description": "包含换行\\n和引号\\"的内容",\n'
    '  "count": 42\n'
    "}\n"
    "输出完毕。"
)
print(f"  3.2 多行 JSON（贪婪匹配）→ {result}")
assert_metrics("多行 JSON", expected_success=1, expected_fallback=1, expected_deepseek=0)

# ==================== 测试 4: 全部三层失败（无 DeepSeek 时） ====================
print("\n" + "=" * 60)
print("测试 4: 三层正则全失败 → 最终降级返回 {}")
print("=" * 60)
reset_metrics()

# 4.1 完全不包含任何大括号的文本
result = robust_json_parse("这是一段完全不包含 JSON 结构的纯文本。")
print(f"  4.1 纯文本无大括号 → {result}")

# 4.2 损坏的 JSON（缺少引号、逗号等）
reset_metrics()
result = robust_json_parse("{plan: [关键词1, 关键词2]}")
print(f"  4.2 损坏的 JSON（无引号）→ {result}")
print("     （注意：如果配置了 DEEPSEEK_API_KEY，策略4 会尝试修复）")

# ==================== 测试 5: 策略4 - DeepSeek LLM 修复（条件执行） ====================
print("\n" + "=" * 60)
print("测试 5: 策略4 - DeepSeek LLM 模型降级修复")
print("=" * 60)

if not DEEPSEEK_API_KEY:
    print("  ⏭️  跳过：未配置 DEEPSEEK_API_KEY 环境变量")
    print("     设置方式：在 .env 中添加 DEEPSEEK_API_KEY=sk-xxx")
else:
    reset_metrics()

    # 5.1 测试单函数 _repair_with_deepseek
    broken = "{plan: [关键词1, 关键词2]}"
    print(f"  5.1 修复前: {broken}")
    fixed = _repair_with_deepseek(broken)
    print(f"      修复后: {fixed}")

    if fixed:
        # 5.2 通过 robust_json_parse 完整流程测试
        reset_metrics()
        result = robust_json_parse(broken)
        print(f"  5.2 robust_json_parse 完整流程 → {result}")
        assert_metrics(
            "DeepSeek LLM 修复",
            expected_success=1,
            expected_fallback=0,
            expected_deepseek=1,
        )

    # 5.3 更复杂的损坏场景
    reset_metrics()
    broken2 = "the plan is keyword1 and keyword2"
    print(f"  5.3 自然语言文本: {broken2}")
    result = robust_json_parse(broken2)
    print(f"      解析结果: {result}")
    print_metrics("  最终")

# ==================== 测试 6: 各策略独立函数验证 ====================
print("\n" + "=" * 60)
print("测试 6: 独立策略函数单元验证")
print("=" * 60)

# 6.1 _parse_direct
assert _parse_direct('{"a": 1}') == {"a": 1}
assert _parse_direct("not json") is None
assert _parse_direct('["list"]') is None  # 非字典
print("  6.1 _parse_direct  ✅")

# 6.2 _parse_regex_nested
assert _parse_regex_nested('x {"a": 1} y') == {"a": 1}
assert _parse_regex_nested('x {"a": {"b": 2}} y') == {"a": {"b": 2}}
assert _parse_regex_nested("no braces") is None
print("  6.2 _parse_regex_nested  ✅")

# 6.3 _parse_regex_greedy
assert _parse_regex_greedy('x {"a": 1} y') == {"a": 1}
assert _parse_regex_greedy("x\ny\n{\n  'key': 'val'\n}\nz") is None  # 单引号非法
print("  6.3 _parse_regex_greedy  ✅")

# 6.4 _repair_with_deepseek (仅验证函数存在，不实际调用)
assert callable(_repair_with_deepseek)
print("  6.4 _repair_with_deepseek 可调用  ✅")

# ==================== 测试完成 ====================
print("\n" + "=" * 60)
print("所有 json_parser 测试完成")
print("=" * 60)
print("\nMetrics 最终状态:")
print(f"  直接解析成功: {metrics.parse_success}")
print(f"  正则回退成功: {metrics.parse_fallback}")
print(f"  DeepSeek回退:  {metrics.parse_deepseek_fallback}")

# 清理测试产生的 metrics 文件
if metrics.data_path.exists():
    metrics.data_path.unlink()
    print(f"\n已清理 metrics 文件: {metrics.data_path}")
