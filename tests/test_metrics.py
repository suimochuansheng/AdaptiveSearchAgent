"""Metrics 指标统计模块单元测试 — 覆盖计数逻辑和成功率计算。"""

from pathlib import Path

from src.utils.metrics import Metrics

# =============================================================================
# record_parse — 各策略计数
# =============================================================================


def test_record_parse_increments_total(tmp_path: Path) -> None:
    """每次调用 record_parse 应使 parse_total += 1。"""
    m = Metrics(data_path=tmp_path / "metrics.json")
    assert m.parse_total == 0

    m.record_parse(success=True)
    m.record_parse(success=True)
    m.record_parse(success=False)

    assert m.parse_total == 3


def test_record_parse_strategy1(tmp_path: Path) -> None:
    """策略1成功：仅 parse_success 递增。"""
    m = Metrics(data_path=tmp_path / "metrics.json")
    m.record_parse(success=True)
    assert m.parse_total == 1
    assert m.parse_success == 1
    assert m.parse_fallback == 0
    assert m.parse_deepseek_fallback == 0


def test_record_parse_strategy2_or_3(tmp_path: Path) -> None:
    """策略2/3成功：parse_success 和 parse_fallback 同时递增。"""
    m = Metrics(data_path=tmp_path / "metrics.json")
    m.record_parse(success=True, fallback=True)
    assert m.parse_total == 1
    assert m.parse_success == 1
    assert m.parse_fallback == 1
    assert m.parse_deepseek_fallback == 0


def test_record_parse_strategy4(tmp_path: Path) -> None:
    """策略4成功：parse_success 和 parse_deepseek_fallback 同时递增。"""
    m = Metrics(data_path=tmp_path / "metrics.json")
    m.record_parse(success=True, deepseek_fallback=True)
    assert m.parse_total == 1
    assert m.parse_success == 1
    assert m.parse_fallback == 0
    assert m.parse_deepseek_fallback == 1


def test_record_parse_failure(tmp_path: Path) -> None:
    """解析失败：仅 parse_total 递增，其他计数器不变。"""
    m = Metrics(data_path=tmp_path / "metrics.json")
    m.record_parse(success=False)
    assert m.parse_total == 1
    assert m.parse_success == 0
    assert m.parse_fallback == 0
    assert m.parse_deepseek_fallback == 0


# =============================================================================
# get_parse_success_rate — 成功率计算
# =============================================================================


def test_rate_all_success(tmp_path: Path) -> None:
    """10次全部成功 → 1.0。"""
    m = Metrics(data_path=tmp_path / "metrics.json")
    for _ in range(10):
        m.record_parse(success=True)
    assert m.get_parse_success_rate() == 1.0


def test_rate_partial(tmp_path: Path) -> None:
    """8次成功 + 2次失败 → 0.8。"""
    m = Metrics(data_path=tmp_path / "metrics.json")
    for _ in range(8):
        m.record_parse(success=True)
    for _ in range(2):
        m.record_parse(success=False)
    assert m.get_parse_success_rate() == 0.8


def test_rate_zero_calls(tmp_path: Path) -> None:
    """无调用 → 0.0（避免除零）。"""
    m = Metrics(data_path=tmp_path / "metrics.json")
    assert m.get_parse_success_rate() == 0.0


# =============================================================================
# record_iterations — 迭代深度记录
# =============================================================================


def test_record_iterations(tmp_path: Path) -> None:
    """每次查询的迭代次数应追加到列表。"""
    m = Metrics(data_path=tmp_path / "metrics.json")
    m.record_iterations(1)
    m.record_iterations(3)
    m.record_iterations(2)
    assert m.iterations_per_query == [1, 3, 2]


# =============================================================================
# 持久化 — save / load 往返
# =============================================================================


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """保存后再加载，计数器值应一致。"""
    path = tmp_path / "metrics.json"
    m1 = Metrics(data_path=path)
    m1.record_parse(success=True)
    m1.record_parse(success=True, fallback=True)
    m1.record_parse(success=False)
    m1.record_iterations(2)
    m1.save()

    m2 = Metrics(data_path=path)
    assert m2.parse_total == 3
    assert m2.parse_success == 2
    assert m2.parse_fallback == 1
    assert m2.parse_deepseek_fallback == 0
    assert m2.iterations_per_query == [2]
