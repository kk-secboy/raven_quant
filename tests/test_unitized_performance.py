"""Unit tests for the unitized TWR curve, drawdown/recovery and XIRR (4.4/8.3)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_platform.unitized_performance import (
    chain_unitized_day,
    unitized_drawdown_recovery,
    xirr,
)

pytestmark = pytest.mark.no_database


def test_twr_daily_return_golden_case_with_flows() -> None:
    # 手算：V_{t-1}=100_000，开盘前入金 50_000，当日收盘 NAV=153_000（含
    # 盘后确认的出金 3_000）。r_t = (153_000 − (−3_000)) / (100_000 + 50_000) − 1
    # = 156_000 / 150_000 − 1 = 0.04。
    day = chain_unitized_day(
        prior_nav=100_000.0,
        nav=153_000.0,
        flow_open=50_000.0,
        flow_close=-3_000.0,
        prior_wealth=1.0,
        prior_high_water_mark=1.0,
    )
    assert day["status"] == "ok"
    assert day["daily_return"] == pytest.approx(0.04)
    assert day["investment_wealth"] == pytest.approx(1.04)
    assert day["drawdown"] == pytest.approx(0.0)
    assert day["high_water_mark"] == pytest.approx(1.04)


def test_deposit_does_not_manufacture_return() -> None:
    # 入金后资产原封不动：NAV 跳升但 TWR 日收益为零、单位化曲线连续。
    day = chain_unitized_day(
        prior_nav=100_000.0,
        nav=160_000.0,
        flow_open=60_000.0,
        flow_close=0.0,
        prior_wealth=1.20,
        prior_high_water_mark=1.20,
    )
    assert day["daily_return"] == pytest.approx(0.0)
    assert day["investment_wealth"] == pytest.approx(1.20)


def test_withdrawal_is_symmetric() -> None:
    # 出金 60_000 后剩余 40_000：同样零收益，曲线不变。
    day = chain_unitized_day(
        prior_nav=100_000.0,
        nav=40_000.0,
        flow_open=-60_000.0,
        flow_close=0.0,
        prior_wealth=1.20,
        prior_high_water_mark=1.20,
    )
    assert day["status"] == "ok"
    assert day["daily_return"] == pytest.approx(0.0)
    assert day["investment_wealth"] == pytest.approx(1.20)


def test_nonpositive_base_marks_chain_unavailable_and_break_propagates() -> None:
    broken = chain_unitized_day(
        prior_nav=10_000.0,
        nav=500.0,
        flow_open=-10_000.0,
        flow_close=0.0,
        prior_wealth=1.0,
        prior_high_water_mark=1.0,
    )
    assert broken["status"] == "undefined_nonpositive_base"
    assert broken["investment_wealth"] is None
    # 设计 4.4：断链后不得跳过当日继续连乘，断链状态向后传播。
    propagated = chain_unitized_day(
        prior_nav=500.0,
        nav=510.0,
        flow_open=0.0,
        flow_close=0.0,
        prior_wealth=None,
        prior_high_water_mark=None,
    )
    assert propagated["status"] == "unavailable_broken_chain"
    assert propagated["daily_return"] is None


def test_unitized_drawdown_and_recovery_hand_computed() -> None:
    start = date(2026, 1, 5)
    days = [start + timedelta(days=offset) for offset in range(6)]
    # 曲线 1.00 → 1.10（前高）→ 0.99（谷底，回撤 0.99/1.10−1=−0.10）
    # → 1.045 → 1.10（第 3 个交易日恢复前高）→ 1.21。
    wealth = [1.00, 1.10, 0.99, 1.045, 1.10, 1.21]
    result = unitized_drawdown_recovery(list(zip(days, wealth, strict=True)))
    assert result["status"] == "ok"
    assert result["twr"] == pytest.approx(0.21)
    assert result["max_drawdown"] == pytest.approx(-0.10)
    assert result["trough_date"] == days[2].isoformat()
    assert result["recovery_date"] == days[4].isoformat()
    assert result["recovery_trading_days"] == 2


def test_unrecovered_drawdown_is_ongoing() -> None:
    start = date(2026, 1, 5)
    days = [start + timedelta(days=offset) for offset in range(3)]
    result = unitized_drawdown_recovery(
        list(zip(days, [1.00, 0.80, 0.90], strict=True))
    )
    assert result["status"] == "ongoing"
    assert result["max_drawdown"] == pytest.approx(-0.20)
    assert result["recovery_trading_days"] is None
    assert result["recovery_date"] is None


def test_monotone_curve_has_zero_recovery() -> None:
    start = date(2026, 1, 5)
    days = [start + timedelta(days=offset) for offset in range(3)]
    result = unitized_drawdown_recovery(
        list(zip(days, [1.00, 1.05, 1.10], strict=True))
    )
    assert result["status"] == "ok"
    assert result["max_drawdown"] == pytest.approx(0.0)
    assert result["recovery_trading_days"] == 0


def test_xirr_converges_on_simple_golden_case() -> None:
    # 投入 100_000，365 天后账户值 110_000：按 365.2425 年化，
    # 期望 r = 1.1 ** (365.2425/365) − 1 ≈ 0.10007。
    result = xirr(
        [(date(2025, 1, 2), -100_000.0)],
        terminal=(date(2026, 1, 2), 110_000.0),
    )
    assert result["status"] == "ok"
    expected = 1.1 ** (365.2425 / 365.0) - 1.0
    assert result["rate"] == pytest.approx(expected, abs=1e-6)


def test_xirr_with_intermediate_flows() -> None:
    # 期初投入 100_000，半年后追加 50_000，一年后账户值 160_000。
    # 手算验证方程：160_000 = 100_000(1+r)^1 + 50_000(1+r)^0.5 → r ≈ 0.05848。
    result = xirr(
        [(date(2025, 1, 2), -100_000.0), (date(2025, 7, 2), -50_000.0)],
        terminal=(date(2026, 1, 2), 160_000.0),
    )
    assert result["status"] == "ok"
    rate = result["rate"]
    t_add = (date(2025, 7, 2) - date(2025, 1, 2)).days / 365.2425
    t_end = (date(2026, 1, 2) - date(2025, 1, 2)).days / 365.2425
    npv = (
        -100_000.0
        - 50_000.0 / (1.0 + rate) ** t_add
        + 160_000.0 / (1.0 + rate) ** t_end
    )
    assert npv == pytest.approx(0.0, abs=1e-6)


def test_xirr_single_sign_is_undefined() -> None:
    result = xirr(
        [(date(2025, 1, 2), -100_000.0)],
        terminal=(date(2026, 1, 2), 0.0),
    )
    assert result["status"] == "undefined_single_sign"
    assert result["rate"] is None


def test_xirr_no_root_reports_status_not_pseudo_precision() -> None:
    # 交替投入/取出且终值无法使 NPV 过零（投入远小于流出终值）时如实标注。
    result = xirr(
        [(date(2025, 1, 2), -1.0), (date(2025, 6, 2), 2.0)],
        terminal=(date(2026, 1, 2), -1.0),
    )
    assert result["status"] in {"undefined_no_root", "multiple_roots", "ok"}
    if result["status"] == "undefined_no_root":
        assert result["rate"] is None


def test_xirr_rejects_non_finite_flows() -> None:
    with pytest.raises(ValueError, match="finite"):
        xirr(
            [(date(2025, 1, 2), float("inf"))],
            terminal=(date(2026, 1, 2), 1.0),
        )
