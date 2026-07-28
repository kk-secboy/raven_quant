from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.pair_trading import (
    PairTradingConfig,
    evaluate_pair,
    run_pair_backtest,
    run_pair_robustness_suite,
)

pytestmark = pytest.mark.no_database


def _markets(
    *, shortable: bool = True, include_minutes: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random = np.random.default_rng(17)
    dates = pd.bdate_range("2024-01-02", periods=150)
    x_log = np.log(20.0) + np.cumsum(random.normal(0.0, 0.006, len(dates)))
    spread = np.zeros(len(dates))
    for index in range(1, len(dates)):
        spread[index] = 0.72 * spread[index - 1] + random.normal(0.0, 0.012)
    for index, shock in ((70, 0.10), (95, -0.11), (120, 0.09)):
        spread[index : index + 3] += np.array([shock, shock * 0.6, shock * 0.3])
    y_price = np.exp(0.15 + 0.95 * x_log + spread)
    x_price = np.exp(x_log)
    rows: list[dict] = []
    minute_rows: list[dict] = []
    for date, y_value, x_value in zip(dates, y_price, x_price, strict=True):
        for instrument, price in (("SH510300", y_value), ("SZ159919", x_value)):
            rows.append(
                {
                    "datetime": date,
                    "instrument": instrument,
                    "open": price * 0.999,
                    "close": price,
                    "volume": 20_000_000.0,
                    "amount": price * 20_000_000.0,
                    "paused": 0,
                    "up_limit": price * 1.10,
                    "down_limit": price * 0.90,
                    "shortable": shortable,
                }
            )
            if include_minutes:
                for clock, multiplier in (("10:00", 0.999), ("10:40", 1.001), ("14:00", 1.0)):
                    timestamp = pd.Timestamp(f"{date.date()} {clock}")
                    minute_price = price * multiplier
                    minute_rows.append(
                        {
                            "datetime": timestamp,
                            "instrument": instrument,
                            "close": minute_price,
                            "volume": 5_000_000.0,
                            "amount": minute_price * 5_000_000.0,
                        }
                    )
    daily = pd.DataFrame(rows).set_index(["datetime", "instrument"])
    minute = pd.DataFrame(
        minute_rows,
        columns=["datetime", "instrument", "close", "volume", "amount"],
    ).set_index(["datetime", "instrument"])
    return daily, minute


def _config(**overrides) -> PairTradingConfig:
    values = {
        "formation_window": 40,
        "min_correlation": 0.0,
        "max_cointegration_pvalue": 1.0,
        "cointegration_recheck_days": 5,
        "entry_zscore": 1.2,
        "exit_zscore": 0.4,
        "stop_zscore": 3.5,
        "max_holding_days": 5,
        "pair_gross_fraction": 0.10,
        "max_volume_participation": 0.02,
        "min_backtest_days": 60,
        "min_closed_trades": 1,
    }
    values.update(overrides)
    return PairTradingConfig(**values)


def test_pair_evidence_reports_cointegration_and_hedge_ratio() -> None:
    daily, _ = _markets()
    close = daily["close"].unstack("instrument")
    evidence = evaluate_pair(close["SH510300"], close["SZ159919"], min_observations=60)
    assert evidence.observations == 150
    assert 0.5 < evidence.hedge_ratio < 1.5
    assert 0 <= evidence.cointegration_pvalue <= 1
    assert evidence.half_life_days is not None


def test_pair_backtest_uses_minute_execution_costs_and_atomic_legs() -> None:
    daily, minute = _markets()
    result = run_pair_backtest(
        daily,
        minute,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=_config(),
    )
    metrics = result["metrics"]
    assert metrics["backtest_engine"] == "quantlab_pair"
    assert metrics["pair_native_backtest"] is True
    assert metrics["trade_count"] >= 2
    assert metrics["closed_trade_count"] >= 1
    assert metrics["minute_execution_enforced"] is True
    assert metrics["shortability_enforced"] is True
    assert metrics["borrow_cost_enforced"] is True
    assert metrics["capacity_fill_ratio"] >= 0.95
    assert all(len(item["orders"]) == 2 for item in result["trades"])
    assert (result["daily"]["nav"] > 0).all()


def test_pair_backtest_rejects_unapproved_short_borrow_atomically() -> None:
    daily, minute = _markets(shortable=False)
    result = run_pair_backtest(
        daily,
        minute,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=_config(),
    )
    assert result["metrics"]["trade_count"] == 0
    assert result["metrics"]["rejected_signal_count"] > 0
    assert any("short_borrow_not_authorized" in item["reason"] for item in result["rejections"])


def test_pair_backtest_treats_unknown_shortability_as_not_authorized() -> None:
    daily, minute = _markets()
    daily["shortable"] = np.nan
    result = run_pair_backtest(
        daily,
        minute,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=_config(),
    )
    assert result["metrics"]["trade_count"] == 0
    assert any("short_borrow_not_authorized" in item["reason"] for item in result["rejections"])


def test_pair_backtest_rejects_missing_minute_execution_evidence() -> None:
    daily, minute = _markets(include_minutes=False)
    result = run_pair_backtest(
        daily,
        minute,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=_config(),
    )
    assert result["metrics"]["trade_count"] == 0
    assert any(
        "missing_valid_minute_execution_window" in item["reason"]
        for item in result["rejections"]
    )


def test_pair_entry_sizing_does_not_use_execution_day_close() -> None:
    daily, minute = _markets()
    baseline = run_pair_backtest(
        daily,
        minute,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=_config(),
    )
    first_entry = next(item for item in baseline["trades"] if item["action"] == "entry")
    trade_date = pd.Timestamp(first_entry["trade_date"]).tz_localize(None)
    changed = daily.copy()
    for instrument in ("SH510300", "SZ159919"):
        changed.loc[(trade_date, instrument), "close"] *= 2.0

    repeated = run_pair_backtest(
        changed,
        minute,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=_config(),
    )
    repeated_entry = next(item for item in repeated["trades"] if item["action"] == "entry")

    assert repeated_entry["signal_date"] == first_entry["signal_date"]
    assert [item["requested_delta"] for item in repeated_entry["orders"]] == [
        item["requested_delta"] for item in first_entry["orders"]
    ]


def test_pair_capacity_uses_one_common_partial_fill_ratio() -> None:
    daily, minute = _markets()
    config = _config(min_capacity_fill_ratio=0.80)
    baseline = run_pair_backtest(
        daily,
        minute,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=config,
    )
    entry = next(item for item in baseline["trades"] if item["action"] == "entry")
    trade_date = pd.Timestamp(entry["trade_date"]).tz_localize(None)
    constrained = minute.copy()
    for order in entry["orders"]:
        instrument = order["instrument"]
        mask = (
            constrained.index.get_level_values("datetime").normalize() == trade_date
        ) & (constrained.index.get_level_values("instrument") == instrument)
        bar_count = int(mask.sum())
        constrained.loc[mask, "volume"] = (
            abs(order["requested_delta"])
            * 0.90
            / config.max_volume_participation
            / bar_count
        )
        constrained.loc[mask, "amount"] = (
            constrained.loc[mask, "close"] * constrained.loc[mask, "volume"]
        )

    result = run_pair_backtest(
        daily,
        constrained,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=config,
    )
    constrained_entry = next(item for item in result["trades"] if item["action"] == "entry")
    ratios = [item["fill_ratio"] for item in constrained_entry["orders"]]

    assert max(ratios) - min(ratios) <= 0.01
    assert all(abs(item["delta"]) <= item["capacity"] for item in constrained_entry["orders"])
    assert any(
        abs(item["delta"]) < abs(item["requested_delta"])
        for item in constrained_entry["orders"]
    )


def test_pair_partial_exit_keeps_position_open_until_both_legs_are_flat() -> None:
    daily, minute = _markets()
    config = _config(min_capacity_fill_ratio=0.80)
    baseline = run_pair_backtest(
        daily,
        minute,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=config,
    )
    first_exit = next(item for item in baseline["trades"] if item["action"] == "exit")
    exit_date = pd.Timestamp(first_exit["trade_date"]).tz_localize(None)
    constrained = minute.copy()
    for order in first_exit["orders"]:
        instrument = order["instrument"]
        mask = (
            constrained.index.get_level_values("datetime").normalize() == exit_date
        ) & (constrained.index.get_level_values("instrument") == instrument)
        bar_count = int(mask.sum())
        constrained.loc[mask, "volume"] = (
            abs(order["requested_delta"])
            * 0.90
            / config.max_volume_participation
            / bar_count
        )
        constrained.loc[mask, "amount"] = (
            constrained.loc[mask, "close"] * constrained.loc[mask, "volume"]
        )

    result = run_pair_backtest(
        daily,
        constrained,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=config,
    )
    partial_exit = next(
        item
        for item in result["trades"]
        if item["action"] == "exit"
        and pd.Timestamp(item["trade_date"]).tz_localize(None) == exit_date
    )
    exit_state = result["daily"].loc[exit_date]

    assert any(
        abs(item["delta"]) < abs(item["requested_delta"])
        for item in partial_exit["orders"]
    )
    assert exit_state["position"] != 0
    assert exit_state["quantity_y"] != 0
    assert exit_state["quantity_x"] != 0


def test_pair_config_fails_closed_on_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="exit < entry < stop"):
        PairTradingConfig(exit_zscore=2.0, entry_zscore=1.5, stop_zscore=3.0)


def test_pair_robustness_suite_runs_cost_and_parameter_stress() -> None:
    daily, minute = _markets()
    result = run_pair_robustness_suite(
        daily,
        minute,
        leg_y="SH510300",
        leg_x="SZ159919",
        config=_config(
            min_sharpe_ratio=-5.0,
            min_rolling_cointegration_pass_rate=0.0,
        ),
    )
    assert result["scenario_count"] == 4
    assert {item["name"] for item in result["scenarios"]} == {
        "base",
        "double_costs",
        "lower_entry",
        "higher_entry",
    }
