"""Past-only quantity factors for holdings without current quotes."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from quant_platform.eligibility import ELIGIBILITY_CONTRACT_VERSION
from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig
from quant_platform.qlib_policy_strategy import qlib_execution_lot_context
from scripts import run_multifactor_backtest as backtest

pytestmark = pytest.mark.no_database


def test_factor_history_is_per_instrument_past_only_with_no_future_backfill():
    dates = pd.date_range("2024-01-02", periods=4, tz="Asia/Shanghai")
    frame = pd.DataFrame(
        {"$factor": np.array([0.5, np.nan, np.nan, 0.25, 0.6, 0, np.inf, -1], dtype=np.float32)},
        index=pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"]),
    ).sample(frac=1, random_state=4)
    original = frame.copy(deep=True)
    lookup = backtest._PreparedQlibFactorHistory(frame)
    # Query latest first: a stateful last-call cache would leak these values backwards.
    for when, expected in (
        ("2024-01-05", {"A": float(np.float32(0.6)), "B": 0.25}),
        ("2024-01-02", {"A": 0.5, "B": np.nan}),
        ("2024-01-03", {"A": 0.5, "B": 0.25}),
        ("2024-01-01", {"A": np.nan, "B": np.nan}),
    ):
        actual = lookup.lookup(pd.Timestamp(when), pd.Index(["B", "A", "missing"]))
        target = pd.Series(expected, dtype=float).reindex(actual.index)
        assert_series_equal(actual, target, check_names=False, check_exact=True)
    assert_frame_equal(frame, original, check_exact=True)


def test_factor_history_handles_missing_rows_and_retains_float32_without_aliasing():
    index = pd.MultiIndex.from_tuples([
        ("A", pd.Timestamp("2024-01-02")), ("B", pd.Timestamp("2024-01-03")),
        ("A", pd.Timestamp("2024-01-05")),
    ], names=["instrument", "datetime"])
    frame = pd.DataFrame({"$factor": np.array([0.125, 0.25, 0.5], dtype=np.float32)}, index=index)
    lookup = backtest._PreparedQlibFactorHistory(frame)
    assert all(dtype == np.dtype("float32") for dtype in lookup._values.dtypes)
    frame.iloc[:] = 999
    actual = lookup.lookup(pd.Timestamp("2024-01-04 23:59"), pd.Index(["B", "A"]))
    assert actual.to_dict() == {"B": 0.25, "A": 0.125}
    actual.iloc[:] = 999
    assert lookup.lookup(pd.Timestamp("2024-01-04"), pd.Index(["A"]))["A"] == 0.125


def test_factor_history_rejects_duplicate_or_unknown_times():
    index = pd.MultiIndex.from_tuples([
        (pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-02"), "A"),
    ], names=["datetime", "instrument"])
    frame = pd.DataFrame({"$factor": [0.5, 0.6]}, index=index)
    with pytest.raises(ValueError):
        backtest._PreparedQlibFactorHistory(frame)
    frame.index = pd.MultiIndex.from_tuples([(pd.NaT, "A"), (pd.Timestamp("2024-01-02"), "B")],
                                           names=["datetime", "instrument"])
    with pytest.raises(ValueError):
        backtest._PreparedQlibFactorHistory(frame)


def test_real_20160105_quote_gap_uses_only_last_valid_20160104_factor():
    # Captured from the failed private IS input, not a fabricated suspension flag:
    # SZ002308 factor/open/close/volume are all missing on January 5 and 6.
    dates = pd.date_range("2016-01-04", periods=3)
    factor = float(np.float32(0.1260743886232376))
    frame = pd.DataFrame({
        "$factor": np.array([factor, np.nan, np.nan], dtype=np.float32),
        "$open": np.array([3.1014301777, np.nan, np.nan], dtype=np.float32),
        "$close": np.array([2.7975907326, np.nan, np.nan], dtype=np.float32),
        "$volume": np.array([131309040.0, np.nan, np.nan], dtype=np.float32),
    }, index=pd.MultiIndex.from_product([dates, ["SZ002308"]], names=["datetime", "instrument"]))
    original = frame.copy(deep=True)
    quotes = backtest._PreparedQlibCrossSections(frame)
    history = backtest._PreparedQlibFactorHistory(frame)
    with patch.object(pd.DataFrame, "ffill", side_effect=AssertionError("per-query history fill")):
        for when in reversed(dates[1:]):
            assert history.lookup(when, pd.Index(["SZ002308"]))["SZ002308"] == factor
            for column in ("$open", "$close", "$volume"):
                assert pd.isna(quotes.lookup(when, column)["SZ002308"])
    assert_frame_equal(frame, original, check_exact=True)


def _provider(monkeypatch):
    dates = pd.date_range("2016-01-04", periods=3)
    instruments = ["SZ002308", "SH600000"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    factors = np.array([0.125, 0.5, np.nan, 0.5, 0.15, 0.5], dtype=np.float32)
    execution = pd.DataFrame({
        "$factor": factors, "$open/$factor": [15, 10, np.nan, 11, 16, 12],
        "$close/$factor": [15.5, 10.5, np.nan, 11.5, 16.5, 12.5],
        "Ref(Mean($amount, 20), 1)": [1_000_000.0] * 6,
    }, index=index)
    close_history = pd.DataFrame({"$close": [2, 5, np.nan, 5.5, 2.25, 6]}, index=index)
    eligibility = pd.DataFrame([{
        "datetime": day, "instrument": instrument, "eligible": not suspended,
        "reasons": '["suspended"]' if suspended else "[]", "is_st": False,
        "delisted": False, "normal_listing_status": True, "suspended": suspended,
        "equity": np.nan, "audit_opinion": np.nan, "financial_gate_required": False,
        "regulatory_data_available": True, "major_violation": False,
        "contract_version": ELIGIBILITY_CONTRACT_VERSION,
    } for day, instrument in index
        for suspended in [day == dates[1] and instrument == "SZ002308"]])
    membership = pd.DataFrame({
        "instrument": instruments, "in_date": [dates[0]] * 2,
        "out_date": [pd.NaT] * 2, "industry": ["one", "two"],
    })
    monkeypatch.setattr(backtest, "filter_available", lambda _name, frame, _when: frame)
    provider = backtest._metadata_provider(
        membership, None, pd.DataFrame(), eligibility, execution, close_history, None,
        strategy_config={"portfolio_construction": "topk_equal_weight"},
    )
    return provider, dates, pd.Index(instruments), execution


def test_suspended_holding_uses_past_factor_without_filling_any_quote_or_tradability(monkeypatch):
    provider, dates, instruments, execution = _provider(monkeypatch)
    original = execution.copy(deep=True)
    # Inspect the future resume first; it must not change a preceding suspension query.
    resumed = provider(dates[2], instruments)
    result = provider(dates[1], instruments)
    assert result["qlib_factors"].to_dict() == {"SZ002308": 0.125, "SH600000": 0.5}
    assert resumed["qlib_factors"]["SZ002308"] == float(np.float32(0.15))
    assert pd.isna(result["prices"]["SZ002308"])
    assert pd.isna(result["current_prices"]["SZ002308"])
    assert pd.isna(result["average_daily_values"]["SZ002308"])
    assert result["instrument_risk_states"]["SZ002308"] == "watch"
    assert result["prices"]["SH600000"] == 11.0
    assert result["current_prices"]["SH600000"] == 11.5
    current = SimpleNamespace(
        get_stock_list=lambda: ["SZ002308"], get_stock_amount=lambda _instrument: 123_200.0,
        get_stock_price=lambda _instrument: 1.9375,
    )
    context = qlib_execution_lot_context(
        current, instruments=instruments, factors=result["qlib_factors"], prices=result["prices"],
        current_prices=result["current_prices"], trade_date=dates[2].date(), locked_amounts={},
    )
    assert context.previous_quantities["SZ002308"] == 15_400
    assert context.valuation_prices["SZ002308"] == 15.5
    assert pd.isna(context.execution_prices["SZ002308"])
    policy_inputs = {key: value for key, value in result.items() if key != "qlib_factors"}
    previous = {"SZ002308": 15_400 * 15.5 / 1_000_000}
    decision = PortfolioPolicy(PortfolioPolicyConfig(
        topk=1, n_drop=0, max_position_weight=0.5, max_daily_turnover=1.0,
    )).decide(
        pd.Series({"SZ002308": 1.0, "SH600000": 0.0}), previous,
        **policy_inputs, execution_lot_context=context, portfolio_value=1_000_000,
        holding_age_sessions={"SZ002308": 3},
    )
    assert decision.target_weights == previous
    assert decision.position_state["execution_lot_target_quantities"]["SZ002308"] == 15_400
    assert_frame_equal(execution, original, check_exact=True)
    minute = provider(dates[1], instruments, execution_quantity_mode="minute_raw")
    assert minute["qlib_factors"].eq(1.0).all()
    assert_series_equal(minute["prices"], result["prices"], check_exact=True)
    assert_series_equal(minute["current_prices"], result["current_prices"], check_exact=True)


def test_holding_without_any_past_factor_keeps_existing_fail_closed_adapter(monkeypatch):
    provider, dates, instruments, _execution = _provider(monkeypatch)
    result = provider(dates[1], instruments)
    current = SimpleNamespace(get_stock_list=lambda: ["missing"],
                              get_stock_amount=lambda _: 100.0, get_stock_price=lambda _: 1.0)
    with pytest.raises(ValueError, match="no valid signal factor for missing"):
        qlib_execution_lot_context(
            current, instruments=pd.Index(["missing"]), factors=result["qlib_factors"],
            prices=result["prices"], current_prices=result["current_prices"],
            trade_date=dates[2].date(), locked_amounts={},
        )


def test_metadata_provider_prepares_factor_history_only_once(monkeypatch):
    original = backtest._PreparedQlibFactorHistory
    constructions = []

    def prepare(frame):
        constructions.append(frame)
        return original(frame)

    monkeypatch.setattr(backtest, "_PreparedQlibFactorHistory", prepare)
    provider, dates, instruments, _execution = _provider(monkeypatch)
    for when in (dates[2], dates[0], dates[1], dates[2]):
        provider(when, instruments)
    assert len(constructions) == 1
