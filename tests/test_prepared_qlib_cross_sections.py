from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from quant_platform.eligibility import (
    ELIGIBILITY_CONTRACT_VERSION,
    PreparedPointInTimeRiskStates,
    project_point_in_time_risk_states,
)
from scripts import run_multifactor_backtest as backtest
from scripts.run_multifactor_backtest import _PreparedQlibCrossSections, _qlib_cross_section

pytestmark = pytest.mark.no_database


def _frame(timezone: str | None = None) -> pd.DataFrame:
    dates = pd.DatetimeIndex([
        "2024-01-04 09:30", "2024-01-02", "2024-01-04 09:30",
        "2024-01-03", "2024-01-04 09:30", "2024-01-04 09:31",
    ], tz=timezone)
    return pd.DataFrame(
        {"$open": ["4", "invalid", "5", None, np.inf, "-0"],
         "$close": pd.array([4, 1, None, 3, -np.inf, 6], dtype="Float32"),
         "$amount": [0.0, -0.0, np.nan, 30.0, 40.0, 50.0]},
        index=pd.MultiIndex.from_arrays(
            [[2, "A", "A", "B", "A", "A"], dates],
            names=["instrument", "datetime"],
        ),
    )


def _compare_call(frame: pd.DataFrame, prepared: _PreparedQlibCrossSections, when, column):
    try:
        expected = _qlib_cross_section(frame, when, column)
    except Exception as exc:
        with pytest.raises(type(exc)) as caught:
            prepared.lookup(when, column)
        assert str(caught.value) == str(exc)
    else:
        actual = prepared.lookup(when, column)
        assert_series_equal(actual, expected, check_exact=True)
        assert np.array_equal(np.signbit(actual.to_numpy()), np.signbit(expected.to_numpy()))


@pytest.mark.parametrize("timezone", [None, "Asia/Shanghai", "America/New_York"])
def test_prepared_lookup_exactly_matches_legacy_dates_values_and_duplicates(timezone):
    frame = _frame(timezone)
    original = frame.copy(deep=True)
    prepared = _PreparedQlibCrossSections(frame)
    # Deliberately query out of order, including duplicates, gaps, intraday
    # points, before-first/after-last, timezone errors and NaT slice behavior.
    queries = [pd.Timestamp(value) for value in (
        "2024-01-04 09:30", "2024-01-03 16:00", "2024-01-01", "2025-01-01",
        "2024-01-02", "2024-01-04 09:30:30", "2024-01-04 09:31",
    )] + [pd.Timestamp("2024-01-03", tz="Asia/Shanghai"), pd.NaT]
    for when in queries:
        for column in ("$open", "$close", "$amount", "missing"):
            _compare_call(frame, prepared, when, column)
    assert_frame_equal(frame, original, check_exact=True)


@pytest.mark.parametrize("empty", [False, True])
def test_missing_and_empty_datetime_indexes_keep_legacy_failure(empty):
    frame = _frame()
    if empty:
        frame = frame.iloc[:0]
    else:
        dates = frame.index.get_level_values("datetime").to_list()
        dates[0] = pd.NaT
        frame.index = pd.MultiIndex.from_arrays(
            [frame.index.get_level_values("instrument"), dates],
            names=["instrument", "datetime"],
        )
    prepared = _PreparedQlibCrossSections(frame)
    for when in (pd.Timestamp("2024-01-03"), pd.NaT):
        for column in ("$open", "missing"):
            _compare_call(frame, prepared, when, column)


def test_prepared_panel_is_private_and_queries_do_not_sort_or_copy_the_full_frame():
    frame = _frame()
    when = pd.Timestamp("2024-01-04 09:30")
    expected = _qlib_cross_section(frame, when, "$amount")
    prepared = _PreparedQlibCrossSections(frame)
    frame.iloc[:, :] = 999.0
    with patch.object(pd.DataFrame, "copy", side_effect=AssertionError("per-query frame copy")), \
            patch.object(pd.DataFrame, "sort_index", side_effect=AssertionError("per-query sort")):
        result = prepared.lookup(when, "$amount")
        assert_series_equal(result, expected, check_exact=True)
        result.iloc[:] = 123.0
        assert_series_equal(prepared.lookup(when, "$amount"), expected, check_exact=True)


@pytest.mark.parametrize("intraday", [False, True])
def test_metadata_provider_prepares_each_panel_once_and_matches_legacy(monkeypatch, intraday):
    dates = pd.date_range("2024-01-02", periods=2)
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
    execution = pd.DataFrame({
        "$open": [1.0, 2.0, 3.0, 4.0], "$close": [1.5, 2.5, 3.5, 4.5],
        "$open/$factor": [2.0, 4.0, 6.0, 8.0],
        "$close/$factor": [3.0, 5.0, 7.0, 9.0], "$factor": [0.5] * 4,
        "Ref(Mean($amount, 20), 1)": [100.0, 200.0, 300.0, np.nan],
    }, index=index)
    minute = pd.DataFrame({"$vwap": [10.0, 20.0], "$close": [11.0, 21.0]},
        index=pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-03 09:30")], ["A", "B"]],
            names=["datetime", "instrument"],
        )) if intraday else None
    eligibility = pd.DataFrame([
        {"datetime": day, "instrument": instrument, "eligible": True, "reasons": "[]",
         "is_st": False, "delisted": False, "normal_listing_status": True,
         "suspended": False, "equity": np.nan, "audit_opinion": np.nan,
         "financial_gate_required": False, "regulatory_data_available": True,
         "major_violation": False, "contract_version": ELIGIBILITY_CONTRACT_VERSION}
        for day, instrument in index
    ])
    memberships = pd.DataFrame({
        "instrument": ["A", "B"], "in_date": [dates[0], dates[0]],
        "out_date": [pd.NaT, pd.NaT], "industry": ["one", "two"],
    })
    monkeypatch.setattr(backtest, "filter_available", lambda _name, frame, _when: frame)
    args = (memberships, None, pd.DataFrame(), eligibility, execution, execution, None)
    kwargs = {"strategy_config": {"portfolio_construction": "topk_equal_weight"},
              "intraday_prices": minute}
    instances = []
    risk_instances = []

    def prepare(frame):
        instance = _PreparedQlibCrossSections(frame)
        instances.append(instance)
        return instance

    def prepare_risk(frame):
        instance = PreparedPointInTimeRiskStates(frame)
        risk_instances.append(instance)
        return instance

    monkeypatch.setattr(backtest, "_PreparedQlibCrossSections", prepare)
    monkeypatch.setattr(backtest, "PreparedPointInTimeRiskStates", prepare_risk)
    actual_provider = backtest._metadata_provider(*args, **kwargs)
    assert len(instances) == (2 if intraday else 1)
    assert len(risk_instances) == 1

    class LegacyLookup:
        def __init__(self, frame):
            self.frame = frame

        def lookup(self, when, column):
            return _qlib_cross_section(self.frame, when, column)

    class LegacyRiskLookup:
        def __init__(self, frame):
            self.frame = frame

        def project(self, *, as_of, instruments):
            return project_point_in_time_risk_states(
                self.frame, as_of=as_of, instruments=instruments,
            )

    monkeypatch.setattr(backtest, "_PreparedQlibCrossSections", LegacyLookup)
    monkeypatch.setattr(backtest, "PreparedPointInTimeRiskStates", LegacyRiskLookup)
    expected_provider = backtest._metadata_provider(*args, **kwargs)
    queries = [pd.Timestamp("2024-01-03 09:30"), pd.Timestamp("2024-01-03 09:31")]
    if not intraday:
        queries.extend(reversed(dates.to_list()))
    for when in queries:
        mode = {"execution_quantity_mode": "minute_raw" if intraday else "daily_adjusted"}
        actual = actual_provider(when, pd.Index(["B", "A"]), **mode)
        expected = expected_provider(when, pd.Index(["B", "A"]), **mode)
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            if isinstance(value, pd.Series):
                assert_series_equal(actual[name], value, check_exact=True)
            elif isinstance(value, pd.DataFrame):
                assert_frame_equal(actual[name], value, check_exact=True)
            else:
                assert actual[name] == value
    assert len(instances) == (2 if intraday else 1)
    assert len(risk_instances) == 1
