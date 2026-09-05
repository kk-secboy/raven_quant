from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant_platform import eligibility

pytestmark = pytest.mark.no_database


def _row(instrument="sh600000", **changes):
    row = {
        "datetime": pd.Timestamp("2025-03-08"), "instrument": instrument,
        "eligible": True, "reasons": "[]", "is_st": False, "suspended": False,
        "delisted": False, "normal_listing_status": True, "equity": 10_000_000.0,
        "audit_opinion": "standard_unqualified", "financial_gate_required": True,
        "regulatory_data_available": True, "major_violation": False,
        "contract_version": eligibility.ELIGIBILITY_CONTRACT_VERSION,
    }
    row.update(changes)
    return row


@pytest.mark.parametrize("timezone", [None, "Asia/Shanghai", "America/New_York"])
def test_prepared_risk_lookup_matches_unchanged_projection_in_any_query_order(timezone):
    dates = pd.date_range("2025-03-06", periods=9, tz=timezone)
    variants = [
        {}, {"suspended": True}, {"eligible": False, "reasons": '["new_listing"]'},
        {"is_st": True, "suspended": True}, {"equity": -1}, {"equity": np.nan},
        {"audit_opinion": "qualified"}, {"audit_opinion": "adverse"},
        {"audit_opinion": None}, {"major_violation": True, "normal_listing_status": False},
        {"delisted": True}, {"reasons": '["unknown_reason", "insufficient_liquidity"]'},
        {"eligible": False}, {"reasons": "not-json"}, {"reasons": "[1]"},
        {"financial_gate_required": False, "equity": None, "audit_opinion": None},
        {"financial_gate_required": False, "reasons": '["negative_or_missing_equity"]'},
        {"financial_gate_required": False, "reasons": '["nonstandard_or_missing_audit"]'},
        {"eligible": "true", "is_st": 1, "suspended": 0},
        {"normal_listing_status": None, "reasons": '["regulatory_data_missing"]'},
    ]
    values = pd.DataFrame([
        _row(f"sh{600000 + i}", datetime=day.replace(hour=12), **variant)
        for i, variant in enumerate(variants)
        for day_index, day in enumerate(dates)
        if (day_index + i) % 3 != 0  # gaps must carry stale evidence, not future rows
    ]).sample(frac=1, random_state=37)
    original = values.copy(deep=True)
    lookup = eligibility.PreparedPointInTimeRiskStates(values)
    queries = [dates[6], pd.Timestamp("2025-03-05", tz=timezone), dates[2], dates[-1], dates[1],
               pd.Timestamp("2025-03-17", tz=timezone), dates[6]]
    for when in queries:
        for requested in [None, [], ["sh600000", "SH600000", "sh600003", "unknown"]]:
            expected = eligibility.project_point_in_time_risk_states(
                values, as_of=when, instruments=requested,
            )
            actual = lookup.project(as_of=when, instruments=requested)
            assert_frame_equal(actual, expected, check_exact=True)
    assert_frame_equal(values, original, check_exact=True)


@pytest.mark.parametrize("empty", [False, True])
def test_prepared_risk_lookup_preserves_cross_timezone_and_nanosecond_date_boundaries(empty):
    values = pd.DataFrame([
        _row(datetime=pd.Timestamp("2025-03-09 16:30", tz="America/New_York")),
    ])
    if empty:
        values = values.iloc[:0]
    lookup = eligibility.PreparedPointInTimeRiskStates(values)
    for when in [
        pd.Timestamp("2025-03-09 00:00", tz="UTC"),
        pd.Timestamp("2025-03-10 01:00", tz="Asia/Shanghai"),
        pd.Timestamp("2025-03-09 23:59:59.999999999", tz="America/New_York"),
    ]:
        assert_frame_equal(
            lookup.project(as_of=when, instruments=["SH600000", "missing"]),
            eligibility.project_point_in_time_risk_states(
                values, as_of=when, instruments=["SH600000", "missing"],
            ), check_exact=True,
        )


@pytest.mark.parametrize("time_unit", ["ns", "us", "ms", "s"])
def test_prepared_risk_lookup_preserves_timestamp_resolution_and_instrument_string_rules(time_unit):
    values = pd.DataFrame([
        _row("sh600000", datetime=datetime(2025, 3, 7, 12)),
        _row(" SH600000 ", datetime=datetime(2025, 3, 8, 9)),
        _row(None, datetime=datetime(2025, 3, 8, 16)),
        _row(np.nan, datetime=datetime(2025, 3, 8, 1)),
    ])
    values["datetime"] = values["datetime"].astype(f"datetime64[{time_unit}]")
    lookup = eligibility.PreparedPointInTimeRiskStates(values)
    requests = ["sh600000", " SH600000 ", None, np.nan, "missing"]
    for when in ["2025-03-08", "2025-03-06", "2025-03-07", "2025-03-09"]:
        assert_frame_equal(
            lookup.project(as_of=when, instruments=iter(requests)),
            eligibility.project_point_in_time_risk_states(
                values, as_of=when, instruments=requests,
            ), check_exact=True,
        )


@pytest.mark.parametrize("bad_input", ["missing_column", "bad_date", "duplicate", "contract"])
def test_prepared_risk_lookup_rejects_invalid_history_even_outside_requested_scope(bad_input):
    values = pd.DataFrame([_row(), _row("other", datetime=pd.Timestamp("2028-01-01"))])
    if bad_input == "missing_column":
        values = values.drop(columns="eligible")
    elif bad_input == "bad_date":
        values.loc[1, "datetime"] = pd.NaT
    elif bad_input == "duplicate":
        values.loc[1, "datetime"] = pd.Timestamp("2025-03-08 23:00")
        values.loc[1, "instrument"] = "SH600000"
    else:
        values.loc[1, "contract_version"] = "old"
    with pytest.raises(ValueError) as expected:
        eligibility.project_point_in_time_risk_states(
            values, as_of="2024-01-01", instruments=["missing"],
        )
    with pytest.raises(ValueError, match=str(expected.value)):
        eligibility.PreparedPointInTimeRiskStates(values)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("when", [pd.NaT, None, "not-a-date", pd.Timestamp("2025-03-08", tz="UTC")])
def test_prepared_risk_lookup_retains_invalid_query_rejection(when, empty):
    values = pd.DataFrame([_row()])
    if empty:
        values = values.iloc[:0]
    lookup = eligibility.PreparedPointInTimeRiskStates(values)
    with pytest.raises((AttributeError, TypeError, ValueError)) as expected:
        eligibility.project_point_in_time_risk_states(values, as_of=when)
    with pytest.raises(type(expected.value)):
        lookup.project(as_of=when)


def test_prepared_risk_lookup_is_frozen_and_does_not_rescan_history(monkeypatch):
    values = pd.DataFrame([
        _row(datetime=day, instrument=f"SH{600000 + i}")
        for i in range(4) for day in pd.date_range("2025-01-01", periods=90)
    ])
    frozen = values.copy()
    lookup = eligibility.PreparedPointInTimeRiskStates(values)
    values.loc[:, "is_st"] = True
    original_projection = eligibility.project_point_in_time_risk_states
    selected_sizes = []

    def bounded_projection(selected, **kwargs):
        selected_sizes.append(len(selected))
        assert len(selected) <= 2
        return original_projection(selected, **kwargs)

    monkeypatch.setattr(eligibility, "project_point_in_time_risk_states", bounded_projection)
    for when in ["2025-03-20", "2025-01-10", "2024-12-01", "2025-04-01"]:
        requested = ["SH600000", "SH600002", "missing"]
        assert_frame_equal(
            lookup.project(as_of=when, instruments=requested),
            original_projection(frozen, as_of=when, instruments=requested), check_exact=True,
        )
    assert selected_sizes == [2, 2, 0, 2]
