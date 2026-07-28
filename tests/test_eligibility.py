import json

import numpy as np
import pandas as pd
import pytest

from quant_platform.eligibility import EligibilityPolicy, build_point_in_time_eligibility
from quant_platform.strategy_backtest import build_governed_signal

pytestmark = pytest.mark.no_database


def _inputs():
    dates = pd.date_range("2025-01-02", periods=80, freq="B")
    market = pd.DataFrame(
        {
            "datetime": dates,
            "instrument": "SH600000",
            "amount": 600_000_000.0,
            "paused": False,
        }
    )
    return dates, {
        "market": market,
        "listings": pd.DataFrame(
            [
                {
                    "instrument": "SH600000",
                    "list_date": dates[0],
                    "delist_date": pd.NaT,
                }
            ]
        ),
        "st_intervals": pd.DataFrame(
            columns=["instrument", "start_date", "end_date", "is_st"]
        ),
        "suspensions": pd.DataFrame(columns=["datetime", "instrument", "suspended"]),
        "financials": pd.DataFrame(
            [
                {
                    "instrument": "SH600000",
                    "announcement_date": dates[10],
                    "equity": 10_000_000.0,
                }
            ]
        ),
        "audits": pd.DataFrame(
            [
                {
                    "instrument": "SH600000",
                    "announcement_date": dates[10],
                    "audit_opinion": "standard_unqualified",
                }
            ]
        ),
        "regulatory_events": pd.DataFrame(
            columns=["instrument", "event_date", "known_date", "major"]
        ),
    }


def test_announcement_date_is_not_usable_until_the_following_day() -> None:
    dates, inputs = _inputs()
    result = build_point_in_time_eligibility(
        **inputs,
        policy=EligibilityPolicy(min_listing_trading_days=1, liquidity_lookback_days=2),
    ).set_index("datetime")
    on_announcement = json.loads(result.loc[dates[10], "reasons"])
    after_announcement = json.loads(result.loc[dates[11], "reasons"])
    assert "negative_or_missing_equity" in on_announcement
    assert "nonstandard_or_missing_audit" in on_announcement
    assert "negative_or_missing_equity" not in after_announcement
    assert result.loc[dates[11], "financial_announcement_date"] == dates[10]


def test_new_stock_st_suspension_and_liquidity_are_point_in_time_filters() -> None:
    dates, inputs = _inputs()
    inputs["st_intervals"] = pd.DataFrame(
        [
            {
                "instrument": "SH600000",
                "start_date": dates[65],
                "end_date": dates[66],
                "is_st": True,
            }
        ]
    )
    inputs["suspensions"] = pd.DataFrame(
        [{"datetime": dates[67], "instrument": "SH600000", "suspended": True}]
    )
    inputs["market"].loc[inputs["market"]["datetime"] >= dates[68], "amount"] = 1.0
    result = build_point_in_time_eligibility(**inputs).set_index("datetime")
    assert "new_listing" in json.loads(result.loc[dates[58], "reasons"])
    assert result.loc[dates[59], "eligible"]
    assert "st" in json.loads(result.loc[dates[65], "reasons"])
    assert "suspended" in json.loads(result.loc[dates[67], "reasons"])
    assert "insufficient_liquidity" in json.loads(result.loc[dates[-1], "reasons"])


def test_delisting_does_not_leak_backwards() -> None:
    dates, inputs = _inputs()
    inputs["listings"].loc[0, "delist_date"] = dates[70]
    result = build_point_in_time_eligibility(**inputs).set_index("datetime")
    assert result.loc[dates[69], "delisted"] is False or not result.loc[dates[69], "delisted"]
    assert result.loc[dates[70], "delisted"]
    assert "abnormal_listing" in json.loads(result.loc[dates[70], "reasons"])


def test_regulatory_event_applies_only_from_known_date_and_missing_source_fails_closed() -> None:
    dates, inputs = _inputs()
    inputs["regulatory_events"] = pd.DataFrame(
        [
            {
                "instrument": "SH600000",
                "event_date": dates[50],
                "known_date": dates[62],
                "major": True,
            }
        ]
    )
    result = build_point_in_time_eligibility(**inputs).set_index("datetime")
    assert "major_violation" not in json.loads(result.loc[dates[61], "reasons"])
    assert "major_violation" in json.loads(result.loc[dates[62], "reasons"])

    inputs["regulatory_events"] = None
    missing = build_point_in_time_eligibility(
        **inputs, policy=EligibilityPolicy(require_regulatory_events=True)
    )
    assert not missing["eligible"].any()
    assert all("regulatory_data_missing" in json.loads(value) for value in missing["reasons"])


def test_governed_signal_cannot_select_an_ineligible_high_score() -> None:
    timestamp = pd.Timestamp("2025-06-03")
    scores = pd.Series(
        [2.0, 1.0],
        index=pd.MultiIndex.from_tuples(
            [(timestamp, "SH600000"), (timestamp, "SZ000001")],
            names=["datetime", "instrument"],
        ),
    )
    eligibility = pd.DataFrame(
        [
            {
                "datetime": timestamp,
                "instrument": "SH600000",
                "eligible": False,
                "contract_version": "ashare-point-in-time-eligibility-v1",
            },
            {
                "datetime": timestamp,
                "instrument": "SZ000001",
                "eligible": True,
                "contract_version": "ashare-point-in-time-eligibility-v1",
            },
        ]
    )
    result = build_governed_signal(scores, topk=1, eligibility_matrix=eligibility)
    assert result.index.get_level_values("instrument").tolist() == ["SZ000001"]


def test_governed_signal_neutralizes_point_in_time_industry_bias() -> None:
    timestamp = pd.Timestamp("2026-07-10")
    instruments = [
        "SH600001",
        "SH600002",
        "SH600003",
        "SZ000001",
        "SZ000002",
        "SZ000003",
    ]
    scores = pd.Series(
        [100.0, 90.0, 80.0, 3.0, 2.0, 1.0],
        index=pd.MultiIndex.from_product(
            [[timestamp], instruments], names=["datetime", "instrument"]
        ),
    )
    memberships = pd.DataFrame(
        {
            "instrument": instruments,
            "industry": ["bank", "bank", "bank", "technology", "technology", "technology"],
            "in_date": [pd.Timestamp("2020-01-01")] * 6,
            "out_date": [pd.NaT] * 6,
        }
    )

    governed = build_governed_signal(
        scores,
        topk=6,
        industry_memberships=memberships,
        max_industry_weight=1.0,
        max_industry_deviation=1.0,
    ).droplevel("datetime")
    by_industry = memberships.set_index("instrument")["industry"]

    assert governed.groupby(by_industry).mean().abs().max() < 1e-10


def test_governed_signal_drops_missing_required_style_exposures() -> None:
    timestamp = pd.Timestamp("2026-07-10")
    instruments = [f"SH{600000 + index:06d}" for index in range(6)]
    scores = pd.Series(
        np.arange(6, dtype=float),
        index=pd.MultiIndex.from_product(
            [[timestamp], instruments], names=["datetime", "instrument"]
        ),
    )
    styles = pd.DataFrame(
        {
            "datetime": [timestamp] * 6,
            "instrument": instruments,
            "size": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            # An unused, unavailable style must not erase valid size evidence.
            "growth": [np.nan] * 6,
        }
    )

    governed = build_governed_signal(
        scores,
        topk=5,
        style_exposures=styles,
        neutralize_industry=False,
        neutralize_style_columns=("size",),
    )

    assert "SH600005" not in set(governed.index.get_level_values("instrument"))
    assert len(governed) == 5
