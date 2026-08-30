import json

import numpy as np
import pandas as pd
import pytest

from quant_platform.eligibility import (
    ELIGIBILITY_CONTRACT_VERSION,
    INSTRUMENT_RISK_STATES,
    EligibilityPolicy,
    build_point_in_time_eligibility,
    project_point_in_time_risk_states,
)
from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig
from quant_platform.strategy_backtest import (
    build_governed_signal,
    governed_score_neutralization,
)

pytestmark = pytest.mark.no_database


def _risk_row(instrument: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "datetime": pd.Timestamp("2025-06-03"),
        "instrument": instrument,
        "eligible": True,
        "reasons": "[]",
        "is_st": False,
        "suspended": False,
        "delisted": False,
        "normal_listing_status": True,
        "equity": 10_000_000.0,
        "audit_opinion": "standard_unqualified",
        "financial_gate_required": True,
        "regulatory_data_available": True,
        "major_violation": False,
        "contract_version": ELIGIBILITY_CONTRACT_VERSION,
    }
    row.update(overrides)
    return row


def test_point_in_time_risk_projection_distinguishes_bad_news_from_missing_data() -> None:
    values = pd.DataFrame(
        [
            _risk_row("SH600000"),
            _risk_row(
                "SH600001",
                eligible=False,
                reasons='["suspended"]',
                suspended=True,
            ),
            _risk_row(
                "SH600002",
                eligible=False,
                reasons='["new_listing"]',
            ),
            _risk_row(
                "SH600003",
                eligible=False,
                reasons=(
                    '["negative_or_missing_equity", '
                    '"nonstandard_or_missing_audit"]'
                ),
                equity=np.nan,
                audit_opinion=None,
            ),
            _risk_row(
                "SH600004",
                eligible=False,
                reasons='["negative_or_missing_equity"]',
                equity=-1.0,
            ),
            _risk_row(
                "SH600005",
                eligible=False,
                reasons='["nonstandard_or_missing_audit"]',
                audit_opinion="qualified",
            ),
            _risk_row(
                "SH600006",
                eligible=False,
                reasons='["nonstandard_or_missing_audit"]',
                audit_opinion="adverse",
            ),
            _risk_row(
                "SH600007",
                eligible=False,
                reasons='["major_violation"]',
                major_violation=True,
            ),
            _risk_row(
                "SH600008",
                eligible=False,
                reasons='["st"]',
                is_st=True,
            ),
            _risk_row(
                "SH600009",
                eligible=False,
                reasons='["abnormal_listing"]',
                delisted=True,
                normal_listing_status=False,
            ),
            _risk_row(
                "SH600010",
                eligible=False,
                reasons='["insufficient_liquidity"]',
            ),
            _risk_row(
                "SH600011",
                eligible=False,
                reasons='["regulatory_data_missing"]',
                regulatory_data_available=False,
            ),
        ]
    )

    result = project_point_in_time_risk_states(
        values,
        as_of="2025-06-03",
        instruments=[*values["instrument"], "SZ000001"],
    ).set_index("instrument")

    assert set(result["risk_state"]).issubset(INSTRUMENT_RISK_STATES)
    assert result.loc["SH600000", "risk_state"] == "normal"
    assert result.loc["SH600001", "risk_state"] == "watch"
    assert not result.loc["SH600001", "tradable"]
    assert result.loc["SH600002", "risk_state"] == "restricted"
    assert result.loc["SH600003", "risk_state"] == "restricted"
    assert json.loads(result.loc["SH600003", "risk_reasons"]) == [
        "audit_evidence_missing",
        "equity_evidence_missing",
    ]
    assert result.loc["SH600004", "risk_state"] == "exit"
    assert result.loc["SH600005", "risk_state"] == "reduce"
    assert result.loc["SH600006", "risk_state"] == "exit"
    assert result.loc["SH600007", "risk_state"] == "exit"
    assert result.loc["SH600008", "risk_state"] == "exit"
    assert result.loc["SH600009", "risk_state"] == "exit"
    assert result.loc["SH600010", "risk_state"] == "restricted"
    assert result.loc["SH600011", "risk_state"] == "restricted"
    assert result.loc["SZ000001", "risk_state"] == "restricted"
    assert not result.loc["SZ000001", "tradable"]
    assert not result.loc["SZ000001", "allow_new_risk"]


def test_point_in_time_risk_projection_does_not_read_future_bad_news() -> None:
    values = pd.DataFrame(
        [
            _risk_row("SH600000", datetime=pd.Timestamp("2025-06-02")),
            _risk_row(
                "SH600000",
                datetime=pd.Timestamp("2025-06-04"),
                eligible=False,
                reasons='["st", "suspended"]',
                is_st=True,
                suspended=True,
            ),
        ]
    )

    before = project_point_in_time_risk_states(values, as_of="2025-06-03").iloc[0]
    after = project_point_in_time_risk_states(values, as_of="2025-06-04").iloc[0]

    assert before["risk_state"] == "restricted"
    assert before["evidence_datetime"] == pd.Timestamp("2025-06-02")
    assert not before["tradable"]
    assert json.loads(before["risk_reasons"]) == ["eligibility_evidence_stale"]
    assert after["risk_state"] == "exit"
    assert not after["tradable"]
    assert set(json.loads(after["risk_reasons"])) == {"st", "suspended"}


def test_point_in_time_risk_projection_does_not_reuse_stale_normal_state() -> None:
    values = pd.DataFrame([_risk_row("SH600000")])

    result = project_point_in_time_risk_states(
        values,
        as_of="2025-06-04",
        instruments=["SH600000"],
    ).iloc[0]

    assert result["risk_state"] == "restricted"
    assert not result["tradable"]
    assert not result["allow_new_risk"]
    assert json.loads(result["risk_reasons"]) == ["eligibility_evidence_stale"]


def test_point_in_time_risk_projection_does_not_apply_stock_financial_gates_to_etf() -> None:
    values = pd.DataFrame(
        [
            _risk_row(
                "SH510300",
                equity=np.nan,
                audit_opinion=None,
                financial_gate_required=False,
            )
        ]
    )

    result = project_point_in_time_risk_states(values, as_of="2025-06-03").iloc[0]

    assert result["risk_state"] == "normal"
    assert result["tradable"]
    assert result["allow_new_risk"]


def test_score_neutralization_contract_is_shared_by_recipe_mode() -> None:
    assert governed_score_neutralization(
        {"portfolio_construction": "topk_equal_weight"}
    ) == (False, ())
    assert governed_score_neutralization(
        {
            "portfolio_construction": "topk_equal_weight",
            "industry_relative_rank": True,
        }
    ) == (True, ())
    assert governed_score_neutralization(
        {"portfolio_construction": "benchmark_relative_qp"}
    ) == (True, ("size",))


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


def test_explicit_global_calendar_preserves_listing_day_math_across_batches() -> None:
    dates, inputs = _inputs()
    inputs["market"] = inputs["market"].iloc[[0, 2]].copy()
    result = build_point_in_time_eligibility(
        **inputs,
        policy=EligibilityPolicy(min_listing_trading_days=3, liquidity_lookback_days=2),
        trading_calendar=dates,
    ).set_index("datetime")

    assert result.loc[dates[2], "listing_trading_days"] == 3
    assert "new_listing" not in json.loads(result.loc[dates[2], "reasons"])


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
                "contract_version": ELIGIBILITY_CONTRACT_VERSION,
            },
            {
                "datetime": timestamp,
                "instrument": "SZ000001",
                "eligible": True,
                "contract_version": ELIGIBILITY_CONTRACT_VERSION,
            },
        ]
    )
    result = build_governed_signal(scores, topk=1, eligibility_matrix=eligibility)
    assert result.index.get_level_values("instrument").tolist() == ["SZ000001"]


def test_governed_signal_reuses_latest_prior_eligibility_snapshot() -> None:
    snapshot_dates = pd.to_datetime(["2025-06-02", "2025-06-04"])
    score_dates = pd.to_datetime(["2025-06-03", "2025-06-05"])
    instruments = ["SH600000", "SZ000001"]
    scores = pd.Series(
        [2.0, 1.0, 2.0, 1.0],
        index=pd.MultiIndex.from_product(
            [score_dates, instruments], names=["datetime", "instrument"]
        ),
    )
    eligibility = pd.DataFrame(
        [
            {
                "datetime": snapshot_dates[0],
                "instrument": "SH600000",
                "eligible": False,
                "contract_version": ELIGIBILITY_CONTRACT_VERSION,
            },
            {
                "datetime": snapshot_dates[0],
                "instrument": "SZ000001",
                "eligible": True,
                "contract_version": ELIGIBILITY_CONTRACT_VERSION,
            },
            {
                "datetime": snapshot_dates[1],
                "instrument": "SH600000",
                "eligible": True,
                "contract_version": ELIGIBILITY_CONTRACT_VERSION,
            },
            {
                "datetime": snapshot_dates[1],
                "instrument": "SZ000001",
                "eligible": False,
                "contract_version": ELIGIBILITY_CONTRACT_VERSION,
            },
        ]
    )

    result = build_governed_signal(scores, topk=1, eligibility_matrix=eligibility)

    assert result.index.tolist() == [
        (score_dates[0], "SZ000001"),
        (score_dates[1], "SH600000"),
    ]


def test_governed_signal_skips_an_empty_eligible_day_before_later_candidates() -> None:
    dates = pd.to_datetime(["2025-06-02", "2025-06-03"])
    instruments = ["SH600000", "SZ000001"]
    scores = pd.Series(
        [4.0, 3.0, 2.0, 1.0],
        index=pd.MultiIndex.from_product(
            [dates, instruments], names=["datetime", "instrument"]
        ),
    )
    eligibility = pd.DataFrame(
        [
            {
                "datetime": timestamp,
                "instrument": instrument,
                "eligible": timestamp == dates[1],
                "contract_version": ELIGIBILITY_CONTRACT_VERSION,
            }
            for timestamp in dates
            for instrument in instruments
        ]
    )
    memberships = pd.DataFrame(
        {
            "instrument": instruments,
            "industry": ["bank", "bank"],
            "in_date": [pd.Timestamp("2020-01-01")] * 2,
            "out_date": [pd.NaT] * 2,
        }
    )

    result = build_governed_signal(
        scores,
        topk=1,
        eligibility_matrix=eligibility,
        industry_memberships=memberships,
        neutralize_industry=True,
        metadata_availability_lag_days=0,
    )

    assert result.index.get_level_values("datetime").unique().tolist() == [dates[1]]
    assert result.index.get_level_values("instrument").tolist() == ["SH600000"]


def test_liquid_whitelisted_etf_can_reach_the_shared_governed_signal_path() -> None:
    dates = pd.date_range("2025-01-02", periods=80, freq="B")
    instrument = "SH510300"
    matrix = build_point_in_time_eligibility(
        market=pd.DataFrame(
            {
                "datetime": dates,
                "instrument": instrument,
                "asset_type": "etf",
                "amount": 700_000_000.0,
                "paused": False,
            }
        ),
        listings=pd.DataFrame(
            [
                {
                    "instrument": instrument,
                    "list_date": "2012-05-28",
                    "delist_date": None,
                }
            ]
        ),
        st_intervals=pd.DataFrame(
            columns=["instrument", "start_date", "end_date", "is_st"]
        ),
        suspensions=pd.DataFrame(
            columns=["datetime", "instrument", "suspended"]
        ),
        financials=pd.DataFrame(
            columns=["instrument", "announcement_date", "equity"]
        ),
        audits=pd.DataFrame(
            columns=["instrument", "announcement_date", "audit_opinion"]
        ),
        regulatory_events=None,
        trading_calendar=dates,
    )
    latest = matrix[matrix["datetime"].eq(dates[-1])]
    assert latest.iloc[0]["eligible"]
    assert not latest.iloc[0]["financial_gate_required"]

    scores = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples(
            [(dates[-1], instrument)], names=["datetime", "instrument"]
        ),
    )
    selected = build_governed_signal(scores, topk=1, eligibility_matrix=matrix)
    assert selected.index.get_level_values("instrument").tolist() == [instrument]


def test_governed_signal_keeps_ndrop_candidates_visible_to_portfolio_policy() -> None:
    timestamp = pd.Timestamp("2025-06-03")
    scores = pd.Series(
        [3.0, 2.0, 1.0],
        index=pd.MultiIndex.from_tuples(
            [
                (timestamp, "SH600002"),
                (timestamp, "SH600001"),
                (timestamp, "SH600000"),
            ],
            names=["datetime", "instrument"],
        ),
    )

    governed = build_governed_signal(scores, topk=2, n_drop=1).xs(
        timestamp, level="datetime"
    )
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=1,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
        )
    )
    decision = policy.decide(
        governed,
        {"SH600000": 0.50, "SH600001": 0.50},
    )

    assert set(governed.index) == {"SH600000", "SH600001", "SH600002"}
    assert set(decision.target_weights) == {"SH600000", "SH600001"}


def test_governed_signal_keeps_industry_feasible_substitutes_beyond_buffer() -> None:
    timestamp = pd.Timestamp("2025-06-03")
    instruments = ["SH600000", "SH600001", "SH600002", "SZ000001"]
    scores = pd.Series(
        [4.0, 3.0, 2.0, 1.0],
        index=pd.MultiIndex.from_product(
            [[timestamp], instruments], names=["datetime", "instrument"]
        ),
    )
    memberships = pd.DataFrame(
        {
            "instrument": instruments,
            "industry": ["bank", "bank", "bank", "technology"],
            "in_date": [pd.Timestamp("2020-01-01")] * 4,
            "out_date": [pd.NaT] * 4,
        }
    )

    governed = build_governed_signal(
        scores,
        topk=2,
        n_drop=1,
        industry_memberships=memberships,
        max_industry_weight=0.50,
        max_industry_deviation=1.0,
        neutralize_industry=False,
    ).xs(timestamp, level="datetime")
    no_buffer = build_governed_signal(
        scores,
        topk=2,
        n_drop=0,
        industry_memberships=memberships,
        max_industry_weight=0.50,
        max_industry_deviation=1.0,
        neutralize_industry=False,
    ).xs(timestamp, level="datetime")
    industries = memberships.set_index("instrument")["industry"]
    previous = {"SH600002": 0.50, "SZ000001": 0.50}
    with_buffer = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=1,
            max_position_weight=0.50,
            max_industry_weight=0.50,
            max_daily_turnover=1.0,
        )
    ).decide(governed, previous, industries=industries)
    without_buffer = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_industry_weight=0.50,
            max_daily_turnover=1.0,
        )
    ).decide(no_buffer, previous, industries=industries)

    assert set(governed.index) == set(instruments)
    assert set(no_buffer.index) == {"SH600000", "SH600001", "SZ000001"}
    assert set(with_buffer.target_weights) == {"SH600002", "SZ000001"}
    assert set(without_buffer.target_weights) == {"SH600000", "SZ000001"}


def test_reporting_benchmark_does_not_add_relative_cap_to_topk_candidates() -> None:
    timestamp = pd.Timestamp("2025-06-03")
    instruments = ["SH600000", "SH600001", "SH600002"]
    scores = pd.Series(
        [3.0, 2.0, 1.0],
        index=pd.MultiIndex.from_product(
            [[timestamp], instruments], names=["datetime", "instrument"]
        ),
    )
    memberships = pd.DataFrame(
        {
            "instrument": instruments,
            "industry": ["rare", "bank", "bank"],
            "in_date": [pd.Timestamp("2020-01-01")] * 3,
            "out_date": [pd.NaT] * 3,
        }
    )
    benchmark = pd.DataFrame(
        {
            "datetime": [timestamp],
            "instrument": ["SH600001"],
            "weight": [1.0],
        }
    )

    governed = build_governed_signal(
        scores,
        topk=2,
        n_drop=0,
        industry_memberships=memberships,
        benchmark_weights=benchmark,
        max_position_weight=0.50,
        max_industry_weight=1.0,
        max_industry_deviation=0.10,
        metadata_availability_lag_days=0,
        neutralize_industry=False,
        benchmark_relative_industry_constraints=False,
    ).xs(timestamp, level="datetime")
    decision = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_industry_weight=1.0,
            max_industry_deviation=0.10,
            max_daily_turnover=1.0,
        )
    ).decide(
        governed,
        {},
        industries=memberships.set_index("instrument")["industry"],
        benchmark_industry_weights=pd.Series({"bank": 1.0}),
    )

    assert set(governed.index) == {"SH600000", "SH600001"}
    assert set(decision.target_weights) == {"SH600000", "SH600001"}


def test_candidate_builder_requires_explicit_benchmark_relative_constraint() -> None:
    timestamp = pd.Timestamp("2025-06-03")
    instruments = ["SH600000", "SH600001", "SH600002"]
    scores = pd.Series(
        [3.0, 2.0, 1.0],
        index=pd.MultiIndex.from_product(
            [[timestamp], instruments], names=["datetime", "instrument"]
        ),
    )
    memberships = pd.DataFrame(
        {
            "instrument": instruments,
            "industry": ["rare", "bank", "bank"],
            "in_date": [pd.Timestamp("2020-01-01")] * 3,
            "out_date": [pd.NaT] * 3,
        }
    )
    benchmark = pd.DataFrame(
        {
            "datetime": [timestamp],
            "instrument": ["SH600001"],
            "weight": [1.0],
        }
    )

    governed = build_governed_signal(
        scores,
        topk=2,
        n_drop=0,
        industry_memberships=memberships,
        benchmark_weights=benchmark,
        max_position_weight=0.50,
        max_industry_weight=1.0,
        max_industry_deviation=0.10,
        metadata_availability_lag_days=0,
        neutralize_industry=False,
        benchmark_relative_industry_constraints=True,
    ).xs(timestamp, level="datetime")

    # The raw TopK retention set contains rare+bank, while the explicitly
    # benchmark-relative feasible set contributes the lower-ranked bank.
    assert set(governed.index) == set(instruments)


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
