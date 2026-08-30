from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.strategy_rule_runtime import (
    apply_strategy_rule_alpha_weights,
    build_portfolio_policy_runtime_metadata,
    build_strategy_rule_runtime_metadata,
    required_rule_history_sessions,
)
from scripts.run_multifactor_backtest import (
    _latest_style_cross_section as _backtest_latest_style_cross_section,
)
from scripts.run_multifactor_backtest import (
    _load_governed_style_exposures as _backtest_load_governed_style_exposures,
)
from scripts.run_multifactor_backtest import _market_trend_close_history
from scripts.run_recommendation_refresh import (
    _latest_style_cross_section as _recommendation_latest_style_cross_section,
)
from scripts.run_recommendation_refresh import (
    _load_governed_style_exposures as _recommendation_load_governed_style_exposures,
)
from scripts.run_recommendation_refresh import (
    _market_trend_close_history as _recommendation_market_trend_close_history,
)
from scripts.run_recommendation_refresh import (
    _recommendation_rule_runtime_metadata,
)

pytestmark = pytest.mark.no_database


class _FakeQlibDataApi:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def features(
        self,
        instruments: list[str],
        fields: list[str],
        **kwargs: object,
    ) -> pd.DataFrame:
        self.calls.append(
            {"instruments": instruments, "fields": fields, **kwargs}
        )
        index = pd.MultiIndex.from_product(
            [instruments, pd.bdate_range("2026-01-02", periods=20)],
            names=["instrument", "datetime"],
        )
        return pd.DataFrame({"$close": np.linspace(1.0, 1.1, len(index))}, index=index)


def test_runner_loads_the_exact_compiled_market_trend_benchmark() -> None:
    data_api = _FakeQlibDataApi()

    closes = _market_trend_close_history(
        data_api,
        strategy_config={
            "market_trend_lookback_sessions": 20,
            "market_trend_benchmark": "SH000300",
        },
        start_time="2025-01-01",
        end_time="2026-01-31",
    )

    assert closes is not None
    assert list(closes.columns) == ["SH000300"]
    assert data_api.calls == [
        {
            "instruments": ["SH000300"],
            "fields": ["$close"],
            "start_time": "2025-01-01",
            "end_time": "2026-01-31",
            "freq": "day",
        }
    ]


def test_backtest_and_recommendation_load_the_same_bound_market_benchmark() -> None:
    backtest_api = _FakeQlibDataApi()
    recommendation_api = _FakeQlibDataApi()
    config = {
        "market_trend_lookback_sessions": 20,
        "market_trend_benchmark": "SH000300",
    }

    backtest = _market_trend_close_history(
        backtest_api,
        strategy_config=config,
        start_time="2025-01-01",
        end_time="2026-01-31",
    )
    recommendation = _recommendation_market_trend_close_history(
        recommendation_api,
        strategy_config=config,
        start_time="2025-01-01",
        end_time="2026-01-31",
    )

    assert backtest is not None and recommendation is not None
    pd.testing.assert_frame_equal(backtest, recommendation)
    assert backtest_api.calls == recommendation_api.calls


def test_recommendation_runtime_passes_exact_bound_benchmark_history() -> None:
    data_api = _FakeQlibDataApi()
    dates = pd.bdate_range("2026-01-02", periods=20)
    constituent_closes = pd.DataFrame(
        {"A": np.linspace(10.0, 12.0, len(dates))}, index=dates
    )

    metadata = _recommendation_rule_runtime_metadata(
        data_api,
        config={
            "market_trend_lookback_sessions": 20,
            "market_trend_benchmark": "SH000300",
        },
        instruments=pd.Index(["A"]),
        close_history=constituent_closes,
        value_exposures=None,
        start_time="2025-01-01",
        end_time="2026-01-31",
    )

    assert metadata["market_regime_allows_entries"] is True
    assert data_api.calls[0]["instruments"] == ["SH000300"]


def test_backtest_and_recommendation_share_standardized_style_artifact(tmp_path) -> None:
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    instruments = [f"S{index:04d}" for index in range(100)]
    source = pd.DataFrame(
        {
            "datetime": pd.Timestamp("2026-01-30"),
            "instrument": instruments,
            "size": np.linspace(-2.0, 2.0, len(instruments)),
            "value": np.linspace(1.0, -1.0, len(instruments)),
            "growth": 0.25,
            "volatility": -0.10,
        }
    )
    source.loc[0, "growth"] = np.nan
    source.to_parquet(metadata_root / "style_exposures.parquet", index=False)

    backtest_frame, backtest_evidence = _backtest_load_governed_style_exposures(
        tmp_path
    )
    recommendation_frame, recommendation_evidence = (
        _recommendation_load_governed_style_exposures(tmp_path)
    )
    backtest_cross_section = _backtest_latest_style_cross_section(
        backtest_frame, pd.Timestamp("2026-01-30")
    )
    recommendation_cross_section = _recommendation_latest_style_cross_section(
        recommendation_frame, pd.Timestamp("2026-01-30")
    )
    recommendation_raw_cross_section = _recommendation_latest_style_cross_section(
        recommendation_frame,
        pd.Timestamp("2026-01-30"),
        preserve_missing=True,
    )

    pd.testing.assert_frame_equal(backtest_frame, recommendation_frame)
    pd.testing.assert_frame_equal(
        backtest_cross_section, recommendation_cross_section
    )
    assert backtest_cross_section.loc["S0000", "growth"] == 0.0
    assert pd.isna(recommendation_raw_cross_section.loc["S0000", "growth"])
    assert backtest_evidence == recommendation_evidence
    assert backtest_evidence["max_cross_section_missing_rate"] == 0.05


def test_topk_requires_only_signal_and_holding_industries() -> None:
    metadata = build_portfolio_policy_runtime_metadata(
        {"portfolio_construction": "topk_equal_weight"},
        instruments=pd.Index(["SIGNAL", "HELD"]),
        industries=pd.Series({"SIGNAL": "tech", "HELD": "bank"}),
        benchmark_weights=None,
        style_exposures=None,
        return_covariance=None,
    )

    assert metadata["industries"].to_dict() == {
        "SIGNAL": "tech",
        "HELD": "bank",
    }
    assert "benchmark_weights" not in metadata
    assert "benchmark_industry_weights" not in metadata
    assert "benchmark_style_exposure" not in metadata
    assert "return_covariance" not in metadata

    with pytest.raises(ValueError, match="signal or current holdings"):
        build_portfolio_policy_runtime_metadata(
            {"portfolio_construction": "topk_equal_weight"},
            instruments=pd.Index(["SIGNAL", "HELD"]),
            industries=pd.Series({"SIGNAL": "tech"}),
            benchmark_weights=None,
            style_exposures=None,
            return_covariance=None,
        )


def test_qp_remains_strict_about_benchmark_industry_style_and_covariance() -> None:
    config = {"portfolio_construction": "benchmark_relative_qp"}
    covariance = pd.DataFrame(
        [[0.1, 0.0], [0.0, 0.1]],
        index=["SIGNAL", "BENCHMARK"],
        columns=["SIGNAL", "BENCHMARK"],
    )
    with pytest.raises(ValueError, match="benchmark constituents"):
        build_portfolio_policy_runtime_metadata(
            config,
            instruments=pd.Index(["SIGNAL"]),
            industries=pd.Series({"SIGNAL": "tech"}),
            benchmark_weights=pd.Series({"BENCHMARK": 1.0}),
            style_exposures=pd.DataFrame(
                {"size": [0.1, 0.2]}, index=["SIGNAL", "BENCHMARK"]
            ),
            return_covariance=covariance,
        )

    with pytest.raises(ValueError, match="incomplete point-in-time styles"):
        build_portfolio_policy_runtime_metadata(
            config,
            instruments=pd.Index(["SIGNAL"]),
            industries=pd.Series({"SIGNAL": "tech", "BENCHMARK": "bank"}),
            benchmark_weights=pd.Series({"BENCHMARK": 1.0}),
            style_exposures=pd.DataFrame(
                {"size": [0.1, np.nan]}, index=["SIGNAL", "BENCHMARK"]
            ),
            return_covariance=covariance,
        )

    with pytest.raises(ValueError, match="return covariance"):
        build_portfolio_policy_runtime_metadata(
            config,
            instruments=pd.Index(["SIGNAL"]),
            industries=pd.Series({"SIGNAL": "tech", "BENCHMARK": "bank"}),
            benchmark_weights=pd.Series({"BENCHMARK": 1.0}),
            style_exposures=pd.DataFrame(
                {"size": [0.1, 0.2]}, index=["SIGNAL", "BENCHMARK"]
            ),
            return_covariance=None,
        )


def test_rule_alpha_weights_change_the_shared_factor_grid_deterministically() -> None:
    recipe = get_strategy_recipe("short_relative_strength")
    config = {
        **recipe["config_overrides"],
        "recipe_id": recipe["id"],
        "recipe_version": recipe["version"],
    }
    columns = [item["id"] for item in recipe["factor_baseline"]]
    normalized = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
        columns=columns,
    )
    public_weights = {
        item["id"]: float(item["weight"]) for item in recipe["factor_baseline"]
    }
    baseline = sum(
        (normalized[key] * weight for key, weight in public_weights.items()),
        start=pd.Series(0.0, index=normalized.index),
    )

    actual = apply_strategy_rule_alpha_weights(normalized, baseline, config)

    assert actual.equals(baseline.rename("score"))
    changed = dict(config)
    changed_ir = {
        **config["strategy_rule_ir"],
        "slots": {
            **config["strategy_rule_ir"]["slots"],
            "alpha_rank": {
                **config["strategy_rule_ir"]["slots"]["alpha_rank"],
                "components": [
                    {
                        "component": "weighted_factor_rank",
                        "parameters": {
                            "weights": {
                                columns[0]: 0.50,
                                columns[1]: 0.20,
                                columns[2]: 0.15,
                                columns[3]: 0.15,
                            }
                        },
                    }
                ],
            },
        },
    }
    from quant_platform.strategy_rule_compiler import compile_strategy_rule_policy

    policy = compile_strategy_rule_policy(
        "short_1_5d", changed_ir, allowed_factor_ids=set(columns)
    )
    changed_ir["rules_sha256"] = policy["strategy_rules_sha256"]
    changed.update(
        {
            "source_research_artifact_id": "artifact-1",
            "strategy_rule_ir": changed_ir,
            "strategy_rules_sha256": policy["strategy_rules_sha256"],
            "strategy_rule_policy_sha256": policy["policy_sha256"],
        }
    )
    for key, value in policy.items():
        if key in changed and value is not None:
            changed[key] = value
    assert not apply_strategy_rule_alpha_weights(normalized, baseline, changed).equals(
        baseline.rename("score")
    )


def test_rule_runtime_builds_pit_entry_and_exit_inputs() -> None:
    dates = pd.bdate_range("2026-01-02", periods=70)
    closes = pd.DataFrame(
        {
            "A": np.linspace(10.0, 14.0, len(dates)),
            "B": np.linspace(20.0, 18.0, len(dates)),
        },
        index=dates,
    )
    config = {
        "extension_guard_max_return_5d": 0.12,
        "trend_break_lookback_sessions": 20,
        "market_trend_lookback_sessions": 60,
        "market_trend_benchmark": "SH000300",
        "valuation_regime_max_percentile": 0.90,
        "liquidity_lookback_days": 20,
    }

    metadata = build_strategy_rule_runtime_metadata(
        config,
        instruments=pd.Index(["A", "B"]),
        close_history=closes,
        benchmark_weights=pd.Series({"A": 0.6, "B": 0.4}),
        benchmark_close_history=closes[["A"]].rename(columns={"A": "SH000300"}),
        value_exposures=pd.Series({"A": 0.5, "B": 0.1}),
    )

    assert required_rule_history_sessions(config) == 60
    assert metadata["market_regime_allows_entries"] is True
    assert metadata["trend_intact"].to_dict() == {"A": True, "B": False}
    assert metadata["five_day_returns"]["A"] > 0
    assert metadata["valuation_percentiles"]["B"] > metadata["valuation_percentiles"]["A"]


def test_market_trend_uses_bound_index_not_incomplete_constituent_history() -> None:
    dates = pd.bdate_range("2026-01-02", periods=20)
    constituent_closes = pd.DataFrame(
        {
            "A": np.linspace(10.0, 12.0, len(dates)),
            "NEW_MEMBER": [np.nan] * 19 + [8.0],
        },
        index=dates,
    )
    benchmark_closes = pd.DataFrame(
        {"SH000300": np.linspace(1.0, 1.1, len(dates))}, index=dates
    )

    metadata = build_strategy_rule_runtime_metadata(
        {
            "market_trend_lookback_sessions": 20,
            "market_trend_benchmark": "SH000300",
        },
        instruments=pd.Index(["A", "NEW_MEMBER"]),
        close_history=constituent_closes,
        benchmark_weights=pd.Series({"A": 0.5, "NEW_MEMBER": 0.5}),
        benchmark_close_history=benchmark_closes,
    )

    assert metadata["market_regime_allows_entries"] is True


def test_market_trend_fails_closed_on_wrong_or_incomplete_bound_index() -> None:
    dates = pd.bdate_range("2026-01-02", periods=20)
    closes = pd.DataFrame({"A": np.linspace(10.0, 12.0, len(dates))}, index=dates)
    config = {
        "market_trend_lookback_sessions": 20,
        "market_trend_benchmark": "SH000300",
    }

    with pytest.raises(ValueError, match="differs from the bound benchmark"):
        build_strategy_rule_runtime_metadata(
            config,
            instruments=pd.Index(["A"]),
            close_history=closes,
            benchmark_close_history=pd.DataFrame(
                {"SH000905": np.linspace(1.0, 1.1, len(dates))}, index=dates
            ),
        )

    incomplete = pd.DataFrame(
        {"SH000300": [np.nan, *np.linspace(1.0, 1.1, len(dates) - 1)]},
        index=dates,
    )
    with pytest.raises(ValueError, match="incomplete benchmark close history"):
        build_strategy_rule_runtime_metadata(
            config,
            instruments=pd.Index(["A"]),
            close_history=closes,
            benchmark_close_history=incomplete,
        )


def test_extension_guard_emits_only_complete_per_stock_evidence() -> None:
    metadata = build_strategy_rule_runtime_metadata(
        {"extension_guard_max_return_5d": 0.10},
        instruments=pd.Index(["A", "INCOMPLETE"]),
        close_history=pd.DataFrame(
            {
                "A": np.linspace(1.0, 1.05, 6),
                "INCOMPLETE": [np.nan, 1.0, 1.01, 1.02, 1.03, 1.04],
            }
        ),
    )

    assert list(metadata["five_day_returns"].index) == ["A"]


def test_extension_gap_blocks_only_new_entry_and_preserves_existing_holding() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
            extension_guard_max_return_5d=0.10,
        )
    )
    evidence = pd.Series({"COMPLETE": 0.05})
    fresh = policy.decide(
        pd.Series({"INCOMPLETE": 2.0, "COMPLETE": 1.0}),
        {},
        five_day_returns=evidence,
    )
    held = policy.decide(
        pd.Series({"INCOMPLETE": 2.0, "COMPLETE": 1.0}),
        {"INCOMPLETE": 1.0},
        five_day_returns=evidence,
    )

    assert fresh.target_weights == {"COMPLETE": 1.0}
    assert held.target_weights == {"INCOMPLETE": 1.0}


def test_trend_gap_blocks_new_entry_but_does_not_abort_existing_holding() -> None:
    dates = pd.bdate_range("2026-01-02", periods=20)
    closes = pd.DataFrame(
        {
            "COMPLETE": np.linspace(10.0, 12.0, len(dates)),
            "INCOMPLETE": [np.nan, *np.linspace(10.0, 11.0, len(dates) - 1)],
        },
        index=dates,
    )
    metadata = build_strategy_rule_runtime_metadata(
        {"trend_break_lookback_sessions": 20},
        instruments=pd.Index(["COMPLETE", "INCOMPLETE"]),
        close_history=closes,
    )

    assert metadata["trend_intact"].to_dict() == {"COMPLETE": True}
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
            trend_break_lookback_sessions=20,
        )
    )
    first = policy.decide(
        pd.Series({"INCOMPLETE": 2.0, "COMPLETE": 1.0}),
        {},
        trend_intact=metadata["trend_intact"],
    )
    second = policy.decide(
        pd.Series({"INCOMPLETE": 2.0, "COMPLETE": 1.0}),
        {"INCOMPLETE": 1.0},
        trend_intact=metadata["trend_intact"],
    )

    assert first.target_weights == {"COMPLETE": 1.0}
    assert second.target_weights == {"INCOMPLETE": 1.0}
    assert any(
        event["rule"] == "trend_evidence_unavailable"
        and event["instrument"] == "INCOMPLETE"
        for event in second.risk_events
    )


def test_suspended_holding_is_preserved_until_fresh_trend_evidence_recovers() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
            trend_break_lookback_sessions=3,
        )
    )
    initial = build_strategy_rule_runtime_metadata(
        {"trend_break_lookback_sessions": 3},
        instruments=pd.Index(["A"]),
        close_history=pd.DataFrame({"A": [10.0, 10.5, 11.0]}),
    )
    entered = policy.decide(
        pd.Series({"A": 1.0}), {}, trend_intact=initial["trend_intact"]
    )
    suspended = build_strategy_rule_runtime_metadata(
        {"trend_break_lookback_sessions": 3},
        instruments=pd.Index(["A"]),
        close_history=pd.DataFrame({"A": [10.5, 11.0, np.nan]}),
    )
    during_halt = policy.decide(
        pd.Series({"A": 1.0}),
        entered.target_weights,
        trend_intact=suspended["trend_intact"],
    )
    resumed = build_strategy_rule_runtime_metadata(
        {"trend_break_lookback_sessions": 3},
        instruments=pd.Index(["A"]),
        close_history=pd.DataFrame({"A": [11.0, 11.2, 11.4]}),
    )
    after_resume = policy.decide(
        pd.Series({"A": 1.0}),
        during_halt.target_weights,
        trend_intact=resumed["trend_intact"],
    )

    assert entered.target_weights == {"A": 1.0}
    assert during_halt.target_weights == {"A": 1.0}
    assert any(
        event["rule"] == "trend_evidence_unavailable"
        for event in during_halt.risk_events
    )
    assert after_resume.target_weights == {"A": 1.0}
    assert not any(
        event["rule"] == "trend_evidence_unavailable"
        for event in after_resume.risk_events
    )


def test_missing_valuation_rejects_only_new_entry_without_imputation() -> None:
    config = {"valuation_regime_max_percentile": 0.9}
    metadata = build_strategy_rule_runtime_metadata(
        config,
        instruments=pd.Index(["VALUED", "MISSING"]),
        close_history=pd.DataFrame({"VALUED": [1.0], "MISSING": [1.0]}),
        value_exposures=pd.Series({"VALUED": 0.5, "MISSING": np.nan}),
    )
    percentiles = metadata["valuation_percentiles"]
    assert percentiles["VALUED"] == 1.0
    assert pd.isna(percentiles["MISSING"])

    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
            valuation_regime_max_percentile=1.0,
        )
    )
    new_entry = policy.decide(
        pd.Series({"MISSING": 2.0, "VALUED": 1.0}),
        {},
        valuation_percentiles=percentiles,
    )
    existing_holding = policy.decide(
        pd.Series({"MISSING": 2.0, "VALUED": 1.0}),
        {"MISSING": 1.0},
        valuation_percentiles=percentiles,
    )

    assert set(new_entry.target_weights) == {"VALUED"}
    assert existing_holding.target_weights == {"MISSING": 1.0}


def test_all_missing_valuations_produce_cash_instead_of_fabricated_values() -> None:
    metadata = build_strategy_rule_runtime_metadata(
        {"valuation_regime_max_percentile": 0.9},
        instruments=pd.Index(["A", "B"]),
        close_history=pd.DataFrame({"A": [1.0], "B": [1.0]}),
        value_exposures=None,
    )
    assert metadata["valuation_percentiles"].isna().all()

    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=1,
            n_drop=0,
            max_position_weight=1.0,
            max_daily_turnover=1.0,
            valuation_regime_max_percentile=0.9,
        )
    )
    decision = policy.decide(
        pd.Series({"A": 2.0, "B": 1.0}),
        {},
        valuation_percentiles=metadata["valuation_percentiles"],
    )

    assert decision.target_weights == {}


def test_policy_executes_holding_age_score_and_trend_exits() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.5,
            max_daily_turnover=1.0,
            entry_score_min_percentile=0.75,
            score_drop_exit_percentile=0.50,
            max_holding_sessions=5,
            trend_break_lookback_sessions=20,
        )
    )
    decision = policy.decide(
        pd.Series({"new": 4.0, "runner_up": 3.0, "aged": 2.0, "broken": 1.0}),
        {"aged": 0.5, "broken": 0.5},
        holding_age_sessions={"aged": 5, "broken": 2},
        trend_intact=pd.Series(
            {"new": True, "runner_up": True, "aged": True, "broken": False}
        ),
    )

    assert set(decision.target_weights) == {"new", "runner_up"}
    assert {event["rule"] for event in decision.risk_events} >= {
        "max_holding_sessions",
        "score_drop_exit",
        "trend_break",
    }
    assert decision.position_state["holding_age_sessions"] == {
        "new": 0,
        "runner_up": 0,
    }


def test_policy_returns_cash_when_regime_has_no_edge() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            market_trend_lookback_sessions=20,
            cash_when_no_edge=True,
        )
    )

    decision = policy.decide(
        pd.Series({"A": 2.0, "B": 1.0}),
        {},
        market_regime_allows_entries=False,
    )

    assert decision.target_weights == {}
    assert decision.expected_turnover == 0.0


def test_long_thesis_break_runs_only_on_a_governed_review_decision() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.5,
            max_daily_turnover=1.0,
            thesis_min_holding_sessions=252,
            thesis_break_score_percentile=0.75,
            thesis_review_frequency="month",
        )
    )
    signal = pd.Series({"strong": 2.0, "weak": 1.0})
    previous = {"weak": 0.5}

    daily_check = policy.decide(
        signal,
        previous,
        holding_age_sessions={"weak": 252},
        rebalance_due=False,
    )
    monthly_review = policy.decide(
        signal,
        previous,
        holding_age_sessions={"weak": 252},
        rebalance_due=True,
    )

    assert daily_check.target_weights == previous
    assert not any(
        event["rule"] == "quant_quality_value_thesis_break_proxy"
        for event in daily_check.risk_events
    )
    assert "weak" not in monthly_review.target_weights
    assert any(
        event["rule"] == "quant_quality_value_thesis_break_proxy"
        for event in monthly_review.risk_events
    )
