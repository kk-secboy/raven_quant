from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from quant_platform.factor_library import compile_qlib_expression
from quant_platform.feature_set_registry import get_feature_set
from quant_platform.research_automation import (
    ResearchWindowUnavailableError,
    normalize_research_period_policy,
    normalize_research_schedule_payload,
    resolve_research_periods,
    resolve_research_window_contract,
)
from quant_platform.research_horizon import (
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    canonical_sha256,
)


def _calendar(count: int = 5500) -> list[str]:
    result: list[str] = []
    current = date(2010, 1, 4)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def _pre_2022_unopened_calendar() -> list[str]:
    """Mirror the server audit: 3,527 sessions, 1,683 cost-covered."""

    pre_cost = [
        item.date().isoformat()
        for item in pd.bdate_range(end="2015-07-31", periods=1844)
    ]
    candidates = [
        item.date().isoformat()
        for item in pd.bdate_range("2015-08-03", "2022-07-05")
    ]
    indices = sorted(
        {
            round(index * (len(candidates) - 1) / (1683 - 1))
            for index in range(1683)
        }
    )
    cost_covered = [candidates[index] for index in indices]
    assert len(pre_cost) == 1844
    assert len(cost_covered) == 1683
    return pre_cost + cost_covered


def _dataset(feature_set: dict | None = None) -> dict:
    required_fields = sorted(
        {
            field
            for expression in (feature_set or {"features": {"close": "$close"}})[
                "features"
            ].values()
            for field in compile_qlib_expression(str(expression)).required_fields
        }
    )
    coverage = {
        "version": "qlib-field-year-source-coverage-v1",
        "source_attribution_policy": "normalized-staging-and-snapshot-contracts-v1",
        "legacy_overlap_policy_version": "overlap-v1",
        "primary_market_history_start": "2016-01-01",
        "fields": {
            field: {
                "source_family": "test",
                "available_from": "2010-01-04",
                "available_to": "2099-12-31",
                "continuous_from": "2010-01-04",
                "research_available_from": "2010-01-04",
                "years": [
                    {
                        "year": 2010,
                        "observed_rows": 1,
                        "non_null_rows": 1,
                        "coverage_ratio": 1.0,
                        "first_session": "2010-01-04",
                        "last_session": "2099-12-31",
                        "source_contracts": ["test-source"],
                    }
                ],
            }
            for field in required_fields
        },
    }
    coverage_sha256 = canonical_sha256(coverage)
    coverage = {**coverage, "coverage_sha256": coverage_sha256}
    return {
        "name": "cn-governed-day",
        "lineage_id": "b" * 64,
        "provenance": {
            "dataset_identity_sha256": "a" * 64,
            "dataset_contract_sha256": "c" * 64,
            "dataset_lineage_id": "b" * 64,
            "field_contract_version": "qlib-daily-v4",
            "fields": required_fields,
            "field_units": {
                "open": "CNY/share",
                "close": "CNY/share",
                "volume": "share",
                "pit_roe": "ratio",
            },
            "research_features": {"version": "pit-research-features-v1"},
            "field_coverage_sha256": coverage_sha256,
            "field_year_coverage": coverage,
            "source_start_date": "2010-01-04",
            "source_end_date": "2031-01-01",
        },
    }


@pytest.mark.no_database
@pytest.mark.parametrize(
    ("profile", "labels", "purge", "embargo", "oos"),
    (
        (SHORT_1_5D, [1, 2, 3, 5], 6, 20, 252),
        (SWING_1_6M, [21, 63, 126], 127, 127, 504),
        (LONG_1_3Y, [63, 126, 252], 253, 253, 756),
    ),
)
def test_three_horizon_windows_freeze_labels_gaps_oos_and_maturity(
    profile: str,
    labels: list[int],
    purge: int,
    embargo: int,
    oos: int,
) -> None:
    calendar = _calendar()
    periods, evidence = resolve_research_periods(calendar, horizon_profile=profile)

    assert evidence["horizon_profile"] == profile
    assert evidence["label_horizons_sessions"] == labels
    assert evidence["purge_trading_days"] == purge
    assert evidence["embargo_trading_days"] == embargo
    assert evidence["label_maturity_tail_trading_days"] == max(labels)
    assert periods["test_end"] == calendar[-max(labels) - 1]
    assert evidence["latest_mature_label_sessions"][str(max(labels))] == periods["test_end"]
    assert (
        calendar.index(periods["valid_start"])
        - calendar.index(periods["train_end"])
        - 1
        == purge
    )
    assert (
        calendar.index(periods["test_start"])
        - calendar.index(periods["valid_end"])
        - 1
        == embargo
    )
    assert calendar.index(periods["test_end"]) - calendar.index(periods["test_start"]) + 1 == oos


@pytest.mark.no_database
def test_pre_2022_unopened_history_runs_short_and_swing_without_fake_10y() -> None:
    calendar = _pre_2022_unopened_calendar()

    short_periods, short = resolve_research_periods(
        calendar, horizon_profile=SHORT_1_5D
    )
    swing_periods, swing = resolve_research_periods(
        calendar, horizon_profile=SWING_1_6M
    )

    assert len(calendar) == 3527
    assert sum(day >= "2015-08-03" for day in calendar) == 1683
    assert short_periods["test_end"] < "2022-07-06"
    assert swing_periods["test_end"] < "2022-07-06"
    assert [item["id"] for item in short["evaluation_profiles"]] == [
        "recent_3y",
        "robust_10y",
        "balanced_5y",
    ]
    assert [item["id"] for item in swing["evaluation_profiles"]] == [
        "recent_3y",
        "robust_10y",
        "balanced_5y",
    ]
    swing_resolution = swing["evaluation_profile_resolution"]
    assert swing_resolution["capital_evaluation_eligible"] is True
    assert swing_resolution["effective_profiles"] == [
        "recent_3y",
        "robust_10y",
        "balanced_5y",
    ]
    assert swing_resolution["unavailable_profiles"] == []
    robust = next(
        item for item in swing["evaluation_profiles"] if item["id"] == "robust_10y"
    )
    assert robust["requested_validation_trading_days"] == 2520
    assert robust["effective_validation_trading_days"] == 926
    assert robust["validation_window_truncated"] is True
    confirmation = next(
        item
        for item in swing["evaluation_profiles"]
        if item["id"] == "balanced_5y"
    )
    assert confirmation["requested_validation_trading_days"] == 1260
    assert confirmation["effective_validation_trading_days"] == 841
    assert confirmation["binding_constraints"][-1] == (
        "confirmation_shortened_to_distinct_cost_covered_midpoint"
    )


@pytest.mark.no_database
def test_pre_2022_long_is_explicitly_unavailable_without_weakening_oos() -> None:
    calendar = _pre_2022_unopened_calendar()

    with pytest.raises(ResearchWindowUnavailableError) as captured:
        resolve_research_periods(calendar, horizon_profile=LONG_1_3Y)

    evidence = captured.value.evidence
    assert evidence["calendar_trading_days"] == 3527
    assert evidence["capital_evaluation_eligible"] is False
    assert evidence["capital_evaluation_unavailable_reason"] == (
        "primary_and_distinct_stress_profiles_required"
    )
    assert evidence["effective_profiles"] == ["recent_3y"]
    assert {item["id"] for item in evidence["unavailable_profiles"]} == {
        "robust_10y",
        "balanced_5y",
    }
    primary = next(
        item
        for item in evidence["requested_profiles"]
        if item["id"] == "recent_3y"
    )
    assert primary["periods"]["test_end"] == calendar[-253]
    assert sum(
        primary["periods"]["test_start"]
        <= day
        <= primary["periods"]["test_end"]
        for day in calendar
    ) == 756


@pytest.mark.no_database
def test_research_window_contract_binds_dataset_features_and_is_deterministic() -> None:
    calendar = _calendar()
    feature_set = get_feature_set("unified-research-v1")

    left_periods, left = resolve_research_window_contract(
        _dataset(feature_set),
        calendar,
        horizon_profile=SWING_1_6M,
        feature_set=feature_set,
    )
    right_periods, right = resolve_research_window_contract(
        _dataset(feature_set),
        calendar,
        horizon_profile=SWING_1_6M,
        feature_set=feature_set,
    )

    assert left_periods == right_periods
    assert left["research_window_contract_sha256"] == right[
        "research_window_contract_sha256"
    ]
    contract = left["research_window_contract"]
    assert contract["dataset_identity_sha256"] == "a" * 64
    assert contract["dataset_lineage_id"] == "b" * 64
    assert contract["feature_set_id"] == "unified-research-v1"
    assert contract["feature_set_sha256"] == feature_set["definition_sha256"]
    assert contract["signal_time_semantics"] == "complete_daily_bar_after_exchange_close"
    assert contract["execution_lag_sessions"] == 1
    assert contract["label_maturity_enforced"] is True
    assert contract["periods"] == left_periods


@pytest.mark.no_database
def test_research_window_starts_only_after_all_selected_fields_are_usable() -> None:
    calendar = _calendar()
    feature_set = {
        "id": "quality-only-test",
        "definition_sha256": "d" * 64,
        "features": {"quality": "$fund_roe"},
    }
    dataset = _dataset(feature_set)
    matrix = dataset["provenance"]["field_year_coverage"]
    matrix["fields"]["fund_roe"].update(
        {
            "available_from": "2012-04-30",
            "continuous_from": "2016-04-30",
            "research_available_from": "2016-04-30",
        }
    )
    unsigned = dict(matrix)
    unsigned.pop("coverage_sha256")
    digest = canonical_sha256(unsigned)
    matrix["coverage_sha256"] = digest
    dataset["provenance"]["field_coverage_sha256"] = digest

    periods, evidence = resolve_research_window_contract(
        dataset,
        calendar,
        horizon_profile=SWING_1_6M,
        feature_set=feature_set,
    )

    coverage = evidence["required_field_coverage"]
    assert coverage["required_fields"] == ["fund_roe"]
    assert coverage["effective_field_start_session"] == "2016-04-30"
    assert evidence["research_window_contract"]["calendar_start"] == "2016-05-02"
    assert periods["train_start"] >= "2016-05-02"

    explicit = dict(periods)
    explicit["train_start"] = "2016-04-29"
    with pytest.raises(ValueError, match="before all selected factor fields"):
        resolve_research_window_contract(
            dataset,
            calendar,
            periods=explicit,
            horizon_profile=SWING_1_6M,
            feature_set=feature_set,
        )


@pytest.mark.no_database
def test_research_window_rejects_tampered_field_coverage() -> None:
    calendar = _calendar()
    feature_set = {
        "id": "close-only-test",
        "definition_sha256": "e" * 64,
        "features": {"close": "$close"},
    }
    dataset = _dataset(feature_set)
    dataset["provenance"]["field_year_coverage"]["fields"]["close"][
        "research_available_from"
    ] = "2018-01-01"

    with pytest.raises(ValueError, match="coverage digest is invalid"):
        resolve_research_window_contract(
            dataset,
            calendar,
            horizon_profile=SHORT_1_5D,
            feature_set=feature_set,
        )


@pytest.mark.no_database
def test_active_explicit_window_cannot_use_immature_final_labels() -> None:
    calendar = _calendar()
    periods, _ = resolve_research_periods(calendar, horizon_profile=SHORT_1_5D)
    periods["test_end"] = calendar[-1]

    with pytest.raises(ValueError, match="post-OOS sessions for label maturity"):
        resolve_research_periods(
            calendar,
            periods=periods,
            horizon_profile=SHORT_1_5D,
        )


@pytest.mark.no_database
def test_horizon_policy_defaults_and_schedule_binding_are_canonical() -> None:
    assert normalize_research_period_policy(horizon_profile=SHORT_1_5D) == {
        "test_trading_days": 252,
        "embargo_trading_days": 20,
    }
    assert normalize_research_period_policy(horizon_profile=SWING_1_6M) == {
        "test_trading_days": 504,
        "embargo_trading_days": 127,
    }
    assert normalize_research_period_policy(horizon_profile=LONG_1_3Y) == {
        "test_trading_days": 756,
        "embargo_trading_days": 253,
    }
    normalized = normalize_research_schedule_payload(
        {
            "scenario": "fin_strategy",
            "objective": "Research deterministic long-horizon quality-value entry rules.",
            "dataset": "cn-governed-day",
            "feature_set_id": "unified-research-v1",
            "horizon": "long",
        },
        max_loops=3,
    )
    assert normalized["horizon_profile"] == LONG_1_3Y
    assert normalized["period_policy"] == {
        "test_trading_days": 756,
        "embargo_trading_days": 253,
    }


@pytest.mark.no_database
def test_legacy_window_remains_explicitly_ambiguous() -> None:
    calendar = _calendar(4000)
    periods, evidence = resolve_research_periods(calendar)

    assert periods["test_end"] == calendar[-1]
    assert evidence["horizon_profile"] == "legacy_ambiguous"
    assert evidence["label_maturity_enforced"] is False
    assert evidence["label_maturity_tail_trading_days"] == 0
