from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_platform.api import RDAgentRunRequest, ResearchPeriodPolicy
from quant_platform.research_automation import (
    derive_multi_profile_research_periods,
    derive_rolling_research_periods,
    normalize_research_schedule_payload,
    rank_factor_candidates,
    rank_multi_profile_candidates,
    required_multi_profile_trading_days,
    resolve_research_periods,
    select_latest_program_dataset,
    select_latest_program_rebase_dataset,
)


@pytest.mark.no_database
def test_default_research_period_policy_locks_one_year_final_oos() -> None:
    policy = ResearchPeriodPolicy()

    assert policy.embargo_trading_days == 20
    assert policy.test_trading_days == 252
    assert required_multi_profile_trading_days() == 3046


def _payload() -> dict:
    return {
        "objective": "Research a low-turnover quality factor for CSI 300 enhancement.",
        "dataset": "cn-research",
        "feature_set_id": "governed-baseline",
        "horizon": "short",
        "loop_n": 2,
        "duration": "1h",
        "requested_by": "research-scheduler",
        "period_mode": "explicit",
        "periods": {
            "train_start": "2018-01-01",
            "train_end": "2021-12-31",
            "valid_start": "2022-01-01",
            "valid_end": "2023-12-31",
            "test_start": "2024-01-01",
            "test_end": "2025-01-02",
        },
    }


@pytest.mark.no_database
def test_research_schedule_payload_is_normalized() -> None:
    normalized = normalize_research_schedule_payload(
        _payload(), max_loops=3, allow_explicit_periods=True
    )
    assert normalized["loop_n"] == 2
    assert normalized["duration"] == "1h"
    assert normalized["periods"]["test_end"] == "2025-01-02"
    assert normalized["horizon_profile"] == "short_1_5d"


@pytest.mark.no_database
def test_research_schedule_without_dates_keeps_a_rolling_policy() -> None:
    payload = _payload()
    payload.pop("periods")
    payload["period_mode"] = "rolling"

    normalized = normalize_research_schedule_payload(payload, max_loops=3)

    assert "periods" not in normalized
    assert normalized["period_mode"] == "rolling"
    assert normalized["period_policy"] == {
        "test_trading_days": 252,
        "embargo_trading_days": 20,
    }


@pytest.mark.no_database
def test_legacy_schedule_dates_roll_forward_unless_explicitly_frozen() -> None:
    payload = _payload()
    payload.pop("period_mode")

    normalized = normalize_research_schedule_payload(payload, max_loops=3)

    assert "periods" not in normalized
    assert normalized["period_mode"] == "rolling"


@pytest.mark.no_database
def test_research_schedule_rejects_unbounded_or_overlapping_requests() -> None:
    payload = _payload()
    payload["loop_n"] = 4
    with pytest.raises(ValueError, match="between 1 and 3"):
        normalize_research_schedule_payload(payload, max_loops=3)

    payload = _payload()
    payload["periods"]["valid_start"] = "2021-12-31"
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        normalize_research_schedule_payload(
            payload, max_loops=3, allow_explicit_periods=True
        )

    payload = _payload()
    payload["duration"] = "3h"
    with pytest.raises(ValueError, match="configured limit"):
        normalize_research_schedule_payload(
            payload,
            max_loops=3,
            max_duration="2h",
            allow_explicit_periods=True,
        )

    payload = _payload()
    payload.update(
        {
            "scenario": "fin_factor_report",
            "asset_ids": ["report-a", "report-b", "report-c"],
        }
    )
    payload.pop("horizon")
    with pytest.raises(ValueError, match="one report per loop"):
        normalize_research_schedule_payload(
            payload,
            max_loops=3,
            max_duration="2h",
            allow_explicit_periods=True,
        )


@pytest.mark.no_database
def test_operator_schedule_cannot_bypass_governed_windows_with_explicit_dates() -> None:
    with pytest.raises(ValueError, match="platform-internal"):
        normalize_research_schedule_payload(_payload(), max_loops=3)


@pytest.mark.no_database
def test_direct_research_api_cannot_accept_operator_selected_dates() -> None:
    with pytest.raises(ValueError, match="periods"):
        RDAgentRunRequest(
            objective="Research a low-turnover quality factor for CSI 300 enhancement.",
            dataset="cn-research",
            periods=_payload()["periods"],
        )


@pytest.mark.no_database
def test_campaign_factor_ranking_uses_only_passed_qlib_evidence() -> None:
    candidates = [
        {
            "id": "stable",
            "latest_evaluation": {
                "gate_status": "passed",
                "metrics": {
                    "icir": 0.8,
                    "rank_icir": 0.7,
                    "cost_adjusted_return": 0.05,
                    "turnover": 0.2,
                },
            },
        },
        {
            "id": "high-turnover",
            "latest_evaluation": {
                "gate_status": "passed",
                "metrics": {
                    "icir": 0.9,
                    "rank_icir": 0.8,
                    "cost_adjusted_return": 0.01,
                    "turnover": 0.6,
                },
            },
        },
        {
            "id": "rejected",
            "latest_evaluation": {
                "gate_status": "failed",
                "metrics": {"icir": 10.0, "rank_icir": 10.0},
            },
        },
    ]
    ranked = rank_factor_candidates(candidates, limit=5)
    assert [item["id"] for item in ranked] == ["stable", "high-turnover"]
    assert all(item["id"] != "rejected" for item in ranked)


@pytest.mark.no_database
def test_continuous_research_windows_use_actual_trading_days() -> None:
    calendar = [f"2024-01-{day:02d}" for day in range(1, 10)]
    periods = derive_rolling_research_periods(
        calendar,
        train_days=3,
        validation_days=2,
        test_days=2,
        embargo_days=1,
    )
    assert periods == {
        "train_start": "2024-01-02",
        "train_end": "2024-01-04",
        "valid_start": "2024-01-05",
        "valid_end": "2024-01-06",
        "test_start": "2024-01-08",
        "test_end": "2024-01-09",
    }
    assert calendar.index(periods["test_start"]) - calendar.index(periods["valid_end"]) == 2
    with pytest.raises(ValueError, match="requires 11"):
        derive_rolling_research_periods(
            calendar,
            train_days=5,
            validation_days=3,
            test_days=2,
            embargo_days=1,
        )


@pytest.mark.no_database
def test_default_periods_are_resolved_from_the_latest_qlib_calendar() -> None:
    calendar = [
        f"{year:04d}-{month:02d}-{day:02d}"
        for year in range(2010, 2030)
        for month in range(1, 13)
        for day in range(1, 29)
    ]
    periods, evidence = resolve_research_periods(calendar)

    assert periods["test_end"] == calendar[-1]
    assert calendar.index(periods["test_start"]) == len(calendar) - 252
    assert calendar.index(periods["valid_end"]) == len(calendar) - 273
    assert evidence["mode"] == "rolling_multi_profile_qlib_calendar_v2"
    assert {item["id"] for item in evidence["evaluation_profiles"]} == {
        "recent_3y",
        "balanced_5y",
        "robust_10y",
    }


@pytest.mark.no_database
def test_multi_profile_windows_share_one_final_oos() -> None:
    calendar: list[str] = []
    day = date(2010, 1, 1)
    while len(calendar) < 4000:
        if day.weekday() < 5:
            calendar.append(day.isoformat())
        day += timedelta(days=1)
    discovery, profiles = derive_multi_profile_research_periods(
        calendar,
        test_days=252,
        embargo_days=20,
    )

    assert discovery == next(item["periods"] for item in profiles if item["id"] == "recent_3y")
    assert len({item["periods"]["test_start"] for item in profiles}) == 1
    assert len({item["periods"]["test_end"] for item in profiles}) == 1
    effective_days = {
        item["id"]: calendar.index(item["periods"]["valid_end"])
        - calendar.index(item["periods"]["valid_start"])
        + 1
        for item in profiles
    }
    assert effective_days["recent_3y"] == 756
    assert effective_days["balanced_5y"] == 1260
    robust = next(item for item in profiles if item["id"] == "robust_10y")
    assert robust["validation_trading_days"] == 2520
    assert robust["requested_validation_trading_days"] == 2520
    assert robust["effective_validation_trading_days"] == effective_days["robust_10y"]
    assert robust["effective_validation_trading_days"] < 2520
    assert robust["periods"]["valid_start"] == "2015-08-03"
    assert robust["authoritative_cost_schedule_effective_from"] == "2015-08-01"
    assert robust["authoritative_cost_schedule_first_trading_day"] == "2015-08-03"
    assert robust["validation_window_truncated"] is True
    assert robust["validation_window_truncation_reason"] == (
        "authoritative_cn_cost_schedule_starts_after_requested_validation"
    )
    robust_train_days = (
        calendar.index(robust["periods"]["train_end"])
        - calendar.index(robust["periods"]["train_start"])
        + 1
    )
    assert robust_train_days == robust["effective_training_trading_days"]
    assert robust_train_days >= 252
    assert (
        calendar.index(robust["periods"]["valid_start"])
        - calendar.index(robust["periods"]["train_end"])
        - 1
        == 2
    )
    assert (
        calendar.index(robust["periods"]["test_start"])
        - calendar.index(robust["periods"]["valid_end"])
        - 1
        == 20
    )


@pytest.mark.no_database
def test_explicit_pre_cost_validation_is_rejected_instead_of_truncated() -> None:
    calendar = [
        (date(2010, 1, 1) + timedelta(days=offset)).isoformat()
        for offset in range(4000)
    ]
    explicit = {
        "train_start": "2010-01-01",
        "train_end": "2015-07-28",
        "valid_start": "2015-07-31",
        "valid_end": "2018-12-31",
        "test_start": "2019-01-07",
        "test_end": "2019-12-31",
    }

    with pytest.raises(ValueError, match="explicit research validation starts before"):
        resolve_research_periods(calendar, periods=explicit)

    assert explicit["valid_start"] == "2015-07-31"


@pytest.mark.no_database
def test_rolling_profiles_fail_closed_when_non_robust_history_predates_costs() -> None:
    calendar: list[str] = []
    day = date(2005, 1, 3)
    while len(calendar) < 4000:
        if day.weekday() < 5:
            calendar.append(day.isoformat())
        day += timedelta(days=1)

    with pytest.raises(ValueError, match="balanced_5y validation starts before"):
        derive_multi_profile_research_periods(
            calendar,
            test_days=252,
            embargo_days=5,
        )


@pytest.mark.no_database
def test_factor_and_model_use_the_same_cost_covered_rolling_window_contract() -> None:
    calendar: list[str] = []
    day = date(2010, 1, 1)
    while len(calendar) < 4000:
        if day.weekday() < 5:
            calendar.append(day.isoformat())
        day += timedelta(days=1)
    resolutions = []
    for scenario, feature_set_id in (
        ("fin_factor", "governed-baseline"),
        ("fin_model", "governed-baseline"),
    ):
        payload = {
            "scenario": scenario,
            "objective": "Research reproducible A-share signals under governed windows.",
            "dataset": "cn-research",
            "loop_n": 1,
            "duration": "30m",
            "requested_by": "test-scheduler",
            "period_mode": "rolling",
            "horizon": "swing",
        }
        if feature_set_id is not None:
            payload["feature_set_id"] = feature_set_id
        normalized = normalize_research_schedule_payload(payload, max_loops=2)
        resolutions.append(
            resolve_research_periods(
                calendar,
                period_policy=normalized["period_policy"],
                horizon_profile=normalized["horizon_profile"],
            )
        )

    assert resolutions[0] == resolutions[1]
    robust = next(
        item
        for item in resolutions[0][1]["evaluation_profiles"]
        if item["id"] == "robust_10y"
    )
    assert resolutions[0][1]["horizon_profile"] == "swing_1_6m"
    assert robust["periods"]["valid_start"] < robust["periods"]["valid_end"]


@pytest.mark.no_database
def test_multi_profile_ranking_requires_recent_and_balanced_consensus() -> None:
    def evaluation(
        profile_id: str,
        *,
        candidate_id: str = "passing",
        gate: str = "passed",
        direction: str = "original",
    ) -> dict:
        suffix = {"recent_3y": "1", "robust_10y": "2", "balanced_5y": "3"}[profile_id]
        valid_start = {
            "recent_3y": "2022-01-03",
            "balanced_5y": "2020-01-02",
            "robust_10y": "2015-01-05",
        }[profile_id]
        train_end = {
            "recent_3y": "2021-12-31",
            "balanced_5y": "2019-12-31",
            "robust_10y": "2015-01-02",
        }[profile_id]
        return {
            "id": suffix * 32,
            "factor_candidate_id": candidate_id,
            "dataset_identity_sha256": "d" * 64,
            "train_start": "2010-01-04",
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": "2024-12-20",
            "test_start": "2025-01-02",
            "test_end": "2025-12-31",
            "candidate_code_sha256": "a" * 64,
            "candidate_values_sha256": "b" * 64,
            "evidence_sha256": suffix * 64,
            "metrics_sha256": suffix * 64,
            "gate_status": gate,
            "metrics": {
                "research_profile": {"id": profile_id},
                "direction": direction,
                "coverage_gate_passed": True,
                "icir": 0.5,
                "rank_icir": 0.4,
                "cost_adjusted_return": 0.03,
                "turnover": 0.1,
            },
        }

    passing = {
        "id": "passing",
        "code_sha256": "a" * 64,
        "values_sha256": "b" * 64,
        "profile_evaluations": [
            evaluation("recent_3y"),
            evaluation("robust_10y", gate="failed"),
            evaluation("balanced_5y"),
        ],
    }
    unstable = {
        "id": "unstable",
        "code_sha256": "a" * 64,
        "values_sha256": "b" * 64,
        "profile_evaluations": [
            evaluation("recent_3y", candidate_id="unstable"),
            evaluation("robust_10y", candidate_id="unstable", direction="inverted"),
            evaluation("balanced_5y", candidate_id="unstable"),
        ],
    }

    ranked = rank_multi_profile_candidates([unstable, passing], limit=5)

    assert [item["id"] for item in ranked] == ["passing"]
    assert ranked[0]["profile_consensus"]["evaluation_ids"] == {
        "balanced_5y": "3" * 32,
        "recent_3y": "1" * 32,
        "robust_10y": "2" * 32,
    }


@pytest.mark.no_database
def test_continuous_research_reserves_configured_trading_day_embargo() -> None:
    calendar = [f"2024-01-{day:02d}" for day in range(1, 13)]
    periods = derive_rolling_research_periods(
        calendar,
        train_days=3,
        validation_days=2,
        test_days=2,
        embargo_days=5,
    )

    assert periods == {
        "train_start": "2024-01-01",
        "train_end": "2024-01-03",
        "valid_start": "2024-01-04",
        "valid_end": "2024-01-05",
        "test_start": "2024-01-11",
        "test_end": "2024-01-12",
    }


@pytest.mark.no_database
def test_continuous_research_never_crosses_dataset_lineage() -> None:
    datasets = [
        {
            "name": "wrong-newer",
            "ready": True,
            "reproducible": True,
            "lineage_verified": True,
            "lineage_id": "other",
            "end_date": "2026-07-14",
            "provenance": {"dataset_identity_sha256": "b" * 64},
        },
        {
            "name": "approved",
            "ready": True,
            "reproducible": True,
            "lineage_verified": True,
            "lineage_id": "lineage-a",
            "end_date": "2026-07-13",
            "provenance": {"dataset_identity_sha256": "a" * 64},
        },
    ]
    selected = select_latest_program_dataset(datasets, lineage_id="lineage-a")
    assert selected and selected["name"] == "approved"


@pytest.mark.no_database
def test_continuous_research_rebase_allows_only_compatible_contract_extension() -> None:
    contract = {
        "dataset_contract_sha256": "c" * 64,
        "field_contract_version": "daily-v1",
        "frequency": "day",
        "eligibility_contract_version": "eligibility-v1",
        "fields": ["open", "close"],
        "field_units": {"open": "CNY", "close": "CNY"},
        "source_start_date": "2008-01-01",
    }
    anchor = {
        "name": "old",
        "ready": True,
        "reproducible": True,
        "lineage_verified": True,
        "lineage_id": "a" * 64,
        "end_date": "2026-08-21",
        "provenance": {**contract, "dataset_identity_sha256": "1" * 64},
    }
    compatible = {
        **anchor,
        "name": "new",
        "lineage_id": "b" * 64,
        "end_date": "2026-08-25",
        "provenance": {
            **contract,
            "dataset_contract_sha256": "e" * 64,
            "fields": ["open", "close", "pit_valuation"],
            "field_units": {
                "open": "CNY",
                "close": "CNY",
                "pit_valuation": "ratio",
            },
            "dataset_identity_sha256": "2" * 64,
        },
    }
    incompatible = {
        **compatible,
        "name": "changed-fields",
        "end_date": "2026-08-26",
        "lineage_id": "d" * 64,
        "provenance": {
            **compatible["provenance"],
            "dataset_identity_sha256": "3" * 64,
            "field_units": {
                "open": "USD",
                "close": "CNY",
                "pit_valuation": "ratio",
            },
        },
    }
    selected = select_latest_program_rebase_dataset(
        [anchor, compatible, incompatible], anchor=anchor
    )
    assert selected and selected["name"] == "new"
