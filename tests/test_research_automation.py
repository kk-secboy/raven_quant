from __future__ import annotations

import pytest

from quant_platform.api import ResearchPeriods
from quant_platform.research_automation import (
    derive_rolling_research_periods,
    normalize_research_schedule_payload,
    rank_factor_candidates,
    select_latest_program_dataset,
)


@pytest.mark.no_database
def test_default_research_periods_cover_the_2008_multiregime_history() -> None:
    periods = ResearchPeriods()

    assert periods.train_start.isoformat() == "2008-01-01"
    assert periods.train_end.isoformat() == "2017-12-31"
    assert periods.valid_start.isoformat() == "2018-01-01"
    assert periods.valid_end.isoformat() == "2020-12-31"
    assert periods.test_start.isoformat() == "2021-01-01"
    assert (periods.test_end - periods.train_start).days >= 3652


def _payload() -> dict:
    return {
        "objective": "Research a low-turnover quality factor for CSI 300 enhancement.",
        "dataset": "cn-research",
        "loop_n": 2,
        "duration": "1h",
        "requested_by": "research-scheduler",
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
    normalized = normalize_research_schedule_payload(_payload(), max_loops=3)
    assert normalized["loop_n"] == 2
    assert normalized["duration"] == "1h"
    assert normalized["periods"]["test_end"] == "2025-01-02"


@pytest.mark.no_database
def test_research_schedule_rejects_unbounded_or_overlapping_requests() -> None:
    payload = _payload()
    payload["loop_n"] = 4
    with pytest.raises(ValueError, match="between 1 and 3"):
        normalize_research_schedule_payload(payload, max_loops=3)

    payload = _payload()
    payload["periods"]["valid_start"] = "2021-12-31"
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        normalize_research_schedule_payload(payload, max_loops=3)


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
    )
    assert periods == {
        "train_start": "2024-01-03",
        "train_end": "2024-01-05",
        "valid_start": "2024-01-06",
        "valid_end": "2024-01-07",
        "test_start": "2024-01-08",
        "test_end": "2024-01-09",
    }
    with pytest.raises(ValueError, match="requires 10"):
        derive_rolling_research_periods(
            calendar,
            train_days=5,
            validation_days=3,
            test_days=2,
        )


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
