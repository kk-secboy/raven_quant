from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from quant_platform.promotion import (
    _count_closed_round_trips,
    _review_period_key,
    build_horizon_review_evidence,
    forward_gate_thresholds_for_horizon,
)
from quant_platform.research_horizon import (
    LEGACY_AMBIGUOUS,
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    horizon_columns_from_config,
    normalize_horizon_config,
    primary_label_horizon_sessions,
    primary_label_policy_contract,
    require_horizon_row,
    require_label_horizon,
    research_horizon_contract,
)

pytestmark = pytest.mark.no_database


@pytest.mark.parametrize(
    ("profile", "labels", "decision", "review", "holding", "purge", "oos"),
    [
        (SHORT_1_5D, [1, 2, 3, 5], 1, 1, (1, 3, 5), 6, 252),
        (SWING_1_6M, [21, 63, 126], 5, 5, (21, 63, 126), 127, 504),
        (LONG_1_3Y, [63, 126, 252], 21, 21, (252, 504, 756), 253, 756),
    ],
)
def test_supported_horizon_profiles_are_explicit_and_leakage_safe(
    profile: str,
    labels: list[int],
    decision: int,
    review: int,
    holding: tuple[int, int, int],
    purge: int,
    oos: int,
) -> None:
    contract = research_horizon_contract(profile)

    assert contract.to_dict()["label_horizons_sessions"] == labels
    assert contract.decision_interval_sessions == decision
    assert contract.review_interval_sessions == review
    assert (
        contract.holding_min_sessions,
        contract.holding_target_sessions,
        contract.holding_max_sessions,
    ) == holding
    assert contract.execution_lag_sessions == 1
    assert contract.purge_sessions == purge
    assert contract.embargo_sessions >= max(labels)
    assert contract.sealed_oos_required is True
    assert contract.sealed_oos_sessions == oos
    assert len(contract.sha256) == 64


def test_missing_profile_is_canonical_legacy_without_invented_values() -> None:
    normalized = normalize_horizon_config({"signal_period": 21})
    contract = normalized["horizon_contract"]

    assert normalized["horizon_profile"] == LEGACY_AMBIGUOUS
    assert contract["label_horizons_sessions"] == []
    assert contract["holding_target_sessions"] is None
    assert contract["sealed_oos_required"] is False
    assert normalized["horizon_contract_sha256"] == research_horizon_contract(
        LEGACY_AMBIGUOUS
    ).sha256


def test_submitted_horizon_contract_cannot_drift_from_profile() -> None:
    canonical = normalize_horizon_config({"horizon_profile": SHORT_1_5D})
    changed = dict(canonical["horizon_contract"])
    changed["holding_max_sessions"] = 6

    with pytest.raises(ValueError, match="differs from the canonical"):
        normalize_horizon_config(
            {"horizon_profile": SHORT_1_5D, "horizon_contract": changed}
        )
    with pytest.raises(ValueError, match="SHA-256"):
        normalize_horizon_config(
            {"horizon_profile": SHORT_1_5D, "horizon_contract_sha256": "0" * 64}
        )


def test_denormalized_strategy_columns_are_verified_against_seal() -> None:
    row = horizon_columns_from_config({"horizon_profile": SWING_1_6M})
    assert require_horizon_row(row).horizon_profile == SWING_1_6M

    changed = dict(row)
    changed["holding_target_sessions"] = 61
    with pytest.raises(ValueError, match="holding_target_sessions"):
        require_horizon_row(changed)


def test_label_horizon_must_belong_to_selected_track() -> None:
    require_label_horizon(SHORT_1_5D, 5)
    require_label_horizon(SWING_1_6M, 63)
    require_label_horizon(LONG_1_3Y, 252)

    with pytest.raises(ValueError, match="not allowed"):
        require_label_horizon(SWING_1_6M, 2)
    with pytest.raises(ValueError, match="no admissible"):
        require_label_horizon(LEGACY_AMBIGUOUS, 1)


def test_primary_prediction_labels_are_stable_per_horizon() -> None:
    assert primary_label_horizon_sessions(SHORT_1_5D) == 5
    assert primary_label_horizon_sessions(SWING_1_6M) == 63
    assert primary_label_horizon_sessions(LONG_1_3Y) == 252

    with pytest.raises(ValueError, match="no primary"):
        primary_label_horizon_sessions(LEGACY_AMBIGUOUS)


def test_primary_label_policy_has_an_independent_frozen_digest() -> None:
    policy = primary_label_policy_contract()

    assert policy == {
        "contract_version": "primary-label-policy-v1",
        "horizon_primary_labels_sessions": {
            "long_1_3y": 252,
            "short_1_5d": 5,
            "swing_1_6m": 63,
        },
        "legacy_ambiguous_executable": False,
        "policy_sha256": (
            "f90f34e67b4721c0e7b82181007872cc099093e80ea92f8ed3d2d88e3e7adfdc"
        ),
    }
    assert research_horizon_contract(SHORT_1_5D).sha256 == (
        "a895312d55f19ffaf35e90c4bd7003af337c94e18b61e27b4ee62a584860e1e9"
    )


def test_forward_gate_minima_are_horizon_specific_not_calendar_day_proxies() -> None:
    short = forward_gate_thresholds_for_horizon(SHORT_1_5D)
    swing = forward_gate_thresholds_for_horizon(SWING_1_6M)
    long = forward_gate_thresholds_for_horizon(LONG_1_3Y)

    assert (
        short.min_forward_calendar_days,
        short.min_forward_trading_days,
        short.min_decision_batches,
        short.min_closed_round_trips,
    ) == (0, 60, 0, 0)
    assert (
        swing.min_forward_trading_days,
        swing.min_review_events,
        swing.min_closed_round_trips,
    ) == (120, 0, 0)
    assert (
        long.min_forward_trading_days,
        long.min_review_events,
        long.min_financial_report_reviews,
    ) == (120, 0, 0)
    # The live recommendation gate does not shorten the historically sealed
    # strategy OOS evidence.
    assert research_horizon_contract(LONG_1_3Y).sealed_oos_sessions == 756


def test_closed_round_trips_require_a_prior_buy_and_full_exit() -> None:
    def fill(instrument: str, side: str, quantity: int, day: int) -> SimpleNamespace:
        return SimpleNamespace(
            instrument=instrument,
            side=side,
            quantity=quantity,
            executed_at=datetime(2026, 1, day, 7, 0, tzinfo=UTC),
        )

    fills = [
        fill("A", "buy", 100, 2),
        fill("A", "sell", 40, 3),
        fill("B", "sell", 100, 3),
        fill("A", "sell", 60, 4),
        fill("C", "buy", 100, 2),
        fill("C", "sell", 120, 3),
        fill("C", "buy", 50, 4),
        fill("C", "sell", 50, 5),
        fill("D", "buy", 100, 2),
        fill("D", "sell", 50, 3),
    ]

    assert _count_closed_round_trips(fills) == (2, 2)


def test_closed_round_trips_reject_same_day_t_plus_one_violation() -> None:
    fills = [
        SimpleNamespace(
            instrument="A",
            side="buy",
            quantity=100,
            executed_at=datetime(2026, 1, 2, 2, 0, tzinfo=UTC),
        ),
        SimpleNamespace(
            instrument="A",
            side="sell",
            quantity=100,
            executed_at=datetime(2026, 1, 2, 7, 0, tzinfo=UTC),
        ),
    ]

    assert _count_closed_round_trips(fills) == (0, 1)


def test_forward_reviews_are_bucketed_by_horizon_cadence() -> None:
    first = datetime(2026, 8, 3, 3, 0, tzinfo=UTC)
    same_week = datetime(2026, 8, 9, 3, 0, tzinfo=UTC)
    next_week = datetime(2026, 8, 10, 3, 0, tzinfo=UTC)

    assert _review_period_key(SWING_1_6M, first) == _review_period_key(
        SWING_1_6M, same_week
    )
    assert _review_period_key(SWING_1_6M, first) != _review_period_key(
        SWING_1_6M, next_week
    )
    assert _review_period_key(LONG_1_3Y, first) == "2026-08"


def test_financial_review_marker_is_sealed_and_requires_report_identity() -> None:
    review = build_horizon_review_evidence(
        event_id="2026q2-review",
        review_type="financial_report_review",
        horizon_profile=LONG_1_3Y,
        completed_at=datetime(2026, 8, 28, 9, 0, tzinfo=UTC),
        strategy_version_id="long-version",
        signal_date=date(2026, 8, 28),
        dataset_identity_sha256="a" * 64,
        trigger_source="pit_financial_announcement",
        trigger_effective_date=date(2026, 8, 28),
        report_period="2026Q2",
        announcement_date=date(2026, 8, 27),
        previous_signal_date=date(2026, 8, 26),
        source_datasets=["fina_indicator"],
        source_event_count=1,
        source_event_sha256="b" * 64,
        report_periods=["2026Q2"],
        reviewed_instruments=["SH600000"],
        review_scope_sha256="c" * 64,
    )

    assert review["status"] == "completed"
    assert review["announcement_date"] == "2026-08-27"
    assert len(review["evidence_sha256"]) == 64

    with pytest.raises(ValueError, match="report period"):
        build_horizon_review_evidence(
            event_id="invalid",
            review_type="financial_report_review",
            horizon_profile=LONG_1_3Y,
            completed_at=datetime(2026, 8, 28, 9, 0, tzinfo=UTC),
            strategy_version_id="long-version",
            signal_date=date(2026, 8, 28),
            dataset_identity_sha256="a" * 64,
            trigger_source="pit_financial_announcement",
            trigger_effective_date=date(2026, 8, 28),
            previous_signal_date=date(2026, 8, 26),
            source_datasets=["fina_indicator"],
            source_event_count=1,
            source_event_sha256="b" * 64,
            report_periods=["2026Q2"],
            reviewed_instruments=["SH600000"],
            review_scope_sha256="c" * 64,
        )
