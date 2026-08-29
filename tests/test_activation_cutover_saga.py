from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from quant_platform.promotion import PromotionStore
from quant_platform.scheduler import SchedulerEngine

pytestmark = pytest.mark.no_database


class _Promotions:
    def __init__(self) -> None:
        self.pending = [
            {
                "strategy_version_id": "new-short",
                "activation_token": "token-short",
            }
        ]
        self.completed: list[dict[str, Any]] = []
        self.rolled_back: list[dict[str, Any]] = []

    def pending_activation_cutovers(self) -> list[dict[str, Any]]:
        return list(self.pending)

    def complete_activation_cutover(self, version_id: str, **kwargs: Any) -> None:
        self.completed.append({"version_id": version_id, **kwargs})

    def rollback_activation_cutover(self, version_id: str, **kwargs: Any) -> None:
        self.rolled_back.append({"version_id": version_id, **kwargs})


class _Alerts:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> None:
        self.items.append(kwargs)


def _scheduler() -> tuple[SchedulerEngine, _Promotions, _Alerts]:
    scheduler = object.__new__(SchedulerEngine)
    promotions = _Promotions()
    alerts = _Alerts()
    scheduler.promotions = promotions
    scheduler.alerts = alerts
    return scheduler, promotions, alerts


def test_pending_member_snapshots_neither_commits_nor_rolls_back_cutover() -> None:
    scheduler, promotions, alerts = _scheduler()

    scheduler._settle_activation_cutovers(
        {"status": "waiting_for_member_targets", "advanced": False},
        now=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert promotions.completed == []
    assert promotions.rolled_back == []
    assert alerts.items == []


@pytest.mark.parametrize(
    "status",
    [
        "onboarding_required",
        "waiting_for_allocation_evidence",
        "waiting_for_member_targets",
        "waiting_for_verified_horizons",
    ],
)
def test_expected_account_prerequisites_keep_cutover_pending(status: str) -> None:
    scheduler, promotions, alerts = _scheduler()

    scheduler._settle_activation_cutovers(
        {"status": status, "advanced": False},
        now=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert promotions.completed == []
    assert promotions.rolled_back == []
    assert alerts.items == []


def test_usable_account_commits_only_the_versions_in_its_frozen_membership() -> None:
    scheduler, promotions, alerts = _scheduler()
    evidence = {
        "status": "no_action",
        "allocation_id": "allocation-1",
        "netting_plan_id": "plan-1",
        "strategy_version_ids": {
            "short_1_5d": "new-short",
            "swing_1_6m": "middle",
            "long_1_3y": "long",
        },
        "member_snapshot_evidence": {
            "new-short": {"snapshot_id": "snapshot-short"},
            "middle": {"snapshot_id": "snapshot-middle"},
            "long": {"snapshot_id": "snapshot-long"},
        },
    }

    scheduler._settle_activation_cutovers(
        evidence,
        now=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert promotions.completed == [
        {
            "version_id": "new-short",
            "activation_token": "token-short",
            "account_evidence": evidence,
        }
    ]
    assert promotions.rolled_back == []
    assert alerts.items == []


def test_hard_account_failure_requests_atomic_cutover_rollback() -> None:
    scheduler, promotions, alerts = _scheduler()

    scheduler._settle_activation_cutovers(
        {"status": "blocked", "reason": "daily dataset unavailable"},
        now=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert promotions.completed == []
    assert promotions.rolled_back == [
        {
            "version_id": "new-short",
            "activation_token": "token-short",
            "reason": (
                "Three-horizon account cutover failed: daily dataset unavailable"
            ),
        }
    ]
    assert alerts.items == []


def test_pending_cutover_projects_exactly_one_replaced_incumbent() -> None:
    store = object.__new__(PromotionStore)
    store.pending_activation_cutovers = lambda **_kwargs: [
        {
            "replaced_incumbents": [
                {"strategy_version_id": "last-verified-short"}
            ]
        }
    ]

    assert (
        store.serving_incumbent_for_pending_cutover("short_1_5d")
        == "last-verified-short"
    )


def test_pending_cutover_never_guesses_between_multiple_incumbents() -> None:
    store = object.__new__(PromotionStore)
    store.pending_activation_cutovers = lambda **_kwargs: [
        {
            "replaced_incumbents": [
                {"strategy_version_id": "one"},
                {"strategy_version_id": "two"},
            ]
        }
    ]

    assert store.serving_incumbent_for_pending_cutover("short_1_5d") is None
