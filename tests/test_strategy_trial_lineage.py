from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from quant_platform.strategy_trial_lineage import build_strategy_trial_lineage

pytestmark = pytest.mark.no_database


RECEIPT_SHA256 = "a" * 64


def _version_configs() -> dict[str, dict[str, str]]:
    return {
        "source-version": {
            "recipe_id": "short_relative_strength",
            "recipe_version": "source-v1",
        },
        "target-version": {
            "recipe_id": "short_relative_strength",
            "recipe_version": "target-v2",
        },
        "ordinary-version": {
            "recipe_id": "short_relative_strength",
            "recipe_version": "ordinary-v3",
        },
    }


def _repair(*, performance_information_used: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        receipt_sha256=RECEIPT_SHA256,
        source_audit_event_id=17,
        source_backtest_ids_json=["source-backtest"],
        target_strategy_version_ids_json=["target-version"],
        verification_json={
            "source_backtest_ids": ["source-backtest"],
            "target_strategy_version_ids": ["target-version"],
            "performance_information_used": performance_information_used,
        },
    )


def _receipt(*, performance_information_used: bool = False) -> dict[str, Any]:
    return {
        "target_recipe_version": "target-v2",
        "performance_information_used": performance_information_used,
        "members": [
            {
                "backtest_id": "source-backtest",
                "strategy_version_id": "source-version",
            }
        ],
    }


def _lineage(
    *,
    repair: SimpleNamespace | None = None,
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repair = repair or _repair()
    receipt = receipt or _receipt()

    def validate(_: Any, *, expected_receipt_sha256: str) -> dict[str, Any]:
        assert expected_receipt_sha256 == RECEIPT_SHA256
        return receipt

    return build_strategy_trial_lineage(
        version_configs=_version_configs(),
        backtests_by_id={
            "source-backtest": SimpleNamespace(
                id="source-backtest",
                strategy_version_id="source-version",
            )
        },
        repair_rows=[repair],
        repair_audits={17: SimpleNamespace(id=17)},
        audit_validator=validate,
    )


def test_valid_pre_result_repair_shares_one_trial_but_keeps_physical_history() -> None:
    result = _lineage()

    assert result["strategy_version_count"] == 3
    assert result["strategy_trial_count"] == 2
    assert result["strategy_trial_components"] == [
        {
            "trial_root_strategy_version_id": "ordinary-version",
            "strategy_version_ids": ["ordinary-version"],
        },
        {
            "trial_root_strategy_version_id": "source-version",
            "strategy_version_ids": ["source-version", "target-version"],
        },
    ]
    assert result["accepted_pre_result_repair_links"] == [
        {
            "classification": "pre_result_implementation_repair",
            "receipt_sha256": RECEIPT_SHA256,
            "repair_source_backtest_ids": ["source-backtest"],
            "source_backtest_ids": ["source-backtest"],
            "source_strategy_version_ids": ["source-version"],
            "target_strategy_version_ids": ["target-version"],
            "performance_information_used": False,
        }
    ]
    assert result["rejected_pre_result_repair_receipts"] == []


@pytest.mark.parametrize("performance_location", ["registry", "receipt"])
def test_performance_informed_repair_fails_closed(performance_location: str) -> None:
    result = _lineage(
        repair=_repair(performance_information_used=performance_location == "registry"),
        receipt=_receipt(performance_information_used=performance_location == "receipt"),
    )

    assert result["strategy_trial_count"] == 3
    assert result["accepted_pre_result_repair_links"] == []
    assert len(result["rejected_pre_result_repair_receipts"]) == 1
    assert result["rejected_pre_result_repair_receipts"][0]["receipt_sha256"] == (
        RECEIPT_SHA256
    )


def test_cross_recipe_repair_fails_closed() -> None:
    versions = _version_configs()
    versions["target-version"]["recipe_id"] = "different_economic_strategy"

    result = build_strategy_trial_lineage(
        version_configs=versions,
        backtests_by_id={
            "source-backtest": SimpleNamespace(
                id="source-backtest",
                strategy_version_id="source-version",
            )
        },
        repair_rows=[_repair()],
        repair_audits={17: SimpleNamespace(id=17)},
        audit_validator=lambda *_args, **_kwargs: _receipt(),
    )

    assert result["strategy_trial_count"] == 3
    assert result["accepted_pre_result_repair_links"] == []
    assert result["rejected_pre_result_repair_receipts"] == [
        {
            "receipt_sha256": RECEIPT_SHA256,
            "reason": "repair receipt economic family changed",
        }
    ]


def test_audit_without_append_only_registry_never_reduces_trials() -> None:
    result = build_strategy_trial_lineage(
        version_configs=_version_configs(),
        backtests_by_id={},
        repair_rows=[],
        repair_audits={17: SimpleNamespace(id=17)},
    )

    assert result["strategy_version_count"] == 3
    assert result["strategy_trial_count"] == 3
    assert result["accepted_pre_result_repair_links"] == []
    assert result["rejected_pre_result_repair_receipts"] == []
