from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime, timedelta

import pytest

from quant_platform import transparent_baseline_lockbox as lockbox
from quant_platform import transparent_baseline_repair as repair

pytestmark = pytest.mark.no_database


def _files() -> list[dict[str, object]]:
    return [
        {"path": "manifest.json", "bytes": 2, "sha256": "a" * 64},
        {"path": "baseline/runtime.json", "bytes": 3, "sha256": "b" * 64},
    ]


def _patch_inventory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_selection_sha256: str | None = None,
) -> None:
    backtest_id = next(iter(lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BACKTEST_IDS))
    binding = dict(lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS[backtest_id])
    binding["artifact_inventory_sha256"] = lockbox.canonical_sha256(_files())
    monkeypatch.setitem(
        lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS,
        backtest_id,
        binding,
    )
    aggregate = lockbox.canonical_sha256(
        [
            {
                "backtest_id": backtest_id,
                "artifact_inventory_sha256": binding["artifact_inventory_sha256"],
            }
        ]
    )
    monkeypatch.setattr(
        lockbox,
        "FILL_AWARE_HOLDING_AGE_SOURCE_ARTIFACT_INVENTORIES_SHA256",
        aggregate,
    )
    monkeypatch.setattr(
        repair,
        "FILL_AWARE_HOLDING_AGE_SOURCE_ARTIFACT_INVENTORIES_SHA256",
        aggregate,
    )
    if source_selection_sha256 is not None:
        monkeypatch.setattr(
            lockbox,
            "FILL_AWARE_HOLDING_AGE_SOURCE_SELECTION_SHA256",
            source_selection_sha256,
        )
        monkeypatch.setattr(
            repair,
            "FILL_AWARE_HOLDING_AGE_SOURCE_SELECTION_SHA256",
            source_selection_sha256,
        )


def _member() -> dict[str, object]:
    backtest_id = next(iter(lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BACKTEST_IDS))
    binding = lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS[backtest_id]
    return {
        "backtest_id": backtest_id,
        "strategy_version_id": binding["strategy_version_id"],
        "job_id": binding["job_id"],
        "dataset": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET,
        "periods": dict(binding["periods"]),
        "status": "failed",
        "job_status": "failed",
        "error": lockbox.FILL_AWARE_HOLDING_AGE_ERROR,
        "metrics_absent": True,
        "result_absent": True,
        "files": _files(),
    }


def _receipt(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    _patch_inventory(monkeypatch)
    return repair.build_fill_aware_holding_age_receipt([_member()])


def _calendar() -> list[str]:
    day = date(2019, 11, 27)
    days: list[date] = []
    while len(days) < 2897:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    return [value.isoformat() for value in reversed(days)]


def _source_batch() -> dict[str, object]:
    member = {
        "oos_vintage_id": "0" * 32,
        "strategy_version_id": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS[
                next(iter(lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BACKTEST_IDS))
            ]["strategy_version_id"]
        ),
        "recipe_id": "short_relative_strength",
        "horizon_profile": "short_1_5d",
        "recipe_version": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_RECIPE_VERSION,
        "test_start": "2018-11-08",
        "test_end": "2019-11-20",
        "first_opened_at": datetime(2026, 8, 30, tzinfo=UTC).isoformat(),
        "sealed_member_set_sha256": "c" * 64,
    }
    return {
        "batch_sha256": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256,
        "recipe_version": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_RECIPE_VERSION,
        "earliest_final_oos_start": "2018-11-08",
        "latest_final_oos_end": "2019-11-20",
        "members": [member],
        "members_sha256": lockbox.canonical_sha256([member]),
    }


def test_v6_receipt_is_exactly_one_allowlisted_failed_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(monkeypatch)

    assert lockbox.validate_pre_result_repair_receipt(receipt) == receipt
    assert receipt["contract_version"] == lockbox.PRE_RESULT_REPAIR_CONTRACT_VERSION_V6
    assert len(receipt["members"]) == 1

    duplicated = deepcopy(receipt)
    duplicated["members"] = [deepcopy(receipt["members"][0])] * 2
    payload = {key: value for key, value in duplicated.items() if key != "receipt_sha256"}
    duplicated["receipt_sha256"] = lockbox.canonical_sha256(payload)
    with pytest.raises(ValueError, match="exactly 1 attempt"):
        lockbox.validate_pre_result_repair_receipt(duplicated)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backtest_id", "1" * 32, "not exact"),
        ("job_id", "2" * 32, "source binding"),
        ("strategy_version_id", "3" * 32, "source binding"),
        ("error", "ValueError: another failure", "error is not exact"),
        ("result_absent", False, "had a result"),
    ],
)
def test_v6_receipt_rejects_any_other_retry(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    receipt = _receipt(monkeypatch)
    receipt["members"][0][field] = value
    payload = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = lockbox.canonical_sha256(payload)

    with pytest.raises(ValueError, match=message):
        lockbox.validate_pre_result_repair_receipt(receipt)


def test_v2_history_selection_names_only_the_exact_ignored_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = _calendar()
    source_selection = lockbox.build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_RECIPE_VERSION,
        prior_batches=[],
    )
    _patch_inventory(
        monkeypatch,
        source_selection_sha256=source_selection["selection_sha256"],
    )
    receipt = repair.build_fill_aware_holding_age_receipt([_member()])

    evidence = lockbox.build_pre_result_repair_history_selection(
        calendar_days=calendar,
        current_recipe_version=lockbox.FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
        source_selection=source_selection,
        repaired_source_batch=_source_batch(),
        repair_receipt=receipt,
    )

    assert lockbox.validate_unopened_history_selection(
        evidence,
        calendar_days=calendar,
    ) == evidence
    assert evidence["repaired_source_batch_sha256"] == (
        lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256
    )
    assert evidence["performance_information_used"] is False
    assert evidence["selected_calendar_end"] == source_selection["selected_calendar_end"]

    tampered = deepcopy(evidence)
    tampered["repaired_source_batch"]["batch_sha256"] = "d" * 64
    payload = {key: value for key, value in tampered.items() if key != "selection_sha256"}
    tampered["selection_sha256"] = lockbox.canonical_sha256(payload)
    with pytest.raises(ValueError, match="source batch changed"):
        lockbox.validate_unopened_history_selection(tampered, calendar_days=calendar)


def test_v6_registry_accepts_one_member_and_rejects_an_extra_lane() -> None:
    backtest_id = next(iter(lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BACKTEST_IDS))
    binding = lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS[backtest_id]
    target_version_id = "4" * 32
    target_batch = "5" * 64
    receipt_sha = "6" * 64
    verification = {
        "contract_version": lockbox.PRE_RESULT_REPAIR_REGISTRY_VERSION,
        "receipt_sha256": receipt_sha,
        "source_audit_event_id": 17,
        "source_batch_sha256": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256,
        "target_batch_sha256": target_batch,
        "source_dataset_lineage_id": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
        ),
        "source_dataset_identity_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "target_dataset": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET,
        "target_dataset_identity_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "target_dataset_lineage_id": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
        ),
        "target_recipe_version": lockbox.FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
        "receipt_contract_version": lockbox.PRE_RESULT_REPAIR_CONTRACT_VERSION_V6,
        "repair_generation": lockbox.FILL_AWARE_HOLDING_AGE_REPAIR_GENERATION,
        "source_release_commit": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_COMMIT,
        "source_runner_expected_sha256": None,
        "source_runner_observed_sha256": None,
        "source_runner_sha256": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_RUNNER_SHA256,
        "source_runtime_bundle_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BUNDLE_SHA256
        ),
        "target_runtime_bundle_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_TARGET_BUNDLE_SHA256
        ),
        lockbox.TRANSPARENT_BASELINE_RUNNER_FIELD: (
            lockbox.FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256
        ),
        "packaging_contract_version": None,
        "runtime_contract_version": (
            lockbox.FILL_AWARE_HOLDING_AGE_RUNTIME_CONTRACT_VERSION
        ),
        "source_artifact_inventories_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_ARTIFACT_INVENTORIES_SHA256
        ),
        "target_change_codes": list(
            lockbox.FILL_AWARE_HOLDING_AGE_TARGET_CHANGE_CODES
        ),
        "source_unopened_history_selection_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_SELECTION_SHA256
        ),
        "source_unavailable_horizons_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_UNAVAILABLE_HORIZONS_SHA256
        ),
        "source_unavailable_evidence_sha256s": sorted(
            lockbox.FILL_AWARE_HOLDING_AGE_UNAVAILABLE_EVIDENCE_SHA256S
        ),
        "source_bindings": [
            {
                "backtest_id": backtest_id,
                "job_id": binding["job_id"],
                "strategy_version_id": binding["strategy_version_id"],
            }
        ],
        "source_backtest_ids": [backtest_id],
        "target_strategy_version_ids": [target_version_id],
        "receipt_created_at": datetime(2026, 8, 30, tzinfo=UTC).isoformat(),
        "failed_without_results": [backtest_id],
        "results_created_after_preregistration": [],
        "performance_information_used": False,
    }
    row = {
        "verification_json": verification,
        "target_strategy_version_ids_json": [target_version_id],
        "source_backtest_ids_json": [backtest_id],
        "target_dataset_lineage_id": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
        ),
        "source_dataset_lineage_id": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
        ),
        "receipt_sha256": receipt_sha,
        "source_audit_event_id": 17,
        "source_batch_sha256": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256,
        "target_batch_sha256": target_batch,
        "target_recipe_version": lockbox.FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
    }

    assert lockbox.validate_repair_registry_binding(
        row,
        lockbox_batch_sha256=target_batch,
        strategy_version_id=target_version_id,
        batch_strategy_version_ids={target_version_id},
        batch_dataset_identity_sha256s={
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256
        },
        dataset=lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET,
        dataset_lineage_id=lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID,
    ) == verification

    with pytest.raises(ValueError, match="no exact append-only"):
        lockbox.validate_repair_registry_binding(
            row,
            lockbox_batch_sha256=target_batch,
            strategy_version_id=target_version_id,
            batch_strategy_version_ids={target_version_id, "7" * 32},
            batch_dataset_identity_sha256s={
                lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256
            },
            dataset=lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET,
            dataset_lineage_id=(
                lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
            ),
        )
