from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform import transparent_baseline_lockbox as lockbox_module
from quant_platform import transparent_baseline_repair as repair_module
from quant_platform.transparent_baseline_lockbox import (
    PRE_RESULT_REPAIR_CONTRACT_VERSION_V5,
    PRE_RESULT_REPAIR_REGISTRY_VERSION,
    RUNTIME_INPUT_SCOPE_CONTRACT_VERSION,
    RUNTIME_INPUT_SCOPE_REPAIR_GENERATION,
    RUNTIME_INPUT_SCOPE_SOURCE_ARTIFACT_INVENTORIES_SHA256,
    RUNTIME_INPUT_SCOPE_SOURCE_BACKTEST_IDS,
    RUNTIME_INPUT_SCOPE_SOURCE_BATCH_SHA256,
    RUNTIME_INPUT_SCOPE_SOURCE_BINDINGS,
    RUNTIME_INPUT_SCOPE_SOURCE_COMMIT,
    RUNTIME_INPUT_SCOPE_SOURCE_DATASET,
    RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256,
    RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID,
    RUNTIME_INPUT_SCOPE_TARGET_CHANGE_CODES,
    RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION,
    RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256,
    RUNTIME_INPUT_SCOPE_TOPK_BENCHMARK_REASON,
    RUNTIME_INPUT_SCOPE_TREND_REASON,
    RUNTIME_INPUT_SCOPE_VALUATION_REASON,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    canonical_sha256,
    validate_pre_result_repair_receipt,
    validate_repair_registry_binding,
)
from quant_platform.transparent_baseline_repair import (
    build_runtime_input_scope_receipt,
)
from quant_platform.transparent_baseline_runner import (
    RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256,
    runtime_alignment_bundle_sha256,
)

pytestmark = pytest.mark.no_database


def _members(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    members: list[dict] = []
    inventory_digests: list[dict[str, str]] = []
    for index, backtest_id in enumerate(
        sorted(RUNTIME_INPUT_SCOPE_SOURCE_BACKTEST_IDS)
    ):
        binding = RUNTIME_INPUT_SCOPE_SOURCE_BINDINGS[backtest_id]
        files = [
            {
                "path": "baseline/composite.parquet",
                "bytes": 100 + index,
                "sha256": str(index + 1) * 64,
            },
            {
                "path": "manifest.json",
                "bytes": 20 + index,
                "sha256": str(index + 4) * 64,
            },
        ]
        inventory_sha256 = canonical_sha256(files)
        monkeypatch.setitem(
            binding,
            "artifact_inventory_sha256",
            inventory_sha256,
        )
        inventory_digests.append(
            {
                "backtest_id": backtest_id,
                "artifact_inventory_sha256": inventory_sha256,
            }
        )
        members.append(
            {
                "backtest_id": backtest_id,
                "strategy_version_id": binding["strategy_version_id"],
                "job_id": binding["job_id"],
                "dataset": RUNTIME_INPUT_SCOPE_SOURCE_DATASET,
                "periods": deepcopy(binding["periods"]),
                "status": "failed",
                "job_status": "failed",
                "error": binding["error"],
                "metrics_absent": True,
                "result_absent": True,
                "files": files,
            }
        )
    aggregate = canonical_sha256(
        sorted(inventory_digests, key=lambda item: item["backtest_id"])
    )
    monkeypatch.setattr(
        lockbox_module,
        "RUNTIME_INPUT_SCOPE_SOURCE_ARTIFACT_INVENTORIES_SHA256",
        aggregate,
    )
    monkeypatch.setattr(
        repair_module,
        "RUNTIME_INPUT_SCOPE_SOURCE_ARTIFACT_INVENTORIES_SHA256",
        aggregate,
    )
    return members


def test_v5_receipt_seals_one_exact_three_member_v10_failure_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = build_runtime_input_scope_receipt(_members(monkeypatch))

    assert validate_pre_result_repair_receipt(receipt) == receipt
    assert receipt["contract_version"] == PRE_RESULT_REPAIR_CONTRACT_VERSION_V5
    assert receipt["repair_generation"] == RUNTIME_INPUT_SCOPE_REPAIR_GENERATION
    assert receipt["source_batch_sha256"] == RUNTIME_INPUT_SCOPE_SOURCE_BATCH_SHA256
    assert receipt["source_dataset_identity_sha256"] == (
        RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256
    )
    assert receipt["source_dataset_lineage_id"] == (
        RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID
    )
    assert receipt["target_recipe_version"] == RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION
    assert receipt[TRANSPARENT_BASELINE_RUNNER_FIELD] == (
        RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256
    )
    assert receipt["runtime_contract_version"] == RUNTIME_INPUT_SCOPE_CONTRACT_VERSION
    assert receipt["reason_codes"] == sorted(
        [
            RUNTIME_INPUT_SCOPE_TOPK_BENCHMARK_REASON,
            RUNTIME_INPUT_SCOPE_TREND_REASON,
            RUNTIME_INPUT_SCOPE_VALUATION_REASON,
        ]
    )
    assert receipt["target_change_codes"] == list(
        RUNTIME_INPUT_SCOPE_TARGET_CHANGE_CODES
    )
    assert tuple(sorted(RUNTIME_INPUT_SCOPE_TARGET_CHANGE_CODES)) == (
        RUNTIME_INPUT_SCOPE_TARGET_CHANGE_CODES
    )
    assert len(receipt["reason_codes"]) == 3


@pytest.mark.parametrize(
    "mutation",
    [
        "backtest",
        "job",
        "version",
        "dataset",
        "period",
        "error",
        "metrics",
        "result",
        "artifact_path",
        "artifact_size",
        "artifact_sha256",
        "source_commit",
        "source_batch",
        "source_identity",
        "source_lineage",
        "source_runner",
        "target_runner",
        "source_bundle",
        "target_bundle",
        "aggregate_inventory",
        "target_change_missing",
        "target_change_unknown",
        "target_change_order",
    ],
)
def test_v5_receipt_rejects_source_performance_or_runtime_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    receipt = build_runtime_input_scope_receipt(_members(monkeypatch))
    member = receipt["members"][0]
    if mutation == "backtest":
        member["backtest_id"] = "f" * 32
    elif mutation == "job":
        member["job_id"] = "f" * 32
    elif mutation == "version":
        member["strategy_version_id"] = "f" * 32
    elif mutation == "dataset":
        member["dataset"] = "different-dataset"
    elif mutation == "period":
        member["periods"]["historical_start"] = "2008-01-04"
    elif mutation == "error":
        member["error"] += " changed"
    elif mutation == "metrics":
        member["metrics_absent"] = False
    elif mutation == "result":
        member["result_absent"] = False
    elif mutation == "artifact_path":
        member["files"][0]["path"] = "result.json"
    elif mutation == "artifact_size":
        member["files"][0]["bytes"] += 1
    elif mutation == "artifact_sha256":
        member["files"][0]["sha256"] = "f" * 64
    elif mutation == "source_commit":
        receipt["source_release_commit"] = "f" * 40
    elif mutation == "source_batch":
        receipt["source_batch_sha256"] = "f" * 64
    elif mutation == "source_identity":
        receipt["source_dataset_identity_sha256"] = "f" * 64
    elif mutation == "source_lineage":
        receipt["source_dataset_lineage_id"] = "f" * 64
    elif mutation == "source_runner":
        receipt["source_runner_sha256"] = "f" * 64
    elif mutation == "target_runner":
        receipt[TRANSPARENT_BASELINE_RUNNER_FIELD] = "f" * 64
    elif mutation == "source_bundle":
        receipt["source_runtime_bundle_sha256"] = "f" * 64
    elif mutation == "target_bundle":
        receipt["target_runtime_bundle_sha256"] = "f" * 64
    elif mutation == "aggregate_inventory":
        receipt["source_artifact_inventories_sha256"] = "f" * 64
    elif mutation == "target_change_missing":
        receipt["target_change_codes"] = receipt["target_change_codes"][:-1]
    elif mutation == "target_change_unknown":
        receipt["target_change_codes"][0] = "unknown-target-change"
    elif mutation == "target_change_order":
        receipt["target_change_codes"].reverse()
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(payload)

    with pytest.raises(ValueError):
        validate_pre_result_repair_receipt(receipt)


def test_v5_registry_binding_is_exact_and_same_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = build_runtime_input_scope_receipt(_members(monkeypatch))
    version_ids = ["1" * 32, "2" * 32, "3" * 32]
    target_batch = "c" * 64
    verification = {
        "contract_version": PRE_RESULT_REPAIR_REGISTRY_VERSION,
        "receipt_sha256": receipt["receipt_sha256"],
        "source_audit_event_id": 77,
        "source_batch_sha256": RUNTIME_INPUT_SCOPE_SOURCE_BATCH_SHA256,
        "target_batch_sha256": target_batch,
        "source_dataset_lineage_id": RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID,
        "source_dataset_identity_sha256": (
            RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "target_dataset": RUNTIME_INPUT_SCOPE_SOURCE_DATASET,
        "target_dataset_identity_sha256": (
            RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "target_dataset_lineage_id": RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID,
        "target_recipe_version": RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION,
        "receipt_contract_version": PRE_RESULT_REPAIR_CONTRACT_VERSION_V5,
        "repair_generation": RUNTIME_INPUT_SCOPE_REPAIR_GENERATION,
        "source_release_commit": RUNTIME_INPUT_SCOPE_SOURCE_COMMIT,
        "source_runner_sha256": receipt["source_runner_sha256"],
        "source_runtime_bundle_sha256": receipt["source_runtime_bundle_sha256"],
        "target_runtime_bundle_sha256": receipt["target_runtime_bundle_sha256"],
        TRANSPARENT_BASELINE_RUNNER_FIELD: RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256,
        "runtime_contract_version": receipt["runtime_contract_version"],
        "source_artifact_inventories_sha256": receipt[
            "source_artifact_inventories_sha256"
        ],
        "target_change_codes": list(RUNTIME_INPUT_SCOPE_TARGET_CHANGE_CODES),
        "source_backtest_ids": sorted(RUNTIME_INPUT_SCOPE_SOURCE_BACKTEST_IDS),
        "target_strategy_version_ids": version_ids,
        "receipt_created_at": "2026-08-30T06:00:00+00:00",
        "failed_without_results": sorted(RUNTIME_INPUT_SCOPE_SOURCE_BACKTEST_IDS),
        "results_created_after_preregistration": [],
        "performance_information_used": False,
    }
    repair = SimpleNamespace(
        receipt_sha256=receipt["receipt_sha256"],
        source_audit_event_id=77,
        source_batch_sha256=RUNTIME_INPUT_SCOPE_SOURCE_BATCH_SHA256,
        target_batch_sha256=target_batch,
        source_dataset_lineage_id=RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID,
        target_dataset_lineage_id=RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID,
        target_recipe_version=RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION,
        source_backtest_ids_json=sorted(RUNTIME_INPUT_SCOPE_SOURCE_BACKTEST_IDS),
        target_strategy_version_ids_json=version_ids,
        verification_json=verification,
    )

    assert validate_repair_registry_binding(
        repair,
        lockbox_batch_sha256=target_batch,
        strategy_version_id="1" * 32,
        batch_strategy_version_ids=set(version_ids),
        batch_dataset_identity_sha256s={
            RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256
        },
        dataset=RUNTIME_INPUT_SCOPE_SOURCE_DATASET,
        dataset_lineage_id=RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID,
    ) == verification

    verification["source_batch_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="exact append-only pre-result registry"):
        validate_repair_registry_binding(
            repair,
            lockbox_batch_sha256=target_batch,
            strategy_version_id="1" * 32,
            batch_strategy_version_ids=set(version_ids),
            batch_dataset_identity_sha256s={
                RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256
            },
            dataset=RUNTIME_INPUT_SCOPE_SOURCE_DATASET,
            dataset_lineage_id=RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID,
        )

    verification["source_batch_sha256"] = RUNTIME_INPUT_SCOPE_SOURCE_BATCH_SHA256
    verification["target_change_codes"] = verification["target_change_codes"][:-1]
    with pytest.raises(ValueError, match="exact append-only pre-result registry"):
        validate_repair_registry_binding(
            repair,
            lockbox_batch_sha256=target_batch,
            strategy_version_id="1" * 32,
            batch_strategy_version_ids=set(version_ids),
            batch_dataset_identity_sha256s={
                RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256
            },
            dataset=RUNTIME_INPUT_SCOPE_SOURCE_DATASET,
            dataset_lineage_id=RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID,
        )


def test_v5_real_inventory_and_runtime_hashes_are_frozen() -> None:
    inventories = sorted(
        (
            {
                "backtest_id": backtest_id,
                "artifact_inventory_sha256": binding[
                    "artifact_inventory_sha256"
                ],
            }
            for backtest_id, binding in RUNTIME_INPUT_SCOPE_SOURCE_BINDINGS.items()
        ),
        key=lambda item: item["backtest_id"],
    )
    root = Path(__file__).parents[1]

    assert canonical_sha256(inventories) == (
        RUNTIME_INPUT_SCOPE_SOURCE_ARTIFACT_INVENTORIES_SHA256
    )
    assert runtime_alignment_bundle_sha256(root) == (
        RUNTIME_INPUT_SCOPE_TARGET_RUNTIME_BUNDLE_SHA256
    )
    assert (
        __import__("hashlib")
        .sha256((root / "scripts" / "run_multifactor_backtest.py").read_bytes())
        .hexdigest()
        == RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256
    )


def test_v5_migration_constraint_pins_only_the_exact_generation() -> None:
    migration_path = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0077_transparent_baseline_runtime_input_scope_repair.py"
    )
    spec = importlib.util.spec_from_file_location(
        "baseline_input_scope_repair_0077", migration_path
    )
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    definition = migration._constraint(
        migration._v2_generation(),
        migration._v3_generation(),
        migration._v4_generation(),
        migration._v5_generation(),
    )

    assert PRE_RESULT_REPAIR_CONTRACT_VERSION_V5 in definition
    assert RUNTIME_INPUT_SCOPE_REPAIR_GENERATION in definition
    assert RUNTIME_INPUT_SCOPE_SOURCE_COMMIT in definition
    assert RUNTIME_INPUT_SCOPE_SOURCE_BATCH_SHA256 in definition
    assert RUNTIME_INPUT_SCOPE_SOURCE_DATASET_IDENTITY_SHA256 in definition
    assert RUNTIME_INPUT_SCOPE_SOURCE_DATASET_LINEAGE_ID in definition
    assert RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION in definition
    assert RUNTIME_INPUT_SCOPE_TARGET_RUNNER_SHA256 in definition
    assert "source_runtime_bundle_sha256" in definition
    assert "target_runtime_bundle_sha256" in definition
    assert "source_artifact_inventories_sha256" in definition
    assert "target_change_codes" in definition
    for change_code in RUNTIME_INPUT_SCOPE_TARGET_CHANGE_CODES:
        assert change_code in definition
    assert str(sorted(RUNTIME_INPUT_SCOPE_SOURCE_BACKTEST_IDS)[0]) in definition
