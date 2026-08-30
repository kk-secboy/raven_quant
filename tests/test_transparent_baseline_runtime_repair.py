from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform import transparent_baseline_lockbox as lockbox_module
from quant_platform.transparent_baseline_lockbox import (
    PRE_RESULT_REPAIR_CONTRACT_VERSION_V4,
    PRE_RESULT_REPAIR_REGISTRY_VERSION,
    RUNTIME_ALIGNMENT_BENCHMARK_REASON,
    RUNTIME_ALIGNMENT_INDUSTRY_REASON,
    RUNTIME_ALIGNMENT_REPAIR_GENERATION,
    RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS,
    RUNTIME_ALIGNMENT_SOURCE_BINDINGS,
    RUNTIME_ALIGNMENT_SOURCE_COMMIT,
    RUNTIME_ALIGNMENT_SOURCE_DATASET,
    RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION,
    RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    canonical_sha256,
    validate_pre_result_repair_receipt,
    validate_repair_registry_binding,
)
from quant_platform.transparent_baseline_repair import (
    build_runtime_alignment_receipt,
)

pytestmark = pytest.mark.no_database


def _members(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    members: list[dict] = []
    for index, backtest_id in enumerate(sorted(RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS)):
        binding = RUNTIME_ALIGNMENT_SOURCE_BINDINGS[backtest_id]
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
        monkeypatch.setitem(
            binding,
            "artifact_inventory_sha256",
            canonical_sha256(files),
        )
        members.append(
            {
                "backtest_id": backtest_id,
                "strategy_version_id": binding["strategy_version_id"],
                "job_id": binding["job_id"],
                "dataset": RUNTIME_ALIGNMENT_SOURCE_DATASET,
                "periods": deepcopy(binding["periods"]),
                "status": "failed",
                "job_status": "failed",
                "error": binding["error"],
                "metrics_absent": True,
                "result_absent": True,
                "files": files,
            }
        )
    return members


def test_v4_receipt_seals_exact_v9_runtime_failures_and_all_partial_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = build_runtime_alignment_receipt(_members(monkeypatch))

    assert validate_pre_result_repair_receipt(receipt) == receipt
    assert receipt["contract_version"] == PRE_RESULT_REPAIR_CONTRACT_VERSION_V4
    assert receipt["repair_generation"] == RUNTIME_ALIGNMENT_REPAIR_GENERATION
    assert receipt["source_release_commit"] == RUNTIME_ALIGNMENT_SOURCE_COMMIT
    assert receipt["target_recipe_version"] == RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION
    assert receipt[TRANSPARENT_BASELINE_RUNNER_FIELD] == (
        RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256
    )
    assert receipt["reason_codes"] == sorted(
        [RUNTIME_ALIGNMENT_BENCHMARK_REASON, RUNTIME_ALIGNMENT_INDUSTRY_REASON]
    )
    assert all(member["files"] for member in receipt["members"])


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
        "source_runner",
        "target_runner",
        "source_bundle",
        "target_bundle",
        "aggregate_inventory",
    ],
)
def test_v4_receipt_rejects_any_source_or_performance_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    receipt = build_runtime_alignment_receipt(_members(monkeypatch))
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
        member["periods"]["historical_start"] = "2010-01-04"
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
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(payload)

    with pytest.raises(ValueError):
        validate_pre_result_repair_receipt(receipt)


def test_v4_registry_binding_uses_a_distinct_exact_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = _members(monkeypatch)
    receipt = build_runtime_alignment_receipt(members)
    version_ids = ["1" * 32, "2" * 32, "3" * 32]
    verification = {
        "contract_version": PRE_RESULT_REPAIR_REGISTRY_VERSION,
        "receipt_sha256": receipt["receipt_sha256"],
        "source_audit_event_id": 75,
        "source_batch_sha256": "b" * 64,
        "target_batch_sha256": "c" * 64,
        "source_dataset_lineage_id": "d" * 64,
        "target_dataset": RUNTIME_ALIGNMENT_SOURCE_DATASET,
        "target_dataset_identity_sha256": "e" * 64,
        "target_dataset_lineage_id": "d" * 64,
        "target_recipe_version": RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION,
        "receipt_contract_version": PRE_RESULT_REPAIR_CONTRACT_VERSION_V4,
        "repair_generation": RUNTIME_ALIGNMENT_REPAIR_GENERATION,
        "source_release_commit": RUNTIME_ALIGNMENT_SOURCE_COMMIT,
        "source_runner_sha256": receipt["source_runner_sha256"],
        "source_runtime_bundle_sha256": receipt[
            "source_runtime_bundle_sha256"
        ],
        "target_runtime_bundle_sha256": receipt[
            "target_runtime_bundle_sha256"
        ],
        TRANSPARENT_BASELINE_RUNNER_FIELD: RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256,
        "runtime_contract_version": receipt["runtime_contract_version"],
        "source_artifact_inventories_sha256": receipt[
            "source_artifact_inventories_sha256"
        ],
        "source_backtest_ids": sorted(RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS),
        "target_strategy_version_ids": version_ids,
        "receipt_created_at": "2026-08-30T03:00:00+00:00",
        "failed_without_results": sorted(RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS),
        "results_created_after_preregistration": [],
        "performance_information_used": False,
    }
    repair = SimpleNamespace(
        receipt_sha256=receipt["receipt_sha256"],
        source_audit_event_id=75,
        source_batch_sha256="b" * 64,
        target_batch_sha256="c" * 64,
        source_dataset_lineage_id="d" * 64,
        target_dataset_lineage_id="d" * 64,
        target_recipe_version=RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION,
        source_backtest_ids_json=sorted(RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS),
        target_strategy_version_ids_json=version_ids,
        verification_json=verification,
    )

    assert validate_repair_registry_binding(
        repair,
        lockbox_batch_sha256="c" * 64,
        strategy_version_id="1" * 32,
        batch_strategy_version_ids=set(version_ids),
        batch_dataset_identity_sha256s={"e" * 64},
        dataset=RUNTIME_ALIGNMENT_SOURCE_DATASET,
        dataset_lineage_id="d" * 64,
    ) == verification

    monkeypatch.setattr(
        lockbox_module,
        "RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION",
        "changed-v10",
    )
    with pytest.raises(ValueError, match="exact append-only pre-result registry"):
        validate_repair_registry_binding(
            repair,
            lockbox_batch_sha256="c" * 64,
            strategy_version_id="1" * 32,
            batch_strategy_version_ids=set(version_ids),
            batch_dataset_identity_sha256s={"e" * 64},
            dataset=RUNTIME_ALIGNMENT_SOURCE_DATASET,
            dataset_lineage_id="d" * 64,
        )


def test_v4_migration_constraint_pins_only_the_exact_runtime_generation() -> None:
    migration_path = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "0076_transparent_baseline_runtime_alignment_repair.py"
    )
    spec = importlib.util.spec_from_file_location("baseline_runtime_repair_0076", migration_path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    definition = migration._constraint(
        migration._v2_generation(),
        migration._v3_generation(),
        migration._v4_generation(),
    )

    assert PRE_RESULT_REPAIR_CONTRACT_VERSION_V4 in definition
    assert RUNTIME_ALIGNMENT_REPAIR_GENERATION in definition
    assert RUNTIME_ALIGNMENT_SOURCE_COMMIT in definition
    assert RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION in definition
    assert RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256 in definition
    assert "source_runtime_bundle_sha256" in definition
    assert "target_runtime_bundle_sha256" in definition
    assert "source_artifact_inventories_sha256" in definition
    assert str(sorted(RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS)[0]) in definition
