from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform.transparent_baseline_lockbox import (
    CANONICAL_LF_PACKAGING_CONTRACT_VERSION,
    CANONICAL_LF_PACKAGING_REPAIR_GENERATION,
    CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS,
    CANONICAL_LF_PACKAGING_SOURCE_BINDINGS,
    CANONICAL_LF_PACKAGING_SOURCE_COMMIT,
    CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256,
    CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
    PRE_RESULT_REPAIR_CONTRACT_VERSION_V3,
    PRE_RESULT_REPAIR_REGISTRY_VERSION,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    _repair_bootstrap_semantics,
    canonical_sha256,
    validate_pre_result_repair_receipt,
    validate_repair_registry_binding,
)
from quant_platform.transparent_baseline_repair import (
    build_canonical_lf_packaging_receipt,
)
from quant_platform.transparent_baseline_runner import (
    CANONICAL_LF_TARGET_RUNNER_SHA256,
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
    require_transparent_baseline_runner,
    target_runner_for_recipe,
)

pytestmark = pytest.mark.no_database


def _members() -> list[dict]:
    result = []
    for backtest_id in sorted(CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS):
        binding = CANONICAL_LF_PACKAGING_SOURCE_BINDINGS[backtest_id]
        result.append(
            {
                "backtest_id": backtest_id,
                "strategy_version_id": binding["strategy_version_id"],
                "job_id": binding["job_id"],
                "dataset": "same-daily",
                "periods": {
                    "historical_start": "2008-01-02",
                    "historical_end": "2021-12-31",
                    "start": "2022-01-04",
                    "end": "2025-12-31",
                },
                "status": "failed",
                "job_status": "failed",
                "error": "transparent v8 runner bytes differ from the repair authorization",
                "metrics_absent": True,
                "result_absent": True,
                "files": [],
            }
        )
    return result


def test_v3_receipt_seals_exact_empty_v8_packaging_failures() -> None:
    receipt = build_canonical_lf_packaging_receipt(_members())

    assert validate_pre_result_repair_receipt(receipt) == receipt
    assert receipt["contract_version"] == PRE_RESULT_REPAIR_CONTRACT_VERSION_V3
    assert receipt["repair_generation"] == CANONICAL_LF_PACKAGING_REPAIR_GENERATION
    assert receipt["source_runner_expected_sha256"] == (
        OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
    )
    assert receipt["source_runner_observed_sha256"] == (
        CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256
    )
    assert receipt[TRANSPARENT_BASELINE_RUNNER_FIELD] == (
        CANONICAL_LF_TARGET_RUNNER_SHA256
    )
    assert all(member["files"] == [] for member in receipt["members"])


def test_v3_receipt_rejects_fabricated_post_failure_manifest() -> None:
    receipt = build_canonical_lf_packaging_receipt(_members())
    receipt["members"][0]["files"] = [
        {"path": "manifest.json", "bytes": 2, "sha256": "a" * 64}
    ]
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(payload)

    with pytest.raises(ValueError, match="packaging repair artifacts must be empty"):
        validate_pre_result_repair_receipt(receipt)


def test_v3_receipt_rejects_any_source_job_or_version_rebinding() -> None:
    members = _members()
    members[0]["job_id"] = "f" * 32

    with pytest.raises(ValueError, match="source binding changed"):
        build_canonical_lf_packaging_receipt(members)


@pytest.mark.parametrize(
    "mutation",
    [
        "backtest",
        "version",
        "source_commit",
        "expected_runner",
        "observed_runner",
        "target_runner",
        "error",
        "metrics",
        "result",
        "status",
    ],
)
def test_v3_receipt_rejects_every_non_packaging_source_change(mutation: str) -> None:
    receipt = build_canonical_lf_packaging_receipt(_members())
    member = receipt["members"][0]
    if mutation == "backtest":
        member["backtest_id"] = "f" * 32
    elif mutation == "version":
        member["strategy_version_id"] = "f" * 32
    elif mutation == "source_commit":
        receipt["source_release_commit"] = "f" * 40
    elif mutation == "expected_runner":
        receipt["source_runner_expected_sha256"] = "f" * 64
    elif mutation == "observed_runner":
        receipt["source_runner_observed_sha256"] = "f" * 64
    elif mutation == "target_runner":
        receipt[TRANSPARENT_BASELINE_RUNNER_FIELD] = "f" * 64
    elif mutation == "error":
        member["error"] += " changed"
    elif mutation == "metrics":
        member["metrics_absent"] = False
    elif mutation == "result":
        member["result_absent"] = False
    elif mutation == "status":
        member["status"] = "running"
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(payload)

    with pytest.raises(ValueError):
        validate_pre_result_repair_receipt(receipt)


def test_v3_bootstrap_comparison_ignores_only_version_derived_packaging_hashes() -> None:
    source = {
        "transparent_baseline_bootstrap": {
            "recipe_version": "v8",
            "recipe_sha256": "a" * 64,
            TRANSPARENT_BASELINE_RUNNER_FIELD: "b" * 64,
            "dataset": "same-daily",
            "dataset_identity_sha256": "c" * 64,
            "dataset_lineage_id": "d" * 64,
            "formal_periods": {
                "historical_start": "2008-01-02",
                "historical_end": "2021-12-31",
                "start": "2022-01-04",
                "end": "2025-12-31",
            },
            "research_periods": {
                "train_start": "2008-01-02",
                "train_end": "2018-12-31",
                "valid_start": "2019-01-02",
                "valid_end": "2021-12-31",
                "test_start": "2022-01-04",
                "test_end": "2025-12-31",
            },
            "feature_set": {
                "id": "transparent-baseline:short_relative_strength",
                "recipe_version": "v8",
                "definition_sha256": "e" * 64,
                "features": {"relative_strength_5d": "$close/Ref($close,5)-1"},
            },
            "research_window_contract": {
                "contract_version": "research-window-v1",
                "feature_set_sha256": "e" * 64,
                "dataset_contract_sha256": "f" * 64,
                "periods": {"test_start": "2022-01-04", "test_end": "2025-12-31"},
            },
            "research_window_contract_sha256": "1" * 64,
        }
    }
    target = deepcopy(source)
    target_bootstrap = target["transparent_baseline_bootstrap"]
    target_bootstrap["recipe_version"] = "v9"
    target_bootstrap["recipe_sha256"] = "2" * 64
    target_bootstrap[TRANSPARENT_BASELINE_RUNNER_FIELD] = "3" * 64
    target_bootstrap["feature_set"]["recipe_version"] = "v9"
    target_bootstrap["feature_set"]["definition_sha256"] = "4" * 64
    target_bootstrap["research_window_contract"]["feature_set_sha256"] = "4" * 64
    target_bootstrap["research_window_contract_sha256"] = "5" * 64
    assert _repair_bootstrap_semantics(source) == _repair_bootstrap_semantics(target)

    target["transparent_baseline_bootstrap"]["research_periods"]["train_start"] = (
        "2009-01-05"
    )
    assert _repair_bootstrap_semantics(source) != _repair_bootstrap_semantics(target)


def test_v9_uses_canonical_lf_runner_without_rebinding_historical_v8() -> None:
    assert target_runner_for_recipe(
        "short_relative_strength", OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION
    ) == OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
    assert target_runner_for_recipe(
        "short_relative_strength", CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION
    ) == CANONICAL_LF_TARGET_RUNNER_SHA256

    runner = Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"
    config = {
        "recipe_id": "short_relative_strength",
        "recipe_version": CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
        "transparent_baseline_bootstrap": {
            TRANSPARENT_BASELINE_RUNNER_FIELD: CANONICAL_LF_TARGET_RUNNER_SHA256,
        },
    }
    assert require_transparent_baseline_runner(
        config=config,
        job_payload={
            "transparent_baseline_runner_sha256": CANONICAL_LF_TARGET_RUNNER_SHA256
        },
        runner_path=runner,
    ) == CANONICAL_LF_TARGET_RUNNER_SHA256


def test_v3_registry_binding_is_exact_and_generation_aware() -> None:
    version_ids = ["1" * 32, "2" * 32, "3" * 32]
    verification = {
        "contract_version": PRE_RESULT_REPAIR_REGISTRY_VERSION,
        "receipt_sha256": "a" * 64,
        "source_audit_event_id": 73,
        "source_batch_sha256": "b" * 64,
        "target_batch_sha256": "c" * 64,
        "source_dataset_lineage_id": "d" * 64,
        "target_dataset": "same-daily",
        "target_dataset_identity_sha256": "e" * 64,
        "target_dataset_lineage_id": "d" * 64,
        "target_recipe_version": CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
        "receipt_contract_version": PRE_RESULT_REPAIR_CONTRACT_VERSION_V3,
        "repair_generation": CANONICAL_LF_PACKAGING_REPAIR_GENERATION,
        "source_release_commit": CANONICAL_LF_PACKAGING_SOURCE_COMMIT,
        "source_runner_expected_sha256": (
            OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
        ),
        "source_runner_observed_sha256": (
            CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256
        ),
        TRANSPARENT_BASELINE_RUNNER_FIELD: CANONICAL_LF_TARGET_RUNNER_SHA256,
        "packaging_contract_version": CANONICAL_LF_PACKAGING_CONTRACT_VERSION,
        "source_backtest_ids": sorted(CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS),
        "target_strategy_version_ids": version_ids,
        "receipt_created_at": "2026-08-30T01:10:00+00:00",
        "failed_without_results": sorted(
            CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS
        ),
        "results_created_after_preregistration": [],
        "performance_information_used": False,
    }
    repair = SimpleNamespace(
        receipt_sha256="a" * 64,
        source_audit_event_id=73,
        source_batch_sha256="b" * 64,
        target_batch_sha256="c" * 64,
        source_dataset_lineage_id="d" * 64,
        target_dataset_lineage_id="d" * 64,
        target_recipe_version=CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
        source_backtest_ids_json=sorted(CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS),
        target_strategy_version_ids_json=version_ids,
        verification_json=verification,
    )

    assert validate_repair_registry_binding(
        repair,
        lockbox_batch_sha256="c" * 64,
        strategy_version_id="1" * 32,
        batch_strategy_version_ids=set(version_ids),
        batch_dataset_identity_sha256s={"e" * 64},
        dataset="same-daily",
        dataset_lineage_id="d" * 64,
    ) == verification

    verification["source_runner_observed_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="exact append-only pre-result registry"):
        validate_repair_registry_binding(
            repair,
            lockbox_batch_sha256="c" * 64,
            strategy_version_id="1" * 32,
            batch_strategy_version_ids=set(version_ids),
            batch_dataset_identity_sha256s={"e" * 64},
            dataset="same-daily",
            dataset_lineage_id="d" * 64,
        )
    verification["source_runner_observed_sha256"] = (
        CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256
    )
    verification["source_release_commit"] = "f" * 40
    with pytest.raises(ValueError, match="exact append-only pre-result registry"):
        validate_repair_registry_binding(
            repair,
            lockbox_batch_sha256="c" * 64,
            strategy_version_id="1" * 32,
            batch_strategy_version_ids=set(version_ids),
            batch_dataset_identity_sha256s={"e" * 64},
            dataset="same-daily",
            dataset_lineage_id="d" * 64,
        )
    verification["source_release_commit"] = CANONICAL_LF_PACKAGING_SOURCE_COMMIT
    verification["failed_without_results"] = verification[
        "failed_without_results"
    ][1:]
    with pytest.raises(ValueError, match="exact append-only pre-result registry"):
        validate_repair_registry_binding(
            repair,
            lockbox_batch_sha256="c" * 64,
            strategy_version_id="1" * 32,
            batch_strategy_version_ids=set(version_ids),
            batch_dataset_identity_sha256s={"e" * 64},
            dataset="same-daily",
            dataset_lineage_id="d" * 64,
        )
    verification["failed_without_results"] = sorted(
        CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS
    )
    verification["results_created_after_preregistration"] = ["unexpected"]
    with pytest.raises(ValueError, match="exact append-only pre-result registry"):
        validate_repair_registry_binding(
            repair,
            lockbox_batch_sha256="c" * 64,
            strategy_version_id="1" * 32,
            batch_strategy_version_ids=set(version_ids),
            batch_dataset_identity_sha256s={"e" * 64},
            dataset="same-daily",
            dataset_lineage_id="d" * 64,
        )
