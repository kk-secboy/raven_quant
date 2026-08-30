from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quant_platform import transparent_baseline_bootstrap as bootstrap_module
from quant_platform import transparent_baseline_lockbox as lockbox
from quant_platform import transparent_baseline_repair as repair
from quant_platform.strategy_store import _transparent_worker_runtime_failures
from quant_platform.transparent_baseline_bootstrap import (
    TransparentBaselineBootstrapService,
)
from quant_platform.transparent_baseline_runner import (
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
)

pytestmark = pytest.mark.no_database


def _files() -> list[dict[str, object]]:
    return [
        {"path": "manifest.json", "bytes": 2, "sha256": "a" * 64},
        {"path": "baseline/runtime.json", "bytes": 3, "sha256": "b" * 64},
    ]


def _production_files() -> list[dict[str, object]]:
    return [
        {
            "path": "baseline/composite.parquet",
            "bytes": 57_532_420,
            "sha256": "3e682431fe9844927d30ba039409b12d733a9adc39c6a1192c4a74d6a222028d",
        },
        {
            "path": "baseline/normalized/amount_expansion_5d.parquet",
            "bytes": 57_206_461,
            "sha256": "0c7a7fec840be9d26153c2725f40345022cb531a2b918ff3568133a4799c68f3",
        },
        {
            "path": "baseline/normalized/close_location_5d.parquet",
            "bytes": 53_130_406,
            "sha256": "12ecd49faa1e6a5fb3926f827472c9ea161634f35ac94f6ab1504bb3c42129af",
        },
        {
            "path": "baseline/normalized/extension_penalty_5d.parquet",
            "bytes": 57_217_048,
            "sha256": "f70f2a59471187e5e54506210828c893e0c5d6076d2cad69c583c376573534c1",
        },
        {
            "path": "baseline/normalized/relative_strength_5d.parquet",
            "bytes": 55_912_111,
            "sha256": "50c49cbc1223554ba3b0b1a3cd6667cded6407edb89b257b61cc81a26ad3ac12",
        },
        {
            "path": "baseline/raw/amount_expansion_5d.parquet",
            "bytes": 33_356_095,
            "sha256": "d166122759b5ea42ed582fe217867de2937f62d03142d6d8630b2bd10d24fe2d",
        },
        {
            "path": "baseline/raw/close_location_5d.parquet",
            "bytes": 30_035_531,
            "sha256": "3be7552ff64404260908b1cc196524fe06dd2f7c6dbc3b2478edd956521f48f2",
        },
        {
            "path": "baseline/raw/extension_penalty_5d.parquet",
            "bytes": 32_735_138,
            "sha256": "a7697b9a94e333881c2921071e12e28a8adf09a33a2411afe97fb352593e0d8c",
        },
        {
            "path": "baseline/raw/relative_strength_5d.parquet",
            "bytes": 29_600_893,
            "sha256": "55fddb6562cd756f82f18c268e82a2719fccfd46986996de83826e8f9ef11836",
        },
        {
            "path": "manifest.json",
            "bytes": 68_929,
            "sha256": "66537cadb62a8cf5a9c6fe5a76ce76a901c55a1807aad8f2261f3558ea6ec81e",
        },
    ]


def _calendar() -> list[str]:
    day = date(2019, 11, 27)
    days: list[date] = []
    while len(days) < 2897:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    return [value.isoformat() for value in reversed(days)]


def _patch_inventory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prefix: str,
    source_selection_sha256: str | None = None,
) -> None:
    bindings_name = f"{prefix}_SOURCE_BINDINGS"
    aggregate_name = f"{prefix}_SOURCE_ARTIFACT_INVENTORIES_SHA256"
    selection_name = f"{prefix}_SOURCE_SELECTION_SHA256"
    bindings = getattr(lockbox, bindings_name)
    backtest_id = next(iter(bindings))
    binding = dict(bindings[backtest_id])
    binding["artifact_inventory_sha256"] = lockbox.canonical_sha256(_files())
    monkeypatch.setitem(bindings, backtest_id, binding)
    aggregate = lockbox.canonical_sha256(
        [
            {
                "backtest_id": backtest_id,
                "artifact_inventory_sha256": binding["artifact_inventory_sha256"],
            }
        ]
    )
    monkeypatch.setattr(lockbox, aggregate_name, aggregate)
    monkeypatch.setattr(repair, aggregate_name, aggregate)
    if source_selection_sha256 is not None:
        monkeypatch.setattr(lockbox, selection_name, source_selection_sha256)
        monkeypatch.setattr(repair, selection_name, source_selection_sha256)


def _receipt_member(prefix: str) -> dict[str, object]:
    bindings = getattr(lockbox, f"{prefix}_SOURCE_BINDINGS")
    backtest_id = next(iter(bindings))
    binding = bindings[backtest_id]
    return {
        "backtest_id": backtest_id,
        "strategy_version_id": binding["strategy_version_id"],
        "job_id": binding["job_id"],
        "dataset": getattr(lockbox, f"{prefix}_SOURCE_DATASET"),
        "periods": dict(binding["periods"]),
        "status": "failed",
        "job_status": "failed",
        "error": binding["error"],
        "metrics_absent": True,
        "result_absent": True,
        "files": _files(),
    }


def _source_batch(prefix: str, *, oos_vintage_id: str) -> dict[str, object]:
    bindings = getattr(lockbox, f"{prefix}_SOURCE_BINDINGS")
    binding = next(iter(bindings.values()))
    recipe_version = getattr(lockbox, f"{prefix}_SOURCE_RECIPE_VERSION")
    periods = binding["periods"]
    member = {
        "oos_vintage_id": oos_vintage_id,
        "strategy_version_id": binding["strategy_version_id"],
        "recipe_id": "short_relative_strength",
        "horizon_profile": "short_1_5d",
        "recipe_version": recipe_version,
        "test_start": periods["start"],
        "test_end": periods["end"],
        "first_opened_at": datetime(2026, 8, 30, tzinfo=UTC).isoformat(),
        "sealed_member_set_sha256": "c" * 64,
    }
    return {
        "batch_sha256": getattr(lockbox, f"{prefix}_SOURCE_BATCH_SHA256"),
        "recipe_version": recipe_version,
        "earliest_final_oos_start": periods["start"],
        "latest_final_oos_end": periods["end"],
        "members": [member],
        "members_sha256": lockbox.canonical_sha256([member]),
    }


def _v7_receipt(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    _patch_inventory(monkeypatch, prefix="DISCRETE_MAX_POSITION")
    return repair.build_discrete_max_position_receipt(
        [_receipt_member("DISCRETE_MAX_POSITION")]
    )


def _v15_source_selection(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calendar = _calendar()
    source = lockbox.build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_RECIPE_VERSION,
        prior_batches=[],
    )
    _patch_inventory(
        monkeypatch,
        prefix="FILL_AWARE_HOLDING_AGE",
        source_selection_sha256=source["selection_sha256"],
    )
    receipt = repair.build_fill_aware_holding_age_receipt(
        [_receipt_member("FILL_AWARE_HOLDING_AGE")]
    )
    monkeypatch.setattr(
        lockbox,
        "FILL_AWARE_HOLDING_AGE_RECEIPT_SHA256",
        receipt["receipt_sha256"],
    )
    return lockbox.build_pre_result_repair_history_selection(
        calendar_days=calendar,
        current_recipe_version=lockbox.FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
        source_selection=source,
        repaired_source_batch=_source_batch(
            "FILL_AWARE_HOLDING_AGE", oos_vintage_id="0" * 32
        ),
        repair_receipt=receipt,
    )


def test_v7_receipt_binds_only_the_exact_failed_v15_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _v7_receipt(monkeypatch)
    binding = next(iter(lockbox.DISCRETE_MAX_POSITION_SOURCE_BINDINGS.values()))

    assert lockbox.validate_pre_result_repair_receipt(receipt) == receipt
    assert receipt["contract_version"] == lockbox.PRE_RESULT_REPAIR_CONTRACT_VERSION_V7
    assert receipt["performance_information_used"] is False
    assert len(receipt["members"]) == 1
    assert receipt["members"][0]["strategy_version_id"] == binding["strategy_version_id"]
    assert receipt["members"][0]["error"] == lockbox.DISCRETE_MAX_POSITION_ERROR


def test_v7_production_inventory_and_receipt_are_immutably_sealed() -> None:
    binding = next(iter(lockbox.DISCRETE_MAX_POSITION_SOURCE_BINDINGS.values()))
    files = _production_files()
    member = {
        "backtest_id": next(iter(lockbox.DISCRETE_MAX_POSITION_SOURCE_BACKTEST_IDS)),
        "strategy_version_id": binding["strategy_version_id"],
        "job_id": binding["job_id"],
        "dataset": lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET,
        "periods": dict(binding["periods"]),
        "status": "failed",
        "job_status": "failed",
        "error": lockbox.DISCRETE_MAX_POSITION_ERROR,
        "metrics_absent": True,
        "result_absent": True,
        "files": files,
    }

    assert lockbox.canonical_sha256(files) == binding["artifact_inventory_sha256"]
    receipt = repair.build_discrete_max_position_receipt([member])
    assert receipt["source_artifact_inventories_sha256"] == (
        lockbox.DISCRETE_MAX_POSITION_SOURCE_ARTIFACT_INVENTORIES_SHA256
    )
    assert receipt["receipt_sha256"] == (
        "dbe47a338ee6fd75c5b6dc471775aaa0155697fa7c31012561f362c7cfb5128a"
    )


def test_v16_formal_admission_binds_worker_image_across_all_artifacts() -> None:
    digest = "sha256:" + "d" * 64
    version = {
        "config": {
            "recipe_id": "short_relative_strength",
            "recipe_version": lockbox.DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
            "transparent_baseline_bootstrap": {
                TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: digest,
            },
        }
    }
    manifest = {TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: digest}
    provenance = {TRANSPARENT_BASELINE_RESULT_WORKER_RUNTIME_IMAGE_FIELD: digest}

    assert _transparent_worker_runtime_failures(version, manifest, provenance) == []
    assert _transparent_worker_runtime_failures(version, {}, provenance) == [
        "strategy backtest manifest worker runtime image differs from the sealed version"
    ]
    assert _transparent_worker_runtime_failures(version, manifest, {}) == [
        "formal result worker runtime image differs from the sealed version"
    ]


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    [
        ("member", "backtest_id", "1" * 32),
        ("member", "job_id", "2" * 32),
        ("member", "strategy_version_id", "3" * 32),
        ("member", "error", "ValueError: another failure"),
        ("member", "result_absent", False),
        ("receipt", "source_unopened_history_selection_sha256", "4" * 64),
        ("receipt", "target_runtime_bundle_sha256", "5" * 64),
    ],
)
def test_v7_receipt_rejects_rehashed_tampering(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    field: str,
    value: object,
) -> None:
    receipt = _v7_receipt(monkeypatch)
    if scope == "member":
        receipt["members"][0][field] = value
    else:
        receipt[field] = value
    payload = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = lockbox.canonical_sha256(payload)

    with pytest.raises(ValueError):
        lockbox.validate_pre_result_repair_receipt(receipt)


def test_v16_chained_history_selection_preserves_the_v15_oos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = _calendar()
    source = _v15_source_selection(monkeypatch)
    _patch_inventory(
        monkeypatch,
        prefix="DISCRETE_MAX_POSITION",
        source_selection_sha256=source["selection_sha256"],
    )
    receipt = repair.build_discrete_max_position_receipt(
        [_receipt_member("DISCRETE_MAX_POSITION")]
    )

    evidence = lockbox.build_pre_result_repair_history_selection(
        calendar_days=calendar,
        current_recipe_version=lockbox.DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
        source_selection=source,
        repaired_source_batch=_source_batch(
            "DISCRETE_MAX_POSITION", oos_vintage_id="1" * 32
        ),
        repair_receipt=receipt,
    )

    assert lockbox.validate_unopened_history_selection(
        evidence, calendar_days=calendar
    ) == evidence
    assert evidence["contract_version"] == (
        lockbox.UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V3
    )
    assert evidence["source_history_selection_sha256"] == source["selection_sha256"]
    assert evidence["source_repair_receipt_sha256"] == source["repair_receipt_sha256"]
    assert evidence["repair_receipt_sha256"] == receipt["receipt_sha256"]
    assert evidence["selected_calendar_end"] == source["selected_calendar_end"]
    assert evidence["selected_calendar_trading_days"] == source[
        "selected_calendar_trading_days"
    ]
    assert evidence["performance_information_used"] is False

    tampered = deepcopy(evidence)
    tampered["source_repair_receipt_sha256"] = "d" * 64
    payload = {key: item for key, item in tampered.items() if key != "selection_sha256"}
    tampered["selection_sha256"] = lockbox.canonical_sha256(payload)
    with pytest.raises(ValueError, match="repair history selection changed"):
        lockbox.validate_unopened_history_selection(tampered, calendar_days=calendar)


class _EmptyConnection:
    def __enter__(self) -> _EmptyConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, _statement: object) -> _EmptyConnection:
        return self

    @staticmethod
    def all() -> list[object]:
        return []


class _EmptyEngine:
    @staticmethod
    def connect() -> _EmptyConnection:
        return _EmptyConnection()


class _Rows:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def all(self) -> list[object]:
        return self.rows

    def one(self) -> object:
        assert len(self.rows) == 1
        return self.rows[0]


class _RepairConnection:
    def __init__(self, rows: dict[str, object]) -> None:
        self.rows = rows

    def __enter__(self) -> _RepairConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: object) -> _Rows:
        sql = str(statement)
        if "FROM quantlab.audit_events" in sql:
            return _Rows([self.rows["audit"]])
        if "FROM quantlab.oos_vintages" in sql:
            return _Rows([self.rows["oos"]])
        if "FROM quantlab.transparent_baseline_pre_result_repairs" in sql:
            return _Rows([self.rows["predecessor"]])
        if "FROM quantlab.strategy_versions" in sql:
            return _Rows([self.rows["version"]])
        if "FROM quantlab.backtest_runs" in sql:
            return _Rows([self.rows["backtest"]])
        raise AssertionError(f"unexpected resolver query: {sql}")


class _RepairEngine:
    def __init__(self, rows: dict[str, object]) -> None:
        self.rows = rows

    def connect(self) -> _RepairConnection:
        return _RepairConnection(self.rows)


def test_v16_repair_selection_fails_closed_without_preregistered_receipt() -> None:
    store = object.__new__(lockbox.TransparentBaselineLockboxStore)
    store.engine = _EmptyEngine()

    with pytest.raises(ValueError, match="v16 discrete max-position repair receipt"):
        store.resolve_preregistered_single_member_repair(
            calendar_days=_calendar(),
            current_recipe_version=lockbox.DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
        )


def test_v16_repair_selection_uses_the_exact_receipt_and_chained_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = _calendar()
    source_selection = _v15_source_selection(monkeypatch)
    evidence_sha256s = sorted(
        lockbox.DISCRETE_MAX_POSITION_UNAVAILABLE_EVIDENCE_SHA256S
    )
    unavailable_horizons = [
        {
            "recipe_id": "swing_trend",
            "evidence_sha256": evidence_sha256s[0],
        },
        {
            "recipe_id": "long_quality_value",
            "evidence_sha256": evidence_sha256s[1],
        },
    ]
    unavailable_sha256 = lockbox.canonical_sha256(unavailable_horizons)
    monkeypatch.setattr(
        lockbox,
        "DISCRETE_MAX_POSITION_SOURCE_UNAVAILABLE_HORIZONS_SHA256",
        unavailable_sha256,
    )
    monkeypatch.setattr(
        repair,
        "DISCRETE_MAX_POSITION_SOURCE_UNAVAILABLE_HORIZONS_SHA256",
        unavailable_sha256,
    )
    _patch_inventory(
        monkeypatch,
        prefix="DISCRETE_MAX_POSITION",
        source_selection_sha256=source_selection["selection_sha256"],
    )
    receipt = repair.build_discrete_max_position_receipt(
        [_receipt_member("DISCRETE_MAX_POSITION")]
    )
    binding = next(iter(lockbox.DISCRETE_MAX_POSITION_SOURCE_BINDINGS.values()))
    source_version_id = str(binding["strategy_version_id"])
    opened_at = datetime(2026, 8, 30, 8, tzinfo=UTC)
    registered_at = opened_at + timedelta(hours=1)
    source_lockbox = {
        "contract_version": lockbox.LOCKBOX_CONTRACT_VERSION_V3,
        "batch_sha256": lockbox.DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256,
        "members": [
            {
                "recipe_id": "short_relative_strength",
                "horizon_profile": "short_1_5d",
            }
        ],
        "unavailable_horizons": unavailable_horizons,
    }
    # The resolver separately compares the canonical unavailable-horizon digest.
    # Keep this unit fixture small while preserving that exact binding.
    monkeypatch.setattr(
        lockbox,
        "validate_joint_lockbox",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        lockbox,
        "validate_lockbox_link",
        lambda value: dict(value),
    )
    predecessor_checked: list[str] = []

    def validate_predecessor(_row: object, *, source_version_id: str) -> dict[str, Any]:
        predecessor_checked.append(source_version_id)
        return {}

    monkeypatch.setattr(
        lockbox,
        "validate_discrete_max_position_predecessor_registry",
        validate_predecessor,
    )
    monkeypatch.setattr(lockbox, "_artifact_inventory", lambda _path: _files())
    monkeypatch.setattr(lockbox, "_artifact_result_exists", lambda _path: False)
    worker_image = "sha256:" + "e" * 64
    source_config = {
        "recipe_id": "short_relative_strength",
        "recipe_version": lockbox.DISCRETE_MAX_POSITION_SOURCE_RECIPE_VERSION,
        lockbox.BOOTSTRAP_CONFIG_KEY: {
            lockbox.TRANSPARENT_BASELINE_RUNNER_FIELD: (
                lockbox.DISCRETE_MAX_POSITION_SOURCE_RUNNER_SHA256
            ),
            lockbox.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
                lockbox.DISCRETE_MAX_POSITION_SOURCE_BUNDLE_SHA256
            ),
            lockbox.TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: worker_image,
            "dataset": lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET,
            "dataset_identity_sha256": (
                lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256
            ),
            "dataset_lineage_id": (
                lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID
            ),
            "formal_periods": dict(binding["periods"]),
            "unopened_history_selection": source_selection,
        },
        lockbox.LOCKBOX_CONFIG_KEY: source_lockbox,
    }
    source_oos = SimpleNamespace(
        id="1" * 32,
        consumed_at=registered_at,
        first_opened_at=opened_at,
        dataset_identity=lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256,
        dataset_lineage_id=lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID,
        test_start=date.fromisoformat(binding["periods"]["start"]),
        test_end=date.fromisoformat(binding["periods"]["end"]),
        sealed_candidate_set_sha256="c" * 64,
        sealed_candidate_set_json={
            "strategy_version_id": source_version_id,
            "transparent_baseline_lockbox": {
                "batch_sha256": lockbox.DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256,
                "recipe_id": "short_relative_strength",
                "horizon_profile": "short_1_5d",
            },
        },
    )
    backtest_id = next(iter(lockbox.DISCRETE_MAX_POSITION_SOURCE_BACKTEST_IDS))
    rows = {
        "audit": SimpleNamespace(
            details_json=receipt,
            method="INTERNAL",
            path="transparent-baseline/pre-result-repair",
            status_code=201,
            created_at=registered_at,
        ),
        "oos": source_oos,
        "predecessor": SimpleNamespace(),
        "version": SimpleNamespace(config_json=source_config),
        "backtest": SimpleNamespace(
            id=backtest_id,
            strategy_version_id=source_version_id,
            job_id=binding["job_id"],
            dataset=lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET,
            periods_json=dict(binding["periods"]),
            status="failed",
            job_kind="strategy_backtest",
            job_status="failed",
            error=binding["error"],
            job_error=binding["error"],
            job_payload_json={
                lockbox.TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: worker_image
            },
            metrics_json=None,
            artifact_path="unused-by-patched-inventory",
            created_at=opened_at,
        ),
    }
    store = object.__new__(lockbox.TransparentBaselineLockboxStore)
    store.engine = _RepairEngine(rows)

    selected = store.resolve_preregistered_single_member_repair(
        calendar_days=calendar,
        current_recipe_version=lockbox.DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
    )

    assert selected is not None
    assert selected["repair_receipt"] == receipt
    assert selected["source_batch_sha256"] == (
        lockbox.DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256
    )
    assert selected["evidence"]["contract_version"] == (
        lockbox.UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V3
    )
    assert selected["evidence"]["source_history_selection_sha256"] == (
        source_selection["selection_sha256"]
    )
    assert len(selected["calendar"]) == source_selection[
        "selected_calendar_trading_days"
    ]
    assert predecessor_checked == [source_version_id]


def test_v16_predecessor_registry_must_be_the_exact_v6_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _v15_source_selection(monkeypatch)
    source_binding = next(
        iter(lockbox.DISCRETE_MAX_POSITION_SOURCE_BINDINGS.values())
    )
    source_version_id = str(source_binding["strategy_version_id"])
    v6_bindings = [
        {
            "backtest_id": backtest_id,
            "job_id": binding["job_id"],
            "strategy_version_id": binding["strategy_version_id"],
        }
        for backtest_id, binding in sorted(
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BINDINGS.items()
        )
    ]
    verification = {
        "receipt_sha256": lockbox.FILL_AWARE_HOLDING_AGE_RECEIPT_SHA256,
        "receipt_contract_version": lockbox.PRE_RESULT_REPAIR_CONTRACT_VERSION_V6,
        "repair_generation": lockbox.FILL_AWARE_HOLDING_AGE_REPAIR_GENERATION,
        "source_release_commit": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_COMMIT,
        "source_batch_sha256": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256,
        "source_dataset_identity_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "source_dataset_lineage_id": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
        ),
        "source_runner_sha256": lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_RUNNER_SHA256,
        lockbox.TRANSPARENT_BASELINE_RUNNER_FIELD: (
            lockbox.FILL_AWARE_HOLDING_AGE_TARGET_RUNNER_SHA256
        ),
        "source_runtime_bundle_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BUNDLE_SHA256
        ),
        "target_runtime_bundle_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_TARGET_BUNDLE_SHA256
        ),
        "runtime_contract_version": (
            lockbox.FILL_AWARE_HOLDING_AGE_RUNTIME_CONTRACT_VERSION
        ),
        "target_recipe_version": (
            lockbox.FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION
        ),
        "source_artifact_inventories_sha256": (
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_ARTIFACT_INVENTORIES_SHA256
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
        "target_change_codes": list(
            lockbox.FILL_AWARE_HOLDING_AGE_TARGET_CHANGE_CODES
        ),
        "source_bindings": v6_bindings,
    }
    row = SimpleNamespace(
        receipt_sha256=lockbox.FILL_AWARE_HOLDING_AGE_RECEIPT_SHA256,
        source_batch_sha256=lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BATCH_SHA256,
        target_batch_sha256=lockbox.DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256,
        source_dataset_lineage_id=(
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_DATASET_LINEAGE_ID
        ),
        target_dataset_lineage_id=lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID,
        target_recipe_version=lockbox.FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
        source_backtest_ids_json=sorted(
            lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_BACKTEST_IDS
        ),
        target_strategy_version_ids_json=[source_version_id],
        verification_json=verification,
    )

    assert lockbox.validate_discrete_max_position_predecessor_registry(
        row, source_version_id=source_version_id
    ) == verification
    row.target_strategy_version_ids_json = ["f" * 32]
    with pytest.raises(ValueError, match="predecessor repair registry changed"):
        lockbox.validate_discrete_max_position_predecessor_registry(
            row, source_version_id=source_version_id
        )


def test_v16_bootstrap_without_exact_receipt_writes_no_version_backtest_or_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = {"versions": 0, "backtests": 0, "jobs": 0, "ordinary": 0}
    calendar = _calendar()
    dataset = {
        "name": lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET,
        "start_date": calendar[0],
        "end_date": calendar[-1],
        "trading_days": len(calendar),
        "dataset_identity_sha256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "dataset_lineage_id": lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID,
        "calendar": calendar,
    }

    class Strategies:
        @staticmethod
        def get_by_name(_name: str) -> None:
            return None

        @staticmethod
        def create(**_values: object) -> None:
            writes["versions"] += 1

        @staticmethod
        def create_version_if_absent(*_args: object, **_values: object) -> None:
            writes["versions"] += 1

        @staticmethod
        def create_backtest(**_values: object) -> None:
            writes["backtests"] += 1

    class Jobs:
        @staticmethod
        def create(**_values: object) -> None:
            writes["jobs"] += 1

    class Lockboxes:
        @staticmethod
        def resolve_preregistered_single_member_repair(**_values: object) -> None:
            raise ValueError("v16 discrete max-position repair receipt is not registered")

        @staticmethod
        def resolve_unopened_history_selection(**_values: object) -> None:
            writes["ordinary"] += 1

    monkeypatch.setattr(
        bootstrap_module,
        "_select_dataset",
        lambda *_args, **_values: dataset,
    )
    service = TransparentBaselineBootstrapService.__new__(
        TransparentBaselineBootstrapService
    )
    service.data_root = tmp_path
    service.dataset_loader = lambda _root: [dataset]
    service.strategies = Strategies()
    service.jobs = Jobs()
    service.lockboxes = Lockboxes()

    result = service.reconcile(actor="test-v16-missing-receipt")

    assert result["status"] == "failed"
    assert result["members"] == []
    assert result["errors"] == [
        "v16 discrete max-position repair receipt is not registered"
    ]
    assert writes == {"versions": 0, "backtests": 0, "jobs": 0, "ordinary": 0}
