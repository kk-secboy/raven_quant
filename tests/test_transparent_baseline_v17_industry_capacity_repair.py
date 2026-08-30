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
from quant_platform.strategy_store import _bind_current_transparent_runtime_identity
from quant_platform.transparent_baseline_bootstrap import (
    TransparentBaselineBootstrapService,
)
from quant_platform.transparent_baseline_runner import (
    TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
    TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    WORKER_RUNTIME_IMAGE_DIGEST_ENV,
)

pytestmark = pytest.mark.no_database

_V8_RECEIPT_SHA256 = (
    "980ea643755d261cc7ee39ffec5e23af3f8e27ebdc1e2f2a647e0802b9dc636d"
)


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
            "bytes": 69_231,
            "sha256": "0e64e37876a1ee81a853bbd13fcac4165abf7c0e960929c99e3d32d33683eef1",
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


def _receipt_member(
    prefix: str, *, files: list[dict[str, object]] | None = None
) -> dict[str, object]:
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
        "files": list(files if files is not None else _files()),
    }


def _source_batch(prefix: str, *, oos_vintage_id: str) -> dict[str, object]:
    binding = next(iter(getattr(lockbox, f"{prefix}_SOURCE_BINDINGS").values()))
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


def _v16_source_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calendar = _calendar()
    ordinary = lockbox.build_unopened_history_selection(
        calendar_days=calendar,
        current_recipe_version=lockbox.FILL_AWARE_HOLDING_AGE_SOURCE_RECIPE_VERSION,
        prior_batches=[],
    )
    _patch_inventory(
        monkeypatch,
        prefix="FILL_AWARE_HOLDING_AGE",
        source_selection_sha256=ordinary["selection_sha256"],
    )
    v6_receipt = repair.build_fill_aware_holding_age_receipt(
        [_receipt_member("FILL_AWARE_HOLDING_AGE")]
    )
    monkeypatch.setattr(
        lockbox,
        "FILL_AWARE_HOLDING_AGE_RECEIPT_SHA256",
        v6_receipt["receipt_sha256"],
    )
    v15_selection = lockbox.build_pre_result_repair_history_selection(
        calendar_days=calendar,
        current_recipe_version=lockbox.FILL_AWARE_HOLDING_AGE_TARGET_RECIPE_VERSION,
        source_selection=ordinary,
        repaired_source_batch=_source_batch(
            "FILL_AWARE_HOLDING_AGE", oos_vintage_id="0" * 32
        ),
        repair_receipt=v6_receipt,
    )
    _patch_inventory(
        monkeypatch,
        prefix="DISCRETE_MAX_POSITION",
        source_selection_sha256=v15_selection["selection_sha256"],
    )
    v7_receipt = repair.build_discrete_max_position_receipt(
        [_receipt_member("DISCRETE_MAX_POSITION")]
    )
    monkeypatch.setattr(
        lockbox,
        "DISCRETE_MAX_POSITION_RECEIPT_SHA256",
        v7_receipt["receipt_sha256"],
    )
    v16_selection = lockbox.build_pre_result_repair_history_selection(
        calendar_days=calendar,
        current_recipe_version=lockbox.DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
        source_selection=v15_selection,
        repaired_source_batch=_source_batch(
            "DISCRETE_MAX_POSITION", oos_vintage_id="1" * 32
        ),
        repair_receipt=v7_receipt,
    )
    return v16_selection, v7_receipt


def test_v8_production_inventory_and_receipt_are_immutably_sealed() -> None:
    files = _production_files()
    binding = next(iter(lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS.values()))

    assert lockbox.canonical_sha256(files) == binding["artifact_inventory_sha256"]
    receipt = repair.build_topk_industry_capacity_receipt(
        [_receipt_member("TOPK_INDUSTRY_CAPACITY", files=files)]
    )

    assert receipt["contract_version"] == lockbox.PRE_RESULT_REPAIR_CONTRACT_VERSION_V8
    assert receipt["source_artifact_inventories_sha256"] == (
        lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_ARTIFACT_INVENTORIES_SHA256
    )
    assert receipt["target_change_codes"] == [
        "topk_industry_capacity_partial_cash"
    ]
    assert receipt["performance_information_used"] is False
    assert receipt["receipt_sha256"] == _V8_RECEIPT_SHA256


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
        ("receipt", "target_change_codes", ["loosen_industry_risk"]),
    ],
)
def test_v8_receipt_rejects_rehashed_tampering(
    scope: str,
    field: str,
    value: object,
) -> None:
    receipt = repair.build_topk_industry_capacity_receipt(
        [_receipt_member("TOPK_INDUSTRY_CAPACITY", files=_production_files())]
    )
    if scope == "member":
        receipt["members"][0][field] = value
    else:
        receipt[field] = value
    payload = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = lockbox.canonical_sha256(payload)

    with pytest.raises(ValueError):
        lockbox.validate_pre_result_repair_receipt(receipt)


def test_v17_history_selection_preserves_v16_oos_and_direct_v7_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = _calendar()
    source, v7_receipt = _v16_source_selection(monkeypatch)
    _patch_inventory(
        monkeypatch,
        prefix="TOPK_INDUSTRY_CAPACITY",
        source_selection_sha256=source["selection_sha256"],
    )
    receipt = repair.build_topk_industry_capacity_receipt(
        [_receipt_member("TOPK_INDUSTRY_CAPACITY")]
    )

    evidence = lockbox.build_pre_result_repair_history_selection(
        calendar_days=calendar,
        current_recipe_version=lockbox.TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION,
        source_selection=source,
        repaired_source_batch=_source_batch(
            "TOPK_INDUSTRY_CAPACITY", oos_vintage_id="2" * 32
        ),
        repair_receipt=receipt,
    )

    assert lockbox.validate_unopened_history_selection(
        evidence, calendar_days=calendar
    ) == evidence
    assert evidence["contract_version"] == (
        lockbox.UNOPENED_HISTORY_SELECTION_CONTRACT_VERSION_V4
    )
    assert evidence["source_history_selection_sha256"] == source["selection_sha256"]
    assert evidence["source_repair_receipt_sha256"] == v7_receipt["receipt_sha256"]
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


def test_v17_repair_selection_fails_closed_without_preregistered_receipt() -> None:
    store = object.__new__(lockbox.TransparentBaselineLockboxStore)
    store.engine = _EmptyEngine()

    with pytest.raises(ValueError, match="v17 topk industry-capacity repair receipt"):
        store.resolve_preregistered_single_member_repair(
            calendar_days=_calendar(),
            current_recipe_version=lockbox.TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION,
        )


def _v7_predecessor_registry_and_audit() -> tuple[SimpleNamespace, SimpleNamespace]:
    source_binding = next(
        iter(lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS.values())
    )
    source_version_id = str(source_binding["strategy_version_id"])
    v7_files = _production_files()
    v7_files[-1] = {
        "path": "manifest.json",
        "bytes": 68_929,
        "sha256": "66537cadb62a8cf5a9c6fe5a76ce76a901c55a1807aad8f2261f3558ea6ec81e",
    }
    receipt = repair.build_discrete_max_position_receipt(
        [_receipt_member("DISCRETE_MAX_POSITION", files=v7_files)]
    )
    assert receipt["receipt_sha256"] == lockbox.DISCRETE_MAX_POSITION_RECEIPT_SHA256
    audit_id = 77
    created_at = datetime(2026, 8, 30, 16, tzinfo=UTC)
    source_bindings = [
        {
            "backtest_id": backtest_id,
            "job_id": binding["job_id"],
            "strategy_version_id": binding["strategy_version_id"],
        }
        for backtest_id, binding in sorted(
            lockbox.DISCRETE_MAX_POSITION_SOURCE_BINDINGS.items()
        )
    ]
    verification = {
        "contract_version": lockbox.PRE_RESULT_REPAIR_REGISTRY_VERSION,
        "receipt_sha256": lockbox.DISCRETE_MAX_POSITION_RECEIPT_SHA256,
        "source_audit_event_id": audit_id,
        "receipt_contract_version": lockbox.PRE_RESULT_REPAIR_CONTRACT_VERSION_V7,
        "repair_generation": lockbox.DISCRETE_MAX_POSITION_REPAIR_GENERATION,
        "source_release_commit": lockbox.DISCRETE_MAX_POSITION_SOURCE_COMMIT,
        "source_batch_sha256": lockbox.DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256,
        "target_batch_sha256": lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256,
        "source_dataset_identity_sha256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "source_dataset_lineage_id": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID
        ),
        "source_runner_sha256": lockbox.DISCRETE_MAX_POSITION_SOURCE_RUNNER_SHA256,
        lockbox.TRANSPARENT_BASELINE_RUNNER_FIELD: (
            lockbox.DISCRETE_MAX_POSITION_TARGET_RUNNER_SHA256
        ),
        "source_runtime_bundle_sha256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_BUNDLE_SHA256
        ),
        "target_runtime_bundle_sha256": (
            lockbox.DISCRETE_MAX_POSITION_TARGET_BUNDLE_SHA256
        ),
        "runtime_contract_version": (
            lockbox.DISCRETE_MAX_POSITION_RUNTIME_CONTRACT_VERSION
        ),
        "target_recipe_version": (
            lockbox.DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION
        ),
        "target_dataset": lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET,
        "target_dataset_identity_sha256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "target_dataset_lineage_id": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID
        ),
        "source_artifact_inventories_sha256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_ARTIFACT_INVENTORIES_SHA256
        ),
        "source_unopened_history_selection_sha256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_SELECTION_SHA256
        ),
        "source_unavailable_horizons_sha256": (
            lockbox.DISCRETE_MAX_POSITION_SOURCE_UNAVAILABLE_HORIZONS_SHA256
        ),
        "source_unavailable_evidence_sha256s": sorted(
            lockbox.DISCRETE_MAX_POSITION_UNAVAILABLE_EVIDENCE_SHA256S
        ),
        "target_change_codes": list(
            lockbox.DISCRETE_MAX_POSITION_TARGET_CHANGE_CODES
        ),
        "source_bindings": source_bindings,
        "source_backtest_ids": sorted(
            lockbox.DISCRETE_MAX_POSITION_SOURCE_BACKTEST_IDS
        ),
        "target_strategy_version_ids": [source_version_id],
        "receipt_created_at": created_at.isoformat(),
        "failed_without_results": sorted(
            lockbox.DISCRETE_MAX_POSITION_SOURCE_BACKTEST_IDS
        ),
        "results_created_after_preregistration": [],
        "performance_information_used": False,
    }
    row = SimpleNamespace(
        source_audit_event_id=audit_id,
        receipt_sha256=lockbox.DISCRETE_MAX_POSITION_RECEIPT_SHA256,
        source_batch_sha256=lockbox.DISCRETE_MAX_POSITION_SOURCE_BATCH_SHA256,
        target_batch_sha256=lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256,
        source_dataset_lineage_id=(
            lockbox.DISCRETE_MAX_POSITION_SOURCE_DATASET_LINEAGE_ID
        ),
        target_dataset_lineage_id=(
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID
        ),
        target_recipe_version=lockbox.DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
        source_backtest_ids_json=sorted(
            lockbox.DISCRETE_MAX_POSITION_SOURCE_BACKTEST_IDS
        ),
        target_strategy_version_ids_json=[source_version_id],
        verification_json=verification,
    )
    audit = SimpleNamespace(
        id=audit_id,
        action=lockbox.PRE_RESULT_REPAIR_ACTION,
        method="INTERNAL",
        path="transparent-baseline/pre-result-repair",
        status_code=201,
        created_at=created_at,
        details_json=receipt,
    )
    return row, audit


def test_v17_predecessor_registry_and_audit_are_the_exact_canonical_v7_source() -> None:
    row, audit = _v7_predecessor_registry_and_audit()
    source_version_id = next(
        iter(lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS.values())
    )["strategy_version_id"]

    assert lockbox.validate_topk_industry_capacity_predecessor_registry(
        row,
        source_version_id=source_version_id,
        audit_event=audit,
    ) == row.verification_json


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("verification", "target_batch_sha256", None),
        ("verification", "target_dataset", "changed-dataset"),
        ("row", "source_audit_event_id", 78),
        ("audit", "method", "POST"),
        ("audit", "created_at", datetime(2026, 8, 30, 16)),
        ("audit", "details_json", {"receipt_sha256": "d" * 64}),
    ],
)
def test_v17_predecessor_registry_rejects_missing_fields_and_bad_audit_envelope(
    target: str,
    field: str,
    value: object,
) -> None:
    row, audit = _v7_predecessor_registry_and_audit()
    source_version_id = next(
        iter(lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS.values())
    )["strategy_version_id"]
    if target == "verification":
        if value is None:
            row.verification_json.pop(field)
        else:
            row.verification_json[field] = value
    else:
        setattr(row if target == "row" else audit, field, value)

    with pytest.raises(ValueError, match="predecessor repair registry changed"):
        lockbox.validate_topk_industry_capacity_predecessor_registry(
            row,
            source_version_id=source_version_id,
            audit_event=audit,
        )


class _Rows:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def one_or_none(self) -> object | None:
        assert len(self.rows) <= 1
        return self.rows[0] if self.rows else None

    def all(self) -> list[object]:
        return self.rows

    def scalar_one(self) -> int:
        assert self.rows == [101]
        return 101


class _RegisterConnection:
    def __init__(
        self,
        source: object,
        *,
        predecessor: object,
        predecessor_audit: object,
        existing_audits: list[object],
    ) -> None:
        self.source = source
        self.predecessor = predecessor
        self.predecessor_audit = predecessor_audit
        self.existing_audits = existing_audits
        self.statements: list[str] = []

    def __enter__(self) -> _RegisterConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: object) -> _Rows:
        sql = str(statement)
        self.statements.append(sql)
        if sql.startswith("INSERT INTO quantlab.audit_events"):
            return _Rows([101])
        if "FROM quantlab.backtest_runs JOIN quantlab.jobs" in sql:
            return _Rows([self.source])
        if "FROM quantlab.transparent_baseline_pre_result_repairs" in sql:
            return _Rows([self.predecessor])
        if "FROM quantlab.audit_events" in sql:
            if "quantlab.audit_events.id =" in sql:
                return _Rows([self.predecessor_audit])
            return _Rows(self.existing_audits)
        raise AssertionError(f"unexpected registration query: {sql}")


class _RegisterEngine:
    def __init__(
        self,
        source: object,
        *,
        predecessor: object,
        predecessor_audit: object,
        existing_audits: list[object],
    ) -> None:
        self.connection = _RegisterConnection(
            source,
            predecessor=predecessor,
            predecessor_audit=predecessor_audit,
            existing_audits=existing_audits,
        )

    def begin(self) -> _RegisterConnection:
        return self.connection


@pytest.mark.parametrize(
    ("existing_mode", "expected_status", "expected_audit_id", "expected_inserts"),
    [
        ("none", "registered", 101, 1),
        ("valid", "already_registered", 202, 0),
        ("collision", None, None, 0),
    ],
)
def test_v8_registration_is_append_only_idempotent_and_rejects_hash_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_mode: str,
    expected_status: str | None,
    expected_audit_id: int | None,
    expected_inserts: int,
) -> None:
    source = next(iter(lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BINDINGS.values()))
    backtest_id = next(iter(lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BACKTEST_IDS))
    worker_image = "sha256:" + "e" * 64
    unavailable = [
        {
            "recipe_id": "long_quality_value",
            "evidence_sha256": sorted(
                lockbox.TOPK_INDUSTRY_CAPACITY_UNAVAILABLE_EVIDENCE_SHA256S
            )[1],
        },
        {
            "recipe_id": "swing_trend",
            "evidence_sha256": sorted(
                lockbox.TOPK_INDUSTRY_CAPACITY_UNAVAILABLE_EVIDENCE_SHA256S
            )[0],
        },
    ]
    validated_lockbox = {
        "contract_version": lockbox.LOCKBOX_CONTRACT_VERSION_V3,
        "batch_sha256": lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BATCH_SHA256,
        "members": [{"recipe_id": "short_relative_strength"}],
        "unavailable_horizons": unavailable,
    }
    bootstrap = {
        lockbox.TRANSPARENT_BASELINE_RUNNER_FIELD: (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_RUNNER_SHA256
        ),
        lockbox.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BUNDLE_SHA256
        ),
        TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: worker_image,
        "dataset": lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET,
        "dataset_identity_sha256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "dataset_lineage_id": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID
        ),
        "formal_periods": dict(source["periods"]),
        "unopened_history_selection": {
            "selection_sha256": lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_SELECTION_SHA256
        },
    }
    row = SimpleNamespace(
        id=backtest_id,
        strategy_version_id=source["strategy_version_id"],
        job_id=source["job_id"],
        dataset=lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET,
        periods_json=dict(source["periods"]),
        status="failed",
        metrics_json=None,
        artifact_path=str(tmp_path),
        error=lockbox.TOPK_INDUSTRY_CAPACITY_ERROR,
        job_kind="strategy_backtest",
        job_status="failed",
        job_error=lockbox.TOPK_INDUSTRY_CAPACITY_ERROR,
        job_payload_json={
            "backtest_id": backtest_id,
            "strategy_version_id": source["strategy_version_id"],
            "dataset": lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET,
            "periods": dict(source["periods"]),
            TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: (
                lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_RUNNER_SHA256
            ),
            TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: (
                lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_BUNDLE_SHA256
            ),
            TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: worker_image,
        },
        config_json={
            "recipe_id": "short_relative_strength",
            "recipe_version": lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_RECIPE_VERSION,
            lockbox.BOOTSTRAP_CONFIG_KEY: bootstrap,
            lockbox.LOCKBOX_CONFIG_KEY: {"validated-by-test": True},
        },
    )
    original_row = deepcopy(vars(row))
    predecessor, predecessor_audit = _v7_predecessor_registry_and_audit()
    expected_receipt = repair.build_topk_industry_capacity_receipt(
        [_receipt_member("TOPK_INDUSTRY_CAPACITY", files=_production_files())]
    )
    existing_audits: list[object] = []
    if existing_mode != "none":
        details = (
            expected_receipt
            if existing_mode == "valid"
            else {
                "receipt_sha256": expected_receipt["receipt_sha256"],
                "members": [],
            }
        )
        existing_audits = [
            SimpleNamespace(
                id=202,
                action=lockbox.PRE_RESULT_REPAIR_ACTION,
                method="INTERNAL",
                path="transparent-baseline/pre-result-repair",
                status_code=201,
                created_at=datetime(2026, 8, 31, tzinfo=UTC),
                details_json=details,
            )
        ]
    engine = _RegisterEngine(
        row,
        predecessor=predecessor,
        predecessor_audit=predecessor_audit,
        existing_audits=existing_audits,
    )
    original_canonical_sha256 = repair.canonical_sha256

    def canonical_sha256(value: Any) -> str:
        if value == unavailable:
            return lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_UNAVAILABLE_HORIZONS_SHA256
        return original_canonical_sha256(value)

    predecessor_calls: list[tuple[str, object]] = []

    def validate_predecessor(
        _row: object,
        *,
        source_version_id: str,
        audit_event: object,
    ) -> dict[str, Any]:
        predecessor_calls.append((source_version_id, audit_event))
        return {}

    monkeypatch.setattr(repair, "open_database", lambda _url: engine)
    monkeypatch.setattr(
        repair, "validate_joint_lockbox", lambda _value: validated_lockbox
    )
    monkeypatch.setattr(
        repair,
        "validate_unopened_history_selection",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        repair,
        "validate_topk_industry_capacity_predecessor_registry",
        validate_predecessor,
    )
    monkeypatch.setattr(repair, "_artifact_inventory", lambda _path: _production_files())
    monkeypatch.setattr(repair, "canonical_sha256", canonical_sha256)

    if existing_mode == "collision":
        with pytest.raises(ValueError, match="hash collides with an invalid audit"):
            repair.register_topk_industry_capacity_repair(
                "postgresql://unused",
                backtest_ids=[backtest_id],
                actor="test-v17-registration",
                target_runtime_root=Path(__file__).parents[1],
            )
    else:
        result = repair.register_topk_industry_capacity_repair(
            "postgresql://unused",
            backtest_ids=[backtest_id],
            actor="test-v17-registration",
            target_runtime_root=Path(__file__).parents[1],
        )
        assert result["status"] == expected_status
        assert result["audit_event_id"] == expected_audit_id
        assert result["receipt"]["receipt_sha256"] == _V8_RECEIPT_SHA256
    assert predecessor_calls == [
        (source["strategy_version_id"], predecessor_audit)
    ]
    assert vars(row) == original_row
    inserted_audits = sum(
        sql.startswith("INSERT INTO quantlab.audit_events")
        for sql in engine.connection.statements
    )
    assert inserted_audits == expected_inserts
    assert not any(
        sql.lstrip().upper().startswith(("UPDATE ", "DELETE "))
        for sql in engine.connection.statements
    )


def test_current_v17_runtime_is_bound_without_rebinding_historical_v16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_image = "sha256:" + "d" * 64
    monkeypatch.setenv(WORKER_RUNTIME_IMAGE_DIGEST_ENV, worker_image)
    current = {
        "recipe_id": "short_relative_strength",
        "recipe_version": lockbox.TOPK_INDUSTRY_CAPACITY_TARGET_RECIPE_VERSION,
    }

    bound = _bind_current_transparent_runtime_identity(current)

    assert bound["transparent_baseline_bootstrap"] == {
        lockbox.TRANSPARENT_BASELINE_RUNNER_FIELD: (
            lockbox.TOPK_INDUSTRY_CAPACITY_TARGET_RUNNER_SHA256
        ),
        lockbox.TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
            lockbox.TOPK_INDUSTRY_CAPACITY_TARGET_BUNDLE_SHA256
        ),
        TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: worker_image,
    }
    historical = {
        "recipe_id": "short_relative_strength",
        "recipe_version": lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_RECIPE_VERSION,
        "transparent_baseline_bootstrap": {"historical": True},
    }
    assert _bind_current_transparent_runtime_identity(historical) == historical


def test_v17_bootstrap_without_exact_receipt_writes_no_version_backtest_or_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = {"versions": 0, "backtests": 0, "jobs": 0, "ordinary": 0}
    calendar = _calendar()
    dataset = {
        "name": lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET,
        "start_date": calendar[0],
        "end_date": calendar[-1],
        "trading_days": len(calendar),
        "dataset_identity_sha256": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_IDENTITY_SHA256
        ),
        "dataset_lineage_id": (
            lockbox.TOPK_INDUSTRY_CAPACITY_SOURCE_DATASET_LINEAGE_ID
        ),
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
            raise ValueError(
                "v17 topk industry-capacity repair receipt is not registered"
            )

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

    result = service.reconcile(actor="test-v17-missing-receipt")

    assert result["status"] == "failed"
    assert result["members"] == []
    assert result["errors"] == [
        "v17 topk industry-capacity repair receipt is not registered"
    ]
    assert writes == {"versions": 0, "backtests": 0, "jobs": 0, "ordinary": 0}
