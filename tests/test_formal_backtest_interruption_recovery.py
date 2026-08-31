from __future__ import annotations

import copy
import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import update
from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg
from sqlalchemy.sql.elements import Null

from quant_data.database import jobs
from quant_platform.formal_backtest_interruption_recovery import (
    EXTERNAL_EVIDENCE_CONTRACT_VERSION,
    INTERRUPTED_JOB_ERROR,
    RECOVERY_CONTROLLER_FAILURE_ERROR,
    V17_EXTERNAL_INTERRUPTION_OBSERVED_AT,
    V17_EXTERNAL_JOURNAL_EXCERPT,
    V17_EXTERNAL_JOURNAL_SHA256,
    V17_INTERRUPTED_ARTIFACT_INVENTORY,
    V17_INTERRUPTION_RECOVERY_PROFILE,
    V17_INTERRUPTION_RECOVERY_RECEIPT_SHA256,
    V17_RECOVERY_DATABASE_APPLICATION_NAME,
    V17_REPAIR_RECEIPT_SHA256,
    FormalBacktestInterruptionRecoveryStore,
    _job_requeue_values,
    _source_backtest_snapshot,
    _source_job_snapshot,
    _validate_snapshot_hash,
    _version_binding,
    build_interruption_recovery_receipt,
    validate_external_interruption_evidence,
    validate_interruption_recovery_receipt,
)
from quant_platform.transparent_baseline_lockbox import canonical_sha256
from quant_platform.transparent_baseline_runner import (
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
)

pytestmark = pytest.mark.no_database


def test_authorized_requeue_uses_sql_null_for_json_progress() -> None:
    values = _job_requeue_values()

    assert isinstance(values["progress_json"], Null)
    assert values["status"] == "queued"
    assert values["max_attempts"] == 2
    compiled = update(jobs).values(**values).compile(dialect=PGDialect_psycopg())
    assert "progress_json=NULL" in str(compiled)
    assert "progress_json" not in compiled.params


def _external_evidence() -> dict:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    return {
        "contract_version": EXTERNAL_EVIDENCE_CONTRACT_VERSION,
        "source": "systemd-docker-journal",
        "service": "evaluation-worker",
        "stop_owner": "quantlab-backup.service",
        "signal": "SIGTERM",
        "has_been_manually_stopped": True,
        "oom_killed": False,
        "backtest_id": profile.backtest_id,
        "job_id": profile.job_id,
        "observed_at": V17_EXTERNAL_INTERRUPTION_OBSERVED_AT,
        "journal_sha256": V17_EXTERNAL_JOURNAL_SHA256,
        "journal_excerpt": copy.deepcopy(V17_EXTERNAL_JOURNAL_EXCERPT),
    }


def _version() -> SimpleNamespace:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    return SimpleNamespace(
        id=profile.strategy_version_id,
        status="draft",
        strategy_type="multifactor",
        horizon_profile="short_1_5d",
        strategy_rules_sha256=profile.strategy_rules_sha256,
        execution_contract_hash=profile.execution_contract_hash,
        qlib_version=profile.qlib_version,
        qlib_commit=profile.qlib_commit,
        rdagent_version=profile.rdagent_version,
        rdagent_commit=profile.rdagent_commit,
        config_json={
            "recipe_id": profile.recipe_id,
            "recipe_version": profile.recipe_version,
            "transparent_baseline_bootstrap": {
                "recipe_id": profile.recipe_id,
                "recipe_version": profile.recipe_version,
                "recipe_sha256": profile.recipe_sha256,
                TRANSPARENT_BASELINE_RUNNER_FIELD: profile.runner_sha256,
                TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
                    profile.runtime_bundle_sha256
                ),
                TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: (
                    profile.worker_runtime_image_digest
                ),
                "dataset": profile.dataset,
                "dataset_identity_sha256": profile.dataset_identity_sha256,
                "dataset_lineage_id": profile.dataset_lineage_id,
                "formal_periods": dict(profile.periods),
            },
        },
    )


def test_version_binding_reads_recipe_hash_from_the_bootstrap_contract() -> None:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    version = _version()

    assert "recipe_sha256" not in version.config_json
    binding = _version_binding(version, profile)

    assert binding["recipe_sha256"] == profile.recipe_sha256


def test_version_binding_rejects_a_top_level_recipe_hash_decoy() -> None:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    version = _version()
    version.config_json["recipe_sha256"] = profile.recipe_sha256
    version.config_json["transparent_baseline_bootstrap"]["recipe_sha256"] = "0" * 64

    with pytest.raises(
        ValueError,
        match="v17 formal backtest immutable strategy binding changed",
    ):
        _version_binding(version, profile)


@pytest.mark.parametrize("field", ["recipe_id", "recipe_version"])
def test_version_binding_rejects_bootstrap_recipe_identity_mismatch(
    field: str,
) -> None:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    version = _version()
    version.config_json["transparent_baseline_bootstrap"][field] = "changed"

    with pytest.raises(
        ValueError,
        match="v17 formal backtest immutable strategy binding changed",
    ):
        _version_binding(version, profile)


def _receipt() -> dict:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    version = _version()
    source_job = _source_job_snapshot(
        SimpleNamespace(
            id=profile.job_id,
            kind="strategy_backtest",
            idempotency_key=profile.idempotency_key,
            status="failed",
            attempts=1,
            max_attempts=1,
            exit_code=143,
            error=INTERRUPTED_JOB_ERROR,
            progress_json=None,
            next_attempt_at=None,
            cancel_requested_at=None,
            log_path=profile.source_log_path,
            payload_json=profile.expected_payload,
            created_at=datetime.fromisoformat(profile.source_job_created_at),
            started_at=datetime.fromisoformat(profile.source_job_started_at),
            finished_at=datetime.fromisoformat(profile.source_job_finished_at),
        ),
        profile,
    )
    source_backtest = _source_backtest_snapshot(
        SimpleNamespace(
            id=profile.backtest_id,
            job_id=profile.job_id,
            strategy_version_id=profile.strategy_version_id,
            dataset=profile.dataset,
            execution_dataset=None,
            status="running",
            periods_json=dict(profile.periods),
            artifact_path=profile.source_artifact_path,
            metrics_json=None,
            error=None,
            finished_at=None,
            execution_contract_hash=profile.execution_contract_hash,
            qlib_version=profile.qlib_version,
            qlib_commit=profile.qlib_commit,
            rdagent_version=profile.rdagent_version,
            rdagent_commit=profile.rdagent_commit,
            created_at=datetime.fromisoformat(profile.source_backtest_created_at),
            started_at=datetime.fromisoformat(profile.source_backtest_started_at),
        ),
        version,
        profile,
    )
    return build_interruption_recovery_receipt(
        source_job=source_job,
        source_backtest=source_backtest,
        immutable_binding=_version_binding(version, profile),
        log_prefix={
            "path": profile.source_log_path,
            "bytes": profile.source_log_prefix_bytes,
            "sha256": profile.source_log_prefix_sha256,
        },
        files=profile.source_artifact_inventory,
        external_interruption=_external_evidence(),
        profile=profile,
    )


def _resign(value: dict) -> dict:
    resigned = copy.deepcopy(value)
    resigned.pop("receipt_sha256", None)
    resigned["receipt_sha256"] = canonical_sha256(resigned)
    return resigned


def test_v17_recovery_profile_pins_every_source_digest_and_receipt() -> None:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE

    assert canonical_sha256(profile.expected_payload) == profile.source_payload_sha256
    assert canonical_sha256([dict(item) for item in V17_INTERRUPTED_ARTIFACT_INVENTORY]) == (
        profile.source_artifact_inventory_sha256
    )
    assert canonical_sha256(V17_EXTERNAL_JOURNAL_EXCERPT) == V17_EXTERNAL_JOURNAL_SHA256
    receipt = _receipt()
    assert receipt["receipt_sha256"] == V17_INTERRUPTION_RECOVERY_RECEIPT_SHA256
    assert receipt["performance_information_used"] is False
    assert receipt["immutable_binding"]["source_repair_receipt_sha256"] == (
        V17_REPAIR_RECEIPT_SHA256
    )
    assert validate_interruption_recovery_receipt(receipt) == receipt


@pytest.mark.parametrize("snapshot_key", ["source_job", "source_backtest"])
def test_nested_source_row_hash_cannot_be_replaced_and_resigned(
    snapshot_key: str,
) -> None:
    receipt = _receipt()
    snapshot = dict(receipt[snapshot_key])
    expected_keys = set(snapshot) - {"row_sha256"}
    snapshot["row_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="hash changed"):
        _validate_snapshot_hash(
            snapshot,
            expected_keys=expected_keys,
            label=snapshot_key,
        )
    receipt[snapshot_key] = snapshot
    with pytest.raises(ValueError, match="receipt hash changed|hash changed"):
        validate_interruption_recovery_receipt(_resign(receipt))


def test_receipt_cannot_hide_extra_performance_information() -> None:
    receipt = _receipt()
    receipt["pre_result_evidence"]["observed_sharpe"] = 3.0

    with pytest.raises(ValueError, match="receipt hash changed|sealed v17 recovery"):
        validate_interruption_recovery_receipt(_resign(receipt))


def test_receipt_cannot_claim_that_performance_information_was_used() -> None:
    receipt = _receipt()
    receipt["performance_information_used"] = True

    with pytest.raises(ValueError, match="receipt hash changed|sealed v17 recovery"):
        validate_interruption_recovery_receipt(_resign(receipt))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("service", "worker"),
        ("signal", "SIGKILL"),
        ("oom_killed", True),
        ("observed_at", "2026-08-30T19:30:10+00:00"),
        ("journal_sha256", "a" * 64),
    ],
)
def test_external_evidence_is_exact_not_a_generic_signal_claim(
    key: str, value: object
) -> None:
    evidence = _external_evidence()
    evidence[key] = value

    with pytest.raises(ValueError, match="not the v17 SIGTERM"):
        validate_external_interruption_evidence(
            evidence, profile=V17_INTERRUPTION_RECOVERY_PROFILE
        )


def test_external_journal_records_are_embedded_and_rehashed() -> None:
    evidence = _external_evidence()
    evidence["journal_excerpt"]["records"][1]["message"] = "different service stopped"

    with pytest.raises(ValueError, match="not the v17 SIGTERM"):
        validate_external_interruption_evidence(
            json.loads(json.dumps(evidence)),
            profile=V17_INTERRUPTION_RECOVERY_PROFILE,
        )


class _SettlementResult:
    def __init__(self, *, row: SimpleNamespace | None = None, rowcount: int = 0) -> None:
        self._row = row
        self.rowcount = rowcount

    def one(self) -> SimpleNamespace:
        assert self._row is not None
        return self._row


class _SettlementConnection:
    def __init__(self, job: SimpleNamespace, backtest: SimpleNamespace) -> None:
        self.job = job
        self.backtest = backtest
        self.updates: list[object] = []

    def scalar(self, _statement: object) -> str:
        return V17_RECOVERY_DATABASE_APPLICATION_NAME

    def execute(self, statement: object) -> _SettlementResult:
        if getattr(statement, "is_select", False):
            table = statement.get_final_froms()[0].name
            return _SettlementResult(
                row=self.job if table == "jobs" else self.backtest
            )
        if getattr(statement, "is_update", False):
            self.updates.append(statement)
            return _SettlementResult(rowcount=1)
        raise AssertionError(f"unexpected settlement statement: {statement}")


class _SettlementTransaction:
    def __init__(self, connection: _SettlementConnection) -> None:
        self.connection = connection

    def __enter__(self) -> _SettlementConnection:
        return self.connection

    def __exit__(self, *_args: object) -> None:
        return None


class _SettlementEngine:
    def __init__(self, connection: _SettlementConnection) -> None:
        self.connection = connection

    def begin(self) -> _SettlementTransaction:
        return _SettlementTransaction(self.connection)


@pytest.mark.parametrize(
    ("backtest_status", "existing_error"),
    [
        ("succeeded", None),
        ("failed", "worker failed after producing evidence"),
        ("cancelled", "worker cancellation after producing evidence"),
    ],
)
def test_controller_failure_settlement_preserves_partial_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    backtest_status: str,
    existing_error: str | None,
) -> None:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    metrics = {"annualized_return": 0.12, "evidence_sha256": "a" * 64}
    produced_artifact = tmp_path / "attempt-2" / "metrics.json"
    produced_artifact.parent.mkdir()
    produced_artifact.write_text(json.dumps(metrics), encoding="utf-8")
    connection = _SettlementConnection(
        SimpleNamespace(
            status="running",
            attempts=2,
            max_attempts=2,
            finished_at=None,
        ),
        SimpleNamespace(
            status=backtest_status,
            artifact_path=profile.target_artifact_path,
            metrics_json=metrics,
            error=existing_error,
            finished_at=datetime.fromisoformat("2026-08-31T01:00:00+00:00"),
        ),
    )
    store = object.__new__(FormalBacktestInterruptionRecoveryStore)
    store.engine = _SettlementEngine(connection)
    store.data_root = tmp_path
    store.profile = profile
    monkeypatch.setattr(store, "_registered_row", lambda _connection: object())
    monkeypatch.setattr(
        store,
        "_validate_registered",
        lambda _connection, _row: {"receipt_sha256": "b" * 64},
    )

    assert store.settle_controller_failure() == {
        "status": "settled_failed",
        "job_id": profile.job_id,
    }

    updates = {statement.table.name: statement for statement in connection.updates}
    backtest_values = {
        key: value.value for key, value in updates["backtest_runs"]._values.items()
    }
    assert backtest_values["status"] == "failed"
    assert backtest_values["error"] == RECOVERY_CONTROLLER_FAILURE_ERROR
    assert "metrics_json" not in backtest_values
    assert "finished_at" not in backtest_values
    assert json.loads(produced_artifact.read_text(encoding="utf-8")) == metrics
