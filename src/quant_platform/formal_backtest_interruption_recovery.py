"""One governed continuation for the interrupted production v17 formal OOS.

The statistical identity does not change: the same StrategyVersion, OOS
vintage, backtest and durable job receive exactly one additional process
execution.  Before that mutable state is requeued, the complete pre-result
interruption evidence is sealed in an append-only database receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import insert, select, text, update

from quant_data.database import (
    audit_events,
    backtest_runs,
    formal_backtest_interruption_recoveries,
    jobs,
    open_database,
    strategy_versions,
    transparent_baseline_pre_result_repairs,
)
from quant_platform.transparent_baseline_lockbox import (
    canonical_sha256,
    validate_pre_result_repair_audit_event,
)
from quant_platform.transparent_baseline_runner import (
    TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
    TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
)

RECOVERY_CONTRACT_VERSION = "transparent-baseline-service-interruption-recovery-v1"
RECOVERY_GENERATION = "v17-control-plane-backup-sigterm-20260831"
RECOVERY_REASON_CODE = "external_service_sigterm_before_formal_result"
RECOVERY_AUDIT_ACTION = "formal_backtest_interruption_recovery_registered"
EXTERNAL_EVIDENCE_CONTRACT_VERSION = "quantlab-external-service-interruption-evidence-v1"
INTERRUPTED_JOB_ERROR = (
    "Worker restarted after the bounded attempt limit; operator review is required"
)
RECOVERY_CONTROLLER_FAILURE_ERROR = (
    "Sealed v17 recovery controller terminated before worker finalization; "
    "no further execution is authorized"
)
V17_REPAIR_RECEIPT_SHA256 = (
    "980ea643755d261cc7ee39ffec5e23af3f8e27ebdc1e2f2a647e0802b9dc636d"
)
V17_EXTERNAL_INTERRUPTION_OBSERVED_AT = "2026-08-30T19:30:09.189629+00:00"
V17_EXTERNAL_JOURNAL_SHA256 = (
    "c7a93f1fe86a57ed6fcd51f32fe74be55ef7115cb4a79454e12ccbee338efcc7"
)
V17_RECOVERY_CONTROLLER_CONTRACT_VERSION = (
    "quantlab-v17-sealed-one-shot-controller-v1"
)
V17_RECOVERY_DATABASE_APPLICATION_NAME = "quantlab-v17-recovery-b41f78f9"
V17_RECOVERY_AUTHORIZER_APPLICATION_NAME = "quantlab-v17-recovery-authorizer"
V17_RECOVERY_CONTROLLER_SHA256 = (
    "1329aaf46bc9db8a3172abf6ecbce2837a12f1b555aad7c33c4ff0a9e70a2c0d"
)
V17_INTERRUPTION_RECOVERY_RECEIPT_SHA256 = (
    "6345c455862d8bb587e12f8ce0be7c1da291f2dde82fbdbb31e954bb88b9d3df"
)
V17_EXTERNAL_JOURNAL_EXCERPT: Mapping[str, Any] = {
    "contract_version": "quantlab-systemd-docker-journal-excerpt-v1",
    "records": [
        {
            "message": (
                "Starting quantlab-backup.service - QuantLab bounded control-plane "
                "backup..."
            ),
            "observed_at": "2026-08-30T19:29:57.618523+00:00",
            "source": "systemd",
            "unit": "quantlab-backup.service",
        },
        {
            "message": "Container quantlab-platform-evaluation-worker-1 Stopping",
            "observed_at": V17_EXTERNAL_INTERRUPTION_OBSERVED_AT,
            "source": "systemd",
            "unit": "quantlab-backup.service",
        },
        {
            "container_id": (
                "710730b1ce29bdd43e081f967860e7f49a791e9e62e62aeeaa5e2b98e3f03218"
            ),
            "daemon_shutting_down": False,
            "exit_status": 0,
            "has_been_manually_stopped": True,
            "observed_at": "2026-08-30T19:30:14.800407712+00:00",
            "source": "dockerd",
        },
        {
            "message": "Container quantlab-platform-evaluation-worker-1 Stopped",
            "observed_at": "2026-08-30T19:30:14.869740+00:00",
            "source": "systemd",
            "unit": "quantlab-backup.service",
        },
        {
            "message": "Container quantlab-platform-evaluation-worker-1 Started",
            "observed_at": "2026-08-30T19:34:33.195752+00:00",
            "source": "systemd",
            "unit": "quantlab-backup.service",
        },
        {
            "message": (
                "Finished quantlab-backup.service - QuantLab bounded control-plane "
                "backup."
            ),
            "observed_at": "2026-08-30T19:35:56.019550+00:00",
            "source": "systemd",
            "unit": "quantlab-backup.service",
        },
    ],
}


@dataclass(frozen=True)
class InterruptionRecoveryProfile:
    backtest_id: str
    job_id: str
    strategy_version_id: str
    idempotency_key: str
    dataset: str
    dataset_path: str
    dataset_identity_sha256: str
    dataset_lineage_id: str
    periods: Mapping[str, str]
    recipe_id: str
    recipe_version: str
    recipe_sha256: str
    runner_sha256: str
    runtime_bundle_sha256: str
    worker_runtime_image_digest: str
    strategy_rules_sha256: str
    execution_contract_hash: str
    qlib_version: str
    qlib_commit: str
    rdagent_version: str
    rdagent_commit: str
    source_job_created_at: str
    source_job_started_at: str
    source_job_finished_at: str
    source_backtest_created_at: str
    source_backtest_started_at: str
    source_payload_sha256: str
    source_log_path: str
    source_log_prefix_bytes: int
    source_log_prefix_sha256: str
    source_artifact_path: str
    source_manifest_sha256: str
    source_artifact_inventory: tuple[Mapping[str, Any], ...]
    source_artifact_inventory_sha256: str
    target_artifact_path: str
    target_execution_log_path: str

    @property
    def expected_payload(self) -> dict[str, Any]:
        return {
            "backtest_id": self.backtest_id,
            "strategy_version_id": self.strategy_version_id,
            "dataset": self.dataset,
            "dataset_path": self.dataset_path,
            "execution_dataset": None,
            "periods": dict(self.periods),
            TRANSPARENT_BASELINE_JOB_RUNNER_FIELD: self.runner_sha256,
            TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD: self.runtime_bundle_sha256,
            TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD: (
                self.worker_runtime_image_digest
            ),
        }


V17_INTERRUPTED_ARTIFACT_INVENTORY: tuple[Mapping[str, Any], ...] = (
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
        "bytes": 69_327,
        "sha256": "a4b72701a88247a7bf783b3bfe2950131c936d6d3dcea7d42601ab5bb8d115c7",
    },
)

V17_INTERRUPTION_RECOVERY_PROFILE = InterruptionRecoveryProfile(
    backtest_id="0a113fe28ca741b6be9c09ab046c9d02",
    job_id="858a75a6f1994c359fa9c3567ed09f57",
    strategy_version_id="4414d202dbb641608975e5305bc18da4",
    idempotency_key=(
        "transparent-baseline:4414d202dbb641608975e5305bc18da4:"
        "0a113fe28ca741b6be9c09ab046c9d02"
    ),
    dataset="cn-20080101-20260828-v7-failclosed-ed5c8b3",
    dataset_path="/data/qlib/cn-20080101-20260828-v7-failclosed-ed5c8b3",
    dataset_identity_sha256=(
        "eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2"
    ),
    dataset_lineage_id=(
        "1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e"
    ),
    periods={
        "start": "2018-11-08",
        "end": "2019-11-20",
        "historical_start": "2008-01-02",
        "historical_end": "2018-10-10",
    },
    recipe_id="short_relative_strength",
    recipe_version="qlib-rdagent-single-mainline-2026-08-31-v17",
    recipe_sha256=(
        "dee3551a73f2ebb3fbbbddfafdf99f4618e3dfdd98981fdb4b4d8849723d5fd8"
    ),
    runner_sha256=(
        "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
    ),
    runtime_bundle_sha256=(
        "c3d6aeb00a8f5286ee117f885231b43ad49e461c5183f5cbf32c4601824bb9fc"
    ),
    worker_runtime_image_digest=(
        "sha256:b41f78f9c99dd9853998d85a52593bcab247907ac60f9d721d863e20904e8bb7"
    ),
    strategy_rules_sha256=(
        "644d9ee73ea4c167c7d8f58b2e8b1707289cb569c48ae73ce131cce139f7e756"
    ),
    execution_contract_hash=(
        "0ffa8939a70f0c46499e3ae472b877f68cce1b9fb95dfe81886ab8430ddcce61"
    ),
    qlib_version="0.0.dev0+gd5379c520f66a39953bad76234a7019a72796fd0",
    qlib_commit="d5379c520f66a39953bad76234a7019a72796fd0",
    rdagent_version="0.0.dev0+g4f9ecb005881cddc08df0124a2e894c018007679",
    rdagent_commit="4f9ecb005881cddc08df0124a2e894c018007679",
    source_job_created_at="2026-08-30T18:56:25.236023+00:00",
    source_job_started_at="2026-08-30T18:56:26.193585+00:00",
    source_job_finished_at="2026-08-30T19:34:35.198471+00:00",
    source_backtest_created_at="2026-08-30T18:56:25.223043+00:00",
    source_backtest_started_at="2026-08-30T18:56:26.196535+00:00",
    source_payload_sha256=(
        "f99fa6c8f2a1364d56a0d0bff9d7401b1f504f3b020c321d996f05a70a05df42"
    ),
    source_log_path=(
        "/data/platform/logs/strategy-backtest-0a113fe28ca741b6be9c09ab046c9d02.log"
    ),
    source_log_prefix_bytes=25_665,
    source_log_prefix_sha256=(
        "c287af99368bf95d16330513f155f795b5fe77bc1e595c47d2fed7a1a447addb"
    ),
    source_artifact_path=(
        "/data/artifacts/backtests/0a113fe28ca741b6be9c09ab046c9d02"
    ),
    source_manifest_sha256=(
        "a4b72701a88247a7bf783b3bfe2950131c936d6d3dcea7d42601ab5bb8d115c7"
    ),
    source_artifact_inventory=V17_INTERRUPTED_ARTIFACT_INVENTORY,
    source_artifact_inventory_sha256=(
        "c0272f59ca8878d26e95db2b4328cfc9551c4b3dff4382a237c38cd87f00f63e"
    ),
    target_artifact_path=(
        "/data/artifacts/formal-backtest-recoveries/"
        "0a113fe28ca741b6be9c09ab046c9d02/attempt-2"
    ),
    target_execution_log_path=(
        "/data/artifacts/formal-backtest-recoveries/"
        "0a113fe28ca741b6be9c09ab046c9d02/attempt-2.log"
    ),
)

_TERMINAL_ARTIFACTS = (
    "artifact_manifest.json",
    "daily_returns.parquet",
    "result.json",
)
_EXTERNAL_EVIDENCE_KEYS = {
    "contract_version",
    "source",
    "service",
    "stop_owner",
    "signal",
    "has_been_manually_stopped",
    "oom_killed",
    "backtest_id",
    "job_id",
    "observed_at",
    "journal_sha256",
    "journal_excerpt",
}


def _sha256_file(path: Path, *, byte_limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = byte_limit
    with path.open("rb") as handle:
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            chunk = handle.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if remaining not in {None, 0}:
        raise ValueError("formal backtest interruption log prefix is truncated")
    return digest.hexdigest()


def _iso_utc(value: Any, *, field: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware database timestamp")
    return value.astimezone(UTC).isoformat()


def _is_sha256(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return len(normalized) == 64 and all(item in "0123456789abcdef" for item in normalized)


def _host_data_path(logical_path: str, *, data_root: Path) -> Path:
    logical = PurePosixPath(logical_path)
    if not logical.is_absolute() or logical.parts[:2] != ("/", "data"):
        raise ValueError("formal backtest recovery paths must be rooted at /data")
    root = data_root.resolve()
    candidate = root.joinpath(*logical.parts[2:])
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError("formal backtest recovery path escapes the data root") from exc
    return candidate


def validate_external_interruption_evidence(
    value: Mapping[str, Any], *, profile: InterruptionRecoveryProfile
) -> dict[str, Any]:
    evidence = json.loads(json.dumps(dict(value), ensure_ascii=False))
    if set(evidence) != _EXTERNAL_EVIDENCE_KEYS:
        raise ValueError("external service-interruption evidence shape is invalid")
    expected = {
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
        "journal_excerpt": json.loads(
            json.dumps(V17_EXTERNAL_JOURNAL_EXCERPT, ensure_ascii=False)
        ),
    }
    if evidence != expected:
        raise ValueError("external service-interruption evidence is not the v17 SIGTERM")
    if canonical_sha256(evidence["journal_excerpt"]) != V17_EXTERNAL_JOURNAL_SHA256:
        raise ValueError("external service-interruption journal excerpt hash changed")
    return evidence


def _verify_log_prefix(
    profile: InterruptionRecoveryProfile,
    *,
    data_root: Path,
    require_exact_size: bool,
) -> dict[str, Any]:
    path = _host_data_path(profile.source_log_path, data_root=data_root)
    if not path.is_file() or path.is_symlink():
        raise ValueError("formal backtest interruption log is missing or not regular")
    size = path.stat().st_size
    if size < profile.source_log_prefix_bytes or (
        require_exact_size and size != profile.source_log_prefix_bytes
    ):
        raise ValueError("formal backtest interruption log size changed")
    observed = _sha256_file(path, byte_limit=profile.source_log_prefix_bytes)
    if observed != profile.source_log_prefix_sha256:
        raise ValueError("formal backtest interruption log prefix changed")
    return {
        "path": profile.source_log_path,
        "bytes": profile.source_log_prefix_bytes,
        "sha256": observed,
    }


def _verify_source_artifacts(
    profile: InterruptionRecoveryProfile,
    *,
    data_root: Path,
) -> list[dict[str, Any]]:
    root = _host_data_path(profile.source_artifact_path, data_root=data_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("formal backtest interruption artifact root is missing")
    expected = {
        str(item["path"]): {
            "path": str(item["path"]),
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in profile.source_artifact_inventory
    }
    observed_paths: set[str] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if directory_path.is_symlink():
            raise ValueError("formal backtest interruption artifacts contain a symlink")
        for name in names:
            child = directory_path / name
            if child.is_symlink():
                raise ValueError("formal backtest interruption artifacts contain a symlink")
        for filename in filenames:
            path = directory_path / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("formal backtest interruption artifacts are not regular")
            relative = path.relative_to(root).as_posix()
            observed_paths.add(relative)
    if observed_paths != set(expected):
        raise ValueError("formal backtest interruption artifact inventory changed")
    files: list[dict[str, Any]] = []
    for relative, item in sorted(expected.items()):
        path = root / relative
        if path.stat().st_size != item["bytes"] or _sha256_file(path) != item["sha256"]:
            raise ValueError("formal backtest interruption artifact content changed")
        files.append(item)
    if canonical_sha256(files) != profile.source_artifact_inventory_sha256:
        raise ValueError("formal backtest interruption artifact inventory digest changed")
    if expected["manifest.json"]["sha256"] != profile.source_manifest_sha256:
        raise ValueError("formal backtest interruption manifest identity changed")
    return files


def _validate_target_is_unopened(
    profile: InterruptionRecoveryProfile, *, data_root: Path
) -> None:
    target = _host_data_path(profile.target_artifact_path, data_root=data_root)
    if target.is_symlink():
        raise ValueError("formal backtest recovery target cannot be a symlink")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("formal backtest recovery target already contains execution output")
    log_path = _host_data_path(profile.target_execution_log_path, data_root=data_root)
    if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
        raise ValueError("formal backtest recovery execution log is not a regular file")
    if log_path.exists() and log_path.stat().st_size:
        raise ValueError("formal backtest recovery execution log is not empty")


def _version_binding(version: Any, profile: InterruptionRecoveryProfile) -> dict[str, Any]:
    config = dict(version.config_json or {})
    bootstrap = dict(config.get("transparent_baseline_bootstrap") or {})
    if (
        str(version.id) != profile.strategy_version_id
        or str(version.status) != "draft"
        or str(version.strategy_type) != "multifactor"
        or str(version.horizon_profile) != "short_1_5d"
        or str(version.strategy_rules_sha256 or "") != profile.strategy_rules_sha256
        or str(version.execution_contract_hash) != profile.execution_contract_hash
        or config.get("recipe_id") != profile.recipe_id
        or config.get("recipe_version") != profile.recipe_version
        or bootstrap.get("recipe_sha256") != profile.recipe_sha256
        or bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD) != profile.runner_sha256
        or bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
        != profile.runtime_bundle_sha256
        or bootstrap.get(TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD)
        != profile.worker_runtime_image_digest
        or bootstrap.get("dataset") != profile.dataset
        or bootstrap.get("dataset_identity_sha256") != profile.dataset_identity_sha256
        or bootstrap.get("dataset_lineage_id") != profile.dataset_lineage_id
        or dict(bootstrap.get("formal_periods") or {}) != dict(profile.periods)
    ):
        raise ValueError("v17 formal backtest immutable strategy binding changed")
    return {
        "strategy_version_id": profile.strategy_version_id,
        "recipe_id": profile.recipe_id,
        "recipe_version": profile.recipe_version,
        "recipe_sha256": profile.recipe_sha256,
        "strategy_rules_sha256": profile.strategy_rules_sha256,
        "execution_contract_hash": profile.execution_contract_hash,
        "runner_sha256": profile.runner_sha256,
        "runtime_bundle_sha256": profile.runtime_bundle_sha256,
        "worker_runtime_image_digest": profile.worker_runtime_image_digest,
        "dataset": profile.dataset,
        "dataset_identity_sha256": profile.dataset_identity_sha256,
        "dataset_lineage_id": profile.dataset_lineage_id,
        "periods": dict(profile.periods),
        "source_repair_receipt_sha256": V17_REPAIR_RECEIPT_SHA256,
    }


def _source_job_snapshot(row: Any, profile: InterruptionRecoveryProfile) -> dict[str, Any]:
    payload = dict(row.payload_json or {})
    if (
        str(row.id) != profile.job_id
        or str(row.kind) != "strategy_backtest"
        or str(row.idempotency_key or "") != profile.idempotency_key
        or str(row.status) != "failed"
        or int(row.attempts) != 1
        or int(row.max_attempts) != 1
        or int(row.exit_code or 0) != 143
        or str(row.error or "") != INTERRUPTED_JOB_ERROR
        or row.progress_json is not None
        or row.next_attempt_at is not None
        or row.cancel_requested_at is not None
        or str(row.log_path or "") != profile.source_log_path
        or payload != profile.expected_payload
        or canonical_sha256(payload) != profile.source_payload_sha256
        or _iso_utc(row.created_at, field="job.created_at")
        != profile.source_job_created_at
        or _iso_utc(row.started_at, field="job.started_at")
        != profile.source_job_started_at
        or _iso_utc(row.finished_at, field="job.finished_at")
        != profile.source_job_finished_at
    ):
        raise ValueError("v17 formal backtest interrupted job evidence changed")
    payload = {
        "id": profile.job_id,
        "kind": "strategy_backtest",
        "idempotency_key": profile.idempotency_key,
        "status": "failed",
        "attempts": 1,
        "max_attempts": 1,
        "exit_code": 143,
        "error": INTERRUPTED_JOB_ERROR,
        "progress_absent": True,
        "payload_sha256": profile.source_payload_sha256,
        "log_path": profile.source_log_path,
        "created_at": profile.source_job_created_at,
        "started_at": profile.source_job_started_at,
        "finished_at": profile.source_job_finished_at,
    }
    return {**payload, "row_sha256": canonical_sha256(payload)}


def _source_backtest_snapshot(
    row: Any, version: Any, profile: InterruptionRecoveryProfile
) -> dict[str, Any]:
    if (
        str(row.id) != profile.backtest_id
        or str(row.job_id or "") != profile.job_id
        or str(row.strategy_version_id) != profile.strategy_version_id
        or str(row.dataset) != profile.dataset
        or row.execution_dataset is not None
        or str(row.status) != "running"
        or dict(row.periods_json or {}) != dict(profile.periods)
        or str(row.artifact_path or "") != profile.source_artifact_path
        or row.metrics_json is not None
        or row.error is not None
        or row.finished_at is not None
        or str(row.execution_contract_hash) != profile.execution_contract_hash
        or _iso_utc(row.created_at, field="backtest.created_at")
        != profile.source_backtest_created_at
        or _iso_utc(row.started_at, field="backtest.started_at")
        != profile.source_backtest_started_at
        or row.qlib_version != version.qlib_version
        or row.qlib_commit != version.qlib_commit
        or row.rdagent_version != version.rdagent_version
        or row.rdagent_commit != version.rdagent_commit
        or str(row.qlib_version or "") != profile.qlib_version
        or str(row.qlib_commit or "") != profile.qlib_commit
        or str(row.rdagent_version or "") != profile.rdagent_version
        or str(row.rdagent_commit or "") != profile.rdagent_commit
    ):
        raise ValueError("v17 formal backtest interrupted aggregate evidence changed")
    payload = {
        "id": profile.backtest_id,
        "job_id": profile.job_id,
        "strategy_version_id": profile.strategy_version_id,
        "status": "running",
        "dataset": profile.dataset,
        "execution_dataset": None,
        "periods": dict(profile.periods),
        "artifact_path": profile.source_artifact_path,
        "metrics_absent": True,
        "error_absent": True,
        "finished_at_absent": True,
        "execution_contract_hash": profile.execution_contract_hash,
        "qlib_version": row.qlib_version,
        "qlib_commit": row.qlib_commit,
        "rdagent_version": row.rdagent_version,
        "rdagent_commit": row.rdagent_commit,
        "created_at": profile.source_backtest_created_at,
        "started_at": profile.source_backtest_started_at,
    }
    return {**payload, "row_sha256": canonical_sha256(payload)}


def build_interruption_recovery_receipt(
    *,
    source_job: Mapping[str, Any],
    source_backtest: Mapping[str, Any],
    immutable_binding: Mapping[str, Any],
    log_prefix: Mapping[str, Any],
    files: Sequence[Mapping[str, Any]],
    external_interruption: Mapping[str, Any],
    profile: InterruptionRecoveryProfile = V17_INTERRUPTION_RECOVERY_PROFILE,
) -> dict[str, Any]:
    payload = {
        "contract_version": RECOVERY_CONTRACT_VERSION,
        "recovery_generation": RECOVERY_GENERATION,
        "reason_code": RECOVERY_REASON_CODE,
        "performance_information_used": False,
        "source_job": dict(source_job),
        "source_backtest": dict(source_backtest),
        "immutable_binding": dict(immutable_binding),
        "pre_result_evidence": {
            "job_progress_absent": True,
            "backtest_metrics_absent": True,
            "result_absent": True,
            "terminal_artifacts_absent": list(_TERMINAL_ARTIFACTS),
            "log_prefix": dict(log_prefix),
            "artifact_path": profile.source_artifact_path,
            "artifact_inventory": [dict(item) for item in files],
            "artifact_inventory_sha256": profile.source_artifact_inventory_sha256,
            "manifest_sha256": profile.source_manifest_sha256,
        },
        "external_interruption": dict(external_interruption),
        "execution_controller": {
            "contract_version": V17_RECOVERY_CONTROLLER_CONTRACT_VERSION,
            "application_name": V17_RECOVERY_DATABASE_APPLICATION_NAME,
            "authorization_application_name": (
                V17_RECOVERY_AUTHORIZER_APPLICATION_NAME
            ),
            "controller_sha256": V17_RECOVERY_CONTROLLER_SHA256,
            "sealed_worker_image_digest": profile.worker_runtime_image_digest,
            "canonical_output_path": profile.source_artifact_path,
            "target_artifact_path": profile.target_artifact_path,
            "target_execution_log_path": profile.target_execution_log_path,
            "data_mount_mode": (
                "volumes-from-read-only-with-persistent-target-samefile-bind"
            ),
            "target_execution_log_mount_mode": "single-file-read-write-bind",
        },
        "target": {
            "job_id": profile.job_id,
            "backtest_id": profile.backtest_id,
            "strategy_version_id": profile.strategy_version_id,
            "authorized_attempt": 2,
            "max_attempts": 2,
            "artifact_path": profile.target_artifact_path,
            "same_formal_oos_identity": True,
        },
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_interruption_recovery_receipt(receipt, profile=profile)


def _validate_snapshot_hash(
    value: Mapping[str, Any], *, expected_keys: set[str], label: str
) -> dict[str, Any]:
    snapshot = dict(value)
    if set(snapshot) != expected_keys | {"row_sha256"}:
        raise ValueError(f"formal backtest interruption {label} shape is invalid")
    supplied = str(snapshot.pop("row_sha256", "")).lower()
    if not _is_sha256(supplied) or canonical_sha256(snapshot) != supplied:
        raise ValueError(f"formal backtest interruption {label} hash changed")
    return {**snapshot, "row_sha256": supplied}


def validate_interruption_recovery_receipt(
    value: Mapping[str, Any],
    *,
    profile: InterruptionRecoveryProfile = V17_INTERRUPTION_RECOVERY_PROFILE,
) -> dict[str, Any]:
    receipt = json.loads(json.dumps(dict(value), ensure_ascii=False))
    expected_keys = {
        "contract_version",
        "recovery_generation",
        "reason_code",
        "performance_information_used",
        "source_job",
        "source_backtest",
        "immutable_binding",
        "pre_result_evidence",
        "external_interruption",
        "execution_controller",
        "target",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("formal backtest interruption receipt shape is invalid")
    supplied_hash = str(receipt.pop("receipt_sha256", "")).lower()
    if (
        supplied_hash != V17_INTERRUPTION_RECOVERY_RECEIPT_SHA256
        or canonical_sha256(receipt) != supplied_hash
    ):
        raise ValueError("formal backtest interruption receipt hash changed")
    receipt["receipt_sha256"] = supplied_hash
    source_job = _validate_snapshot_hash(
        dict(receipt.get("source_job") or {}),
        expected_keys={
            "id",
            "kind",
            "idempotency_key",
            "status",
            "attempts",
            "max_attempts",
            "exit_code",
            "error",
            "progress_absent",
            "payload_sha256",
            "log_path",
            "created_at",
            "started_at",
            "finished_at",
        },
        label="source job",
    )
    source_backtest = _validate_snapshot_hash(
        dict(receipt.get("source_backtest") or {}),
        expected_keys={
            "id",
            "job_id",
            "strategy_version_id",
            "status",
            "dataset",
            "execution_dataset",
            "periods",
            "artifact_path",
            "metrics_absent",
            "error_absent",
            "finished_at_absent",
            "execution_contract_hash",
            "qlib_version",
            "qlib_commit",
            "rdagent_version",
            "rdagent_commit",
            "created_at",
            "started_at",
        },
        label="source backtest",
    )
    binding = dict(receipt.get("immutable_binding") or {})
    evidence = dict(receipt.get("pre_result_evidence") or {})
    controller = dict(receipt.get("execution_controller") or {})
    target = dict(receipt.get("target") or {})
    files = [dict(item) for item in evidence.get("artifact_inventory") or []]
    if (
        receipt["contract_version"] != RECOVERY_CONTRACT_VERSION
        or receipt["recovery_generation"] != RECOVERY_GENERATION
        or receipt["reason_code"] != RECOVERY_REASON_CODE
        or receipt["performance_information_used"] is not False
        or source_job.get("id") != profile.job_id
        or source_job.get("kind") != "strategy_backtest"
        or source_job.get("idempotency_key") != profile.idempotency_key
        or source_job.get("status") != "failed"
        or source_job.get("attempts") != 1
        or source_job.get("max_attempts") != 1
        or source_job.get("exit_code") != 143
        or source_job.get("error") != INTERRUPTED_JOB_ERROR
        or source_job.get("progress_absent") is not True
        or source_job.get("payload_sha256") != profile.source_payload_sha256
        or source_job.get("log_path") != profile.source_log_path
        or source_job.get("created_at") != profile.source_job_created_at
        or source_job.get("started_at") != profile.source_job_started_at
        or source_job.get("finished_at") != profile.source_job_finished_at
        or source_backtest.get("id") != profile.backtest_id
        or source_backtest.get("job_id") != profile.job_id
        or source_backtest.get("strategy_version_id") != profile.strategy_version_id
        or source_backtest.get("status") != "running"
        or source_backtest.get("dataset") != profile.dataset
        or source_backtest.get("execution_dataset") is not None
        or source_backtest.get("periods") != dict(profile.periods)
        or source_backtest.get("artifact_path") != profile.source_artifact_path
        or source_backtest.get("metrics_absent") is not True
        or source_backtest.get("error_absent") is not True
        or source_backtest.get("finished_at_absent") is not True
        or source_backtest.get("execution_contract_hash")
        != profile.execution_contract_hash
        or source_backtest.get("qlib_version") != profile.qlib_version
        or source_backtest.get("qlib_commit") != profile.qlib_commit
        or source_backtest.get("rdagent_version") != profile.rdagent_version
        or source_backtest.get("rdagent_commit") != profile.rdagent_commit
        or source_backtest.get("created_at") != profile.source_backtest_created_at
        or source_backtest.get("started_at") != profile.source_backtest_started_at
        or binding
        != {
            "strategy_version_id": profile.strategy_version_id,
            "recipe_id": profile.recipe_id,
            "recipe_version": profile.recipe_version,
            "recipe_sha256": profile.recipe_sha256,
            "strategy_rules_sha256": profile.strategy_rules_sha256,
            "execution_contract_hash": profile.execution_contract_hash,
            "runner_sha256": profile.runner_sha256,
            "runtime_bundle_sha256": profile.runtime_bundle_sha256,
            "worker_runtime_image_digest": profile.worker_runtime_image_digest,
            "dataset": profile.dataset,
            "dataset_identity_sha256": profile.dataset_identity_sha256,
            "dataset_lineage_id": profile.dataset_lineage_id,
            "periods": dict(profile.periods),
            "source_repair_receipt_sha256": V17_REPAIR_RECEIPT_SHA256,
        }
        or set(evidence)
        != {
            "job_progress_absent",
            "backtest_metrics_absent",
            "result_absent",
            "terminal_artifacts_absent",
            "log_prefix",
            "artifact_path",
            "artifact_inventory",
            "artifact_inventory_sha256",
            "manifest_sha256",
        }
        or evidence.get("job_progress_absent") is not True
        or evidence.get("backtest_metrics_absent") is not True
        or evidence.get("result_absent") is not True
        or evidence.get("terminal_artifacts_absent") != list(_TERMINAL_ARTIFACTS)
        or evidence.get("artifact_path") != profile.source_artifact_path
        or evidence.get("artifact_inventory_sha256")
        != profile.source_artifact_inventory_sha256
        or evidence.get("manifest_sha256") != profile.source_manifest_sha256
        or files != [dict(item) for item in profile.source_artifact_inventory]
        or canonical_sha256(files) != profile.source_artifact_inventory_sha256
        or dict(evidence.get("log_prefix") or {})
        != {
            "path": profile.source_log_path,
            "bytes": profile.source_log_prefix_bytes,
            "sha256": profile.source_log_prefix_sha256,
        }
        or target
        != {
            "job_id": profile.job_id,
            "backtest_id": profile.backtest_id,
            "strategy_version_id": profile.strategy_version_id,
            "authorized_attempt": 2,
            "max_attempts": 2,
            "artifact_path": profile.target_artifact_path,
            "same_formal_oos_identity": True,
        }
        or controller
        != {
            "contract_version": V17_RECOVERY_CONTROLLER_CONTRACT_VERSION,
            "application_name": V17_RECOVERY_DATABASE_APPLICATION_NAME,
            "authorization_application_name": (
                V17_RECOVERY_AUTHORIZER_APPLICATION_NAME
            ),
            "controller_sha256": V17_RECOVERY_CONTROLLER_SHA256,
            "sealed_worker_image_digest": profile.worker_runtime_image_digest,
            "canonical_output_path": profile.source_artifact_path,
            "target_artifact_path": profile.target_artifact_path,
            "target_execution_log_path": profile.target_execution_log_path,
            "data_mount_mode": (
                "volumes-from-read-only-with-persistent-target-samefile-bind"
            ),
            "target_execution_log_mount_mode": "single-file-read-write-bind",
        }
    ):
        raise ValueError("formal backtest interruption receipt is not the sealed v17 recovery")
    validate_external_interruption_evidence(
        dict(receipt.get("external_interruption") or {}), profile=profile
    )
    return receipt


class FormalBacktestInterruptionRecoveryStore:
    """Register and consume the one production-specific recovery receipt."""

    def __init__(
        self,
        database_url: str,
        *,
        data_root: Path = Path("/data"),
        profile: InterruptionRecoveryProfile = V17_INTERRUPTION_RECOVERY_PROFILE,
    ) -> None:
        self.engine = open_database(database_url)
        self.data_root = data_root.expanduser().resolve()
        self.profile = profile

    def _validate_repair_chain(self, connection: Any) -> None:
        profile = self.profile
        repair = connection.execute(
            select(transparent_baseline_pre_result_repairs).where(
                transparent_baseline_pre_result_repairs.c.receipt_sha256
                == V17_REPAIR_RECEIPT_SHA256
            )
        ).one_or_none()
        if repair is None:
            raise ValueError("v17 source repair receipt is not registered")
        audit = connection.execute(
            select(audit_events).where(audit_events.c.id == repair.source_audit_event_id)
        ).one_or_none()
        if audit is None:
            raise ValueError("v17 source repair audit event is missing")
        source_receipt = validate_pre_result_repair_audit_event(
            audit, expected_receipt_sha256=V17_REPAIR_RECEIPT_SHA256
        )
        if (
            list(repair.target_strategy_version_ids_json) != [profile.strategy_version_id]
            or str(repair.target_recipe_version) != profile.recipe_version
            or str(repair.target_dataset_lineage_id) != profile.dataset_lineage_id
            or source_receipt.get("target_recipe_version") != profile.recipe_version
        ):
            raise ValueError("v17 source repair registry binding changed")

    def _registered_row(self, connection: Any) -> Any | None:
        return connection.execute(
            select(formal_backtest_interruption_recoveries).where(
                formal_backtest_interruption_recoveries.c.backtest_id
                == self.profile.backtest_id
            )
        ).one_or_none()

    def preflight_source(
        self, *, external_interruption: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Read and hash every source before the service is stopped."""

        external = validate_external_interruption_evidence(
            external_interruption, profile=self.profile
        )
        profile = self.profile
        with self.engine.begin() as connection:
            existing = self._registered_row(connection)
            if existing is not None:
                receipt = self._validate_registered(
                    connection,
                    existing,
                    external_interruption=external,
                )
                return {"status": "already_registered", "receipt": receipt}
            job = connection.execute(
                select(jobs).where(jobs.c.id == profile.job_id)
            ).one_or_none()
            backtest = connection.execute(
                select(backtest_runs).where(backtest_runs.c.id == profile.backtest_id)
            ).one_or_none()
            version = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == profile.strategy_version_id
                )
            ).one_or_none()
            if job is None or backtest is None or version is None:
                raise ValueError("v17 formal backtest interruption source rows are incomplete")
            self._validate_repair_chain(connection)
            _version_binding(version, profile)
            _source_job_snapshot(job, profile)
            _source_backtest_snapshot(backtest, version, profile)
            _verify_log_prefix(
                profile,
                data_root=self.data_root,
                require_exact_size=True,
            )
            _verify_source_artifacts(profile, data_root=self.data_root)
            _validate_target_is_unopened(profile, data_root=self.data_root)
        return {"status": "source_verified", "job_id": profile.job_id}

    def register_and_requeue(
        self, *, actor: str, external_interruption: Mapping[str, Any]
    ) -> dict[str, Any]:
        username = str(actor or "").strip()
        if not username:
            raise ValueError("formal backtest interruption recovery actor is required")
        external = validate_external_interruption_evidence(
            external_interruption, profile=self.profile
        )
        profile = self.profile
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "SET LOCAL application_name = "
                    "'quantlab-v17-recovery-authorizer'"
                )
            )
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:identity))"),
                {"identity": f"formal-backtest-interruption:{profile.backtest_id}"},
            )
            existing = self._registered_row(connection)
            if existing is not None:
                receipt = self._validate_registered(
                    connection, existing, external_interruption=external
                )
                current_job = connection.execute(
                    select(jobs).where(jobs.c.id == profile.job_id)
                ).one()
                current_backtest = connection.execute(
                    select(backtest_runs).where(backtest_runs.c.id == profile.backtest_id)
                ).one()
                return {
                    "status": "already_registered",
                    "receipt": receipt,
                    "job_status": str(current_job.status),
                    "job_attempts": int(current_job.attempts),
                    "backtest_status": str(current_backtest.status),
                }

            job = connection.execute(
                select(jobs).where(jobs.c.id == profile.job_id).with_for_update()
            ).one_or_none()
            backtest = connection.execute(
                select(backtest_runs)
                .where(backtest_runs.c.id == profile.backtest_id)
                .with_for_update()
            ).one_or_none()
            version = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.id == profile.strategy_version_id)
                .with_for_update()
            ).one_or_none()
            if job is None or backtest is None or version is None:
                raise ValueError("v17 formal backtest interruption source rows are incomplete")
            self._validate_repair_chain(connection)
            immutable_binding = _version_binding(version, profile)
            source_job = _source_job_snapshot(job, profile)
            source_backtest = _source_backtest_snapshot(backtest, version, profile)
            log_prefix = _verify_log_prefix(
                profile,
                data_root=self.data_root,
                require_exact_size=True,
            )
            files = _verify_source_artifacts(profile, data_root=self.data_root)
            _validate_target_is_unopened(profile, data_root=self.data_root)
            receipt = build_interruption_recovery_receipt(
                source_job=source_job,
                source_backtest=source_backtest,
                immutable_binding=immutable_binding,
                log_prefix=log_prefix,
                files=files,
                external_interruption=external,
                profile=profile,
            )
            now = datetime.now(UTC)
            event_id = connection.execute(
                insert(audit_events)
                .values(
                    user_id=None,
                    username=username,
                    action=RECOVERY_AUDIT_ACTION,
                    method="INTERNAL",
                    path="transparent-baseline/formal-backtest-interruption-recovery",
                    status_code=201,
                    ip_hash=None,
                    user_agent="recover_transparent_baseline_v17_interruption.py",
                    details_json=receipt,
                    created_at=now,
                )
                .returning(audit_events.c.id)
            ).scalar_one()
            connection.execute(
                insert(formal_backtest_interruption_recoveries).values(
                    receipt_sha256=receipt["receipt_sha256"],
                    source_audit_event_id=event_id,
                    backtest_id=profile.backtest_id,
                    job_id=profile.job_id,
                    strategy_version_id=profile.strategy_version_id,
                    contract_version=RECOVERY_CONTRACT_VERSION,
                    recovery_generation=RECOVERY_GENERATION,
                    reason_code=RECOVERY_REASON_CODE,
                    source_job_row_sha256=source_job["row_sha256"],
                    source_backtest_row_sha256=source_backtest["row_sha256"],
                    source_payload_sha256=profile.source_payload_sha256,
                    source_log_prefix_sha256=profile.source_log_prefix_sha256,
                    source_log_prefix_bytes=profile.source_log_prefix_bytes,
                    source_artifact_inventory_sha256=(
                        profile.source_artifact_inventory_sha256
                    ),
                    target_artifact_path=profile.target_artifact_path,
                    verification_json=receipt,
                    created_at=now,
                )
            )
            backtest_update = connection.execute(
                update(backtest_runs)
                .where(
                    backtest_runs.c.id == profile.backtest_id,
                    backtest_runs.c.status == "running",
                    backtest_runs.c.job_id == profile.job_id,
                    backtest_runs.c.metrics_json.is_(None),
                    backtest_runs.c.finished_at.is_(None),
                )
                .values(
                    status="queued",
                    artifact_path=profile.target_artifact_path,
                    error=None,
                    started_at=None,
                    finished_at=None,
                )
            )
            job_update = connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == profile.job_id,
                    jobs.c.kind == "strategy_backtest",
                    jobs.c.status == "failed",
                    jobs.c.attempts == 1,
                    jobs.c.max_attempts == 1,
                    jobs.c.exit_code == 143,
                    jobs.c.error == INTERRUPTED_JOB_ERROR,
                )
                .values(
                    status="queued",
                    max_attempts=2,
                    progress_json=None,
                    exit_code=None,
                    error=None,
                    started_at=None,
                    finished_at=None,
                    cancel_requested_at=None,
                    next_attempt_at=None,
                )
            )
            if backtest_update.rowcount != 1 or job_update.rowcount != 1:
                raise ValueError("v17 formal backtest recovery source changed during registration")
        return {
            "status": "registered_and_queued",
            "audit_event_id": int(event_id),
            "receipt": receipt,
            "job_id": profile.job_id,
            "backtest_id": profile.backtest_id,
            "authorized_attempt": 2,
        }

    def _validate_registered(
        self,
        connection: Any,
        row: Any,
        *,
        external_interruption: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = self.profile
        receipt = validate_interruption_recovery_receipt(
            dict(row.verification_json or {}), profile=profile
        )
        if external_interruption is not None and receipt["external_interruption"] != dict(
            external_interruption
        ):
            raise ValueError("v17 interruption recovery already has different external evidence")
        audit = connection.execute(
            select(audit_events).where(audit_events.c.id == row.source_audit_event_id)
        ).one_or_none()
        if (
            audit is None
            or str(audit.action) != RECOVERY_AUDIT_ACTION
            or str(audit.method) != "INTERNAL"
            or str(audit.path)
            != "transparent-baseline/formal-backtest-interruption-recovery"
            or int(audit.status_code) != 201
            or str(audit.user_agent)
            != "recover_transparent_baseline_v17_interruption.py"
            or audit.user_id is not None
            or audit.ip_hash is not None
            or audit.created_at != row.created_at
            or dict(audit.details_json or {}) != receipt
            or str(row.receipt_sha256) != receipt["receipt_sha256"]
            or str(row.job_id) != profile.job_id
            or str(row.backtest_id) != profile.backtest_id
            or str(row.strategy_version_id) != profile.strategy_version_id
            or str(row.source_job_row_sha256) != receipt["source_job"]["row_sha256"]
            or str(row.source_backtest_row_sha256)
            != receipt["source_backtest"]["row_sha256"]
            or str(row.contract_version) != RECOVERY_CONTRACT_VERSION
            or str(row.recovery_generation) != RECOVERY_GENERATION
            or str(row.reason_code) != RECOVERY_REASON_CODE
            or str(row.source_payload_sha256) != profile.source_payload_sha256
            or str(row.source_log_prefix_sha256) != profile.source_log_prefix_sha256
            or int(row.source_log_prefix_bytes) != profile.source_log_prefix_bytes
            or str(row.source_artifact_inventory_sha256)
            != profile.source_artifact_inventory_sha256
            or str(row.target_artifact_path) != profile.target_artifact_path
        ):
            raise ValueError("v17 interruption recovery registry or audit changed")
        self._validate_repair_chain(connection)
        version = connection.execute(
            select(strategy_versions).where(
                strategy_versions.c.id == profile.strategy_version_id
            )
        ).one_or_none()
        if version is None:
            raise ValueError("v17 interruption recovery strategy version is missing")
        _version_binding(version, profile)
        current_job = connection.execute(
            select(jobs).where(jobs.c.id == profile.job_id)
        ).one_or_none()
        current_backtest = connection.execute(
            select(backtest_runs).where(backtest_runs.c.id == profile.backtest_id)
        ).one_or_none()
        if (
            current_job is None
            or current_backtest is None
            or str(current_job.kind) != "strategy_backtest"
            or str(current_job.idempotency_key or "") != profile.idempotency_key
            or dict(current_job.payload_json or {}) != profile.expected_payload
            or canonical_sha256(dict(current_job.payload_json or {}))
            != profile.source_payload_sha256
            or str(current_job.log_path or "") != profile.source_log_path
            or int(current_job.attempts) not in {1, 2}
            or int(current_job.max_attempts) != 2
            or str(current_backtest.job_id or "") != profile.job_id
            or str(current_backtest.strategy_version_id) != profile.strategy_version_id
            or str(current_backtest.dataset) != profile.dataset
            or current_backtest.execution_dataset is not None
            or dict(current_backtest.periods_json or {}) != dict(profile.periods)
            or str(current_backtest.artifact_path or "") != profile.target_artifact_path
            or str(current_backtest.execution_contract_hash)
            != profile.execution_contract_hash
            or str(current_backtest.qlib_version or "") != profile.qlib_version
            or str(current_backtest.qlib_commit or "") != profile.qlib_commit
            or str(current_backtest.rdagent_version or "") != profile.rdagent_version
            or str(current_backtest.rdagent_commit or "") != profile.rdagent_commit
        ):
            raise ValueError("v17 interruption recovery mutable identity changed")
        _verify_log_prefix(
            profile,
            data_root=self.data_root,
            require_exact_size=False,
        )
        _verify_source_artifacts(profile, data_root=self.data_root)
        return receipt

    def require_authorized_queue(self) -> dict[str, Any]:
        profile = self.profile
        with self.engine.begin() as connection:
            row = self._registered_row(connection)
            if row is None:
                raise ValueError("formal backtest repeated execution has no recovery receipt")
            receipt = self._validate_registered(connection, row)
            current_job = connection.execute(
                select(jobs).where(jobs.c.id == profile.job_id)
            ).one()
            backtest = connection.execute(
                select(backtest_runs).where(backtest_runs.c.id == profile.backtest_id)
            ).one()
            version = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == profile.strategy_version_id
                )
            ).one()
            _version_binding(version, profile)
            if (
                str(current_job.status) != "queued"
                or int(current_job.attempts) != 1
                or int(current_job.max_attempts) != 2
                or dict(current_job.payload_json or {}) != profile.expected_payload
                or current_job.exit_code is not None
                or current_job.error is not None
                or current_job.finished_at is not None
                or str(backtest.status) != "queued"
                or str(backtest.job_id or "") != profile.job_id
                or str(backtest.strategy_version_id) != profile.strategy_version_id
                or str(backtest.artifact_path or "") != profile.target_artifact_path
                or backtest.metrics_json is not None
                or backtest.error is not None
                or backtest.finished_at is not None
            ):
                raise ValueError("authorized v17 recovery queue/aggregate is inconsistent")
            _validate_target_is_unopened(profile, data_root=self.data_root)
        return receipt

    def settle_controller_failure(self) -> dict[str, Any]:
        """Fail a claimed attempt two; never make it executable again."""

        profile = self.profile
        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            application_name = connection.scalar(
                text("SELECT current_setting('application_name', true)")
            )
            if str(application_name or "") != V17_RECOVERY_DATABASE_APPLICATION_NAME:
                raise ValueError("v17 recovery failure settler has the wrong application_name")
            row = self._registered_row(connection)
            if row is None:
                raise ValueError("v17 recovery failure settler has no receipt")
            self._validate_registered(connection, row)
            current_job = connection.execute(
                select(jobs).where(jobs.c.id == profile.job_id).with_for_update()
            ).one()
            backtest = connection.execute(
                select(backtest_runs)
                .where(backtest_runs.c.id == profile.backtest_id)
                .with_for_update()
            ).one()
            job_status = str(current_job.status)
            backtest_status = str(backtest.status)
            consistent_terminal = (
                job_status == "succeeded"
                and backtest_status == "succeeded"
                and current_job.finished_at is not None
                and backtest.finished_at is not None
                and backtest.metrics_json is not None
            ) or (
                job_status in {"failed", "cancelled"}
                and backtest_status in {"failed", "cancelled"}
                and current_job.finished_at is not None
                and backtest.finished_at is not None
            )
            if consistent_terminal:
                return {
                    "status": "already_terminal",
                    "job_status": job_status,
                    "backtest_status": backtest_status,
                }
            if (
                int(current_job.attempts) != 2
                or int(current_job.max_attempts) != 2
                or job_status not in {"running", "failed", "cancelled"}
                or backtest_status
                not in {"queued", "running", "succeeded", "failed", "cancelled"}
                or str(backtest.artifact_path or "") != profile.target_artifact_path
            ):
                raise ValueError("v17 recovery cannot safely settle the observed state")
            if job_status == "running":
                job_update = connection.execute(
                    update(jobs)
                    .where(
                        jobs.c.id == profile.job_id,
                        jobs.c.status == "running",
                        jobs.c.attempts == 2,
                        jobs.c.max_attempts == 2,
                    )
                    .values(
                        status="failed",
                        exit_code=125,
                        error=RECOVERY_CONTROLLER_FAILURE_ERROR,
                        finished_at=now,
                        cancel_requested_at=None,
                        next_attempt_at=None,
                    )
                )
                if job_update.rowcount != 1:
                    raise ValueError("v17 recovery job changed during failure settlement")
            # The historical worker commits the aggregate and job in separate
            # transactions.  A controller/process death can therefore leave a
            # completed aggregate (including formal metrics and artifacts)
            # beside a non-terminal job.  Fail closed without erasing that
            # already-produced evidence.  A terminal worker failure is already
            # authoritative and must likewise keep its original error/timing.
            if backtest_status in {
                "queued",
                "running",
                "succeeded",
                "failed",
                "cancelled",
            }:
                backtest_values: dict[str, Any] = {
                    "status": "failed",
                    "error": RECOVERY_CONTROLLER_FAILURE_ERROR,
                }
                if backtest_status in {"queued", "running"}:
                    backtest_values["finished_at"] = now
                backtest_update = connection.execute(
                    update(backtest_runs)
                    .where(
                        backtest_runs.c.id == profile.backtest_id,
                        backtest_runs.c.status == backtest_status,
                        backtest_runs.c.artifact_path == profile.target_artifact_path,
                    )
                    .values(**backtest_values)
                )
                if backtest_update.rowcount != 1:
                    raise ValueError(
                        "v17 recovery aggregate changed during failure settlement"
                    )
        return {"status": "settled_failed", "job_id": profile.job_id}

    def verify_terminal_execution(self) -> dict[str, Any]:
        profile = self.profile
        with self.engine.begin() as connection:
            row = self._registered_row(connection)
            if row is None:
                raise ValueError("v17 recovery terminal verification has no receipt")
            receipt = self._validate_registered(connection, row)
            current_job = connection.execute(
                select(jobs).where(jobs.c.id == profile.job_id)
            ).one()
            backtest = connection.execute(
                select(backtest_runs).where(backtest_runs.c.id == profile.backtest_id)
            ).one()
            job_status = str(current_job.status)
            backtest_status = str(backtest.status)
            if (
                job_status not in {"succeeded", "failed", "cancelled"}
                or int(current_job.attempts) != 2
                or int(current_job.max_attempts) != 2
                or current_job.finished_at is None
                or backtest.finished_at is None
                or (
                    job_status == "succeeded"
                    and (backtest_status != "succeeded" or backtest.metrics_json is None)
                )
                or (
                    job_status in {"failed", "cancelled"}
                    and backtest_status not in {"failed", "cancelled"}
                )
            ):
                raise ValueError("v17 recovery did not reach one consistent terminal state")
        if job_status == "succeeded":
            target = _host_data_path(profile.target_artifact_path, data_root=self.data_root)
            missing = [name for name in _TERMINAL_ARTIFACTS if not (target / name).is_file()]
            if missing:
                raise ValueError("v17 recovery succeeded without complete terminal artifacts")
        return {
            "status": "terminal_verified",
            "job_status": job_status,
            "backtest_status": backtest_status,
            "attempts": 2,
            "receipt_sha256": receipt["receipt_sha256"],
        }


def register_v17_interruption_recovery(
    database_url: str,
    *,
    actor: str,
    external_interruption: Mapping[str, Any],
    data_root: Path = Path("/data"),
) -> dict[str, Any]:
    return FormalBacktestInterruptionRecoveryStore(
        database_url,
        data_root=data_root,
    ).register_and_requeue(
        actor=actor,
        external_interruption=external_interruption,
    )
