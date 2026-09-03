"""Replay-governance evidence rules for sealed and consumed-history backtests.

These symbols were lifted verbatim out of the retired
``forward_only_rehabilitation`` module during weight-reduction phase C3; the
standalone transparent-baseline admission line is gone, so this module only
carries the *live* governance surface:

- ``worker`` and ``scripts/run_multifactor_backtest.py`` use
  ``EVIDENCE_MODE_REPLAY``/``EVIDENCE_MODE_SEALED``, ``REPLAY_MARKERS``,
  ``require_replay_config``/``require_replay_markers`` and the
  incomplete-family eligibility helpers on the live formal-OOS backtest path.
- ``strategy_store`` uses the ``EVIDENCE_MODE_*`` constants plus the
  incomplete-family and consumed-vintage validators during approval and
  backtest admission.
- ``promotion`` and ``simulation_store`` use ``EVIDENCE_MODE_REPLAY`` and
  ``require_qualification`` to handle historical replay versions read-only.
- ``deployment_readiness`` uses the ``TERMINAL_CASH_ONLY_*`` constants and
  ``require_terminal_cash_only_receipt`` for readiness checks.

No new replay admissions can occur; the admission entry point and its
qualification builders stayed behind with the deleted module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import insert, select, update

from quant_data.database import (
    backtest_runs,
    formal_backtest_interruption_recoveries,
    jobs,
    oos_vintages,
    row_dict,
    strategies,
    strategy_events,
    strategy_forward_only_rehabilitations,
    strategy_incomplete_family_eligibilities,
    strategy_versions,
)
from quant_platform.strategy_artifact_manifest import (
    STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION,
    validate_backtest_artifact_manifest,
)
from quant_platform.transparent_baseline_governance import (
    BOOTSTRAP_CONFIG_KEY,
    LOCKBOX_CONFIG_KEY,
    lockbox_member_link,
    validate_joint_lockbox,
    validate_unopened_history_selection,
)
from quant_platform.transparent_baseline_governance import (
    canonical_sha256 as lockbox_canonical_sha256,
)
from quant_platform.transparent_baseline_runner import (
    FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION,
    FORWARD_ONLY_REHABILITATION_TARGET_RUNNER_SHA256,
    FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
)

CONTRACT_VERSION = "forward-only-rehabilitation-v1"
EVIDENCE_MODE_LEGACY = "legacy_ambiguous"
EVIDENCE_MODE_SEALED = "sealed_final_oos"
EVIDENCE_MODE_REPLAY = "consumed_historical_replay"
REPLAY_AUTHORITY = "historical_description_only"
INCOMPLETE_FAMILY_ELIGIBILITY_VERSION = "incomplete-factor-family-eligibility-v1"
INCOMPLETE_FAMILY_AUTHORITY = "conservative_bonferroni_only"
_REQUIRED_MISSING_FAMILY_ARTIFACTS = frozenset(
    {
        "trial_daily_returns_matrix",
        "trial_score_grid_matrix",
        "trial_candidate_manifest_matrix",
    }
)

SOURCE_VERSION_ID = "4414d202dbb641608975e5305bc18da4"
SOURCE_BACKTEST_ID = "0a113fe28ca741b6be9c09ab046c9d02"
SOURCE_JOB_ID = "858a75a6f1994c359fa9c3567ed09f57"
SOURCE_INTERRUPTION_RECOVERY_RECEIPT_SHA256 = (
    "6345c455862d8bb587e12f8ce0be7c1da291f2dde82fbdbb31e954bb88b9d3df"
)
SOURCE_INTERRUPTION_RECEIPT_AUTHORITY = "interruption_identity_only_not_pre_result"
SOURCE_CASH_ONLY_SCOPE = "cash_only_projection_only"
SOURCE_DATASET = "cn-20080101-20260828-v7-failclosed-ed5c8b3"
SOURCE_DATASET_IDENTITY_SHA256 = (
    "eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2"
)
SOURCE_DATASET_LINEAGE_ID = (
    "1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e"
)
SOURCE_RULES_SHA256 = (
    "644d9ee73ea4c167c7d8f58b2e8b1707289cb569c48ae73ce131cce139f7e756"
)
SOURCE_EXECUTION_CONTRACT_HASH = (
    "0ffa8939a70f0c46499e3ae472b877f68cce1b9fb95dfe81886ab8430ddcce61"
)
SOURCE_PERIODS = {
    "start": "2018-11-08",
    "end": "2019-11-20",
    "historical_start": "2008-01-02",
    "historical_end": "2018-10-10",
}
SOURCE_ATTEMPT_ARTIFACT_RELATIVE_PATH = (
    "artifacts/formal-backtest-recoveries/"
    f"{SOURCE_BACKTEST_ID}/attempt-2"
)

# One exact v18 public-control attempt reached all four primary robustness
# scenarios before the nominal-result reconstruction rejected legitimate
# zero-amount execution evidence.  The independently recomputed scenarios are
# all economically negative, so rerunning that already-opened historical
# window cannot make the public control eligible.  This receipt records only a
# terminal cash/NO_ACTION projection.  It is deliberately stored on the
# already-failed backtest and never turns that row into a successful result.
TERMINAL_CASH_ONLY_CONTRACT_VERSION = (
    "transparent-baseline-terminal-cash-only-v1"
)
TERMINAL_CASH_ONLY_AUTHORITY = SOURCE_CASH_ONLY_SCOPE
TERMINAL_CASH_ONLY_AUDIT_ACTION = "strategy.terminal_cash_only_registered"
TERMINAL_CASH_ONLY_WRAPPER_KEY = "terminal_cash_only_receipt"
TERMINAL_CASH_ONLY_VERSION_ID = "47eb2ea1152e438f9579843d2f3fb16f"
TERMINAL_CASH_ONLY_BACKTEST_ID = "2daa5d6d976f4a5d9fdc2d9e6d279884"
TERMINAL_CASH_ONLY_JOB_ID = "4ebf57ad734a4fcc90b7b4c163819baf"
TERMINAL_CASH_ONLY_FAILURE = (
    "ValueError: formal fill ledger contains invalid position evidence"
)
TERMINAL_CASH_ONLY_REASON_CODE = (
    "public_control_failed_all_primary_robustness_scenarios"
)
TERMINAL_CASH_ONLY_CORE_SCENARIOS = (
    "double_cost",
    "turnover_75pct",
    "topk_80pct",
    "zero_retention_buffer",
)
TERMINAL_CASH_ONLY_RECIPE_ID = "short_relative_strength"
TERMINAL_CASH_ONLY_HORIZON = "short_1_5d"
TERMINAL_CASH_ONLY_ARTIFACT_RELATIVE_PATH = (
    f"artifacts/backtests/{TERMINAL_CASH_ONLY_BACKTEST_ID}"
)

REPLAY_MARKERS: dict[str, Any] = {
    "evidence_mode": EVIDENCE_MODE_REPLAY,
    "historical_replay_opened": True,
    "consumed_oos_replayed": True,
    "sealed_final_oos": False,
    "unseen_oos": False,
    "authority": REPLAY_AUTHORITY,
}


def require_source_cancellation(connection: Any) -> dict[str, Any]:
    """Bind the exact v17 cancellation without reviving its pre-result claim.

    Attempt 2 later opened and consumed the historical performance window.  The
    old interruption receipt is therefore useful only as immutable interruption
    identity; it is never authorization to reuse an unseen/final OOS.
    """

    source_version = connection.execute(
        select(strategy_versions).where(strategy_versions.c.id == SOURCE_VERSION_ID)
    ).first()
    source_backtest = connection.execute(
        select(backtest_runs).where(backtest_runs.c.id == SOURCE_BACKTEST_ID)
    ).first()
    source_job = connection.execute(
        select(jobs).where(jobs.c.id == SOURCE_JOB_ID)
    ).first()
    interruption = connection.execute(
        select(formal_backtest_interruption_recoveries).where(
            formal_backtest_interruption_recoveries.c.receipt_sha256
            == SOURCE_INTERRUPTION_RECOVERY_RECEIPT_SHA256,
            formal_backtest_interruption_recoveries.c.strategy_version_id
            == SOURCE_VERSION_ID,
            formal_backtest_interruption_recoveries.c.backtest_id
            == SOURCE_BACKTEST_ID,
            formal_backtest_interruption_recoveries.c.job_id == SOURCE_JOB_ID,
        )
    ).first()
    if (
        source_version is None
        or source_backtest is None
        or source_job is None
        or interruption is None
        or str(source_version.strategy_rules_sha256) != SOURCE_RULES_SHA256
        or str(source_version.execution_contract_hash)
        != SOURCE_EXECUTION_CONTRACT_HASH
        or str(source_version.horizon_profile) != "short_1_5d"
        or str(source_backtest.strategy_version_id) != SOURCE_VERSION_ID
        or str(source_backtest.job_id) != SOURCE_JOB_ID
        or str(source_backtest.dataset) != SOURCE_DATASET
        or dict(source_backtest.periods_json or {}) != SOURCE_PERIODS
        or str(source_backtest.status) not in {"failed", "cancelled"}
        or str(source_job.status) != "cancelled"
        or str(source_job.error or "") != "Cancelled by operator"
        or int(source_job.attempts or 0) != int(source_job.max_attempts or -1)
        or int(source_job.attempts or 0) != 2
        or str(interruption.receipt_sha256)
        != SOURCE_INTERRUPTION_RECOVERY_RECEIPT_SHA256
    ):
        raise ValueError("source v17 cancellation or interruption receipt changed")
    return {
        "source_strategy_version_id": SOURCE_VERSION_ID,
        "source_backtest_id": SOURCE_BACKTEST_ID,
        "source_job_id": SOURCE_JOB_ID,
        "source_interruption_recovery_receipt_sha256": (
            SOURCE_INTERRUPTION_RECOVERY_RECEIPT_SHA256
        ),
        "source_interruption_receipt_authority": (
            SOURCE_INTERRUPTION_RECEIPT_AUTHORITY
        ),
    }


def require_source_cash_only_lockbox(connection: Any) -> dict[str, Any]:
    """Recompute the exact v17 cash-only facts without creating a v18 lockbox."""

    source = connection.execute(
        select(strategy_versions).where(strategy_versions.c.id == SOURCE_VERSION_ID)
    ).first()
    if source is None:
        raise ValueError("source v17 StrategyVersion is unavailable")
    config = dict(source.config_json or {})
    lockbox = validate_joint_lockbox(config.get(LOCKBOX_CONFIG_KEY))
    link = lockbox_member_link(config)
    selection = validate_unopened_history_selection(
        lockbox.get("unopened_history_selection")
    )
    members = [
        item
        for item in lockbox["members"]
        if item["recipe_id"] == "short_relative_strength"
    ]
    unavailable = list(lockbox.get("unavailable_horizons") or [])
    unavailable_pairs = {
        (str(item.get("recipe_id") or ""), str(item.get("horizon_profile") or ""))
        for item in unavailable
    }
    evidence_hashes = {
        str(item["horizon_profile"]): str(item["evidence_sha256"])
        for item in unavailable
    }
    if (
        lockbox.get("contract_version")
        != "transparent-baseline-available-horizons-lockbox-v3"
        or lockbox.get("dataset") != SOURCE_DATASET
        or lockbox.get("dataset_identity_sha256")
        != SOURCE_DATASET_IDENTITY_SHA256
        or lockbox.get("dataset_lineage_id") != SOURCE_DATASET_LINEAGE_ID
        or len(members) != 1
        or members[0]["test_start"] != SOURCE_PERIODS["start"]
        or members[0]["test_end"] != SOURCE_PERIODS["end"]
        or members[0]["historical_start"] != SOURCE_PERIODS["historical_start"]
        or members[0]["historical_end"] != SOURCE_PERIODS["historical_end"]
        or unavailable_pairs
        != {
            ("swing_trend", "swing_1_6m"),
            ("long_quality_value", "long_1_3y"),
        }
        or set(evidence_hashes) != {"swing_1_6m", "long_1_3y"}
        or not all(is_sha256(value) for value in evidence_hashes.values())
        or not isinstance(link, Mapping)
        or link.get("batch_sha256") != lockbox["batch_sha256"]
        or selection.get("current_recipe_version")
        != "qlib-rdagent-single-mainline-2026-08-31-v17"
    ):
        raise ValueError("source v17 cash-only lockbox changed or is incomplete")
    return {
        "source_lockbox_contract_version": lockbox["contract_version"],
        "source_lockbox_batch_sha256": lockbox["batch_sha256"],
        "source_lockbox_member_sha256": link["member_sha256"],
        "source_history_selection_sha256": selection["selection_sha256"],
        "source_unavailable_horizons_sha256": lockbox_canonical_sha256(unavailable),
        "source_unavailable_evidence_sha256s": evidence_hashes,
        "source_cash_only_scope": SOURCE_CASH_ONLY_SCOPE,
    }


def _validate_missing_family_artifacts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("incomplete family eligibility artifact inventory is missing")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "artifact_kind",
            "path",
            "status",
            "sha256",
            "bytes",
            "observed_at",
        }:
            raise ValueError("incomplete family artifact inventory entry is malformed")
        status = str(item.get("status") or "")
        digest = item.get("sha256")
        size = item.get("bytes")
        try:
            observed_at = datetime.fromisoformat(str(item["observed_at"]).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("incomplete family artifact observation time is invalid") from exc
        if (
            status not in {"missing", "partial"}
            or not str(item.get("path") or "").strip()
            or observed_at.tzinfo is None
            or (
                status == "missing"
                and (digest is not None or size is not None)
            )
            or (
                status == "partial"
                and (not is_sha256(digest) or not isinstance(size, int) or size <= 0)
            )
        ):
            raise ValueError("incomplete family artifact inventory is not fail-closed")
        normalized.append(dict(item))
    missing_kinds = {
        str(item["artifact_kind"])
        for item in normalized
        if item["status"] == "missing"
    }
    if not _REQUIRED_MISSING_FAMILY_ARTIFACTS.issubset(missing_kinds):
        raise ValueError("incomplete family artifact inventory omits required matrices")
    if normalized != sorted(
        normalized,
        key=lambda item: (str(item["artifact_kind"]), str(item["path"])),
    ):
        raise ValueError("incomplete family artifact inventory is not canonical")
    return normalized


def audit_incomplete_family_artifacts(
    *,
    data_root: Path,
    observed_at: datetime,
) -> list[dict[str, Any]]:
    """Hash the allowed partial baseline files and prove trial matrices absent."""

    if observed_at.tzinfo is None:
        raise ValueError("incomplete family artifact audit requires an aware time")
    governed_root = data_root.resolve()
    lexical_root = data_root / SOURCE_ATTEMPT_ARTIFACT_RELATIVE_PATH
    cursor = data_root
    for part in Path(SOURCE_ATTEMPT_ARTIFACT_RELATIVE_PATH).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("source attempt artifact path contains a symlink")
    root = lexical_root.resolve()
    try:
        root.relative_to(governed_root)
    except ValueError as exc:
        raise ValueError("source attempt artifact path escapes the governed data root") from exc
    if not root.is_dir():
        raise ValueError("source attempt artifact directory is unavailable")
    forbidden = {
        "result.json",
        "daily_returns.parquet",
        "artifact_manifest.json",
    }
    existing: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("source attempt artifact tree contains a symlink")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        lowered = relative.lower()
        if (
            relative in forbidden
            or lowered.startswith("robustness/")
            or "trial" in lowered and (
                "return" in lowered or "score" in lowered or "candidate" in lowered
            )
        ):
            raise ValueError(
                "historical family artifacts exist; conservative fallback is forbidden"
            )
        if relative != "manifest.json" and "baseline" not in lowered:
            raise ValueError(
                f"unclassified source attempt artifact blocks fallback: {relative}"
            )
        existing.append(candidate)
    if not existing:
        raise ValueError("source attempt has no hashable baseline partial artifacts")
    timestamp = observed_at.astimezone(UTC).isoformat()
    inventory = [
        {
            "artifact_kind": kind,
            "path": f"{SOURCE_ATTEMPT_ARTIFACT_RELATIVE_PATH}/{filename}",
            "status": "missing",
            "sha256": None,
            "bytes": None,
            "observed_at": timestamp,
        }
        for kind, filename in (
            ("trial_daily_returns_matrix", "trial_daily_returns_matrix.parquet"),
            ("trial_score_grid_matrix", "trial_score_grid_matrix.parquet"),
            (
                "trial_candidate_manifest_matrix",
                "trial_candidate_manifest_matrix.json",
            ),
        )
    ]
    inventory.extend(
        {
            "artifact_kind": f"baseline_partial:{path.relative_to(root).as_posix()}",
            "path": path.relative_to(governed_root).as_posix(),
            "status": "partial",
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "observed_at": timestamp,
        }
        for path in sorted(existing)
    )
    return sorted(
        inventory,
        key=lambda item: (str(item["artifact_kind"]), str(item["path"])),
    )


def build_incomplete_family_eligibility(
    connection: Any,
    *,
    strategy_version_id: str,
    hypothesis_group_evidence: Mapping[str, Any],
    missing_artifacts: Any,
    cutoff_at: datetime,
    created_by: str,
) -> dict[str, Any]:
    """Freeze the sole pre-run permission for conservative family statistics."""

    actor = created_by.strip()
    if len(actor) < 2 or cutoff_at.tzinfo is None:
        raise ValueError("incomplete family eligibility requires actor and aware cutoff")
    version = connection.execute(
        select(
            strategy_versions,
            strategies.c.economic_hypothesis_group,
        )
        .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
        .where(strategy_versions.c.id == strategy_version_id)
        .with_for_update()
    ).first()
    if (
        version is None
        or str(version.status) != "draft"
        or version.promotion_stage is not None
        or bool(version.is_legacy)
        or str(version.evidence_mode) != EVIDENCE_MODE_REPLAY
    ):
        raise ValueError("incomplete family eligibility requires an inert replay draft")
    require_replay_config(dict(version.config_json or {}))
    version_ids = list(hypothesis_group_evidence.get("strategy_version_ids") or [])
    trial_count_audit = hypothesis_group_evidence.get("trial_count_audit")
    try:
        trial_count = int(hypothesis_group_evidence["shared_experiment_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("incomplete family trial count is missing") from exc
    if (
        version_ids != sorted(set(str(item) for item in version_ids))
        or strategy_version_id not in version_ids
        or SOURCE_VERSION_ID not in version_ids
        or trial_count <= 1
        or not isinstance(trial_count_audit, Mapping)
        or str(hypothesis_group_evidence.get("economic_hypothesis_group") or "")
        != str(version.economic_hypothesis_group)
    ):
        raise ValueError("incomplete family eligibility does not bind the full family")
    inventory = _validate_missing_family_artifacts(missing_artifacts)
    audit = dict(trial_count_audit)
    audit_sha256 = canonical_sha256(audit)
    core = {
        "contract_version": INCOMPLETE_FAMILY_ELIGIBILITY_VERSION,
        "evidence_mode": EVIDENCE_MODE_REPLAY,
        "authority": INCOMPLETE_FAMILY_AUTHORITY,
        "source_strategy_version_id": SOURCE_VERSION_ID,
        "source_backtest_id": SOURCE_BACKTEST_ID,
        "source_job_id": SOURCE_JOB_ID,
        "strategy_version_id": strategy_version_id,
        "economic_hypothesis_group": str(version.economic_hypothesis_group),
        "eligible_strategy_version_ids": version_ids,
        "strategy_trial_count": trial_count,
        "trial_count_audit": audit,
        "trial_count_audit_sha256": audit_sha256,
        "missing_artifacts": inventory,
        "cutoff_at": cutoff_at.astimezone(UTC).isoformat(),
    }
    receipt_sha256 = canonical_sha256(core)
    return {
        **core,
        "receipt_sha256": receipt_sha256,
        "created_by": actor,
    }


def insert_incomplete_family_eligibility(
    connection: Any,
    value: Mapping[str, Any],
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    eligibility = dict(value)
    receipt = str(eligibility.get("receipt_sha256") or "")
    core = {
        key: item
        for key, item in eligibility.items()
        if key not in {"receipt_sha256", "created_by"}
    }
    if not is_sha256(receipt) or canonical_sha256(core) != receipt:
        raise ValueError("incomplete family eligibility receipt is invalid")
    existing = connection.execute(
        select(strategy_incomplete_family_eligibilities).where(
            strategy_incomplete_family_eligibilities.c.strategy_version_id
            == eligibility["strategy_version_id"]
        )
    ).first()
    if existing is not None:
        if str(existing.receipt_sha256) != receipt:
            raise ValueError("strategy already has another incomplete-family eligibility")
        return row_dict(existing)
    connection.execute(
        insert(strategy_incomplete_family_eligibilities).values(
            receipt_sha256=receipt,
            strategy_version_id=eligibility["strategy_version_id"],
            contract_version=INCOMPLETE_FAMILY_ELIGIBILITY_VERSION,
            evidence_mode=EVIDENCE_MODE_REPLAY,
            authority=INCOMPLETE_FAMILY_AUTHORITY,
            source_strategy_version_id=SOURCE_VERSION_ID,
            source_backtest_id=SOURCE_BACKTEST_ID,
            source_job_id=SOURCE_JOB_ID,
            economic_hypothesis_group=eligibility["economic_hypothesis_group"],
            eligible_strategy_version_ids_json=eligibility[
                "eligible_strategy_version_ids"
            ],
            strategy_trial_count=eligibility["strategy_trial_count"],
            trial_count_audit_json=eligibility["trial_count_audit"],
            trial_count_audit_sha256=eligibility["trial_count_audit_sha256"],
            missing_artifacts_json=eligibility["missing_artifacts"],
            cutoff_at=datetime.fromisoformat(eligibility["cutoff_at"]),
            qualification_json={
                key: item for key, item in eligibility.items() if key != "created_by"
            },
            created_by=eligibility["created_by"],
            created_at=created_at or datetime.now(UTC),
        )
    )
    return eligibility


def require_incomplete_family_eligibility(
    value: Mapping[str, Any],
    *,
    hypothesis_group_evidence: Mapping[str, Any],
    strategy_version_id: str,
) -> dict[str, Any]:
    """Validate a DB receipt projected into the execution manifest."""

    eligibility = dict(value)
    receipt = str(eligibility.pop("receipt_sha256", ""))
    _validate_missing_family_artifacts(eligibility.get("missing_artifacts"))
    version_ids = list(hypothesis_group_evidence.get("strategy_version_ids") or [])
    audit = hypothesis_group_evidence.get("trial_count_audit")
    try:
        trial_count = int(hypothesis_group_evidence["shared_experiment_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("incomplete family manifest trial count is invalid") from exc
    if (
        not is_sha256(receipt)
        or canonical_sha256(eligibility) != receipt
        or eligibility.get("contract_version")
        != INCOMPLETE_FAMILY_ELIGIBILITY_VERSION
        or eligibility.get("evidence_mode") != EVIDENCE_MODE_REPLAY
        or eligibility.get("authority") != INCOMPLETE_FAMILY_AUTHORITY
        or eligibility.get("source_strategy_version_id") != SOURCE_VERSION_ID
        or eligibility.get("source_backtest_id") != SOURCE_BACKTEST_ID
        or eligibility.get("source_job_id") != SOURCE_JOB_ID
        or eligibility.get("strategy_version_id") != strategy_version_id
        or eligibility.get("eligible_strategy_version_ids") != version_ids
        or int(eligibility.get("strategy_trial_count") or 0) != trial_count
        or not isinstance(audit, Mapping)
        or eligibility.get("trial_count_audit") != dict(audit)
        or eligibility.get("trial_count_audit_sha256")
        != canonical_sha256(dict(audit))
    ):
        raise ValueError("incomplete factor-family eligibility is absent or changed")
    return {**eligibility, "receipt_sha256": receipt}


def incomplete_family_eligibility_for_version(
    connection: Any,
    *,
    strategy_version_id: str,
    hypothesis_group_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    row = connection.execute(
        select(strategy_incomplete_family_eligibilities).where(
            strategy_incomplete_family_eligibilities.c.strategy_version_id
            == strategy_version_id
        )
    ).first()
    if row is None:
        raise ValueError("multi-trial replay has no frozen incomplete-family eligibility")
    projected = dict(row.qualification_json or {})
    require_incomplete_family_eligibility(
        projected,
        hypothesis_group_evidence=hypothesis_group_evidence,
        strategy_version_id=strategy_version_id,
    )
    if (
        str(row.receipt_sha256) != projected.get("receipt_sha256")
        or str(row.trial_count_audit_sha256)
        != projected.get("trial_count_audit_sha256")
    ):
        raise ValueError("incomplete-family eligibility database binding changed")
    return projected


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _resolve_terminal_evidence_path(data_root: Path, stored_path: str) -> Path:
    root = data_root.resolve()
    raw = Path(stored_path)
    normalized = stored_path.replace("\\", "/")
    if normalized == "/data":
        candidate = root
    elif normalized.startswith("/data/"):
        candidate = root.joinpath(*normalized.removeprefix("/data/").split("/"))
    elif raw.is_absolute():
        candidate = raw
    else:
        candidate = root / raw
    resolved = candidate.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError("terminal cash-only evidence escapes DATA_ROOT")
    if not resolved.is_file():
        raise ValueError("terminal cash-only evidence is not a regular file")
    return resolved


def _terminal_file_record(data_root: Path, path: Path) -> dict[str, Any]:
    root = data_root.resolve()
    resolved = path.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError("terminal cash-only artifact escapes DATA_ROOT")
    relative = resolved.relative_to(root).as_posix()
    return {
        "path": relative,
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _recompute_terminal_scenario(report_path: Path) -> dict[str, Any]:
    report = pd.read_parquet(report_path)
    required = {"return", "cost", "bench"}
    if report.empty or not required.issubset(report.columns):
        raise ValueError("terminal robustness daily report is incomplete")
    net = pd.to_numeric(report["return"], errors="coerce") - pd.to_numeric(
        report["cost"], errors="coerce"
    )
    benchmark = pd.to_numeric(report["bench"], errors="coerce")
    if (
        net.isna().any()
        or benchmark.isna().any()
        or not all(isfinite(float(value)) for value in net)
        or not all(isfinite(float(value)) for value in benchmark)
    ):
        raise ValueError("terminal robustness daily report contains non-finite returns")
    annualized_excess = float((net - benchmark).mean() * 252.0)
    nav = (1.0 + net).cumprod()
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    if not isfinite(annualized_excess) or not isfinite(max_drawdown):
        raise ValueError("terminal robustness recomputation is non-finite")
    return {
        "trading_days": int(len(report)),
        "annualized_excess_return": annualized_excess,
        "max_drawdown": max_drawdown,
    }


def validate_terminal_cash_only_receipt(
    value: Mapping[str, Any],
    *,
    expected_backtest_id: str = TERMINAL_CASH_ONLY_BACKTEST_ID,
    expected_job_id: str = TERMINAL_CASH_ONLY_JOB_ID,
    expected_version_id: str = TERMINAL_CASH_ONLY_VERSION_ID,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the exact terminal rejection receipt without granting authority."""

    receipt = dict(value)
    receipt_sha256 = str(receipt.pop("receipt_sha256", ""))
    gate = receipt.get("robustness_gate")
    scenarios = receipt.get("scenarios")
    files = receipt.get("files")
    if (
        not is_sha256(receipt_sha256)
        or canonical_sha256(receipt) != receipt_sha256
        or receipt.get("contract_version") != TERMINAL_CASH_ONLY_CONTRACT_VERSION
        or receipt.get("strategy_version_id") != expected_version_id
        or receipt.get("backtest_id") != expected_backtest_id
        or receipt.get("job_id") != expected_job_id
        or receipt.get("recipe_id") != TERMINAL_CASH_ONLY_RECIPE_ID
        or receipt.get("horizon_profile") != TERMINAL_CASH_ONLY_HORIZON
        or receipt.get("authority") != TERMINAL_CASH_ONLY_AUTHORITY
        or receipt.get("cash_only_scope") != SOURCE_CASH_ONLY_SCOPE
        or receipt.get("reason_code") != TERMINAL_CASH_ONLY_REASON_CODE
        or receipt.get("failure") != TERMINAL_CASH_ONLY_FAILURE
        or receipt.get("formal_result_complete") is not False
        or receipt.get("approval_eligible") is not False
        or receipt.get("rerun_allowed") is not False
        or receipt.get("recommendation_eligible") is not False
        or receipt.get("paper_eligible") is not False
        or receipt.get("dataset") != SOURCE_DATASET
        or receipt.get("dataset_identity_sha256") != SOURCE_DATASET_IDENTITY_SHA256
        or receipt.get("dataset_lineage_id") != SOURCE_DATASET_LINEAGE_ID
        or receipt.get("periods") != SOURCE_PERIODS
        or receipt.get("strategy_rules_sha256") != SOURCE_RULES_SHA256
        or receipt.get("execution_contract_hash") != SOURCE_EXECUTION_CONTRACT_HASH
        or receipt.get("recipe_version")
        != FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
        or receipt.get("runner_sha256")
        != FORWARD_ONLY_REHABILITATION_TARGET_RUNNER_SHA256
        or receipt.get("runtime_bundle_sha256")
        != FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256
        or not str(receipt.get("worker_runtime_image_digest") or "").startswith(
            "sha256:"
        )
        or not is_sha256(
            str(receipt.get("worker_runtime_image_digest") or "").removeprefix(
                "sha256:"
            )
        )
        or receipt.get("source_lockbox_contract_version")
        != "transparent-baseline-available-horizons-lockbox-v3"
        or not is_sha256(receipt.get("source_lockbox_batch_sha256"))
        or not is_sha256(receipt.get("source_lockbox_member_sha256"))
        or not is_sha256(receipt.get("source_history_selection_sha256"))
        or not is_sha256(receipt.get("source_unavailable_horizons_sha256"))
        or receipt.get("source_cash_only_scope") != SOURCE_CASH_ONLY_SCOPE
        or not isinstance(receipt.get("source_unavailable_evidence_sha256s"), Mapping)
        or set(receipt["source_unavailable_evidence_sha256s"])
        != {"swing_1_6m", "long_1_3y"}
        or not all(
            is_sha256(digest)
            for digest in receipt["source_unavailable_evidence_sha256s"].values()
        )
        or not isinstance(gate, Mapping)
        or dict(gate)
        != {
            "passed": 0,
            "total": 4,
            "pass_rate": 0.0,
            "min_pass_rate": 1.0,
            "passed_gate": False,
        }
        or not isinstance(scenarios, list)
        or not isinstance(files, Mapping)
    ):
        raise ValueError("terminal cash-only receipt is malformed or has changed")

    expected_names = list(TERMINAL_CASH_ONLY_CORE_SCENARIOS)
    observed_names: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ValueError("terminal cash-only scenario is malformed")
        name = str(scenario.get("name") or "")
        reported = scenario.get("reported_metrics")
        recomputed = scenario.get("recomputed_metrics")
        if not isinstance(reported, Mapping) or not isinstance(recomputed, Mapping):
            raise ValueError("terminal cash-only scenario metrics are missing")
        try:
            reported_excess = float(reported["annualized_excess_return"])
            reported_drawdown = float(reported["max_drawdown"])
            reported_days = int(reported["trading_days"])
            recomputed_excess = float(recomputed["annualized_excess_return"])
            recomputed_drawdown = float(recomputed["max_drawdown"])
            recomputed_days = int(recomputed["trading_days"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("terminal cash-only scenario metrics are invalid") from exc
        if (
            name not in TERMINAL_CASH_ONLY_CORE_SCENARIOS
            or scenario.get("passed") is not False
            or not all(
                isfinite(number)
                for number in (
                    reported_excess,
                    reported_drawdown,
                    recomputed_excess,
                    recomputed_drawdown,
                )
            )
            or reported_excess >= 0.0
            or recomputed_excess >= 0.0
            or reported_days != 252
            or recomputed_days != reported_days
            or not isclose(reported_excess, recomputed_excess, rel_tol=1e-9, abs_tol=1e-12)
            or not isclose(
                reported_drawdown,
                recomputed_drawdown,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("terminal robustness result is not an exact 0/4 rejection")
        observed_names.append(name)
    if observed_names != expected_names:
        raise ValueError("terminal cash-only receipt does not contain the four core scenarios")

    expected_file_keys = {"manifest", "job_log"}
    for scenario_name in TERMINAL_CASH_ONLY_CORE_SCENARIOS:
        expected_file_keys.update(
            {
                f"{scenario_name}:daily_report",
                f"{scenario_name}:fills",
                f"{scenario_name}:metrics",
            }
        )
    if set(files) != expected_file_keys:
        raise ValueError("terminal cash-only artifact inventory is incomplete")
    expected_paths = {
        "manifest": f"{TERMINAL_CASH_ONLY_ARTIFACT_RELATIVE_PATH}/manifest.json",
        "job_log": (
            "platform/logs/strategy-backtest-"
            f"{TERMINAL_CASH_ONLY_BACKTEST_ID}.log"
        ),
    }
    for scenario_name in TERMINAL_CASH_ONLY_CORE_SCENARIOS:
        scenario_root = (
            f"{TERMINAL_CASH_ONLY_ARTIFACT_RELATIVE_PATH}/robustness/{scenario_name}"
        )
        expected_paths.update(
            {
                f"{scenario_name}:daily_report": f"{scenario_root}/daily_report.parquet",
                f"{scenario_name}:fills": f"{scenario_root}/fills.parquet",
                f"{scenario_name}:metrics": f"{scenario_root}/metrics.json",
            }
        )
    observed_paths: set[str] = set()
    for file_key, entry in files.items():
        if not isinstance(entry, Mapping) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("terminal cash-only artifact entry is malformed")
        path_text = str(entry.get("path") or "")
        if (
            path_text != expected_paths[file_key]
            or Path(path_text).is_absolute()
            or ".." in Path(path_text).parts
            or path_text in observed_paths
            or not isinstance(entry.get("bytes"), int)
            or int(entry["bytes"]) <= 0
            or not is_sha256(entry.get("sha256"))
        ):
            raise ValueError("terminal cash-only artifact entry is unsafe")
        observed_paths.add(path_text)
        if artifact_root is not None:
            path = _resolve_terminal_evidence_path(artifact_root, path_text)
            if (
                int(path.stat().st_size) != int(entry["bytes"])
                or sha256_file(path) != str(entry["sha256"])
            ):
                raise ValueError("terminal cash-only artifact changed after registration")
    return {**receipt, "receipt_sha256": receipt_sha256}


def _terminal_rows(connection: Any, *, lock: bool = False) -> tuple[Any, Any, Any]:
    statements = (
        select(strategy_versions).where(
            strategy_versions.c.id == TERMINAL_CASH_ONLY_VERSION_ID
        ),
        select(backtest_runs).where(
            backtest_runs.c.id == TERMINAL_CASH_ONLY_BACKTEST_ID
        ),
        select(jobs).where(jobs.c.id == TERMINAL_CASH_ONLY_JOB_ID),
    )
    if lock:
        statements = tuple(statement.with_for_update() for statement in statements)
    version = connection.execute(statements[0]).first()
    backtest = connection.execute(statements[1]).first()
    job = connection.execute(statements[2]).first()
    if version is None or backtest is None or job is None:
        raise ValueError("terminal cash-only target rows are missing")
    return version, backtest, job


def build_terminal_cash_only_receipt(
    connection: Any,
    *,
    data_root: Path,
) -> dict[str, Any]:
    """Build a deterministic receipt from the exact failed public control."""

    version, backtest, job = _terminal_rows(connection)
    config = dict(version.config_json or {})
    bootstrap = dict(config.get(BOOTSTRAP_CONFIG_KEY) or {})
    if (
        str(version.status) != "draft"
        or version.promotion_stage is not None
        or version.approved_at is not None
        or version.approved_by is not None
        or bool(version.is_legacy)
        or str(version.evidence_mode) != EVIDENCE_MODE_REPLAY
        or str(version.horizon_profile) != TERMINAL_CASH_ONLY_HORIZON
        or str(version.strategy_rules_sha256) != SOURCE_RULES_SHA256
        or str(version.execution_contract_hash) != SOURCE_EXECUTION_CONTRACT_HASH
        or str(config.get("recipe_id") or "") != TERMINAL_CASH_ONLY_RECIPE_ID
        or str(config.get("recipe_version") or "")
        != FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
        or float(config.get("min_robustness_pass_rate") or 0.0) != 1.0
        or str(backtest.status) != "failed"
        or str(backtest.strategy_version_id) != TERMINAL_CASH_ONLY_VERSION_ID
        or str(backtest.job_id) != TERMINAL_CASH_ONLY_JOB_ID
        or str(backtest.dataset) != SOURCE_DATASET
        or dict(backtest.periods_json or {}) != SOURCE_PERIODS
        or backtest.metrics_json is not None
        or str(backtest.error or "") != TERMINAL_CASH_ONLY_FAILURE
        or str(backtest.evidence_mode) != EVIDENCE_MODE_REPLAY
        or str(job.kind) != "strategy_backtest"
        or str(job.status) != "failed"
        or str(job.error or "") != TERMINAL_CASH_ONLY_FAILURE
        or int(job.exit_code or 0) != 1
        or int(job.attempts or 0) != 1
        or int(job.max_attempts or 0) != 1
    ):
        raise ValueError("terminal cash-only target is not the exact inert failed replay")
    if (
        bootstrap.get("dataset") != SOURCE_DATASET
        or bootstrap.get("dataset_identity_sha256") != SOURCE_DATASET_IDENTITY_SHA256
        or bootstrap.get("dataset_lineage_id") != SOURCE_DATASET_LINEAGE_ID
        or bootstrap.get("recipe_id") != TERMINAL_CASH_ONLY_RECIPE_ID
        or bootstrap.get("recipe_version")
        != FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
        or bootstrap.get("formal_periods") != SOURCE_PERIODS
        or bootstrap.get("target_runner_sha256")
        != FORWARD_ONLY_REHABILITATION_TARGET_RUNNER_SHA256
        or bootstrap.get("target_runtime_bundle_sha256")
        != FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256
        or not str(bootstrap.get("target_worker_runtime_image_digest") or "").startswith(
            "sha256:"
        )
    ):
        raise ValueError("terminal cash-only runtime binding changed")

    root = _resolve_terminal_evidence_path(
        data_root,
        f"{str(backtest.artifact_path).rstrip('/')}/manifest.json",
    ).parent
    expected_relative_root = TERMINAL_CASH_ONLY_ARTIFACT_RELATIVE_PATH
    if root.relative_to(data_root.resolve()).as_posix() != expected_relative_root:
        raise ValueError("terminal cash-only artifact root changed")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("backtest_id") != TERMINAL_CASH_ONLY_BACKTEST_ID
        or manifest.get("strategy_version_id") != TERMINAL_CASH_ONLY_VERSION_ID
        or manifest.get("dataset") != SOURCE_DATASET
        or manifest.get("strategy_rules_sha256") != SOURCE_RULES_SHA256
        or manifest.get("periods") != {
            "start": SOURCE_PERIODS["start"],
            "end": SOURCE_PERIODS["end"],
        }
        or manifest.get("transparent_baseline_runner_sha256")
        != bootstrap["target_runner_sha256"]
        or manifest.get("transparent_baseline_runtime_bundle_sha256")
        != bootstrap["target_runtime_bundle_sha256"]
        or manifest.get("transparent_baseline_worker_runtime_image_digest")
        != bootstrap["target_worker_runtime_image_digest"]
        or manifest.get("authority") != REPLAY_AUTHORITY
        or manifest.get("final_oos_opened") is not True
        or manifest.get("sealed_final_oos") is not False
        or manifest.get("unseen_oos") is not False
    ):
        raise ValueError("terminal cash-only manifest identity changed")

    files: dict[str, dict[str, Any]] = {
        "manifest": _terminal_file_record(data_root, manifest_path),
        "job_log": _terminal_file_record(
            data_root,
            _resolve_terminal_evidence_path(data_root, str(job.log_path or "")),
        ),
    }
    scenarios: list[dict[str, Any]] = []
    max_drawdown = float(config.get("max_drawdown") or 0.0)
    for scenario_name in TERMINAL_CASH_ONLY_CORE_SCENARIOS:
        scenario_root = root / "robustness" / scenario_name
        report_path = scenario_root / "daily_report.parquet"
        fills_path = scenario_root / "fills.parquet"
        metrics_path = scenario_root / "metrics.json"
        reported_all = json.loads(metrics_path.read_text(encoding="utf-8"))
        recomputed = _recompute_terminal_scenario(report_path)
        reported = {
            "trading_days": int(reported_all.get("trading_days") or 0),
            "annualized_excess_return": float(
                reported_all.get("annualized_excess_return")
            ),
            "max_drawdown": float(reported_all.get("max_drawdown")),
        }
        passed = (
            recomputed["annualized_excess_return"] > 0.0
            and recomputed["max_drawdown"] >= -max_drawdown
        )
        scenarios.append(
            {
                "name": scenario_name,
                "passed": passed,
                "reported_metrics": reported,
                "recomputed_metrics": recomputed,
            }
        )
        files[f"{scenario_name}:daily_report"] = _terminal_file_record(
            data_root, report_path
        )
        files[f"{scenario_name}:fills"] = _terminal_file_record(data_root, fills_path)
        files[f"{scenario_name}:metrics"] = _terminal_file_record(
            data_root, metrics_path
        )
    if any(bool(item["passed"]) for item in scenarios):
        raise ValueError("terminal public control did not fail all core robustness scenarios")

    source_cash_only = require_source_cash_only_lockbox(connection)
    core = {
        "contract_version": TERMINAL_CASH_ONLY_CONTRACT_VERSION,
        "strategy_version_id": TERMINAL_CASH_ONLY_VERSION_ID,
        "backtest_id": TERMINAL_CASH_ONLY_BACKTEST_ID,
        "job_id": TERMINAL_CASH_ONLY_JOB_ID,
        "recipe_id": TERMINAL_CASH_ONLY_RECIPE_ID,
        "horizon_profile": TERMINAL_CASH_ONLY_HORIZON,
        "authority": TERMINAL_CASH_ONLY_AUTHORITY,
        "cash_only_scope": SOURCE_CASH_ONLY_SCOPE,
        "reason_code": TERMINAL_CASH_ONLY_REASON_CODE,
        "failure": TERMINAL_CASH_ONLY_FAILURE,
        "formal_result_complete": False,
        "approval_eligible": False,
        "paper_eligible": False,
        "recommendation_eligible": False,
        "rerun_allowed": False,
        "dataset": SOURCE_DATASET,
        "dataset_identity_sha256": SOURCE_DATASET_IDENTITY_SHA256,
        "dataset_lineage_id": SOURCE_DATASET_LINEAGE_ID,
        "periods": SOURCE_PERIODS,
        "strategy_rules_sha256": SOURCE_RULES_SHA256,
        "execution_contract_hash": SOURCE_EXECUTION_CONTRACT_HASH,
        "recipe_version": FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION,
        "runner_sha256": bootstrap["target_runner_sha256"],
        "runtime_bundle_sha256": bootstrap["target_runtime_bundle_sha256"],
        "worker_runtime_image_digest": bootstrap[
            "target_worker_runtime_image_digest"
        ],
        **source_cash_only,
        "robustness_gate": {
            "passed": 0,
            "total": 4,
            "pass_rate": 0.0,
            "min_pass_rate": 1.0,
            "passed_gate": False,
        },
        "scenarios": scenarios,
        "files": files,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    return validate_terminal_cash_only_receipt(receipt, artifact_root=data_root)


def require_terminal_cash_only_receipt(
    connection: Any,
    *,
    data_root: Path,
    verify_artifact_hashes: bool = True,
) -> dict[str, Any]:
    """Revalidate the registered receipt against database and source lockbox."""

    version, backtest, job = _terminal_rows(connection)
    config = dict(version.config_json or {})
    bootstrap = dict(config.get(BOOTSTRAP_CONFIG_KEY) or {})
    wrapper = backtest.metrics_json
    if not isinstance(wrapper, Mapping) or set(wrapper) != {
        TERMINAL_CASH_ONLY_WRAPPER_KEY
    }:
        raise ValueError("terminal cash-only receipt is not registered")
    raw_receipt = wrapper.get(TERMINAL_CASH_ONLY_WRAPPER_KEY)
    if not isinstance(raw_receipt, Mapping):
        raise ValueError("terminal cash-only receipt wrapper is malformed")
    receipt = validate_terminal_cash_only_receipt(
        raw_receipt,
        artifact_root=data_root if verify_artifact_hashes else None,
    )
    source_cash_only = require_source_cash_only_lockbox(connection)
    if (
        str(version.status) != "rejected"
        or version.promotion_stage is not None
        or version.approved_at is not None
        or version.approved_by is not None
        or bool(version.is_legacy)
        or str(version.evidence_mode) != EVIDENCE_MODE_REPLAY
        or str(version.horizon_profile) != TERMINAL_CASH_ONLY_HORIZON
        or str(version.strategy_rules_sha256) != SOURCE_RULES_SHA256
        or str(version.execution_contract_hash) != SOURCE_EXECUTION_CONTRACT_HASH
        or str(config.get("recipe_id") or "") != TERMINAL_CASH_ONLY_RECIPE_ID
        or str(config.get("recipe_version") or "")
        != FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
        or bootstrap.get("dataset") != SOURCE_DATASET
        or bootstrap.get("dataset_identity_sha256") != SOURCE_DATASET_IDENTITY_SHA256
        or bootstrap.get("dataset_lineage_id") != SOURCE_DATASET_LINEAGE_ID
        or bootstrap.get("target_runner_sha256") != receipt.get("runner_sha256")
        or bootstrap.get("target_runtime_bundle_sha256")
        != receipt.get("runtime_bundle_sha256")
        or bootstrap.get("target_worker_runtime_image_digest")
        != receipt.get("worker_runtime_image_digest")
        or str(backtest.status) != "failed"
        or str(backtest.job_id) != TERMINAL_CASH_ONLY_JOB_ID
        or str(backtest.strategy_version_id) != TERMINAL_CASH_ONLY_VERSION_ID
        or str(backtest.dataset) != SOURCE_DATASET
        or dict(backtest.periods_json or {}) != SOURCE_PERIODS
        or str(backtest.error or "") != TERMINAL_CASH_ONLY_FAILURE
        or str(job.status) != "failed"
        or str(job.error or "") != TERMINAL_CASH_ONLY_FAILURE
        or int(job.attempts or 0) != 1
        or int(job.max_attempts or 0) != 1
        or any(receipt.get(key) != item for key, item in source_cash_only.items())
    ):
        raise ValueError("terminal cash-only database or source-lockbox binding changed")
    return receipt


def register_terminal_cash_only_receipt(
    connection: Any,
    *,
    data_root: Path,
    actor: str,
) -> dict[str, Any]:
    """CAS-register the rejection while keeping formal and production states inert."""

    responsible = actor.strip()
    if len(responsible) < 2:
        raise ValueError("terminal cash-only registration requires a responsible actor")
    version, backtest, _job = _terminal_rows(connection, lock=True)
    if backtest.metrics_json is not None:
        return require_terminal_cash_only_receipt(
            connection,
            data_root=data_root,
            verify_artifact_hashes=True,
        )
    receipt = build_terminal_cash_only_receipt(connection, data_root=data_root)
    backtest_result = connection.execute(
        update(backtest_runs)
        .where(
            backtest_runs.c.id == TERMINAL_CASH_ONLY_BACKTEST_ID,
            backtest_runs.c.strategy_version_id == TERMINAL_CASH_ONLY_VERSION_ID,
            backtest_runs.c.job_id == TERMINAL_CASH_ONLY_JOB_ID,
            backtest_runs.c.status == "failed",
            backtest_runs.c.metrics_json.is_(None),
        )
        .values(metrics_json={TERMINAL_CASH_ONLY_WRAPPER_KEY: receipt})
    )
    version_result = connection.execute(
        update(strategy_versions)
        .where(
            strategy_versions.c.id == TERMINAL_CASH_ONLY_VERSION_ID,
            strategy_versions.c.status == "draft",
            strategy_versions.c.promotion_stage.is_(None),
            strategy_versions.c.approved_at.is_(None),
            strategy_versions.c.approved_by.is_(None),
        )
        .values(status="rejected")
    )
    if backtest_result.rowcount != 1 or version_result.rowcount != 1:
        raise ValueError("terminal cash-only compare-and-set lost its exact target")
    connection.execute(
        insert(strategy_events).values(
            strategy_id=str(version.strategy_id),
            strategy_version_id=TERMINAL_CASH_ONLY_VERSION_ID,
            event_type=TERMINAL_CASH_ONLY_AUDIT_ACTION,
            actor=responsible,
            payload_json={
                "receipt_sha256": receipt["receipt_sha256"],
                "backtest_id": TERMINAL_CASH_ONLY_BACKTEST_ID,
                "job_id": TERMINAL_CASH_ONLY_JOB_ID,
                "reason_code": TERMINAL_CASH_ONLY_REASON_CODE,
            },
            created_at=datetime.now(UTC),
        )
    )
    return receipt


def require_replay_markers(value: Mapping[str, Any], *, label: str) -> None:
    failures = [
        key for key, expected in REPLAY_MARKERS.items() if value.get(key) != expected
    ]
    if failures:
        raise ValueError(
            f"{label} misstates consumed historical replay authority: "
            + ", ".join(failures)
        )


def require_forward_criteria(value: Mapping[str, Any]) -> None:
    thresholds = value.get("thresholds")
    if (
        value.get("contract_version") != "strategy-forward-gate-v2"
        or value.get("horizon_profile") != "short_1_5d"
        or not is_sha256(value.get("horizon_contract_sha256"))
        or not isinstance(thresholds, Mapping)
        or int(thresholds.get("min_forward_calendar_days") or 0) < 365
        or int(thresholds.get("min_forward_trading_days") or 0) < 252
        or int(thresholds.get("min_decision_batches") or 0) < 60
        or int(thresholds.get("min_closed_round_trips") or 0) < 30
        or float(thresholds.get("min_data_completeness") or 0.0) < 0.95
        or float(thresholds.get("min_reconciliation_rate") or 0.0) < 1.0
    ):
        raise ValueError("forward-only rehabilitation criteria are weaker or malformed")


def require_replay_config(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config.get("forward_only_rehabilitation")
    bootstrap = config.get(BOOTSTRAP_CONFIG_KEY)
    expected = {
        "contract_version": CONTRACT_VERSION,
        "source_strategy_version_id": SOURCE_VERSION_ID,
        "source_backtest_id": SOURCE_BACKTEST_ID,
        "source_job_id": SOURCE_JOB_ID,
        "source_interruption_recovery_receipt_sha256": (
            SOURCE_INTERRUPTION_RECOVERY_RECEIPT_SHA256
        ),
        "source_interruption_receipt_authority": (
            SOURCE_INTERRUPTION_RECEIPT_AUTHORITY
        ),
        "dataset": SOURCE_DATASET,
        "dataset_identity_sha256": SOURCE_DATASET_IDENTITY_SHA256,
        "dataset_lineage_id": SOURCE_DATASET_LINEAGE_ID,
        "strategy_rules_sha256": SOURCE_RULES_SHA256,
        "execution_contract_hash": SOURCE_EXECUTION_CONTRACT_HASH,
        "replay_periods": SOURCE_PERIODS,
        "source_attempt_artifact_relative_path": (
            SOURCE_ATTEMPT_ARTIFACT_RELATIVE_PATH
        ),
    }
    if (
        config.get("evidence_mode") != EVIDENCE_MODE_REPLAY
        or config.get("recipe_id") != "short_relative_strength"
        or config.get("recipe_version")
        != FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
        or config.get("factor_source_mode") != "qlib_baseline"
        or config.get("horizon_profile") != "short_1_5d"
        or config.get("strategy_rules_sha256") != SOURCE_RULES_SHA256
        or config.get("execution_contract_hash") != SOURCE_EXECUTION_CONTRACT_HASH
        or LOCKBOX_CONFIG_KEY in config
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
        != FORWARD_ONLY_REHABILITATION_TARGET_RUNNER_SHA256
        or bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
        != FORWARD_ONLY_REHABILITATION_TARGET_RUNTIME_BUNDLE_SHA256
        or not str(
            bootstrap.get(TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD) or ""
        ).startswith("sha256:")
        or not is_sha256(
            str(
                bootstrap.get(TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD)
                or ""
            ).removeprefix("sha256:")
        )
        or not isinstance(binding, Mapping)
        or any(binding.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(
            "consumed historical replay is not the exact forward-only rehabilitation binding"
        )
    return dict(binding)


def require_consumed_vintage(
    connection: Any,
    *,
    version_config: Mapping[str, Any],
    dataset_identity_sha256: str,
    dataset_lineage_id: str,
) -> Any:
    """Return the exact already-consumed source vintage; never create a new one."""

    binding = require_replay_config(version_config)
    if (
        dataset_identity_sha256 != SOURCE_DATASET_IDENTITY_SHA256
        or dataset_lineage_id != SOURCE_DATASET_LINEAGE_ID
    ):
        raise ValueError("historical replay dataset identity or lineage changed")
    configured_vintage_id = str(binding.get("consumed_oos_vintage_id") or "")
    statement = select(oos_vintages).where(
        oos_vintages.c.test_start == date.fromisoformat(SOURCE_PERIODS["start"]),
        oos_vintages.c.test_end == date.fromisoformat(SOURCE_PERIODS["end"]),
        oos_vintages.c.dataset_identity == SOURCE_DATASET_IDENTITY_SHA256,
        oos_vintages.c.dataset_lineage_id == SOURCE_DATASET_LINEAGE_ID,
        oos_vintages.c.consumed_at.is_not(None),
    )
    if configured_vintage_id:
        statement = statement.where(oos_vintages.c.id == configured_vintage_id)
    rows = connection.execute(statement.with_for_update()).all()
    matches = []
    for row in rows:
        members = dict(row.sealed_candidate_set_json or {})
        if str(members.get("strategy_version_id") or "") == SOURCE_VERSION_ID:
            matches.append(row)
            continue
        link = members.get("transparent_baseline_lockbox")
        if isinstance(link, Mapping) and SOURCE_VERSION_ID in {
            str(item) for item in link.get("strategy_version_ids", [])
        }:
            matches.append(row)
    # Older transparent baseline lockboxes do not always project the version id
    # at top level. The exact source cancellation plus one unique consumed row
    # for the frozen dataset/lineage/periods is the only safe fallback.
    if not matches and len(rows) == 1:
        require_source_cancellation(connection)
        matches = rows
    if len(matches) != 1:
        raise ValueError(
            "forward-only rehabilitation requires one exact consumed source OOS vintage"
        )
    return matches[0]


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"historical replay {label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"historical replay {label} must be a JSON object")
    return value


def _artifact_hashes(
    backtest: Mapping[str, Any], *, metrics: Mapping[str, Any]
) -> dict[str, str]:
    root = Path(str(backtest.get("artifact_path") or "")).resolve()
    paths = {
        "replay_manifest_sha256": root / "manifest.json",
        "replay_result_sha256": root / "result.json",
        "replay_artifact_manifest_sha256": root / "artifact_manifest.json",
        "replay_daily_returns_sha256": root / "daily_returns.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(
            "historical replay terminal artifacts are incomplete: " + ", ".join(missing)
        )
    manifest = _read_json_object(paths["replay_manifest_sha256"], label="manifest")
    result = _read_json_object(paths["replay_result_sha256"], label="result")
    if result.get("status") != "ok" or not isinstance(result.get("metrics"), Mapping):
        raise ValueError("historical replay result is not a successful governed result")
    if canonical_sha256(result["metrics"]) != canonical_sha256(dict(metrics)):
        raise ValueError("historical replay result metrics differ from the database")
    provenance = metrics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("historical replay provenance is missing")
    for label, value in (
        ("manifest", manifest),
        ("result", result),
        ("metrics", metrics),
        ("provenance", provenance),
    ):
        require_replay_markers(value, label=f"historical replay {label}")
        if value.get("final_oos_opened") is not True:
            raise ValueError(
                f"historical replay {label} falsely claims that the opened window stayed closed"
            )
    if (
        result.get("evaluation_mode") != EVIDENCE_MODE_REPLAY
        or manifest.get("evaluation_mode") != EVIDENCE_MODE_REPLAY
        or metrics.get("evaluation_mode") != EVIDENCE_MODE_REPLAY
        or provenance.get("evaluation_mode") != EVIDENCE_MODE_REPLAY
    ):
        raise ValueError("historical replay evaluation mode is inconsistent")
    expected_artifact_sha256 = str(
        provenance.get("artifact_manifest_sha256") or ""
    )
    artifact_manifest = validate_backtest_artifact_manifest(
        root,
        expected_sha256=expected_artifact_sha256,
    )
    if (
        provenance.get("artifact_manifest_version")
        != STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION
        or int(provenance.get("artifact_manifest_file_count") or -1)
        != len(artifact_manifest["files"])
        or sha256_file(paths["replay_artifact_manifest_sha256"])
        != expected_artifact_sha256
    ):
        raise ValueError("historical replay artifact manifest provenance is inconsistent")
    daily_entry = next(
        (
            item
            for item in artifact_manifest["files"]
            if item.get("path") == "daily_returns.parquet"
        ),
        None,
    )
    if (
        not isinstance(daily_entry, Mapping)
        or daily_entry.get("sha256") != sha256_file(paths["replay_daily_returns_sha256"])
    ):
        raise ValueError("historical replay daily returns are not manifest-bound")
    execution_manifest_sha256 = str(
        provenance.get("execution_manifest_sha256") or ""
    )
    if (
        not is_sha256(execution_manifest_sha256)
        or execution_manifest_sha256 != sha256_file(paths["replay_manifest_sha256"])
    ):
        raise ValueError("historical replay execution manifest provenance is inconsistent")
    return {key: sha256_file(path) for key, path in paths.items()}


def require_qualification(
    connection: Any,
    *,
    version: Any,
    backtest: Any | None = None,
) -> dict[str, Any]:
    row = connection.execute(
        select(strategy_forward_only_rehabilitations).where(
            strategy_forward_only_rehabilitations.c.strategy_version_id == version.id
        )
    ).first()
    if row is None:
        raise ValueError(
            "consumed historical replay has no forward-only rehabilitation qualification"
        )
    criteria = dict(row.forward_criteria_json or {})
    version_config = dict(version.config_json or {})
    require_replay_config(version_config)
    if backtest is None:
        backtest = connection.execute(
            select(backtest_runs).where(backtest_runs.c.id == row.backtest_id)
        ).first()
    qualification = dict(row.qualification_json or {})
    receipt = str(qualification.pop("receipt_sha256", ""))
    try:
        require_replay_markers(
            qualification,
            label="forward-only rehabilitation qualification",
        )
    except ValueError as exc:
        raise ValueError(
            "forward-only rehabilitation qualification has drifted"
        ) from exc
    bootstrap = dict(version_config.get(BOOTSTRAP_CONFIG_KEY) or {})
    if (
        str(row.receipt_sha256) != receipt
        or canonical_sha256(qualification) != receipt
        or str(row.contract_version) != CONTRACT_VERSION
        or str(row.evidence_mode) != EVIDENCE_MODE_REPLAY
        or str(row.authority) != REPLAY_AUTHORITY
        or str(row.source_strategy_version_id) != SOURCE_VERSION_ID
        or str(row.source_backtest_id) != SOURCE_BACKTEST_ID
        or str(row.source_job_id) != SOURCE_JOB_ID
        or str(row.source_interruption_recovery_receipt_sha256)
        != SOURCE_INTERRUPTION_RECOVERY_RECEIPT_SHA256
        or str(row.source_interruption_receipt_authority)
        != SOURCE_INTERRUPTION_RECEIPT_AUTHORITY
        or str(row.source_lockbox_contract_version)
        != str(qualification.get("source_lockbox_contract_version") or "")
        or str(row.source_lockbox_batch_sha256)
        != str(qualification.get("source_lockbox_batch_sha256") or "")
        or str(row.source_lockbox_member_sha256)
        != str(qualification.get("source_lockbox_member_sha256") or "")
        or str(row.source_history_selection_sha256)
        != str(qualification.get("source_history_selection_sha256") or "")
        or str(row.source_unavailable_horizons_sha256)
        != str(qualification.get("source_unavailable_horizons_sha256") or "")
        or dict(row.source_unavailable_evidence_sha256s_json or {})
        != dict(qualification.get("source_unavailable_evidence_sha256s") or {})
        or str(row.source_cash_only_scope) != SOURCE_CASH_ONLY_SCOPE
        or str(row.dataset) != SOURCE_DATASET
        or str(row.dataset_identity_sha256) != SOURCE_DATASET_IDENTITY_SHA256
        or str(row.dataset_lineage_id) != SOURCE_DATASET_LINEAGE_ID
        or qualification.get("final_oos_opened") is not True
        or qualification.get("capital_eligible") is not False
        or dict(row.replay_periods_json or {}) != SOURCE_PERIODS
        or qualification.get("replay_periods") != SOURCE_PERIODS
        or str(row.runner_sha256)
        != str(bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD) or "")
        or str(row.runtime_bundle_sha256)
        != str(bootstrap.get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD) or "")
        or str(row.worker_runtime_image_digest)
        != str(bootstrap.get(TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD) or "")
        or int(row.strategy_trial_count or 0)
        != int(qualification.get("strategy_trial_count") or 0)
        or str(row.trial_count_audit_sha256)
        != str(qualification.get("trial_count_audit_sha256") or "")
        or str(row.incomplete_family_eligibility_sha256)
        != str(qualification.get("incomplete_family_eligibility_sha256") or "")
        or dict(row.forward_criteria_json or {})
        != dict(qualification.get("forward_criteria") or {})
        or str(row.forward_criteria_sha256)
        != str(qualification.get("forward_criteria_sha256") or "")
        or str(row.incomplete_family_eligibility_sha256)
        != str(
            connection.scalar(
                select(
                    strategy_incomplete_family_eligibilities.c.receipt_sha256
                ).where(
                    strategy_incomplete_family_eligibilities.c.strategy_version_id
                    == version.id
                )
            )
            or ""
        )
        or str(row.strategy_rules_sha256) != str(version.strategy_rules_sha256)
        or str(row.execution_contract_hash) != str(version.execution_contract_hash)
        or str(row.horizon_profile) != str(version.horizon_profile)
        or canonical_sha256(criteria) != str(row.forward_criteria_sha256)
    ):
        raise ValueError("forward-only rehabilitation qualification has drifted")
    require_source_cancellation(connection)
    source_cash_only = require_source_cash_only_lockbox(connection)
    if any(qualification.get(key) != value for key, value in source_cash_only.items()):
        raise ValueError("source cash-only lockbox changed after qualification")
    if backtest is None or (
        str(row.backtest_id) != str(backtest.id)
        or str(backtest.evidence_mode) != EVIDENCE_MODE_REPLAY
        or str(backtest.status) != "succeeded"
        or str(backtest.strategy_version_id) != str(version.id)
    ):
        raise ValueError("forward-only qualification does not bind the approval replay")
    backtest_value = row_dict(backtest)
    backtest_value["metrics"] = dict(backtest.metrics_json or {})
    current_hashes = _artifact_hashes(
        backtest_value,
        metrics=backtest_value["metrics"],
    )
    for field, observed in current_hashes.items():
        if (
            str(getattr(row, field)) != observed
            or str(qualification.get(field) or "") != observed
        ):
            raise ValueError("forward-only rehabilitation artifacts changed after qualification")
    require_forward_criteria(criteria)
    return row_dict(row)
