"""Narrow admission for a consumed-history transparent-baseline replay.

This module does not introduce another promotion or simulation lifecycle.  It
only proves that one exact historical replay is descriptive evidence and
freezes the stricter criteria for the existing forward paper stage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import insert, select

from quant_data.database import (
    audit_events,
    backtest_runs,
    formal_backtest_interruption_recoveries,
    jobs,
    oos_vintages,
    row_dict,
    strategies,
    strategy_forward_only_rehabilitations,
    strategy_incomplete_family_eligibilities,
    strategy_versions,
)
from quant_platform.strategy_artifact_manifest import (
    STRATEGY_BACKTEST_ARTIFACT_MANIFEST_VERSION,
    validate_backtest_artifact_manifest,
)
from quant_platform.transparent_baseline_lockbox import (
    BOOTSTRAP_CONFIG_KEY,
    LOCKBOX_CONFIG_KEY,
    lockbox_member_link,
    validate_joint_lockbox,
    validate_unopened_history_selection,
)
from quant_platform.transparent_baseline_lockbox import (
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
REHABILITATION_AUDIT_ACTION = "strategy.forward_only_rehabilitation_admitted"
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


def require_replay_markers(value: Mapping[str, Any], *, label: str) -> None:
    failures = [
        key for key, expected in REPLAY_MARKERS.items() if value.get(key) != expected
    ]
    if failures:
        raise ValueError(
            f"{label} misstates consumed historical replay authority: "
            + ", ".join(failures)
        )


def rehabilitation_forward_thresholds(base: Mapping[str, Any]) -> dict[str, Any]:
    """Strengthen, never replace, the existing horizon-specific paper gate."""

    thresholds = dict(base)
    thresholds["min_forward_calendar_days"] = max(
        365, int(thresholds.get("min_forward_calendar_days") or 0)
    )
    thresholds["min_forward_trading_days"] = max(
        252, int(thresholds.get("min_forward_trading_days") or 0)
    )
    return thresholds


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


def build_qualification(
    connection: Any,
    *,
    version: Mapping[str, Any],
    backtest: Mapping[str, Any],
    forward_criteria: Mapping[str, Any],
    created_by: str,
) -> dict[str, Any]:
    """Rebuild the exact forward-only receipt from governed rows and artifacts."""

    actor = created_by.strip()
    if len(actor) < 2:
        raise ValueError("forward-only rehabilitation requires a responsible actor")
    config = version.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("strategy replay config is missing")
    binding = require_replay_config(config)
    if (
        version.get("evidence_mode") != EVIDENCE_MODE_REPLAY
        or version.get("status") != "draft"
        or version.get("promotion_stage") is not None
        or bool(version.get("is_legacy"))
        or backtest.get("evidence_mode") != EVIDENCE_MODE_REPLAY
        or backtest.get("status") != "succeeded"
        or bool(backtest.get("is_legacy"))
        or str(backtest.get("strategy_version_id")) != str(version.get("id"))
        or str(backtest.get("dataset")) != SOURCE_DATASET
        or dict(backtest.get("periods") or {}) != SOURCE_PERIODS
    ):
        raise ValueError("target StrategyVersion/backtest is not an inert succeeded replay")
    metrics = backtest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("historical replay metrics are missing")
    require_replay_markers(metrics, label="backtest metrics")
    formal = metrics.get("formal_validation")
    multiple = formal.get("multiple_testing") if isinstance(formal, Mapping) else None
    deflated = metrics.get("deflated_sharpe")
    try:
        trial_count = int(metrics.get("strategy_trial_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("historical replay has no real strategy trial count") from exc
    audit_sha256 = str((multiple or {}).get("trial_count_audit_sha256") or "")
    eligibility_sha256 = str(
        (multiple or {}).get("eligibility_receipt_sha256") or ""
    )
    eligibility_row = connection.execute(
        select(strategy_incomplete_family_eligibilities).where(
            strategy_incomplete_family_eligibilities.c.strategy_version_id
            == str(version["id"])
        )
    ).first()
    if (
        trial_count <= 1
        or int((multiple or {}).get("trial_count") or 0) != trial_count
        or int((deflated or {}).get("trials") or 0) != trial_count
        or not is_sha256(audit_sha256)
        or not is_sha256(eligibility_sha256)
        or eligibility_row is None
        or str(eligibility_row.receipt_sha256) != eligibility_sha256
    ):
        raise ValueError(
            "historical replay must use the complete real trial count and conservative gate"
        )
    require_forward_criteria(forward_criteria)
    vintage = require_consumed_vintage(
        connection,
        version_config=config,
        dataset_identity_sha256=str(binding["dataset_identity_sha256"]),
        dataset_lineage_id=str(binding["dataset_lineage_id"]),
    )
    source_cancellation = require_source_cancellation(connection)
    source_cash_only = require_source_cash_only_lockbox(connection)
    bootstrap = config.get("transparent_baseline_bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("historical replay runtime binding is missing")
    runner_sha256 = str(bootstrap.get("target_runner_sha256") or "")
    runtime_bundle_sha256 = str(
        bootstrap.get("target_runtime_bundle_sha256") or ""
    )
    image_digest = str(bootstrap.get("target_worker_runtime_image_digest") or "")
    if (
        not is_sha256(runner_sha256)
        or not is_sha256(runtime_bundle_sha256)
        or not image_digest.startswith("sha256:")
        or not is_sha256(image_digest.removeprefix("sha256:"))
    ):
        raise ValueError("historical replay runtime identity is incomplete")
    artifact_hashes = _artifact_hashes(backtest, metrics=metrics)
    criteria = dict(forward_criteria)
    criteria_sha256 = canonical_sha256(criteria)
    core = {
        "contract_version": CONTRACT_VERSION,
        **REPLAY_MARKERS,
        "final_oos_opened": True,
        "capital_eligible": False,
        "source_strategy_version_id": SOURCE_VERSION_ID,
        "source_backtest_id": SOURCE_BACKTEST_ID,
        "source_job_id": SOURCE_JOB_ID,
        "source_interruption_recovery_receipt_sha256": source_cancellation[
            "source_interruption_recovery_receipt_sha256"
        ],
        "source_interruption_receipt_authority": source_cancellation[
            "source_interruption_receipt_authority"
        ],
        **source_cash_only,
        "strategy_version_id": str(version["id"]),
        "backtest_id": str(backtest["id"]),
        "consumed_oos_vintage_id": str(vintage.id),
        "recipe_id": "short_relative_strength",
        "horizon_profile": "short_1_5d",
        "dataset": SOURCE_DATASET,
        "dataset_identity_sha256": SOURCE_DATASET_IDENTITY_SHA256,
        "dataset_lineage_id": SOURCE_DATASET_LINEAGE_ID,
        "strategy_rules_sha256": SOURCE_RULES_SHA256,
        "execution_contract_hash": SOURCE_EXECUTION_CONTRACT_HASH,
        "runner_sha256": runner_sha256,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "worker_runtime_image_digest": image_digest,
        "replay_periods": SOURCE_PERIODS,
        **artifact_hashes,
        "strategy_trial_count": trial_count,
        "trial_count_audit_sha256": audit_sha256,
        "incomplete_family_eligibility_sha256": eligibility_sha256,
        "forward_criteria": criteria,
        "forward_criteria_sha256": criteria_sha256,
    }
    receipt_sha256 = canonical_sha256(core)
    return {**core, "receipt_sha256": receipt_sha256, "created_by": actor}


def insert_qualification(
    connection: Any,
    qualification: Mapping[str, Any],
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Append the audit event and qualification receipt in the caller transaction."""

    value = dict(qualification)
    receipt_sha256 = str(value.get("receipt_sha256") or "")
    if not is_sha256(receipt_sha256):
        raise ValueError("forward-only rehabilitation receipt SHA-256 is invalid")
    core = {key: item for key, item in value.items() if key not in {"receipt_sha256", "created_by"}}
    if canonical_sha256(core) != receipt_sha256:
        raise ValueError("forward-only rehabilitation receipt is not canonical")
    existing = connection.execute(
        select(strategy_forward_only_rehabilitations).where(
            strategy_forward_only_rehabilitations.c.strategy_version_id
            == value["strategy_version_id"]
        )
    ).first()
    if existing is not None:
        if str(existing.receipt_sha256) != receipt_sha256:
            raise ValueError("StrategyVersion is bound to another rehabilitation receipt")
        return row_dict(existing)
    now = created_at or datetime.now(UTC)
    audit_id = connection.scalar(
        insert(audit_events)
        .values(
            user_id=None,
            username=str(value["created_by"]),
            action=REHABILITATION_AUDIT_ACTION,
            method="INTERNAL",
            path="/internal/strategies/forward-only-rehabilitation",
            status_code=201,
            ip_hash=None,
            user_agent="quantlab-forward-only-rehabilitation",
            details_json={
                "receipt_sha256": receipt_sha256,
                "strategy_version_id": value["strategy_version_id"],
                "backtest_id": value["backtest_id"],
                "consumed_oos_vintage_id": value["consumed_oos_vintage_id"],
                "authority": REPLAY_AUTHORITY,
            },
            created_at=now,
        )
        .returning(audit_events.c.id)
    )
    connection.execute(
        insert(strategy_forward_only_rehabilitations).values(
            receipt_sha256=receipt_sha256,
            source_audit_event_id=audit_id,
            source_strategy_version_id=value["source_strategy_version_id"],
            source_backtest_id=value["source_backtest_id"],
            source_job_id=value["source_job_id"],
            source_interruption_recovery_receipt_sha256=value[
                "source_interruption_recovery_receipt_sha256"
            ],
            source_interruption_receipt_authority=value[
                "source_interruption_receipt_authority"
            ],
            source_lockbox_contract_version=value[
                "source_lockbox_contract_version"
            ],
            source_lockbox_batch_sha256=value["source_lockbox_batch_sha256"],
            source_lockbox_member_sha256=value["source_lockbox_member_sha256"],
            source_history_selection_sha256=value[
                "source_history_selection_sha256"
            ],
            source_unavailable_horizons_sha256=value[
                "source_unavailable_horizons_sha256"
            ],
            source_unavailable_evidence_sha256s_json=value[
                "source_unavailable_evidence_sha256s"
            ],
            source_cash_only_scope=value["source_cash_only_scope"],
            strategy_version_id=value["strategy_version_id"],
            backtest_id=value["backtest_id"],
            consumed_oos_vintage_id=value["consumed_oos_vintage_id"],
            contract_version=CONTRACT_VERSION,
            evidence_mode=EVIDENCE_MODE_REPLAY,
            authority=REPLAY_AUTHORITY,
            recipe_id=value["recipe_id"],
            horizon_profile=value["horizon_profile"],
            dataset=value["dataset"],
            dataset_identity_sha256=value["dataset_identity_sha256"],
            dataset_lineage_id=value["dataset_lineage_id"],
            strategy_rules_sha256=value["strategy_rules_sha256"],
            execution_contract_hash=value["execution_contract_hash"],
            runner_sha256=value["runner_sha256"],
            runtime_bundle_sha256=value["runtime_bundle_sha256"],
            worker_runtime_image_digest=value["worker_runtime_image_digest"],
            replay_periods_json=value["replay_periods"],
            replay_manifest_sha256=value["replay_manifest_sha256"],
            replay_result_sha256=value["replay_result_sha256"],
            replay_artifact_manifest_sha256=value[
                "replay_artifact_manifest_sha256"
            ],
            replay_daily_returns_sha256=value["replay_daily_returns_sha256"],
            strategy_trial_count=value["strategy_trial_count"],
            trial_count_audit_sha256=value["trial_count_audit_sha256"],
            incomplete_family_eligibility_sha256=value[
                "incomplete_family_eligibility_sha256"
            ],
            forward_criteria_json=value["forward_criteria"],
            forward_criteria_sha256=value["forward_criteria_sha256"],
            qualification_json={
                key: item for key, item in value.items() if key != "created_by"
            },
            created_by=value["created_by"],
            created_at=now,
        )
    )
    return value


def qualification_for_version(connection: Any, version_id: str) -> dict[str, Any] | None:
    row = connection.execute(
        select(strategy_forward_only_rehabilitations).where(
            strategy_forward_only_rehabilitations.c.strategy_version_id == version_id
        )
    ).first()
    return row_dict(row) if row is not None else None


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
