"""Registration helper for the allowlisted v7 baseline pre-result repair."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import insert, select

from quant_data.database import (
    audit_events,
    backtest_runs,
    jobs,
    open_database,
    strategy_versions,
)
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_platform.eligibility import ELIGIBILITY_CONTRACT_VERSION
from quant_platform.transparent_baseline_lockbox import (
    CANONICAL_LF_PACKAGING_CONTRACT_VERSION,
    CANONICAL_LF_PACKAGING_ERROR,
    CANONICAL_LF_PACKAGING_REASON,
    CANONICAL_LF_PACKAGING_REPAIR_GENERATION,
    CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS,
    CANONICAL_LF_PACKAGING_SOURCE_COMMIT,
    CANONICAL_LF_PACKAGING_SOURCE_EXPECTED_RUNNER_SHA256,
    CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256,
    CANONICAL_LF_PACKAGING_SOURCE_RECIPE_VERSION,
    CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
    CANONICAL_LF_PACKAGING_TARGET_RUNNER_SHA256,
    OPTIMIZER_APPLICABILITY_ERROR,
    OPTIMIZER_APPLICABILITY_REASON,
    OPTIMIZER_APPLICABILITY_REPAIR_GENERATION,
    OPTIMIZER_APPLICABILITY_SOURCE_BACKTEST_IDS,
    OPTIMIZER_APPLICABILITY_SOURCE_COMMIT,
    OPTIMIZER_APPLICABILITY_SOURCE_RECIPE_VERSION,
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
    PRE_RESULT_REPAIR_ACTION,
    PRE_RESULT_REPAIR_CONTRACT_VERSION_V2,
    PRE_RESULT_REPAIR_CONTRACT_VERSION_V3,
    PRE_RESULT_REPAIR_CONTRACT_VERSION_V4,
    RUNTIME_ALIGNMENT_BENCHMARK_REASON,
    RUNTIME_ALIGNMENT_CONTRACT_VERSION,
    RUNTIME_ALIGNMENT_INDUSTRY_REASON,
    RUNTIME_ALIGNMENT_REPAIR_GENERATION,
    RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS,
    RUNTIME_ALIGNMENT_SOURCE_BINDINGS,
    RUNTIME_ALIGNMENT_SOURCE_BUNDLE_SHA256,
    RUNTIME_ALIGNMENT_SOURCE_COMMIT,
    RUNTIME_ALIGNMENT_SOURCE_DATASET,
    RUNTIME_ALIGNMENT_SOURCE_RECIPE_VERSION,
    RUNTIME_ALIGNMENT_SOURCE_RUNNER_SHA256,
    RUNTIME_ALIGNMENT_TARGET_BUNDLE_SHA256,
    RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION,
    RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    canonical_sha256,
    validate_pre_result_repair_receipt,
)
from quant_platform.transparent_baseline_runner import (
    TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
    runtime_alignment_bundle_sha256,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_inventory(
    root_value: Any, *, allow_empty: bool = False
) -> list[dict[str, Any]]:
    root = Path(str(root_value or "")).resolve()
    if not root.is_dir():
        raise ValueError("transparent baseline repair artifact root is missing")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative != "manifest.json" and not relative.startswith("baseline/"):
            raise ValueError(
                "transparent baseline repair artifact inventory contains performance output"
            )
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not files and not allow_empty:
        raise ValueError("transparent baseline repair artifact inventory is missing")
    return files


def build_optimizer_applicability_receipt(
    members: Sequence[dict[str, Any]],
    *,
    source_release_commit: str = OPTIMIZER_APPLICABILITY_SOURCE_COMMIT,
    target_recipe_version: str = OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
) -> dict[str, Any]:
    """Build and validate the immutable, production-specific v2 receipt."""

    payload = {
        "contract_version": PRE_RESULT_REPAIR_CONTRACT_VERSION_V2,
        "repair_generation": OPTIMIZER_APPLICABILITY_REPAIR_GENERATION,
        "source_release_commit": source_release_commit,
        "target_recipe_version": target_recipe_version,
        TRANSPARENT_BASELINE_RUNNER_FIELD: OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
        "target_eligibility_contract": ELIGIBILITY_CONTRACT_VERSION,
        "target_stock_scope_contract": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
        "reason_codes": [OPTIMIZER_APPLICABILITY_REASON],
        "performance_information_used": False,
        "members": list(members),
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_pre_result_repair_receipt(receipt)


def build_canonical_lf_packaging_receipt(
    members: Sequence[dict[str, Any]],
    *,
    source_release_commit: str = CANONICAL_LF_PACKAGING_SOURCE_COMMIT,
    target_recipe_version: str = CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
) -> dict[str, Any]:
    """Build the exact v8-to-v9 canonical-LF pre-result receipt."""

    payload = {
        "contract_version": PRE_RESULT_REPAIR_CONTRACT_VERSION_V3,
        "repair_generation": CANONICAL_LF_PACKAGING_REPAIR_GENERATION,
        "source_release_commit": source_release_commit,
        "target_recipe_version": target_recipe_version,
        "source_runner_expected_sha256": (
            CANONICAL_LF_PACKAGING_SOURCE_EXPECTED_RUNNER_SHA256
        ),
        "source_runner_observed_sha256": (
            CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256
        ),
        TRANSPARENT_BASELINE_RUNNER_FIELD: (
            CANONICAL_LF_PACKAGING_TARGET_RUNNER_SHA256
        ),
        "packaging_contract_version": CANONICAL_LF_PACKAGING_CONTRACT_VERSION,
        "target_eligibility_contract": ELIGIBILITY_CONTRACT_VERSION,
        "target_stock_scope_contract": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
        "reason_codes": [CANONICAL_LF_PACKAGING_REASON],
        "performance_information_used": False,
        "members": list(members),
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_pre_result_repair_receipt(receipt)


def build_runtime_alignment_receipt(
    members: Sequence[dict[str, Any]],
    *,
    source_release_commit: str = RUNTIME_ALIGNMENT_SOURCE_COMMIT,
    target_recipe_version: str = RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION,
) -> dict[str, Any]:
    """Build the exact v9-to-v10 no-performance runtime repair receipt."""

    inventories = sorted(
        (
            {
                "backtest_id": str(member.get("backtest_id") or ""),
                "artifact_inventory_sha256": canonical_sha256(
                    list(member.get("files") or [])
                ),
            }
            for member in members
        ),
        key=lambda item: item["backtest_id"],
    )
    payload = {
        "contract_version": PRE_RESULT_REPAIR_CONTRACT_VERSION_V4,
        "repair_generation": RUNTIME_ALIGNMENT_REPAIR_GENERATION,
        "source_release_commit": source_release_commit,
        "target_recipe_version": target_recipe_version,
        "source_runner_sha256": RUNTIME_ALIGNMENT_SOURCE_RUNNER_SHA256,
        TRANSPARENT_BASELINE_RUNNER_FIELD: RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256,
        "source_runtime_bundle_sha256": RUNTIME_ALIGNMENT_SOURCE_BUNDLE_SHA256,
        "target_runtime_bundle_sha256": RUNTIME_ALIGNMENT_TARGET_BUNDLE_SHA256,
        "runtime_contract_version": RUNTIME_ALIGNMENT_CONTRACT_VERSION,
        "source_artifact_inventories_sha256": canonical_sha256(inventories),
        "target_eligibility_contract": ELIGIBILITY_CONTRACT_VERSION,
        "target_stock_scope_contract": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
        "reason_codes": sorted(
            [RUNTIME_ALIGNMENT_BENCHMARK_REASON, RUNTIME_ALIGNMENT_INDUSTRY_REASON]
        ),
        "performance_information_used": False,
        "members": list(members),
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_pre_result_repair_receipt(receipt)


def register_optimizer_applicability_repair(
    database_url: str,
    *,
    backtest_ids: Sequence[str],
    actor: str,
    source_release_commit: str = OPTIMIZER_APPLICABILITY_SOURCE_COMMIT,
    target_recipe_version: str = OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
) -> dict[str, Any]:
    """Append the exact v7 no-performance evidence to the audit ledger.

    This helper cannot register arbitrary retries: the validator pins the
    source commit, target recipe, repair generation, failure marker and the
    three production backtest identifiers.
    """

    normalized_ids = [str(value or "").strip().lower() for value in backtest_ids]
    if len(normalized_ids) != 3 or set(normalized_ids) != (
        OPTIMIZER_APPLICABILITY_SOURCE_BACKTEST_IDS
    ):
        raise ValueError("optimizer applicability repair requires the exact three v7 backtests")
    username = str(actor or "").strip()
    if not username:
        raise ValueError("transparent baseline repair actor is required")

    engine = open_database(database_url)
    with engine.begin() as connection:
        rows = connection.execute(
            select(
                backtest_runs.c.id,
                backtest_runs.c.strategy_version_id,
                backtest_runs.c.job_id,
                backtest_runs.c.dataset,
                backtest_runs.c.periods_json,
                backtest_runs.c.status,
                backtest_runs.c.metrics_json,
                backtest_runs.c.artifact_path,
                backtest_runs.c.error,
                jobs.c.status.label("job_status"),
                jobs.c.error.label("job_error"),
                strategy_versions.c.config_json,
            )
            .join(jobs, jobs.c.id == backtest_runs.c.job_id)
            .join(
                strategy_versions,
                strategy_versions.c.id == backtest_runs.c.strategy_version_id,
            )
            .where(backtest_runs.c.id.in_(normalized_ids))
            .with_for_update()
        ).all()
        if len(rows) != 3:
            raise ValueError("optimizer applicability repair backtests are incomplete")

        members: list[dict[str, Any]] = []
        for row in rows:
            config = dict(row.config_json or {})
            if config.get("recipe_version") != OPTIMIZER_APPLICABILITY_SOURCE_RECIPE_VERSION:
                raise ValueError("optimizer applicability repair source recipe changed")
            if (
                str(row.status) != "failed"
                or str(row.job_status) != "failed"
                or row.metrics_json is not None
                or str(row.error or "") != OPTIMIZER_APPLICABILITY_ERROR
                or str(row.job_error or "") != OPTIMIZER_APPLICABILITY_ERROR
            ):
                raise ValueError(
                    "optimizer applicability repair requires failed no-metrics evidence"
                )
            members.append(
                {
                    "backtest_id": str(row.id),
                    "strategy_version_id": str(row.strategy_version_id),
                    "job_id": str(row.job_id),
                    "dataset": str(row.dataset),
                    "periods": {
                        key: str(dict(row.periods_json or {})[key])
                        for key in ("historical_start", "historical_end", "start", "end")
                    },
                    "status": "failed",
                    "job_status": "failed",
                    "error": OPTIMIZER_APPLICABILITY_ERROR,
                    "metrics_absent": True,
                    "result_absent": True,
                    "files": _artifact_inventory(row.artifact_path),
                }
            )
        members.sort(key=lambda item: item["backtest_id"])
        receipt = build_optimizer_applicability_receipt(
            members,
            source_release_commit=source_release_commit,
            target_recipe_version=target_recipe_version,
        )

        for audit_row in connection.execute(
            select(audit_events).where(audit_events.c.action == PRE_RESULT_REPAIR_ACTION)
        ).all():
            details = dict(audit_row.details_json or {})
            if details.get("receipt_sha256") == receipt["receipt_sha256"]:
                return {
                    "status": "already_registered",
                    "audit_event_id": int(audit_row.id),
                    "receipt": receipt,
                }
            raw_members = details.get("members")
            if isinstance(raw_members, list) and {
                str(item.get("backtest_id") or "")
                for item in raw_members
                if isinstance(item, dict)
            } == set(normalized_ids):
                raise ValueError("v7 backtests already have a different repair receipt")

        event_id = connection.execute(
            insert(audit_events)
            .values(
                user_id=None,
                username=username,
                action=PRE_RESULT_REPAIR_ACTION,
                method="INTERNAL",
                path="transparent-baseline/pre-result-repair",
                status_code=201,
                ip_hash=None,
                user_agent="register_transparent_baseline_repair.py",
                details_json=receipt,
                created_at=datetime.now(UTC),
            )
            .returning(audit_events.c.id)
        ).scalar_one()
    return {
        "status": "registered",
        "audit_event_id": int(event_id),
        "receipt": receipt,
    }


def register_canonical_lf_packaging_repair(
    database_url: str,
    *,
    backtest_ids: Sequence[str],
    actor: str,
    source_runner_observed_sha256: str,
    target_runner_path: Path | None = None,
    source_release_commit: str = CANONICAL_LF_PACKAGING_SOURCE_COMMIT,
    target_recipe_version: str = CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
) -> dict[str, Any]:
    """Register only the three exact v8 pre-run packaging failures."""

    normalized_ids = [str(value or "").strip().lower() for value in backtest_ids]
    if len(normalized_ids) != 3 or set(normalized_ids) != (
        CANONICAL_LF_PACKAGING_SOURCE_BACKTEST_IDS
    ):
        raise ValueError("canonical LF packaging repair requires the exact three v8 backtests")
    username = str(actor or "").strip()
    if not username:
        raise ValueError("transparent baseline repair actor is required")
    observed_source_runner = str(source_runner_observed_sha256 or "").strip().lower()
    if observed_source_runner != CANONICAL_LF_PACKAGING_SOURCE_OBSERVED_RUNNER_SHA256:
        raise ValueError("canonical LF packaging observed source runner changed")
    governed_target_runner = (
        target_runner_path
        if target_runner_path is not None
        else Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_multifactor_backtest.py"
    )
    if _file_sha256(governed_target_runner) != (
        CANONICAL_LF_PACKAGING_TARGET_RUNNER_SHA256
    ):
        raise ValueError("canonical LF packaging target runner bytes changed")

    engine = open_database(database_url)
    with engine.begin() as connection:
        rows = connection.execute(
            select(
                backtest_runs.c.id,
                backtest_runs.c.strategy_version_id,
                backtest_runs.c.job_id,
                backtest_runs.c.dataset,
                backtest_runs.c.periods_json,
                backtest_runs.c.status,
                backtest_runs.c.metrics_json,
                backtest_runs.c.artifact_path,
                backtest_runs.c.error,
                jobs.c.status.label("job_status"),
                jobs.c.error.label("job_error"),
                jobs.c.log_path.label("job_log_path"),
                jobs.c.payload_json.label("job_payload_json"),
                strategy_versions.c.config_json,
            )
            .join(jobs, jobs.c.id == backtest_runs.c.job_id)
            .join(
                strategy_versions,
                strategy_versions.c.id == backtest_runs.c.strategy_version_id,
            )
            .where(backtest_runs.c.id.in_(normalized_ids))
            .with_for_update()
        ).all()
        if len(rows) != 3:
            raise ValueError("canonical LF packaging repair backtests are incomplete")

        members: list[dict[str, Any]] = []
        for row in rows:
            config = dict(row.config_json or {})
            if config.get("recipe_version") != (
                CANONICAL_LF_PACKAGING_SOURCE_RECIPE_VERSION
            ):
                raise ValueError("canonical LF packaging repair source recipe changed")
            bootstrap = dict(config.get("transparent_baseline_bootstrap") or {})
            if bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD) != (
                CANONICAL_LF_PACKAGING_SOURCE_EXPECTED_RUNNER_SHA256
            ):
                raise ValueError("canonical LF packaging source runner changed")
            periods = {
                key: str(dict(row.periods_json or {})[key])
                for key in ("historical_start", "historical_end", "start", "end")
            }
            job_payload = dict(row.job_payload_json or {})
            if (
                str(job_payload.get("backtest_id") or "") != str(row.id)
                or str(job_payload.get("strategy_version_id") or "")
                != str(row.strategy_version_id)
                or str(job_payload.get("dataset") or "") != str(row.dataset)
                or dict(job_payload.get("periods") or {}) != periods
                or job_payload.get("transparent_baseline_runner_sha256")
                != CANONICAL_LF_PACKAGING_SOURCE_EXPECTED_RUNNER_SHA256
            ):
                raise ValueError("canonical LF packaging source job binding changed")
            if (
                str(row.status) != "failed"
                or str(row.job_status) != "failed"
                or row.metrics_json is not None
                or str(row.error or "") != CANONICAL_LF_PACKAGING_ERROR
                or str(row.job_error or "") != CANONICAL_LF_PACKAGING_ERROR
            ):
                raise ValueError(
                    "canonical LF packaging repair requires failed no-metrics evidence"
                )
            files = _artifact_inventory(row.artifact_path, allow_empty=True)
            if files:
                raise ValueError("canonical LF packaging repair artifacts must be empty")
            log_path = Path(str(row.job_log_path or "")).resolve()
            if log_path.exists() and (not log_path.is_file() or log_path.stat().st_size != 0):
                raise ValueError("canonical LF packaging repair job log must be empty")
            members.append(
                {
                    "backtest_id": str(row.id),
                    "strategy_version_id": str(row.strategy_version_id),
                    "job_id": str(row.job_id),
                    "dataset": str(row.dataset),
                    "periods": periods,
                    "status": "failed",
                    "job_status": "failed",
                    "error": CANONICAL_LF_PACKAGING_ERROR,
                    "metrics_absent": True,
                    "result_absent": True,
                    "files": [],
                }
            )
        members.sort(key=lambda item: item["backtest_id"])
        receipt = build_canonical_lf_packaging_receipt(
            members,
            source_release_commit=source_release_commit,
            target_recipe_version=target_recipe_version,
        )

        for audit_row in connection.execute(
            select(audit_events).where(audit_events.c.action == PRE_RESULT_REPAIR_ACTION)
        ).all():
            details = dict(audit_row.details_json or {})
            if details.get("receipt_sha256") == receipt["receipt_sha256"]:
                return {
                    "status": "already_registered",
                    "audit_event_id": int(audit_row.id),
                    "receipt": receipt,
                }
            raw_members = details.get("members")
            if isinstance(raw_members, list) and {
                str(item.get("backtest_id") or "")
                for item in raw_members
                if isinstance(item, dict)
            } == set(normalized_ids):
                raise ValueError("v8 backtests already have a different repair receipt")

        event_id = connection.execute(
            insert(audit_events)
            .values(
                user_id=None,
                username=username,
                action=PRE_RESULT_REPAIR_ACTION,
                method="INTERNAL",
                path="transparent-baseline/pre-result-repair",
                status_code=201,
                ip_hash=None,
                user_agent="register_transparent_baseline_packaging_repair.py",
                details_json=receipt,
                created_at=datetime.now(UTC),
            )
            .returning(audit_events.c.id)
        ).scalar_one()
    return {
        "status": "registered",
        "audit_event_id": int(event_id),
        "receipt": receipt,
    }


def register_runtime_alignment_repair(
    database_url: str,
    *,
    backtest_ids: Sequence[str],
    actor: str,
    source_runtime_root: Path,
    target_runner_path: Path | None = None,
    source_release_commit: str = RUNTIME_ALIGNMENT_SOURCE_COMMIT,
    target_recipe_version: str = RUNTIME_ALIGNMENT_TARGET_RECIPE_VERSION,
) -> dict[str, Any]:
    """Register only the exact three v9 runtime-contract failures.

    The source IDs, jobs, versions, dataset, OOS periods, exact errors and the
    complete partial-artifact inventories are production-specific constants.
    The helper refuses registration if any performance result exists or the
    final corrected runner bytes have not yet been sealed.
    """

    normalized_ids = [str(value or "").strip().lower() for value in backtest_ids]
    if len(normalized_ids) != 3 or set(normalized_ids) != (
        RUNTIME_ALIGNMENT_SOURCE_BACKTEST_IDS
    ):
        raise ValueError("runtime alignment repair requires the exact three v9 backtests")
    username = str(actor or "").strip()
    if not username:
        raise ValueError("transparent baseline repair actor is required")
    governed_source_root = source_runtime_root.resolve()
    source_runner = governed_source_root / "scripts" / "run_multifactor_backtest.py"
    if _file_sha256(source_runner) != RUNTIME_ALIGNMENT_SOURCE_RUNNER_SHA256:
        raise ValueError("runtime alignment source runner bytes changed or are not sealed")
    if runtime_alignment_bundle_sha256(governed_source_root) != (
        RUNTIME_ALIGNMENT_SOURCE_BUNDLE_SHA256
    ):
        raise ValueError("runtime alignment source runtime bundle changed or is not sealed")
    governed_target_runner = (
        target_runner_path
        if target_runner_path is not None
        else Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_multifactor_backtest.py"
    )
    if _file_sha256(governed_target_runner) != RUNTIME_ALIGNMENT_TARGET_RUNNER_SHA256:
        raise ValueError("runtime alignment target runner bytes changed or are not sealed")
    if runtime_alignment_bundle_sha256(governed_target_runner.parents[1]) != (
        RUNTIME_ALIGNMENT_TARGET_BUNDLE_SHA256
    ):
        raise ValueError("runtime alignment target runtime bundle changed or is not sealed")
    if governed_source_root == governed_target_runner.parents[1].resolve():
        raise ValueError("runtime alignment source and target roots must be distinct")

    engine = open_database(database_url)
    with engine.begin() as connection:
        rows = connection.execute(
            select(
                backtest_runs.c.id,
                backtest_runs.c.strategy_version_id,
                backtest_runs.c.job_id,
                backtest_runs.c.dataset,
                backtest_runs.c.periods_json,
                backtest_runs.c.status,
                backtest_runs.c.metrics_json,
                backtest_runs.c.artifact_path,
                backtest_runs.c.error,
                jobs.c.status.label("job_status"),
                jobs.c.error.label("job_error"),
                jobs.c.payload_json.label("job_payload_json"),
                strategy_versions.c.config_json,
            )
            .join(jobs, jobs.c.id == backtest_runs.c.job_id)
            .join(
                strategy_versions,
                strategy_versions.c.id == backtest_runs.c.strategy_version_id,
            )
            .where(backtest_runs.c.id.in_(normalized_ids))
            .with_for_update()
        ).all()
        if len(rows) != 3:
            raise ValueError("runtime alignment repair backtests are incomplete")

        members: list[dict[str, Any]] = []
        for row in rows:
            source = RUNTIME_ALIGNMENT_SOURCE_BINDINGS.get(str(row.id))
            if source is None:
                raise ValueError("runtime alignment repair source binding is unknown")
            config = dict(row.config_json or {})
            bootstrap = dict(config.get("transparent_baseline_bootstrap") or {})
            periods = {
                key: str(dict(row.periods_json or {})[key])
                for key in ("historical_start", "historical_end", "start", "end")
            }
            job_payload = dict(row.job_payload_json or {})
            if (
                config.get("recipe_version") != RUNTIME_ALIGNMENT_SOURCE_RECIPE_VERSION
                or bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
                != RUNTIME_ALIGNMENT_SOURCE_RUNNER_SHA256
                or str(row.strategy_version_id) != source["strategy_version_id"]
                or str(row.job_id) != source["job_id"]
                or str(row.dataset) != RUNTIME_ALIGNMENT_SOURCE_DATASET
                or periods != source["periods"]
            ):
                raise ValueError("runtime alignment repair source contract changed")
            if (
                str(job_payload.get("backtest_id") or "") != str(row.id)
                or str(job_payload.get("strategy_version_id") or "")
                != str(row.strategy_version_id)
                or str(job_payload.get("dataset") or "") != str(row.dataset)
                or dict(job_payload.get("periods") or {}) != periods
                or job_payload.get("transparent_baseline_runner_sha256")
                != RUNTIME_ALIGNMENT_SOURCE_RUNNER_SHA256
                or job_payload.get(TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD)
                is not None
            ):
                raise ValueError("runtime alignment repair source job binding changed")
            if (
                str(row.status) != "failed"
                or str(row.job_status) != "failed"
                or row.metrics_json is not None
                or str(row.error or "") != source["error"]
                or str(row.job_error or "") != source["error"]
            ):
                raise ValueError(
                    "runtime alignment repair requires exact failed no-metrics evidence"
                )
            root = Path(str(row.artifact_path or "")).resolve()
            if any(root.rglob("result.json")):
                raise ValueError("runtime alignment repair source already has a result")
            files = _artifact_inventory(root)
            if canonical_sha256(files) != source["artifact_inventory_sha256"]:
                raise ValueError("runtime alignment repair artifact inventory changed")
            members.append(
                {
                    "backtest_id": str(row.id),
                    "strategy_version_id": str(row.strategy_version_id),
                    "job_id": str(row.job_id),
                    "dataset": str(row.dataset),
                    "periods": periods,
                    "status": "failed",
                    "job_status": "failed",
                    "error": str(source["error"]),
                    "metrics_absent": True,
                    "result_absent": True,
                    "files": files,
                }
            )
        members.sort(key=lambda item: item["backtest_id"])
        receipt = build_runtime_alignment_receipt(
            members,
            source_release_commit=source_release_commit,
            target_recipe_version=target_recipe_version,
        )

        for audit_row in connection.execute(
            select(audit_events).where(audit_events.c.action == PRE_RESULT_REPAIR_ACTION)
        ).all():
            details = dict(audit_row.details_json or {})
            if details.get("receipt_sha256") == receipt["receipt_sha256"]:
                return {
                    "status": "already_registered",
                    "audit_event_id": int(audit_row.id),
                    "receipt": receipt,
                }
            raw_members = details.get("members")
            if isinstance(raw_members, list) and {
                str(item.get("backtest_id") or "")
                for item in raw_members
                if isinstance(item, dict)
            } == set(normalized_ids):
                raise ValueError("v9 backtests already have a different repair receipt")

        event_id = connection.execute(
            insert(audit_events)
            .values(
                user_id=None,
                username=username,
                action=PRE_RESULT_REPAIR_ACTION,
                method="INTERNAL",
                path="transparent-baseline/pre-result-repair",
                status_code=201,
                ip_hash=None,
                user_agent="register_transparent_baseline_runtime_repair.py",
                details_json=receipt,
                created_at=datetime.now(UTC),
            )
            .returning(audit_events.c.id)
        ).scalar_one()
    return {
        "status": "registered",
        "audit_event_id": int(event_id),
        "receipt": receipt,
    }
