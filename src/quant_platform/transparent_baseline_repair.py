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
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    canonical_sha256,
    validate_pre_result_repair_receipt,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_inventory(root_value: Any) -> list[dict[str, Any]]:
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
    if not files:
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
