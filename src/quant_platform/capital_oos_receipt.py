"""Small, fail-closed contracts for the capital-facing final OOS boundary.

This module intentionally knows nothing about a database.  The ledger owns
alpha allocation and immutability; callers use these helpers to turn the
immutable formal-backtest artifact into the *one* paired return series and to
verify a compact ledger receipt before capital can move to paper.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_oos_paired_returns(
    artifact_path: Path,
    *,
    expected_trading_dates: Sequence[str],
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Read the formal OOS artifact as candidate net vs benchmark net.

    The Qlib report is deliberately not compounded or resampled here.  The
    capital ledger tests the exact daily paired observations it preregistered:
    candidate = ``return - cost`` and baseline = ``bench``.  Any ambiguous
    date column, missing observation, duplicate or non-finite value blocks
    settlement rather than silently changing the sample.
    """

    path = Path(artifact_path)
    if not path.is_file():
        raise ValueError("formal OOS daily_returns artifact is unavailable")
    expected = [str(item) for item in expected_trading_dates]
    if not expected or expected != sorted(expected) or len(expected) != len(set(expected)):
        raise ValueError("capital OOS expected trading dates are invalid")
    frame = pd.read_parquet(path)
    date_columns = [column for column in ("datetime", "date") if column in frame.columns]
    if len(date_columns) != 1:
        raise ValueError("formal OOS daily_returns needs exactly one datetime/date column")
    required = {"return", "cost", "bench"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("formal OOS daily_returns is missing " + ", ".join(missing))
    dates = pd.to_datetime(frame[date_columns[0]], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    if dates.isna().any():
        raise ValueError("formal OOS daily_returns contains an invalid date")
    normalized = dates.dt.normalize()
    if normalized.duplicated().any():
        raise ValueError("formal OOS daily_returns contains duplicate trading dates")
    observed = [item.date().isoformat() for item in normalized]
    if observed != expected:
        raise ValueError("formal OOS daily_returns differs from the preregistered window")
    candidate = pd.to_numeric(frame["return"], errors="coerce") - pd.to_numeric(
        frame["cost"], errors="coerce"
    )
    baseline = pd.to_numeric(frame["bench"], errors="coerce")
    candidate.index = pd.DatetimeIndex(normalized)
    baseline.index = pd.DatetimeIndex(normalized)
    if candidate.isna().any() or baseline.isna().any():
        raise ValueError("formal OOS paired returns contain non-numeric values")
    candidate = candidate.astype(float).rename("candidate_net_return")
    baseline = baseline.astype(float).rename("baseline_net_return")
    evidence = {
        "contract_version": "capital-oos-artifact-paired-returns-v1",
        "formal_oos_artifact_sha256": sha256_file(path),
        "daily_returns_filename": path.name,
        "trading_day_count": len(expected),
        "trading_dates": expected,
        "candidate_return_definition": "return_minus_cost",
        "baseline_return_definition": "bench",
    }
    return candidate, baseline, evidence


def capital_oos_receipt(
    batch: Mapping[str, Any],
    *,
    backtest_id: str,
    strategy_version_id: str,
    dataset: str,
    periods: Mapping[str, Any],
    formal_oos_artifact_sha256: str,
) -> dict[str, Any]:
    """Produce a compact immutable approval receipt from a settled batch."""

    if str(batch.get("status") or "") != "settled" or batch.get("passed") is not True:
        raise ValueError("capital OOS batch is not a settled passing receipt")
    evidence = batch.get("settlement_evidence_json")
    if not isinstance(evidence, Mapping):
        raise ValueError("capital OOS settlement evidence is unavailable")
    support = evidence.get("supporting_evidence")
    if not isinstance(support, Mapping) or str(
        support.get("formal_oos_artifact_sha256") or ""
    ).lower() != str(formal_oos_artifact_sha256).lower():
        raise ValueError("capital OOS receipt is not bound to the formal artifact")
    expected_dates = [str(item) for item in batch.get("trading_dates_json") or []]
    start = str(periods.get("start") or "")
    end = str(periods.get("end") or "")
    if (
        not backtest_id
        or not strategy_version_id
        or not dataset
        or start != str(batch.get("final_oos_start") or "")
        or end != str(batch.get("final_oos_end") or "")
        or not expected_dates
    ):
        raise ValueError("capital OOS receipt has an invalid strategy/backtest binding")
    return {
        "contract_version": "capital-oos-approval-receipt-v1",
        "batch_id": str(batch.get("id") or ""),
        "batch_settlement_evidence_sha256": str(
            batch.get("settlement_evidence_sha256") or ""
        ),
        "passed": True,
        "backtest_id": str(backtest_id),
        "strategy_version_id": str(strategy_version_id),
        "dataset": str(dataset),
        "dataset_identity_sha256": str(batch.get("dataset_identity_sha256") or ""),
        "dataset_lineage_id": str(batch.get("dataset_lineage_id") or ""),
        "final_oos_start": start,
        "final_oos_end": end,
        "trading_dates_sha256": str(batch.get("trading_dates_sha256") or ""),
        "formal_oos_artifact_sha256": str(formal_oos_artifact_sha256).lower(),
        "frozen_bundle_manifest_sha256": str(
            batch.get("frozen_bundle_manifest_sha256") or ""
        ),
        "frozen_baseline_manifest_sha256": str(
            batch.get("frozen_baseline_manifest_sha256") or ""
        ),
    }


def require_capital_oos_receipt(
    receipt: Mapping[str, Any] | None,
    *,
    backtest_id: str,
    strategy_version_id: str,
    dataset: str,
    periods: Mapping[str, Any],
) -> dict[str, Any]:
    """Check only the immutable identifiers an approval can observe."""

    if not isinstance(receipt, Mapping):
        raise ValueError("capital OOS settled receipt is required before paper approval")
    result = dict(receipt)
    if (
        result.get("contract_version") != "capital-oos-approval-receipt-v1"
        or result.get("passed") is not True
        or not all(
            isinstance(result.get(name), str) and len(str(result[name])) == 64
            for name in (
                "batch_settlement_evidence_sha256",
                "dataset_identity_sha256",
                "dataset_lineage_id",
                "trading_dates_sha256",
                "formal_oos_artifact_sha256",
                "frozen_bundle_manifest_sha256",
                "frozen_baseline_manifest_sha256",
            )
        )
        or str(result.get("backtest_id") or "") != str(backtest_id)
        or str(result.get("strategy_version_id") or "") != str(strategy_version_id)
        or str(result.get("dataset") or "") != str(dataset)
        or str(result.get("final_oos_start") or "") != str(periods.get("start") or "")
        or str(result.get("final_oos_end") or "") != str(periods.get("end") or "")
    ):
        raise ValueError("capital OOS receipt does not match the formal backtest")
    return result
