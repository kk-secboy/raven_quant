from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

FINANCIAL_REVIEW_TRIGGER_VERSION = "pit-financial-review-trigger-v3"
SCOPED_FINANCIAL_REVIEW_TRIGGER_VERSION = "pit-financial-review-trigger-v2"
LEGACY_FINANCIAL_REVIEW_TRIGGER_VERSION = "pit-financial-review-trigger-v1"
FINANCIAL_REVIEW_TRIGGER_SOURCE = "pit_financial_announcement"
FINANCIAL_REVIEW_SCOPE_VERSION = "pit-financial-review-scope-v1"

_FINANCIAL_DATASETS = (
    "fina_indicator",
    "fina_indicator_nondefault",
    "income",
    "balancesheet",
    "cashflow",
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_sql(column: str) -> str:
    identifier = '"' + column.replace('"', '""') + '"'
    return (
        f"coalesce(try_cast({identifier} AS DATE), "
        f"try_strptime(CAST({identifier} AS VARCHAR), '%Y%m%d')::DATE)"
    )


def _report_period(value: date) -> str:
    quarter = (value.month - 1) // 3 + 1
    return f"{value.year:04d}Q{quarter}"


def _report_period_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("financial review report periods must be a list")
    periods = [str(item or "").strip() for item in value]
    if (
        periods != sorted(set(periods))
        or not periods
        or any(
            len(period) != 6
            or not period[:4].isdigit()
            or period[4] != "Q"
            or period[5] not in "1234"
            for period in periods
        )
    ):
        raise ValueError("financial review report periods are invalid")
    return periods


def _instrument(value: Any) -> str:
    """Normalize a Tushare/engine symbol to the Qlib engine identity."""

    text = str(value or "").strip().upper()
    if "." in text:
        digits, exchange = text.split(".", 1)
        text = f"{exchange}{digits}"
    if (
        len(text) < 3
        or text[:2] not in {"SH", "SZ", "BJ"}
        or not text[2:].isdigit()
    ):
        raise ValueError("financial review instrument identity is invalid")
    return text


def _instrument_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("financial review instruments must be a list")
    normalized = [_instrument(item) for item in value]
    if normalized != sorted(set(normalized)):
        raise ValueError("financial review instruments must be sorted and unique")
    return normalized


def validate_financial_review_trigger(
    value: Mapping[str, Any],
    *,
    expected_signal_date: date,
    expected_dataset_identity_sha256: str,
) -> dict[str, Any]:
    """Validate the immutable event interval that makes a PIT review due."""

    trigger = dict(value or {})
    contract_version = str(trigger.get("contract_version") or "")
    if contract_version not in {
        FINANCIAL_REVIEW_TRIGGER_VERSION,
        SCOPED_FINANCIAL_REVIEW_TRIGGER_VERSION,
        LEGACY_FINANCIAL_REVIEW_TRIGGER_VERSION,
    }:
        raise ValueError("financial review trigger contract is invalid")
    if trigger.get("trigger_source") != FINANCIAL_REVIEW_TRIGGER_SOURCE:
        raise ValueError("financial review trigger source is invalid")
    if (
        str(trigger.get("dataset_identity_sha256") or "")
        != expected_dataset_identity_sha256
        or len(expected_dataset_identity_sha256) != 64
    ):
        raise ValueError("financial review trigger dataset identity is invalid")
    try:
        previous_signal_date = date.fromisoformat(str(trigger["previous_signal_date"]))
        announcement_date = date.fromisoformat(str(trigger["announcement_date"]))
        effective_date = date.fromisoformat(str(trigger["trigger_effective_date"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("financial review trigger dates are invalid") from exc
    if (
        effective_date != expected_signal_date
        or not previous_signal_date <= announcement_date < effective_date
    ):
        raise ValueError("financial review trigger violates PIT availability")
    report_period = str(trigger.get("report_period") or "").strip()
    source_datasets = trigger.get("source_datasets")
    source_event_count = trigger.get("source_event_count")
    source_event_sha256 = str(trigger.get("source_event_sha256") or "").lower()
    if contract_version in {
        FINANCIAL_REVIEW_TRIGGER_VERSION,
        SCOPED_FINANCIAL_REVIEW_TRIGGER_VERSION,
    }:
        affected_instruments = _instrument_list(trigger.get("affected_instruments"))
        affected_instrument_count = trigger.get("affected_instrument_count")
        if (
            not affected_instruments
            or isinstance(affected_instrument_count, bool)
            or not isinstance(affected_instrument_count, int)
            or affected_instrument_count != len(affected_instruments)
        ):
            raise ValueError("financial review affected-instrument identity is invalid")
    elif "affected_instruments" in trigger or "affected_instrument_count" in trigger:
        raise ValueError("legacy financial review trigger cannot claim scoped instruments")
    if contract_version == FINANCIAL_REVIEW_TRIGGER_VERSION:
        report_periods = _report_period_list(trigger.get("report_periods"))
        if report_period != report_periods[-1]:
            raise ValueError("financial review newest report period is invalid")
    elif "report_periods" in trigger:
        raise ValueError("legacy financial review trigger cannot claim report-period coverage")
    if (
        not report_period
        or not isinstance(source_datasets, list)
        or not source_datasets
        or source_datasets != sorted(set(str(item) for item in source_datasets))
        or isinstance(source_event_count, bool)
        or not isinstance(source_event_count, int)
        or source_event_count < 1
        or len(source_event_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_event_sha256)
    ):
        raise ValueError("financial review trigger event identity is invalid")
    return trigger


def build_financial_review_scope(
    trigger: Mapping[str, Any],
    *,
    current_holdings: list[str] | set[str] | tuple[str, ...],
    qualified_candidates: list[str] | set[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Bind a PIT filing batch to instruments relevant to this decision.

    A-share filing season produces announcements on most trading days.  The
    source batch remains governed evidence, but it becomes an off-cadence
    portfolio decision only when it affects an existing holding or a candidate
    that already passed the current strategy's eligibility/ranking gates.
    """

    normalized_trigger = dict(trigger or {})
    contract_version = str(normalized_trigger.get("contract_version") or "")
    if contract_version in {
        FINANCIAL_REVIEW_TRIGGER_VERSION,
        SCOPED_FINANCIAL_REVIEW_TRIGGER_VERSION,
    }:
        affected = set(_instrument_list(normalized_trigger.get("affected_instruments")))
    elif contract_version == LEGACY_FINANCIAL_REVIEW_TRIGGER_VERSION:
        # Old queued artifacts did not preserve instrument scope.  They remain
        # auditable source batches, but must never force a broad rebalance.
        affected = set()
    else:
        raise ValueError("financial review trigger contract is invalid")
    holdings = {_instrument(item) for item in current_holdings}
    candidates = {_instrument(item) for item in qualified_candidates}
    affected_holdings = sorted(affected & holdings)
    affected_candidates = sorted((affected & candidates) - set(affected_holdings))
    reviewed = sorted({*affected_holdings, *affected_candidates})
    ignored = sorted(affected - set(reviewed))
    payload = {
        "contract_version": FINANCIAL_REVIEW_SCOPE_VERSION,
        "review_mode": "decision_review" if reviewed else "governance_only",
        "source_event_sha256": str(normalized_trigger.get("source_event_sha256") or ""),
        "affected_holdings": affected_holdings,
        "affected_candidates": affected_candidates,
        "reviewed_instruments": reviewed,
        "ignored_instruments": ignored,
    }
    return {**payload, "scope_sha256": _canonical_sha256(payload)}


def validate_financial_review_scope(
    value: Mapping[str, Any],
    *,
    trigger: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a decision/governance scope against its immutable event batch."""

    scope = dict(value or {})
    scope_sha256 = str(scope.pop("scope_sha256", "")).lower()
    if (
        len(scope_sha256) != 64
        or any(character not in "0123456789abcdef" for character in scope_sha256)
        or _canonical_sha256(scope) != scope_sha256
    ):
        raise ValueError("financial review scope seal is invalid")
    if scope.get("contract_version") != FINANCIAL_REVIEW_SCOPE_VERSION:
        raise ValueError("financial review scope contract is invalid")
    if scope.get("source_event_sha256") != trigger.get("source_event_sha256"):
        raise ValueError("financial review scope changed its source event")
    holdings = _instrument_list(scope.get("affected_holdings"))
    candidates = _instrument_list(scope.get("affected_candidates"))
    reviewed = _instrument_list(scope.get("reviewed_instruments"))
    ignored = _instrument_list(scope.get("ignored_instruments"))
    if set(holdings) & set(candidates) or reviewed != sorted({*holdings, *candidates}):
        raise ValueError("financial review scope partitions are invalid")
    trigger_version = str(trigger.get("contract_version") or "")
    affected = (
        _instrument_list(trigger.get("affected_instruments"))
        if trigger_version
        in {
            FINANCIAL_REVIEW_TRIGGER_VERSION,
            SCOPED_FINANCIAL_REVIEW_TRIGGER_VERSION,
        }
        else []
    )
    if set(reviewed) & set(ignored) or sorted({*reviewed, *ignored}) != affected:
        raise ValueError("financial review scope does not cover its source batch")
    expected_mode = "decision_review" if reviewed else "governance_only"
    if scope.get("review_mode") != expected_mode:
        raise ValueError("financial review scope mode is invalid")
    return {**scope, "scope_sha256": scope_sha256}


def resolve_financial_review_trigger(
    *,
    data_root: Path,
    dataset_provenance: Mapping[str, Any],
    previous_signal_date: date | None,
    signal_date: date,
) -> dict[str, Any] | None:
    """Resolve newly PIT-effective financial announcements from the source snapshot.

    The Qlib daily contract makes a filing dated ``D`` visible only on a
    trading date strictly after ``D``.  Consequently the current signal sees
    rows with announcement dates in ``[previous_signal_date, signal_date)``.
    A first paper decision has no prior forward boundary and deliberately does
    not backfill historical filings as new evidence.
    """

    if previous_signal_date is None:
        return None
    if previous_signal_date >= signal_date:
        raise ValueError("financial review requires an advancing signal date")
    provenance = dict(dataset_provenance or {})
    dataset_identity = str(provenance.get("dataset_identity_sha256") or "")
    snapshot_name = str(provenance.get("snapshot_name") or "").strip()
    snapshot_digest = str(provenance.get("snapshot_manifest_sha256") or "").lower()
    if (
        len(dataset_identity) != 64
        or len(snapshot_digest) != 64
        or not snapshot_name
    ):
        raise ValueError("financial review requires immutable source-snapshot provenance")
    snapshots_root = (Path(data_root) / "snapshots").resolve()
    snapshot = (snapshots_root / snapshot_name).resolve()
    try:
        snapshot.relative_to(snapshots_root)
    except ValueError as exc:
        raise ValueError("financial review source snapshot path is unsafe") from exc
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file() or _sha256_file(manifest_path) != snapshot_digest:
        raise ValueError("financial review source snapshot failed immutable verification")

    rows: list[dict[str, str]] = []
    connection = duckdb.connect()
    try:
        for source_dataset in _FINANCIAL_DATASETS:
            paths = sorted((snapshot / "parquet" / source_dataset).rglob("*.parquet"))
            if not paths:
                continue
            serialized = [str(path.resolve()) for path in paths]
            columns = {
                str(item[0])
                for item in connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)",
                    [serialized],
                ).fetchall()
            }
            if not {"ts_code", "ann_date", "end_date"}.issubset(columns):
                continue
            announcement = _date_sql("ann_date")
            report_end = _date_sql("end_date")
            values = connection.execute(
                f"""
                SELECT DISTINCT
                    CAST(ts_code AS VARCHAR) AS instrument,
                    {announcement} AS announcement_date,
                    {report_end} AS report_end
                FROM read_parquet(?, union_by_name=true)
                WHERE ts_code IS NOT NULL
                  AND {announcement} >= ?
                  AND {announcement} < ?
                  AND {report_end} IS NOT NULL
                  AND {report_end} <= {announcement}
                ORDER BY report_end, announcement_date, instrument
                """,
                [serialized, previous_signal_date, signal_date],
            ).fetchall()
            rows.extend(
                {
                    "source_dataset": source_dataset,
                    "instrument": str(instrument),
                    "announcement_date": announcement_date.isoformat(),
                    "report_end": report_end.isoformat(),
                }
                for instrument, announcement_date, report_end in values
            )
    finally:
        connection.close()
    if not rows:
        return None

    report_periods = sorted(
        {_report_period(date.fromisoformat(item["report_end"])) for item in rows}
    )
    report_period = report_periods[-1]
    selected = sorted(
        rows,
        key=lambda item: (
            item["source_dataset"],
            item["instrument"],
            item["announcement_date"],
            item["report_end"],
        ),
    )
    announcement_date = max(
        date.fromisoformat(item["announcement_date"]) for item in selected
    )
    trigger = {
        "contract_version": FINANCIAL_REVIEW_TRIGGER_VERSION,
        "trigger_source": FINANCIAL_REVIEW_TRIGGER_SOURCE,
        "trigger_effective_date": signal_date.isoformat(),
        "previous_signal_date": previous_signal_date.isoformat(),
        "dataset_identity_sha256": dataset_identity,
        "report_period": report_period,
        "report_periods": report_periods,
        "announcement_date": announcement_date.isoformat(),
        "source_datasets": sorted({item["source_dataset"] for item in selected}),
        "source_event_count": len(selected),
        "source_event_sha256": _canonical_sha256(selected),
        "affected_instruments": sorted(
            {_instrument(item["instrument"]) for item in selected}
        ),
    }
    trigger["affected_instrument_count"] = len(trigger["affected_instruments"])
    return validate_financial_review_trigger(
        trigger,
        expected_signal_date=signal_date,
        expected_dataset_identity_sha256=dataset_identity,
    )
