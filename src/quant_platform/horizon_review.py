from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

FINANCIAL_REVIEW_TRIGGER_VERSION = "pit-financial-review-trigger-v1"
FINANCIAL_REVIEW_TRIGGER_SOURCE = "pit_financial_announcement"

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


def validate_financial_review_trigger(
    value: Mapping[str, Any],
    *,
    expected_signal_date: date,
    expected_dataset_identity_sha256: str,
) -> dict[str, Any]:
    """Validate the immutable event interval that makes a PIT review due."""

    trigger = dict(value or {})
    if trigger.get("contract_version") != FINANCIAL_REVIEW_TRIGGER_VERSION:
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

    newest_report_end = max(date.fromisoformat(item["report_end"]) for item in rows)
    report_period = _report_period(newest_report_end)
    selected = sorted(
        (
            item
            for item in rows
            if _report_period(date.fromisoformat(item["report_end"])) == report_period
        ),
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
        "announcement_date": announcement_date.isoformat(),
        "source_datasets": sorted({item["source_dataset"] for item in selected}),
        "source_event_count": len(selected),
        "source_event_sha256": _canonical_sha256(selected),
    }
    return validate_financial_review_trigger(
        trigger,
        expected_signal_date=signal_date,
        expected_dataset_identity_sha256=dataset_identity,
    )
