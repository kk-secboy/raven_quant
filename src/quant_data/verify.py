from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

from .catalog import ALL_DEFINITIONS
from .checkpoint import CheckpointStore
from .coverage_data import COVERAGE_DATASETS, coverage_primary_key_candidates
from .reference_data import select_current_reference_units


def verify_downloads(
    checkpoint: CheckpointStore,
    data_root: Path,
    *,
    snapshot_end: date | None = None,
    require_all_planned: bool = True,
) -> dict[str, Any]:
    """Verify durable files and the exact generation a successor snapshot will use.

    A shared checkpoint database may contain dormant plans for later pipeline
    stages.  A download stage proves its own explicit plan before invoking this
    verifier, so a chained snapshot can demote those unrelated pending rows to
    warnings while still validating every immutable successful file.  Duplicate
    checks always use the same current-generation selector as snapshot building;
    retained old reference generations and provider-capped page groups therefore
    remain auditable without being counted twice.
    """

    effective_snapshot_end = snapshot_end or date.today()
    datasets: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for row in checkpoint.verification_rows():
        item = dict(row)
        if item["succeeded"] != item["planned"]:
            message = f"{item['dataset']}: {item['succeeded']}/{item['planned']} units succeeded"
            (errors if require_all_planned else warnings).append(message)
        definition = ALL_DEFINITIONS.get(item["dataset"])
        if item["empty"] and definition and not definition.allow_empty:
            errors.append(f"{item['dataset']}: {item['empty']} unexpected empty units")
        elif item["empty"]:
            warnings.append(f"{item['dataset']}: {item['empty']} allowed empty units")
        datasets.append(item)

    successful_rows = checkpoint.successful()
    selected_rows = select_current_reference_units(
        successful_rows, snapshot_end=effective_snapshot_end
    )
    selected_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        selected_by_dataset.setdefault(str(row["dataset"]), []).append(row)

    missing_files = 0
    bad_checksums = 0
    for row in successful_rows:
        path = data_root / row["output_path"]
        if not path.exists():
            missing_files += 1
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            bad_checksums += 1
    if missing_files:
        errors.append(f"{missing_files} successful unit files are missing")
    if bad_checksums:
        errors.append(f"{bad_checksums} successful unit files failed checksum validation")

    duplicate_checks: dict[str, int] = {}
    completeness_checks: dict[str, int] = {}
    connection = duckdb.connect()
    try:
        for dataset, rows in sorted(selected_by_dataset.items()):
            definition = ALL_DEFINITIONS.get(dataset)
            if not (definition and definition.primary_key) and dataset not in COVERAGE_DATASETS:
                continue
            unit_dir = data_root / "units" / dataset
            if not unit_dir.exists() or not any(unit_dir.glob("*.parquet")):
                continue
            selected_files = sorted(
                {
                    str((data_root / str(row["output_path"])).resolve())
                    for row in rows
                    if (data_root / str(row["output_path"])).exists()
                }
            )
            if not selected_files:
                continue
            connection.execute("DROP TABLE IF EXISTS selected_unit_files")
            connection.execute("CREATE TEMP TABLE selected_unit_files(path VARCHAR PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO selected_unit_files VALUES (?)",
                [(path,) for path in selected_files],
            )
            glob = str((unit_dir / "*.parquet").resolve()).replace("'", "''")
            columns = {
                str(row[0])
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{glob}', union_by_name=true)"
                ).fetchall()
            }
            primary_key = definition.primary_key if definition else ()
            if not primary_key:
                primary_key = next(
                    (
                        candidate
                        for candidate in coverage_primary_key_candidates(dataset)
                        if set(candidate) <= columns
                    ),
                    (),
                )
            if not primary_key:
                errors.append(
                    f"{dataset}: no supported primary key is present in provider columns"
                )
                continue
            missing_key_columns = sorted(set(primary_key) - columns)
            if missing_key_columns:
                errors.append(
                    f"{dataset}: primary-key columns are missing: "
                    f"{', '.join(missing_key_columns)}"
                )
                continue
            key = ",".join(f'"{column}"' for column in primary_key)
            duplicates = connection.execute(
                f"""
                SELECT count(*) - count(DISTINCT ({key}))
                FROM read_parquet(
                    '{glob}', union_by_name=true, filename=true
                ) AS materialized
                INNER JOIN selected_unit_files AS selected
                    ON materialized.filename = selected.path
                """
            ).fetchone()[0]
            duplicate_checks[dataset] = int(duplicates)
            if duplicates:
                errors.append(f"{dataset}: {duplicates} duplicate primary-key rows")

        completeness_errors, completeness_checks = _verify_daily_completeness(
            connection,
            selected_by_dataset,
            data_root,
            snapshot_end=effective_snapshot_end,
        )
        errors.extend(completeness_errors)
    finally:
        connection.close()

    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "ok": not errors,
        "datasets": datasets,
        "duplicate_checks": duplicate_checks,
        "completeness_checks": completeness_checks,
        "errors": errors,
        "warnings": warnings,
    }


def _verify_daily_completeness(
    connection: duckdb.DuckDBPyConnection,
    selected_by_dataset: dict[str, list[dict[str, Any]]],
    data_root: Path,
    *,
    snapshot_end: date,
) -> tuple[list[str], dict[str, int]]:
    """Prove that open dates and daily stock cross-sections are complete."""

    errors: list[str] = []
    checks = {
        "open_trading_days": 0,
        "daily_trading_days": 0,
        "missing_trading_days": 0,
        "daily_rows": 0,
        "daily_basic_rows": 0,
        "stocks_missing_daily_quotes": 0,
        "stocks_missing_daily_basic": 0,
    }
    paths = {
        dataset: _selected_parquet_paths(selected_by_dataset, dataset, data_root)
        for dataset in ("trade_cal", "daily", "daily_basic")
    }
    if paths["trade_cal"] and paths["daily"]:
        calendar = _parquet_relation(paths["trade_cal"])
        daily = _parquet_relation(paths["daily"])
        calendar_date = _date_sql("cal_date")
        trade_date = _date_sql("trade_date")
        open_days_sql = f"""
            SELECT DISTINCT {calendar_date} AS trade_date
            FROM {calendar}
            WHERE lower(CAST(is_open AS VARCHAR)) IN ('1', 'true', 't', 'yes')
              AND {calendar_date} <= DATE {_sql_string(snapshot_end.isoformat())}
        """
        daily_days_sql = f"""
            SELECT DISTINCT {trade_date} AS trade_date
            FROM {daily}
            WHERE {trade_date} IS NOT NULL
              AND {trade_date} <= DATE {_sql_string(snapshot_end.isoformat())}
        """
        checks["open_trading_days"] = int(
            connection.execute(f"SELECT count(*) FROM ({open_days_sql})").fetchone()[0]
        )
        checks["daily_trading_days"] = int(
            connection.execute(f"SELECT count(*) FROM ({daily_days_sql})").fetchone()[0]
        )
        missing_days_sql = (
            f"SELECT trade_date FROM ({open_days_sql}) "
            f"EXCEPT SELECT trade_date FROM ({daily_days_sql})"
        )
        missing_days = int(
            connection.execute(f"SELECT count(*) FROM ({missing_days_sql})").fetchone()[0]
        )
        checks["missing_trading_days"] = missing_days
        if missing_days:
            sample = ", ".join(
                str(row[0])
                for row in connection.execute(
                    f"SELECT trade_date FROM ({missing_days_sql}) ORDER BY trade_date LIMIT 10"
                ).fetchall()
            )
            errors.append(
                f"daily: {missing_days} open trading days have no quotes (sample: {sample})"
            )

    if paths["daily"] and paths["daily_basic"]:
        daily = _parquet_relation(paths["daily"])
        daily_basic = _parquet_relation(paths["daily_basic"])
        trade_date = _date_sql("trade_date")
        daily_keys = f"""
            SELECT DISTINCT ts_code, {trade_date} AS trade_date
            FROM {daily}
            WHERE ts_code IS NOT NULL AND {trade_date} IS NOT NULL
              AND {trade_date} <= DATE {_sql_string(snapshot_end.isoformat())}
        """
        basic_keys = f"""
            SELECT DISTINCT ts_code, {trade_date} AS trade_date
            FROM {daily_basic}
            WHERE ts_code IS NOT NULL AND {trade_date} IS NOT NULL
              AND {trade_date} <= DATE {_sql_string(snapshot_end.isoformat())}
        """
        checks["daily_rows"] = int(
            connection.execute(f"SELECT count(*) FROM ({daily_keys})").fetchone()[0]
        )
        checks["daily_basic_rows"] = int(
            connection.execute(f"SELECT count(*) FROM ({basic_keys})").fetchone()[0]
        )
        missing_quotes_sql = (
            f"SELECT ts_code, trade_date FROM ({basic_keys}) "
            f"EXCEPT SELECT ts_code, trade_date FROM ({daily_keys})"
        )
        missing_quotes = int(
            connection.execute(f"SELECT count(*) FROM ({missing_quotes_sql})").fetchone()[0]
        )
        checks["stocks_missing_daily_quotes"] = missing_quotes
        if missing_quotes:
            sample = _key_sample(connection, missing_quotes_sql)
            errors.append(
                f"daily: {missing_quotes} stock/date quotes are missing versus daily_basic "
                f"(sample: {sample})"
            )

        missing_basic_sql = (
            f"SELECT ts_code, trade_date FROM ({daily_keys}) "
            f"EXCEPT SELECT ts_code, trade_date FROM ({basic_keys})"
        )
        missing_basic = int(
            connection.execute(f"SELECT count(*) FROM ({missing_basic_sql})").fetchone()[0]
        )
        checks["stocks_missing_daily_basic"] = missing_basic
        if missing_basic:
            sample = _key_sample(connection, missing_basic_sql)
            errors.append(
                f"daily_basic: {missing_basic} stock/date rows are missing versus daily "
                f"(sample: {sample})"
            )
    return errors, checks


def _selected_parquet_paths(
    selected_by_dataset: dict[str, list[dict[str, Any]]],
    dataset: str,
    data_root: Path,
) -> list[str]:
    return sorted(
        {
            str((data_root / str(row["output_path"])).resolve())
            for row in selected_by_dataset.get(dataset, [])
            if str(row.get("output_path") or "").endswith(".parquet")
            and (data_root / str(row["output_path"])).exists()
        }
    )


def _parquet_relation(paths: list[str]) -> str:
    quoted = ",".join(_sql_string(path) for path in paths)
    return f"read_parquet([{quoted}], union_by_name=true)"


def _date_sql(column: str) -> str:
    identifier = '"' + column.replace('"', '""') + '"'
    return (
        f"coalesce(try_cast({identifier} AS DATE), "
        f"try_strptime(CAST({identifier} AS VARCHAR), '%Y%m%d')::DATE)"
    )


def _key_sample(connection: duckdb.DuckDBPyConnection, query: str) -> str:
    return ", ".join(
        f"{row[0]}@{row[1]}"
        for row in connection.execute(
            f"SELECT ts_code, trade_date FROM ({query}) "
            "ORDER BY trade_date, ts_code LIMIT 10"
        ).fetchall()
    )


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
