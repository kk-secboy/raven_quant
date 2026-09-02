from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

from .catalog import (
    ALL_DEFINITIONS,
    GLOBAL_REFERENCE_CALENDARS,
    GLOBAL_REFERENCE_DATASETS,
)
from .checkpoint import CheckpointStore
from .coverage_data import COVERAGE_DATASETS, coverage_primary_key_candidates
from .execution_contract import (
    MINUTE_AMOUNT_ROUNDING_TOLERANCE_CNY,
    MINUTE_CANONICALIZATION_POLICY_VERSION,
    MINUTE_PRICE_TICK_TOLERANCE_CNY,
    MINUTE_VWAP_RELATIVE_TOLERANCE,
    SIMULATION_MINUTE_SOURCE_DATASETS,
    TUSHARE_HAND_SIZE,
)
from .history_bounds import BSE_GOVERNED_HISTORY_START
from .release_window import select_release_window_units, summarize_release_plan
from .row_identity import (
    LATEST_GENERATION_KEYS,
    SNAPSHOT_QUARANTINE_KEYS,
    provider_row_completeness_sql,
    semantic_provider_columns,
)

# Interfaces whose provider pagination reorders rows between pages, whose
# intraday snapshots drift between polls, or whose requested date partitions
# overlap at their boundaries. Snapshot builds already deduplicate identical
# provider fields while retaining the earliest ingestion timestamp, so only
# semantic duplicates for these datasets can be warnings.
UNSTABLE_PAGINATION_DATASETS = frozenset(
    {
        "share_float",
        "ccass_hold",
        "ccass_hold_detail",
        "dc_member",
        "dc_hot",
        "eco_cal",
        # An explicit-field contract can overlap an older provider-default
        # request for the same ETF session. Both immutable source units remain
        # selected and are compared here: exact semantic rows are safe to
        # collapse in snapshots, while any price/factor disagreement blocks.
        "fund_adj",
        "fund_daily",
        "moneyflow_ind_dc",
        "irm_qa_sh",
        "irm_qa_sz",
        "research_report",
        "us_tbr",
        "us_tltr",
        "us_trltr",
        "us_trycr",
        # Measured on production 2026-09-02: exact-duplicate rows from
        # overlapping date pages / repeated pulls, with any residual semantic
        # conflicts adjudicated by LATEST_GENERATION_KEYS below.
        # hk_tradecal/us_tradecal: 925 exact duplicates each (re-pulled
        # calendars), zero semantic variants.
        "hk_tradecal",
        "us_tradecal",
        # us_tycr: yearly window boundaries overlap; earlier duplicate primary
        # keys were exact repeats of the same provider row.
        "us_tycr",
        # us_daily: 69 exact duplicates plus generational revisions.
        # hk_daily: daily-paged interface; boundary overlaps are possible even
        # though the 2026-09-02 probe found none.
        # hk_daily_adj: 732,486 exact duplicates plus generational repricing.
        # us_daily_adj: generational repricing after corporate actions.
        "us_daily",
        "hk_daily",
        "us_daily_adj",
        "hk_daily_adj",
    }
)

# Tushare daily_basic has a complete Beijing Stock Exchange cross-section from
# 2023 onward.  Earlier BJ history is sparse (including entire missing years)
# even though the daily quotes endpoint backfills those securities.  Keep that
# audited provider gap outside the cross-dataset completeness contract instead
# of treating unavailable history as an investable-data failure.
DAILY_BASIC_BSE_COMPLETE_FROM = BSE_GOVERNED_HISTORY_START
# A handful of isolated provider holes can be safely masked by factor/rebalance
# eligibility, but a systematic cross-section gap must still block publication.
DAILY_BASIC_HARD_MISSING_RATE = 0.0001


def verify_downloads(
    checkpoint: CheckpointStore,
    data_root: Path,
    *,
    snapshot_start: date | None = None,
    snapshot_end: date | None = None,
    require_all_planned: bool = True,
    dataset_filter: set[str] | frozenset[str] | None = None,
    required_datasets: set[str] | frozenset[str] | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Verify durable files and the exact generation a successor snapshot will use.

    Production verification and snapshot publication must require every active
    unit in their plan to be complete.  ``require_all_planned=False`` remains
    available only to narrowly scoped stages that provide ``dataset_filter``
    and separately prove their exact work-unit plan.  The report always records
    whether the inspected dataset plan is complete, even when such a bounded
    caller treats unrelated incompleteness as a warning.  Duplicate checks use
    the same current-generation selector as snapshot building; retained old
    reference generations and provider-capped page groups therefore remain
    auditable without being counted twice.
    """

    if not require_all_planned and dataset_filter is None:
        raise ValueError(
            "relaxed verification requires an explicit dataset_filter and cannot "
            "authorize an unscoped production snapshot"
        )
    if (
        required_datasets is not None
        and dataset_filter is not None
        and not set(required_datasets) <= set(dataset_filter)
    ):
        raise ValueError("required_datasets must be a subset of dataset_filter")
    effective_snapshot_end = snapshot_end or date.today()
    effective_required = (
        set(required_datasets)
        if required_datasets is not None
        else set(dataset_filter or ())
    )
    active_rows = checkpoint.active_units(
        set(dataset_filter) if dataset_filter is not None else None
    )
    selection = select_release_window_units(
        active_rows,
        snapshot_start=snapshot_start,
        snapshot_end=effective_snapshot_end,
        datasets=dataset_filter,
        profile=profile,
    )
    datasets: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    incomplete_datasets: list[str] = []
    observed_datasets: set[str] = set()
    for row in summarize_release_plan(selection.rows):
        item = dict(row)
        observed_datasets.add(str(item["dataset"]))
        plan_complete = int(item["succeeded"] or 0) == int(item["planned"] or 0)
        item["plan_status"] = "complete" if plan_complete else "incomplete"
        if not plan_complete:
            incomplete_datasets.append(str(item["dataset"]))
            message = f"{item['dataset']}: {item['succeeded']}/{item['planned']} units succeeded"
            (errors if require_all_planned else warnings).append(message)
        unexpected_empty = int(item.get("unexpected_empty") or 0)
        allowed_empty = int(item.get("allowed_empty") or 0)
        if unexpected_empty:
            errors.append(
                f"{item['dataset']}: {unexpected_empty} unexpected empty units"
            )
        if allowed_empty:
            warnings.append(f"{item['dataset']}: {allowed_empty} allowed empty units")
        if (
            profile == "research-assets"
            and item["dataset"] == "research_report"
            and int(item.get("rows") or 0) == 0
        ):
            errors.append(
                "research_report: the isolated research-asset source is empty; "
                "provider entitlement and downloadable metadata are not proven"
            )
        datasets.append(item)

    missing_planned_datasets = sorted(effective_required - observed_datasets)
    if missing_planned_datasets:
        errors.append(
            "required snapshot datasets have no active plan in the release window: "
            + ", ".join(missing_planned_datasets)
        )

    selected_rows = [
        dict(row) for row in selection.rows if str(row.get("status")) == "succeeded"
    ]
    selected_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        selected_by_dataset.setdefault(str(row["dataset"]), []).append(row)

    missing_files = 0
    bad_checksums = 0
    for row in selected_rows:
        output_path = str(row.get("output_path") or "")
        path = data_root / output_path
        if not output_path or not path.is_file():
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
    conflicting_duplicate_checks: dict[str, int] = {}
    quarantined_conflict_checks: dict[str, int] = {}
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
                errors.append(f"{dataset}: no supported primary key is present in provider columns")
                continue
            missing_key_columns = sorted(set(primary_key) - columns)
            if missing_key_columns:
                errors.append(
                    f"{dataset}: primary-key columns are missing: {', '.join(missing_key_columns)}"
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
            if duplicates and dataset in UNSTABLE_PAGINATION_DATASETS:
                provider_columns = ", ".join(
                    f'materialized."{column.replace(chr(34), chr(34) * 2)}"'
                    for column in sorted(semantic_provider_columns(dataset, columns))
                )
                deduplicated_sql = f"""
                    SELECT DISTINCT {provider_columns}
                    FROM read_parquet(
                        '{glob}', union_by_name=true, filename=true
                    ) AS materialized
                    INNER JOIN selected_unit_files AS selected
                        ON materialized.filename = selected.path
                """
                conflicting_duplicates = int(
                    connection.execute(
                        f"""
                        SELECT count(*) - count(DISTINCT ({key}))
                        FROM ({deduplicated_sql}) AS deduplicated
                        """
                    ).fetchone()[0]
                )
                conflicting_duplicate_checks[dataset] = conflicting_duplicates
                if conflicting_duplicates:
                    conflicting_keys = int(
                        connection.execute(
                            f"""
                            SELECT count(*)
                            FROM (
                                SELECT {key}
                                FROM ({deduplicated_sql}) AS deduplicated
                                GROUP BY {key}
                                HAVING count(*) > 1
                            )
                            """
                        ).fetchone()[0]
                    )
                if conflicting_duplicates and dataset in LATEST_GENERATION_KEYS:
                    unresolved = _latest_generation_unresolved_keys(
                        connection, dataset, glob, columns
                    )
                    resolved = conflicting_keys - unresolved
                    if resolved:
                        warnings.append(
                            f"{dataset}: {resolved} conflicting business keys resolve "
                            "to the latest ingestion generation; the snapshot keeps "
                            "the newest row per key"
                        )
                    if unresolved:
                        errors.append(
                            f"{dataset}: {unresolved} conflicting business keys have "
                            "no unique latest-generation row (true same-generation "
                            "conflict; publication blocked)"
                        )
                elif conflicting_duplicates:
                    if dataset in SNAPSHOT_QUARANTINE_KEYS:
                        quarantined_conflict_checks[dataset] = conflicting_keys
                        warnings.append(
                            f"{dataset}: {conflicting_keys} conflicting business keys "
                            f"({conflicting_duplicates} extra semantic variants) are "
                            "quarantined from successor snapshots"
                        )
                    else:
                        errors.append(
                            f"{dataset}: {conflicting_keys} conflicting business keys "
                            f"({conflicting_duplicates} extra semantic variants) remain "
                            "after exact-row deduplication"
                        )
                else:
                    warnings.append(
                        f"{dataset}: {duplicates} exact duplicate primary-key rows "
                        "(provider paginates this interface with an unstable sort order "
                        "or drifts intraday snapshots between polls, or requested windows "
                        "overlap at their boundaries; snapshot semantic-row "
                        "deduplication removes the identical provider rows)"
                    )
            elif duplicates:
                errors.append(f"{dataset}: {duplicates} duplicate primary-key rows")

        completeness_errors, completeness_warnings, completeness_checks = (
            _verify_daily_completeness(
                connection,
                selected_by_dataset,
                data_root,
                snapshot_start=snapshot_start,
                snapshot_end=effective_snapshot_end,
                require_end_boundary=snapshot_end is not None,
            )
        )
        errors.extend(completeness_errors)
        warnings.extend(completeness_warnings)
        ohlc_errors, ohlc_warnings, ohlc_checks = _verify_daily_ohlc(
            connection,
            selected_by_dataset,
            data_root,
            snapshot_end=effective_snapshot_end,
        )
        errors.extend(ohlc_errors)
        warnings.extend(ohlc_warnings)
        global_errors, global_warnings, global_reference_checks = (
            _verify_global_reference_frames(
                connection,
                selected_by_dataset,
                data_root,
                snapshot_end=effective_snapshot_end,
            )
        )
        errors.extend(global_errors)
        warnings.extend(global_warnings)
        minute_errors, minute_warnings, minute_daily_checks = _verify_minute_daily_consistency(
            connection,
            selected_by_dataset,
            data_root,
            snapshot_end=effective_snapshot_end,
        )
        errors.extend(minute_errors)
        warnings.extend(minute_warnings)
        minute_source_audits: dict[str, dict[str, Any]] = {}
        ashare_5m_paths = _selected_parquet_paths(
            selected_by_dataset, "ashare_5m", data_root
        )
        if ashare_5m_paths:
            source_errors, source_warnings, source_audit = _verify_ashare_5m_relation(
                connection,
                _parquet_relation(ashare_5m_paths),
                snapshot_start=snapshot_start,
                snapshot_end=effective_snapshot_end,
            )
            errors.extend(source_errors)
            warnings.extend(source_warnings)
            minute_source_audits["ashare_5m"] = source_audit
        disclosure_warnings, disclosure_checks = _verify_disclosure_reconciliation(
            connection,
            selected_by_dataset,
            data_root,
        )
        warnings.extend(disclosure_warnings)
    finally:
        connection.close()

    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "ok": not errors,
        "plan_gate": {
            "status": (
                "block"
                if missing_planned_datasets
                else (
                    "pass"
                    if not incomplete_datasets
                    else ("block" if require_all_planned else "warning")
                )
            ),
            "require_all_planned": require_all_planned,
            "incomplete_datasets": sorted(incomplete_datasets),
            "missing_planned_datasets": missing_planned_datasets,
            "dataset_filter": sorted(dataset_filter) if dataset_filter is not None else None,
            "required_datasets": sorted(effective_required),
        },
        "release_window": selection.report(),
        "datasets": datasets,
        "duplicate_checks": duplicate_checks,
        "conflicting_duplicate_checks": conflicting_duplicate_checks,
        "quarantined_conflict_checks": quarantined_conflict_checks,
        "completeness_checks": completeness_checks,
        "ohlc_checks": ohlc_checks,
        "global_reference_checks": global_reference_checks,
        "minute_daily_checks": minute_daily_checks,
        "minute_source_audits": minute_source_audits,
        "disclosure_checks": disclosure_checks,
        "errors": errors,
        "warnings": warnings,
    }


def _verify_daily_completeness(
    connection: duckdb.DuckDBPyConnection,
    selected_by_dataset: dict[str, list[dict[str, Any]]],
    data_root: Path,
    *,
    snapshot_start: date | None,
    snapshot_end: date,
    require_end_boundary: bool,
) -> tuple[list[str], list[str], dict[str, int | float]]:
    """Prove that open dates and daily stock cross-sections are complete."""

    errors: list[str] = []
    warnings: list[str] = []
    checks = {
        "open_trading_days": 0,
        "daily_trading_days": 0,
        "missing_trading_days": 0,
        "calendar_start_covered": 0,
        "calendar_end_covered": 0,
        "daily_rows": 0,
        "daily_basic_rows": 0,
        "stocks_missing_daily_quotes": 0,
        "stocks_missing_daily_basic": 0,
        "daily_rows_outside_daily_basic_history": 0,
        "stocks_missing_daily_basic_rate": 0.0,
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
        calendar_bounds = connection.execute(
            f"SELECT min({calendar_date}), max({calendar_date}) FROM {calendar} "
            f"WHERE {calendar_date} IS NOT NULL"
        ).fetchone()
        calendar_min = calendar_bounds[0] if calendar_bounds is not None else None
        calendar_max = calendar_bounds[1] if calendar_bounds is not None else None
        if snapshot_start is not None:
            checks["calendar_start_covered"] = int(
                calendar_min is not None and calendar_min <= snapshot_start
            )
            if not checks["calendar_start_covered"]:
                errors.append(
                    "trade_cal: selected history does not reach snapshot_start "
                    f"{snapshot_start.isoformat()}"
                )
        if require_end_boundary:
            checks["calendar_end_covered"] = int(
                calendar_max is not None and calendar_max >= snapshot_end
            )
            if not checks["calendar_end_covered"]:
                errors.append(
                    "trade_cal: selected history does not reach snapshot_end "
                    f"{snapshot_end.isoformat()}"
                )
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
        # B-share codes (200xxx.SZ, 900xxx.SH) are outside the product's market
        # scope (A-shares and exchange-traded ETFs only) and the provider's
        # per-date coverage for them is inherently incomplete, so the
        # cross-dataset completeness check is defined on the A-share universe.
        b_share_filter = "left(ts_code, 3) NOT IN ('200', '900')"
        # Codes absent from the security master (upstream ghost rows that have
        # no quotes anywhere) cannot be required to have daily quotes either.
        master_paths = _selected_parquet_paths(selected_by_dataset, "stock_basic", data_root)
        master_filter = "1 = 1"
        if master_paths:
            master = _parquet_relation(master_paths)
            master_filter = f"ts_code IN (SELECT DISTINCT ts_code FROM {master})"
        daily_all_keys = f"""
            SELECT DISTINCT ts_code, {trade_date} AS trade_date
            FROM {daily}
            WHERE ts_code IS NOT NULL AND {trade_date} IS NOT NULL
              AND {b_share_filter}
              AND {master_filter}
              AND {trade_date} <= DATE {_sql_string(snapshot_end.isoformat())}
        """
        basic_all_keys = f"""
            SELECT DISTINCT ts_code, {trade_date} AS trade_date
            FROM {daily_basic}
            WHERE ts_code IS NOT NULL AND {trade_date} IS NOT NULL
              AND {b_share_filter}
              AND {master_filter}
              AND {trade_date} <= DATE {_sql_string(snapshot_end.isoformat())}
        """
        supported_history_filter = (
            "right(ts_code, 3) <> '.BJ' OR trade_date >= DATE "
            f"{_sql_string(DAILY_BASIC_BSE_COMPLETE_FROM.isoformat())}"
        )
        daily_keys = (
            f"SELECT ts_code, trade_date FROM ({daily_all_keys}) WHERE {supported_history_filter}"
        )
        basic_keys = (
            f"SELECT ts_code, trade_date FROM ({basic_all_keys}) WHERE {supported_history_filter}"
        )
        checks["daily_rows_outside_daily_basic_history"] = int(
            connection.execute(
                f"SELECT count(*) FROM ({daily_all_keys}) WHERE NOT ({supported_history_filter})"
            ).fetchone()[0]
        )
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
        missing_basic_rate = (
            missing_basic / int(checks["daily_rows"]) if checks["daily_rows"] else 1.0
        )
        checks["stocks_missing_daily_basic_rate"] = missing_basic_rate
        if missing_basic:
            sample = _key_sample(connection, missing_basic_sql)
            message = (
                f"daily_basic: {missing_basic} stock/date rows are missing versus daily "
                f"(sample: {sample})"
            )
            if missing_basic_rate > DAILY_BASIC_HARD_MISSING_RATE:
                errors.append(message)
            else:
                warnings.append(
                    f"{message}; missing rate {missing_basic_rate:.6%} is below the "
                    f"{DAILY_BASIC_HARD_MISSING_RATE:.2%} blocking threshold"
                )
    return errors, warnings, checks


def _verify_daily_ohlc(
    connection: duckdb.DuckDBPyConnection,
    selected_by_dataset: dict[str, list[dict[str, Any]]],
    data_root: Path,
    *,
    snapshot_end: date,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Check OHLC relationships and adjustment-factor coverage on selected daily units."""

    errors: list[str] = []
    warnings: list[str] = []
    checks = {
        "daily_ohlc_rows": 0,
        "daily_nonpositive_price_rows": 0,
        "daily_high_below_low_rows": 0,
        "daily_open_close_outside_range_rows": 0,
        "daily_missing_adj_factor_keys": 0,
        "daily_large_pct_chg_rows": 0,
    }
    daily_paths = _selected_parquet_paths(selected_by_dataset, "daily", data_root)
    if not daily_paths:
        return errors, warnings, checks
    daily = _parquet_relation(daily_paths)
    trade_date = _date_sql("trade_date")
    daily_columns = {
        str(row[0]) for row in connection.execute(f"DESCRIBE SELECT * FROM {daily}").fetchall()
    }
    # Referencing an absent pct_chg binds to the SELECT alias itself and fails,
    # so fall back to NULL and simply produce no jump warnings on sparse fixtures.
    pct_chg_select = (
        "try_cast(pct_chg AS DOUBLE) AS pct_chg"
        if "pct_chg" in daily_columns
        else "NULL AS pct_chg"
    )
    daily_sql = f"""
        SELECT ts_code, {trade_date} AS trade_date,
               try_cast(open AS DOUBLE) AS open,
               try_cast(high AS DOUBLE) AS high,
               try_cast(low AS DOUBLE) AS low,
               try_cast(close AS DOUBLE) AS close,
               {pct_chg_select}
        FROM {daily}
        WHERE ts_code IS NOT NULL AND {trade_date} IS NOT NULL
          AND {trade_date} <= DATE {_sql_string(snapshot_end.isoformat())}
    """
    checks["daily_ohlc_rows"] = int(
        connection.execute(f"SELECT count(*) FROM ({daily_sql})").fetchone()[0]
    )
    violations = {
        "daily_nonpositive_price_rows": (
            "open <= 0 OR high <= 0 OR low <= 0 OR close <= 0",
            "non-positive OHLC prices",
        ),
        "daily_high_below_low_rows": ("high < low", "high below low"),
        "daily_open_close_outside_range_rows": (
            "open > high OR open < low OR close > high OR close < low",
            "open/close outside the [low, high] range",
        ),
    }
    for check, (predicate, label) in violations.items():
        query = f"SELECT ts_code, trade_date FROM ({daily_sql}) WHERE {predicate}"
        count = int(connection.execute(f"SELECT count(*) FROM ({query})").fetchone()[0])
        checks[check] = count
        if count:
            errors.append(
                f"daily: {count} rows have {label} (sample: {_key_sample(connection, query)})"
            )

    adj_paths = _selected_parquet_paths(selected_by_dataset, "adj_factor", data_root)
    if adj_paths:
        adj = _parquet_relation(adj_paths)
        adj_sql = f"""
            SELECT ts_code, {trade_date} AS trade_date,
                   try_cast(adj_factor AS DOUBLE) AS adj_factor
            FROM {adj}
        """
        missing_sql = f"""
            SELECT d.ts_code, d.trade_date
            FROM ({daily_sql}) d
            LEFT JOIN ({adj_sql}) a
              ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
            WHERE a.adj_factor IS NULL
        """
    else:
        missing_sql = f"SELECT ts_code, trade_date FROM ({daily_sql})"
    missing_adj = int(connection.execute(f"SELECT count(*) FROM ({missing_sql})").fetchone()[0])
    checks["daily_missing_adj_factor_keys"] = missing_adj
    if missing_adj:
        errors.append(
            f"daily: {missing_adj} stock/date keys have no adjustment factor "
            f"(sample: {_key_sample(connection, missing_sql)})"
        )

    jumps_sql = f"SELECT ts_code, trade_date FROM ({daily_sql}) WHERE abs(pct_chg) > 35.0"
    jumps = int(connection.execute(f"SELECT count(*) FROM ({jumps_sql})").fetchone()[0])
    checks["daily_large_pct_chg_rows"] = jumps
    if jumps:
        warnings.append(
            f"daily: {jumps} rows move more than 35% in one session "
            f"(board rules differ; review the price-limit data, sample: "
            f"{_key_sample(connection, jumps_sql)})"
        )
    return errors, warnings, checks



# Peripheral global-reference datasets (us/hk dailies, global indexes, US
# treasury yields, us/hk trade calendars): primary-key duplicates are already
# covered by the generic catalog-driven check; this layer adds date and OHLC
# sanity so a peripheral snapshot cannot publish unparseable, future-dated or
# structurally impossible rows.
GLOBAL_REFERENCE_OHLC_DATASETS = frozenset(
    {"us_daily", "us_daily_adj", "hk_daily", "hk_daily_adj", "index_global"}
)


def _verify_global_reference_frames(
    connection: duckdb.DuckDBPyConnection,
    selected_by_dataset: dict[str, list[dict[str, Any]]],
    data_root: Path,
    *,
    snapshot_end: date,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Date/OHLC sanity for selected peripheral units; fail closed."""

    errors: list[str] = []
    warnings: list[str] = []
    checks = {
        "global_reference_rows": 0,
        "global_reference_bad_date_rows": 0,
        "global_reference_future_rows": 0,
        "global_reference_missing_ohlc_datasets": 0,
        "global_reference_nonpositive_price_rows": 0,
        "global_reference_high_below_low_rows": 0,
        "global_reference_open_close_outside_range_rows": 0,
        "global_reference_large_pct_chg_rows": 0,
    }
    for dataset in sorted(GLOBAL_REFERENCE_DATASETS & set(selected_by_dataset)):
        paths = _selected_parquet_paths(selected_by_dataset, dataset, data_root)
        if not paths:
            continue
        definition = ALL_DEFINITIONS[dataset]
        date_column = str(definition.date_field or "trade_date")
        relation = _parquet_relation(paths)
        columns = {
            str(row[0])
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM {relation}"
            ).fetchall()
        }
        if date_column not in columns:
            errors.append(
                f"{dataset}: availability date column {date_column!r} is missing"
            )
            continue
        date_sql = _date_sql(date_column)
        base = (
            f"SELECT *, {date_sql} AS _parsed_date, "
            f'CAST("{date_column}" AS VARCHAR) AS _raw_date FROM {relation}'
        )
        total = int(connection.execute(f"SELECT count(*) FROM ({base})").fetchone()[0])
        checks["global_reference_rows"] += total
        bad_dates = int(
            connection.execute(
                f"SELECT count(*) FROM ({base}) "
                "WHERE _raw_date IS NULL OR _parsed_date IS NULL"
            ).fetchone()[0]
        )
        checks["global_reference_bad_date_rows"] += bad_dates
        if bad_dates:
            errors.append(
                f"{dataset}: {bad_dates} rows have a missing or unparseable "
                f"{date_column}"
            )
        # Trade calendars legitimately carry future sessions; market data must not.
        if dataset not in GLOBAL_REFERENCE_CALENDARS:
            future = int(
                connection.execute(
                    f"SELECT count(*) FROM ({base}) "
                    f"WHERE _parsed_date > DATE {_sql_string(snapshot_end.isoformat())}"
                ).fetchone()[0]
            )
            checks["global_reference_future_rows"] += future
            if future:
                errors.append(
                    f"{dataset}: {future} rows are dated after the snapshot end"
                )
        if dataset not in GLOBAL_REFERENCE_OHLC_DATASETS:
            continue
        ohlc = {"open", "high", "low", "close"}
        if not ohlc <= columns:
            checks["global_reference_missing_ohlc_datasets"] += 1
            errors.append(
                f"{dataset}: provider columns lack the expected OHLC fields: "
                f"{sorted(ohlc - columns)}"
            )
            continue
        price_sql = f"""
            SELECT {date_sql} AS _parsed_date,
                   try_cast(open AS DOUBLE) AS open,
                   try_cast(high AS DOUBLE) AS high,
                   try_cast(low AS DOUBLE) AS low,
                   try_cast(close AS DOUBLE) AS close
            FROM {relation}
            WHERE {date_sql} IS NOT NULL
        """
        violations = {
            "global_reference_nonpositive_price_rows": (
                "open <= 0 OR high <= 0 OR low <= 0 OR close <= 0",
                "non-positive OHLC prices",
            ),
            "global_reference_high_below_low_rows": ("high < low", "high below low"),
            "global_reference_open_close_outside_range_rows": (
                "open > high OR open < low OR close > high OR close < low",
                "open/close outside the [low, high] range",
            ),
        }
        # Provider noise on OTC/illiquid names (e.g. us_daily reports 0-price
        # rows with non-zero volume for OTC tickers) is quarantined, not
        # fatal: peripheral factor producers consume whitelisted symbols only,
        # so junk rows never enter factor inputs.  Counts stay visible in the
        # report for audit.  Structural problems (missing date column, future
        # rows, provider schema drift) remain errors above.
        for check, (predicate, label) in violations.items():
            count = int(
                connection.execute(
                    f"SELECT count(*) FROM ({price_sql}) WHERE {predicate}"
                ).fetchone()[0]
            )
            checks[check] += count
            if count:
                warnings.append(
                    f"{dataset}: {count} rows have {label} "
                    "(quarantined provider noise; whitelist-only factor consumption)"
                )
        if "pct_chg" in columns:
            jumps = int(
                connection.execute(
                    f"SELECT count(*) FROM {relation} "
                    "WHERE abs(try_cast(pct_chg AS DOUBLE)) > 35.0"
                ).fetchone()[0]
            )
            checks["global_reference_large_pct_chg_rows"] += jumps
            if jumps:
                warnings.append(
                    f"{dataset}: {jumps} rows move more than 35% in one session "
                    "(peripheral boards differ; review before factor use)"
                )
    return errors, warnings, checks

# 设计 §3.5 质量门：日线↔分钟聚合一致性。只比对换算契约明确的股票/ETF
# 分钟数据集（execution_contract.SIMULATION_MINUTE_SOURCE_DATASETS，分钟
# vol 为股、amount 为 CNY）；指数 amount 是成分均价、期货/期权有合约乘数，
# 均不在日线换算契约内。日线 vol 为手（=100 股）、amount 为千元。
MINUTE_DAILY_PRICE_TOLERANCE = 1e-4
MINUTE_DAILY_HARD_MISMATCH_RATE = 0.05
_DAILY_AMOUNT_CNY_PER_UNIT = 1000.0
ASHARE_5M_EXPECTED_BARS_PER_FULL_SESSION = 49
_ASHARE_5M_REQUIRED_COLUMNS = frozenset(
    {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}
)


def _verify_minute_daily_consistency(
    connection: duckdb.DuckDBPyConnection,
    selected_by_dataset: dict[str, list[dict[str, Any]]],
    data_root: Path,
    *,
    snapshot_end: date,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Reconcile minute bars aggregated to daily against the daily dataset.

    对同一 (ts_code, trade_date)：开=首 bar 开、高=max、低=min、收=末 bar
    收、量额=Σ（按换算契约折算成日线单位）。价格相对容差 1e-4；量额折算
    后同容差（分母取 max(|日线值|, 1) 防零）。公共键上的取值矛盾记 error
    （与日线 OHLC 矛盾/重复主键同级）；分钟覆盖缺失只记 coverage 计数，
    不硬失败——分钟本来就是子集/区间下载（1min 流动性子集、5min 全市场
    分区间），反向键（分钟有、日线无）同样只计数。
    """

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, int] = {}
    daily_paths = _selected_parquet_paths(selected_by_dataset, "daily", data_root)
    if not daily_paths:
        return errors, warnings, checks
    daily = _parquet_relation(daily_paths)
    trade_date = _date_sql("trade_date")
    daily_sql = f"""
        SELECT ts_code, {trade_date} AS trade_date,
               try_cast(open AS DOUBLE) AS open,
               try_cast(high AS DOUBLE) AS high,
               try_cast(low AS DOUBLE) AS low,
               try_cast(close AS DOUBLE) AS close,
               try_cast(vol AS DOUBLE) AS vol_hands,
               try_cast(amount AS DOUBLE) AS amount_thousand_cny
        FROM {daily}
        WHERE ts_code IS NOT NULL AND {trade_date} IS NOT NULL
          AND {trade_date} <= DATE {_sql_string(snapshot_end.isoformat())}
    """
    tolerance = MINUTE_DAILY_PRICE_TOLERANCE
    for dataset in sorted(SIMULATION_MINUTE_SOURCE_DATASETS):
        minute_paths = _selected_parquet_paths(selected_by_dataset, dataset, data_root)
        if not minute_paths:
            continue
        minute = _parquet_relation(minute_paths)
        columns = {
            str(row[0]) for row in connection.execute(f"DESCRIBE SELECT * FROM {minute}").fetchall()
        }
        required = {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}
        if not required.issubset(columns):
            warnings.append(
                f"{dataset}: minute units lack the columns required for the daily "
                "aggregation consistency check"
            )
            continue
        stamp = "try_cast(trade_time AS TIMESTAMP)"
        # A small subset of historical stk_mins responses reports volume at
        # 100x the documented share unit. Amount/volume then implies a price
        # exactly 1/100 of the bar price. Canonicalize those rows using the
        # provider's own amount and OHLC envelope before comparing with daily
        # hands. Zero-volume placeholder bars are excluded from OHLC because
        # they can carry stale pre-open or post-close prices.
        normalized_volume = f"""
            CASE
                WHEN try_cast(vol AS DOUBLE) > 0
                 AND try_cast(amount AS DOUBLE) > 0
                 AND try_cast(amount AS DOUBLE) / try_cast(vol AS DOUBLE)
                     NOT BETWEEN try_cast(low AS DOUBLE) * 0.95
                         AND try_cast(high AS DOUBLE) * 1.05
                 AND try_cast(amount AS DOUBLE)
                     / (try_cast(vol AS DOUBLE) / {TUSHARE_HAND_SIZE})
                     BETWEEN try_cast(low AS DOUBLE) * 0.95
                         AND try_cast(high AS DOUBLE) * 1.05
                THEN try_cast(vol AS DOUBLE) / {TUSHARE_HAND_SIZE}
                ELSE try_cast(vol AS DOUBLE)
            END
        """
        active_bar = f"({normalized_volume}) > 0 OR try_cast(amount AS DOUBLE) > 0"
        minute_sql = f"""
            SELECT ts_code,
                   CAST({stamp} AS DATE) AS trade_date,
                   arg_min(
                       CASE WHEN {active_bar} THEN try_cast(open AS DOUBLE) END,
                       {stamp}
                   ) AS open,
                   max(
                       CASE WHEN {active_bar} THEN try_cast(high AS DOUBLE) END
                   ) AS high,
                   min(
                       CASE WHEN {active_bar} THEN try_cast(low AS DOUBLE) END
                   ) AS low,
                   arg_max(
                       CASE WHEN {active_bar} THEN try_cast(close AS DOUBLE) END,
                       {stamp}
                   ) AS close,
                   sum({normalized_volume}) AS vol_shares,
                   sum(try_cast(amount AS DOUBLE)) AS amount_cny
            FROM {minute}
            WHERE ts_code IS NOT NULL AND {stamp} IS NOT NULL
              AND CAST({stamp} AS DATE)
                  <= DATE {_sql_string(snapshot_end.isoformat())}
            GROUP BY ts_code, CAST({stamp} AS DATE)
        """
        compared_sql = f"""
            SELECT m.ts_code, m.trade_date
            FROM ({minute_sql}) m
            INNER JOIN ({daily_sql}) d
              ON m.ts_code = d.ts_code AND m.trade_date = d.trade_date
        """
        compared = int(connection.execute(f"SELECT count(*) FROM ({compared_sql})").fetchone()[0])
        mismatch_predicates = {
            "open": f"abs(m.open - d.open) > {tolerance} * abs(d.open)",
            "high": f"abs(m.high - d.high) > {tolerance} * abs(d.high)",
            "low": f"abs(m.low - d.low) > {tolerance} * abs(d.low)",
            "close": f"abs(m.close - d.close) > {tolerance} * abs(d.close)",
            "volume": (
                f"abs(m.vol_shares / {TUSHARE_HAND_SIZE} - d.vol_hands) "
                f"> {tolerance} * greatest(abs(d.vol_hands), 1)"
            ),
            "amount": (
                f"abs(m.amount_cny / {_DAILY_AMOUNT_CNY_PER_UNIT} "
                f"- d.amount_thousand_cny) "
                f"> {tolerance} * greatest(abs(d.amount_thousand_cny), 1)"
            ),
        }
        mismatch_predicate = " OR ".join(
            f"({predicate})" for predicate in mismatch_predicates.values()
        )
        mismatched_sql = f"{compared_sql} WHERE {mismatch_predicate}"
        mismatch_row = connection.execute(
            f"""
            SELECT
                {
                ", ".join(
                    f"count(*) FILTER (WHERE {predicate})"
                    for predicate in mismatch_predicates.values()
                )
            },
                count(*) FILTER (WHERE {mismatch_predicate})
            FROM ({minute_sql}) m
            INNER JOIN ({daily_sql}) d
              ON m.ts_code = d.ts_code AND m.trade_date = d.trade_date
            """
        )
        *field_counts, mismatched = [int(value) for value in mismatch_row.fetchone()]
        daily_keys_sql = f"SELECT ts_code, trade_date FROM ({daily_sql})"
        minute_keys_sql = f"SELECT ts_code, trade_date FROM ({minute_sql})"
        daily_without_minute = int(
            connection.execute(
                f"SELECT count(*) FROM ({daily_keys_sql} EXCEPT {minute_keys_sql})"
            ).fetchone()[0]
        )
        minute_without_daily = int(
            connection.execute(
                f"SELECT count(*) FROM ({minute_keys_sql} EXCEPT {daily_keys_sql})"
            ).fetchone()[0]
        )
        checks[f"minute_daily_{dataset}_compared_keys"] = compared
        checks[f"minute_daily_{dataset}_mismatched_keys"] = mismatched
        for field, count in zip(mismatch_predicates, field_counts, strict=True):
            checks[f"minute_daily_{dataset}_{field}_mismatches"] = count
        checks[f"minute_daily_{dataset}_daily_keys_without_minute_coverage"] = daily_without_minute
        checks[f"minute_daily_{dataset}_minute_keys_without_daily"] = minute_without_daily
        if mismatched:
            message = (
                f"{dataset}: {mismatched} stock/date keys disagree between the "
                f"aggregated minute bars and the daily dataset "
                f"(sample: {_key_sample(connection, mismatched_sql)})"
            )
            mismatch_rate = mismatched / compared if compared else 1.0
            if mismatch_rate > MINUTE_DAILY_HARD_MISMATCH_RATE:
                errors.append(message)
            else:
                warnings.append(
                    f"{message}; mismatch rate {mismatch_rate:.2%} is below the "
                    f"{MINUTE_DAILY_HARD_MISMATCH_RATE:.0%} blocking threshold"
                )
    return errors, warnings, checks


def verify_ashare_5m_source_files(
    paths: list[Path],
    *,
    snapshot_start: date | None,
    snapshot_end: date,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Audit raw A-share 5-minute files under the publication canonicalizer.

    Provider files are intentionally retained byte-for-byte.  Known provider
    additions (off-grid one-minute rows and post-close/block-trade records) are
    excluded by a deterministic policy rather than aggregated into an already
    present 5-minute bar.  Every exclusion and non-tradable content decision is
    counted and sealed into ``audit_sha256`` so a passing gate cannot hide raw
    source anomalies.
    """

    connection = duckdb.connect()
    try:
        return _verify_ashare_5m_relation(
            connection,
            _parquet_relation([str(path.resolve()) for path in paths]),
            snapshot_start=snapshot_start,
            snapshot_end=snapshot_end,
        )
    finally:
        connection.close()


def _verify_ashare_5m_relation(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    *,
    snapshot_start: date | None,
    snapshot_end: date,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return fail-closed cadence/session/content evidence for ``ashare_5m``."""

    columns = {
        str(row[0])
        for row in connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    }
    missing_columns = sorted(_ASHARE_5M_REQUIRED_COLUMNS - columns)
    policy = {
        "version": MINUTE_CANONICALIZATION_POLICY_VERSION,
        "frequency": "5min",
        "timestamp_semantics": "bar_end_local_asia_shanghai",
        "continuous_session_endpoints": ["09:30", "09:35-11:30", "13:05-15:00"],
        "expected_bars_per_full_session": ASHARE_5M_EXPECTED_BARS_PER_FULL_SESSION,
        "off_grid_policy": "exclude_do_not_resample",
        "outside_session_policy": "exclude_nontradable",
        "severe_content_policy": "retain_price_mark_bar_nontradable",
        "vwap_relative_tolerance": MINUTE_VWAP_RELATIVE_TOLERANCE,
        "amount_rounding_tolerance_cny": MINUTE_AMOUNT_ROUNDING_TOLERANCE_CNY,
        "price_tick_tolerance_cny": MINUTE_PRICE_TICK_TOLERANCE_CNY,
        "hand_size": TUSHARE_HAND_SIZE,
    }
    if missing_columns:
        audit = {
            "dataset": "ashare_5m",
            "policy": policy,
            "missing_columns": missing_columns,
            "source_rows": 0,
            "canonical_rows": 0,
            "audit_status": "block",
        }
        audit["audit_sha256"] = _canonical_audit_sha256(audit)
        return (
            ["ashare_5m: canonical source audit lacks columns: " + ", ".join(missing_columns)],
            [],
            audit,
        )

    start_predicate = (
        f"CAST(stamp AS DATE) >= DATE {_sql_string(snapshot_start.isoformat())} AND "
        if snapshot_start is not None
        else ""
    )
    end_literal = _sql_string(snapshot_end.isoformat())
    relative = MINUTE_VWAP_RELATIVE_TOLERANCE
    amount_tolerance = MINUTE_AMOUNT_ROUNDING_TOLERANCE_CNY
    tick_tolerance = MINUTE_PRICE_TICK_TOLERANCE_CNY

    direct_guard = "vol_d > 0 AND amount_d > 0 AND low_d > 0 AND high_d > 0"
    direct_strict = (
        f"{direct_guard} AND amount_d / vol_d BETWEEN low_d AND high_d"
    )
    direct_relative = f"""
        {direct_guard} AND amount_d / vol_d
            BETWEEN low_d * {1.0 - relative} AND high_d * {1.0 + relative}
    """
    direct_amount_rounding = f"""
        {direct_guard} AND greatest(low_d * vol_d - amount_d,
            amount_d - high_d * vol_d, 0.0) <= {amount_tolerance}
    """
    direct_price_tick = f"""
        {direct_guard} AND greatest(low_d - amount_d / vol_d,
            amount_d / vol_d - high_d, 0.0) <= {tick_tolerance}
    """
    normalized_volume = f"vol_d / {TUSHARE_HAND_SIZE}"
    hand_guard = direct_guard
    hand_strict = f"""
        {hand_guard} AND amount_d / ({normalized_volume}) BETWEEN low_d AND high_d
    """
    hand_relative = f"""
        {hand_guard} AND amount_d / ({normalized_volume})
            BETWEEN low_d * {1.0 - relative} AND high_d * {1.0 + relative}
    """
    hand_amount_rounding = f"""
        {hand_guard} AND greatest(low_d * ({normalized_volume}) - amount_d,
            amount_d - high_d * ({normalized_volume}), 0.0) <= {amount_tolerance}
    """
    hand_price_tick = f"""
        {hand_guard} AND greatest(low_d - amount_d / ({normalized_volume}),
            amount_d / ({normalized_volume}) - high_d, 0.0) <= {tick_tolerance}
    """
    typed_sql = f"""
        SELECT trim(CAST(ts_code AS VARCHAR)) AS ts_code,
               CAST(trade_time AS VARCHAR) AS raw_trade_time,
               try_cast(trade_time AS TIMESTAMP) AS stamp,
               try_cast(open AS DOUBLE) AS open_d,
               try_cast(high AS DOUBLE) AS high_d,
               try_cast(low AS DOUBLE) AS low_d,
               try_cast(close AS DOUBLE) AS close_d,
               try_cast(vol AS DOUBLE) AS vol_d,
               try_cast(amount AS DOUBLE) AS amount_d
        FROM {relation}
    """
    classified_sql = f"""
        WITH typed AS ({typed_sql}), bounded AS (
            SELECT *,
                   extract(hour FROM stamp)::INTEGER * 60
                       + extract(minute FROM stamp)::INTEGER AS minute_of_day
            FROM typed
            WHERE stamp IS NULL OR (
                {start_predicate} CAST(stamp AS DATE) <= DATE {end_literal}
            )
        ), flags AS (
            SELECT *,
                   ts_code IS NOT NULL AND ts_code <> '' AS identity_ok,
                   stamp IS NOT NULL AS timestamp_ok,
                   stamp IS NOT NULL AND (
                       minute_of_day BETWEEN 570 AND 690
                       OR minute_of_day BETWEEN 785 AND 900
                   ) AS session_ok,
                   stamp IS NOT NULL
                       AND date_trunc('minute', stamp) = stamp
                       AND extract(minute FROM stamp)::INTEGER % 5 = 0 AS cadence_ok,
                   open_d IS NOT NULL AND high_d IS NOT NULL
                       AND low_d IS NOT NULL AND close_d IS NOT NULL
                       AND vol_d IS NOT NULL AND amount_d IS NOT NULL
                       AND isfinite(open_d) AND isfinite(high_d)
                       AND isfinite(low_d) AND isfinite(close_d)
                       AND isfinite(vol_d) AND isfinite(amount_d)
                       AND open_d > 0 AND high_d > 0 AND low_d > 0 AND close_d > 0
                       AND high_d >= greatest(open_d, low_d, close_d)
                       AND low_d <= least(open_d, high_d, close_d)
                       AND vol_d >= 0 AND amount_d >= 0 AS numeric_ohlc_ok,
                   ({direct_strict}) AS direct_strict_ok,
                   ({direct_relative}) AS direct_relative_ok,
                   ({direct_amount_rounding}) AS direct_amount_rounding_ok,
                   ({direct_price_tick}) AS direct_price_tick_ok,
                   ({hand_strict}) AS hand_strict_ok,
                   ({hand_relative}) AS hand_relative_ok,
                   ({hand_amount_rounding}) AS hand_amount_rounding_ok,
                   ({hand_price_tick}) AS hand_price_tick_ok
            FROM bounded
        ), decisions AS (
            SELECT *,
                   direct_relative_ok OR direct_amount_rounding_ok
                       OR direct_price_tick_ok AS direct_amount_ok,
                   hand_relative_ok OR hand_amount_rounding_ok
                       OR hand_price_tick_ok AS hand_amount_ok
            FROM flags
        )
        SELECT *,
               identity_ok AND timestamp_ok AND session_ok AND cadence_ok
                   AS canonical_key_ok,
               numeric_ohlc_ok AND (
                   (vol_d = 0 AND amount_d = 0)
                   OR direct_amount_ok OR hand_amount_ok
               ) AS content_ok,
               NOT direct_amount_ok AND hand_amount_ok AS volume_normalized,
               CASE WHEN direct_amount_ok THEN direct_strict_ok
                    ELSE hand_strict_ok END AS selected_strict_ok,
               CASE WHEN direct_amount_ok THEN direct_relative_ok
                    ELSE hand_relative_ok END AS selected_relative_ok,
               CASE WHEN direct_amount_ok THEN direct_amount_rounding_ok
                    ELSE hand_amount_rounding_ok END AS selected_amount_rounding_ok,
               CASE WHEN direct_amount_ok THEN direct_price_tick_ok
                    ELSE hand_price_tick_ok END AS selected_price_tick_ok
        FROM decisions
    """
    key_expr = "ts_code || '@' || strftime(stamp, '%Y-%m-%dT%H:%M:%S.%f')"
    fingerprint_expr = """
        hash(coalesce(ts_code, ''), coalesce(raw_trade_time, ''),
             coalesce(open_d, 'NaN'::DOUBLE), coalesce(high_d, 'NaN'::DOUBLE),
             coalesce(low_d, 'NaN'::DOUBLE), coalesce(close_d, 'NaN'::DOUBLE),
             coalesce(vol_d, 'NaN'::DOUBLE), coalesce(amount_d, 'NaN'::DOUBLE),
             identity_ok, timestamp_ok, session_ok, cadence_ok,
             numeric_ohlc_ok, direct_amount_ok, hand_amount_ok,
             volume_normalized, selected_strict_ok, selected_relative_ok,
             selected_amount_rounding_ok, selected_price_tick_ok)
    """
    summary_row = connection.execute(
        f"""
        SELECT count(*) AS source_rows,
               count(*) FILTER (WHERE NOT identity_ok) AS invalid_identity_rows,
               count(*) FILTER (WHERE NOT timestamp_ok) AS invalid_timestamp_rows,
               count(*) FILTER (WHERE timestamp_ok AND NOT session_ok)
                   AS outside_session_rows,
               count(*) FILTER (WHERE timestamp_ok AND session_ok AND NOT cadence_ok)
                   AS off_cadence_rows,
               count(DISTINCT CAST(stamp AS DATE) || ':' || ts_code)
                   FILTER (WHERE identity_ok AND timestamp_ok AND session_ok
                                   AND NOT cadence_ok) AS offgrid_symbol_days,
               count(*) FILTER (WHERE NOT canonical_key_ok) AS excluded_rows,
               count(*) FILTER (WHERE canonical_key_ok) AS canonical_source_rows,
               count(DISTINCT {key_expr}) FILTER (WHERE canonical_key_ok)
                   AS canonical_rows,
               count(*) FILTER (
                   WHERE canonical_key_ok AND volume_normalized
               ) AS hand_normalized_rows,
               count(*) FILTER (
                   WHERE canonical_key_ok AND content_ok AND NOT selected_strict_ok
                     AND selected_relative_ok
               ) AS relative_fallback_rows,
               count(*) FILTER (
                   WHERE canonical_key_ok AND content_ok AND NOT selected_relative_ok
                     AND selected_amount_rounding_ok
               ) AS amount_rounding_fallback_rows,
               count(*) FILTER (
                   WHERE canonical_key_ok AND content_ok AND NOT selected_relative_ok
                     AND NOT selected_amount_rounding_ok AND selected_price_tick_ok
               ) AS price_tick_fallback_rows,
               count(*) FILTER (WHERE canonical_key_ok AND NOT content_ok)
                   AS nontradable_rows,
               count(DISTINCT CAST(stamp AS DATE) || ':' || ts_code)
                   FILTER (WHERE canonical_key_ok AND NOT content_ok)
                   AS nontradable_symbol_days,
               coalesce(CAST(sum({fingerprint_expr}) AS VARCHAR), '0')
                   AS source_fingerprint_sum
        FROM ({classified_sql})
        """
    ).fetchone()
    names = [item[0] for item in connection.description]
    summary = dict(zip(names, summary_row, strict=True))
    for name, value in list(summary.items()):
        if name != "source_fingerprint_sum":
            summary[name] = int(value or 0)

    duplicate_row = connection.execute(
        f"""
        SELECT coalesce(sum(row_count - 1), 0) AS duplicate_excess_rows,
               count(*) FILTER (WHERE content_variants > 1)
                   AS conflicting_canonical_keys
        FROM (
            SELECT ts_code, stamp, count(*) AS row_count,
                   count(DISTINCT hash(open_d, high_d, low_d, close_d, vol_d, amount_d))
                       AS content_variants
            FROM ({classified_sql})
            WHERE canonical_key_ok
            GROUP BY ts_code, stamp
            HAVING count(*) > 1
        ) duplicates
        """
    ).fetchone()
    summary["duplicate_excess_rows"] = int(duplicate_row[0] or 0)
    summary["conflicting_canonical_keys"] = int(duplicate_row[1] or 0)

    incomplete_sessions = int(
        connection.execute(
            f"""
            SELECT count(*)
            FROM (
                SELECT ts_code, CAST(stamp AS DATE) AS trade_date,
                       count(DISTINCT stamp) AS bars
                FROM ({classified_sql})
                WHERE canonical_key_ok
                GROUP BY ts_code, CAST(stamp AS DATE)
            ) sessions
            WHERE bars <> {ASHARE_5M_EXPECTED_BARS_PER_FULL_SESSION}
            """
        ).fetchone()[0]
    )
    summary["incomplete_symbol_days"] = incomplete_sessions
    summary.update(
        {
            # Builder-aligned names are retained beside the source-audit names
            # so provenance can compare one policy without translating units.
            "input_rows": summary["source_rows"],
            "output_rows": summary["canonical_rows"],
            "session_excluded_rows": summary["outside_session_rows"],
            "offgrid_excluded_rows": summary["off_cadence_rows"],
            "volume_normalized_rows": summary["hand_normalized_rows"],
        }
    )

    errors: list[str] = []
    warnings: list[str] = []
    if summary["source_rows"] == 0 or summary["canonical_rows"] == 0:
        errors.append("ashare_5m: canonical source audit produced no usable 5-minute rows")
    if summary["conflicting_canonical_keys"]:
        errors.append(
            "ashare_5m: "
            f"{summary['conflicting_canonical_keys']} canonical timestamps have "
            "conflicting bar content"
        )

    event_payload = {
        "policy_version": MINUTE_CANONICALIZATION_POLICY_VERSION,
        "source_fingerprint_sum": summary["source_fingerprint_sum"],
        "input_rows": summary["input_rows"],
        "session_excluded_rows": summary["session_excluded_rows"],
        "offgrid_excluded_rows": summary["offgrid_excluded_rows"],
        "volume_normalized_rows": summary["volume_normalized_rows"],
        "relative_fallback_rows": summary["relative_fallback_rows"],
        "amount_rounding_fallback_rows": summary["amount_rounding_fallback_rows"],
        "price_tick_fallback_rows": summary["price_tick_fallback_rows"],
        "nontradable_rows": summary["nontradable_rows"],
    }
    summary_payload = {
        **event_payload,
        "output_rows": summary["output_rows"],
        "offgrid_symbol_days": summary["offgrid_symbol_days"],
        "event_sha256": _canonical_audit_sha256(event_payload),
    }
    audit = {
        "dataset": "ashare_5m",
        "policy_version": MINUTE_CANONICALIZATION_POLICY_VERSION,
        "rules": {key: value for key, value in policy.items() if key != "version"},
        "policy": policy,
        **summary,
        "event_sha256": summary_payload["event_sha256"],
        "summary_sha256": _canonical_audit_sha256(summary_payload),
        "audit_status": "block" if errors else "pass_with_canonicalization",
    }
    audit["audit_sha256"] = _canonical_audit_sha256(audit)
    evidence = audit["audit_sha256"]
    if summary["excluded_rows"]:
        warnings.append(
            "ashare_5m: deterministic session/cadence canonicalization excludes "
            f"{summary['excluded_rows']} raw rows (audit {evidence})"
        )
    if summary["hand_normalized_rows"]:
        warnings.append(
            "ashare_5m: "
            f"{summary['hand_normalized_rows']} rows require the governed 100-share "
            f"volume correction (audit {evidence})"
        )
    if summary["nontradable_rows"]:
        warnings.append(
            "ashare_5m: "
            f"{summary['nontradable_rows']} severe content rows are fail-closed as "
            f"non-tradable (audit {evidence})"
        )
    if summary["duplicate_excess_rows"] and not summary["conflicting_canonical_keys"]:
        warnings.append(
            "ashare_5m: "
            f"{summary['duplicate_excess_rows']} byte-equivalent canonical duplicates "
            f"collapse deterministically (audit {evidence})"
        )
    if incomplete_sessions:
        warnings.append(
            "ashare_5m: "
            f"{incomplete_sessions} symbol-days do not contain all "
            f"{ASHARE_5M_EXPECTED_BARS_PER_FULL_SESSION} continuous-auction endpoints; "
            f"missing bars remain unavailable/non-tradable (audit {evidence})"
        )
    return errors, warnings, audit


def _canonical_audit_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_FINANCIAL_STATEMENT_DATASETS = (
    "income",
    "balancesheet",
    "cashflow",
    "fina_indicator",
    "forecast",
    "express",
)


def _verify_disclosure_reconciliation(
    connection: duckdb.DuckDBPyConnection,
    selected_by_dataset: dict[str, list[dict[str, Any]]],
    data_root: Path,
) -> tuple[list[str], dict[str, int]]:
    """Reconcile financial ann_date rows against the disclosure_date calendar.

    Flag-only (never blocking): rows whose ann_date disagrees with the
    disclosure calendar's actual_date (or planned ann_date when no actual date
    was recorded) are counted as warnings so availability drift is visible
    without failing the download gate.
    """

    warnings: list[str] = []
    checks = {
        "disclosure_calendar_rows": 0,
        "compared_rows": 0,
        "mismatched_ann_date_rows": 0,
        "rows_without_calendar_entry": 0,
    }
    disclosure_paths = _selected_parquet_paths(selected_by_dataset, "disclosure_date", data_root)
    if not disclosure_paths:
        return warnings, checks
    disclosure = _parquet_relation(disclosure_paths)
    calendar_sql = f"""
        SELECT ts_code, {_date_sql("end_date")} AS end_date,
               max(coalesce({_date_sql("actual_date")}, {_date_sql("ann_date")}))
                   AS disclosed_date
        FROM {disclosure}
        WHERE ts_code IS NOT NULL AND {_date_sql("end_date")} IS NOT NULL
        GROUP BY ts_code, {_date_sql("end_date")}
    """
    checks["disclosure_calendar_rows"] = int(
        connection.execute(f"SELECT count(*) FROM ({calendar_sql})").fetchone()[0]
    )
    for dataset in _FINANCIAL_STATEMENT_DATASETS:
        paths = _selected_parquet_paths(selected_by_dataset, dataset, data_root)
        if not paths:
            continue
        relation = _parquet_relation(paths)
        columns = {
            str(row[0])
            for row in connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
        }
        if not {"ts_code", "ann_date", "end_date"}.issubset(columns):
            continue
        financial_sql = f"""
            SELECT ts_code, {_date_sql("ann_date")} AS ann_date,
                   {_date_sql("end_date")} AS end_date
            FROM {relation}
            WHERE ts_code IS NOT NULL
              AND {_date_sql("ann_date")} IS NOT NULL
              AND {_date_sql("end_date")} IS NOT NULL
        """
        compared = connection.execute(
            f"""
            SELECT count(*),
                   count(*) FILTER (f.ann_date <> c.disclosed_date)
            FROM ({financial_sql}) f
            INNER JOIN ({calendar_sql}) c
              ON f.ts_code = c.ts_code AND f.end_date = c.end_date
            """
        ).fetchone()
        unmatched = connection.execute(
            f"""
            SELECT count(*)
            FROM ({financial_sql}) f
            LEFT JOIN ({calendar_sql}) c
              ON f.ts_code = c.ts_code AND f.end_date = c.end_date
            WHERE c.disclosed_date IS NULL
            """
        ).fetchone()[0]
        checks["compared_rows"] += int(compared[0])
        checks["mismatched_ann_date_rows"] += int(compared[1])
        checks["rows_without_calendar_entry"] += int(unmatched)
        if int(compared[1]):
            warnings.append(
                f"{dataset}: {int(compared[1])} rows have ann_date disagreeing with the "
                "disclosure_date calendar (flagged only; availability still uses ann_date)"
            )
    return warnings, checks


def quality_gate_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Compact quality-gate marker stored in snapshot manifests."""

    release = dict(report.get("release_window") or {})
    payload = {
        "ok": bool(report["ok"]),
        "verified_at": str(report["checked_at"]),
        "errors": list(report["errors"]),
    }
    if release:
        plan_scope_sha256 = release.get("plan_scope_sha256") or release.get(
            "scope_sha256"
        )
        payload.update(
            {
                "plan_scope_sha256": plan_scope_sha256,
                # Compatibility for consumers deployed before the plan-scope
                # name made the publication binding explicit.
                "release_window_scope_sha256": plan_scope_sha256,
                "selected_unit_set_sha256": release.get(
                    "selected_unit_set_sha256"
                ),
                "release_window": {
                    "selector_version": release.get("selector_version"),
                    "snapshot_start": release.get("snapshot_start"),
                    "snapshot_end": release.get("snapshot_end"),
                    "profile": release.get("profile"),
                    "requested_datasets": list(
                        release.get("requested_datasets") or []
                    ),
                    "plan_scope_sha256": plan_scope_sha256,
                    "selected_unit_set_sha256": release.get(
                        "selected_unit_set_sha256"
                    ),
                    "selected_unit_count": release.get("selected_unit_count"),
                    "selected_unit_identities": list(
                        release.get("selected_unit_identities") or []
                    ),
                },
            }
        )
    return payload


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


def _latest_generation_unresolved_keys(
    connection: duckdb.DuckDBPyConnection,
    dataset: str,
    glob: str,
    columns: set[str],
) -> int:
    """Count conflicting keys with no unique (generation, completeness) winner.

    A conflict is resolvable when exactly one distinct provider row tops the
    (latest ingested_at, highest provider-field completeness) ranking.  Rows
    that still tie at the top disagree within one generation and stay blocked.
    """

    key = LATEST_GENERATION_KEYS[dataset]
    key_columns = ", ".join(f'"{column}"' for column in key)
    using = ", ".join(f'"{column}"' for column in key)
    completeness = provider_row_completeness_sql(dataset, columns)
    return int(
        connection.execute(
            f"""
            WITH d AS (
                SELECT DISTINCT * EXCLUDE (filename) FROM read_parquet(
                    '{glob}', union_by_name=true, filename=true
                ) AS materialized
                INNER JOIN selected_unit_files AS selected
                    ON materialized.filename = selected.path
            ),
            proj AS (SELECT * EXCLUDE (ingested_at) FROM d),
            conf AS (
                SELECT {key_columns} FROM proj
                GROUP BY {key_columns} HAVING count(*) > 1
            ),
            scored AS (
                SELECT d.*, ({completeness}) AS _comp
                FROM d JOIN conf USING ({using})
            ),
            top_ing AS (
                SELECT {key_columns}, max(ingested_at) AS m_ing
                FROM scored GROUP BY {key_columns}
            ),
            top_rows AS (
                SELECT s.* FROM scored s
                JOIN top_ing USING ({using})
                WHERE s.ingested_at = top_ing.m_ing
            ),
            top_comp AS (
                SELECT {key_columns}, max(_comp) AS m_comp
                FROM top_rows GROUP BY {key_columns}
            )
            SELECT count(*) FROM (
                SELECT {key_columns} FROM (
                    SELECT * EXCLUDE (_comp, ingested_at) FROM top_rows t
                    JOIN top_comp USING ({using})
                    WHERE t._comp = top_comp.m_comp
                ) GROUP BY {key_columns} HAVING count(*) > 1
            )
            """
        ).fetchone()[0]
    )


def _key_sample(connection: duckdb.DuckDBPyConnection, query: str) -> str:
    return ", ".join(
        f"{row[0]}@{row[1]}"
        for row in connection.execute(
            f"SELECT ts_code, trade_date FROM ({query}) ORDER BY trade_date, ts_code LIMIT 10"
        ).fetchall()
    )


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
