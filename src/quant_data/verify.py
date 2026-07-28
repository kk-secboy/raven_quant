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
from .execution_contract import SIMULATION_MINUTE_SOURCE_DATASETS, TUSHARE_HAND_SIZE
from .reference_data import select_current_reference_units

# Interfaces whose provider pagination reorders rows between pages, or whose
# intraday snapshots drift between polls, making overlapping pages (and
# therefore duplicate primary keys) inherent.  Snapshot builds already
# deduplicate identical provider fields while retaining the earliest ingestion
# timestamp, so only semantic duplicates for these datasets can be warnings.
UNSTABLE_PAGINATION_DATASETS = frozenset(
    {
        "share_float",
        "ccass_hold",
        "ccass_hold_detail",
        "dc_member",
        "dc_hot",
        "eco_cal",
        "moneyflow_ind_dc",
        "irm_qa_sh",
        "irm_qa_sz",
    }
)

# Tushare daily_basic has a complete Beijing Stock Exchange cross-section from
# 2023 onward.  Earlier BJ history is sparse (including entire missing years)
# even though the daily quotes endpoint backfills those securities.  Keep that
# audited provider gap outside the cross-dataset completeness contract instead
# of treating unavailable history as an investable-data failure.
DAILY_BASIC_BSE_COMPLETE_FROM = date(2023, 1, 1)
# A handful of isolated provider holes can be safely masked by factor/rebalance
# eligibility, but a systematic cross-section gap must still block publication.
DAILY_BASIC_HARD_MISSING_RATE = 0.0001


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
    conflicting_duplicate_checks: dict[str, int] = {}
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
                    for column in sorted(columns - {"ingested_at"})
                )
                conflicting_duplicates = int(
                    connection.execute(
                        f"""
                        SELECT count(*) - count(DISTINCT ({key}))
                        FROM (
                            SELECT DISTINCT {provider_columns}
                            FROM read_parquet(
                                '{glob}', union_by_name=true, filename=true
                            ) AS materialized
                            INNER JOIN selected_unit_files AS selected
                                ON materialized.filename = selected.path
                        ) AS deduplicated
                        """
                    ).fetchone()[0]
                )
                conflicting_duplicate_checks[dataset] = conflicting_duplicates
                if conflicting_duplicates:
                    errors.append(
                        f"{dataset}: {conflicting_duplicates} conflicting "
                        "primary-key rows remain after exact-row deduplication"
                    )
                else:
                    warnings.append(
                        f"{dataset}: {duplicates} exact duplicate primary-key rows "
                        "(provider paginates this interface with an unstable sort order "
                        "or drifts intraday snapshots between polls; snapshot semantic-row "
                        "deduplication removes the identical provider rows)"
                    )
            elif duplicates:
                errors.append(f"{dataset}: {duplicates} duplicate primary-key rows")

        completeness_errors, completeness_warnings, completeness_checks = (
            _verify_daily_completeness(
                connection,
                selected_by_dataset,
                data_root,
                snapshot_end=effective_snapshot_end,
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
        minute_errors, minute_warnings, minute_daily_checks = _verify_minute_daily_consistency(
            connection,
            selected_by_dataset,
            data_root,
            snapshot_end=effective_snapshot_end,
        )
        errors.extend(minute_errors)
        warnings.extend(minute_warnings)
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
        "datasets": datasets,
        "duplicate_checks": duplicate_checks,
        "conflicting_duplicate_checks": conflicting_duplicate_checks,
        "completeness_checks": completeness_checks,
        "ohlc_checks": ohlc_checks,
        "minute_daily_checks": minute_daily_checks,
        "disclosure_checks": disclosure_checks,
        "errors": errors,
        "warnings": warnings,
    }


def _verify_daily_completeness(
    connection: duckdb.DuckDBPyConnection,
    selected_by_dataset: dict[str, list[dict[str, Any]]],
    data_root: Path,
    *,
    snapshot_end: date,
) -> tuple[list[str], list[str], dict[str, int | float]]:
    """Prove that open dates and daily stock cross-sections are complete."""

    errors: list[str] = []
    warnings: list[str] = []
    checks = {
        "open_trading_days": 0,
        "daily_trading_days": 0,
        "missing_trading_days": 0,
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
            f"SELECT ts_code, trade_date FROM ({daily_all_keys}) "
            f"WHERE {supported_history_filter}"
        )
        basic_keys = (
            f"SELECT ts_code, trade_date FROM ({basic_all_keys}) "
            f"WHERE {supported_history_filter}"
        )
        checks["daily_rows_outside_daily_basic_history"] = int(
            connection.execute(
                f"SELECT count(*) FROM ({daily_all_keys}) "
                f"WHERE NOT ({supported_history_filter})"
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


# 设计 §3.5 质量门：日线↔分钟聚合一致性。只比对换算契约明确的股票/ETF
# 分钟数据集（execution_contract.SIMULATION_MINUTE_SOURCE_DATASETS，分钟
# vol 为股、amount 为 CNY）；指数 amount 是成分均价、期货/期权有合约乘数，
# 均不在日线换算契约内。日线 vol 为手（=100 股）、amount 为千元。
MINUTE_DAILY_PRICE_TOLERANCE = 1e-4
MINUTE_DAILY_HARD_MISMATCH_RATE = 0.05
_DAILY_AMOUNT_CNY_PER_UNIT = 1000.0


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

    return {
        "ok": bool(report["ok"]),
        "verified_at": str(report["checked_at"]),
        "errors": list(report["errors"]),
    }


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
            f"SELECT ts_code, trade_date FROM ({query}) ORDER BY trade_date, ts_code LIMIT 10"
        ).fetchall()
    )


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
