from __future__ import annotations

import json
import math
import re
import time
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Lock
from typing import Any

import duckdb

INDEX_NAMES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "000016.SH": "上证50",
}
DEFAULT_WATCHLIST = (
    "000300.SH",
    "000905.SH",
    "000852.SH",
    "000016.SH",
    "510300.SH",
    "159919.SZ",
    "510500.SH",
    "512100.SH",
)
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9_.-]{1,31}$")


class MarketOverviewService:
    """Build a compact research-market view from an immutable daily snapshot."""

    def __init__(self, data_root: Path, *, cache_seconds: int = 30) -> None:
        self.data_root = data_root.resolve()
        self.cache_seconds = max(0, cache_seconds)
        self._lock = Lock()
        self._cache: dict[tuple[str, tuple[str, ...], int], tuple[float, dict[str, Any]]] = {}

    def get(
        self,
        *,
        snapshot_name: str | None = None,
        symbols: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        selected = self._select_snapshot(snapshot_name)
        if selected is None:
            return self._empty("尚无包含 A 股日线的不可变快照，请先完成数据收口。")
        snapshot, manifest, manifest_mtime = selected
        watchlist = self._normalize_symbols(symbols)
        key = (snapshot.name, watchlist, manifest_mtime)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] <= self.cache_seconds:
                return cached[1]
        result = self._build(snapshot, manifest, watchlist)
        with self._lock:
            self._cache = {key: (now, result)}
        return result

    def _select_snapshot(
        self, snapshot_name: str | None
    ) -> tuple[Path, dict[str, Any], int] | None:
        root = (self.data_root / "snapshots").resolve()
        if snapshot_name:
            candidate = (root / snapshot_name).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError("snapshot name resolves outside the snapshot root") from exc
            if not root.exists():
                return None
            candidates = [candidate]
        else:
            if not root.exists():
                return None
            candidates = sorted(
                (item for item in root.iterdir() if item.is_dir()),
                key=lambda item: item.stat().st_mtime_ns,
                reverse=True,
            )
        valid: list[tuple[datetime, Path, dict[str, Any], int]] = []
        for candidate in candidates:
            manifest_path = candidate / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            datasets = manifest.get("datasets")
            if not isinstance(datasets, dict) or not self._entry_files(
                candidate, datasets.get("daily")
            ):
                continue
            if str(manifest.get("frequency") or "day") != "day":
                continue
            try:
                created = datetime.fromisoformat(
                    str(manifest.get("created_at") or "").replace("Z", "+00:00")
                )
            except ValueError:
                created = datetime.fromtimestamp(manifest_path.stat().st_mtime, UTC)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            valid.append((created, candidate, manifest, manifest_path.stat().st_mtime_ns))
        if not valid:
            return None
        _, candidate, manifest, mtime = max(valid, key=lambda item: item[0])
        return candidate, manifest, mtime

    def _build(
        self,
        snapshot: Path,
        manifest: dict[str, Any],
        watchlist: tuple[str, ...],
    ) -> dict[str, Any]:
        datasets = manifest.get("datasets", {})
        connection = duckdb.connect()
        try:
            daily = self._relation(connection, snapshot, datasets, "daily")
            if daily is None or not {"ts_code", "trade_date", "close"}.issubset(daily[1]):
                return self._empty(
                    "所选快照缺少可读取的 A 股日线字段。",
                    snapshot_name=snapshot.name,
                )
            relation, columns = daily
            pct = self._pct_expression(columns)
            amount = self._number_expression(columns, "amount")
            trade_date = self._date_expression("trade_date")
            breadth_row = self._one(
                connection,
                f"""
                WITH bars AS (
                    SELECT {trade_date} AS trade_date, {pct} AS pct_chg,
                           {amount} AS amount
                    FROM {relation}
                ), latest AS (SELECT max(trade_date) AS trade_date FROM bars)
                SELECT latest.trade_date,
                       count(*) FILTER (WHERE bars.pct_chg IS NOT NULL) AS instruments,
                       count(*) FILTER (WHERE bars.pct_chg > 0) AS advances,
                       count(*) FILTER (WHERE bars.pct_chg < 0) AS declines,
                       count(*) FILTER (WHERE bars.pct_chg = 0) AS unchanged,
                       count(*) FILTER (WHERE bars.pct_chg >= 9.5) AS limit_up,
                       count(*) FILTER (WHERE bars.pct_chg <= -9.5) AS limit_down,
                       avg(bars.pct_chg) AS average_pct_chg,
                       median(bars.pct_chg) AS median_pct_chg,
                       sum(bars.amount) AS amount
                FROM latest LEFT JOIN bars USING (trade_date)
                GROUP BY latest.trade_date
                """,
            )
            as_of = self._date_text(breadth_row.get("trade_date"))
            pulse = self._rows(
                connection,
                f"""
                SELECT * FROM (
                    SELECT {trade_date} AS trade_date,
                           avg({pct}) AS average_pct_chg,
                           100.0 * count(*) FILTER (WHERE {pct} > 0)
                               / nullif(count(*) FILTER (WHERE {pct} IS NOT NULL), 0)
                               AS advance_ratio,
                           sum({amount}) AS amount
                    FROM {relation}
                    GROUP BY {trade_date}
                    ORDER BY trade_date DESC
                    LIMIT 20
                ) history ORDER BY trade_date
                """,
            )
            names, industries = self._instrument_metadata(connection, snapshot, datasets)
            indices = self._latest_rows(
                connection,
                snapshot,
                datasets,
                "index_daily",
                symbols=tuple(INDEX_NAMES),
                limit=10,
            )
            for item in indices:
                item["name"] = INDEX_NAMES.get(str(item.get("ts_code")), str(item.get("ts_code")))
            etfs = self._latest_rows(
                connection, snapshot, datasets, "fund_daily", limit=8, order_by_amount=True
            )
            for item in etfs:
                item["name"] = names.get(str(item.get("ts_code")), str(item.get("ts_code")))
            futures = self._futures(connection, snapshot, datasets)
            sectors = self._sectors(connection, snapshot, datasets, relation, columns)
            watch_rows = self._watchlist(
                connection, snapshot, datasets, watchlist, names, industries
            )
            source_date = self._parse_date(as_of)
            lag_days = (date.today() - source_date).days if source_date else None
            freshness = (
                "current"
                if lag_days is not None and lag_days <= 4
                else "delayed"
                if lag_days is not None and lag_days <= 10
                else "historical"
            )
            available = [
                name
                for name in ("daily", "index_daily", "fund_daily", "fut_daily", "stock_basic")
                if self._entry_files(snapshot, datasets.get(name))
            ]
            return {
                "status": "ready",
                "source": {
                    "mode": "research_snapshot",
                    "snapshot_name": snapshot.name,
                    "snapshot_created_at": manifest.get("created_at"),
                    "as_of": as_of,
                    "generated_at": datetime.now(UTC).isoformat(),
                    "is_realtime": False,
                    "freshness": freshness,
                    "calendar_days_behind": lag_days,
                    "available_datasets": available,
                },
                "breadth": self._json_row(breadth_row),
                "indices": [self._json_row(item) for item in indices],
                "pulse": [self._json_row(item) for item in pulse],
                "sectors": [self._json_row(item) for item in sectors],
                "etfs": [self._json_row(item) for item in etfs],
                "futures": [self._json_row(item) for item in futures],
                "watchlist": [self._json_row(item) for item in watch_rows],
            }
        finally:
            connection.close()

    def _instrument_metadata(
        self,
        connection: duckdb.DuckDBPyConnection,
        snapshot: Path,
        datasets: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, str]]:
        names: dict[str, str] = {}
        industries: dict[str, str] = {}
        for dataset in ("stock_basic", "fund_basic"):
            resolved = self._relation(connection, snapshot, datasets, dataset)
            if resolved is None or "ts_code" not in resolved[1]:
                continue
            relation, columns = resolved
            name = "CAST(name AS VARCHAR)" if "name" in columns else "CAST(ts_code AS VARCHAR)"
            industry = "CAST(industry AS VARCHAR)" if "industry" in columns else "NULL"
            rows = self._rows(
                connection,
                f"SELECT CAST(ts_code AS VARCHAR) AS ts_code, {name} AS name, "
                f"{industry} AS industry FROM {relation}",
            )
            for row in rows:
                code = str(row.get("ts_code") or "")
                if code:
                    names[code] = str(row.get("name") or code)
                    if row.get("industry"):
                        industries[code] = str(row["industry"])
        names.update(INDEX_NAMES)
        return names, industries

    def _latest_rows(
        self,
        connection: duckdb.DuckDBPyConnection,
        snapshot: Path,
        datasets: dict[str, Any],
        dataset: str,
        *,
        symbols: tuple[str, ...] = (),
        limit: int = 8,
        order_by_amount: bool = False,
    ) -> list[dict[str, Any]]:
        resolved = self._relation(connection, snapshot, datasets, dataset)
        if resolved is None:
            return []
        relation, columns = resolved
        if not {"ts_code", "trade_date", "close"}.issubset(columns):
            return []
        pct = self._pct_expression(columns)
        amount = self._number_expression(columns, "amount")
        trade_date = self._date_expression("trade_date")
        where = ""
        if symbols:
            where = (
                "WHERE CAST(ts_code AS VARCHAR) IN ("
                + ",".join(self._sql_string(symbol) for symbol in symbols)
                + ")"
            )
        order = "amount DESC NULLS LAST, ts_code" if order_by_amount else "ts_code"
        return self._rows(
            connection,
            f"""
            WITH ranked AS (
                SELECT CAST(ts_code AS VARCHAR) AS ts_code,
                       {trade_date} AS trade_date,
                       try_cast(close AS DOUBLE) AS close,
                       {pct} AS pct_chg,
                       {amount} AS amount,
                       row_number() OVER (
                           PARTITION BY ts_code ORDER BY {trade_date} DESC
                       ) AS rank
                FROM {relation} {where}
            )
            SELECT ts_code, trade_date, close, pct_chg, amount
            FROM ranked WHERE rank = 1 ORDER BY {order} LIMIT {int(limit)}
            """,
        )

    def _futures(
        self,
        connection: duckdb.DuckDBPyConnection,
        snapshot: Path,
        datasets: dict[str, Any],
    ) -> list[dict[str, Any]]:
        resolved = self._relation(connection, snapshot, datasets, "fut_daily")
        if resolved is None:
            return []
        relation, columns = resolved
        if not {"ts_code", "trade_date", "close"}.issubset(columns):
            return []
        pct = self._pct_expression(columns)
        amount = self._number_expression(columns, "amount")
        trade_date = self._date_expression("trade_date")
        rows = self._rows(
            connection,
            f"""
            WITH source AS (
                SELECT CAST(ts_code AS VARCHAR) AS ts_code,
                       regexp_extract(CAST(ts_code AS VARCHAR), '^(IF|IC|IM|IH)', 1) AS product,
                       {trade_date} AS trade_date,
                       try_cast(close AS DOUBLE) AS close,
                       {pct} AS pct_chg,
                       {amount} AS amount
                FROM {relation}
                WHERE regexp_matches(CAST(ts_code AS VARCHAR), '^(IF|IC|IM|IH)[0-9]')
            ), latest AS (
                SELECT *, row_number() OVER (
                    PARTITION BY product ORDER BY trade_date DESC, amount DESC NULLS LAST
                ) AS rank
                FROM source
            )
            SELECT ts_code, product, trade_date, close, pct_chg, amount
            FROM latest WHERE rank = 1 ORDER BY product
            """,
        )
        labels = {
            "IF": "沪深300期货",
            "IC": "中证500期货",
            "IM": "中证1000期货",
            "IH": "上证50期货",
        }
        for row in rows:
            row["name"] = labels.get(str(row.get("product")), str(row.get("ts_code")))
        return rows

    def _sectors(
        self,
        connection: duckdb.DuckDBPyConnection,
        snapshot: Path,
        datasets: dict[str, Any],
        daily_relation: str,
        daily_columns: set[str],
    ) -> list[dict[str, Any]]:
        basic = self._relation(connection, snapshot, datasets, "stock_basic")
        if basic is None or not {"ts_code", "industry"}.issubset(basic[1]):
            return []
        pct = self._pct_expression(daily_columns)
        amount = self._number_expression(daily_columns, "amount")
        trade_date = self._date_expression("trade_date")
        source_trade_date = self._date_expression("source.trade_date")
        return self._rows(
            connection,
            f"""
            WITH latest_date AS (
                SELECT max({trade_date}) AS trade_date FROM {daily_relation}
            ), bars AS (
                SELECT CAST(source.ts_code AS VARCHAR) AS ts_code,
                       {pct} AS pct_chg, {amount} AS amount
                FROM {daily_relation} AS source, latest_date
                WHERE {source_trade_date} = latest_date.trade_date
            ), grouped AS (
                SELECT CAST(b.industry AS VARCHAR) AS industry,
                       count(*) AS members,
                       avg(d.pct_chg) AS pct_chg,
                       sum(d.amount) AS amount,
                       100.0 * count(*) FILTER (WHERE d.pct_chg > 0)
                           / nullif(count(*), 0) AS advance_ratio
                FROM bars d JOIN {basic[0]} b USING (ts_code)
                WHERE b.industry IS NOT NULL AND trim(CAST(b.industry AS VARCHAR)) <> ''
                GROUP BY b.industry HAVING count(*) >= 3
            ), ranked AS (
                SELECT *, dense_rank() OVER (ORDER BY pct_chg DESC) AS best_rank,
                       dense_rank() OVER (ORDER BY pct_chg ASC) AS worst_rank
                FROM grouped
            )
            SELECT industry, members, pct_chg, amount, advance_ratio
            FROM ranked WHERE best_rank <= 5 OR worst_rank <= 5
            ORDER BY pct_chg DESC
            """,
        )

    def _watchlist(
        self,
        connection: duckdb.DuckDBPyConnection,
        snapshot: Path,
        datasets: dict[str, Any],
        symbols: tuple[str, ...],
        names: dict[str, str],
        industries: dict[str, str],
    ) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for dataset, asset_type in (
            ("daily", "stock"),
            ("fund_daily", "etf"),
            ("index_daily", "index"),
        ):
            for row in self._latest_rows(
                connection, snapshot, datasets, dataset, symbols=symbols, limit=len(symbols)
            ):
                code = str(row.get("ts_code"))
                row["asset_type"] = asset_type
                row["name"] = names.get(code, code)
                row["industry"] = industries.get(code)
                found.setdefault(code, row)
        return [found[symbol] for symbol in symbols if symbol in found]

    def _relation(
        self,
        connection: duckdb.DuckDBPyConnection,
        snapshot: Path,
        datasets: dict[str, Any],
        dataset: str,
    ) -> tuple[str, set[str]] | None:
        files = self._entry_files(snapshot, datasets.get(dataset))
        if not files:
            return None
        relation = (
            "read_parquet(["
            + ",".join(self._sql_string(str(path)) for path in files)
            + "] , union_by_name=true)"
        )
        try:
            columns = {
                str(row[0])
                for row in connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
            }
        except duckdb.Error:
            return None
        return relation, columns

    @staticmethod
    def _entry_files(snapshot: Path, entry: Any) -> list[Path]:
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), list):
            return []
        files: list[Path] = []
        for item in entry["files"]:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            target = (snapshot / str(item["path"])).resolve()
            try:
                target.relative_to(snapshot.resolve())
            except ValueError:
                continue
            if target.is_file():
                files.append(target)
        return files

    @staticmethod
    def _pct_expression(columns: set[str]) -> str:
        if "pct_chg" in columns:
            return "try_cast(pct_chg AS DOUBLE)"
        if {"close", "pre_close"}.issubset(columns):
            return (
                "100.0 * (try_cast(close AS DOUBLE) / nullif(try_cast(pre_close AS DOUBLE), 0) - 1)"
            )
        return "NULL::DOUBLE"

    @staticmethod
    def _number_expression(columns: set[str], name: str) -> str:
        return f"try_cast({name} AS DOUBLE)" if name in columns else "NULL::DOUBLE"

    @staticmethod
    def _date_expression(name: str) -> str:
        return (
            f"coalesce(try_cast({name} AS DATE), "
            f"try_strptime(CAST({name} AS VARCHAR), '%Y%m%d')::DATE)"
        )

    @staticmethod
    def _rows(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
        cursor = connection.execute(sql)
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    @classmethod
    def _one(cls, connection: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
        rows = cls._rows(connection, sql)
        return rows[0] if rows else {}

    @staticmethod
    def _normalize_symbols(symbols: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        values = symbols or DEFAULT_WATCHLIST
        normalized: list[str] = []
        for value in values:
            symbol = str(value).strip().upper()
            if symbol and _SYMBOL_PATTERN.fullmatch(symbol) and symbol not in normalized:
                normalized.append(symbol)
            if len(normalized) == 30:
                break
        return tuple(normalized or DEFAULT_WATCHLIST)

    @staticmethod
    def _json_row(row: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (date, datetime)):
                result[key] = value.isoformat()
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                result[key] = None
            else:
                result[key] = value
        return result

    @staticmethod
    def _date_text(value: Any) -> str | None:
        if isinstance(value, (date, datetime)):
            return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
        return str(value) if value else None

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        try:
            return date.fromisoformat(value or "")
        except ValueError:
            return None

    @staticmethod
    def _sql_string(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    @staticmethod
    def _empty(message: str, *, snapshot_name: str | None = None) -> dict[str, Any]:
        return {
            "status": "not_ready",
            "message": message,
            "source": {
                "mode": "research_snapshot",
                "snapshot_name": snapshot_name,
                "as_of": None,
                "generated_at": datetime.now(UTC).isoformat(),
                "is_realtime": False,
                "freshness": "unavailable",
                "calendar_days_behind": None,
                "available_datasets": [],
            },
            "breadth": {},
            "indices": [],
            "pulse": [],
            "sectors": [],
            "etfs": [],
            "futures": [],
            "watchlist": [],
        }
