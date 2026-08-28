from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any

import duckdb

INDEX_NAMES = {
    "000001.SH": "上证指数",
    "000016.SH": "上证50",
    "000300.SH": "沪深300",
    "000688.SH": "科创50",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "899050.BJ": "北证50",
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
_PARTITION_PATTERN = re.compile(
    r"(?:^|/)partition_year=(\d{4})/partition_month=(\d{1,2})(?:/|$)"
)
_MATERIALIZED_SCHEMA_VERSION = "market-overview-v1"
_RECENT_MARKET_MONTHS = 4


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
        """Read only a compact publisher-owned projection.

        This method is called from the HTTP request path.  It must never open a
        snapshot manifest or DuckDB: a cold or damaged projection is a display
        cache miss, not permission to scan the immutable research lake inside
        the API process.
        """

        watchlist = self._normalize_symbols(symbols)
        normalized_snapshot = self._validate_snapshot_name(snapshot_name)
        materialized = self._read_published_materialized(
            snapshot_name=normalized_snapshot,
            watchlist=watchlist,
        )
        if materialized is not None:
            return materialized
        return self._empty(
            "行情总览投影尚未由后台生成；页面不会现场扫描历史数据。",
            snapshot_name=normalized_snapshot,
        )

    def materialize(
        self,
        *,
        snapshot_name: str,
        symbols: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Create or reuse a compact artifact outside the HTTP request path."""

        watchlist = self._normalize_symbols(symbols)
        normalized_snapshot = self._validate_snapshot_name(snapshot_name)
        selected = self._select_snapshot(normalized_snapshot)
        if selected is None:
            return self._empty(
                "所选快照不包含可物化的 A 股日线数据。",
                snapshot_name=normalized_snapshot,
            )
        snapshot, manifest, manifest_mtime, manifest_sha256 = selected
        key = (snapshot.name, watchlist, manifest_mtime)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] <= self.cache_seconds:
                return cached[1]
            result = self._read_materialized(
                snapshot=snapshot,
                manifest_mtime=manifest_mtime,
                manifest_sha256=manifest_sha256,
                watchlist=watchlist,
            )
            if result is None:
                result = self._build(snapshot, manifest, watchlist)
            if result.get("status") == "ready":
                # Re-publish even when the immutable, content-addressed result
                # already exists.  The stable request/latest pointers may be
                # absent after an upgrade or interrupted publication.
                self._write_materialized(
                    snapshot=snapshot,
                    manifest_mtime=manifest_mtime,
                    manifest_sha256=manifest_sha256,
                    watchlist=watchlist,
                    result=result,
                )
            self._remember(key, now, result)
            return result

    def _validate_snapshot_name(self, snapshot_name: str | None) -> str | None:
        if snapshot_name is None:
            return None
        normalized = str(snapshot_name).strip()
        root = (self.data_root / "snapshots").resolve()
        candidate = (root / normalized).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("snapshot name resolves outside the snapshot root") from exc
        return normalized

    def _remember(
        self,
        key: tuple[str, tuple[str, ...], int],
        now: float,
        result: dict[str, Any],
    ) -> None:
        self._cache[key] = (now, result)
        if len(self._cache) > 16:
            oldest = min(self._cache, key=lambda item: self._cache[item][0])
            self._cache.pop(oldest, None)

    def _select_snapshot(
        self, snapshot_name: str | None
    ) -> tuple[Path, dict[str, Any], int, str] | None:
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
        manifests: list[tuple[int, Path, Path]] = []
        for candidate in candidates:
            manifest_path = candidate / "manifest.json"
            try:
                manifests.append((manifest_path.stat().st_mtime_ns, candidate, manifest_path))
            except FileNotFoundError:
                continue
        for _, candidate, manifest_path in sorted(
            manifests, key=lambda item: item[0], reverse=True
        ):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            datasets = manifest.get("datasets")
            if not isinstance(datasets, dict) or not self._entry_has_files(datasets.get("daily")):
                continue
            if str(manifest.get("frequency") or "day") != "day":
                continue
            if not self._entry_files(
                candidate, datasets.get("daily"), recent_months=1
            ):
                continue
            raw_manifest = manifest_path.read_bytes()
            return (
                candidate,
                manifest,
                manifest_path.stat().st_mtime_ns,
                sha256(raw_manifest).hexdigest(),
            )
        return None

    def _build(
        self,
        snapshot: Path,
        manifest: dict[str, Any],
        watchlist: tuple[str, ...],
    ) -> dict[str, Any]:
        datasets = manifest.get("datasets", {})
        connection = duckdb.connect()
        try:
            daily = self._relation(
                connection,
                snapshot,
                datasets,
                "daily",
                recent_months=_RECENT_MARKET_MONTHS,
            )
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
                if self._entry_files(snapshot, datasets.get(name), recent_months=1)
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
        resolved = self._relation(
            connection,
            snapshot,
            datasets,
            dataset,
            recent_months=_RECENT_MARKET_MONTHS,
        )
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
        resolved = self._relation(
            connection,
            snapshot,
            datasets,
            "fut_daily",
            recent_months=_RECENT_MARKET_MONTHS,
        )
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
        *,
        recent_months: int | None = None,
    ) -> tuple[str, set[str]] | None:
        files = self._entry_files(
            snapshot,
            datasets.get(dataset),
            recent_months=recent_months,
        )
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
    def _entry_has_files(entry: Any) -> bool:
        return bool(
            isinstance(entry, dict)
            and isinstance(entry.get("files"), list)
            and entry["files"]
        )

    @staticmethod
    def _entry_files(
        snapshot: Path,
        entry: Any,
        *,
        recent_months: int | None = None,
    ) -> list[Path]:
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), list):
            return []
        candidates: list[tuple[Path, tuple[int, int] | None]] = []
        for item in entry["files"]:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            relative = str(item["path"]).replace("\\", "/")
            target = (snapshot / relative).resolve()
            try:
                target.relative_to(snapshot.resolve())
            except ValueError:
                continue
            match = _PARTITION_PATTERN.search(relative)
            partition = (int(match.group(1)), int(match.group(2))) if match else None
            candidates.append((target, partition))
        if recent_months and candidates and all(item[1] is not None for item in candidates):
            partitions = sorted({item[1] for item in candidates if item[1] is not None})
            keep = set(partitions[-max(1, int(recent_months)) :])
            candidates = [item for item in candidates if item[1] in keep]
        return [target for target, _ in candidates if target.is_file()]

    def _materialized_path(
        self,
        *,
        snapshot: Path,
        manifest_sha256: str,
        watchlist: tuple[str, ...],
    ) -> Path:
        symbols_sha256 = self._symbols_sha256(watchlist)
        return (
            self.data_root
            / "artifacts"
            / "market-overview"
            / snapshot.name
            / manifest_sha256[:16]
            / f"{symbols_sha256}.json"
        )

    @staticmethod
    def _symbols_sha256(watchlist: tuple[str, ...]) -> str:
        return sha256("\n".join(watchlist).encode("utf-8")).hexdigest()

    def _published_request_path(
        self,
        *,
        snapshot_name: str,
        watchlist: tuple[str, ...],
    ) -> Path:
        snapshot_sha256 = sha256(snapshot_name.encode("utf-8")).hexdigest()
        return (
            self.data_root
            / "artifacts"
            / "market-overview"
            / "requests"
            / snapshot_sha256
            / f"{self._symbols_sha256(watchlist)}.json"
        )

    def _latest_path(self, watchlist: tuple[str, ...], *, previous: bool = False) -> Path:
        if watchlist == DEFAULT_WATCHLIST:
            filename = "previous.json" if previous else "latest.json"
        else:
            prefix = "previous" if previous else "latest"
            filename = f"{prefix}-{self._symbols_sha256(watchlist)}.json"
        return self.data_root / "artifacts" / "market-overview" / filename

    def _read_materialized(
        self,
        *,
        snapshot: Path,
        manifest_mtime: int,
        manifest_sha256: str,
        watchlist: tuple[str, ...],
    ) -> dict[str, Any] | None:
        path = self._materialized_path(
            snapshot=snapshot,
            manifest_sha256=manifest_sha256,
            watchlist=watchlist,
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if (
            payload.get("schema_version") != _MATERIALIZED_SCHEMA_VERSION
            or payload.get("snapshot_name") != snapshot.name
            or payload.get("manifest_sha256") != manifest_sha256
            or payload.get("manifest_mtime_ns") != manifest_mtime
            or payload.get("symbols") != list(watchlist)
            or not isinstance(payload.get("result"), dict)
        ):
            return None
        return payload["result"]

    def _read_published_payload(
        self,
        path: Path,
        *,
        snapshot_name: str | None,
        watchlist: tuple[str, ...],
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if (
            payload.get("schema_version") != _MATERIALIZED_SCHEMA_VERSION
            or payload.get("symbols") != list(watchlist)
            or (snapshot_name is not None and payload.get("snapshot_name") != snapshot_name)
            or not isinstance(payload.get("result"), dict)
            or payload["result"].get("status") != "ready"
        ):
            return None
        return payload, payload["result"]

    def _read_published_materialized(
        self,
        *,
        snapshot_name: str | None,
        watchlist: tuple[str, ...],
    ) -> dict[str, Any] | None:
        if snapshot_name is not None:
            paths = [
                self._published_request_path(
                    snapshot_name=snapshot_name,
                    watchlist=watchlist,
                )
            ]
        else:
            # `previous` is deliberately retained across publications.  It is
            # a small, known path and provides a bounded fallback if `latest`
            # is damaged; no directory or snapshot discovery occurs here.
            paths = [
                self._latest_path(watchlist),
                self._latest_path(watchlist, previous=True),
            ]
        for path in paths:
            published = self._read_published_payload(
                path,
                snapshot_name=snapshot_name,
                watchlist=watchlist,
            )
            if published is not None:
                return published[1]
        return None

    def _read_latest_materialized(
        self, watchlist: tuple[str, ...]
    ) -> dict[str, Any] | None:
        """Compatibility wrapper for callers that only need the latest projection."""

        return self._read_published_materialized(
            snapshot_name=None,
            watchlist=watchlist,
        )

    @staticmethod
    def _publication_order(payload: dict[str, Any]) -> tuple[str, int]:
        result = payload.get("result")
        source = result.get("source") if isinstance(result, dict) else None
        as_of = str(source.get("as_of") or "") if isinstance(source, dict) else ""
        try:
            manifest_mtime = int(payload.get("manifest_mtime_ns") or 0)
        except (TypeError, ValueError):
            manifest_mtime = 0
        return as_of, manifest_mtime

    def _write_materialized(
        self,
        *,
        snapshot: Path,
        manifest_mtime: int,
        manifest_sha256: str,
        watchlist: tuple[str, ...],
        result: dict[str, Any],
    ) -> None:
        path = self._materialized_path(
            snapshot=snapshot,
            manifest_sha256=manifest_sha256,
            watchlist=watchlist,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _MATERIALIZED_SCHEMA_VERSION,
            "snapshot_name": snapshot.name,
            "manifest_sha256": manifest_sha256,
            "manifest_mtime_ns": manifest_mtime,
            "symbols": list(watchlist),
            "generated_at": datetime.now(UTC).isoformat(),
            "result": result,
        }
        self._write_json_atomic(path, payload)
        self._write_json_atomic(
            self._published_request_path(
                snapshot_name=snapshot.name,
                watchlist=watchlist,
            ),
            payload,
        )
        latest_path = self._latest_path(watchlist)
        current = self._read_published_payload(
            latest_path,
            snapshot_name=None,
            watchlist=watchlist,
        )
        if current is None or self._publication_order(payload) >= self._publication_order(
            current[0]
        ):
            if current is not None and current[0] != payload:
                self._write_json_atomic(
                    self._latest_path(watchlist, previous=True),
                    current[0],
                )
            self._write_json_atomic(
                latest_path,
                payload,
            )

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)

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
