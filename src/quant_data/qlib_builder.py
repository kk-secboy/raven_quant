from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .path_utils import to_wsl_path as _to_wsl_path


class QlibBuilder:
    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path.resolve()

    def build_staging(self, staging_path: Path) -> Path:
        daily_glob = self.snapshot_path / "parquet" / "daily" / "**" / "*.parquet"
        adj_glob = self.snapshot_path / "parquet" / "adj_factor" / "**" / "*.parquet"
        limit_glob = self.snapshot_path / "parquet" / "stk_limit" / "**" / "*.parquet"
        if not list((self.snapshot_path / "parquet" / "daily").rglob("*.parquet")):
            raise FileNotFoundError("snapshot does not contain daily Parquet data")
        if not list((self.snapshot_path / "parquet" / "adj_factor").rglob("*.parquet")):
            raise FileNotFoundError("snapshot does not contain adj_factor Parquet data")
        if not list((self.snapshot_path / "parquet" / "stk_limit").rglob("*.parquet")):
            raise FileNotFoundError("snapshot does not contain A-share price-limit Parquet data")

        staging_path = staging_path.resolve()
        temporary = staging_path.with_name(f".{staging_path.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        partitions = temporary / "partitions"
        by_symbol = temporary / "by_symbol"
        partitions.mkdir(parents=True, exist_ok=True)
        by_symbol.mkdir(parents=True, exist_ok=True)

        connection = duckdb.connect()
        try:
            query = self._normalized_query(daily_glob, adj_glob, limit_glob)
            missing = connection.execute(
                self._missing_market_controls_query(daily_glob, adj_glob, limit_glob)
            ).fetchone()[0]
            if missing:
                raise RuntimeError(
                    f"{missing} daily rows have no valid adjustment factor or price limits"
                )
            connection.execute(
                f"COPY ({query}) TO {_sql_string(str(partitions))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (symbol), "
                "ROW_GROUP_SIZE 100000)"
            )
        finally:
            connection.close()

        for partition in sorted(partitions.glob("symbol=*")):
            symbol = partition.name.split("=", 1)[1]
            files = sorted(partition.glob("*.parquet"))
            if not files:
                continue
            if len(files) == 1:
                frame = pd.read_parquet(files[0])
            else:
                frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
            frame.insert(1, "symbol", symbol)
            frame.sort_values("date", inplace=True)
            frame.to_parquet(
                by_symbol / f"{symbol}.parquet",
                index=False,
                compression="zstd",
            )
        self._write_index_staging(by_symbol)
        shutil.rmtree(partitions)
        if not any(by_symbol.glob("*.parquet")):
            raise RuntimeError("Qlib staging produced no per-symbol Parquet files")
        if staging_path.exists():
            shutil.rmtree(staging_path)
        os.replace(temporary, staging_path)
        return staging_path / "by_symbol"

    def dump_bin(
        self,
        *,
        staging_by_symbol: Path,
        qlib_dir: Path,
        qlib_repo: Path,
        qlib_python: str,
        wsl_distro: str,
        max_workers: int = 8,
    ) -> Path:
        script = qlib_repo.resolve() / "scripts" / "dump_bin.py"
        if not script.exists():
            raise FileNotFoundError(f"Qlib dump script not found: {script}")
        qlib_dir = qlib_dir.resolve()
        qlib_dir.parent.mkdir(parents=True, exist_ok=True)
        if qlib_dir.exists():
            raise FileExistsError(f"Qlib output already exists: {qlib_dir}")
        self._snapshot_manifest_digest()
        if os.name == "nt" and qlib_python.startswith("/"):
            command = [
                "wsl",
                "-d",
                wsl_distro,
                "--exec",
                qlib_python,
                _to_wsl_path(script),
                "dump_all",
                "--data_path",
                _to_wsl_path(staging_by_symbol),
                "--qlib_dir",
                _to_wsl_path(qlib_dir),
            ]
        else:
            command = [
                qlib_python,
                str(script),
                "dump_all",
                "--data_path",
                str(staging_by_symbol),
                "--qlib_dir",
                str(qlib_dir),
            ]
        command.extend(
            [
                "--freq",
                "day",
                "--file_suffix",
                ".parquet",
                "--date_field_name",
                "date",
                "--symbol_field_name",
                "symbol",
                "--include_fields",
                "open,high,low,close,vwap,volume,factor,change,amount,paused,up_limit,down_limit",
                "--max_workers",
                str(max_workers),
            ]
        )
        try:
            subprocess.run(command, check=True)
            required = (
                qlib_dir / "calendars" / "day.txt",
                qlib_dir / "instruments" / "all.txt",
                qlib_dir / "features",
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise RuntimeError(f"Qlib dump completed without required outputs: {missing}")
            if not any((qlib_dir / "features").rglob("*.day.bin")):
                raise RuntimeError("Qlib dump produced no daily feature binaries")
            self._write_stock_universe(qlib_dir)
            self._write_portfolio_metadata(qlib_dir)
            self._write_provenance(qlib_dir)
        except Exception:
            if qlib_dir.exists():
                shutil.rmtree(qlib_dir)
            raise
        return qlib_dir

    def _write_provenance(self, qlib_dir: Path) -> None:
        snapshot_digest = self._snapshot_manifest_digest()
        snapshot_manifest = json.loads(
            (self.snapshot_path / "manifest.json").read_text(encoding="utf-8")
        )
        builder_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        fields = [
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "factor",
            "change",
            "amount",
            "paused",
            "up_limit",
            "down_limit",
        ]
        contract = {"frequency": "day", "fields": fields}
        contract_sha256 = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        source_lineage_id = str(snapshot_manifest.get("lineage_id") or "")
        lineage_verified = len(source_lineage_id) == 64 and all(
            character in "0123456789abcdef" for character in source_lineage_id
        )
        dataset_lineage_id = (
            hashlib.sha256(
                json.dumps(
                    {
                        "source_lineage_id": source_lineage_id,
                        "dataset_contract_sha256": contract_sha256,
                        "qlib_builder_sha256": builder_digest,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if lineage_verified
            else None
        )
        identity = {
            "snapshot_name": self.snapshot_path.name,
            "snapshot_manifest_sha256": snapshot_digest,
            "qlib_builder_sha256": builder_digest,
            "frequency": "day",
            "fields": fields,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        provenance = {
            **identity,
            "dataset_identity_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "dataset_contract_sha256": contract_sha256,
            "dataset_lineage_id": dataset_lineage_id,
            "source_lineage_id": source_lineage_id or None,
            "source_lineage_generation": snapshot_manifest.get("lineage_generation"),
            "source_parent_snapshot": snapshot_manifest.get("parent_snapshot"),
            "source_start_date": snapshot_manifest.get("start_date"),
            "source_end_date": snapshot_manifest.get("end_date"),
            "lineage_verified": lineage_verified,
            "created_at": datetime.now(UTC).isoformat(),
        }
        target = qlib_dir / "metadata" / "provenance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    def _snapshot_manifest_digest(self) -> str:
        snapshot_manifest = self.snapshot_path / "manifest.json"
        if not snapshot_manifest.exists():
            raise FileNotFoundError("immutable source snapshot is missing manifest.json")
        manifest = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
        datasets = manifest.get("datasets")
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError("snapshot manifest has no dataset content identities")
        for dataset, entry in datasets.items():
            if not isinstance(entry, dict) or not entry.get("source_sha256"):
                raise ValueError(f"snapshot dataset {dataset} has no source SHA-256")
            files = entry.get("files")
            if not isinstance(files, list):
                raise ValueError(f"snapshot dataset {dataset} has no file manifest")
            if int(entry.get("rows") or 0) > 0 and not files:
                raise ValueError(f"snapshot dataset {dataset} has rows but no content files")
            for item in files:
                relative = Path(str(item.get("path") or ""))
                target = (self.snapshot_path / relative).resolve()
                try:
                    target.relative_to(self.snapshot_path)
                except ValueError as exc:
                    raise ValueError("snapshot manifest contains an unsafe file path") from exc
                if not target.is_file() or target.stat().st_size != int(item.get("bytes") or -1):
                    raise ValueError(f"snapshot file size mismatch: {relative.as_posix()}")
                if _sha256_file(target) != item.get("sha256"):
                    raise ValueError(f"snapshot file digest mismatch: {relative.as_posix()}")
        return hashlib.sha256(snapshot_manifest.read_bytes()).hexdigest()

    def _write_index_staging(self, by_symbol: Path) -> None:
        index_root = self.snapshot_path / "parquet" / "index_daily"
        files = sorted(index_root.rglob("*.parquet")) if index_root.exists() else []
        if not files:
            return
        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        required = {"ts_code", "trade_date", "open", "high", "low", "close"}
        if frame.empty or not required.issubset(frame.columns):
            return
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame.sort_values(["ts_code", "trade_date"], inplace=True)
        for ts_code, group in frame.groupby("ts_code", sort=True):
            group = group.copy()
            base_price = float(group["close"].dropna().iloc[0])
            if base_price <= 0:
                continue
            exchange, code = str(ts_code).split(".", 1)[1], str(ts_code).split(".", 1)[0]
            symbol = f"{exchange.upper()}{code}"
            volume = pd.to_numeric(group.get("vol", 0.0), errors="coerce").fillna(0.0)
            amount = pd.to_numeric(group.get("amount", 0.0), errors="coerce").fillna(0.0)
            close = pd.to_numeric(group["close"], errors="coerce")
            raw_vwap = (amount * 10.0 / volume.where(volume > 0)).fillna(close)
            normalized = pd.DataFrame(
                {
                    "date": group["trade_date"],
                    "symbol": symbol,
                    "open": pd.to_numeric(group["open"], errors="coerce") / base_price,
                    "high": pd.to_numeric(group["high"], errors="coerce") / base_price,
                    "low": pd.to_numeric(group["low"], errors="coerce") / base_price,
                    "close": close / base_price,
                    "vwap": raw_vwap / base_price,
                    "volume": volume * base_price,
                    "factor": 1.0 / base_price,
                    "change": pd.to_numeric(group.get("pct_chg", 0.0), errors="coerce")
                    .fillna(0.0)
                    .div(100.0),
                    "amount": amount,
                    "paused": (volume <= 0).astype(float),
                }
            )
            normalized.to_parquet(by_symbol / f"{symbol}.parquet", index=False, compression="zstd")

    def _write_stock_universe(self, qlib_dir: Path) -> None:
        daily_glob = self.snapshot_path / "parquet" / "daily" / "**" / "*.parquet"
        connection = duckdb.connect()
        try:
            rows = connection.execute(
                f"""
                SELECT ts_code, min(trade_date), max(trade_date)
                FROM read_parquet({_sql_string(str(daily_glob.resolve()))}, hive_partitioning=true)
                WHERE ts_code IS NOT NULL AND trade_date IS NOT NULL
                GROUP BY ts_code ORDER BY ts_code
                """
            ).fetchall()
        finally:
            connection.close()
        lines = []
        for ts_code, start, end in rows:
            code, exchange = str(ts_code).split(".", 1)
            lines.append(f"{exchange.upper()}{code}\t{start}\t{end}")
        if not lines:
            raise RuntimeError("Qlib stock universe is empty")
        target = qlib_dir / "instruments" / "cn_all.txt"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_portfolio_metadata(self, qlib_dir: Path) -> None:
        target = qlib_dir / "metadata"
        wrote_metadata = False

        industry_source = self.snapshot_path / "parquet" / "index_member_all"
        industry_files = (
            sorted(industry_source.rglob("*.parquet")) if industry_source.exists() else []
        )
        if industry_files:
            frame = pd.concat(
                [pd.read_parquet(path) for path in industry_files], ignore_index=True
            )
            instrument_column = "ts_code" if "ts_code" in frame.columns else "con_code"
            industry_column = next(
                (
                    name
                    for name in ("l1_code", "index_code", "l2_code")
                    if name in frame.columns
                ),
                None,
            )
            if industry_column and instrument_column in frame.columns:
                metadata = pd.DataFrame(
                    {
                        "instrument": frame[instrument_column].map(_qlib_symbol),
                        "industry": frame[industry_column].astype("string"),
                        "in_date": pd.to_datetime(frame.get("in_date"), errors="coerce"),
                        "out_date": pd.to_datetime(frame.get("out_date"), errors="coerce"),
                    }
                ).dropna(subset=["instrument", "industry", "in_date"])
                metadata.drop_duplicates(
                    ["instrument", "industry", "in_date", "out_date"], inplace=True
                )
                metadata.sort_values(["instrument", "in_date", "industry"], inplace=True)
                if not metadata.empty:
                    target.mkdir(parents=True, exist_ok=True)
                    metadata.to_parquet(
                        target / "industry_memberships.parquet", index=False
                    )
                    wrote_metadata = True

        weight_source = self.snapshot_path / "parquet" / "index_weight"
        weight_files = sorted(weight_source.rglob("*.parquet")) if weight_source.exists() else []
        if weight_files:
            frame = pd.concat([pd.read_parquet(path) for path in weight_files], ignore_index=True)
            required = {"index_code", "con_code", "trade_date", "weight"}
            if required.issubset(frame.columns):
                weights = pd.DataFrame(
                    {
                        "benchmark": frame["index_code"].map(_qlib_symbol),
                        "instrument": frame["con_code"].map(_qlib_symbol),
                        "datetime": pd.to_datetime(frame["trade_date"], errors="coerce"),
                        "weight": pd.to_numeric(frame["weight"], errors="coerce"),
                    }
                ).dropna()
                if not weights.empty and float(weights["weight"].max()) > 1.0:
                    weights["weight"] = weights["weight"] / 100.0
                weights = weights[weights["weight"] > 0]
                weights.drop_duplicates(
                    ["benchmark", "instrument", "datetime"], keep="last", inplace=True
                )
                weights.sort_values(
                    ["benchmark", "datetime", "instrument"], inplace=True
                )
                if not weights.empty:
                    target.mkdir(parents=True, exist_ok=True)
                    weights.to_parquet(target / "benchmark_weights.parquet", index=False)
                    wrote_metadata = True

        style_source = self.snapshot_path / "parquet" / "daily_basic"
        style_files = sorted(style_source.rglob("*.parquet")) if style_source.exists() else []
        if style_files:
            frame = pd.concat([pd.read_parquet(path) for path in style_files], ignore_index=True)
            required = {"ts_code", "trade_date", "total_mv"}
            if required.issubset(frame.columns):
                market_cap = pd.to_numeric(frame["total_mv"], errors="coerce")
                styles = pd.DataFrame(
                    {
                        "instrument": frame["ts_code"].map(_qlib_symbol),
                        "datetime": pd.to_datetime(frame["trade_date"], errors="coerce"),
                        "log_market_cap": market_cap.where(market_cap > 0).map(np.log),
                    }
                ).dropna()
                styles.drop_duplicates(["instrument", "datetime"], keep="last", inplace=True)
                styles.sort_values(["datetime", "instrument"], inplace=True)
                if not styles.empty:
                    target.mkdir(parents=True, exist_ok=True)
                    styles.to_parquet(target / "style_exposures.parquet", index=False)
                    wrote_metadata = True

        if not wrote_metadata and target.exists() and not any(target.iterdir()):
            target.rmdir()

    @staticmethod
    def _normalized_query(daily_glob: Path, adj_glob: Path, limit_glob: Path) -> str:
        daily = _sql_string(str(daily_glob.resolve()))
        adj = _sql_string(str(adj_glob.resolve()))
        limits = _sql_string(str(limit_glob.resolve()))
        return f"""
            WITH joined AS (
                SELECT
                    d.ts_code,
                    d.trade_date,
                    d.open,
                    d.high,
                    d.low,
                    d.close,
                    d.vol,
                    d.amount,
                    d.pct_chg,
                    a.adj_factor,
                    l.up_limit,
                    l.down_limit,
                    first_value(d.close * a.adj_factor) OVER (
                        PARTITION BY d.ts_code ORDER BY d.trade_date
                    ) AS base_price
                FROM read_parquet({daily}, hive_partitioning=true) d
                LEFT JOIN read_parquet({adj}, hive_partitioning=true) a
                  ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
                LEFT JOIN read_parquet({limits}, hive_partitioning=true) l
                  ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
                WHERE d.ts_code IS NOT NULL AND d.close IS NOT NULL
            )
            SELECT
                trade_date AS date,
                upper(split_part(ts_code, '.', 2) || split_part(ts_code, '.', 1)) AS symbol,
                open * adj_factor / base_price AS open,
                high * adj_factor / base_price AS high,
                low * adj_factor / base_price AS low,
                close * adj_factor / base_price AS close,
                CASE
                    WHEN vol IS NOT NULL AND vol > 0 AND amount IS NOT NULL
                    THEN amount * 10.0 / vol * adj_factor / base_price
                    ELSE close * adj_factor / base_price
                END AS vwap,
                vol * base_price / adj_factor AS volume,
                adj_factor / base_price AS factor,
                pct_chg / 100.0 AS change,
                amount,
                CASE WHEN vol IS NULL OR vol <= 0 THEN 1.0 ELSE 0.0 END AS paused
                , up_limit * adj_factor / base_price AS up_limit
                , down_limit * adj_factor / base_price AS down_limit
            FROM joined
            WHERE adj_factor IS NOT NULL AND adj_factor > 0 AND base_price > 0
        """

    @staticmethod
    def _missing_market_controls_query(daily_glob: Path, adj_glob: Path, limit_glob: Path) -> str:
        daily = _sql_string(str(daily_glob.resolve()))
        adj = _sql_string(str(adj_glob.resolve()))
        limits = _sql_string(str(limit_glob.resolve()))
        return f"""
            SELECT count(*)
            FROM read_parquet({daily}, hive_partitioning=true) d
            LEFT JOIN read_parquet({adj}, hive_partitioning=true) a
              ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
            LEFT JOIN read_parquet({limits}, hive_partitioning=true) l
              ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
            WHERE d.ts_code IS NOT NULL AND d.close IS NOT NULL
              AND (
                a.adj_factor IS NULL OR a.adj_factor <= 0
                OR l.up_limit IS NULL OR l.down_limit IS NULL
                OR (
                  (l.up_limit <= 0 OR l.down_limit <= 0)
                  AND NOT (l.up_limit >= 99999.0 AND l.down_limit = 0)
                )
              )
        """


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qlib_symbol(value: object) -> str | None:
    text = str(value or "")
    if "." not in text:
        return None
    code, exchange = text.split(".", 1)
    return f"{exchange.upper()}{code}"
