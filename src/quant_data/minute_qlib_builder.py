from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from .execution_data import MINUTE_DATASETS
from .path_utils import to_wsl_path
from .qlib_builder import QlibBuilder, _sql_string

MINUTE_QLIB_FIELDS = (
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
    "oi",
)


class MinuteQlibBuilder:
    """Build a separate Qlib 1-minute dataset from an immutable execution snapshot."""

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path.resolve()
        manifest_path = self.snapshot_path / "manifest.json"
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError("minute snapshot manifest is missing or invalid") from exc
        if self.manifest.get("frequency") != "1min":
            raise ValueError("minute Qlib builder requires a 1min execution snapshot")
        if not set(self.manifest.get("datasets", {})).intersection(MINUTE_DATASETS):
            raise ValueError("minute snapshot contains no supported bar datasets")

    def build_staging(self, staging_path: Path) -> Path:
        staging_path = staging_path.resolve()
        temporary = staging_path.with_name(f".{staging_path.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        partitions = temporary / "partitions"
        by_symbol = temporary / "by_symbol"
        partitions.mkdir(parents=True)
        by_symbol.mkdir()
        sources = []
        for dataset in MINUTE_DATASETS:
            root = self.snapshot_path / "parquet" / dataset
            if not root.exists() or not any(root.rglob("*.parquet")):
                continue
            glob = _sql_string(str((root / "**" / "*.parquet").resolve()))
            oi = "try_cast(oi AS DOUBLE)" if dataset == "futures_1m" else "NULL::DOUBLE"
            sources.append(
                f"SELECT *, {oi} AS normalized_oi, {_sql_string(dataset)} AS source_dataset "
                f"FROM read_parquet({glob}, hive_partitioning=true, union_by_name=true)"
            )
        if not sources:
            raise FileNotFoundError("snapshot does not contain minute Parquet data")
        query = self._normalized_query(" UNION ALL ".join(sources))
        connection = duckdb.connect()
        try:
            connection.execute(
                f"COPY ({query}) TO {_sql_string(str(partitions))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (symbol), ROW_GROUP_SIZE 100000)"
            )
        finally:
            connection.close()
        for partition in sorted(partitions.glob("symbol=*")):
            symbol = partition.name.split("=", 1)[1]
            files = sorted(partition.glob("*.parquet"))
            frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
            frame.insert(1, "symbol", symbol)
            frame.sort_values("date", inplace=True)
            frame.to_parquet(by_symbol / f"{symbol}.parquet", index=False, compression="zstd")
        shutil.rmtree(partitions)
        if not any(by_symbol.glob("*.parquet")):
            raise RuntimeError("minute Qlib staging produced no instruments")
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
        if qlib_dir.exists():
            raise FileExistsError(f"Qlib output already exists: {qlib_dir}")
        command = (
            [
                "wsl",
                "-d",
                wsl_distro,
                "--exec",
                qlib_python,
                to_wsl_path(script),
                "dump_all",
                "--data_path",
                to_wsl_path(staging_by_symbol),
                "--qlib_dir",
                to_wsl_path(qlib_dir),
            ]
            if os.name == "nt" and qlib_python.startswith("/")
            else [
                qlib_python,
                str(script),
                "dump_all",
                "--data_path",
                str(staging_by_symbol),
                "--qlib_dir",
                str(qlib_dir),
            ]
        )
        command.extend(
            [
                "--freq",
                "1min",
                "--file_suffix",
                ".parquet",
                "--date_field_name",
                "date",
                "--symbol_field_name",
                "symbol",
                "--include_fields",
                ",".join(MINUTE_QLIB_FIELDS),
                "--max_workers",
                str(max_workers),
            ]
        )
        try:
            subprocess.run(command, check=True)
            if not any((qlib_dir / "features").rglob("*.1min.bin")):
                raise RuntimeError("Qlib dump produced no minute feature binaries")
            self._write_provenance(qlib_dir)
        except Exception:
            if qlib_dir.exists():
                shutil.rmtree(qlib_dir)
            raise
        return qlib_dir

    def _write_provenance(self, qlib_dir: Path) -> None:
        manifest_path = self.snapshot_path / "manifest.json"
        QlibBuilder(self.snapshot_path)._snapshot_manifest_digest()
        identity = {
            "snapshot_name": self.snapshot_path.name,
            "snapshot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "qlib_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "frequency": "1min",
            "fields": list(MINUTE_QLIB_FIELDS),
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        provenance = {
            **identity,
            "dataset_identity_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "dataset_lineage_id": self.manifest.get("lineage_id"),
            "source_lineage_id": self.manifest.get("lineage_id"),
            "lineage_verified": bool(self.manifest.get("lineage_id")),
            "source_start_date": self.manifest.get("start_date"),
            "source_end_date": self.manifest.get("end_date"),
            "created_at": datetime.now(UTC).isoformat(),
        }
        target = qlib_dir / "metadata" / "provenance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _normalized_query(sources: str) -> str:
        return f"""
            WITH bars AS ({sources}), deduplicated AS (
                SELECT * FROM bars
                WHERE ts_code IS NOT NULL AND try_cast(trade_time AS TIMESTAMP) IS NOT NULL
                QUALIFY row_number() OVER (
                    PARTITION BY ts_code, try_cast(trade_time AS TIMESTAMP)
                    ORDER BY source_dataset
                ) = 1
            )
            SELECT
                try_cast(trade_time AS TIMESTAMP) AS date,
                upper(split_part(ts_code, '.', 2) || split_part(ts_code, '.', 1)) AS symbol,
                try_cast(open AS DOUBLE) AS open,
                try_cast(high AS DOUBLE) AS high,
                try_cast(low AS DOUBLE) AS low,
                try_cast(close AS DOUBLE) AS close,
                CASE WHEN try_cast(vol AS DOUBLE) > 0
                    THEN try_cast(amount AS DOUBLE) / try_cast(vol AS DOUBLE)
                    ELSE try_cast(close AS DOUBLE) END AS vwap,
                try_cast(vol AS DOUBLE) AS volume,
                1.0::DOUBLE AS factor,
                try_cast(close AS DOUBLE) / lag(try_cast(close AS DOUBLE)) OVER (
                    PARTITION BY ts_code ORDER BY try_cast(trade_time AS TIMESTAMP)
                ) - 1.0 AS change,
                try_cast(amount AS DOUBLE) AS amount,
                0.0::DOUBLE AS paused,
                normalized_oi AS oi
            FROM deduplicated
            WHERE try_cast(open AS DOUBLE) > 0 AND try_cast(close AS DOUBLE) > 0
        """
