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

from .execution_contract import (
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
)
from .execution_data import (
    MINUTE_DATASETS,
    MINUTE_FREQUENCIES,
    NATIVE_MINUTE_FREQUENCIES,
    QLIB_RESAMPLED_MINUTE_FREQUENCIES,
)
from .path_utils import to_wsl_path
from .qlib_builder import QlibBuilder, _sql_string
from .qlib_minute_resample import QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION

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
    "up_limit",
    "down_limit",
    "oi",
)

# Explicit per-field unit declarations written into the dataset provenance.
# Minute bars keep source (unadjusted) CNY prices; volume units are declared
# per source dataset in MINUTE_SOURCE_UNIT_CONTRACTS.
MINUTE_QLIB_FIELD_UNITS = {
    "open": "source_price_cny",
    "high": "source_price_cny",
    "low": "source_price_cny",
    "close": "source_price_cny",
    "vwap": "source_price_cny_amount_div_volume",
    "volume": "per_dataset_see_source_unit_contracts",
    "factor": "constant_1_unadjusted",
    "change": "decimal_return",
    "amount": "cny_yuan",
    "paused": "flag_1_when_no_volume",
    "up_limit": "source_price_cny",
    "down_limit": "source_price_cny",
    "oi": "open_interest_contracts",
}


class MinuteQlibBuilder:
    """Build native or Qlib-resampled minute data from one immutable snapshot."""

    def __init__(
        self, snapshot_path: Path, *, target_frequency: str | None = None
    ) -> None:
        self.snapshot_path = snapshot_path.resolve()
        manifest_path = self.snapshot_path / "manifest.json"
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError("minute snapshot manifest is missing or invalid") from exc
        quality_gate = self.manifest.get("quality_gate")
        if quality_gate is not None and (
            not isinstance(quality_gate, dict) or quality_gate.get("ok") is not True
        ):
            raise ValueError("minute snapshot quality gate did not pass")
        self.source_frequency = str(self.manifest.get("frequency") or "")
        if self.source_frequency not in NATIVE_MINUTE_FREQUENCIES:
            raise ValueError(
                "minute Qlib builder requires a supported minute snapshot at "
                "native 1/5-minute frequency"
            )
        self.frequency = str(target_frequency or self.source_frequency).lower()
        if self.frequency not in MINUTE_FREQUENCIES:
            raise ValueError("minute Qlib target frequency is unsupported")
        if (
            self.frequency in NATIVE_MINUTE_FREQUENCIES
            and self.frequency != self.source_frequency
        ):
            raise ValueError("native minute Qlib output must match the snapshot frequency")
        if self.frequency in QLIB_RESAMPLED_MINUTE_FREQUENCIES:
            source_minutes = int(self.source_frequency.removesuffix("min"))
            target_minutes = int(self.frequency.removesuffix("min"))
            if target_minutes % source_minutes:
                raise ValueError(
                    "Qlib resample target must be an integer multiple of the source"
                )
        if not set(self.manifest.get("datasets", {})).intersection(MINUTE_DATASETS):
            raise ValueError("minute snapshot contains no supported bar datasets")

    @property
    def requires_resampling(self) -> bool:
        return self.frequency != self.source_frequency

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
                "SELECT ts_code, trade_time, open, high, low, close, vol, amount, "
                f"{oi} AS normalized_oi, {_sql_string(dataset)} AS source_dataset "
                f"FROM read_parquet({glob}, hive_partitioning=true, union_by_name=true)"
            )
        if not sources:
            raise FileNotFoundError("snapshot does not contain minute Parquet data")
        limit_root = self.snapshot_path / "parquet" / "stk_limit"
        if not limit_root.exists() or not any(limit_root.rglob("*.parquet")):
            raise FileNotFoundError(
                "minute execution snapshot does not contain daily A-share price limits"
            )
        limit_glob = _sql_string(str((limit_root / "**" / "*.parquet").resolve()))
        query = self._normalized_query(" UNION ALL ".join(sources), limit_glob)
        connection = duckdb.connect()
        try:
            invalid_share_units = connection.execute(
                self._invalid_share_unit_query(" UNION ALL ".join(sources))
            ).fetchone()[0]
            if invalid_share_units:
                raise RuntimeError(
                    f"{invalid_share_units} stock/ETF minute rows violate the "
                    "share-volume/CNY-amount contract"
                )
            missing_controls = connection.execute(
                f"SELECT count(*) FROM ({query}) WHERE up_limit IS NULL OR down_limit IS NULL"
            ).fetchone()[0]
            if missing_controls:
                raise RuntimeError(
                    f"{missing_controls} minute rows have no same-lineage daily price limits"
                )
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

    def resample_staging(
        self,
        *,
        native_by_symbol: Path,
        staging_path: Path,
        qlib_python: str,
        wsl_distro: str,
    ) -> Path:
        if not self.requires_resampling:
            raise ValueError("native minute output does not require Qlib resampling")
        script = Path(__file__).resolve().parents[2] / "scripts" / "resample_minute_qlib.py"
        if not script.is_file():
            raise FileNotFoundError(f"Qlib minute resample script not found: {script}")
        staging_path = staging_path.resolve()
        temporary = staging_path.with_name(f".{staging_path.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        output = temporary / "by_symbol"
        command = (
            [
                "wsl",
                "-d",
                wsl_distro,
                "--exec",
                qlib_python,
                to_wsl_path(script),
                "--source",
                to_wsl_path(native_by_symbol),
                "--output",
                to_wsl_path(output),
            ]
            if os.name == "nt" and qlib_python.startswith("/")
            else [
                qlib_python,
                str(script),
                "--source",
                str(native_by_symbol),
                "--output",
                str(output),
            ]
        )
        command.extend(
            [
                "--source-frequency",
                self.source_frequency,
                "--target-frequency",
                self.frequency,
            ]
        )
        try:
            subprocess.run(command, check=True)
            if not any(output.glob("*.parquet")):
                raise RuntimeError("Qlib minute resampling produced no instrument files")
            if staging_path.exists():
                shutil.rmtree(staging_path)
            os.replace(temporary, staging_path)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
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
                self.frequency,
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
            if not any((qlib_dir / "features").rglob(f"*.{self.frequency}.bin")):
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
        builder_files = [Path(__file__)]
        if self.requires_resampling:
            builder_files.extend(
                [
                    Path(__file__).with_name("qlib_minute_resample.py"),
                    Path(__file__).resolve().parents[2]
                    / "scripts"
                    / "resample_minute_qlib.py",
                ]
            )
        builder_digest = hashlib.sha256(
            b"".join(path.read_bytes() for path in builder_files)
        ).hexdigest()
        identity = {
            "snapshot_name": self.snapshot_path.name,
            "snapshot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "qlib_builder_sha256": builder_digest,
            "frequency": self.frequency,
            "source_frequency": self.source_frequency,
            "fields": list(MINUTE_QLIB_FIELDS),
            "field_units": MINUTE_QLIB_FIELD_UNITS,
            "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
            "resampled": self.requires_resampling,
            "resample_contract_version": (
                QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION
                if self.requires_resampling
                else None
            ),
            "resample_engine": (
                "qlib.utils.resam.resam_calendar"
                if self.requires_resampling
                else None
            ),
            "source_datasets": sorted(
                dataset
                for dataset in MINUTE_DATASETS
                if dataset in self.manifest.get("datasets", {})
            ),
        }
        identity["source_unit_contracts"] = {
            dataset: MINUTE_SOURCE_UNIT_CONTRACTS[dataset]
            for dataset in identity["source_datasets"]
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
    def _normalized_query(sources: str, limit_glob: str) -> str:
        return f"""
            WITH bars AS ({sources}), deduplicated AS (
                SELECT * FROM bars
                WHERE ts_code IS NOT NULL AND try_cast(trade_time AS TIMESTAMP) IS NOT NULL
                QUALIFY row_number() OVER (
                    PARTITION BY ts_code, try_cast(trade_time AS TIMESTAMP)
                    ORDER BY source_dataset
                ) = 1
            ), limits AS (
                SELECT ts_code, try_cast(trade_date AS DATE) AS trade_date,
                       try_cast(up_limit AS DOUBLE) AS up_limit,
                       try_cast(down_limit AS DOUBLE) AS down_limit
                FROM read_parquet({limit_glob}, hive_partitioning=true, union_by_name=true)
                QUALIFY row_number() OVER (
                    PARTITION BY ts_code, try_cast(trade_date AS DATE)
                    ORDER BY try_cast(trade_date AS DATE)
                ) = 1
            )
            SELECT
                try_cast(trade_time AS TIMESTAMP) AS date,
                upper(split_part(deduplicated.ts_code, '.', 2) ||
                      split_part(deduplicated.ts_code, '.', 1)) AS symbol,
                try_cast(open AS DOUBLE) AS open,
                try_cast(high AS DOUBLE) AS high,
                try_cast(low AS DOUBLE) AS low,
                try_cast(close AS DOUBLE) AS close,
                CASE WHEN source_dataset IN ('ashare_5m', 'liquid_stocks_1m', 'etf_1m')
                          AND try_cast(vol AS DOUBLE) > 0
                    THEN try_cast(amount AS DOUBLE) / try_cast(vol AS DOUBLE)
                    ELSE try_cast(close AS DOUBLE) END AS vwap,
                try_cast(vol AS DOUBLE) AS volume,
                1.0::DOUBLE AS factor,
                try_cast(close AS DOUBLE) / lag(try_cast(close AS DOUBLE)) OVER (
                    PARTITION BY deduplicated.ts_code
                    ORDER BY try_cast(deduplicated.trade_time AS TIMESTAMP)
                ) - 1.0 AS change,
                try_cast(amount AS DOUBLE) AS amount,
                CASE WHEN try_cast(vol AS DOUBLE) IS NULL OR try_cast(vol AS DOUBLE) <= 0
                    THEN 1.0 ELSE 0.0 END AS paused,
                CASE WHEN source_dataset IN ('ashare_5m', 'liquid_stocks_1m', 'etf_1m')
                    THEN l.up_limit ELSE 99999.0 END AS up_limit,
                CASE WHEN source_dataset IN ('ashare_5m', 'liquid_stocks_1m', 'etf_1m')
                    THEN l.down_limit ELSE 0.0 END AS down_limit,
                normalized_oi AS oi
            FROM deduplicated
            LEFT JOIN limits l
              ON deduplicated.ts_code = l.ts_code
             AND try_cast(deduplicated.trade_time AS DATE) = l.trade_date
            WHERE try_cast(open AS DOUBLE) > 0 AND try_cast(close AS DOUBLE) > 0
        """

    @staticmethod
    def _invalid_share_unit_query(sources: str) -> str:
        return f"""
            WITH bars AS ({sources})
            SELECT count(*)
            FROM bars
            WHERE source_dataset IN ('ashare_5m', 'liquid_stocks_1m', 'etf_1m')
              AND try_cast(vol AS DOUBLE) > 0
              AND (
                  try_cast(amount AS DOUBLE) IS NULL
                  OR try_cast(amount AS DOUBLE) <= 0
                  OR try_cast(low AS DOUBLE) <= 0
                  OR try_cast(high AS DOUBLE) <= 0
                  OR try_cast(amount AS DOUBLE) / try_cast(vol AS DOUBLE)
                     NOT BETWEEN try_cast(low AS DOUBLE) * 0.95
                         AND try_cast(high AS DOUBLE) * 1.05
              )
        """
