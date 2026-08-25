from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .execution_contract import (
    MINUTE_AMOUNT_ROUNDING_TOLERANCE_CNY,
    MINUTE_CANONICALIZATION_POLICY_VERSION,
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_PRICE_TICK_TOLERANCE_CNY,
    MINUTE_SOURCE_UNIT_CONTRACTS,
    MINUTE_VWAP_RELATIVE_TOLERANCE,
    SIMULATION_MINUTE_SOURCE_DATASETS,
    TUSHARE_HAND_SIZE,
)
from .execution_data import (
    MINUTE_DATASETS,
    MINUTE_FREQUENCIES,
    NATIVE_MINUTE_FREQUENCIES,
    QLIB_RESAMPLED_MINUTE_FREQUENCIES,
)
from .path_utils import to_wsl_path
from .qlib_builder import QlibBuilder, _sql_string, build_qlib_output_manifest
from .qlib_minute_resample import QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION
from .snapshot_lineage import verify_snapshot_lineage

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
    "vwap": "source_price_cny_strict_amount_div_volume_else_close",
    "volume": "per_dataset_see_source_unit_contracts",
    "factor": "constant_1_unadjusted",
    "change": "decimal_return",
    "amount": "cny_yuan",
    "paused": "flag_1_when_no_volume",
    "up_limit": "source_price_cny",
    "down_limit": "source_price_cny",
    "oi": "open_interest_contracts",
}

# A full-market minute build can otherwise inherit DuckDB's host-wide defaults
# (roughly 80% of RAM and a relative ``.tmp`` spill directory).  The worker's
# current directory lives on the small container/root filesystem, while the
# staging tree lives on the governed data volume.  Keep both memory and spill
# bounded to this one build so an all-A-share normalization cannot exhaust the
# host or its root disk.
MINUTE_QLIB_DUCKDB_MEMORY_LIMIT = "8GB"
MINUTE_QLIB_DUCKDB_THREADS = 8


class MinuteQlibBuilder:
    """Build native or Qlib-resampled minute data from one immutable snapshot."""

    def __init__(self, snapshot_path: Path, *, target_frequency: str | None = None) -> None:
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
        self.source_minute_audit_sha256: dict[str, str] = {}
        source_audits = (
            quality_gate.get("minute_source_audits", {})
            if isinstance(quality_gate, dict)
            else {}
        )
        if source_audits is not None and not isinstance(source_audits, dict):
            raise ValueError("minute snapshot source-audit registry is invalid")
        for dataset, audit in sorted((source_audits or {}).items()):
            if dataset not in self.manifest.get("datasets", {}):
                continue
            if not isinstance(audit, dict):
                raise ValueError(f"minute snapshot source audit is invalid: {dataset}")
            policy = audit.get("policy")
            if (
                not isinstance(policy, dict)
                or policy.get("version") != MINUTE_CANONICALIZATION_POLICY_VERSION
                or audit.get("audit_status") != "pass_with_canonicalization"
            ):
                raise ValueError(
                    f"minute snapshot source audit policy did not pass: {dataset}"
                )
            claimed_sha256 = str(audit.get("audit_sha256") or "").lower()
            audit_payload = {key: value for key, value in audit.items() if key != "audit_sha256"}
            actual_sha256 = hashlib.sha256(
                json.dumps(audit_payload, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            if claimed_sha256 != actual_sha256:
                raise ValueError(
                    f"minute snapshot source audit digest is inconsistent: {dataset}"
                )
            self.source_minute_audit_sha256[str(dataset)] = claimed_sha256
        self.source_frequency = str(self.manifest.get("frequency") or "")
        if self.source_frequency not in NATIVE_MINUTE_FREQUENCIES:
            raise ValueError(
                "minute Qlib builder requires a supported minute snapshot at "
                "native 1/5-minute frequency"
            )
        self.frequency = str(target_frequency or self.source_frequency).lower()
        if self.frequency not in MINUTE_FREQUENCIES:
            raise ValueError("minute Qlib target frequency is unsupported")
        if self.frequency in NATIVE_MINUTE_FREQUENCIES and self.frequency != self.source_frequency:
            raise ValueError("native minute Qlib output must match the snapshot frequency")
        if self.frequency in QLIB_RESAMPLED_MINUTE_FREQUENCIES:
            source_minutes = int(self.source_frequency.removesuffix("min"))
            target_minutes = int(self.frequency.removesuffix("min"))
            if target_minutes % source_minutes:
                raise ValueError("Qlib resample target must be an integer multiple of the source")
        if not set(self.manifest.get("datasets", {})).intersection(MINUTE_DATASETS):
            raise ValueError("minute snapshot contains no supported bar datasets")
        snapshot_manifest_sha256 = QlibBuilder(
            self.snapshot_path
        )._snapshot_manifest_digest()
        if snapshot_manifest_sha256 != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError("minute snapshot manifest digest changed during validation")
        self.source_lineage_id = str(
            self.manifest.get("source_lineage_id") or ""
        ).lower()
        self.source_lineage_evidence = self.manifest.get("source_lineage_evidence")
        if not self._is_sha256(self.source_lineage_id) or not isinstance(
            self.source_lineage_evidence, dict
        ):
            raise ValueError("minute snapshot has no verified daily-source binding")
        if self.source_lineage_evidence.get("source_lineage_id") != self.source_lineage_id:
            raise ValueError("minute snapshot daily-source lineage evidence disagrees")
        for field in (
            "qlib_dataset_identity_sha256",
            "qlib_dataset_lineage_id",
            "source_snapshot_manifest_sha256",
            "evidence_sha256",
        ):
            if not self._is_sha256(self.source_lineage_evidence.get(field)):
                raise ValueError(f"minute snapshot daily-source evidence has invalid {field}")
        evidence_payload = {
            key: value
            for key, value in self.source_lineage_evidence.items()
            if key != "evidence_sha256"
        }
        evidence_sha256 = hashlib.sha256(
            json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if self.source_lineage_evidence["evidence_sha256"] != evidence_sha256:
            raise ValueError("minute snapshot daily-source evidence digest is inconsistent")
        self.canonicalization_audit: dict[str, object] | None = None

    @property
    def requires_resampling(self) -> bool:
        return self.frequency != self.source_frequency

    def builder_sha256(self) -> str:
        module_root = Path(__file__).resolve().parent
        builder_files = {
            "minute_qlib_builder": Path(__file__).resolve(),
            "execution_contract": module_root / "execution_contract.py",
            "execution_data": module_root / "execution_data.py",
            "qlib_builder": module_root / "qlib_builder.py",
        }
        if self.requires_resampling:
            builder_files.update(
                {
                    "qlib_minute_resample": module_root / "qlib_minute_resample.py",
                    "resample_script": module_root.parents[1]
                    / "scripts"
                    / "resample_minute_qlib.py",
                }
            )
        contract = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(builder_files.items())
        }
        return hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def build_staging(self, staging_path: Path) -> Path:
        staging_path = staging_path.resolve()
        temporary = staging_path.with_name(f".{staging_path.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        partitions = temporary / "partitions"
        by_symbol = temporary / "by_symbol"
        spill_dir = temporary / ".duckdb-spill"
        partitions.mkdir(parents=True)
        by_symbol.mkdir()
        spill_dir.mkdir()
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
        connection: duckdb.DuckDBPyConnection | None = None
        try:
            connection = duckdb.connect()
            connection.execute(
                f"SET memory_limit='{MINUTE_QLIB_DUCKDB_MEMORY_LIMIT}'"
            )
            connection.execute(f"SET threads={MINUTE_QLIB_DUCKDB_THREADS}")
            connection.execute(
                f"SET temp_directory={_sql_string(str(spill_dir.resolve()))}"
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
            if connection is not None:
                connection.close()
            shutil.rmtree(spill_dir, ignore_errors=True)
        audit_counts = {
            "input_rows": 0,
            "output_rows": 0,
            "session_excluded_rows": 0,
            "offgrid_excluded_rows": 0,
            "volume_normalized_rows": 0,
            "relative_fallback_rows": 0,
            "amount_rounding_fallback_rows": 0,
            "price_tick_fallback_rows": 0,
            "nontradable_rows": 0,
        }
        audit_hasher = hashlib.sha256()
        offgrid_symbol_days: set[str] = set()
        for partition in sorted(partitions.glob("symbol=*")):
            symbol = partition.name.split("=", 1)[1]
            files = sorted(partition.glob("*.parquet"))
            frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
            frame.insert(1, "symbol", symbol)
            frame = self._canonicalize_symbol_frame(
                frame,
                audit_counts=audit_counts,
                audit_hasher=audit_hasher,
                offgrid_symbol_days=offgrid_symbol_days,
            )
            if frame.empty:
                continue
            frame.to_parquet(by_symbol / f"{symbol}.parquet", index=False, compression="zstd")
        shutil.rmtree(partitions)
        if not any(by_symbol.glob("*.parquet")):
            raise RuntimeError("minute Qlib staging produced no instruments")
        self.canonicalization_audit = self._finalize_canonicalization_audit(
            audit_counts,
            audit_hasher=audit_hasher,
            offgrid_symbol_days=offgrid_symbol_days,
        )
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
        if self.canonicalization_audit is None:
            raise RuntimeError("minute staging canonicalization audit is missing")
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
        if self.canonicalization_audit is None:
            raise RuntimeError("minute staging canonicalization audit is missing")
        manifest_path = self.snapshot_path / "manifest.json"
        QlibBuilder(self.snapshot_path)._snapshot_manifest_digest()
        builder_digest = self.builder_sha256()
        identity = {
            "snapshot_name": self.snapshot_path.name,
            "snapshot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "qlib_builder_sha256": builder_digest,
            "frequency": self.frequency,
            "source_frequency": self.source_frequency,
            "fields": list(MINUTE_QLIB_FIELDS),
            "field_units": MINUTE_QLIB_FIELD_UNITS,
            "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
            "canonicalization_policy_version": MINUTE_CANONICALIZATION_POLICY_VERSION,
            "canonicalization_audit": self.canonicalization_audit,
            "source_minute_audit_sha256": self.source_minute_audit_sha256,
            "resampled": self.requires_resampling,
            "resample_contract_version": (
                QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION if self.requires_resampling else None
            ),
            "resample_engine": (
                "qlib.utils.resam.resam_calendar" if self.requires_resampling else None
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
        verified_manifest = verify_snapshot_lineage(self.snapshot_path)
        if verified_manifest != self.manifest:
            raise ValueError("execution snapshot lineage verification changed manifest data")
        snapshot_lineage_id = str(self.manifest.get("lineage_id") or "")
        source_lineage_id = self.source_lineage_id
        source_lineage_evidence_sha256 = str(
            self.source_lineage_evidence["evidence_sha256"]
        )
        identity.update(
            {
                "source_lineage_id": source_lineage_id,
                "source_lineage_evidence_sha256": source_lineage_evidence_sha256,
                "source_qlib_dataset_lineage_id": self.source_lineage_evidence[
                    "qlib_dataset_lineage_id"
                ],
            }
        )
        lineage_verified = True
        dataset_lineage_id = (
            hashlib.sha256(
                json.dumps(
                    {
                        "snapshot_lineage_id": snapshot_lineage_id,
                        "qlib_builder_sha256": builder_digest,
                        "frequency": self.frequency,
                        "source_frequency": self.source_frequency,
                        "fields": list(MINUTE_QLIB_FIELDS),
                        "field_units": MINUTE_QLIB_FIELD_UNITS,
                        "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
                        "canonicalization_policy_version": (
                            MINUTE_CANONICALIZATION_POLICY_VERSION
                        ),
                        "canonicalization_audit_sha256": self.canonicalization_audit[
                            "summary_sha256"
                        ],
                        "source_minute_audit_sha256": self.source_minute_audit_sha256,
                        "resample_contract_version": (
                            QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION
                            if self.requires_resampling
                            else None
                        ),
                        "source_unit_contracts": identity["source_unit_contracts"],
                        "source_lineage_id": source_lineage_id,
                        "source_lineage_evidence_sha256": source_lineage_evidence_sha256,
                        "source_qlib_dataset_lineage_id": self.source_lineage_evidence[
                            "qlib_dataset_lineage_id"
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if lineage_verified
            else None
        )
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        provenance = {
            **identity,
            "dataset_identity_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "dataset_lineage_id": dataset_lineage_id,
            "source_snapshot_lineage_id": snapshot_lineage_id or None,
            "source_lineage_id": source_lineage_id or None,
            "source_lineage_evidence": self.source_lineage_evidence,
            "lineage_verified": lineage_verified,
            "source_start_date": self.manifest.get("start_date"),
            "source_end_date": self.manifest.get("end_date"),
            "output_manifest": build_qlib_output_manifest(qlib_dir),
            "created_at": datetime.now(UTC).isoformat(),
        }
        target = qlib_dir / "metadata" / "provenance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    def _canonicalize_symbol_frame(
        self,
        frame: pd.DataFrame,
        *,
        audit_counts: dict[str, int],
        audit_hasher: Any,
        offgrid_symbol_days: set[str],
    ) -> pd.DataFrame:
        """Project one symbol onto the executable minute-bar contract.

        Raw snapshots remain immutable.  Publication excludes rows outside the
        continuous auction and, for a declared 5-minute source, any extra
        1-minute rows returned off the 5-minute grid.  Unit-inconsistent rows
        are retained as explicitly paused/nontradable observations rather than
        silently dropped or assigned a fabricated execution price.
        """

        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame.sort_values(["date", "source_dataset"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        audit_counts["input_rows"] += int(len(frame))
        share = frame["source_dataset"].isin(SIMULATION_MINUTE_SOURCE_DATASETS)
        minute_of_day = frame["date"].dt.hour * 60 + frame["date"].dt.minute
        in_continuous_session = (
            minute_of_day.between(9 * 60 + 30, 11 * 60 + 30)
            | minute_of_day.between(13 * 60 + 1, 15 * 60)
        )
        session_excluded = share & ~in_continuous_session
        self._record_canonicalization_events(
            frame.loc[session_excluded],
            category="session_excluded",
            counts=audit_counts,
            count_key="session_excluded_rows",
            hasher=audit_hasher,
        )
        frame = frame.loc[~session_excluded].copy()

        if self.source_frequency == "5min" and not frame.empty:
            share = frame["source_dataset"].isin(SIMULATION_MINUTE_SOURCE_DATASETS)
            minute_of_day = frame["date"].dt.hour * 60 + frame["date"].dt.minute
            on_grid = (
                minute_of_day.mod(5).eq(0)
                & frame["date"].dt.second.eq(0)
                & frame["date"].dt.microsecond.eq(0)
            )
            offgrid = share & ~on_grid
            offgrid_frame = frame.loc[offgrid]
            for row in offgrid_frame[["symbol", "date"]].itertuples(index=False):
                offgrid_symbol_days.add(f"{row.symbol}|{row.date.date().isoformat()}")
            self._record_canonicalization_events(
                offgrid_frame,
                category="offgrid_excluded",
                counts=audit_counts,
                count_key="offgrid_excluded_rows",
                hasher=audit_hasher,
            )
            frame = frame.loc[~offgrid].copy()

        if frame.empty:
            return frame.drop(columns=["source_dataset"], errors="ignore")

        share = frame["source_dataset"].isin(SIMULATION_MINUTE_SOURCE_DATASETS)
        source_volume = pd.to_numeric(frame["source_volume"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        amount = pd.to_numeric(frame["amount"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        positive_volume = volume.gt(0)
        volume_normalized = (
            share
            & source_volume.gt(0)
            & volume.notna()
            & ~volume.eq(source_volume)
        )
        valid_envelope = low.gt(0) & high.gt(0)
        implied_price = amount / volume.where(positive_volume)
        strict_match = (
            valid_envelope
            & amount.gt(0)
            & implied_price.ge(low)
            & implied_price.le(high)
        )
        relative = MINUTE_VWAP_RELATIVE_TOLERANCE
        relative_match = (
            valid_envelope
            & amount.gt(0)
            & implied_price.ge(low * (1.0 - relative))
            & implied_price.le(high * (1.0 + relative))
        )
        amount_distance = pd.concat(
            [
                (low * volume - amount).clip(lower=0),
                (amount - high * volume).clip(lower=0),
            ],
            axis=1,
        ).max(axis=1)
        amount_rounding = (
            share
            & positive_volume
            & valid_envelope
            & amount.gt(0)
            & ~relative_match
            & amount_distance.le(MINUTE_AMOUNT_ROUNDING_TOLERANCE_CNY + 1e-12)
        )
        price_distance = pd.concat(
            [
                (low - implied_price).clip(lower=0),
                (implied_price - high).clip(lower=0),
            ],
            axis=1,
        ).max(axis=1)
        tick_fallback = (
            share
            & positive_volume
            & valid_envelope
            & amount.gt(0)
            & ~relative_match
            & ~amount_rounding
            & price_distance.le(MINUTE_PRICE_TICK_TOLERANCE_CNY + 1e-12)
        )
        accepted = relative_match | amount_rounding | tick_fallback
        relative_fallback = share & positive_volume & relative_match & ~strict_match
        nontradable = share & positive_volume & (
            ~valid_envelope | amount.isna() | amount.le(0) | ~accepted
        )
        self._record_canonicalization_events(
            frame.loc[volume_normalized],
            category="volume_normalized",
            counts=audit_counts,
            count_key="volume_normalized_rows",
            hasher=audit_hasher,
        )
        self._record_canonicalization_events(
            frame.loc[relative_fallback],
            category="relative_fallback",
            counts=audit_counts,
            count_key="relative_fallback_rows",
            hasher=audit_hasher,
        )
        self._record_canonicalization_events(
            frame.loc[amount_rounding],
            category="amount_rounding_fallback",
            counts=audit_counts,
            count_key="amount_rounding_fallback_rows",
            hasher=audit_hasher,
        )
        self._record_canonicalization_events(
            frame.loc[tick_fallback],
            category="price_tick_fallback",
            counts=audit_counts,
            count_key="price_tick_fallback_rows",
            hasher=audit_hasher,
        )
        self._record_canonicalization_events(
            frame.loc[nontradable],
            category="nontradable",
            counts=audit_counts,
            count_key="nontradable_rows",
            hasher=audit_hasher,
        )

        frame.loc[nontradable, "volume"] = 0.0
        frame.loc[nontradable, "amount"] = 0.0
        frame["vwap"] = frame["close"].astype(float)
        amount_vwap = share & positive_volume & strict_match & ~nontradable
        frame.loc[amount_vwap, "vwap"] = implied_price.loc[amount_vwap]
        frame["paused"] = frame["volume"].isna() | frame["volume"].le(0)
        frame["paused"] = frame["paused"].astype(float)
        frame.sort_values("date", inplace=True)
        frame["change"] = (
            frame["close"].astype(float) / frame["close"].astype(float).shift(1) - 1.0
        )
        frame.drop(columns=["source_dataset", "source_volume"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        audit_counts["output_rows"] += int(len(frame))
        return frame

    @staticmethod
    def _record_canonicalization_events(
        frame: pd.DataFrame,
        *,
        category: str,
        counts: dict[str, int],
        count_key: str,
        hasher: Any,
    ) -> None:
        counts[count_key] += int(len(frame))
        if frame.empty:
            return
        fields = [
            "source_dataset",
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source_volume",
            "amount",
        ]
        ordered = frame[fields].sort_values(["symbol", "date", "source_dataset"])
        for row in ordered.itertuples(index=False, name=None):
            payload = {
                "category": category,
                "source_dataset": str(row[0]),
                "symbol": str(row[1]),
                "date": pd.Timestamp(row[2]).isoformat(),
                "open": None if pd.isna(row[3]) else float(row[3]),
                "high": None if pd.isna(row[4]) else float(row[4]),
                "low": None if pd.isna(row[5]) else float(row[5]),
                "close": None if pd.isna(row[6]) else float(row[6]),
                "volume": None if pd.isna(row[7]) else float(row[7]),
                "source_volume": None if pd.isna(row[8]) else float(row[8]),
                "amount": None if pd.isna(row[9]) else float(row[9]),
            }
            hasher.update(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )

    @staticmethod
    def _finalize_canonicalization_audit(
        counts: dict[str, int],
        *,
        audit_hasher: Any,
        offgrid_symbol_days: set[str],
    ) -> dict[str, object]:
        summary: dict[str, object] = {
            "policy_version": MINUTE_CANONICALIZATION_POLICY_VERSION,
            "rules": {
                "share_sessions": ["09:30-11:30", "13:01-15:00"],
                "five_minute_grid": "minute_mod_5_eq_0",
                "vwap_relative_tolerance": MINUTE_VWAP_RELATIVE_TOLERANCE,
                "amount_rounding_tolerance_cny": (
                    MINUTE_AMOUNT_ROUNDING_TOLERANCE_CNY
                ),
                "price_tick_tolerance_cny": MINUTE_PRICE_TICK_TOLERANCE_CNY,
                "severe_row_policy": "paused_nontradable_zero_volume_amount",
            },
            **{key: int(value) for key, value in sorted(counts.items())},
            "offgrid_symbol_days": len(offgrid_symbol_days),
            "event_sha256": audit_hasher.hexdigest(),
        }
        summary["summary_sha256"] = hashlib.sha256(
            json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return summary

    @staticmethod
    def _is_sha256(value: object) -> bool:
        normalized = str(value or "").lower()
        return len(normalized) == 64 and all(
            character in "0123456789abcdef" for character in normalized
        )

    @staticmethod
    def _normalized_query(sources: str, limit_glob: str) -> str:
        share_volume = MinuteQlibBuilder._normalized_share_volume_expression()
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
                          AND ({share_volume}) > 0
                    THEN try_cast(amount AS DOUBLE) / ({share_volume})
                    ELSE try_cast(close AS DOUBLE) END AS vwap,
                CASE WHEN source_dataset IN ('ashare_5m', 'liquid_stocks_1m', 'etf_1m')
                    THEN ({share_volume})
                    ELSE try_cast(vol AS DOUBLE) END AS volume,
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
                normalized_oi AS oi,
                source_dataset AS source_dataset,
                try_cast(vol AS DOUBLE) AS source_volume
            FROM deduplicated
            LEFT JOIN limits l
              ON deduplicated.ts_code = l.ts_code
             AND try_cast(deduplicated.trade_time AS DATE) = l.trade_date
            WHERE try_cast(open AS DOUBLE) > 0 AND try_cast(close AS DOUBLE) > 0
        """

    @staticmethod
    def _normalized_share_volume_expression() -> str:
        direct_match = MinuteQlibBuilder._share_amount_match_expression(
            "try_cast(vol AS DOUBLE)"
        )
        normalized_match = MinuteQlibBuilder._share_amount_match_expression(
            f"(try_cast(vol AS DOUBLE) / {TUSHARE_HAND_SIZE})"
        )
        return f"""
            CASE
                WHEN try_cast(vol AS DOUBLE) > 0
                  AND try_cast(amount AS DOUBLE) > 0
                  AND NOT ({direct_match})
                  AND ({normalized_match})
                THEN try_cast(vol AS DOUBLE) / {TUSHARE_HAND_SIZE}
                ELSE try_cast(vol AS DOUBLE)
            END
        """

    @staticmethod
    def _share_amount_match_expression(volume_expression: str) -> str:
        amount = "try_cast(amount AS DOUBLE)"
        low = "try_cast(low AS DOUBLE)"
        high = "try_cast(high AS DOUBLE)"
        relative = MINUTE_VWAP_RELATIVE_TOLERANCE
        rounding = MINUTE_AMOUNT_ROUNDING_TOLERANCE_CNY
        tick = MINUTE_PRICE_TICK_TOLERANCE_CNY
        return f"""
            {low} > 0
            AND {high} > 0
            AND (
                {amount} / {volume_expression}
                    BETWEEN {low} * {1.0 - relative}
                        AND {high} * {1.0 + relative}
                OR {amount}
                    BETWEEN {low} * {volume_expression} - {rounding}
                        AND {high} * {volume_expression} + {rounding}
                OR {amount} / {volume_expression}
                    BETWEEN {low} - {tick}
                        AND {high} + {tick}
            )
        """
