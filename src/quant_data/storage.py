from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .availability import availability_contract_label, recoverability_level
from .models import ProviderResult, UnitResult
from .reference_data import reference_manifest_metadata

DATE_COLUMNS = {
    "trade_date",
    "cal_date",
    "pretrade_date",
    "list_date",
    "delist_date",
    "ann_date",
    "f_ann_date",
    "end_date",
    "actual_date",
    "modify_date",
    "pre_date",
    "start_date",
    "in_date",
    "out_date",
    "pub_date",
    "imp_date",
    "publish_date",
    "change_date",
    "ipo_date",
    "issue_date",
    "surv_date",
    "nav_date",
    "date",
}
DATETIME_COLUMNS = {"pub_time", "publish_time", "datetime", "trade_time"}


def _normalize_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows, columns=columns or None)
    for column in frame.columns:
        if column in DATE_COLUMNS:
            values = frame[column].astype("string").str.replace(r"\.0$", "", regex=True)
            frame[column] = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
        elif column in DATETIME_COLUMNS:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


class ParquetStore:
    def __init__(
        self,
        root: Path,
        *,
        keep_raw: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.root = root
        self.keep_raw = keep_raw
        # Row-level ingestion timestamp (design draft 3.3 ``ingested_at``):
        # recorded once per written unit, tz-aware UTC, so every parquet row can
        # answer "when did the platform actually obtain this row".
        self._clock = clock or (lambda: datetime.now(UTC))
        self.units_root = root / "units"
        self.raw_root = root / "raw"
        self.snapshots_root = root / "snapshots"
        self.units_root.mkdir(parents=True, exist_ok=True)

    def write_unit(self, dataset: str, unit_key: str, result: ProviderResult) -> UnitResult:
        directory = self.units_root / dataset
        directory.mkdir(parents=True, exist_ok=True)
        if not result.rows:
            return self._write_empty_marker(dataset, unit_key, result)
        target = directory / f"{unit_key}.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        frame = _normalize_frame(result.rows, result.columns)
        ingested_at = self._clock()
        if ingested_at.tzinfo is None:
            ingested_at = ingested_at.replace(tzinfo=UTC)
        frame["ingested_at"] = pd.Timestamp(ingested_at)
        frame.to_parquet(temporary, index=False, compression="zstd", engine="pyarrow")
        os.replace(temporary, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if self.keep_raw:
            self._write_raw(dataset, unit_key, result.raw_body)
        return UnitResult(
            output_path=target.relative_to(self.root).as_posix(),
            row_count=len(frame),
            sha256=digest,
        )

    def _write_empty_marker(
        self, dataset: str, unit_key: str, result: ProviderResult
    ) -> UnitResult:
        target = self.units_root / dataset / f"{unit_key}.empty.json"
        temporary = target.with_suffix(".empty.json.tmp")
        payload = {
            "api_name": result.api_name,
            "columns": result.columns,
            "metadata": result.metadata,
            "empty": True,
        }
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if self.keep_raw:
            self._write_raw(dataset, unit_key, result.raw_body)
        return UnitResult(
            output_path=target.relative_to(self.root).as_posix(),
            row_count=0,
            sha256=digest,
        )

    def _write_raw(self, dataset: str, unit_key: str, body: bytes) -> None:
        directory = self.raw_root / dataset
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{unit_key}.json.gz"
        temporary = target.with_suffix(".json.gz.tmp")
        with gzip.open(temporary, "wb", compresslevel=1) as stream:
            stream.write(body)
        os.replace(temporary, target)

    def read_units(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        frames = []
        for row in rows:
            output_path = row.get("output_path")
            if output_path and str(output_path).endswith(".parquet"):
                frames.append(pd.read_parquet(self.root / output_path))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def build_snapshot(
        self,
        *,
        name: str,
        successful_units: dict[str, list[dict[str, Any]]],
        manifest_extra: dict[str, Any],
    ) -> Path:
        target = self.snapshots_root / name
        temporary = self.snapshots_root / f".{name}.tmp"
        if target.exists():
            raise FileExistsError(f"snapshot already exists: {target}")
        if temporary.exists():
            shutil.rmtree(temporary)
        (temporary / "parquet").mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "name": name,
            "created_at": datetime.now(UTC).isoformat(),
            "datasets": {},
            **manifest_extra,
        }
        connection = duckdb.connect()
        try:
            for dataset, rows in sorted(successful_units.items()):
                refresh_metadata = reference_manifest_metadata(rows)
                source_identity = [
                    {
                        "unit_key": str(row["unit_key"]),
                        "sha256": str(row.get("sha256") or ""),
                        "row_count": int(row.get("row_count") or 0),
                    }
                    for row in sorted(rows, key=lambda item: str(item["unit_key"]))
                ]
                source_sha256 = hashlib.sha256(
                    json.dumps(
                        source_identity,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                paths = [
                    str((self.root / row["output_path"]).resolve())
                    for row in rows
                    if str(row["output_path"]).endswith(".parquet")
                ]
                if not paths:
                    manifest["datasets"][dataset] = {
                        "rows": 0,
                        "unit_files": 0,
                        "empty_units": len(rows),
                        "date_field": None,
                        "date_min": None,
                        "date_max": None,
                        "ingested_at_min": None,
                        "ingested_at_max": None,
                        "recoverability": recoverability_level(dataset),
                        "availability_policy": availability_contract_label(dataset),
                        "reference_refresh": refresh_metadata,
                        "source_sha256": source_sha256,
                        "source_units": source_identity,
                        "files": [],
                    }
                    continue
                dataset_dir = temporary / "parquet" / dataset
                dataset_dir.mkdir(parents=True, exist_ok=True)
                quoted_paths = "[" + ",".join(_sql_string(path) for path in paths) + "]"
                columns = (
                    connection.execute(
                        f"DESCRIBE SELECT * FROM read_parquet({quoted_paths}, union_by_name=true)"
                    )
                    .fetchdf()["column_name"]
                    .tolist()
                )
                date_field = next(
                    (field for field in _date_field_candidates(dataset) if field in columns),
                    None,
                )
                source_sql = _snapshot_source_query(dataset, quoted_paths, set(columns))
                date_min = None
                date_max = None
                if date_field:
                    date_expression = _date_sql_expression(date_field)
                    date_min, date_max = connection.execute(
                        f"SELECT min({date_expression})::VARCHAR, max({date_expression})::VARCHAR "
                        f"FROM ({source_sql}) WHERE {date_expression} IS NOT NULL"
                    ).fetchone()
                    export_sql = (
                        f"SELECT *, year({date_expression})::INTEGER AS partition_year, "
                        f"month({date_expression})::INTEGER AS partition_month "
                        f"FROM ({source_sql}) "
                        f"WHERE {date_expression} IS NOT NULL"
                    )
                    connection.execute(
                        f"COPY ({export_sql}) TO {_sql_string(str(dataset_dir))} "
                        "(FORMAT PARQUET, COMPRESSION ZSTD, "
                        "PARTITION_BY (partition_year,partition_month), "
                        "ROW_GROUP_SIZE 100000)"
                    )
                else:
                    connection.execute(
                        "COPY ({query}) TO {target} "
                        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)".format(
                            query=source_sql,
                            target=_sql_string(str(dataset_dir / "data.parquet")),
                        )
                    )
                row_count = connection.execute(f"SELECT count(*) FROM ({source_sql})").fetchone()[0]
                # Row-level ingestion provenance: old units predate the column
                # and merge as NULL via union_by_name, so a NULL bound simply
                # means "ingested before this column existed".
                ingested_min = None
                ingested_max = None
                if "ingested_at" in columns:
                    ingested_min, ingested_max = connection.execute(
                        "SELECT min(ingested_at)::VARCHAR, max(ingested_at)::VARCHAR "
                        f"FROM ({source_sql}) WHERE ingested_at IS NOT NULL"
                    ).fetchone()
                files = [
                    {
                        "path": path.relative_to(temporary).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                    for path in sorted(dataset_dir.rglob("*.parquet"))
                ]
                manifest["datasets"][dataset] = {
                    "rows": int(row_count),
                    "unit_files": len(paths),
                    "empty_units": len(rows) - len(paths),
                    "date_field": date_field,
                    "date_min": date_min,
                    "date_max": date_max,
                    "ingested_at_min": ingested_min,
                    "ingested_at_max": ingested_max,
                    "recoverability": recoverability_level(dataset),
                    "availability_policy": availability_contract_label(dataset),
                    "reference_refresh": refresh_metadata,
                    "source_sha256": source_sha256,
                    "source_units": source_identity,
                    "files": files,
                }
            historical = {
                dataset: {
                    "date_field": details["date_field"],
                    "date_min": details["date_min"],
                    "date_max": details["date_max"],
                    "source_sha256": details["source_sha256"],
                }
                for dataset, details in manifest["datasets"].items()
                if details.get("date_min") and str(details["date_min"]) < "2024-01-01"
            }
            manifest["coverage_audit"] = {
                "historical_before_2024_count": len(historical),
                "historical_before_2024": historical,
                "versioned_reference_count": sum(
                    1
                    for details in manifest["datasets"].values()
                    if details.get("reference_refresh")
                ),
            }
        finally:
            connection.close()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.snapshots_root.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
        return target


def _date_field_candidates(dataset: str) -> tuple[str, ...]:
    if dataset == "trade_cal":
        return ("cal_date",)
    if dataset == "stock_basic":
        return ()
    if dataset in {"income", "balancesheet", "cashflow", "fina_indicator", "forecast", "express"}:
        return ("ann_date", "f_ann_date", "end_date")
    return (
        "trade_date",
        "cal_date",
        "ann_date",
        "pub_time",
        "publish_time",
        "pub_date",
        "imp_date",
        "date",
        "datetime",
        "trade_time",
        "month",
        "MONTH",
        "quarter",
        "publish_date",
        "change_date",
        "ipo_date",
        "issue_date",
        "surv_date",
        "nav_date",
        "start_date",
        "in_date",
        "end_date",
        "out_date",
    )


def _date_sql_expression(field: str) -> str:
    value = f"CAST({_identifier(field)} AS VARCHAR)"
    if field in {"month", "MONTH"}:
        return f"try_strptime(regexp_replace({value}, '[^0-9]', '', 'g'), '%Y%m')::DATE"
    if field == "quarter":
        return (
            f"CASE WHEN regexp_matches({value}, '^[0-9]{{4}}Q[1-4]$') THEN "
            f"make_date(CAST(substr({value}, 1, 4) AS INTEGER), "
            f"(CAST(substr({value}, 6, 1) AS INTEGER) - 1) * 3 + 1, 1) "
            f"ELSE try_cast({_identifier(field)} AS DATE) END"
        )
    return (
        f"coalesce(try_cast({_identifier(field)} AS DATE), "
        f"try_strptime({value}, '%Y%m%d')::DATE, "
        f"try_strptime({value}, '%Y-%m-%d %H:%M:%S')::DATE)"
    )


def _snapshot_source_query(dataset: str, quoted_paths: str, columns: set[str]) -> str:
    base = f"SELECT DISTINCT * FROM read_parquet({quoted_paths}, union_by_name=true)"
    news_identity = {"datetime", "content", "title", "source"}
    if dataset != "news" or not news_identity.issubset(columns):
        return base
    # Keep every explicitly sourced record. Legacy all-day news units did not
    # persist the source; retain those only when no new source-aware window has
    # supplied the same timestamp/title/content. This lets immutable old units
    # remain on disk without duplicating repaired snapshots.
    return f"""
        WITH source_rows AS ({base})
        SELECT candidate.*
        FROM source_rows AS candidate
        WHERE candidate.source IS NOT NULL
           OR NOT EXISTS (
                SELECT 1
                FROM source_rows AS tagged
                WHERE tagged.source IS NOT NULL
                  AND candidate.datetime IS NOT DISTINCT FROM tagged.datetime
                  AND candidate.title IS NOT DISTINCT FROM tagged.title
                  AND candidate.content IS NOT DISTINCT FROM tagged.content
           )
    """


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
