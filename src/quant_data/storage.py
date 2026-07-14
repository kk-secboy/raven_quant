from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .models import ProviderResult, UnitResult

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
}
DATETIME_COLUMNS = {"pub_time", "datetime", "trade_time"}


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
    def __init__(self, root: Path, *, keep_raw: bool = False) -> None:
        self.root = root
        self.keep_raw = keep_raw
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
                source_sql = (
                    f"SELECT DISTINCT * FROM read_parquet({quoted_paths}, union_by_name=true)"
                )
                if date_field:
                    export_sql = (
                        f"SELECT *, year({_identifier(date_field)})::INTEGER AS partition_year, "
                        f"month({_identifier(date_field)})::INTEGER AS partition_month "
                        f"FROM ({source_sql}) "
                        f"WHERE {_identifier(date_field)} IS NOT NULL"
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
                    "source_sha256": source_sha256,
                    "source_units": source_identity,
                    "files": files,
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
        "ann_date",
        "pub_time",
        "datetime",
        "trade_time",
        "start_date",
        "in_date",
        "end_date",
        "out_date",
    )


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
