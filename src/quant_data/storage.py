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
        base_snapshot: Path | None = None,
        duckdb_memory_limit: str = "4GB",
        duckdb_threads: int = 4,
    ) -> Path:
        target = self.snapshots_root / name
        temporary = self.snapshots_root / f".{name}.tmp"
        if target.exists():
            raise FileExistsError(f"snapshot already exists: {target}")
        if temporary.exists():
            shutil.rmtree(temporary)
        (temporary / "parquet").mkdir(parents=True, exist_ok=True)

        # Incremental base: when a compatible parent snapshot is supplied,
        # partitions untouched by new units are hard-linked instead of being
        # re-merged. All DISTINCT merges run per (dataset, partition) with a
        # bounded duckdb memory budget and on-disk spill, so peak memory is a
        # function of one partition instead of the whole lake.
        base_root: Path | None = None
        base_manifest: dict[str, Any] = {}
        if base_snapshot is not None:
            try:
                base_manifest = json.loads(
                    (base_snapshot / "manifest.json").read_text(encoding="utf-8")
                )
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                raise ValueError(f"base snapshot {base_snapshot} is incomplete") from exc
            base_root = base_snapshot / "parquet"

        manifest: dict[str, Any] = {
            "name": name,
            "created_at": datetime.now(UTC).isoformat(),
            "datasets": {},
            **manifest_extra,
        }
        connection = duckdb.connect()
        try:
            connection.execute(f"SET memory_limit='{duckdb_memory_limit}'")
            connection.execute(f"SET threads={int(duckdb_threads)}")
            spill_dir = temporary / ".duckdb-spill"
            spill_dir.mkdir(exist_ok=True)
            connection.execute(f"SET temp_directory={_sql_string(str(spill_dir))}")
            for dataset, rows in sorted(successful_units.items()):
                base_entry = None
                if base_root is not None:
                    base_entry = (base_manifest.get("datasets") or {}).get(dataset)
                    if not isinstance(base_entry, dict):
                        base_entry = None
                manifest["datasets"][dataset] = self._build_dataset_snapshot(
                    connection,
                    dataset,
                    rows,
                    temporary,
                    base_root,
                    base_entry,
                )
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

    def _build_dataset_snapshot(
        self,
        connection: duckdb.DuckDBPyConnection,
        dataset: str,
        rows: list[dict[str, Any]],
        temporary: Path,
        base_root: Path | None,
        base_entry: dict[str, Any] | None,
    ) -> dict[str, Any]:
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
        empty_entry = {
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
        if not paths:
            return empty_entry

        current_tuples = {
            (item["unit_key"], item["sha256"], item["row_count"]) for item in source_identity
        }
        base_dir = base_root / dataset if base_root is not None else None
        dataset_dir = temporary / "parquet" / dataset
        if (
            base_entry is not None
            and base_dir is not None
            and base_dir.exists()
            and current_tuples == _source_unit_tuples(base_entry)
        ):
            # Dataset untouched by this build: hard-link the parent's files and
            # reuse its manifest entry verbatim (linked bytes are identical).
            _link_tree(base_dir, dataset_dir)
            return dict(base_entry)

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
        news_identity = {"datetime", "content", "title", "source"}
        legacy_news = dataset == "news" and news_identity.issubset(set(columns))
        if date_field is None or legacy_news:
            # Non-partitioned datasets (and the legacy global news dedup, whose
            # NOT EXISTS semantics are dataset-wide) keep the single-query
            # export; the memory budget and spill directory still apply.
            dataset_dir.mkdir(parents=True, exist_ok=True)
            source_sql = _snapshot_source_query(dataset, quoted_paths, set(columns))
            connection.execute(
                "COPY ({query}) TO {target} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)".format(
                    query=source_sql,
                    target=_sql_string(str(dataset_dir / "data.parquet")),
                )
            )
        else:
            self._export_partitioned_dataset(
                connection,
                dataset,
                rows,
                paths,
                date_field,
                dataset_dir,
                base_dir,
                base_entry,
                current_tuples,
            )
        row_count = connection.execute(
            f"SELECT count(*) FROM read_parquet({quoted_paths}, union_by_name=true)"
        ).fetchone()[0]
        date_min = date_max = None
        if date_field is not None:
            date_expression = _date_sql_expression(date_field)
            source_sql = _snapshot_source_query(dataset, quoted_paths, set(columns))
            date_min, date_max = connection.execute(
                f"SELECT min({date_expression})::VARCHAR, max({date_expression})::VARCHAR "
                f"FROM ({source_sql}) WHERE {date_expression} IS NOT NULL"
            ).fetchone()
        ingested_min = ingested_max = None
        if "ingested_at" in columns:
            ingested_min, ingested_max = connection.execute(
                "SELECT min(ingested_at)::VARCHAR, max(ingested_at)::VARCHAR "
                f"FROM read_parquet({quoted_paths}, union_by_name=true) "
                "WHERE ingested_at IS NOT NULL"
            ).fetchone()
        base_files = {}
        if base_entry is not None:
            base_files = {
                str(item["path"]): item for item in base_entry.get("files") or []
            }
        files = []
        for path in sorted(dataset_dir.rglob("*.parquet")):
            relative = path.relative_to(temporary).as_posix()
            reused = base_files.get(relative)
            if reused is not None and int(reused.get("bytes") or -1) == path.stat().st_size:
                files.append(dict(reused))
            else:
                files.append(
                    {
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
        return {
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

    def _export_partitioned_dataset(
        self,
        connection: duckdb.DuckDBPyConnection,
        dataset: str,
        rows: list[dict[str, Any]],
        paths: list[str],
        date_field: str,
        dataset_dir: Path,
        base_dir: Path | None,
        base_entry: dict[str, Any] | None,
        current_tuples: set[tuple[str, str, int]],
    ) -> None:
        date_expression = _date_sql_expression(date_field)
        # Metadata pass: one single-column min/max scan per unit file, so the
        # merge queries below only read units that intersect their partition.
        unit_ranges: dict[str, tuple[tuple[int, int], tuple[int, int]] | None] = {}
        for path in paths:
            lo, hi = connection.execute(
                f"SELECT min({date_expression}), max({date_expression}) "
                f"FROM read_parquet({_sql_string(path)})"
            ).fetchone()
            unit_ranges[path] = (
                (_year_month(lo), _year_month(hi)) if lo is not None and hi is not None else None
            )

        base_partitions: set[tuple[int, int]] = set()
        if base_dir is not None and base_dir.exists():
            for year_dir in base_dir.glob("partition_year=*"):
                for month_dir in year_dir.glob("partition_month=*"):
                    try:
                        base_partitions.add(
                            (
                                int(year_dir.name.split("=", 1)[1]),
                                int(month_dir.name.split("=", 1)[1]),
                            )
                        )
                    except (IndexError, ValueError):
                        continue

        added_paths = set(paths)
        use_base_partitions = False
        if base_entry is not None:
            base_tuples = _source_unit_tuples(base_entry)
            added_paths = {
                str((self.root / row["output_path"]).resolve())
                for row in rows
                if str(row["output_path"]).endswith(".parquet")
                and (
                    str(row["unit_key"]),
                    str(row.get("sha256") or ""),
                    int(row.get("row_count") or 0),
                )
                in {item for item in current_tuples - base_tuples}
            }
            if base_tuples - current_tuples:
                # Units vanished or changed (e.g. superseded reference
                # generations): parent partitions may hold stale rows, so the
                # whole dataset is rebuilt from current units only.
                dirty: set[tuple[int, int]] | str = "all"
            else:
                dirty = set()
                for path in added_paths:
                    unit_range = unit_ranges.get(path)
                    if unit_range is None:
                        dirty = "all"
                        break
                    dirty.update(_months_between(*unit_range))
                if dirty != "all":
                    use_base_partitions = True
        else:
            dirty = "all"

        if dirty == "all":
            rebuild: set[tuple[int, int]] = set()
            for unit_range in unit_ranges.values():
                if unit_range is not None:
                    rebuild.update(_months_between(*unit_range))
            link: set[tuple[int, int]] = set()
        else:
            rebuild = set(dirty)
            link = base_partitions - rebuild

        for year, month in sorted(link):
            source_dir = (
                base_dir / f"partition_year={year}" / f"partition_month={month}"
            )
            if base_dir is not None and source_dir.exists():
                _link_tree(
                    source_dir,
                    dataset_dir / f"partition_year={year}" / f"partition_month={month}",
                )

        for year, month in sorted(rebuild):
            sources = [
                path
                for path, unit_range in unit_ranges.items()
                if unit_range is not None
                and unit_range[0] <= (year, month) <= unit_range[1]
            ]
            if use_base_partitions and base_dir is not None:
                parent_dir = (
                    base_dir / f"partition_year={year}" / f"partition_month={month}"
                )
                if parent_dir.exists():
                    sources = [
                        str(path) for path in sorted(parent_dir.rglob("*.parquet"))
                    ] + sources
            if not sources:
                continue
            quoted = "[" + ",".join(_sql_string(path) for path in sources) + "]"
            partition_dir = (
                dataset_dir / f"partition_year={year}" / f"partition_month={month}"
            )
            partition_dir.mkdir(parents=True, exist_ok=True)
            connection.execute(
                f"COPY ("
                f"SELECT * FROM ("
                f"SELECT DISTINCT * FROM read_parquet({quoted}, union_by_name=true)"
                f") "
                f"WHERE {date_expression} IS NOT NULL "
                f"AND year({date_expression}) = {year} AND month({date_expression}) = {month}"
                f") TO {_sql_string(str(partition_dir / 'data.parquet'))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
            )


def _source_unit_tuples(entry: dict[str, Any]) -> set[tuple[str, str, int]]:
    return {
        (
            str(item.get("unit_key") or ""),
            str(item.get("sha256") or ""),
            int(item.get("row_count") or 0),
        )
        for item in entry.get("source_units") or []
    }


def _year_month(value: Any) -> tuple[int, int]:
    timestamp = pd.Timestamp(value)
    return (int(timestamp.year), int(timestamp.month))


def _months_between(
    lo: tuple[int, int], hi: tuple[int, int]
) -> list[tuple[int, int]]:
    if hi < lo:
        return []
    months: list[tuple[int, int]] = []
    year, month = lo
    while (year, month) <= hi:
        months.append((year, month))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def _link_tree(source: Path, target: Path) -> None:
    """Hard-link every file under source into target (copy as fallback)."""
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)


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
