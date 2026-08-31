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
from .release_window import PIT_CARRY_IN_DATASETS
from .row_identity import (
    NULL_ON_AMBIGUITY_COLUMNS,
    SEMANTIC_METADATA_COLUMNS,
    SNAPSHOT_QUARANTINE_KEYS,
    semantic_provider_columns,
)

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
        keep_raw: bool = True,
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
        industry_history_anchor: Path | None = None,
        duckdb_memory_limit: str = "4GB",
        duckdb_threads: int = 4,
    ) -> Path:
        target = self.snapshots_root / name
        temporary = self.snapshots_root / f".{name}.tmp"
        if base_snapshot is not None and industry_history_anchor is not None:
            raise ValueError(
                "industry history anchor is only valid for a new lineage root; "
                "a lineage parent is already available"
            )
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
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"base snapshot {base_snapshot} is incomplete") from exc
            base_root = base_snapshot / "parquet"

        industry_anchor_root: Path | None = None
        industry_anchor_manifest: dict[str, Any] = {}
        industry_anchor_evidence: dict[str, Any] | None = None
        if industry_history_anchor is not None:
            try:
                anchor_manifest_raw = (industry_history_anchor / "manifest.json").read_bytes()
                industry_anchor_manifest = json.loads(anchor_manifest_raw)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"industry history anchor {industry_history_anchor} is incomplete"
                ) from exc
            if not isinstance(industry_anchor_manifest, dict):
                raise ValueError("industry history anchor manifest is invalid")
            industry_anchor_evidence = manifest_extra.get("industry_history_anchor")
            if not isinstance(industry_anchor_evidence, dict):
                raise ValueError("industry history anchor has no governed evidence")
            expected_evidence = dict(industry_anchor_evidence)
            evidence_sha256 = str(expected_evidence.pop("evidence_sha256", ""))
            if (
                industry_anchor_evidence.get("snapshot_name")
                != industry_history_anchor.name
                or industry_anchor_evidence.get("manifest_sha256")
                != hashlib.sha256(anchor_manifest_raw).hexdigest()
                or evidence_sha256 != _canonical_sha256(expected_evidence)
            ):
                raise ValueError("industry history anchor evidence does not match")
            industry_anchor_root = industry_history_anchor / "parquet"

        manifest: dict[str, Any] = {
            "name": name,
            "created_at": datetime.now(UTC).isoformat(),
            "datasets": {},
            **manifest_extra,
        }
        snapshot_start = str(manifest_extra.get("start_date") or "")[:10] or None
        snapshot_end = str(manifest_extra.get("end_date") or "")[:10] or None
        connection = duckdb.connect()
        try:
            connection.execute(f"SET memory_limit='{duckdb_memory_limit}'")
            connection.execute(f"SET threads={int(duckdb_threads)}")
            spill_dir = temporary / ".duckdb-spill"
            spill_dir.mkdir(exist_ok=True)
            connection.execute(f"SET temp_directory={_sql_string(str(spill_dir))}")
            delisted_stock_symbols = self._explicitly_delisted_stock_symbols(
                connection,
                successful_units.get("stock_basic", []),
            )
            for dataset, rows in sorted(successful_units.items()):
                base_entry = None
                if base_root is not None:
                    base_entry = (base_manifest.get("datasets") or {}).get(dataset)
                    if not isinstance(base_entry, dict):
                        base_entry = None
                industry_history_root = base_root
                industry_history_entry = base_entry
                industry_history_source = (
                    base_snapshot.name if base_snapshot is not None else None
                )
                industry_history_source_kind = "lineage_parent"
                if (
                    dataset == "index_member_all"
                    and industry_history_entry is None
                    and industry_anchor_root is not None
                ):
                    anchor_entry = (industry_anchor_manifest.get("datasets") or {}).get(
                        dataset
                    )
                    if isinstance(anchor_entry, dict):
                        industry_history_root = industry_anchor_root
                        industry_history_entry = anchor_entry
                        industry_history_source = industry_history_anchor.name
                        industry_history_source_kind = "explicit_anchor"
                manifest["datasets"][dataset] = self._build_dataset_snapshot(
                    connection,
                    dataset,
                    rows,
                    temporary,
                    base_root,
                    base_entry,
                    snapshot_start,
                    snapshot_end,
                    industry_history_root=(
                        industry_history_root
                        if dataset == "index_member_all"
                        else None
                    ),
                    industry_history_entry=(
                        industry_history_entry
                        if dataset == "index_member_all"
                        else None
                    ),
                    industry_history_source=industry_history_source,
                    industry_history_source_kind=industry_history_source_kind,
                    delisted_stock_symbols=delisted_stock_symbols,
                    industry_history_anchor_evidence=industry_anchor_evidence,
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
        snapshot_start: str | None,
        snapshot_end: str | None,
        *,
        industry_history_root: Path | None = None,
        industry_history_entry: dict[str, Any] | None = None,
        industry_history_source: str | None = None,
        industry_history_source_kind: str | None = None,
        delisted_stock_symbols: frozenset[str] = frozenset(),
        industry_history_anchor_evidence: dict[str, Any] | None = None,
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
            "source_rows": 0,
            "unit_files": 0,
            "empty_units": len(rows),
            "date_field": None,
            "date_min": None,
            "date_max": None,
            "date_filter_mode": (
                "interval_overlap"
                if dataset in {"index_member_all", "namechange"}
                else None
            ),
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
        quoted_paths = "[" + ",".join(_sql_string(path) for path in paths) + "]"
        raw_source_sql = (
            f"SELECT * FROM read_parquet({quoted_paths}, union_by_name=true)"
        )
        columns = (
            connection.execute(f"DESCRIBE {raw_source_sql}")
            .fetchdf()["column_name"]
            .tolist()
        )
        carry_result = self._index_member_parent_carry(
            connection,
            dataset=dataset,
            current_source_sql=raw_source_sql,
            current_columns=set(columns),
            history_root=industry_history_root,
            history_entry=industry_history_entry,
            history_source=industry_history_source,
            history_source_kind=industry_history_source_kind,
            delisted_stock_symbols=delisted_stock_symbols,
            industry_history_anchor_evidence=industry_history_anchor_evidence,
        )
        industry_carry: dict[str, Any] | None = None
        if carry_result is not None:
            raw_source_sql, industry_carry = carry_result
            columns = (
                connection.execute(f"DESCRIBE {raw_source_sql}")
                .fetchdf()["column_name"]
                .tolist()
            )
        date_field = next(
            (field for field in _date_field_candidates(dataset) if field in columns),
            None,
        )
        if date_field is not None:
            # Wide provider schemas can contain a date-like column that is
            # unrelated to this dataset and entirely NULL (for example
            # sge_basic carrying an empty trade_time column). Treat that as a
            # non-partitioned reference dataset. Otherwise the partition
            # exporter creates no files while the manifest still reports the
            # source row count, producing an unusable snapshot.
            date_expression = _date_sql_expression(date_field)
            has_valid_dates = bool(
                connection.execute(
                    f"SELECT count(*) > 0 FROM ({raw_source_sql}) "
                    f"WHERE {date_expression} IS NOT NULL"
                ).fetchone()[0]
            )
            if not has_valid_dates:
                date_field = None
        pit_carry_in = (
            dataset in PIT_CARRY_IN_DATASETS
            and date_field in {"ann_date", "f_ann_date"}
        )
        date_filter_mode = (
            "interval_overlap"
            if dataset in {"index_member_all", "namechange"}
            else (
                "announcement_pit_carry_in"
                if pit_carry_in
                else ("point_date" if date_field is not None else None)
            )
        )
        if dataset in {"index_member_all", "namechange"} and base_entry is not None:
            # Interval membership output depends on both requested bounds.
            # Rebuild this small reference dataset rather than linking a
            # parent that may have been built for another range. Older
            # snapshots also bounded memberships by in_date alone, dropping
            # companies that entered before the requested start but remained
            # active inside the requested range.
            base_entry = None
            base_dir = None
        if dataset == "fund_basic" and base_entry is not None and (
            base_entry.get("date_field") is not None
            or base_entry.get("date_filter_mode") is not None
        ):
            # fund_basic is a lifecycle master, not a point-in-time issue
            # event.  Older snapshots clipped it by issue_date, which drops
            # still-listed funds issued before the research window.  Never
            # hard-link that obsolete projection into a corrected successor.
            base_entry = None
            base_dir = None
        if date_field is not None and base_entry is not None:
            base_min = str(base_entry.get("date_min") or "")[:10] or None
            base_max = str(base_entry.get("date_max") or "")[:10] or None
            if (
                (snapshot_start is not None and base_min is not None and base_min < snapshot_start)
                or (snapshot_end is not None and base_max is not None and base_max > snapshot_end)
            ):
                # A parent created for a wider range (or by an older builder
                # that leaked the rest of an overlapping month) cannot be
                # hard-linked safely into this bounded snapshot.
                base_entry = None
                base_dir = None
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
        news_identity = {"datetime", "content", "title", "source"}
        legacy_news = dataset == "news" and news_identity.issubset(set(columns))
        if date_field is None or legacy_news or pit_carry_in:
            # Non-partitioned datasets (and the legacy global news dedup, whose
            # NOT EXISTS semantics are dataset-wide) keep the single-query
            # export; the memory budget and spill directory still apply.
            dataset_dir.mkdir(parents=True, exist_ok=True)
            source_sql = _snapshot_source_query(
                dataset,
                quoted_paths,
                set(columns),
                source_sql=raw_source_sql,
            )
            source_sql = _bounded_snapshot_query(
                source_sql,
                dataset,
                date_field,
                snapshot_start,
                snapshot_end,
                set(columns),
            )
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
                set(columns),
                dataset_dir,
                base_dir,
                base_entry,
                current_tuples,
                snapshot_start,
                snapshot_end,
            )
        bounded_source_sql = _bounded_snapshot_query(
            raw_source_sql,
            dataset,
            date_field,
            snapshot_start,
            snapshot_end,
            set(columns),
        )
        source_row_count = connection.execute(
            f"SELECT count(*) FROM ({bounded_source_sql})"
        ).fetchone()[0]
        snapshot_paths = [str(path.resolve()) for path in sorted(dataset_dir.rglob("*.parquet"))]
        if not snapshot_paths:
            return {
                **empty_entry,
                "source_rows": int(source_row_count),
                "unit_files": len(paths),
                "empty_units": len(rows) - len(paths),
                "date_field": date_field,
                "date_filter_mode": date_filter_mode,
            }
        snapshot_quoted_paths = "[" + ",".join(_sql_string(path) for path in snapshot_paths) + "]"
        row_count = connection.execute(
            f"SELECT count(*) FROM read_parquet({snapshot_quoted_paths}, union_by_name=true)"
        ).fetchone()[0]
        date_min = date_max = None
        coverage_field = date_field
        if coverage_field is None and dataset == "index_member_all" and "in_date" in columns:
            coverage_field = "in_date"
        if coverage_field is None and dataset == "namechange" and "start_date" in columns:
            coverage_field = "start_date"
        if coverage_field is not None:
            date_expression = _date_sql_expression(coverage_field)
            # The compacted files are the published dataset and already carry
            # exact-row deduplication plus any explicit conflict quarantine.
            # Derive their coverage directly instead of re-running the full
            # semantic GROUP BY over every source unit. On very large sources
            # (for example ccass_hold_detail) that redundant global aggregate
            # can exhaust DuckDB's bounded memory after partition export.
            date_min, date_max = connection.execute(
                f"SELECT min({date_expression})::VARCHAR, max({date_expression})::VARCHAR "
                f"FROM read_parquet({snapshot_quoted_paths}, union_by_name=true) "
                f"WHERE {date_expression} IS NOT NULL"
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
            base_files = {str(item["path"]): item for item in base_entry.get("files") or []}
        files = []
        for path in sorted(dataset_dir.rglob("*.parquet")):
            relative = path.relative_to(temporary).as_posix()
            reused = base_files.get(relative)
            base_path = base_root.parent / relative if base_root is not None else None
            if (
                reused is not None
                and base_path is not None
                and base_path.is_file()
                and path.samefile(base_path)
            ):
                # Equal path and byte length do not prove equal content.  A
                # rebuilt Parquet partition can compress to exactly the same
                # size as its parent while carrying different rows.  Reuse
                # the parent's digest only for the actual hard-linked file;
                # copied fallbacks and rebuilt partitions are hashed below.
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
            "source_rows": int(source_row_count),
            "unit_files": len(paths),
            "empty_units": len(rows) - len(paths),
            "date_field": date_field,
            "date_min": date_min,
            "date_max": date_max,
            "date_filter_mode": date_filter_mode,
            "ingested_at_min": ingested_min,
            "ingested_at_max": ingested_max,
            "recoverability": recoverability_level(dataset),
            "availability_policy": availability_contract_label(dataset),
            "reference_refresh": refresh_metadata,
            "source_sha256": _effective_source_sha256(source_sha256, industry_carry),
            "source_units": source_identity,
            "files": files,
            **({"industry_history_carry": industry_carry} if industry_carry else {}),
        }

    def _explicitly_delisted_stock_symbols(
        self,
        connection: duckdb.DuckDBPyConnection,
        rows: list[dict[str, Any]],
    ) -> frozenset[str]:
        paths = [
            str((self.root / row["output_path"]).resolve())
            for row in rows
            if str(row.get("output_path") or "").endswith(".parquet")
        ]
        if not paths:
            return frozenset()
        quoted = "[" + ",".join(_sql_string(path) for path in paths) + "]"
        columns = set(
            connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet({quoted}, union_by_name=true)"
            )
            .fetchdf()["column_name"]
            .tolist()
        )
        if not {"ts_code", "list_status"}.issubset(columns):
            return frozenset()
        records = connection.execute(
            f"""
            SELECT upper(trim(CAST(ts_code AS VARCHAR))) AS instrument
            FROM read_parquet({quoted}, union_by_name=true)
            WHERE nullif(trim(CAST(ts_code AS VARCHAR)), '') IS NOT NULL
              AND nullif(trim(CAST(list_status AS VARCHAR)), '') IS NOT NULL
            GROUP BY instrument
            HAVING count(DISTINCT upper(trim(CAST(list_status AS VARCHAR)))) = 1
               AND max(upper(trim(CAST(list_status AS VARCHAR)))) = 'D'
            """
        ).fetchall()
        return frozenset(str(record[0]) for record in records)

    @staticmethod
    def _index_member_parent_carry(
        connection: duckdb.DuckDBPyConnection,
        *,
        dataset: str,
        current_source_sql: str,
        current_columns: set[str],
        history_root: Path | None,
        history_entry: dict[str, Any] | None,
        history_source: str | None,
        history_source_kind: str | None,
        delisted_stock_symbols: frozenset[str],
        industry_history_anchor_evidence: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]] | None:
        if (
            dataset != "index_member_all"
            or "ts_code" not in current_columns
            or history_root is None
            or not isinstance(history_entry, dict)
            or not history_source
            or not delisted_stock_symbols
        ):
            return None
        history_files = _manifest_dataset_paths(
            history_root,
            dataset=dataset,
            entry=history_entry,
            reject_unmanifested=history_source_kind == "explicit_anchor",
            verify_hashes=history_source_kind == "explicit_anchor",
        )
        if not history_files:
            return None
        history_quoted = "[" + ",".join(
            _sql_string(str(path.resolve())) for path in history_files
        ) + "]"
        history_columns = set(
            connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet({history_quoted}, union_by_name=true)"
            )
            .fetchdf()["column_name"]
            .tolist()
        )
        if "ts_code" not in history_columns:
            return None
        delisted = ",".join(_sql_string(value) for value in sorted(delisted_stock_symbols))
        symbols = [
            str(record[0])
            for record in connection.execute(
                f"""
                WITH current_symbols AS (
                    SELECT DISTINCT upper(trim(CAST(ts_code AS VARCHAR))) AS instrument
                    FROM ({current_source_sql})
                    WHERE nullif(trim(CAST(ts_code AS VARCHAR)), '') IS NOT NULL
                ),
                history_symbols AS (
                    SELECT DISTINCT upper(trim(CAST(ts_code AS VARCHAR))) AS instrument
                    FROM read_parquet({history_quoted}, union_by_name=true)
                    WHERE nullif(trim(CAST(ts_code AS VARCHAR)), '') IS NOT NULL
                )
                SELECT history_symbols.instrument
                FROM history_symbols
                LEFT JOIN current_symbols USING (instrument)
                WHERE current_symbols.instrument IS NULL
                  AND history_symbols.instrument IN ({delisted})
                ORDER BY history_symbols.instrument
                """
            ).fetchall()
        ]
        if not symbols:
            return None
        symbol_sql = ",".join(_sql_string(value) for value in symbols)
        parent_filter = (
            "upper(trim(CAST(ts_code AS VARCHAR))) "
            f"IN ({symbol_sql})"
        )
        carried_rows = int(
            connection.execute(
                "SELECT count(*) FROM read_parquet("
                f"{history_quoted}, union_by_name=true) WHERE {parent_filter}"
            ).fetchone()[0]
        )
        files_evidence = [
            {
                "path": str(item.get("path") or ""),
                "bytes": int(item.get("bytes") or 0),
                "sha256": str(item.get("sha256") or ""),
            }
            for item in history_entry.get("files") or []
        ]
        source_sql = (
            f"SELECT * FROM ({current_source_sql}) UNION ALL BY NAME "
            f"SELECT * FROM read_parquet({history_quoted}, union_by_name=true) "
            f"WHERE {parent_filter}"
        )
        evidence = {
            "rule_version": "index-member-delisted-parent-carry-v1",
            "source_snapshot": history_source,
            "source_kind": history_source_kind,
            "parent_dataset_source_sha256": str(
                history_entry.get("source_sha256") or ""
            ),
            "parent_dataset_files_sha256": _canonical_sha256(files_evidence),
            "symbols": symbols,
            "symbol_count": len(symbols),
            "rows": carried_rows,
            **(
                {
                    "anchor_manifest_sha256": str(
                        industry_history_anchor_evidence.get("manifest_sha256") or ""
                    ),
                    "anchor_evidence_sha256": str(
                        industry_history_anchor_evidence.get("evidence_sha256") or ""
                    ),
                }
                if history_source_kind == "explicit_anchor"
                and isinstance(industry_history_anchor_evidence, dict)
                else {}
            ),
        }
        return source_sql, evidence

    def _export_partitioned_dataset(
        self,
        connection: duckdb.DuckDBPyConnection,
        dataset: str,
        rows: list[dict[str, Any]],
        paths: list[str],
        date_field: str,
        columns: set[str],
        dataset_dir: Path,
        base_dir: Path | None,
        base_entry: dict[str, Any] | None,
        current_tuples: set[tuple[str, str, int]],
        snapshot_start: str | None,
        snapshot_end: str | None,
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

        lower_month = _year_month(snapshot_start) if snapshot_start is not None else None
        upper_month = _year_month(snapshot_end) if snapshot_end is not None else None
        rebuild = {
            item
            for item in rebuild
            if (lower_month is None or item >= lower_month)
            and (upper_month is None or item <= upper_month)
        }
        link = {
            item
            for item in link
            if (lower_month is None or item >= lower_month)
            and (upper_month is None or item <= upper_month)
        }

        for year, month in sorted(link):
            source_dir = base_dir / f"partition_year={year}" / f"partition_month={month}"
            if base_dir is not None and source_dir.exists():
                _link_tree(
                    source_dir,
                    dataset_dir / f"partition_year={year}" / f"partition_month={month}",
                )

        for year, month in sorted(rebuild):
            sources = [
                path
                for path, unit_range in unit_ranges.items()
                if unit_range is not None and unit_range[0] <= (year, month) <= unit_range[1]
            ]
            if use_base_partitions and base_dir is not None:
                parent_dir = base_dir / f"partition_year={year}" / f"partition_month={month}"
                if parent_dir.exists():
                    sources = [
                        str(path) for path in sorted(parent_dir.rglob("*.parquet"))
                    ] + sources
            if not sources:
                continue
            quoted = "[" + ",".join(_sql_string(path) for path in sources) + "]"
            partition_columns = {
                str(row[0])
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet({quoted}, union_by_name=true)"
                ).fetchall()
            }
            source_sql = _snapshot_source_query(dataset, quoted, partition_columns)
            source_sql = _bounded_snapshot_query(
                source_sql,
                dataset,
                date_field,
                snapshot_start,
                snapshot_end,
                partition_columns,
            )
            partition_dir = dataset_dir / f"partition_year={year}" / f"partition_month={month}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            connection.execute(
                f"COPY ("
                f"SELECT * FROM ("
                f"{source_sql}"
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


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _effective_source_sha256(
    current_source_sha256: str,
    industry_carry: dict[str, Any] | None,
) -> str:
    if not industry_carry:
        return current_source_sha256
    return _canonical_sha256(
        {
            "current_source_sha256": current_source_sha256,
            "industry_history_carry": industry_carry,
        }
    )


def _manifest_dataset_paths(
    parquet_root: Path,
    *,
    dataset: str,
    entry: dict[str, Any],
    reject_unmanifested: bool,
    verify_hashes: bool,
) -> list[Path]:
    """Resolve only dataset files sealed by a snapshot manifest."""

    snapshot_root = parquet_root.parent.resolve()
    declared: list[Path] = []
    seen: set[Path] = set()
    files = entry.get("files")
    if not isinstance(files, list):
        raise ValueError(f"history snapshot has no sealed {dataset} file list")
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"history snapshot has an invalid {dataset} file entry")
        relative = Path(str(item.get("path") or ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or tuple(relative.parts[:2]) != ("parquet", dataset)
            or relative.suffix != ".parquet"
        ):
            raise ValueError(f"history snapshot has an unsafe {dataset} file path")
        unresolved = snapshot_root / relative
        if unresolved.is_symlink() or not unresolved.is_file():
            raise ValueError(f"history snapshot {dataset} file is missing or unsafe")
        try:
            resolved = unresolved.resolve(strict=True)
            resolved.relative_to(snapshot_root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"history snapshot {dataset} file escapes its root") from exc
        if resolved in seen:
            raise ValueError(f"history snapshot repeats a {dataset} file path")
        if verify_hashes:
            expected_bytes = item.get("bytes")
            expected_sha256 = str(item.get("sha256") or "").lower()
            if (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 0
                or resolved.stat().st_size != expected_bytes
                or len(expected_sha256) != 64
                or hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_sha256
            ):
                raise ValueError(f"history snapshot {dataset} file hash does not match")
        seen.add(resolved)
        declared.append(resolved)
    if reject_unmanifested:
        actual = {
            path.resolve()
            for path in (parquet_root / dataset).rglob("*.parquet")
            if path.is_file()
        }
        if actual != seen:
            raise ValueError(
                f"industry history anchor contains unmanifested {dataset} parquet files"
            )
    return sorted(declared)


def _year_month(value: Any) -> tuple[int, int]:
    timestamp = pd.Timestamp(value)
    return (int(timestamp.year), int(timestamp.month))


def _months_between(lo: tuple[int, int], hi: tuple[int, int]) -> list[tuple[int, int]]:
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
    if dataset in {"stock_basic", "fund_basic"}:
        # Lifecycle masters describe instruments that can remain active long
        # after their listing/issue date.  Their selected reference generation
        # is already bounded by snapshot_end; clipping rows to snapshot_start
        # would remove valid pre-window constituents such as 510050.SH.
        return ()
    if dataset in {"index_member_all", "namechange"}:
        # Membership/name-state rows describe intervals. They must not be
        # partitioned or clipped as point observations by their start date.
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


def _bounded_snapshot_query(
    source_sql: str,
    dataset: str,
    date_field: str | None,
    snapshot_start: str | None,
    snapshot_end: str | None,
    columns: set[str],
) -> str:
    if snapshot_start is None and snapshot_end is None:
        return source_sql
    interval_columns = {
        "index_member_all": ("in_date", "out_date"),
        "namechange": ("start_date", "end_date"),
    }
    interval = interval_columns.get(dataset)
    if interval is not None and interval[0] in columns:
        start_field, end_field = interval
        interval_start = _date_sql_expression(start_field)
        predicates = [f"{interval_start} IS NOT NULL"]
        if snapshot_end is not None:
            predicates.append(
                f"{interval_start} <= DATE {_sql_string(snapshot_end)}"
            )
        if snapshot_start is not None and end_field in columns:
            interval_end = _date_sql_expression(end_field)
            predicates.append(
                f"({interval_end} IS NULL OR "
                f"{interval_end} >= DATE {_sql_string(snapshot_start)})"
            )
        return f"SELECT * FROM ({source_sql}) WHERE {' AND '.join(predicates)}"
    if date_field is None:
        return source_sql
    expression = _date_sql_expression(date_field)
    if (
        dataset in PIT_CARRY_IN_DATASETS
        and date_field in {"ann_date", "f_ann_date"}
        and snapshot_start is not None
    ):
        if "ts_code" not in columns:
            raise ValueError(
                f"{dataset} requires ts_code to preserve point-in-time carry-in state"
            )
        predicates = [f"{expression} IS NOT NULL"]
        if snapshot_end is not None:
            predicates.append(f"{expression} <= DATE {_sql_string(snapshot_end)}")
        bounded = (
            f"SELECT *, {expression} AS __snapshot_pit_date "
            f"FROM ({source_sql}) WHERE {' AND '.join(predicates)}"
        )
        ranked = (
            "SELECT *, max(CASE WHEN __snapshot_pit_date < "
            f"DATE {_sql_string(snapshot_start)} THEN __snapshot_pit_date END) "
            "OVER (PARTITION BY ts_code) AS __snapshot_pit_carry_date "
            f"FROM ({bounded})"
        )
        return (
            "SELECT * EXCLUDE (__snapshot_pit_date, __snapshot_pit_carry_date) "
            f"FROM ({ranked}) WHERE __snapshot_pit_date >= "
            f"DATE {_sql_string(snapshot_start)} "
            "OR __snapshot_pit_date = __snapshot_pit_carry_date"
        )
    predicates = [f"{expression} IS NOT NULL"]
    if snapshot_start is not None:
        predicates.append(f"{expression} >= DATE {_sql_string(snapshot_start)}")
    if snapshot_end is not None:
        predicates.append(f"{expression} <= DATE {_sql_string(snapshot_end)}")
    return f"SELECT * FROM ({source_sql}) WHERE {' AND '.join(predicates)}"


def _snapshot_source_query(
    dataset: str,
    quoted_paths: str,
    columns: set[str],
    *,
    source_sql: str | None = None,
) -> str:
    source_sql = source_sql or (
        f"SELECT * FROM read_parquet({quoted_paths}, union_by_name=true)"
    )
    if "ingested_at" in columns:
        # ``ingested_at`` is acquisition lineage, not provider row identity.
        # Overlapping/resumed pages can return the exact same provider row at
        # different fetch times. Collapse those semantic duplicates and keep
        # the earliest observation, which is the conservative point-in-time
        # timestamp. Unsafe conflicts are either blocked by verification or
        # removed under an explicit dataset-specific quarantine rule below.
        provider_columns = sorted(semantic_provider_columns(dataset, columns))
        projected = ", ".join(_identifier(column) for column in provider_columns)
        metadata_columns = sorted(set(columns) & set(SEMANTIC_METADATA_COLUMNS.get(dataset, ())))
        null_on_ambiguity = NULL_ON_AMBIGUITY_COLUMNS.get(dataset, ())
        metadata_projections = []
        for column in metadata_columns:
            identifier = _identifier(column)
            if column in null_on_ambiguity:
                metadata_projections.append(
                    f"CASE WHEN count(DISTINCT {identifier}) <= 1 "
                    f"THEN min({identifier}) END AS {identifier}"
                )
            else:
                metadata_projections.append(f"min({identifier}) AS {identifier}")
        selected = ", ".join([projected, *metadata_projections, "min(ingested_at) AS ingested_at"])
        semantic_rows = (
            f"SELECT {selected} "
            f"FROM ({source_sql}) "
            f"GROUP BY {projected}"
        )
        quarantine_key = SNAPSHOT_QUARANTINE_KEYS.get(dataset)
        if quarantine_key:
            quarantine_columns = ", ".join(_identifier(column) for column in quarantine_key)
            base = (
                f"WITH semantic_rows AS ({semantic_rows}) "
                "SELECT * FROM semantic_rows "
                f"QUALIFY count(*) OVER (PARTITION BY {quarantine_columns}) = 1"
            )
        else:
            base = semantic_rows
    else:
        base = f"SELECT DISTINCT * FROM ({source_sql})"
    news_identity = {"datetime", "content", "title", "source"}
    if dataset != "news" or not news_identity.issubset(columns):
        return base
    # Keep every explicitly sourced record. Legacy all-day news units did not
    # persist the source; retain those only when no new source-aware window has
    # supplied the same timestamp/title/content. This lets immutable old units
    # remain on disk without duplicating repaired snapshots.
    candidate_projection = "candidate.*"
    derived_columns = {"display_title", "title_source"} & columns
    if derived_columns:
        excluded = ", ".join(_identifier(column) for column in sorted(derived_columns))
        candidate_projection = f"candidate.* EXCLUDE ({excluded})"
    return f"""
        WITH source_rows AS ({base})
        SELECT
            {candidate_projection},
            coalesce(
                nullif(trim(CAST(candidate.title AS VARCHAR)), ''),
                nullif(substr(trim(CAST(candidate.content AS VARCHAR)), 1, 80), '')
            ) AS display_title,
            CASE
                WHEN nullif(trim(CAST(candidate.title AS VARCHAR)), '') IS NOT NULL
                    THEN 'original'
                WHEN nullif(trim(CAST(candidate.content AS VARCHAR)), '') IS NOT NULL
                    THEN 'content_fallback'
                ELSE 'missing'
            END AS title_source
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
