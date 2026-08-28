from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock, Thread, get_ident
from time import monotonic
from typing import Any

from sqlalchemy import select

from quant_data.database import (
    backtest_runs,
    factor_evaluations,
    jobs,
    open_database,
    paper_portfolios,
    parameter_experiments,
    recommendation_portfolios,
    research_campaigns,
    research_programs,
    research_runs,
    simulation_portfolios,
)

RETENTION_CONFIRMATION = "DELETE_UNREFERENCED_DATASETS"
RETENTION_INVENTORY_VERSION = 1
RETENTION_INVENTORY_MAX_AGE = timedelta(hours=6)
RETENTION_INVENTORY_FAILURE_BACKOFF_SECONDS = 60.0


class DataRetentionManager:
    """Plan and explicitly remove only unreferenced immutable datasets."""

    def __init__(self, data_root: Path, database_url: str) -> None:
        self.data_root = data_root.resolve()
        self.engine = open_database(database_url)
        self._inventory_cache_path = (
            self.data_root / "platform" / "cache" / "retention-inventory-v1.json"
        )
        self._inventory_refresh_lock = Lock()
        self._inventory_refresh_running = False
        self._inventory_retry_after = 0.0

    def plan(
        self,
        *,
        keep_latest: int = 7,
        min_age_days: int = 14,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if keep_latest < 1 or min_age_days < 1:
            raise ValueError("retention limits must be positive")
        current = now or datetime.now(UTC)
        protected = self._protected_datasets()
        names = self._dataset_names()
        snapshot_created_at = self._snapshot_created_at_index()
        entries = [
            self._entry(name, current, snapshot_created_at=snapshot_created_at)
            for name in names
        ]
        return self._plan_from_entries(
            entries,
            protected=protected,
            keep_latest=keep_latest,
            min_age_days=min_age_days,
            current=current,
            inventory_generated_at=current.isoformat(),
            cache_state="strict",
            refreshing=False,
        )

    def display_plan(
        self,
        *,
        keep_latest: int = 7,
        min_age_days: int = 14,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return a bounded display projection without synchronously walking every file.

        Dataset deletion never consumes this cache: :meth:`apply` calls the strict
        :meth:`plan` again.  Protection references are also queried on every display
        request, so a newly referenced dataset is hidden from deletion immediately
        even while byte counts come from the last immutable inventory.
        """

        if keep_latest < 1 or min_age_days < 1:
            raise ValueError("retention limits must be positive")
        current = now or datetime.now(UTC)
        fingerprint, quick_entries = self._quick_inventory(current)
        cached = self._load_inventory_cache()
        cached_entries = {
            str(item.get("name")): dict(item)
            for item in (cached or {}).get("entries", [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        current_names = {str(item["name"]) for item in quick_entries}
        entries: list[dict[str, Any]] = []
        for quick in quick_entries:
            name = str(quick["name"])
            cached_item = cached_entries.get(name)
            if cached_item is not None:
                entries.append(
                    {
                        **cached_item,
                        "locations": quick["locations"],
                        "inventory_complete": True,
                    }
                )
            else:
                entries.append({**quick, "bytes": 0, "inventory_complete": False})
        cached_fingerprint = (cached or {}).get("fingerprint")
        try:
            cached_generated_at = datetime.fromisoformat(str((cached or {})["generated_at"]))
            if cached_generated_at.tzinfo is None:
                cached_generated_at = cached_generated_at.replace(tzinfo=UTC)
            age_is_fresh = current - cached_generated_at <= RETENTION_INVENTORY_MAX_AGE
        except (KeyError, TypeError, ValueError):
            age_is_fresh = False
        cache_is_fresh = bool(cached) and cached_fingerprint == fingerprint and age_is_fresh
        cache_state = "fresh" if cache_is_fresh else "stale" if cached else "building"
        if not cache_is_fresh:
            self._start_inventory_refresh()
        generated_at = str((cached or {}).get("generated_at") or current.isoformat())
        # A removed directory must disappear immediately even if the last cache still
        # contained it; only current shallow inventory names are returned.
        entries = [item for item in entries if str(item.get("name")) in current_names]
        return self._plan_from_entries(
            entries,
            protected=self._protected_datasets(),
            keep_latest=keep_latest,
            min_age_days=min_age_days,
            current=current,
            inventory_generated_at=generated_at,
            cache_state=cache_state,
            refreshing=not cache_is_fresh,
        )

    def refresh_display_inventory(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Rebuild the persisted exact-size inventory outside request latency."""

        current = now or datetime.now(UTC)
        fingerprint, _ = self._quick_inventory(current)
        names = self._dataset_names()
        snapshot_created_at = self._snapshot_created_at_index()
        entries = [
            {
                **self._entry(name, current, snapshot_created_at=snapshot_created_at),
                "inventory_complete": True,
            }
            for name in names
        ]
        payload = {
            "version": RETENTION_INVENTORY_VERSION,
            "generated_at": current.isoformat(),
            "fingerprint": fingerprint,
            "entries": entries,
        }
        self._inventory_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._inventory_cache_path.with_name(
            f".{self._inventory_cache_path.name}.{os.getpid()}.{get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(self._inventory_cache_path)
        return payload

    def _plan_from_entries(
        self,
        entries: list[dict[str, Any]],
        *,
        protected: dict[str, set[str]],
        keep_latest: int,
        min_age_days: int,
        current: datetime,
        inventory_generated_at: str,
        cache_state: str,
        refreshing: bool,
    ) -> dict[str, Any]:
        entries.sort(key=lambda item: item["created_at"], reverse=True)
        latest = {str(item["name"]) for item in entries[:keep_latest]}
        threshold = current - timedelta(days=min_age_days)
        eligible_bytes = 0
        for item in entries:
            name = str(item["name"])
            reasons = sorted(protected.get(name, set()))
            created_at = datetime.fromisoformat(str(item["created_at"]))
            if reasons:
                state = "protected"
            elif name in latest:
                state = "keep_latest"
                reasons = [f"one of the latest {keep_latest} datasets"]
            elif created_at > threshold:
                state = "keep_young"
                reasons = [f"younger than {min_age_days} days"]
            elif not bool(item.get("inventory_complete", True)):
                # Unknown capacity is display-only and must never be advertised as
                # deletable.  The strict apply path rebuilds the inventory anyway.
                state = "inventory_pending"
                reasons = ["capacity inventory is still being calculated"]
            else:
                state = "eligible"
                eligible_bytes += int(item["bytes"])
            item["state"] = state
            item["reasons"] = reasons
        return {
            "generated_at": current.isoformat(),
            "inventory_generated_at": inventory_generated_at,
            "cache_state": cache_state,
            "refreshing": refreshing,
            "keep_latest": keep_latest,
            "min_age_days": min_age_days,
            "total_bytes": sum(int(item["bytes"]) for item in entries),
            "eligible_bytes": eligible_bytes,
            "entries": entries,
        }

    def _load_inventory_cache(self) -> dict[str, Any] | None:
        try:
            if self._inventory_cache_path.stat().st_size > 5_000_000:
                return None
            payload = json.loads(self._inventory_cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != RETENTION_INVENTORY_VERSION
            or not isinstance(payload.get("entries"), list)
            or not isinstance(payload.get("fingerprint"), list)
        ):
            return None
        return payload

    def _start_inventory_refresh(self) -> None:
        with self._inventory_refresh_lock:
            if self._inventory_refresh_running or monotonic() < self._inventory_retry_after:
                return
            self._inventory_refresh_running = True

        def refresh() -> None:
            try:
                self.refresh_display_inventory()
            except Exception:
                with self._inventory_refresh_lock:
                    self._inventory_retry_after = (
                        monotonic() + RETENTION_INVENTORY_FAILURE_BACKOFF_SECONDS
                    )
            finally:
                with self._inventory_refresh_lock:
                    self._inventory_refresh_running = False

        Thread(target=refresh, name="retention-inventory-refresh", daemon=True).start()

    def _quick_inventory(self, fallback: datetime) -> tuple[list[list[Any]], list[dict[str, Any]]]:
        locations: dict[str, list[Path]] = {}
        fingerprint: list[list[Any]] = []
        for root_name in ("snapshots", "qlib", "qlib_staging"):
            root = self.data_root / root_name
            if not root.exists():
                continue
            for path in root.iterdir():
                if not path.is_dir():
                    continue
                stat = path.stat()
                locations.setdefault(path.name, []).append(path)
                fingerprint.append([root_name, path.name, stat.st_mtime_ns])
        fingerprint.sort()
        snapshot_created_at = self._snapshot_created_at_index()
        entries = []
        for name, paths in locations.items():
            created_at = self._created_at(
                name,
                paths,
                fallback,
                snapshot_created_at=snapshot_created_at,
            )
            entries.append(
                {
                    "name": name,
                    "created_at": created_at.isoformat(),
                    "locations": sorted(path.parent.name for path in paths),
                }
            )
        return fingerprint, entries

    def apply(
        self,
        names: list[str],
        *,
        confirmation: str,
        keep_latest: int = 7,
        min_age_days: int = 14,
    ) -> dict[str, Any]:
        if confirmation != RETENTION_CONFIRMATION:
            raise ValueError("retention confirmation phrase is invalid")
        requested = {str(name).strip() for name in names if str(name).strip()}
        if not requested:
            raise ValueError("at least one dataset name is required")
        plan = self.plan(keep_latest=keep_latest, min_age_days=min_age_days)
        eligible = {
            str(item["name"]): item for item in plan["entries"] if item["state"] == "eligible"
        }
        blocked = sorted(requested.difference(eligible))
        if blocked:
            raise ValueError("datasets are protected or not eligible: " + ", ".join(blocked))
        targets: dict[str, list[tuple[str, Path]]] = {}
        # Resolve and validate every requested location before removing anything;
        # otherwise a later symlink/error could leave an earlier location deleted.
        for name in sorted(requested):
            scoped: list[tuple[str, Path]] = []
            for root_name in ("snapshots", "qlib", "qlib_staging"):
                root = (self.data_root / root_name).resolve()
                raw_target = root / name
                if raw_target.is_symlink():
                    raise ValueError(f"refusing to delete symlinked dataset: {name}")
                target = raw_target.resolve()
                target.relative_to(root)
                if target.exists():
                    scoped.append((root_name, target))
            targets[name] = scoped
        deleted: list[dict[str, Any]] = []
        for name in sorted(requested):
            item = eligible[name]
            if name in self._protected_datasets():
                raise ValueError(f"dataset became protected before deletion: {name}")
            removed = []
            for root_name, target in targets[name]:
                shutil.rmtree(target)
                removed.append(root_name)
            deleted.append({"name": name, "bytes": item["bytes"], "removed_locations": removed})
        self._inventory_cache_path.unlink(missing_ok=True)
        return {
            "status": "deleted",
            "deleted": deleted,
            "reclaimed_bytes": sum(int(item["bytes"]) for item in deleted),
        }

    def _protected_datasets(self) -> dict[str, set[str]]:
        protected: dict[str, set[str]] = {}

        def add(value: Any, reason: str) -> None:
            if isinstance(value, dict):
                value = value.get("name")
            name = str(value or "").strip()
            if name:
                protected.setdefault(name, set()).add(reason)

        with self.engine.connect() as connection:
            for value in connection.scalars(select(research_runs.c.dataset)):
                add(value, "RD-Agent research run")
            for value in connection.scalars(select(factor_evaluations.c.dataset)):
                add(value, "factor evaluation")
            for value in connection.scalars(select(backtest_runs.c.dataset)):
                add(value, "strategy backtest")
            for value in connection.scalars(select(backtest_runs.c.execution_dataset)):
                add(value, "strategy execution backtest")
            for value in connection.scalars(select(parameter_experiments.c.dataset)):
                add(value, "parameter experiment")
            for value in connection.scalars(select(research_campaigns.c.dataset)):
                add(value, "research campaign")
            for value in connection.scalars(select(research_programs.c.last_dataset_name)):
                add(value, "continuous research program")
            for value in connection.scalars(select(paper_portfolios.c.dataset)):
                add(value, "paper portfolio")
            for value in connection.scalars(select(recommendation_portfolios.c.dataset)):
                add(value, "recommendation target portfolio")
            for value in connection.scalars(select(simulation_portfolios.c.daily_dataset)):
                add(value, "simulation daily ledger")
            for value in connection.scalars(select(simulation_portfolios.c.execution_dataset)):
                add(value, "simulation execution ledger")
            active_jobs = connection.execute(
                select(jobs.c.kind, jobs.c.payload_json).where(
                    jobs.c.status.in_(("queued", "running"))
                )
            )
            for row in active_jobs:
                payload = dict(row.payload_json or {})
                add(payload.get("dataset"), f"active {row.kind} job")
                add(payload.get("execution_dataset"), f"active {row.kind} job")
                add(payload.get("snapshot_name"), f"active {row.kind} job")
        return protected

    def _dataset_names(self) -> list[str]:
        names: set[str] = set()
        for root_name in ("snapshots", "qlib", "qlib_staging"):
            root = self.data_root / root_name
            if root.exists():
                names.update(path.name for path in root.iterdir() if path.is_dir())
        return sorted(names)

    def _entry(
        self,
        name: str,
        now: datetime,
        *,
        snapshot_created_at: dict[str, datetime] | None = None,
    ) -> dict[str, Any]:
        paths = [
            self.data_root / root_name / name for root_name in ("snapshots", "qlib", "qlib_staging")
        ]
        existing = [path for path in paths if path.exists()]
        created_at = self._created_at(
            name,
            existing,
            now,
            snapshot_created_at=snapshot_created_at,
        )
        return {
            "name": name,
            "created_at": created_at.isoformat(),
            "bytes": sum(self._directory_size(path) for path in existing),
            "locations": sorted(path.parent.name for path in existing),
        }

    def _created_at(
        self,
        name: str,
        paths: list[Path],
        fallback: datetime,
        *,
        snapshot_created_at: dict[str, datetime] | None = None,
    ) -> datetime:
        # Qlib provenance is small; some snapshot manifests contain millions of
        # source-unit records and must never be JSON-loaded by a page request.
        provenance = self.data_root / "qlib" / name / "metadata" / "provenance.json"
        for path in (provenance,):
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                value = json.loads(path.read_text(encoding="utf-8")).get("created_at")
                if value:
                    parsed = datetime.fromisoformat(str(value))
                    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
            except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
                continue
        indexed = (snapshot_created_at or {}).get(name)
        if indexed is not None:
            return indexed
        manifest = self.data_root / "snapshots" / name / "manifest.json"
        try:
            if manifest.stat().st_size <= 2_000_000:
                value = json.loads(manifest.read_text(encoding="utf-8")).get("created_at")
                if value:
                    parsed = datetime.fromisoformat(str(value))
                    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
            pass
        # Unknown age is treated as new.  This may retain extra data but can never
        # make an old/partially published directory eligible for deletion.
        return fallback

    def _snapshot_created_at_index(self) -> dict[str, datetime]:
        cache = self.data_root / "snapshots" / ".catalog-display-v1.json"
        try:
            if cache.stat().st_size > 5_000_000:
                return {}
            payload = json.loads(cache.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        index: dict[str, datetime] = {}
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, dict):
            return index
        for name, record in entries.items():
            summary = record.get("summary") if isinstance(record, dict) else None
            value = summary.get("created_at") if isinstance(summary, dict) else None
            if not value:
                continue
            try:
                parsed = datetime.fromisoformat(str(value))
            except ValueError:
                continue
            index[str(name)] = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return index

    @staticmethod
    def _directory_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
