from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from quant_data.database import (
    backtest_runs,
    factor_evaluations,
    jobs,
    open_database,
    paper_portfolios,
    research_runs,
)

RETENTION_CONFIRMATION = "DELETE_UNREFERENCED_DATASETS"


class DataRetentionManager:
    """Plan and explicitly remove only unreferenced immutable datasets."""

    def __init__(self, data_root: Path, database_url: str) -> None:
        self.data_root = data_root.resolve()
        self.engine = open_database(database_url)

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
        entries = [self._entry(name, current) for name in names]
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
            else:
                state = "eligible"
                eligible_bytes += int(item["bytes"])
            item["state"] = state
            item["reasons"] = reasons
        return {
            "generated_at": current.isoformat(),
            "keep_latest": keep_latest,
            "min_age_days": min_age_days,
            "total_bytes": sum(int(item["bytes"]) for item in entries),
            "eligible_bytes": eligible_bytes,
            "entries": entries,
        }

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
        deleted: list[dict[str, Any]] = []
        for name in sorted(requested):
            item = eligible[name]
            removed = []
            for root_name in ("snapshots", "qlib", "qlib_staging"):
                target = (self.data_root / root_name / name).resolve()
                target.relative_to(self.data_root / root_name)
                if target.is_symlink():
                    raise ValueError(f"refusing to delete symlinked dataset: {name}")
                if target.exists():
                    shutil.rmtree(target)
                    removed.append(root_name)
            deleted.append(
                {"name": name, "bytes": item["bytes"], "removed_locations": removed}
            )
        return {
            "status": "deleted",
            "deleted": deleted,
            "reclaimed_bytes": sum(int(item["bytes"]) for item in deleted),
        }

    def _protected_datasets(self) -> dict[str, set[str]]:
        protected: dict[str, set[str]] = {}

        def add(value: Any, reason: str) -> None:
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
            for value in connection.scalars(select(paper_portfolios.c.dataset)):
                add(value, "paper portfolio")
            active_jobs = connection.execute(
                select(jobs.c.kind, jobs.c.payload_json).where(
                    jobs.c.status.in_(("queued", "running"))
                )
            )
            for row in active_jobs:
                payload = dict(row.payload_json or {})
                add(payload.get("dataset"), f"active {row.kind} job")
                add(payload.get("snapshot_name"), f"active {row.kind} job")
        return protected

    def _dataset_names(self) -> list[str]:
        names: set[str] = set()
        for root_name in ("snapshots", "qlib", "qlib_staging"):
            root = self.data_root / root_name
            if root.exists():
                names.update(path.name for path in root.iterdir() if path.is_dir())
        return sorted(names)

    def _entry(self, name: str, now: datetime) -> dict[str, Any]:
        paths = [
            self.data_root / root_name / name
            for root_name in ("snapshots", "qlib", "qlib_staging")
        ]
        existing = [path for path in paths if path.exists()]
        created_at = self._created_at(name, existing, now)
        return {
            "name": name,
            "created_at": created_at.isoformat(),
            "bytes": sum(self._directory_size(path) for path in existing),
            "locations": [path.parent.name for path in existing],
        }

    def _created_at(self, name: str, paths: list[Path], fallback: datetime) -> datetime:
        candidates = (
            self.data_root / "snapshots" / name / "manifest.json",
            self.data_root / "qlib" / name / "metadata" / "provenance.json",
        )
        for path in candidates:
            try:
                value = json.loads(path.read_text(encoding="utf-8")).get("created_at")
                if value:
                    parsed = datetime.fromisoformat(str(value))
                    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                continue
        if paths:
            return datetime.fromtimestamp(min(path.stat().st_mtime for path in paths), tz=UTC)
        return fallback

    @staticmethod
    def _directory_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
