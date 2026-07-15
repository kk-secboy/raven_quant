from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from quant_data.catalog import (
    CORE_DAILY,
    CORPORATE_EVENTS,
    ETF_DAILY,
    FUNDAMENTALS,
    RESEARCH_DAILY,
)
from quant_data.checkpoint import CheckpointStore
from quant_data.config import Settings


def resolve_snapshot_dataset(
    data_root: Path,
    *,
    snapshot_name: str,
    dataset_name: str,
) -> dict[str, Any]:
    snapshots_root = (data_root / "snapshots").resolve()
    snapshot = (snapshots_root / snapshot_name).resolve()
    try:
        snapshot.relative_to(snapshots_root)
    except ValueError as exc:
        raise ValueError("snapshot name resolves outside the snapshot root") from exc
    manifest_path = snapshot / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("immutable snapshot manifest is missing or invalid") from exc
    entry = manifest.get("datasets", {}).get(dataset_name)
    if not isinstance(entry, dict):
        raise ValueError(f"snapshot does not contain dataset {dataset_name}")
    source_sha256 = str(entry.get("source_sha256") or "")
    if len(source_sha256) != 64:
        raise ValueError("snapshot dataset has no source identity")
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("snapshot dataset has no immutable Parquet files")
    resolved_files: list[str] = []
    parquet_root = (snapshot / "parquet").resolve()
    dataset_path = (parquet_root / dataset_name).resolve()
    try:
        dataset_path.relative_to(parquet_root)
    except ValueError as exc:
        raise ValueError("dataset name resolves outside the snapshot Parquet root") from exc
    for item in files:
        relative = Path(str(item.get("path") or ""))
        target = (snapshot / relative).resolve()
        try:
            target.relative_to(snapshot)
        except ValueError as exc:
            raise ValueError("snapshot manifest contains an unsafe file path") from exc
        if not target.is_file() or target.stat().st_size != int(item.get("bytes") or -1):
            raise ValueError(f"snapshot file is missing or has the wrong size: {relative}")
        resolved_files.append(str(target))
    return {
        "snapshot_name": snapshot_name,
        "dataset_name": dataset_name,
        "snapshot_path": str(snapshot),
        "dataset_path": str(dataset_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_sha256": source_sha256,
        "snapshot_lineage_id": manifest.get("lineage_id"),
        "snapshot_lineage_generation": manifest.get("lineage_generation"),
        "parent_snapshot": manifest.get("parent_snapshot"),
        "start_date": manifest.get("start_date"),
        "end_date": manifest.get("end_date"),
        "rows": int(entry.get("rows") or 0),
        "files": resolved_files,
    }


def resolve_snapshot_manifest(data_root: Path, snapshot_name: str) -> dict[str, Any]:
    """Resolve and validate an immutable snapshot without trusting its name as a path."""

    snapshots_root = (data_root / "snapshots").resolve()
    snapshot = (snapshots_root / snapshot_name).resolve()
    try:
        snapshot.relative_to(snapshots_root)
    except ValueError as exc:
        raise ValueError("snapshot name resolves outside the snapshot root") from exc
    manifest_path = snapshot / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("immutable snapshot manifest is missing or invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("immutable snapshot manifest is missing or invalid")
    return {
        "name": snapshot.name,
        "path": str(snapshot),
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def dataset_catalog(checkpoint: CheckpointStore) -> list[dict[str, Any]]:
    profiles: dict[str, str] = {
        "stock_basic": "core",
        "trade_cal": "core",
        "index_basic": "core",
        "index_daily": "core",
        "index_dailybasic": "core",
        "index_weight": "core",
        "fund_basic": "research",
        "index_classify": "research",
        "index_member_all": "research",
        "disclosure_date": "research",
        "news": "full",
    }
    profiles.update({item.name: "core" for item in CORE_DAILY})
    profiles.update({item.name: "research" for item in (*RESEARCH_DAILY, *ETF_DAILY)})
    profiles.update({item.name: "full" for item in (*FUNDAMENTALS, *CORPORATE_EVENTS)})
    profiles.update(
        {
            "margin_eligibility": "research",
            "indices_1m": "research",
            "etf_1m": "research",
            "futures_1m": "research",
            "options_1m": "research",
            "liquid_stocks_1m": "research",
        }
    )
    aggregates: dict[str, dict[str, int]] = defaultdict(
        lambda: {"planned": 0, "succeeded": 0, "failed": 0, "running": 0, "rows": 0}
    )
    for row in checkpoint.counts():
        item = aggregates[row["dataset"]]
        item["planned"] += int(row["units"])
        item[row["status"]] = int(row["units"])
        item["rows"] += int(row["rows"])
    result = []
    for name, profile in sorted(profiles.items(), key=lambda item: (item[1], item[0])):
        stats = aggregates[name]
        completed = stats["succeeded"]
        planned = stats["planned"]
        state = "ready" if planned and completed == planned else "partial" if completed else "empty"
        result.append(
            {
                "name": name,
                "profile": profile,
                **stats,
                "coverage": round(completed / planned * 100, 1) if planned else 0.0,
                "state": state,
            }
        )
    return result


def list_snapshots(data_root: Path) -> list[dict[str, Any]]:
    root = data_root / "snapshots"
    snapshots = []
    if not root.exists():
        return snapshots
    for path in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
        manifest_path = path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {"name": path.name, "datasets": {}, "invalid": True}
        snapshots.append(manifest)
    return snapshots


def list_qlib_datasets(data_root: Path) -> list[dict[str, Any]]:
    root = data_root / "qlib"
    datasets = []
    if not root.exists():
        return datasets
    for path in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
        provenance_path = path / "metadata" / "provenance.json"
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            provenance = None
        frequency = str((provenance or {}).get("frequency") or "day")
        calendar = path / "calendars" / f"{frequency}.txt"
        instrument_candidates = (
            path / "instruments" / "cn_all.txt",
            path / "instruments" / "liquid_all.txt",
            path / "instruments" / "all.txt",
        )
        instruments = next(
            (candidate for candidate in instrument_candidates if candidate.exists()),
            instrument_candidates[-1],
        )
        features = path / "features"
        days = calendar.read_text(encoding="utf-8").splitlines() if calendar.exists() else []
        stocks = (
            instruments.read_text(encoding="utf-8").splitlines() if instruments.exists() else []
        )
        datasets.append(
            {
                "name": path.name,
                "path": str(path),
                "ready": bool(days and stocks and features.exists()),
                "reproducible": bool(
                    provenance
                    and provenance.get("dataset_identity_sha256")
                    and provenance.get("snapshot_manifest_sha256")
                ),
                "provenance": provenance,
                "frequency": frequency,
                "lineage_id": (provenance or {}).get("dataset_lineage_id"),
                "lineage_verified": bool((provenance or {}).get("lineage_verified")),
                "start_date": days[0] if days else None,
                "end_date": days[-1] if days else None,
                "trading_days": len(days),
                "instruments": len(stocks),
            }
        )
    return datasets


def list_qlib_experiments(data_root: Path) -> list[dict[str, Any]]:
    root = data_root / "artifacts" / "qlib"
    experiments = []
    if not root.exists():
        return experiments
    for path in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
        result_path = path / "result.json"
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        experiments.append({"id": path.name, **result})
    return experiments


def probe_qlib(settings: Settings, project_root: Path) -> dict[str, Any]:
    if settings.qlib_worker_url:
        try:
            response = requests.get(
                f"{settings.qlib_worker_url}/qlib/status",
                timeout=8,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            return {"status": "unavailable", "error": str(exc)}
    script = project_root / "scripts" / "run_qlib_baseline.py"
    is_wsl = os.name == "nt" and settings.qlib_python.startswith("/")
    command = (
        [
            "wsl",
            "-d",
            settings.qlib_wsl_distro,
            "--exec",
            settings.qlib_python,
            _windows_to_wsl(script),
            "--probe",
        ]
        if is_wsl
        else [settings.qlib_python, str(script), "--probe"]
    )
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=45, check=True)
        line = next(
            (item for item in reversed(completed.stdout.splitlines()) if item.startswith("{")),
            "{}",
        )
        return json.loads(line)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "error": str(exc)}


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}{resolved.as_posix().split(':', 1)[1]}" if drive else resolved.as_posix()


def system_summary(
    settings: Settings,
    checkpoint: CheckpointStore,
    jobs: list[dict],
    data_tasks: list[dict] | None = None,
) -> dict:
    catalog = dataset_catalog(checkpoint)
    total_rows = sum(item["rows"] for item in catalog)
    planned = sum(item["planned"] for item in catalog)
    succeeded = sum(item["succeeded"] for item in catalog)
    snapshots = list_snapshots(settings.data_root)
    qlib_datasets = len(list_qlib_datasets(settings.data_root))
    active_jobs = sum(job["status"] in {"queued", "running"} for job in jobs)
    actionable_tasks = [
        task
        for task in (data_tasks or [])
        if task.get("implementation_status")
        not in {"permission_probe", "external_source_required"}
    ]
    ready_tasks = sum(task.get("status") == "succeeded" for task in actionable_tasks)
    partial_tasks = sum(task.get("status") == "partial" for task in actionable_tasks)
    failed_tasks = sum(task.get("status") == "failed" for task in actionable_tasks)
    running_tasks = sum(
        task.get("status") in {"queued", "running"} for task in actionable_tasks
    )
    waiting_tasks = max(
        len(actionable_tasks) - ready_tasks - partial_tasks - failed_tasks - running_tasks,
        0,
    )
    readiness_percent = (
        round(
            sum(
                100.0
                if task.get("status") == "succeeded"
                else float(task.get("coverage") or 0.0)
                for task in actionable_tasks
            )
            / len(actionable_tasks),
            1,
        )
        if actionable_tasks
        else 0.0
    )
    legacy_download_coverage = round(succeeded / planned * 100, 1) if planned else 0.0
    return {
        "mode": "local-research",
        "credentials_configured": bool(settings.api_url and settings.token),
        "data_root": str(settings.data_root),
        "rows": total_rows,
        "planned_units": planned,
        "succeeded_units": succeeded,
        # Kept for API compatibility. The Web UI labels this as the legacy
        # foundational download completion and never as overall readiness.
        "coverage": legacy_download_coverage,
        "legacy_download_coverage": legacy_download_coverage,
        "readiness_percent": readiness_percent,
        "ready_tasks": ready_tasks,
        "actionable_tasks": len(actionable_tasks),
        "partial_tasks": partial_tasks,
        "failed_tasks": failed_tasks,
        "running_tasks": running_tasks,
        "waiting_tasks": waiting_tasks,
        "snapshots": len(snapshots),
        "qlib_datasets": qlib_datasets,
        "active_jobs": active_jobs,
        "components": [
            {"name": "PostgreSQL", "state": "ready"},
            {"name": "Data Center", "state": "ready"},
            {"name": "Qlib", "state": "configured" if settings.qlib_python else "needs_setup"},
            {"name": "RD-Agent", "state": "configured"},
            {"name": "Paper Portfolio", "state": "ready"},
            {"name": "Scheduler", "state": "configured" if settings.scheduler_url else "local"},
        ],
    }
