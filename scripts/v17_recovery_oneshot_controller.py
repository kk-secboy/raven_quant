#!/usr/bin/env python3
"""Execute the single authorized v17 continuation inside the sealed old image.

This file is bind-mounted read-only into the historical worker image.  It does
not run the worker queue loop and does not invoke interrupted-job recovery: it
locks and claims the one sealed job, then calls the old image's worker `_run`
method exactly once.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import text

from quant_data.config import Settings
from quant_data.database import row_dict
from quant_platform.job_store import JobStore
from quant_platform.worker import LocalJobWorker

CONTRACT_VERSION = "quantlab-v17-sealed-one-shot-controller-v1"
APPLICATION_NAME = "quantlab-v17-recovery-b41f78f9"
AUTHORIZATION_APPLICATION_NAME = "quantlab-v17-recovery-authorizer"
JOB_ID = "858a75a6f1994c359fa9c3567ed09f57"
BACKTEST_ID = "0a113fe28ca741b6be9c09ab046c9d02"
STRATEGY_VERSION_ID = "4414d202dbb641608975e5305bc18da4"
PAYLOAD_SHA256 = "f99fa6c8f2a1364d56a0d0bff9d7401b1f504f3b020c321d996f05a70a05df42"
WORKER_IMAGE_DIGEST = (
    "sha256:b41f78f9c99dd9853998d85a52593bcab247907ac60f9d721d863e20904e8bb7"
)
CANONICAL_OUTPUT_PATH = f"/data/artifacts/backtests/{BACKTEST_ID}"
TARGET_ARTIFACT_PATH = (
    f"/data/artifacts/formal-backtest-recoveries/{BACKTEST_ID}/attempt-2"
)
TARGET_LOG_PATH = (
    f"/data/artifacts/formal-backtest-recoveries/{BACKTEST_ID}/attempt-2.log"
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mount_points() -> dict[str, frozenset[str]]:
    escapes = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
    points: dict[str, frozenset[str]] = {}
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split(" - ", 1)[0].split()
        if len(fields) < 6:
            raise RuntimeError("sealed recovery mountinfo record is malformed")
        mount_point = fields[4]
        for encoded, decoded in escapes.items():
            mount_point = mount_point.replace(encoded, decoded)
        points[mount_point] = frozenset(fields[5].split(","))
    return points


def _require_mount_contract() -> str:
    canonical = Path(CANONICAL_OUTPUT_PATH)
    target = Path(TARGET_ARTIFACT_PATH)
    mount_points = _mount_points()
    if (
        not canonical.is_dir()
        or not target.is_dir()
        or canonical.is_symlink()
        or target.is_symlink()
        or "ro" not in mount_points.get("/data", frozenset())
        or "rw" in mount_points.get("/data", frozenset())
        or "rw" not in mount_points.get(str(canonical), frozenset())
        or "ro" in mount_points.get(str(canonical), frozenset())
        or str(target) in mount_points
        or not os.path.samefile(canonical, target)
    ):
        raise RuntimeError("sealed recovery output bind is not the exact same-file target")
    if any(target.iterdir()):
        raise RuntimeError("sealed recovery attempt-2 artifact target is not empty")
    probe = canonical / ".quantlab-v17-recovery-write-probe"
    try:
        with probe.open("xb") as handle:
            handle.write(b"persistent-target-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not (target / probe.name).is_file():
            raise RuntimeError("sealed recovery write probe did not reach persistent target")
    finally:
        probe.unlink(missing_ok=True)
    if any(target.iterdir()):
        raise RuntimeError("sealed recovery write probe cleanup did not restore empty target")
    log_path = Path(TARGET_LOG_PATH)
    if (
        log_path.is_symlink()
        or not log_path.is_file()
        or "rw" not in mount_points.get(str(log_path), frozenset())
        or "ro" in mount_points.get(str(log_path), frozenset())
    ):
        raise RuntimeError("sealed recovery execution log path is not a regular file")
    if log_path.stat().st_size:
        raise RuntimeError("sealed recovery execution log is not empty")
    try:
        with log_path.open("ab"):
            pass
    except OSError as exc:
        raise RuntimeError("sealed recovery execution log bind is not writable") from exc
    return _sha256_file(Path(__file__).resolve())


def _require_receipt_and_claim(store: JobStore, *, controller_sha256: str) -> dict[str, Any]:
    with store.engine.begin() as connection:
        application_name = connection.scalar(
            text("SELECT current_setting('application_name', true)")
        )
        if str(application_name or "") != APPLICATION_NAME:
            raise RuntimeError("sealed recovery database application_name is not authorized")
        receipt_row = connection.execute(
            text(
                "SELECT receipt_sha256, verification_json, target_artifact_path "
                "FROM quantlab.formal_backtest_interruption_recoveries "
                "WHERE job_id = :job_id AND backtest_id = :backtest_id FOR SHARE"
            ),
            {"job_id": JOB_ID, "backtest_id": BACKTEST_ID},
        ).mappings().one_or_none()
        if receipt_row is None:
            raise RuntimeError("sealed recovery receipt is missing")
        receipt = dict(receipt_row["verification_json"] or {})
        supplied_receipt_sha256 = str(receipt.pop("receipt_sha256", ""))
        controller = dict(receipt.get("execution_controller") or {})
        if (
            _canonical_sha256(receipt) != supplied_receipt_sha256
            or str(receipt_row["receipt_sha256"]) != supplied_receipt_sha256
            or str(receipt_row["target_artifact_path"]) != TARGET_ARTIFACT_PATH
            or controller
            != {
                "contract_version": CONTRACT_VERSION,
                "application_name": APPLICATION_NAME,
                "authorization_application_name": AUTHORIZATION_APPLICATION_NAME,
                "controller_sha256": controller_sha256,
                "sealed_worker_image_digest": WORKER_IMAGE_DIGEST,
                "canonical_output_path": CANONICAL_OUTPUT_PATH,
                "target_artifact_path": TARGET_ARTIFACT_PATH,
                "target_execution_log_path": TARGET_LOG_PATH,
                "data_mount_mode": (
                    "volumes-from-read-only-with-persistent-target-samefile-bind"
                ),
                "target_execution_log_mount_mode": "single-file-read-write-bind",
            }
        ):
            raise RuntimeError("sealed recovery receipt/controller binding is invalid")
        backtest = connection.execute(
            text(
                "SELECT status, job_id, strategy_version_id, artifact_path, metrics_json, "
                "error, finished_at FROM quantlab.backtest_runs "
                "WHERE id = :backtest_id FOR UPDATE"
            ),
            {"backtest_id": BACKTEST_ID},
        ).mappings().one_or_none()
        if (
            backtest is None
            or str(backtest["status"]) != "queued"
            or str(backtest["job_id"] or "") != JOB_ID
            or str(backtest["strategy_version_id"]) != STRATEGY_VERSION_ID
            or str(backtest["artifact_path"] or "") != TARGET_ARTIFACT_PATH
            or backtest["metrics_json"] is not None
            or backtest["error"] is not None
            or backtest["finished_at"] is not None
        ):
            raise RuntimeError("sealed recovery backtest aggregate is not queued exactly once")
        claimed = connection.execute(
            text(
                "UPDATE quantlab.jobs SET status = 'running', attempts = 2, "
                "started_at = now(), finished_at = NULL, next_attempt_at = NULL, "
                "cancel_requested_at = NULL, exit_code = NULL, error = NULL "
                "WHERE id = :job_id AND kind = 'strategy_backtest' "
                "AND status = 'queued' AND attempts = 1 AND max_attempts = 2 "
                "RETURNING *"
            ),
            {"job_id": JOB_ID},
        ).first()
        if claimed is None:
            raise RuntimeError("sealed recovery job could not be claimed exactly once")
        job = JobStore._decode(row_dict(claimed))
        if (
            str(job["id"]) != JOB_ID
            or str(job["kind"]) != "strategy_backtest"
            or str(job["status"]) != "running"
            or int(job["attempts"]) != 2
            or int(job["max_attempts"]) != 2
            or _canonical_sha256(job["payload"]) != PAYLOAD_SHA256
        ):
            raise RuntimeError("sealed recovery claimed job identity changed")
    job["log_path"] = TARGET_LOG_PATH
    return job


def main() -> None:
    if os.environ.get("QUANTLAB_WORKER_RUNTIME_IMAGE_DIGEST") != WORKER_IMAGE_DIGEST:
        raise RuntimeError("sealed v17 worker runtime image environment changed")
    settings = Settings.from_env(Path("/nonexistent/quantlab-v17-recovery.env"))
    if settings.data_root != Path("/data"):
        raise RuntimeError("sealed v17 recovery DATA_ROOT must be /data")
    if settings.worker_job_kinds != ("strategy_backtest",):
        raise RuntimeError("sealed v17 recovery worker kind must be strategy_backtest only")
    if settings.worker_concurrency != 1:
        raise RuntimeError("sealed v17 recovery worker concurrency must be one")
    controller_sha256 = _require_mount_contract()
    store = JobStore(settings.database_url)
    job = _require_receipt_and_claim(store, controller_sha256=controller_sha256)
    worker = LocalJobWorker(
        store,
        Path("/app"),
        settings,
        initialize_queue=False,
    )
    worker._run(job)
    current = store.get(JOB_ID)
    backtest = worker.strategies.get_backtest(BACKTEST_ID)
    if (
        int(current["attempts"]) != 2
        or int(current["max_attempts"]) != 2
        or current["status"] not in {"succeeded", "failed", "cancelled"}
        or backtest["status"] != current["status"]
    ):
        raise RuntimeError("sealed v17 recovery did not reach one consistent terminal state")
    if current["status"] != "succeeded":
        raise RuntimeError(f"sealed v17 recovery ended {current['status']}")


if __name__ == "__main__":
    main()
