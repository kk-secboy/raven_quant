"""Observe long model computations without terminating them based on elapsed time."""
from __future__ import annotations

import json
import logging
import stat
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from .model_cell_store import atomic_json

MODEL_EXECUTION_PROGRESS_VERSION = "model-execution-progress-v1-observe-only"
ELAPSED_WARNING_SECONDS = 1800
PROGRESS_WARNING_SECONDS = 1800
POLL_SECONDS = 30
SHUTDOWN_WAIT_SECONDS = 0.2
_LOG = logging.getLogger(__name__)
_PROGRESS_FILES = (
    "data-preparation/audit/progress.jsonl",
    "output/memory_stages.jsonl",
    "output/checkpoint.pt",
    "output/checkpoint.txt",
    "output/checkpoint.json",
    "output/predictions.parquet",
    "output/portfolio_report.parquet",
)


class ModelExecutionMonitor:
    """A missing progress observation requests inspection; it never proves a hang.

    Records are operational only, outside immutable model outputs. Their updates
    cannot count as model progress or reset the model's progress warning clock.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.path = workspace / "execution-progress.json"
        self.started = time.monotonic()
        self.last_progress = self.started
        self.phase = "waiting_for_prepared_data"
        self.status = "running"
        self.signature: tuple = ()
        self.observed: dict[str, tuple] = {}
        self.last_record: dict = {}
        self.warnings: list[str] = []
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def set_phase(self, phase: str) -> None:
        with self.lock:
            if self.phase != phase:
                self.phase = phase
                self.last_progress = time.monotonic()
        self.sample()

    def _observations(self) -> tuple:
        result = []
        for relative in _PROGRESS_FILES:
            try:
                info = (self.workspace / relative).lstat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                result.append((relative, info.st_size, info.st_mtime_ns))
        return tuple(result)

    def sample(self) -> dict:
        try:
            return self._sample()
        except Exception:
            self._log_unavailable()
            return self.last_record

    @staticmethod
    def _log_unavailable() -> None:
        # A custom logging handler is also an observational dependency.
        with suppress(Exception):
            _LOG.exception("Could not observe model execution progress; continuing")

    def _sample(self) -> dict:
        signature = self._observations()
        with self.lock:
            now = time.monotonic()
            # A temporarily unreadable/disappearing file is not computation.
            if any(self.observed.get(row[0]) != row[1:] for row in signature):
                self.last_progress = now
            self.signature = signature
            self.observed.update({row[0]: row[1:] for row in signature})
            elapsed = max(0.0, now - self.started)
            quiet = max(0.0, now - self.last_progress)
            warnings = []
            if self.status == "running":
                if elapsed >= ELAPSED_WARNING_SECONDS:
                    warnings.append("elapsed_warning")
                if quiet >= PROGRESS_WARNING_SECONDS:
                    warnings.append("progress_not_observed")
            record = {
                "contract_version": MODEL_EXECUTION_PROGRESS_VERSION,
                "updated_at": datetime.now(UTC).isoformat(),
                "status": self.status,
                "phase": self.phase,
                "elapsed_seconds": round(elapsed, 3),
                "seconds_without_observed_progress": round(quiet, 3),
                "warnings": warnings,
                "automatic_termination": False,
                "wall_clock_deadline": None,
                "elapsed_warning_seconds": ELAPSED_WARNING_SECONDS,
                "progress_warning_seconds": PROGRESS_WARNING_SECONDS,
                "observed_files": [row[0] for row in signature],
            }
            changed_warning = warnings != self.warnings and warnings
            self.warnings = warnings
            self.last_record = record
        # Never hold the state lock across logging or filesystem operations.
        if changed_warning:
            _LOG.warning("Model computation needs inspection; continuing: %s",
                         json.dumps(record, ensure_ascii=False))
        atomic_json(self.path, record)
        return record

    def finish(self, status: str) -> None:
        with self.lock:
            self.status = status

    def _poll(self) -> None:
        while not self.stop.wait(POLL_SECONDS):
            try:
                self.sample()
            except Exception:
                self._log_unavailable()

    def __enter__(self) -> ModelExecutionMonitor:
        self.sample()
        try:
            self.thread = threading.Thread(target=self._poll, name="model-progress", daemon=True)
            self.thread.start()
        except Exception:
            self.thread = None
            self._log_unavailable()
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        if exc_type is not None:
            self.finish("failed" if issubclass(exc_type, Exception) else "interrupted")
        self.stop.set()
        if self.thread is not None:
            try:
                self.thread.join(timeout=SHUTDOWN_WAIT_SECONDS)
                if self.thread.is_alive():
                    return  # A slow observer cannot delay completion or cancellation.
            except Exception:
                self._log_unavailable()
        self.sample()
