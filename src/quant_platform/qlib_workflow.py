from __future__ import annotations

import math
import os
import re
import sys
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

from .upstream_versions import QLIB_COMMIT, upstream_runtime_identity

QLIB_WORKFLOW_ADAPTER_VERSION = "quantlab-qlib-workflow-v1"
_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _load_qlib_recorder() -> Any:
    try:
        from qlib.workflow import R
    except ImportError as exc:  # pragma: no cover - configured runtime assertion
        raise RuntimeError("Qlib Workflow/Recorder runtime is unavailable") from exc
    return R


def _safe_name(value: str, *, fallback: str) -> str:
    normalized = _NAME_PATTERN.sub("-", str(value).strip()).strip("-.")
    return (normalized or fallback)[:240]


def _parameter_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _finite_metrics(values: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            metrics[_safe_name(key, fallback="metric")] = number
    return metrics


@dataclass(frozen=True)
class QlibRecorderIdentity:
    adapter_version: str
    experiment_id: str
    experiment_name: str
    recorder_id: str
    recorder_name: str
    tracking_backend: str
    artifact_backend: str
    qlib_version: str
    qlib_commit: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class QlibWorkflowRun(AbstractContextManager["QlibWorkflowRun"]):
    """Mandatory production adapter for Qlib Workflow/Recorder evidence."""

    def __init__(
        self,
        *,
        run_kind: str,
        run_id: str,
        tracking_uri: str,
        dataset_identity_sha256: str | None = None,
    ) -> None:
        if not str(tracking_uri).strip():
            raise ValueError("Qlib Workflow/Recorder tracking URI is required")
        dataset_identity = str(dataset_identity_sha256 or "").strip().lower()
        if (
            len(dataset_identity) != 64
            or any(character not in "0123456789abcdef" for character in dataset_identity)
        ):
            raise ValueError("Qlib Workflow/Recorder requires an immutable dataset identity")
        artifact_root = os.getenv("_MLFLOW_SERVER_ARTIFACT_ROOT", "").strip()
        if not artifact_root:
            raise ValueError("Qlib Workflow/Recorder durable artifact root is required")
        if "://" not in artifact_root and not Path(artifact_root).is_absolute():
            raise ValueError(
                "Qlib Workflow/Recorder local artifact root must be an absolute durable path"
            )
        self.run_kind = _safe_name(run_kind, fallback="run")
        self.run_id = _safe_name(run_id, fallback="unknown")
        self.tracking_uri = str(tracking_uri).strip()
        self.dataset_identity_sha256 = dataset_identity
        self.artifact_backend = (
            urlparse(artifact_root).scheme if "://" in artifact_root else "file"
        )
        self.experiment_name = f"quantlab-{self.run_kind}"
        self.recorder_name = f"{self.run_kind}-{self.run_id}"
        self.identity: QlibRecorderIdentity | None = None
        self._recorder_api: Any = None
        self._context: Any = None

    def __enter__(self) -> QlibWorkflowRun:
        runtime = upstream_runtime_identity("qlib")
        recorder_api = _load_qlib_recorder()
        resume = self.identity is not None
        try:
            context = recorder_api.start(
                experiment_id=self.identity.experiment_id if self.identity else None,
                experiment_name=None if resume else self.experiment_name,
                recorder_id=self.identity.recorder_id if self.identity else None,
                recorder_name=None if resume else self.recorder_name,
                uri=self.tracking_uri,
                resume=resume,
            )
        except AttributeError as exc:
            raise RuntimeError(
                "Qlib must be initialized with the governed dataset before Workflow/Recorder"
            ) from exc
        experiment = context.__enter__()
        try:
            recorder = recorder_api.get_recorder()
        except BaseException:
            context.__exit__(*sys.exc_info())
            raise
        self._recorder_api = recorder_api
        self._context = context
        if self.identity is None:
            self.identity = QlibRecorderIdentity(
                adapter_version=QLIB_WORKFLOW_ADAPTER_VERSION,
                experiment_id=str(experiment.id),
                experiment_name=self.experiment_name,
                recorder_id=str(recorder.id),
                recorder_name=self.recorder_name,
                tracking_backend=urlparse(self.tracking_uri).scheme or "file",
                artifact_backend=self.artifact_backend,
                qlib_version=str(runtime["version"]),
                qlib_commit=str(runtime["commit"]),
            )
            try:
                self.log_params(
                    {
                        "workflow_adapter_version": QLIB_WORKFLOW_ADAPTER_VERSION,
                        "run_kind": self.run_kind,
                        "run_id": self.run_id,
                        "qlib_version": runtime["version"],
                        "qlib_commit": runtime["commit"],
                    }
                )
                self.set_tags(
                    {
                        "production_path": "qlib-workflow-recorder",
                        "dataset_identity_sha256": self.dataset_identity_sha256,
                    }
                )
            except BaseException:
                context.__exit__(*sys.exc_info())
                self._context = None
                self._recorder_api = None
                self.identity = None
                raise
        elif (
            str(recorder.id) != self.identity.recorder_id
            or str(experiment.id) != self.identity.experiment_id
            or str(runtime["version"]) != self.identity.qlib_version
            or str(runtime["commit"]) != self.identity.qlib_commit
        ):
            context.__exit__(
                RuntimeError,
                RuntimeError("Qlib Workflow/Recorder resume identity changed"),
                None,
            )
            raise RuntimeError("Qlib Workflow/Recorder resume identity changed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._context is None:
            return None
        try:
            return self._context.__exit__(exc_type, exc_value, traceback)
        finally:
            self._context = None
            self._recorder_api = None

    def _require_active(self) -> Any:
        if self._recorder_api is None or self.identity is None:
            raise RuntimeError("Qlib Workflow/Recorder run is not active")
        return self._recorder_api

    def identity_dict(self) -> dict[str, str]:
        if self.identity is None:
            raise RuntimeError("Qlib Workflow/Recorder identity is unavailable")
        return self.identity.to_dict()

    def log_params(self, values: dict[str, Any]) -> None:
        recorder = self._require_active()
        recorder.log_params(
            **{
                _safe_name(key, fallback="parameter"): _parameter_value(value)
                for key, value in values.items()
            }
        )

    def log_metrics(self, values: dict[str, Any]) -> None:
        recorder = self._require_active()
        metrics = _finite_metrics(values)
        if metrics:
            recorder.log_metrics(**metrics)

    def set_tags(self, values: dict[str, Any]) -> None:
        recorder = self._require_active()
        recorder.set_tags(
            **{
                _safe_name(key, fallback="tag"): str(value)
                for key, value in values.items()
                if value is not None
            }
        )

    def save_artifacts(self, path: str | Path, *, artifact_path: str = "production") -> None:
        recorder = self._require_active()
        source = Path(path).resolve()
        if not source.exists():
            raise ValueError(f"Qlib Recorder artifact path does not exist: {source}")
        recorder.save_objects(local_path=str(source), artifact_path=artifact_path)

    def list_metrics(self) -> dict[str, Any]:
        recorder = self._require_active().get_recorder()
        return dict(recorder.list_metrics())


def qlib_workflow_run(
    *,
    run_kind: str,
    run_id: str,
    tracking_uri: str,
    dataset_identity_sha256: str | None = None,
) -> QlibWorkflowRun:
    return QlibWorkflowRun(
        run_kind=run_kind,
        run_id=run_id,
        tracking_uri=tracking_uri,
        dataset_identity_sha256=dataset_identity_sha256,
    )


def require_qlib_workflow_identity(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Qlib Workflow/Recorder identity is required")
    required = {
        "adapter_version",
        "experiment_id",
        "experiment_name",
        "recorder_id",
        "recorder_name",
        "tracking_backend",
        "artifact_backend",
        "qlib_version",
        "qlib_commit",
    }
    if set(value) != required or any(not str(value.get(key) or "").strip() for key in required):
        raise ValueError("Qlib Workflow/Recorder identity is incomplete")
    if value["adapter_version"] != QLIB_WORKFLOW_ADAPTER_VERSION:
        raise ValueError("Qlib Workflow/Recorder adapter version is unsupported")
    if value["qlib_commit"] != QLIB_COMMIT:
        raise ValueError("Qlib Workflow/Recorder does not identify the pinned Qlib commit")
    if value["qlib_version"] == "unknown":
        raise ValueError("Qlib Workflow/Recorder version is unavailable")
    return {key: str(value[key]) for key in sorted(required)}
