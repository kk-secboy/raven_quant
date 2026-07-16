from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform import qlib_workflow

pytestmark = pytest.mark.no_database


class _FakeContext:
    def __init__(self, api: _FakeRecorderApi) -> None:
        self.api = api

    def __enter__(self) -> SimpleNamespace:
        self.api.entered = True
        return SimpleNamespace(id="experiment-1")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.api.exit_exception = exc_value


class _FakeRecorderApi:
    def __init__(self) -> None:
        self.entered = False
        self.exit_exception = None
        self.start_kwargs: dict = {}
        self.start_calls: list[dict] = []
        self.params: dict = {}
        self.metrics: dict = {}
        self.tags: dict = {}
        self.saved: list[tuple[str, str]] = []

    def start(self, **kwargs):
        self.start_kwargs = kwargs
        self.start_calls.append(kwargs)
        return _FakeContext(self)

    @staticmethod
    def get_recorder() -> SimpleNamespace:
        return SimpleNamespace(id="recorder-1")

    def log_params(self, **kwargs) -> None:
        self.params.update(kwargs)

    def log_metrics(self, **kwargs) -> None:
        self.metrics.update(kwargs)

    def set_tags(self, **kwargs) -> None:
        self.tags.update(kwargs)

    def save_objects(self, *, local_path: str, artifact_path: str) -> None:
        self.saved.append((local_path, artifact_path))


def test_workflow_records_pinned_identity_metrics_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeRecorderApi()
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", str(tmp_path / "mlflow"))
    monkeypatch.setattr(qlib_workflow, "_load_qlib_recorder", lambda: fake)
    monkeypatch.setattr(
        qlib_workflow,
        "upstream_runtime_identity",
        lambda _kind: {
            "version": "0.0.dev0+gd5379c5",
            "commit": qlib_workflow.QLIB_COMMIT,
        },
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    with qlib_workflow.qlib_workflow_run(
        run_kind="formal backtest",
        run_id="strategy/version",
        tracking_uri="postgresql://tracking",
        dataset_identity_sha256="a" * 64,
    ) as workflow:
        workflow.log_params({"benchmark": "SH000300"})
        workflow.log_metrics({"return": 0.12, "ignored": None, "flag": True})
        workflow.save_artifacts(artifact)
        identity = workflow.identity_dict()

    assert fake.start_kwargs == {
        "experiment_id": None,
        "experiment_name": "quantlab-formal-backtest",
        "recorder_id": None,
        "recorder_name": "formal-backtest-strategy-version",
        "uri": "postgresql://tracking",
        "resume": False,
    }
    assert fake.params["workflow_adapter_version"] == qlib_workflow.QLIB_WORKFLOW_ADAPTER_VERSION
    assert fake.params["qlib_commit"] == qlib_workflow.QLIB_COMMIT
    assert fake.params["benchmark"] == "SH000300"
    assert fake.metrics == {"return": 0.12}
    assert fake.tags["production_path"] == "qlib-workflow-recorder"
    assert fake.tags["dataset_identity_sha256"] == "a" * 64
    assert fake.saved == [(str(artifact.resolve()), "production")]
    assert identity["recorder_id"] == "recorder-1"
    assert identity["qlib_version"] == "0.0.dev0+gd5379c5"
    assert identity["tracking_backend"] == "postgresql"
    assert identity["artifact_backend"] == "file"
    assert fake.exit_exception is None


def test_workflow_requires_real_tracking_backend() -> None:
    with pytest.raises(ValueError, match="tracking URI is required"):
        qlib_workflow.qlib_workflow_run(
            run_kind="factor",
            run_id="candidate",
            tracking_uri="",
            dataset_identity_sha256="a" * 64,
        )


def test_workflow_resumes_the_same_recorder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeRecorderApi()
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", str(tmp_path / "mlflow"))
    monkeypatch.setattr(qlib_workflow, "_load_qlib_recorder", lambda: fake)
    monkeypatch.setattr(
        qlib_workflow,
        "upstream_runtime_identity",
        lambda _kind: {
            "version": "0.0.dev0+gd5379c5",
            "commit": qlib_workflow.QLIB_COMMIT,
        },
    )
    workflow = qlib_workflow.qlib_workflow_run(
        run_kind="model-baseline",
        run_id="baseline-1",
        tracking_uri="postgresql://tracking",
        dataset_identity_sha256="a" * 64,
    )

    with workflow:
        pass
    with workflow:
        pass

    assert fake.start_calls[1] == {
        "experiment_id": "experiment-1",
        "experiment_name": None,
        "recorder_id": "recorder-1",
        "recorder_name": None,
        "uri": "postgresql://tracking",
        "resume": True,
    }


def test_workflow_does_not_fallback_when_recorder_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", "E:/data/mlflow")
    monkeypatch.setattr(
        qlib_workflow,
        "upstream_runtime_identity",
        lambda _kind: {"version": "version", "commit": "commit"},
    )

    def unavailable():
        raise RuntimeError("Qlib Workflow/Recorder runtime is unavailable")

    monkeypatch.setattr(qlib_workflow, "_load_qlib_recorder", unavailable)
    with pytest.raises(RuntimeError, match="runtime is unavailable"):
        with qlib_workflow.qlib_workflow_run(
            run_kind="factor",
            run_id="candidate",
            tracking_uri="postgresql://tracking",
            dataset_identity_sha256="a" * 64,
        ):
            pass


def test_workflow_requires_immutable_dataset_identity() -> None:
    with pytest.raises(ValueError, match="immutable dataset identity"):
        qlib_workflow.qlib_workflow_run(
            run_kind="factor",
            run_id="candidate",
            tracking_uri="postgresql://tracking",
            dataset_identity_sha256="not-a-digest",
        )


def test_workflow_requires_durable_artifact_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("_MLFLOW_SERVER_ARTIFACT_ROOT", raising=False)
    with pytest.raises(ValueError, match="durable artifact root"):
        qlib_workflow.qlib_workflow_run(
            run_kind="factor",
            run_id="candidate",
            tracking_uri="postgresql://tracking",
            dataset_identity_sha256="a" * 64,
        )
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", "./mlruns")
    with pytest.raises(ValueError, match="absolute durable path"):
        qlib_workflow.qlib_workflow_run(
            run_kind="factor",
            run_id="candidate",
            tracking_uri="postgresql://tracking",
            dataset_identity_sha256="a" * 64,
        )


def test_only_adapter_imports_the_global_qlib_recorder() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for base in (root / "src", root / "scripts"):
        for path in base.rglob("*.py"):
            if path.resolve() == Path(qlib_workflow.__file__).resolve():
                continue
            if "from qlib.workflow import R" in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(root).as_posix())
    assert offenders == []


def test_workflow_identity_rejects_unpinned_or_partial_evidence() -> None:
    with pytest.raises(ValueError, match="identity is incomplete"):
        qlib_workflow.require_qlib_workflow_identity({"recorder_id": "recorder"})

    identity = {
        "adapter_version": qlib_workflow.QLIB_WORKFLOW_ADAPTER_VERSION,
        "experiment_id": "experiment",
        "experiment_name": "quantlab-formal-backtest",
        "recorder_id": "recorder",
        "recorder_name": "formal-backtest-id",
        "tracking_backend": "postgresql",
        "artifact_backend": "file",
        "qlib_version": "0.0.dev0+gd5379c5",
        "qlib_commit": "b" * 40,
    }
    with pytest.raises(ValueError, match="pinned Qlib commit"):
        qlib_workflow.require_qlib_workflow_identity(identity)
