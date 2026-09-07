from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.no_database
RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "model_sandbox_runner.py"


def runner_module():
    spec = importlib.util.spec_from_file_location("model_sandbox_workflow_contract", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracker_is_durable_sibling_and_preserves_old_records(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://unreachable.invalid/old-tracker")
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", str(tmp_path / "old-artifacts"))
    output = tmp_path / "output"
    old_run = output / "qlib-workflow" / "artifacts" / "tracking" / "old-run.yaml"
    old_run.parent.mkdir(parents=True)
    old_run.write_bytes(b"preserved old failed run")
    calls = []
    runtime = SimpleNamespace(init=lambda **kwargs: calls.append(kwargs))

    uri = runner_module().initialize_model_qlib(
        runtime, output=output, provider_uri="/qlib", kernels=1,
    )

    assert uri == str(output / "qlib-workflow" / "tracking")
    assert Path(uri).is_dir()
    assert "artifacts" not in Path(uri).parts
    assert old_run.read_bytes() == b"preserved old failed run"
    assert os.environ["MLFLOW_TRACKING_URI"] == uri
    assert os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] == str(old_run.parents[1])
    assert calls == [{
        "provider_uri": "/qlib", "region": "cn", "kernels": 1,
        "exp_manager": {"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                        "kwargs": {"uri": uri, "default_exp_name": "Experiment"}},
    }]


@pytest.mark.parametrize("relative", [True, False])
def test_unsafe_tracking_path_rejected_before_initialization(tmp_path, relative):
    output = Path("relative-output") if relative else tmp_path / "artifacts" / "output"
    calls = []
    runtime = SimpleNamespace(init=lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="tracking path must be absolute and outside artifacts"):
        runner_module().initialize_model_qlib(
            runtime, output=output, provider_uri="/qlib", kernels=1,
        )
    assert calls == []
    if not relative:
        assert not output.exists()


def test_main_binds_workflow_to_initialized_uri_instead_of_mutable_environment():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assignments = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                   and n.value.func.id == "initialize_model_qlib"]
    assert len(assignments) == 1
    binding = assignments[0]
    name = binding.targets[0].id
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    governed = [n for n in calls if isinstance(n.func, ast.Name)
                and n.func.id == "qlib_workflow_run"]
    assert len(governed) == 1
    uri = next(k.value for k in governed[0].keywords if k.arg == "tracking_uri")
    assert isinstance(uri, ast.Name) and uri.id == name
    assert not any(isinstance(n.func, ast.Name) and n.func.id == "qlib_workflow_tracking_uri"
                   for n in calls)
    assert all(binding.lineno < n.lineno < governed[0].lineno for n in calls
               if isinstance(n.func, ast.Attribute) and n.func.attr == "fit")


@pytest.mark.parametrize("implicit_training_recorder", [False, True])
def test_real_qlib_uses_same_tracker_with_or_without_implicit_training(
    tmp_path, monkeypatch, implicit_training_recorder,
):
    qlib = pytest.importorskip("qlib")
    mlflow = pytest.importorskip("mlflow")
    from qlib.workflow import R

    module = runner_module()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("GIT_PYTHON_REFRESH", "quiet")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://unreachable.invalid/old-tracker")
    monkeypatch.setenv("_MLFLOW_SERVER_ARTIFACT_ROOT", str(tmp_path / "unrelated"))
    provider = tmp_path / "empty-provider"
    provider.mkdir()
    output = tmp_path / "output"
    uri = module.initialize_model_qlib(
        qlib, output=output, provider_uri=str(provider), kernels=1,
    )
    implicit_id = None
    try:
        if implicit_training_recorder:
            # Exercise the actual Qlib side effect used by LGBM, without fitting.
            R.log_metrics(synthetic_only=0)
            implicit_id = str(R.get_recorder().id)
            module.end_implicit_qlib_recorder()
        assert mlflow.active_run() is None
        # The portfolio stage must ignore a subsequently changed environment.
        monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "artifacts" / "bad"))
        with module.qlib_workflow_run(
            run_kind="model-tracker-fixture", run_id="synthetic",
            tracking_uri=uri, dataset_identity_sha256="a" * 64,
        ) as workflow:
            workflow.log_params({"synthetic_only": True})
            identity = workflow.identity_dict()
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        recorded = client.get_run(identity["recorder_id"])
        assert recorded.info.status == "FINISHED"
        assert recorded.data.params["synthetic_only"] == "True"
        assert recorded.data.tags["dataset_identity_sha256"] == "a" * 64
        assert (Path(uri) / identity["experiment_id"] / identity["recorder_id"]).is_dir()
        if implicit_id is not None:
            assert client.get_run(implicit_id).info.status == "FINISHED"
        assert not (tmp_path / "artifacts" / "bad").exists()
    finally:
        R.end_exp()
