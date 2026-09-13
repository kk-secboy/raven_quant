from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_platform.fin_quant_model_dimensions import (
    BASELINE_MODEL_CONFIG,
    MODEL_CONFIGS,
    bind_model_dimensions,
    enable_fin_quant_model_dimensions,
)

pytestmark = pytest.mark.no_database


def _env(count=24):
    return {"feature_names": str([f"base_{i}" for i in range(count)]),
            "feature_expressions": str([f"Ref($close,{i})" for i in range(count)])}


def _workspace(tmp_path):
    return SimpleNamespace(workspace_path=tmp_path, file_dict={
        name: ('"num_features": 20,' if name == BASELINE_MODEL_CONFIG
               else '"num_features": {{ num_features }}')
        for name in MODEL_CONFIGS
    })


def _factors(tmp_path, names):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=2), ["SH600000", "SZ000001"]],
        names=["datetime", "instrument"])
    frame = pd.DataFrame(np.zeros((4, len(names))), index=index,
                         columns=pd.MultiIndex.from_product([["feature"], names]))
    frame.to_parquet(tmp_path / "combined_factors_df.parquet")


@pytest.mark.parametrize("count", [1, 20, 24, 158])
@pytest.mark.parametrize("dataset", ["DatasetH", "TSDatasetH"])
def test_baseline_dimension_tracks_registered_features_not_sequence_length(
    tmp_path, count, dataset,
):
    workspace = _workspace(tmp_path)
    env = {**_env(count), "dataset_cls": dataset}
    if dataset == "TSDatasetH":
        env.update(num_timesteps=20, step_len=20)
    template, bound = bind_model_dimensions(workspace, BASELINE_MODEL_CONFIG, env)
    assert bound == {**env, "num_features": str(count)}
    assert '"num_features": {{ num_features }},' == template
    assert "num_features" not in env
    workspace.file_dict[BASELINE_MODEL_CONFIG] = template
    assert bind_model_dimensions(workspace, BASELINE_MODEL_CONFIG, bound) == (template, bound)


@pytest.mark.parametrize("config", sorted(MODEL_CONFIGS - {BASELINE_MODEL_CONFIG}))
def test_combined_dimension_uses_deduplicated_parquet_columns_not_rows(tmp_path, config):
    workspace = _workspace(tmp_path)
    _factors(tmp_path, ["new_a", "new_b", "new_c"])
    env = {**_env(), "num_features": "27"}
    template, bound = bind_model_dimensions(workspace, config, env)
    assert bound == env
    assert template == workspace.file_dict[config]
    with pytest.raises(ValueError, match="declared 26, loaded features 27"):
        bind_model_dimensions(workspace, config, {**env, "num_features": 26})


@pytest.mark.parametrize("override", [
    {"feature_names": "[]"},
    {"feature_names": "['dup', 'dup']"},
    {"feature_names": "['one']"},
    {"feature_expressions": "['one']"},
    {"feature_names": "__import__('os').getcwd()"},
    {"num_features": 20},
    {"num_features": True},
])
def test_invalid_binding_is_rejected_before_execution(tmp_path, override):
    with pytest.raises(ValueError, match="fin_quant"):
        bind_model_dimensions(_workspace(tmp_path), BASELINE_MODEL_CONFIG, {**_env(), **override})


@pytest.mark.parametrize("value", ['"num_features": 21,', '"other": 20,',
                                   '"num_features": 20, "num_features": 20,'])
def test_unrecognized_template_is_not_silently_rewritten(tmp_path, value):
    workspace = _workspace(tmp_path)
    workspace.file_dict[BASELINE_MODEL_CONFIG] = value
    with pytest.raises(ValueError, match="template does not bind"):
        bind_model_dimensions(workspace, BASELINE_MODEL_CONFIG, _env())


def test_combined_features_must_not_overlap_baseline(tmp_path):
    _factors(tmp_path, ["base_0"])
    with pytest.raises(ValueError, match="overlap"):
        bind_model_dimensions(_workspace(tmp_path), "conf_sota_factors_model.yaml", _env())


def test_execution_adapter_binds_before_launch_and_preserves_other_experiments(
    monkeypatch, tmp_path,
):
    launched = []

    class Workspace:
        def __init__(self):
            self.__dict__.update(_workspace(tmp_path).__dict__)

        def inject_files(self, **files):
            self.file_dict.update(files)

        def execute(self, qlib_config_name="conf.yaml", run_env=None, *args, **kwargs):
            launched.append((qlib_config_name, run_env, args, kwargs))
            return "result", "log"

    module = ModuleType("rdagent.scenarios.qlib.experiment.workspace")
    module.QlibFBWorkspace = Workspace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    enable_fin_quant_model_dimensions()
    execute = Workspace.execute
    enable_fin_quant_model_dimensions()
    assert Workspace.execute is execute
    workspace = Workspace()
    env = _env()
    assert workspace.execute(BASELINE_MODEL_CONFIG, env, "extra", custom=True) == ("result", "log")
    assert launched == [(BASELINE_MODEL_CONFIG, {**env, "num_features": "24"},
                         ("extra",), {"custom": True})]
    with pytest.raises(ValueError, match="input mismatch"):
        workspace.execute(BASELINE_MODEL_CONFIG, {**env, "num_features": "20"})
    assert len(launched) == 1
    workspace.execute("conf_baseline.yaml", env)
    assert launched[-1][1] is env


def test_installed_upstream_baseline_template_initializes_24_feature_model(monkeypatch, tmp_path):
    rdagent = pytest.importorskip("rdagent")
    torch = pytest.importorskip("torch")
    jinja2 = pytest.importorskip("jinja2")
    import yaml

    root = Path(next(iter(rdagent.__path__)))
    template = (
        root / "scenarios/qlib/experiment/model_template" / BASELINE_MODEL_CONFIG).read_text()
    workspace = _workspace(tmp_path)
    workspace.file_dict[BASELINE_MODEL_CONFIG] = template
    env = {**_env(), "n_epochs": "1", "lr": "0.001", "early_stop": "1",
           "batch_size": "32", "weight_decay": "0.0001"}

    def training_step(source, context):
        config = yaml.safe_load(jinja2.Template(source).render(**context))
        dimension = config["task"]["model"]["kwargs"]["pt_model_kwargs"]["num_features"]
        features = config["data_handler_config"]["data_loader"]["kwargs"]["dataloader_l"][0][
            "kwargs"]["config"]["feature"]
        assert len(features[0]) == len(features[1]) == 24
        model = torch.nn.Sequential(torch.nn.LayerNorm(dimension), torch.nn.Linear(dimension, 1))
        x = torch.randn(32, len(features[0]))
        model(x).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    with pytest.raises(RuntimeError, match="normalized_shape"):
        training_step(template, env)
    patched, bound = bind_model_dimensions(workspace, BASELINE_MODEL_CONFIG, env)
    training_step(patched, bound)
