from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("rdagent")
rdagent_runner = importlib.import_module("quant_platform.rdagent_runner")

pytestmark = pytest.mark.no_database


class _Workspace:
    def __init__(self, names: frozenset[str]) -> None:
        template = (
            "market: &market csi300\n"
            "benchmark: &benchmark SH000300\n"
            "instruments: *market\n"
        )
        self.file_dict = {name: template for name in names}
        self.injected: dict[str, str] = {}

    def inject_files(self, **files: str) -> None:
        self.injected.update(files)
        self.file_dict.update(files)


def _experiment(names: frozenset[str]) -> SimpleNamespace:
    return SimpleNamespace(experiment_workspace=_Workspace(names))


def _govern(
    monkeypatch: pytest.MonkeyPatch,
    experiment: SimpleNamespace,
    names: frozenset[str],
) -> None:
    monkeypatch.setattr(rdagent_runner, "QlibFBWorkspace", _Workspace)
    monkeypatch.setattr(
        rdagent_runner,
        "RD_AGENT_SETTINGS",
        SimpleNamespace(cache_with_pickle=False),
    )
    rdagent_runner._govern_experiment_market(experiment, names)


@pytest.mark.parametrize(
    "names",
    [rdagent_runner._FACTOR_CONFIGS, rdagent_runner._MODEL_CONFIGS],
)
def test_runner_rewrites_only_the_market_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    names: frozenset[str],
) -> None:
    experiment = _experiment(names)

    _govern(monkeypatch, experiment, names)
    _govern(monkeypatch, experiment, names)

    workspace = experiment.experiment_workspace
    assert set(workspace.injected) == set(names)
    for source in workspace.file_dict.values():
        assert source.count("market: &market cn_all") == 1
        assert "csi300" not in source.lower()
        assert source.count("benchmark: &benchmark SH000300") == 1
        assert source.count("instruments: *market") == 1


def test_runner_fails_closed_on_template_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = rdagent_runner._FACTOR_CONFIGS
    experiment = _experiment(names)
    experiment.experiment_workspace.file_dict.pop("conf_baseline.yaml")

    with pytest.raises(RuntimeError, match="template set drifted"):
        _govern(monkeypatch, experiment, names)


def test_runner_rejects_unreviewed_csi300_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = rdagent_runner._MODEL_CONFIGS
    experiment = _experiment(names)
    target = "conf_sota_factors_model.yaml"
    experiment.experiment_workspace.file_dict[target] += "# CSI300 hidden override\n"

    with pytest.raises(RuntimeError, match="still selects CSI300"):
        _govern(monkeypatch, experiment, names)


def test_runner_rejects_pickle_cache_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = rdagent_runner._FACTOR_CONFIGS
    experiment = _experiment(names)
    monkeypatch.setattr(rdagent_runner, "QlibFBWorkspace", _Workspace)
    monkeypatch.setattr(
        rdagent_runner,
        "RD_AGENT_SETTINGS",
        SimpleNamespace(cache_with_pickle=True),
    )

    with pytest.raises(RuntimeError, match="pickle cache disabled"):
        rdagent_runner._govern_experiment_market(experiment, names)
