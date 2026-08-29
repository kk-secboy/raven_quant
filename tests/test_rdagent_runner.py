from __future__ import annotations

import importlib
import pickle
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
            "train: [{{ train_start }}, {{ train_end }}]\n"
            "valid: [{{ valid_start }}, {{ valid_end }}]\n"
            "test: [{{ test_start }}, {{ test_end }}]\n"
        )
        self.file_dict = {name: template for name in names}
        self.injected: dict[str, str] = {}

    def inject_files(self, **files: str) -> None:
        self.injected.update(files)
        self.file_dict.update(files)

    def execute(self, qlib_config_name="conf.yaml", run_env=None, *args, **kwargs):
        return qlib_config_name, run_env


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


def test_execute_guard_rejects_effective_period_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = rdagent_runner._FACTOR_CONFIGS
    workspace = object.__new__(rdagent_runner.QlibFBWorkspace)
    workspace.file_dict = _Workspace(names).file_dict
    periods = {
        "train_start": "2010-01-01",
        "train_end": "2018-12-31",
        "valid_start": "2019-01-01",
        "valid_end": "2020-12-31",
        "test_start": "2021-01-01",
        "test_end": "2022-12-31",
    }
    for prefix in ("QLIB_FACTOR", "QLIB_MODEL", "QLIB_QUANT"):
        for key, value in periods.items():
            monkeypatch.setenv(f"{prefix}_{key.upper()}", value)
    monkeypatch.setattr(
        rdagent_runner.QlibFBWorkspace,
        "execute",
        lambda _self, qlib_config_name="conf.yaml", run_env=None, *args, **kwargs: (
            qlib_config_name,
            run_env,
        ),
    )
    rdagent_runner._install_execute_guard(
        workspace,
        prefix="QLIB_FACTOR",
        expected_configs=names,
    )
    pickle.dumps(workspace)

    assert workspace.execute("conf_baseline.yaml", dict(periods)) == (
        "conf_baseline.yaml",
        periods,
    )
    with pytest.raises(RuntimeError, match="effective dates disagree"):
        workspace.execute(
            "conf_baseline.yaml",
            {**periods, "test_end": "2026-12-31"},
        )
    monkeypatch.setenv("QLIB_QUANT_TEST_END", "2023-12-31")
    with pytest.raises(RuntimeError, match="disagree with QLIB_QUANT"):
        workspace.execute("conf_baseline.yaml", dict(periods))


def test_runner_rejects_template_without_governed_date_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = rdagent_runner._MODEL_CONFIGS
    experiment = _experiment(names)
    target = "conf_sota_factors_model.yaml"
    experiment.experiment_workspace.file_dict[target] = experiment.experiment_workspace.file_dict[
        target
    ].replace("{{ test_end }}", "2026-12-31")

    with pytest.raises(RuntimeError, match="omits governed periods"):
        _govern(monkeypatch, experiment, names)
