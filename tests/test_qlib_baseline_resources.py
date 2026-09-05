from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from quant_platform.job_commands import evaluation
from scripts import run_qlib_baseline

pytestmark = pytest.mark.no_database


@pytest.mark.parametrize("configured, expected", [(None, "1"), ("3", "3")])
@pytest.mark.parametrize("is_wsl", [False, True])
def test_baseline_command_forwards_governed_kernel_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured, expected, is_wsl
) -> None:
    if configured is None:
        monkeypatch.delenv("QUANTLAB_QLIB_KERNELS", raising=False)
    else:
        monkeypatch.setenv("QUANTLAB_QLIB_KERNELS", configured)
    monkeypatch.setattr(evaluation, "os", SimpleNamespace(name="nt", environ=os.environ))
    monkeypatch.setattr(evaluation, "_to_wsl_path", lambda path: f"wsl:{path.as_posix()}")
    settings = SimpleNamespace(
        data_root=tmp_path / "data",
        qlib_python="/venv/bin/python" if is_wsl else "python.exe",
        qlib_wsl_distro="Ubuntu-22.04",
        mlflow_tracking_uri="postgresql://tracking",
    )
    worker = SimpleNamespace(settings=settings, project_root=tmp_path)
    payload = {
        "dataset_path": str(tmp_path / "dataset"),
        "market": "cn_all",
        "benchmark": "SH000300",
        "account": 5_000_000,
        "topk": 50,
        "n_drop": 5,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5.0,
        # Job payloads must not override the deployment resource limit.
        "num_kernels": 64,
    }

    command, result_path, _ = evaluation.qlib_baseline_command(
        worker, {"id": "baseline-a", "payload": payload}
    )

    assert command[command.index("--num-kernels") + 1] == expected
    assert command.count("--num-kernels") == 1
    assert (command[0] == "wsl") is is_wsl
    assert command[command.index("--market") + 1] == "cn_all"
    assert result_path == settings.data_root / "artifacts/qlib/baseline-a/result.json"


@pytest.mark.parametrize(
    "configured, cli, expected",
    [(None, [], 1), ("3", [], 3), ("3", ["--num-kernels", "2"], 2)],
)
def test_baseline_cli_applies_bounded_kernels_to_qlib_init(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured, cli, expected
) -> None:
    if configured is None:
        monkeypatch.delenv("QUANTLAB_QLIB_KERNELS", raising=False)
    else:
        monkeypatch.setenv("QUANTLAB_QLIB_KERNELS", configured)
    provider = tmp_path / "dataset"
    (provider / "metadata").mkdir(parents=True)
    (provider / "metadata/provenance.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qlib_baseline.py",
            "--provider-uri", str(provider),
            "--output", str(tmp_path / "output"),
            "--tracking-uri", "postgresql://tracking",
            *cli,
        ],
    )
    verify = Mock()
    monkeypatch.setattr(run_qlib_baseline, "verify_qlib_output_manifest", verify)

    class Initialized(Exception):
        pass

    init = Mock(side_effect=Initialized)
    modules = {
        "qlib": {"init": init},
        "qlib.constant": {"REG_CN": "cn"},
        "qlib.contrib.data.handler": {"Alpha158": object},
        "qlib.contrib.evaluate": {"backtest_daily": object, "risk_analysis": object},
        "qlib.contrib.model.gbdt": {"LGBModel": object},
        "qlib.contrib.strategy": {"TopkDropoutStrategy": object},
        "qlib.data": {"D": object},
        "qlib.data.dataset": {"DatasetH": object},
        "quant_platform.qlib_baseline_dataset": {"BaselineDataset": object},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.raises(Initialized):
        run_qlib_baseline.main()

    verify.assert_called_once_with(provider.resolve(), {})
    init.assert_called_once_with(
        provider_uri=str(provider.resolve()), region="cn", kernels=expected
    )


@pytest.mark.parametrize("value", ["0", "-1", "65", "all"])
@pytest.mark.parametrize("source", ["cli", "environment"])
def test_baseline_rejects_invalid_kernel_limits_before_initialization(
    monkeypatch: pytest.MonkeyPatch, value, source
) -> None:
    monkeypatch.delenv("QUANTLAB_QLIB_KERNELS", raising=False)
    arguments = ["run_qlib_baseline.py"]
    if source == "environment":
        monkeypatch.setenv("QUANTLAB_QLIB_KERNELS", value)
    else:
        arguments.extend(["--num-kernels", value])
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as exc:
        run_qlib_baseline.parse_args()

    assert exc.value.code == 2
