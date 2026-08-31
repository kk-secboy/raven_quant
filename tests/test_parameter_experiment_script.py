from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from qlib_test_doubles import qlib_workflow_identity

from quant_platform.parameter_experiment_store import _terminal_result_state

pytestmark = pytest.mark.no_database


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "run_parameter_experiment.py"
    spec = importlib.util.spec_from_file_location("run_parameter_experiment_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_segment_is_reused_only_when_period_and_config_match(
    tmp_path: Path,
) -> None:
    script = _script_module()
    config = {"topk": 50, "max_daily_turnover": 0.2}
    periods = {"start": "2024-01-01", "end": "2024-12-31"}
    result = {
        "periods": periods,
        "qlib_workflow": qlib_workflow_identity(),
        "metrics": {
            "backtest_engine": "qlib",
            "qlib_native_backtest": True,
            "provenance": {"strategy_config_sha256": script._canonical_sha256(config)},
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert script._read_completed_result(
        result_path, config=config, periods=periods
    ) == result
    assert script._read_completed_result(
        result_path, config={**config, "topk": 80}, periods=periods
    ) is None
    assert script._read_completed_result(
        result_path,
        config=config,
        periods={"start": "2025-01-01", "end": "2025-12-31"},
    ) is None


def test_child_backtest_manifest_preserves_transparent_runtime_identity() -> None:
    script = _script_module()
    identity = {
        "transparent_baseline_runner_sha256": "a" * 64,
        "transparent_baseline_runtime_bundle_sha256": "b" * 64,
        "transparent_baseline_worker_runtime_image_digest": "sha256:" + "c" * 64,
    }
    base = {
        "strategy_version_id": "version-id",
        "dataset": "dataset-id",
        "benchmark": "SH000300",
        "universe": "cn_all",
        "execution_dataset": None,
        "evaluation_mode": "strategy_policy_only_pre_final",
        "pre_final_cutoff": "2024-12-31",
        "historical_validation_periods": {
            "start": "2015-01-01",
            "end": "2021-12-31",
        },
        "strategy_trial_count": 2,
        "shared_multiple_testing": None,
        "model_signal": None,
        "model_formal_admission": None,
        "model_candidate": None,
        "model_bundle_factors": [],
        "factors": [],
        **identity,
    }

    child = script._build_segment_manifest(
        base_manifest=base,
        config={"recipe_id": "short_relative_strength"},
        periods={"start": "2022-01-01", "end": "2024-12-31"},
    )

    assert {key: child[key] for key in identity} == identity


def test_cross_trial_dsr_rewrites_trial_metrics_and_progress() -> None:
    script = _script_module()
    returns = {
        0: pd.Series([0.001 + (index % 5 - 2) * 0.0002 for index in range(60)]),
        1: pd.Series([0.0005 + (index % 7 - 3) * 0.0003 for index in range(60)]),
    }
    trials = []
    for index in range(2):
        provisional = script.deflated_sharpe_probability(
            returns[index], trials=2
        )
        trials.append(
            {
                "trial_index": index,
                "parameters": {"topk": 20 + index},
                "status": "succeeded",
                "score": 0.0,
                "warnings": ["deflated_sharpe_failed"],
                "error": None,
                "metrics": {
                    "in_sample": {},
                    "out_of_sample": {
                        "deflated_sharpe": provisional,
                        "deflated_sharpe_probability": provisional["probability"],
                    },
                },
            }
        )

    assert script._finalize_cross_trial_dsr(
        trials, returns, trial_count=2
    )
    progress = script._progress_payload(trials, trial_count=2)

    assert progress["trials"][0]["warnings"] == trials[0]["warnings"]
    for trial in trials:
        evidence = trial["metrics"]["out_of_sample"]["deflated_sharpe"]
        assert evidence["status"] == "ok"
        assert evidence["method_version"] == (
            "bailey-lopez-de-prado-cross-trial-v2"
        )
        assert evidence["trial_sharpe_std"] > 0


def test_cross_trial_dsr_counts_prior_admitted_model_trials() -> None:
    script = _script_module()
    returns = {
        0: pd.Series([0.001 + (index % 5 - 2) * 0.0002 for index in range(60)]),
        1: pd.Series([0.0005 + (index % 7 - 3) * 0.0003 for index in range(60)]),
    }
    trials = []
    for index in range(2):
        provisional = script.deflated_sharpe_probability(returns[index], trials=4)
        trials.append(
            {
                "trial_index": index,
                "parameters": {
                    "portfolio_construction": (
                        "topk_equal_weight" if index == 0 else "industry_neutral_qp"
                    )
                },
                "status": "succeeded",
                "score": 0.0,
                "warnings": [],
                "error": None,
                "metrics": {
                    "in_sample": {},
                    "out_of_sample": {
                        "deflated_sharpe": provisional,
                        "deflated_sharpe_probability": provisional["probability"],
                    },
                },
            }
        )

    assert script._finalize_cross_trial_dsr(
        trials,
        returns,
        trial_count=2,
        prior_trial_sharpes=[0.02, 0.03],
    )
    for trial in trials:
        evidence = trial["metrics"]["out_of_sample"]["deflated_sharpe"]
        assert evidence["trials"] == 4
        assert evidence["trial_sharpe_std"] > 0


def test_admitted_trial_distribution_must_be_complete_and_pre_final() -> None:
    script = _script_module()
    evidence = {
        "final_oos_opened": False,
        "trial_count": 2,
        "trial_names": ["model-a", "model-b"],
        "trial_daily_sharpes": [0.01, 0.02],
    }
    assert script._admitted_trial_sharpes({"shared_multiple_testing": evidence}) == [
        0.01,
        0.02,
    ]
    with pytest.raises(ValueError, match="incomplete"):
        script._admitted_trial_sharpes(
            {
                "shared_multiple_testing": {
                    **evidence,
                    "final_oos_opened": True,
                }
            }
        )


def test_portfolio_trials_require_identical_predictions_and_costs() -> None:
    script = _script_module()
    base_config = {
        "signal_frequency": "day",
        "signal_period": 1,
        "rebalance_frequency": "day",
        "execution_frequency": "day",
        "execution_method": "open",
        "execution_days": 1,
        "execution_lag_bars": 1,
        "max_volume_participation": 0.1,
    }
    manifest = {
        "evaluation_mode": "pre_final_portfolio_trial",
        "trials": [
            {
                "trial_index": index,
                "config": {**base_config, "portfolio_construction": construction},
            }
            for index, construction in enumerate(
                ("topk_equal_weight", "industry_neutral_qp")
            )
        ],
    }
    results = []
    for index, construction in enumerate(
        ("topk_equal_weight", "industry_neutral_qp")
    ):
        results.append(
            {
                "trial_index": index,
                "parameters": {"portfolio_construction": construction},
                "status": "succeeded",
                "metrics": {
                    segment: {
                        "cost_model": {"version": "cost-v1", "stamp_tax": 0.001},
                        "provenance": {
                            "evaluation_mode": "pre_final_portfolio_trial",
                            "evaluation_scope": "pre_final_only",
                            "final_oos_opened": False,
                            "formal_model_predictions_sha256": "a" * 64,
                            "formal_model_checkpoint_sha256": "b" * 64,
                            "model_signal_identity_sha256": "c" * 64,
                            "dataset_identity_sha256": "d" * 64,
                            "formal_model_admission_binding_sha256": "e" * 64,
                            "pre_final_cutoff": "2025-12-31",
                        },
                    }
                    for segment in ("in_sample", "out_of_sample")
                },
            }
        )

    evidence = script._validate_portfolio_trial_comparability(manifest, results)
    assert evidence is not None
    assert evidence["segments"]["out_of_sample"][
        "model_predictions_sha256"
    ] == ("a" * 64)

    results[1]["metrics"]["out_of_sample"]["provenance"][
        "formal_model_predictions_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="identical model predictions"):
        script._validate_portfolio_trial_comparability(manifest, results)


def test_partial_trial_execution_failure_builds_a_complete_failed_result() -> None:
    script = _script_module()
    trials = [
        {
            "trial_index": 0,
            "parameters": {"topk": 20},
            "status": "succeeded",
            "score": 0.2,
            "metrics": {"in_sample": {}, "out_of_sample": {}},
            "warnings": ["deflated_sharpe_failed"],
            "error": None,
        },
        {
            "trial_index": 1,
            "parameters": {"topk": 30},
            "status": "failed",
            "score": None,
            "metrics": None,
            "warnings": [],
            "error": "child backtest exited 2: missing price-limit field",
        },
    ]
    summary = {
        "trial_count": 2,
        "succeeded_count": 0,
        "failed_count": 2,
        "statistically_rejected_count": 1,
    }
    result = script._build_terminal_result(
        manifest={
            "experiment_id": "experiment-1",
            "strategy_version_id": "version-1",
            "dataset": "dataset-1",
            "periods": {"in_sample": {}, "out_of_sample": {}},
        },
        evaluation_mode="strategy_policy_only_pre_final",
        trial_results=trials,
        summary=summary,
    )

    assert result["status"] == "failed"
    assert result["failure_kind"] == "trial_execution_error"
    assert "missing price-limit field" in result["error"]
    assert result["trials"] == trials
    assert result["summary"]["execution_succeeded_count"] == 1
    assert result["summary"]["execution_failed_count"] == 1
    assert _terminal_result_state(result, trials, result["summary"]) == (
        "failed",
        result["error"],
    )


def test_statistical_rejection_is_not_reported_as_an_execution_failure() -> None:
    script = _script_module()
    trials = [
        {
            "trial_index": 0,
            "parameters": {"topk": 20},
            "status": "succeeded",
            "score": -0.1,
            "metrics": {"in_sample": {}, "out_of_sample": {}},
            "warnings": ["deflated_sharpe_failed"],
            "error": None,
        }
    ]
    result = script._build_terminal_result(
        manifest={
            "experiment_id": "experiment-2",
            "strategy_version_id": "version-2",
            "dataset": "dataset-2",
            "periods": {"in_sample": {}, "out_of_sample": {}},
        },
        evaluation_mode="strategy_policy_only_pre_final",
        trial_results=trials,
        summary={
            "trial_count": 1,
            "succeeded_count": 0,
            "failed_count": 1,
            "statistically_rejected_count": 1,
            "warnings": ["all_trials_failed_deflated_sharpe"],
        },
    )

    assert result["status"] == "ok"
    assert "failure_kind" not in result
    assert "error" not in result
    assert result["summary"]["execution_succeeded_count"] == 1
    assert result["summary"]["execution_failed_count"] == 0
    assert _terminal_result_state(result, trials, result["summary"]) == (
        "succeeded",
        None,
    )


def test_failed_terminal_result_requires_each_real_trial_error() -> None:
    trials = [
        {
            "trial_index": 0,
            "status": "failed",
            "error": None,
        }
    ]
    with pytest.raises(ValueError, match="retain their execution errors"):
        _terminal_result_state(
            {
                "status": "failed",
                "failure_kind": "trial_execution_error",
                "error": "trial failed",
            },
            trials,
            {"execution_succeeded_count": 0, "execution_failed_count": 1},
        )


def test_main_writes_failed_trial_ledger_before_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script_module()
    provider = tmp_path / "provider"
    metadata = provider / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "provenance.json").write_text(
        json.dumps({"dataset_identity_sha256": "a" * 64}), encoding="utf-8"
    )
    manifest = {
        "experiment_id": "experiment-failed",
        "strategy_version_id": "version-failed",
        "dataset": "dataset-failed",
        "benchmark": "SH000300",
        "periods": {
            "in_sample": {"start": "2022-01-01", "end": "2022-12-31"},
            "out_of_sample": {"start": "2023-01-01", "end": "2023-12-31"},
        },
        "parameter_grid": {"topk": [20]},
        "trials": [
            {
                "trial_index": 0,
                "parameters": {"topk": 20},
                "config": {"topk": 20},
            }
        ],
        "factors": [],
        "strategy_trial_count": 1,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "output"

    class Workflow:
        def log_params(self, _values: dict) -> None:
            pass

        def log_metrics(self, _values: dict) -> None:
            pass

        def identity_dict(self) -> dict:
            return qlib_workflow_identity()

        def save_artifacts(self, _path: Path) -> None:
            pass

    @contextmanager
    def workflow_run(**_kwargs):
        yield Workflow()

    monkeypatch.setattr(script, "verify_qlib_output_manifest", lambda *_args: None)
    monkeypatch.setattr(
        script,
        "_run_segment",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("child backtest error: price-limit column missing")
        ),
    )
    monkeypatch.setattr(script, "qlib_workflow_run", workflow_run)
    monkeypatch.setitem(
        script.sys.modules,
        "qlib",
        SimpleNamespace(init=lambda **_kwargs: None),
    )
    monkeypatch.setattr(
        script.sys,
        "argv",
        [
            "run_parameter_experiment.py",
            "--provider-uri",
            str(provider),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--tracking-uri",
            str(tmp_path / "mlruns"),
        ],
    )

    with pytest.raises(RuntimeError, match="price-limit column missing"):
        script.main()

    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["failure_kind"] == "trial_execution_error"
    assert result["trials"][0]["status"] == "failed"
    assert result["trials"][0]["error"] == (
        "child backtest error: price-limit column missing"
    )
    assert result["summary"]["execution_failed_count"] == 1
