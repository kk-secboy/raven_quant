from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from qlib_test_doubles import qlib_workflow_identity

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
