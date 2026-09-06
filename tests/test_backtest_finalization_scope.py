from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from test_qlib_workflow import _FakeRecorderApi

from quant_platform import qlib_workflow
from quant_platform.strategy_artifact_manifest import validate_backtest_artifact_manifest
from quant_platform.strategy_health_reference import validate_strategy_health_reference
from quant_platform.strategy_research_evaluation import STRATEGY_RESEARCH_EVALUATION_MODES

pytestmark = pytest.mark.no_database
ROOT = Path(__file__).parents[1]
PRE_FINAL_MODES = ("pre_final_portfolio_trial", *sorted(STRATEGY_RESEARCH_EVALUATION_MODES))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _inputs(output, manifest):
    """Only deterministic synthetic data, representing already-computed output."""
    output.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2024-01-02", periods=30)
    index = pd.MultiIndex.from_product(
        [dates, [f"SH{600000 + i:06d}" for i in range(40)]], names=["datetime", "instrument"],
    )
    values = pd.DataFrame({"fixture_factor": np.linspace(-2, 2, len(index))}, index=index)
    raw = output / "fixture_factor.parquet"
    values.to_parquet(raw)
    for name in ("daily_returns", "score_grid", "governed_signal", "qlib_portfolio_report",
                 "execution_fills"):
        pd.DataFrame({"synthetic": [0.0]}).to_parquet(output / f"{name}.parquet")
    (output / "qlib_positions.pkl").write_bytes(pickle.dumps({"synthetic": []}))
    for name in ("execution_model", "robustness", "rolling", "event_stress", "capacity_curve",
                 "formal_validation"):
        (output / f"{name}.json").write_text('{"synthetic":true}', encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mode = manifest["evaluation_mode"]
    pre_final = mode in PRE_FINAL_MODES
    replay = mode == "consumed_historical_replay"
    metrics = {
        "backtest_engine": "qlib", "qlib_native_backtest": True, "fixture_metric": 1.25,
        "formal_validation": {
            "status": "not_applicable_pre_final_only" if pre_final else "failed",
            "capital_eligible": False,
        },
        "provenance": {
            "strategy_config_sha256": _load_script("run_parameter_experiment")
            ._canonical_sha256(manifest["config"]),
            "evaluation_mode": mode,
            "evaluation_scope": (
                "pre_final_only" if pre_final
                else "historical_description_only" if replay else "final_oos_once"
            ),
            "final_oos_opened": not pre_final,
        },
    }
    return {
        "output": output, "manifest": manifest, "manifest_path": manifest_path,
        "provider_provenance": {"dataset_identity_sha256": "a" * 64,
                                "dataset_lineage_id": "b" * 64},
        "periods": manifest["periods"], "metrics": metrics, "evaluation_mode": mode,
        "signal_source": "factor_score", "execution_method": "open", "execution_frequency": "day",
        "historical_window_opened": mode not in PRE_FINAL_MODES,
        "tracking_uri": (output.parent / "synthetic-tracking").as_uri(),
        "health_reference_inputs": {
            "factor_source_mode": "qlib_baseline",
            "baseline_definition": {"factors": [{"id": "fixture_factor"}]},
            "baseline_raw": values,
            "baseline_artifacts": {"raw": {"fixture_factor": {
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}}},
            "challenger_entries": [], "formal_factor_items": [], "model_feature_set": None,
            "model_predictions": None, "model_label_contract": None, "qlib_data_api": None,
        },
    }


def _segment_manifest(mode):
    config = {"evidence_mode": "sealed_final_oos", "challenger_weight": 0.0,
              "strategy_rules_sha256": "c" * 64}
    return _load_script("run_parameter_experiment")._build_segment_manifest(
        base_manifest={"strategy_version_id": "synthetic-version", "dataset": "synthetic-dataset",
                       "benchmark": "SH000300", "factors": [], "evaluation_mode": mode},
        config=config, periods={"start": "2024-01-02", "end": "2024-02-12"},
    )


def _finish(runner, kwargs):
    fake = _FakeRecorderApi()
    with (
        patch.dict("os.environ", {"_MLFLOW_SERVER_ARTIFACT_ROOT": str(kwargs["output"].parent)}),
        patch.object(qlib_workflow, "_load_qlib_recorder", return_value=fake),
        patch.object(qlib_workflow, "upstream_runtime_identity", return_value={
            "version": "synthetic-test-qlib", "commit": qlib_workflow.QLIB_COMMIT}),
    ):
        result = runner._finalize_backtest_output(**kwargs)
    assert fake.metrics == {"fixture_metric": 1.25}
    assert fake.saved == [(str(kwargs["output"].resolve()), "production")]
    assert fake.exit_exception is None
    return result, fake


@pytest.mark.parametrize("mode", PRE_FINAL_MODES)
def test_real_prefinal_segment_completes_tracking_artifact_seal_and_result_without_formal_health(
    tmp_path, mode,
):
    runner = _load_script("run_multifactor_backtest")
    manifest = _segment_manifest(mode)
    assert "backtest_id" not in manifest and "strategy_rules_sha256" not in manifest
    kwargs = _inputs(tmp_path / "segment", manifest)
    with patch.object(runner, "_write_strategy_health_reference",
                      side_effect=AssertionError("pre-final cannot publish a formal reference")):
        result, fake = _finish(runner, kwargs)
    assert result["status"] == "ok" and result["final_oos_opened"] is False
    assert result["metrics"]["formal_validation"]["capital_eligible"] is False
    assert "strategy_health_reference" not in result["artifacts"]
    assert "strategy_health_reference" not in result["metrics"]["provenance"]
    assert not (kwargs["output"] / runner.STRATEGY_HEALTH_REFERENCE_NAME).exists()
    assert fake.params["run_kind"] == "portfolio-experiment-trial"
    assert fake.params["backtest_id"].startswith("synthetic-version-")
    assert "backtest_id" not in json.loads(Path(kwargs["manifest_path"]).read_text())
    sealed = validate_backtest_artifact_manifest(
        kwargs["output"],
        expected_sha256=result["metrics"]["provenance"]["artifact_manifest_sha256"],
    )
    assert runner.STRATEGY_HEALTH_REFERENCE_NAME not in {row["path"] for row in sealed["files"]}
    loaded = _load_script("run_parameter_experiment")._read_completed_result(
        kwargs["output"] / "result.json", config=manifest["config"],
        periods=manifest["periods"], evaluation_mode=mode,
    )
    assert loaded == result


@pytest.mark.parametrize("mode", ["formal_final_oos", "consumed_historical_replay"])
def test_formal_and_replay_still_write_valid_bound_health_reference_before_sealing(tmp_path, mode):
    runner = _load_script("run_multifactor_backtest")
    manifest = _segment_manifest(mode)
    manifest.update(backtest_id="synthetic-formal-backtest", strategy_rules_sha256="c" * 64)
    kwargs = _inputs(tmp_path / "formal", manifest)
    result, _fake = _finish(runner, kwargs)
    receipt = result["metrics"]["provenance"]["strategy_health_reference"]
    reference_path = Path(result["artifacts"]["strategy_health_reference"])
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == receipt["sha256"]
    validate_strategy_health_reference(
        json.loads(reference_path.read_text()), strategy_version_id=manifest["strategy_version_id"],
        formal_backtest_id=manifest["backtest_id"], formal_dataset_identity_sha256="a" * 64,
        formal_dataset_lineage_id="b" * 64, strategy_rules_sha256="c" * 64,
        expected_factor_ids={"fixture_factor"}, signal_source="factor_score",
    )
    sealed = validate_backtest_artifact_manifest(
        kwargs["output"],
        expected_sha256=result["metrics"]["provenance"]["artifact_manifest_sha256"],
    )
    assert runner.STRATEGY_HEALTH_REFERENCE_NAME in {row["path"] for row in sealed["files"]}


@pytest.mark.parametrize("mode", ["formal_final_oos", "consumed_historical_replay"])
@pytest.mark.parametrize("missing", ["backtest_id", "strategy_rules_sha256"])
def test_formal_modes_still_reject_missing_top_level_identity_without_defaults(
    tmp_path, mode, missing,
):
    runner = _load_script("run_multifactor_backtest")
    manifest = _segment_manifest(mode)
    manifest.update(backtest_id="synthetic-formal-backtest", strategy_rules_sha256="c" * 64)
    del manifest[missing]
    kwargs = _inputs(tmp_path / "formal-missing", manifest)
    with pytest.raises(KeyError, match=missing):
        _finish(runner, kwargs)
    assert not (kwargs["output"] / "result.json").exists()
    assert not (kwargs["output"] / runner.STRATEGY_HEALTH_REFERENCE_NAME).exists()


@pytest.mark.parametrize("mode", PRE_FINAL_MODES)
@pytest.mark.parametrize("entry_kind", ["file", "dangling_symlink"])
def test_prefinal_rejects_existing_health_reference_without_changing_it(
    tmp_path, mode, entry_kind,
):
    runner = _load_script("run_multifactor_backtest")
    kwargs = _inputs(tmp_path / "stale", _segment_manifest(mode))
    reference = kwargs["output"] / runner.STRATEGY_HEALTH_REFERENCE_NAME
    if entry_kind == "file":
        reference.write_bytes(b"historical reference retained\n")
    else:
        try:
            reference.symlink_to("missing-historical-reference.json")
        except OSError:
            pytest.skip("host does not permit unprivileged symlinks")
    with pytest.raises(ValueError, match="pre-final output must not contain"):
        _finish(runner, kwargs)
    assert not (kwargs["output"] / "result.json").exists()
    if entry_kind == "file":
        assert reference.read_bytes() == b"historical reference retained\n"
    else:
        assert reference.is_symlink()


@pytest.mark.parametrize("mode", PRE_FINAL_MODES)
def test_main_rejects_stale_prefinal_reference_before_reading_provider(tmp_path, mode):
    runner = _load_script("run_multifactor_backtest")
    output = tmp_path / "stale"
    output.mkdir()
    reference = output / runner.STRATEGY_HEALTH_REFERENCE_NAME
    reference.write_bytes(b"historical reference retained\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_segment_manifest(mode)))
    with (
        patch.object(sys, "argv", [
            "backtest", "--provider-uri", str(tmp_path / "absent-provider"),
            "--manifest", str(manifest), "--output", str(output),
            "--tracking-uri", "synthetic-unused",
        ]),
        pytest.raises(ValueError, match="pre-final output must not contain"),
    ):
        runner.main()
    assert reference.read_bytes() == b"historical reference retained\n"


@pytest.mark.parametrize("mode", PRE_FINAL_MODES)
@pytest.mark.parametrize("location", ["file", "artifacts", "provenance"])
def test_prefinal_reuse_rejects_formal_health_reference(tmp_path, mode, location):
    runner = _load_script("run_multifactor_backtest")
    manifest = _segment_manifest(mode)
    kwargs = _inputs(tmp_path / "reuse", manifest)
    result, _fake = _finish(runner, kwargs)
    if location == "file":
        (kwargs["output"] / runner.STRATEGY_HEALTH_REFERENCE_NAME).write_bytes(b"old reference")
    else:
        container = (
            result["artifacts"] if location == "artifacts" else result["metrics"]["provenance"]
        )
        container["strategy_health_reference"] = None
    result_path = kwargs["output"] / "result.json"
    result_path.write_text(json.dumps(result))
    assert _load_script("run_parameter_experiment")._read_completed_result(
        result_path, config=manifest["config"], periods=manifest["periods"], evaluation_mode=mode,
    ) is None


def _child_main():
    """Actual subprocess fixture: finalize, then append stdout after output sealing."""
    manifest_path = Path(sys.argv[sys.argv.index("--manifest") + 1])
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    runner = _load_script("run_multifactor_backtest")
    kwargs = _inputs(output, json.loads(manifest_path.read_text()))
    result, _fake = _finish(runner, kwargs)
    print(json.dumps(result))
    print("synthetic stdout after artifact sealing")


@pytest.mark.parametrize("segment", ["in_sample", "validation"])
def test_real_child_stdout_after_seal_is_external_and_old_in_tree_log_is_preserved(
    tmp_path, segment,
):
    parameter = _load_script("run_parameter_experiment")
    child = tmp_path / "synthetic_backtest.py"
    child.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'tests')!r})\n"
        "from test_backtest_finalization_scope import _child_main\n_child_main()\n",
        encoding="utf-8",
    )
    output = tmp_path / "trial" / segment
    output.mkdir(parents=True)
    old_log = output / "backtest.log"
    old_log.write_bytes(b"historical log retained unchanged\n")
    old_bytes = old_log.read_bytes()
    manifest = _segment_manifest("strategy_policy_only_pre_final")
    result = parameter._run_segment(
        backtest_script=child, provider_uri="synthetic-unused", execution_provider_uri=None,
        execution_frequency=None, base_manifest=manifest, config=manifest["config"],
        periods=manifest["periods"], output=output, tracking_uri="synthetic-unused",
    )
    current_log = output.parent / f"{segment}-backtest.log"
    assert "synthetic stdout after artifact sealing" in current_log.read_text()
    assert old_log.read_bytes() == old_bytes
    expected = result["metrics"]["provenance"]["artifact_manifest_sha256"]
    validate_backtest_artifact_manifest(output, expected_sha256=expected)
    with current_log.open("a") as stream:
        stream.write("more synthetic stdout after child exit\n")
    validate_backtest_artifact_manifest(output, expected_sha256=expected)
