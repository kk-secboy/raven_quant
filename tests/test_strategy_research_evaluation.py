from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_data.execution_contract import strategy_execution_contract_hash
from quant_platform.cost_model import CostModelConfig
from quant_platform.parameter_experiment_store import ParameterExperimentStore
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.strategy_research_evaluation import (
    STRATEGY_FULL_STACK_MODE,
    STRATEGY_POLICY_ONLY_MODE,
    build_strategy_research_competition_plan,
    build_strategy_stage_artifact_from_parameter_experiment,
    build_strategy_stage_evidence,
    strategy_score_grid_contract,
)
from quant_platform.strategy_rule_compiler import compile_strategy_rule_policy
from quant_platform.strategy_rule_ir import validate_strategy_rule_ir
from quant_platform.transparent_baseline_runner import (
    STRATEGY_RESEARCH_TARGET_RUNNER_SHA256,
    STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    WORKER_RUNTIME_IMAGE_DIGEST_ENV,
)

_WORKER_IMAGE_DIGEST = "sha256:" + "e" * 64


@pytest.fixture(autouse=True)
def _sealed_worker_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKER_RUNTIME_IMAGE_DIGEST_ENV, _WORKER_IMAGE_DIGEST)


def _config() -> dict:
    recipe = get_strategy_recipe("short_relative_strength")
    config = deepcopy(recipe["config_overrides"])
    config.update(
        {
            "recipe_id": recipe["id"],
            "recipe_version": recipe["version"],
            "baseline_definition_sha256": "b" * 64,
            "strategy_evaluation_contract": {
                "minimum_oos_observations": 252,
            },
            "transparent_baseline_bootstrap": {
                TRANSPARENT_BASELINE_RUNNER_FIELD: (
                    STRATEGY_RESEARCH_TARGET_RUNNER_SHA256
                ),
                TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD: (
                    STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256
                ),
                TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD: (
                    _WORKER_IMAGE_DIGEST
                ),
            },
        }
    )
    costs = CostModelConfig().to_dict()
    costs["cost_schedule_version"] = costs.pop("version")
    config.update(costs)
    config["execution_contract_hash"] = strategy_execution_contract_hash(config)
    return config


def _candidate() -> dict:
    config = _config()
    config["source_research_artifact_id"] = "compiled-artifact"
    rule_ir = deepcopy(config["strategy_rule_ir"])
    slots = rule_ir["slots"]
    weights = slots["alpha_rank"]["components"][0]["parameters"]["weights"]
    weights["relative_strength_5d"] = 0.30
    weights["amount_expansion_5d"] = 0.30
    for component in slots["entry_timing"]["components"]:
        if component["component"] == "score_threshold":
            component["parameters"]["minimum_percentile"] = 0.85
    rules = validate_strategy_rule_ir(
        "short_1_5d", rule_ir["slots"], allowed_factor_ids=set(weights)
    )
    policy = compile_strategy_rule_policy(
        "short_1_5d", rules, allowed_factor_ids=set(weights)
    )
    config["strategy_rule_ir"] = rules
    config["strategy_rules_sha256"] = rules["rules_sha256"]
    config["strategy_rule_policy_sha256"] = policy["policy_sha256"]
    for key, value in policy.items():
        if key in config and value is not None:
            config[key] = value
    config["entry_score_min_percentile"] = policy["entry_score_min_percentile"]
    config["execution_contract_hash"] = strategy_execution_contract_hash(config)
    return config


def _periods() -> dict:
    return {
        "in_sample": {"start": "2019-01-02", "end": "2021-12-31"},
        "out_of_sample": {"start": "2022-01-10", "end": "2023-12-29"},
        "governance": {
            "mode": "fin_strategy_pre_final",
            "pre_final_cutoff": "2023-12-29",
            "final_oos_opened": False,
            "historical_validation_periods": {
                "start": "2016-01-04",
                "end": "2018-12-28",
            },
        },
    }


def _plan() -> dict:
    return build_strategy_research_competition_plan(
        research_run_id="run-1",
        compiled_artifact_id="artifact-1",
        compiled_artifact_sha256="a" * 64,
        baseline_config=_config(),
        candidate_config=_candidate(),
        dataset="daily-20260829",
        dataset_identity_sha256="d" * 64,
        score_inputs_sha256=strategy_score_grid_contract(_config())["contract_sha256"],
        periods=_periods(),
    )


def _materialized_competition(tmp_path, *, stage: str = "policy_only") -> dict:
    candidate = _candidate()
    candidate["source_research_artifact_id"] = "artifact-1"
    candidate["strategy_research_artifact_sha256"] = "a" * 64
    candidate["strategy_research_data_contract"] = {
        "dataset_snapshot_id": "d" * 64,
        "feature_set_id": "alpha-short",
        "feature_set_definition_sha256": "f" * 64,
        "research_periods": {
            "train_start": "2016-01-04",
            "train_end": "2018-12-28",
            "valid_start": "2019-01-02",
            "valid_end": "2023-12-29",
            "test_start": "2024-01-02",
            "test_end": "2025-12-31",
        },
        "decision_frequency": "day",
        "label_horizon_trading_days": 5,
    }
    plan = build_strategy_research_competition_plan(
        research_run_id="run-1",
        compiled_artifact_id="artifact-1",
        compiled_artifact_sha256="a" * 64,
        baseline_config=_config(),
        candidate_config=candidate,
        dataset="daily-20260829",
        dataset_identity_sha256="d" * 64,
        score_inputs_sha256=strategy_score_grid_contract(_config())["contract_sha256"],
        periods=_periods(),
    )
    return {
        "strategy_version": {
            "id": "version-1",
            "status": "draft",
            "horizon_profile": "short_1_5d",
            "source_research_artifact_id": "artifact-1",
            "strategy_rules_sha256": candidate["strategy_rules_sha256"],
            "config": candidate,
        },
        "plan": plan,
        "stage": stage,
        "dataset": {
            "name": "daily-20260829",
            "path": str(tmp_path / "qlib"),
            "provenance": {"dataset_identity_sha256": "d" * 64},
        },
        "artifact_root": tmp_path / "experiments",
        "created_by": "system:fin-strategy",
    }


def _trial_results(challenger_role: str) -> list[dict]:
    metrics = {
        "deflated_sharpe_probability": 0.99,
        "robustness_passed": True,
        "component_cost_stress_passed": True,
        "rolling_passed": True,
        "event_stress_passed": True,
        "capacity_curve_passed": True,
    }
    return [
        {
            "role": "public_baseline",
            "status": "succeeded",
            "metrics": {"out_of_sample": dict(metrics)},
        },
        {
            "role": challenger_role,
            "status": "succeeded",
            "metrics": {"out_of_sample": dict(metrics)},
        },
    ]


def _returns(challenger_role: str) -> dict[str, pd.Series]:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2022-01-03", periods=320)
    baseline = pd.Series(rng.normal(0.0001, 0.008, len(index)), index=index)
    candidate = baseline + pd.Series(
        rng.normal(0.0015, 0.0002, len(index)), index=index
    )
    return {"public_baseline": baseline, challenger_role: candidate}


@pytest.mark.no_database
def test_plan_preregisters_policy_then_full_stack_on_existing_job_kind() -> None:
    plan = _plan()

    assert [stage["stage"] for stage in plan["stages"]] == [
        "policy_only",
        "full_stack",
    ]
    assert [stage["evaluation_mode"] for stage in plan["stages"]] == [
        STRATEGY_POLICY_ONLY_MODE,
        STRATEGY_FULL_STACK_MODE,
    ]
    assert {stage["job_kind"] for stage in plan["stages"]} == {
        "parameter_experiment"
    }
    policy_trials = plan["stages"][0]["trials"]
    assert strategy_score_grid_contract(policy_trials[0]["config"]) == (
        strategy_score_grid_contract(policy_trials[1]["config"])
    )
    full_trials = plan["stages"][1]["trials"]
    assert strategy_score_grid_contract(full_trials[0]["config"]) != (
        strategy_score_grid_contract(full_trials[1]["config"])
    )
    assert plan["capital_eligible"] is False
    assert plan["simulation_eligible"] is False


@pytest.mark.no_database
def test_plan_rejects_cost_or_execution_advantage() -> None:
    candidate = _candidate()
    candidate["fixed_slippage_rate"] *= 0.5
    candidate["execution_contract_hash"] = strategy_execution_contract_hash(candidate)

    with pytest.raises(ValueError, match="cost/execution"):
        build_strategy_research_competition_plan(
            research_run_id="run-1",
            compiled_artifact_id="artifact-1",
            compiled_artifact_sha256="a" * 64,
            baseline_config=_config(),
            candidate_config=candidate,
            dataset="daily",
            dataset_identity_sha256="d" * 64,
            score_inputs_sha256="e" * 64,
            periods=_periods(),
        )


@pytest.mark.no_database
def test_full_stack_cannot_run_before_passed_policy_evidence() -> None:
    plan = _plan()
    role = "full_stack_challenger"

    with pytest.raises(ValueError, match="passed policy-only"):
        build_strategy_stage_evidence(
            plan,
            stage_name="full_stack",
            trial_results=_trial_results(role),
            daily_returns=_returns(role),
            governed_score_sha256={"public_baseline": "1" * 64, role: "2" * 64},
        )


@pytest.mark.no_database
def test_two_stage_evidence_is_research_only_and_statistically_gated() -> None:
    plan = _plan()
    policy_role = "policy_challenger"
    policy = build_strategy_stage_evidence(
        plan,
        stage_name="policy_only",
        trial_results=_trial_results(policy_role),
        daily_returns=_returns(policy_role),
        governed_score_sha256={
            "public_baseline": "1" * 64,
            policy_role: "1" * 64,
        },
    )
    assert policy["gate_passed"] is True
    assert policy["next_gate"] == "full_stack_pre_final"
    assert policy["capital_eligible"] is False
    assert policy["final_oos_opened"] is False

    full_role = "full_stack_challenger"
    full = build_strategy_stage_evidence(
        plan,
        stage_name="full_stack",
        trial_results=_trial_results(full_role),
        daily_returns=_returns(full_role),
        governed_score_sha256={
            "public_baseline": "1" * 64,
            full_role: "2" * 64,
        },
        prerequisite_evidence=policy,
    )
    assert full["gate_passed"] is True
    assert full["next_gate"] == "formal_final_oos_once"
    assert full["prerequisite_evidence_sha256"] == policy["evidence_sha256"]


@pytest.mark.no_database
def test_policy_evidence_rejects_different_score_artifacts() -> None:
    plan = _plan()
    role = "policy_challenger"
    with pytest.raises(ValueError, match="same score grid"):
        build_strategy_stage_evidence(
            plan,
            stage_name="policy_only",
            trial_results=_trial_results(role),
            daily_returns=_returns(role),
            governed_score_sha256={"public_baseline": "1" * 64, role: "2" * 64},
        )


@pytest.mark.no_database
def test_existing_parameter_experiment_artifacts_become_research_artifact(
    tmp_path,
) -> None:
    plan = _plan()
    stage = plan["stages"][0]
    returns = _returns("policy_challenger")
    score = pd.DataFrame(
        {"score": np.linspace(-1.0, 1.0, 320)},
        index=pd.bdate_range("2022-01-03", periods=320),
    )
    results = _trial_results("policy_challenger")
    for index, item in enumerate(results):
        role = "public_baseline" if index == 0 else "policy_challenger"
        trial_root = tmp_path / f"trial-{index:03d}" / "out_of_sample"
        trial_root.mkdir(parents=True)
        pd.DataFrame(
            {
                "datetime": returns[role].index,
                "return": returns[role].to_numpy(),
                "cost": 0.0,
            }
        ).to_parquet(trial_root / "daily_returns.parquet", index=False)
        score.to_parquet(trial_root / "score_grid.parquet")
        item["trial_index"] = index
        item["parameters"] = stage["trials"][index]["parameters"]
    experiment = {
        "status": "ok",
        "experiment_id": "experiment-1",
        "evaluation_mode": STRATEGY_POLICY_ONLY_MODE,
        "final_oos_opened": False,
        "periods": stage["periods"],
        "trials": results,
    }

    artifact = build_strategy_stage_artifact_from_parameter_experiment(
        plan,
        stage_name="policy_only",
        experiment_result=experiment,
        artifact_root=tmp_path,
    )

    assert artifact["artifact_type"] == "fin_strategy_policy_only_evaluation"
    assert artifact["capital_eligible"] is False
    assert artifact["evidence"]["gate_passed"] is True


@pytest.mark.no_database
def test_strategy_competition_store_prepares_and_reuses_existing_dag(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = object.__new__(ParameterExperimentStore)
    persisted: dict[str, dict] = {}
    prepared_calls: list[dict] = []

    def persist(prepared):
        prepared_calls.append(prepared)
        key = prepared["competition_spec_sha256"]
        created = key not in persisted
        persisted.setdefault(
            key,
            {
                "id": "experiment-1",
                "status": "queued",
                "job_id": None,
                "periods": prepared["periods"],
            },
        )
        return persisted[key], created

    monkeypatch.setattr(store, "_persist_strategy_research_competition", persist)
    call = _materialized_competition(tmp_path)

    first = store.ensure_strategy_research_competition(**call)
    second = store.ensure_strategy_research_competition(**call)

    assert first["created"] is True
    assert second["created"] is False
    assert first["experiment"]["id"] == second["experiment"]["id"]
    prepared = prepared_calls[0]
    governance = prepared["periods"]["governance"]
    assert governance["mode"] == STRATEGY_POLICY_ONLY_MODE
    assert governance["final_oos_opened"] is False
    assert governance["plan_sha256"] == call["plan"]["plan_sha256"]
    assert governance["stage"] == "policy_only"
    assert governance["compiled_artifact_id"] == "artifact-1"
    assert governance["compiled_artifact_sha256"] == "a" * 64
    assert first["job_payload"]["strategy_evaluation_mode"] == (
        STRATEGY_POLICY_ONLY_MODE
    )
    assert first["job_payload"]["transparent_baseline_runner_sha256"] == (
        STRATEGY_RESEARCH_TARGET_RUNNER_SHA256
    )
    assert first["job_payload"]["transparent_baseline_runtime_bundle_sha256"] == (
        STRATEGY_RESEARCH_TARGET_RUNTIME_BUNDLE_SHA256
    )
    assert first["job_payload"][
        "transparent_baseline_worker_runtime_image_digest"
    ] == _WORKER_IMAGE_DIGEST


@pytest.mark.no_database
def test_strategy_competition_store_binds_full_stack_to_same_plan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = object.__new__(ParameterExperimentStore)
    captured: dict = {}

    def persist(prepared):
        captured.update(prepared)
        return {"id": "experiment-2", "status": "queued", "job_id": None}, True

    monkeypatch.setattr(store, "_persist_strategy_research_competition", persist)
    result = store.ensure_strategy_research_competition(
        **_materialized_competition(tmp_path, stage="full_stack")
    )

    assert captured["evaluation_mode"] == STRATEGY_FULL_STACK_MODE
    assert captured["periods"]["governance"]["stage"] == "full_stack"
    assert result["job_payload"]["strategy_competition_stage"] == "full_stack"


@pytest.mark.no_database
def test_strategy_competition_store_rejects_version_or_dataset_drift(
    tmp_path,
) -> None:
    store = object.__new__(ParameterExperimentStore)
    call = _materialized_competition(tmp_path)
    call["strategy_version"]["config"] = deepcopy(
        call["strategy_version"]["config"]
    )
    call["strategy_version"]["config"]["topk"] = 99
    with pytest.raises(ValueError, match="differs"):
        store.ensure_strategy_research_competition(**call)

    call = _materialized_competition(tmp_path)
    call["dataset"]["provenance"]["dataset_identity_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="dataset or research periods"):
        store.ensure_strategy_research_competition(**call)


def _strategy_result_contract(tmp_path) -> dict:
    call = _materialized_competition(tmp_path)
    prepared = ParameterExperimentStore._prepare_strategy_research_competition(
        **call
    )
    expected = [
        SimpleNamespace(
            trial_index=item["trial_index"],
            parameters_json=item["parameters"],
            config_json=item["config"],
        )
        for item in prepared["trials"]
    ]
    trial_results = []
    for item in prepared["trials"]:
        provenance = {
            "evaluation_mode": prepared["evaluation_mode"],
            "evaluation_scope": "pre_final_only",
            "final_oos_opened": False,
            "dataset_identity_sha256": prepared["dataset_identity_sha256"],
            "pre_final_cutoff": prepared["periods"]["governance"][
                "pre_final_cutoff"
            ],
            "strategy_config_sha256": item["config_sha256"],
        }
        segment = {
            "evaluation_mode": prepared["evaluation_mode"],
            "evaluation_scope": "pre_final_only",
            "final_oos_opened": False,
            "capital_eligible": False,
            "provenance": provenance,
        }
        trial_results.append(
            {
                "trial_index": item["trial_index"],
                "parameters": item["parameters"],
                "status": "succeeded",
                "metrics": {
                    "in_sample": deepcopy(segment),
                    "out_of_sample": deepcopy(segment),
                },
            }
        )
    summary = {
        "trial_count": 2,
        "governed_trial_count": 2,
        "prior_admitted_trial_count": 0,
        "succeeded_count": 2,
        "failed_count": 0,
        "final_oos_opened": False,
    }
    result = {
        "status": "ok",
        "experiment_id": "experiment-1",
        "strategy_version_id": prepared["version_id"],
        "dataset": prepared["dataset_name"],
        "evaluation_mode": prepared["evaluation_mode"],
        "final_oos_opened": False,
        "periods": prepared["periods"],
        "trials": trial_results,
        "summary": summary,
    }
    return {
        "experiment_id": "experiment-1",
        "experiment_row": SimpleNamespace(
            strategy_version_id=prepared["version_id"],
            dataset=prepared["dataset_name"],
        ),
        "periods": prepared["periods"],
        "governance": prepared["periods"]["governance"],
        "result": result,
        "trial_results": trial_results,
        "summary": summary,
        "expected_trials": expected,
    }


@pytest.mark.no_database
def test_strategy_result_contract_accepts_only_bound_pre_final_evidence(
    tmp_path,
) -> None:
    ParameterExperimentStore._validate_strategy_research_result(
        **_strategy_result_contract(tmp_path)
    )


@pytest.mark.no_database
@pytest.mark.parametrize(
    "drift",
    [
        "mode",
        "dataset",
        "periods",
        "top_level_final",
        "summary_final",
        "summary_count",
        "provenance_scope",
        "provenance_cutoff",
        "provenance_dataset",
        "provenance_config",
        "trial_parameters",
    ],
)
def test_strategy_result_contract_rejects_identity_or_authority_drift(
    tmp_path, drift: str
) -> None:
    values = _strategy_result_contract(tmp_path)
    result = values["result"]
    if drift == "mode":
        result["evaluation_mode"] = STRATEGY_FULL_STACK_MODE
    elif drift == "dataset":
        result["dataset"] = "another-dataset"
    elif drift == "periods":
        result["periods"] = deepcopy(result["periods"])
        result["periods"]["out_of_sample"]["end"] = "2023-12-28"
    elif drift == "top_level_final":
        result["final_oos_opened"] = True
    elif drift == "summary_final":
        values["summary"]["final_oos_opened"] = True
    elif drift == "summary_count":
        values["summary"]["governed_trial_count"] = 1
    elif drift == "trial_parameters":
        values["trial_results"][0]["parameters"] = {
            "strategy_comparison_role": "changed"
        }
    else:
        provenance = values["trial_results"][0]["metrics"]["out_of_sample"][
            "provenance"
        ]
        if drift == "provenance_scope":
            provenance["evaluation_scope"] = "final_oos_once"
        elif drift == "provenance_cutoff":
            provenance["pre_final_cutoff"] = "2025-12-31"
        elif drift == "provenance_dataset":
            provenance["dataset_identity_sha256"] = "9" * 64
        elif drift == "provenance_config":
            provenance["strategy_config_sha256"] = "9" * 64

    with pytest.raises(ValueError, match="fin_strategy"):
        ParameterExperimentStore._validate_strategy_research_result(**values)
