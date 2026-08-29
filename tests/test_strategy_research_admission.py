from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.promotion import PromotionStore
from quant_platform.strategy_proposal import STRATEGY_PROPOSAL_VERSION
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.strategy_research_admission import (
    build_fin_strategy_capital_oos_reservation,
    build_fin_strategy_formal_admission,
    build_fin_strategy_winner_artifact,
    validate_fin_strategy_formal_admission,
)
from quant_platform.strategy_research_evaluation import (
    build_public_strategy_control_config,
    build_strategy_research_competition_plan,
    strategy_score_grid_contract,
)
from quant_platform.strategy_rule_compiler import (
    compile_strategy_proposal,
    materialize_strategy_candidate_config,
)
from quant_platform.strategy_rule_ir import canonical_sha256

pytestmark = pytest.mark.no_database


def _proposal() -> dict:
    recipe = get_strategy_recipe("short_relative_strength")
    slots = deepcopy(recipe["strategy_rule_ir"]["slots"])
    weights = slots["alpha_rank"]["components"][0]["parameters"]["weights"]
    weights["relative_strength_5d"] = 0.30
    weights["amount_expansion_5d"] = 0.30
    return {
        "contract_version": STRATEGY_PROPOSAL_VERSION,
        "delivery_status": "research_only",
        "name": "short challenger",
        "description": "A structured policy challenger with a falsifiable entry rule.",
        "horizon": "short_1_5d",
        "economic_hypothesis": "A stricter entry filter may improve after-cost returns.",
        "baseline_recipe_id": recipe["id"],
        "baseline_recipe_version": recipe["version"],
        "baseline_rules_sha256": recipe["strategy_rule_ir"]["rules_sha256"],
        "parent_strategy_version_id": None,
        "changed_slots": ["alpha_rank"],
        "data_contract": {
            "dataset_snapshot_id": "d" * 64,
            "feature_set_id": "governed-short",
            "feature_set_definition_sha256": "f" * 64,
            "research_periods": {
                "train_start": "2008-01-02",
                "train_end": "2018-12-28",
                "valid_start": "2019-01-02",
                "valid_end": "2021-12-31",
                "test_start": "2022-02-01",
                "test_end": "2023-03-31",
            },
            "decision_frequency": "day",
            "label_horizon_trading_days": 5,
        },
        "evaluation_contract": {
            "benchmark": "SH000300",
            "primary_metric": "after_cost_information_ratio",
            "cost_schedule_version": COST_SCHEDULE_VERSION,
            "rolling_folds": 5,
            "minimum_oos_observations": 252,
            "final_oos_visible_during_selection": False,
        },
        "slots": slots,
    }


def _sealed_evaluation(plan: dict, stage: str, prerequisite: str | None = None) -> dict:
    mode = {
        "policy_only": "strategy_policy_only_pre_final",
        "full_stack": "strategy_full_stack_pre_final",
    }[stage]
    evidence = {
        "contract_version": "fin-strategy-stage-evidence-v1",
        "delivery_status": "research_only",
        "capital_eligible": False,
        "final_oos_opened": False,
        "research_run_id": plan["research_run_id"],
        "plan_sha256": plan["plan_sha256"],
        "stage": stage,
        "evaluation_mode": mode,
        "trial_roles": [
            "public_baseline",
            "policy_challenger" if stage == "policy_only" else "full_stack_challenger",
        ],
        "governed_score_sha256": {
            "public_baseline": "1" * 64,
            "policy_challenger" if stage == "policy_only" else "full_stack_challenger": (
                "1" * 64 if stage == "policy_only" else "2" * 64
            ),
        },
        "paired_block_bootstrap": {"status": "ok"},
        "alpha_spending": {"family_alpha": 0.05},
        "pbo": {"status": "ok", "pbo": 0.1},
        "challenger_stress_gates_passed": True,
        "prerequisite_evidence_sha256": prerequisite,
        "gate_passed": True,
        "next_gate": (
            "full_stack_pre_final"
            if stage == "policy_only"
            else "formal_final_oos_once"
        ),
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    artifact = {
        "contract_version": "fin-strategy-evaluation-artifact-v1",
        "delivery_status": "research_only",
        "capital_eligible": False,
        "research_run_id": plan["research_run_id"],
        "artifact_type": f"fin_strategy_{stage}_evaluation",
        "parameter_experiment_id": f"experiment-{stage}",
        "parameter_experiment_result_sha256": (
            "3" * 64 if stage == "policy_only" else "4" * 64
        ),
        "evidence": evidence,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def _admission_inputs() -> dict:
    proposal = _proposal()
    factors = set(
        proposal["slots"]["alpha_rank"]["components"][0]["parameters"]["weights"]
    )
    compiled = compile_strategy_proposal(proposal, allowed_factor_ids=factors)
    config = materialize_strategy_candidate_config(
        compiled,
        source_research_artifact_id="compiled-artifact",
        allowed_factor_ids=factors,
    )
    baseline = build_public_strategy_control_config(config)
    periods = {
        "in_sample": {"start": "2016-01-04", "end": "2018-12-28"},
        "out_of_sample": {"start": "2019-01-02", "end": "2021-12-31"},
        "governance": {
            "final_oos_opened": False,
            "pre_final_cutoff": "2021-12-31",
            "historical_validation_periods": {
                "start": "2008-01-02",
                "end": "2015-12-31",
            },
        },
    }
    plan = build_strategy_research_competition_plan(
        research_run_id="run-1",
        compiled_artifact_id="compiled-artifact",
        compiled_artifact_sha256=compiled["artifact_sha256"],
        baseline_config=baseline,
        candidate_config=config,
        dataset="daily-frozen",
        dataset_identity_sha256="d" * 64,
        score_inputs_sha256=strategy_score_grid_contract(baseline)[
            "contract_sha256"
        ],
        periods=periods,
        benchmark="SH000300",
        universe="cn_all",
    )
    policy = _sealed_evaluation(plan, "policy_only")
    full = _sealed_evaluation(
        plan,
        "full_stack",
        policy["evidence"]["evidence_sha256"],
    )
    winner = build_fin_strategy_winner_artifact(
        research_run_id="run-1",
        branch_outcomes=[
            {
                "strategy_version_id": "version-1",
                "status": "eligible",
                "plan_sha256": plan["plan_sha256"],
                "policy_evidence_sha256": policy["evidence"]["evidence_sha256"],
                "full_stack_evidence_sha256": full["evidence"]["evidence_sha256"],
                "observed_mean_difference": 0.001,
                "adjusted_p_value": 0.01,
                "pbo": 0.10,
            }
        ],
    )
    return {
        "strategy_version": {
            "id": "version-1",
            "strategy_id": "strategy-1",
            "status": "draft",
            "promotion_stage": None,
            "source_research_artifact_id": "compiled-artifact",
            "horizon_profile": "short_1_5d",
            "universe": "cn_all",
            "benchmark": "SH000300",
            "config": config,
        },
        "compiled_artifact": compiled,
        "competition_plan": plan,
        "policy_evaluation_artifact": policy,
        "full_stack_evaluation_artifact": full,
        "governed_winner_artifact": winner,
    }


def test_passed_research_stages_only_authorize_capital_oos_preregistration() -> None:
    admission = build_fin_strategy_formal_admission(**_admission_inputs())
    assert admission["delivery_status"] == "governed_evaluation_winner"
    assert admission["capital_eligible"] is False
    assert admission["simulation_eligible"] is False
    assert admission["recommendation_eligible"] is False
    assert admission["final_oos_opened"] is False
    assert admission["next_gate"] == "preregister_capital_final_oos_once"
    assert validate_fin_strategy_formal_admission(admission) == admission

    calendar = [item.date() for item in pd.bdate_range("2008-01-02", "2023-03-31")]
    request = build_fin_strategy_capital_oos_reservation(
        admission,
        dataset_lineage_id="a" * 64,
        trading_dates=calendar,
    )
    assert request["dataset_identity_sha256"] == "d" * 64
    assert len(request["final_oos_trading_dates"]) >= 252
    assert len(request["embargo_trading_dates"]) >= 20
    assert "strategy_version_id" not in request["stable_mandate"]
    assert "dataset_identity_sha256" not in request["stable_mandate"]


def test_full_stack_cannot_bypass_policy_evidence_or_failed_gate() -> None:
    values = _admission_inputs()
    full = deepcopy(values["full_stack_evaluation_artifact"])
    full["evidence"]["prerequisite_evidence_sha256"] = "9" * 64
    full["evidence"]["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in full["evidence"].items() if key != "evidence_sha256"}
    )
    full["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in full.items() if key != "artifact_sha256"}
    )
    values["full_stack_evaluation_artifact"] = full
    with pytest.raises(ValueError, match="bypassed its policy prerequisite"):
        build_fin_strategy_formal_admission(**values)

    values = _admission_inputs()
    values["policy_evaluation_artifact"]["evidence"]["gate_passed"] = False
    values["policy_evaluation_artifact"]["evidence"]["evidence_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in values["policy_evaluation_artifact"]["evidence"].items()
            if key != "evidence_sha256"
        }
    )
    values["policy_evaluation_artifact"]["artifact_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in values["policy_evaluation_artifact"].items()
            if key != "artifact_sha256"
        }
    )
    with pytest.raises(ValueError, match="gate did not pass"):
        build_fin_strategy_formal_admission(**values)


def test_historical_winner_cannot_claim_approved_or_recommendation_state() -> None:
    values = _admission_inputs()
    values["strategy_version"]["status"] = "approved"
    values["strategy_version"]["promotion_stage"] = "recommendation_enabled"
    with pytest.raises(ValueError, match="inert draft"):
        build_fin_strategy_formal_admission(**values)

    initial = _admission_inputs()
    admission = build_fin_strategy_formal_admission(**initial)
    paper = _admission_inputs()
    paper["strategy_version"]["status"] = "approved"
    paper["strategy_version"]["promotion_stage"] = "paper"
    assert (
        build_fin_strategy_formal_admission(
            **paper,
            allow_approved_paper=True,
        )
        == admission
    )
    tampered = deepcopy(admission)
    tampered["recommendation_eligible"] = True
    with pytest.raises(ValueError, match="content seal"):
        validate_fin_strategy_formal_admission(tampered)


def test_individual_full_stack_pass_cannot_bypass_run_level_winner() -> None:
    values = _admission_inputs()
    other = deepcopy(values["governed_winner_artifact"])
    other["branch_outcomes"][0]["strategy_version_id"] = "version-2"
    other["eligible_ranking"] = ["version-2"]
    other["winner_strategy_version_id"] = "version-2"
    other["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in other.items() if key != "artifact_sha256"}
    )
    values["governed_winner_artifact"] = other
    with pytest.raises(ValueError, match="not the governed run winner"):
        build_fin_strategy_formal_admission(**values)


def test_governed_winner_waits_for_all_branches_and_ranks_deterministically() -> None:
    common = {
        "status": "eligible",
        "plan_sha256": "1" * 64,
        "policy_evidence_sha256": "2" * 64,
        "full_stack_evidence_sha256": "3" * 64,
        "adjusted_p_value": 0.01,
        "pbo": 0.10,
    }
    winner = build_fin_strategy_winner_artifact(
        research_run_id="run-1",
        branch_outcomes=[
            {
                **common,
                "strategy_version_id": "version-b",
                "observed_mean_difference": 0.002,
            },
            {
                **common,
                "strategy_version_id": "version-a",
                "observed_mean_difference": 0.003,
            },
            {
                "strategy_version_id": "version-c",
                "status": "policy_rejected",
                "plan_sha256": "4" * 64,
                "policy_evidence_sha256": "5" * 64,
                "full_stack_evidence_sha256": "",
            },
        ],
    )
    assert winner["all_branches_settled"] is True
    assert winner["winner_strategy_version_id"] == "version-a"
    assert winner["eligible_ranking"] == ["version-a", "version-b"]
    assert winner["capital_eligible"] is False
    assert winner["recommendation_eligible"] is False


def test_governed_winner_records_complete_negative_result() -> None:
    rejected = build_fin_strategy_winner_artifact(
        research_run_id="run-1",
        branch_outcomes=[
            {
                "strategy_version_id": "version-a",
                "status": "policy_rejected",
                "plan_sha256": "1" * 64,
                "policy_evidence_sha256": "2" * 64,
                "full_stack_evidence_sha256": "",
            }
        ],
    )
    assert rejected["delivery_status"] == "research_rejected"
    assert rejected["winner_strategy_version_id"] is None
    assert rejected["next_gate"] == "research_rejected"


def test_auto_promotion_reconciles_the_paper_stage_before_reading_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = object.__new__(PromotionStore)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        store,
        "prepare_paper_stage",
        lambda version_id, actor: calls.append((version_id, actor)),
    )
    monkeypatch.setattr(
        store,
        "evaluate_forward_gate",
        lambda version_id: {
            "passed": False,
            "reasons": ["90 trading days are not complete"],
        },
    )
    result = store.auto_promote_if_ready("version-1")
    assert calls == [("version-1", "system:auto-promotion")]
    assert result["promoted"] is False
    assert result["promotion_stage"] == "paper"
