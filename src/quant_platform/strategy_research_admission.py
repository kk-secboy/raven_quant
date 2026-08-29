from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime
from math import isfinite
from typing import Any

from quant_platform.alpha_spending_integration import capital_oos_family_manifest
from quant_platform.research_horizon import research_horizon_contract
from quant_platform.strategy_research_evaluation import (
    STRATEGY_FULL_STACK_MODE,
    STRATEGY_POLICY_ONLY_MODE,
    STRATEGY_RESEARCH_COMPETITION_VERSION,
)
from quant_platform.strategy_rule_compiler import (
    validate_compiled_strategy_artifact,
    validate_strategy_rule_binding,
)
from quant_platform.strategy_rule_ir import canonical_sha256

FIN_STRATEGY_FORMAL_ADMISSION_VERSION = "fin-strategy-formal-admission-v1"
FIN_STRATEGY_POLICY_ARTIFACT_TYPE = "fin_strategy_policy_only_evaluation"
FIN_STRATEGY_FULL_STACK_ARTIFACT_TYPE = "fin_strategy_full_stack_evaluation"
FIN_STRATEGY_WINNER_ARTIFACT_TYPE = "fin_strategy_governed_winner"


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _require_sealed_document(
    value: Mapping[str, Any],
    *,
    digest_field: str,
    label: str,
) -> dict[str, Any]:
    result = deepcopy(dict(value))
    digest = str(result.pop(digest_field, ""))
    if not _is_sha256(digest) or digest != canonical_sha256(result):
        raise ValueError(f"{label} content seal is invalid")
    result[digest_field] = digest
    return result


def build_fin_strategy_winner_artifact(
    *,
    research_run_id: str,
    branch_outcomes: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal the one deterministic winner after every preregistered branch settles.

    Policy/full-stack evidence is still research-only.  This extra run-level
    decision prevents any individually passing proposal from presenting itself
    as the tournament winner and consuming the one capital-facing OOS window.
    """

    run_id = str(research_run_id or "").strip()
    if not run_id or not branch_outcomes:
        raise ValueError("fin_strategy winner decision requires a run and branches")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in branch_outcomes:
        item = deepcopy(dict(raw))
        version_id = str(item.get("strategy_version_id") or "").strip()
        status = str(item.get("status") or "")
        if not version_id or version_id in seen or status not in {
            "policy_rejected",
            "full_stack_rejected",
            "eligible",
        }:
            raise ValueError("fin_strategy winner branch outcome is invalid")
        seen.add(version_id)
        plan_sha256 = str(item.get("plan_sha256") or "")
        policy_sha256 = str(item.get("policy_evidence_sha256") or "")
        full_sha256 = str(item.get("full_stack_evidence_sha256") or "")
        if not _is_sha256(plan_sha256) or not _is_sha256(policy_sha256):
            raise ValueError("fin_strategy winner branch evidence is incomplete")
        if status == "policy_rejected":
            if full_sha256:
                raise ValueError("policy-rejected branch cannot contain full-stack evidence")
        elif not _is_sha256(full_sha256):
            raise ValueError("settled full-stack branch has no sealed evidence")
        if status == "eligible":
            for field in (
                "observed_mean_difference",
                "adjusted_p_value",
                "pbo",
            ):
                value = item.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(float(value))
                ):
                    raise ValueError("eligible fin_strategy branch has invalid ranking metrics")
        else:
            for field in (
                "observed_mean_difference",
                "adjusted_p_value",
                "pbo",
            ):
                item[field] = None
        normalized.append(item)
    normalized.sort(key=lambda item: str(item["strategy_version_id"]))
    eligible = [item for item in normalized if item["status"] == "eligible"]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -float(item["observed_mean_difference"]),
            float(item["adjusted_p_value"]),
            float(item["pbo"]),
            str(item["strategy_version_id"]),
        ),
    )
    winner_id = str(ranked[0]["strategy_version_id"]) if ranked else None
    artifact = {
        "contract_version": "fin-strategy-governed-winner-v1",
        "artifact_type": FIN_STRATEGY_WINNER_ARTIFACT_TYPE,
        "delivery_status": (
            "governed_evaluation_winner" if winner_id else "research_rejected"
        ),
        "capital_eligible": False,
        "simulation_eligible": False,
        "recommendation_eligible": False,
        "final_oos_opened": False,
        "all_branches_settled": True,
        "research_run_id": run_id,
        "selection_rule": (
            "max_mean_difference_then_min_adjusted_p_then_min_pbo_then_version_id"
        ),
        "branch_outcomes": normalized,
        "eligible_ranking": [str(item["strategy_version_id"]) for item in ranked],
        "winner_strategy_version_id": winner_id,
        "next_gate": (
            "preregister_capital_final_oos_once" if winner_id else "research_rejected"
        ),
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def validate_fin_strategy_winner_artifact(
    value: Mapping[str, Any],
    *,
    research_run_id: str,
    strategy_version_id: str,
) -> dict[str, Any]:
    artifact = _require_sealed_document(
        value,
        digest_field="artifact_sha256",
        label="fin_strategy governed winner artifact",
    )
    if (
        artifact.get("contract_version") != "fin-strategy-governed-winner-v1"
        or artifact.get("artifact_type") != FIN_STRATEGY_WINNER_ARTIFACT_TYPE
        or artifact.get("delivery_status") != "governed_evaluation_winner"
        or artifact.get("capital_eligible") is not False
        or artifact.get("simulation_eligible") is not False
        or artifact.get("recommendation_eligible") is not False
        or artifact.get("final_oos_opened") is not False
        or artifact.get("all_branches_settled") is not True
        or artifact.get("research_run_id") != research_run_id
        or artifact.get("winner_strategy_version_id") != strategy_version_id
        or artifact.get("next_gate") != "preregister_capital_final_oos_once"
    ):
        raise ValueError("fin_strategy proposal is not the governed run winner")
    rebuilt = build_fin_strategy_winner_artifact(
        research_run_id=research_run_id,
        branch_outcomes=list(artifact.get("branch_outcomes") or []),
    )
    if rebuilt != artifact:
        raise ValueError("fin_strategy governed winner ranking changed")
    return artifact


def _evaluation_evidence(
    artifact: Mapping[str, Any],
    *,
    artifact_type: str,
    stage: str,
    evaluation_mode: str,
    research_run_id: str,
    plan_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = _require_sealed_document(
        artifact,
        digest_field="artifact_sha256",
        label=f"fin_strategy {stage} artifact",
    )
    if (
        normalized.get("contract_version") != "fin-strategy-evaluation-artifact-v1"
        or normalized.get("delivery_status") != "research_only"
        or normalized.get("capital_eligible") is not False
        or normalized.get("research_run_id") != research_run_id
        or normalized.get("artifact_type") != artifact_type
        or not str(normalized.get("parameter_experiment_id") or "").strip()
        or not _is_sha256(normalized.get("parameter_experiment_result_sha256"))
    ):
        raise ValueError(f"fin_strategy {stage} artifact contract is invalid")
    evidence_value = normalized.get("evidence")
    if not isinstance(evidence_value, Mapping):
        raise ValueError(f"fin_strategy {stage} evidence is missing")
    evidence = _require_sealed_document(
        evidence_value,
        digest_field="evidence_sha256",
        label=f"fin_strategy {stage} evidence",
    )
    next_gate = (
        "full_stack_pre_final" if stage == "policy_only" else "formal_final_oos_once"
    )
    score_hashes = evidence.get("governed_score_sha256")
    if (
        evidence.get("contract_version") != "fin-strategy-stage-evidence-v1"
        or evidence.get("delivery_status") != "research_only"
        or evidence.get("capital_eligible") is not False
        or evidence.get("final_oos_opened") is not False
        or evidence.get("research_run_id") != research_run_id
        or evidence.get("plan_sha256") != plan_sha256
        or evidence.get("stage") != stage
        or evidence.get("evaluation_mode") != evaluation_mode
        or evidence.get("gate_passed") is not True
        or evidence.get("next_gate") != next_gate
        or not isinstance(score_hashes, Mapping)
        or len(score_hashes) != 2
        or not all(_is_sha256(item) for item in score_hashes.values())
    ):
        raise ValueError(f"fin_strategy {stage} gate did not pass its sealed contract")
    return normalized, evidence


def _stable_capital_mandate(
    *,
    horizon: str,
    baseline_config: Mapping[str, Any],
    stage: Mapping[str, Any],
    recipe_id: str,
    recipe_version: str,
    universe: str,
    benchmark: str,
) -> dict[str, Any]:
    """Build a family identity that cannot be reset by changing a challenger.

    The public control, horizon and platform competition contract define the
    investment mandate.  Candidate rules, run ids, dataset snapshots and OOS
    dates are deliberately excluded so repeated research cannot mint a fresh
    alpha budget.
    """

    cost_contract = stage.get("cost_contract")
    cost_sha256 = (
        str(cost_contract.get("contract_sha256") or "")
        if isinstance(cost_contract, Mapping)
        else ""
    )
    execution_sha256 = str(baseline_config.get("execution_contract_hash") or "")
    horizon_contract = research_horizon_contract(horizon)
    governance_sha256 = canonical_sha256(
        {
            "contract_version": "fin-strategy-capital-governance-v1",
            "competition_contract_version": STRATEGY_RESEARCH_COMPETITION_VERSION,
            "recipe_id": recipe_id,
            "recipe_version": recipe_version,
            "horizon_profile": horizon,
            "horizon_contract_sha256": horizon_contract.sha256,
            "simulation_only": True,
            "broker_connection_enabled": False,
            "real_trading_eligible": False,
        }
    )
    if not _is_sha256(cost_sha256) or not _is_sha256(execution_sha256):
        raise ValueError("fin_strategy capital mandate has no frozen cost or execution contract")
    return capital_oos_family_manifest(
        universe,
        benchmark,
        int(horizon_contract.label_horizons_sessions[-1]),
        cost_sha256,
        execution_sha256,
        governance_sha256,
    )


def build_fin_strategy_formal_admission(
    *,
    strategy_version: Mapping[str, Any],
    compiled_artifact: Mapping[str, Any],
    competition_plan: Mapping[str, Any],
    policy_evaluation_artifact: Mapping[str, Any],
    full_stack_evaluation_artifact: Mapping[str, Any],
    governed_winner_artifact: Mapping[str, Any],
    allow_approved_paper: bool = False,
) -> dict[str, Any]:
    """Bind a passed two-stage research tournament to one unopened formal OOS.

    This remains non-capital evidence.  The returned contract only authorizes
    preregistration of one CapitalOOSAlphaLedger batch; it never approves the
    strategy, opens paper evidence, or enables recommendations.
    """

    version_id = str(strategy_version.get("id") or "")
    config = dict(strategy_version.get("config") or {})
    lifecycle = (
        strategy_version.get("status"),
        strategy_version.get("promotion_stage"),
    )
    lifecycle_allowed = lifecycle == ("draft", None) or (
        allow_approved_paper and lifecycle == ("approved", "paper")
    )
    if not version_id or not lifecycle_allowed:
        raise ValueError("fin_strategy formal admission requires an inert draft version")
    source_artifact_id = str(
        strategy_version.get("source_research_artifact_id")
        or config.get("source_research_artifact_id")
        or ""
    )
    source_artifact_sha256 = str(config.get("strategy_research_artifact_sha256") or "")
    policy = validate_strategy_rule_binding(config)
    normalized_compiled = validate_compiled_strategy_artifact(
        compiled_artifact,
        allowed_factor_ids=set((policy or {}).get("alpha_factor_weights") or {}),
    )
    if (
        not source_artifact_id
        or normalized_compiled["artifact_sha256"] != source_artifact_sha256
    ):
        raise ValueError("fin_strategy draft differs from its compiled source artifact")

    plan = _require_sealed_document(
        competition_plan,
        digest_field="plan_sha256",
        label="fin_strategy competition plan",
    )
    plan_sha256 = str(plan["plan_sha256"])
    research_run_id = str(plan.get("research_run_id") or "")
    winner_artifact = validate_fin_strategy_winner_artifact(
        governed_winner_artifact,
        research_run_id=research_run_id,
        strategy_version_id=version_id,
    )
    horizon = str(config.get("horizon_profile") or "")
    stages = plan.get("stages")
    stage_by_name = {
        str(item.get("stage") or ""): item
        for item in stages or []
        if isinstance(item, Mapping)
    }
    if (
        plan.get("contract_version") != STRATEGY_RESEARCH_COMPETITION_VERSION
        or plan.get("delivery_status") != "research_only"
        or plan.get("capital_eligible") is not False
        or plan.get("simulation_eligible") is not False
        or plan.get("compiled_artifact_id") != source_artifact_id
        or plan.get("compiled_artifact_sha256") != source_artifact_sha256
        or plan.get("horizon") != horizon
        or not research_run_id
        or set(stage_by_name) != {"policy_only", "full_stack"}
        or plan.get("next_gate_after_success") != "formal_final_oos_once"
    ):
        raise ValueError("fin_strategy competition plan is not bound to the draft")
    policy_stage = stage_by_name["policy_only"]
    full_stage = stage_by_name["full_stack"]
    if (
        policy_stage.get("evaluation_mode") != STRATEGY_POLICY_ONLY_MODE
        or full_stage.get("evaluation_mode") != STRATEGY_FULL_STACK_MODE
        or policy_stage.get("final_oos_opened") is not False
        or full_stage.get("final_oos_opened") is not False
        or policy_stage.get("dataset") != full_stage.get("dataset")
        or policy_stage.get("dataset_identity_sha256")
        != full_stage.get("dataset_identity_sha256")
        or policy_stage.get("periods") != full_stage.get("periods")
    ):
        raise ValueError("fin_strategy stages do not share one unopened data contract")

    policy_artifact, policy_evidence = _evaluation_evidence(
        policy_evaluation_artifact,
        artifact_type=FIN_STRATEGY_POLICY_ARTIFACT_TYPE,
        stage="policy_only",
        evaluation_mode=STRATEGY_POLICY_ONLY_MODE,
        research_run_id=research_run_id,
        plan_sha256=plan_sha256,
    )
    full_artifact, full_evidence = _evaluation_evidence(
        full_stack_evaluation_artifact,
        artifact_type=FIN_STRATEGY_FULL_STACK_ARTIFACT_TYPE,
        stage="full_stack",
        evaluation_mode=STRATEGY_FULL_STACK_MODE,
        research_run_id=research_run_id,
        plan_sha256=plan_sha256,
    )
    if (
        full_evidence.get("prerequisite_evidence_sha256")
        != policy_evidence["evidence_sha256"]
        or policy_artifact.get("parameter_experiment_id")
        == full_artifact.get("parameter_experiment_id")
    ):
        raise ValueError("fin_strategy full-stack evidence bypassed its policy prerequisite")

    research_data = config.get("strategy_research_data_contract")
    research_periods = (
        research_data.get("research_periods")
        if isinstance(research_data, Mapping)
        else None
    )
    stage_periods = full_stage.get("periods")
    governance = (
        stage_periods.get("governance")
        if isinstance(stage_periods, Mapping)
        else None
    )
    if not isinstance(research_periods, Mapping) or not isinstance(governance, Mapping):
        raise ValueError("fin_strategy formal periods are missing")
    formal_periods = {
        "start": str(research_periods.get("test_start") or ""),
        "end": str(research_periods.get("test_end") or ""),
        "historical_start": str(research_periods.get("train_start") or ""),
        "historical_end": str(research_periods.get("valid_end") or ""),
    }
    if (
        not all(formal_periods.values())
        or not formal_periods["historical_start"] <= formal_periods["historical_end"]
        < formal_periods["start"]
        <= formal_periods["end"]
        or governance.get("pre_final_cutoff") != formal_periods["historical_end"]
        or governance.get("final_oos_opened") is not False
        or str((stage_periods.get("out_of_sample") or {}).get("end") or "")
        > formal_periods["historical_end"]
    ):
        raise ValueError("fin_strategy formal OOS was exposed or its periods changed")
    dataset_identity_sha256 = str(full_stage.get("dataset_identity_sha256") or "")
    if (
        not _is_sha256(dataset_identity_sha256)
        or research_data.get("dataset_snapshot_id") != dataset_identity_sha256
    ):
        raise ValueError("fin_strategy formal dataset identity changed after selection")

    baseline_trials = [
        item
        for item in full_stage.get("trials") or []
        if isinstance(item, Mapping) and item.get("role") == "public_baseline"
    ]
    if len(baseline_trials) != 1 or not isinstance(
        baseline_trials[0].get("config"), Mapping
    ):
        raise ValueError("fin_strategy plan has no unique public capital control")
    proposal = normalized_compiled["strategy_proposal"]
    capital_mandate = _stable_capital_mandate(
        horizon=horizon,
        baseline_config=baseline_trials[0]["config"],
        stage=full_stage,
        recipe_id=str(proposal["baseline_recipe_id"]),
        recipe_version=str(proposal["baseline_recipe_version"]),
        universe=str(strategy_version.get("universe") or ""),
        benchmark=str(strategy_version.get("benchmark") or ""),
    )
    admission = {
        "contract_version": FIN_STRATEGY_FORMAL_ADMISSION_VERSION,
        "delivery_status": "governed_evaluation_winner",
        "capital_eligible": False,
        "simulation_eligible": False,
        "recommendation_eligible": False,
        "final_oos_opened": False,
        "next_gate": "preregister_capital_final_oos_once",
        "strategy_version_id": version_id,
        "horizon_profile": horizon,
        "universe": str(strategy_version.get("universe") or ""),
        "benchmark": str(strategy_version.get("benchmark") or ""),
        "strategy_config_sha256": canonical_sha256(config),
        "source_research_artifact_id": source_artifact_id,
        "source_research_artifact_sha256": source_artifact_sha256,
        "research_run_id": research_run_id,
        "competition_plan_sha256": plan_sha256,
        "policy_evaluation_artifact_sha256": policy_artifact["artifact_sha256"],
        "policy_evidence_sha256": policy_evidence["evidence_sha256"],
        "full_stack_evaluation_artifact_sha256": full_artifact["artifact_sha256"],
        "full_stack_evidence_sha256": full_evidence["evidence_sha256"],
        "governed_winner_artifact_sha256": winner_artifact["artifact_sha256"],
        "dataset": str(full_stage.get("dataset") or ""),
        "dataset_identity_sha256": dataset_identity_sha256,
        "formal_periods": formal_periods,
        "public_baseline_config_sha256": canonical_sha256(
            baseline_trials[0]["config"]
        ),
        "capital_oos_stable_mandate": capital_mandate,
        "capital_oos_stable_mandate_sha256": canonical_sha256(capital_mandate),
    }
    admission["admission_sha256"] = canonical_sha256(admission)
    return admission


def validate_fin_strategy_formal_admission(value: Mapping[str, Any]) -> dict[str, Any]:
    admission = _require_sealed_document(
        value,
        digest_field="admission_sha256",
        label="fin_strategy formal admission",
    )
    if (
        admission.get("contract_version") != FIN_STRATEGY_FORMAL_ADMISSION_VERSION
        or admission.get("delivery_status") != "governed_evaluation_winner"
        or admission.get("capital_eligible") is not False
        or admission.get("simulation_eligible") is not False
        or admission.get("recommendation_eligible") is not False
        or admission.get("final_oos_opened") is not False
        or admission.get("next_gate") != "preregister_capital_final_oos_once"
        or not all(
            _is_sha256(admission.get(field))
            for field in (
                "strategy_config_sha256",
                "source_research_artifact_sha256",
                "competition_plan_sha256",
                "policy_evaluation_artifact_sha256",
                "policy_evidence_sha256",
                "full_stack_evaluation_artifact_sha256",
                "full_stack_evidence_sha256",
                "governed_winner_artifact_sha256",
                "dataset_identity_sha256",
                "public_baseline_config_sha256",
                "capital_oos_stable_mandate_sha256",
            )
        )
        or canonical_sha256(admission.get("capital_oos_stable_mandate"))
        != admission.get("capital_oos_stable_mandate_sha256")
    ):
        raise ValueError("fin_strategy formal admission contract is invalid")
    return admission


def build_fin_strategy_capital_oos_reservation(
    admission_value: Mapping[str, Any],
    *,
    dataset_lineage_id: str,
    trading_dates: list[date | datetime | str],
) -> dict[str, Any]:
    """Build the exact preregistration request for the persistent alpha ledger."""

    admission = validate_fin_strategy_formal_admission(admission_value)
    lineage = str(dataset_lineage_id or "").strip().lower()
    if not _is_sha256(lineage):
        raise ValueError("fin_strategy capital OOS requires a dataset lineage SHA256")
    try:
        calendar = [
            item.date()
            if isinstance(item, datetime)
            else item
            if isinstance(item, date)
            else date.fromisoformat(str(item))
            for item in trading_dates
        ]
        periods = admission["formal_periods"]
        research_end = date.fromisoformat(str(periods["historical_end"]))
        final_start = date.fromisoformat(str(periods["start"]))
        final_end = date.fromisoformat(str(periods["end"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fin_strategy capital OOS calendar or periods are invalid") from exc
    if calendar != sorted(calendar) or len(calendar) != len(set(calendar)):
        raise ValueError("fin_strategy capital OOS calendar must be increasing and unique")
    embargo_dates = [item for item in calendar if research_end < item < final_start]
    final_dates = [item for item in calendar if final_start <= item <= final_end]
    bundle_sha256 = canonical_sha256(
        {
            "contract_version": "fin-strategy-capital-frozen-bundle-v1",
            "strategy_version_id": admission["strategy_version_id"],
            "strategy_config_sha256": admission["strategy_config_sha256"],
            "source_research_artifact_sha256": admission[
                "source_research_artifact_sha256"
            ],
            "competition_plan_sha256": admission["competition_plan_sha256"],
            "policy_evidence_sha256": admission["policy_evidence_sha256"],
            "full_stack_evidence_sha256": admission[
                "full_stack_evidence_sha256"
            ],
            "governed_winner_artifact_sha256": admission[
                "governed_winner_artifact_sha256"
            ],
            "formal_admission_sha256": admission["admission_sha256"],
        }
    )
    baseline_sha256 = canonical_sha256(
        {
            "contract_version": "fin-strategy-capital-public-baseline-v1",
            "horizon_profile": admission["horizon_profile"],
            "universe": admission["universe"],
            "benchmark": admission["benchmark"],
            "public_baseline_config_sha256": admission[
                "public_baseline_config_sha256"
            ],
        }
    )
    batch_key = canonical_sha256(
        {
            "contract_version": "fin-strategy-capital-final-oos-batch-v1",
            "formal_admission_sha256": admission["admission_sha256"],
            "dataset_identity_sha256": admission["dataset_identity_sha256"],
            "dataset_lineage_id": lineage,
            "final_oos_trading_dates": [item.isoformat() for item in final_dates],
        }
    )
    return {
        "dataset_lineage_id": lineage,
        "dataset_identity_sha256": admission["dataset_identity_sha256"],
        "stable_mandate": admission["capital_oos_stable_mandate"],
        "batch_key": batch_key,
        "frozen_bundle_manifest_sha256": bundle_sha256,
        "frozen_baseline_manifest_sha256": baseline_sha256,
        "research_data_end": research_end,
        "final_oos_trading_dates": final_dates,
        "embargo_trading_dates": embargo_dates,
    }
