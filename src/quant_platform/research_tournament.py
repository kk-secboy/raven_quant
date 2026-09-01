from __future__ import annotations

import hashlib
import itertools
import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    autopilot_branches,
    jobs,
    model_candidates,
    model_ensemble_candidates,
    model_ensemble_evaluations,
    open_database,
    research_runs,
    research_tournament_trials,
    research_tournaments,
    row_dict,
)

from .feature_set_registry import get_feature_set
from .model_ensemble import (
    MODEL_ENSEMBLE_EVALUATION_CONTRACT_VERSION,
    prediction_grid_from_admission,
    validate_model_ensemble_label_contract,
)
from .model_research_governance import (
    REQUIRED_MODEL_SEEDS,
    REQUIRED_RESEARCH_PROFILES,
    file_sha256,
    require_model_metric_gate,
    validate_run_multiple_testing_evidence,
)
from .research_execution_cadence import (
    build_research_execution_cadence_contract,
    validate_research_execution_cadence_contract,
)

FEATURE_SCREEN_IDS = ("qlib-alpha158", "qlib-alpha360", "platform-seed-v1")
FEATURE_SET_COMMON_WINDOW_POLICY_VERSION = "feature-set-common-window-policy-v1"
FEATURE_SET_COMMON_WINDOW_POLICY: dict[str, Any] = {
    "contract_version": FEATURE_SET_COMMON_WINDOW_POLICY_VERSION,
    "candidate_scope": "all_preregistered_feature_sets",
    "field_scope": "union",
    "calendar_policy": "latest_common_continuous_field_coverage",
    "identical_periods_within_horizon": True,
    "missing_history_policy": "exclude_sessions_fail_closed_never_zero_backfill",
}
MODEL_FAMILIES: dict[str, dict[str, Any]] = {
    "ridge": {
        "engine": "ridge_baseline",
        "alpha": 1.0,
        "stochastic": False,
        "requires_train_fitted_standardization": True,
    },
    "lightgbm": {
        "engine": "lightgbm_baseline",
        "preset": "pinned_baseline",
        "stochastic": False,
    },
    "gru": {
        "engine": "platform_gru",
        "layers": 1,
        "hidden_size": 32,
        "recurrent_dropout": 0.0,
        "output_dropout": 0.1,
        "step_len": 20,
    },
    "transformer": {
        "engine": "platform_transformer",
        "layers": 2,
        "d_model": 32,
        "nhead": 4,
        "dropout": 0.1,
        "step_len": 20,
    },
}
SCREEN_SEEDS = (11,)
FULL_SEEDS = (11, 29, 47)
FULL_PROFILES = ("recent_3y", "balanced_5y", "robust_10y")
ENSEMBLE_MAX_CANDIDATES = 4
ENSEMBLE_MAX_MEMBERS = 3
ENSEMBLE_CORRELATION_LIMIT = 0.90
RDAGENT_DYNAMIC_MODEL_TRIAL_CAPACITY = 8
RESEARCH_SCREENING_MARKERS: dict[str, bool] = {
    "research_screening_only": True,
    "not_capital_confirmation": True,
    "cross_cycle_fwer_claimed": False,
    "final_oos_opened": False,
}
_REMEDIABLE_TOURNAMENT_STATUSES = frozenset({"planned", "running", "failed", "blocked"})
_TERMINAL_BRANCH_STATUSES = frozenset({"succeeded", "failed", "blocked", "skipped"})
_TERMINAL_RESEARCH_RUN_STATUSES = frozenset(
    {"succeeded", "failed", "blocked", "cancelled"}
)
_TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
QUANT_SCREENING_ABLATIONS = ("factor_only", "model_only", "joint")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def candidate_feature_sets(active_sota_feature_set_id: str | None = None) -> list[dict[str, Any]]:
    ids = list(FEATURE_SCREEN_IDS)
    if active_sota_feature_set_id:
        ids.append(active_sota_feature_set_id)
    return [get_feature_set(item) for item in ids]


def build_preregistered_manifest(
    *,
    active_sota_feature_set_id: str | None = None,
) -> dict[str, Any]:
    feature_sets = candidate_feature_sets(active_sota_feature_set_id)
    by_id = {str(item["id"]): item for item in feature_sets}
    feature_set_ids = [str(item["id"]) for item in feature_sets]
    feature_trials: list[dict[str, Any]] = []
    # Round one is deliberately one fixed learner across every feature set.
    # Changing the model together with the features would make the source of
    # an apparent improvement unknowable.
    for feature_set_id in feature_set_ids:
        feature = by_id[feature_set_id]
        family = "lightgbm"
        feature_trials.append(
            {
                "name": f"feature-screen:{feature['id']}:{family}",
                "trial_kind": "feature_set",
                "feature_set_id": feature["id"],
                "feature_set_definition_sha256": feature["definition_sha256"],
                "model_family": family,
                "spec": {
                    "round": "feature_screen",
                    "model": MODEL_FAMILIES[family],
                    "selection_profiles": ["recent_3y"],
                    "selection_seeds": list(SCREEN_SEEDS),
                    "selection": "top_two_feature_sets_under_fixed_lightgbm",
                    # Round one is the deliberately cheap screen promised by
                    # the public contract: recent window and fixed seed 11.
                    # The complete three-window/three-seed grid belongs only
                    # to model_full, after the two feature sets are frozen.
                    "evaluation_grid": {
                        "profiles": ["recent_3y"],
                        "seeds": list(SCREEN_SEEDS),
                    },
                },
                "resource": {"cpu_only": True, "evaluation_concurrency_cap": 3},
            }
        )
    model_trials: list[dict[str, Any]] = []
    # The top two feature sets are not known at preregistration time.  Reserve
    # every possible full trial now; after the fixed-model screen, trials for
    # the two losing sets are explicitly rejected and remain in the ledger.
    for feature_set_id in feature_set_ids:
        feature = by_id[feature_set_id]
        for family in MODEL_FAMILIES:
            model = MODEL_FAMILIES[family]
            # Seeds are robustness repetitions inside one preregistered
            # hypothesis, never three independent chances to win.  Ridge
            # is deterministic, but the three cells are kept as cheap
            # integrity repeats so its evidence grid remains compatible
            # with the existing formal StrategyStore contract.
            full_seeds = FULL_SEEDS
            screen_dependency = f"feature-screen:{feature['id']}:lightgbm"
            model_trials.append(
                {
                    "name": f"model-full:{feature['id']}:{family}",
                    "trial_kind": "model",
                    "feature_set_id": feature["id"],
                    "feature_set_definition_sha256": feature["definition_sha256"],
                    "model_family": family,
                    "spec": {
                        "round": "model_full",
                        "depends_on": screen_dependency,
                        "eligible_if": "feature_screen_top_two",
                        "model": model,
                        "profiles": list(FULL_PROFILES),
                        "seeds": list(full_seeds),
                        "max_epochs": 12 if family in {"gru", "transformer"} else None,
                        "early_stop": 3 if family in {"gru", "transformer"} else None,
                    },
                    "resource": {
                        "cpu_only": True,
                        "exclusive_concurrency": 1 if family == "transformer" else None,
                        "evaluation_concurrency_cap": 3,
                        "reserved_service_fraction": 0.25,
                    },
                }
            )
    return {
        "contract_version": "quantlab-tournament-v1",
        "feature_sets": [
            {
                "id": item["id"],
                "definition_sha256": item["definition_sha256"],
                "feature_count": len(item["features"]),
            }
            for item in feature_sets
        ],
        "model_families": MODEL_FAMILIES,
        "competition": {
            "feature_screen": {
                "feature_set_ids": feature_set_ids,
                "model_family": "lightgbm",
                "selected_feature_set_count": 2,
                "selection_profile": "recent_3y",
                "selection_seed": 11,
            },
            "model_full": {
                "model_families": list(MODEL_FAMILIES),
                "profiles": list(FULL_PROFILES),
                "seeds": list(FULL_SEEDS),
            },
        },
        "feature_set_window_policy": dict(FEATURE_SET_COMMON_WINDOW_POLICY),
        "ensemble": {
            "combiner": "equal_rank",
            "candidate_limit": ENSEMBLE_MAX_CANDIDATES,
            "member_limit": ENSEMBLE_MAX_MEMBERS,
            "prediction_correlation_limit": ENSEMBLE_CORRELATION_LIMIT,
            "stacking_allowed": False,
            "adaptive_trial_capacity": ENSEMBLE_MAX_CANDIDATES,
        },
        "resource_policy": {
            "cpu_only": True,
            "qlib_evaluation_concurrency_cap": 3,
            "transformer_concurrency_cap": 1,
            "reserved_service_fraction": 0.25,
        },
        "dynamic_model_trial_capacity": RDAGENT_DYNAMIC_MODEL_TRIAL_CAPACITY,
        "trials": feature_trials + model_trials,
        **RESEARCH_SCREENING_MARKERS,
    }


def build_quant_preregistered_manifest(
    *,
    parent_tournament: Mapping[str, Any],
    dataset_identity_sha256: str,
    baseline_prediction_champion: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the complete fin_quant research family before Qlib executes it.

    This is deliberately a research-accounting manifest.  It does not spend
    capital-confirmation alpha and it never opens the final OOS.  Failed jobs
    remain members of the batch and are represented by p=1 in the independent
    evaluator's run-level Holm/PBO evidence.
    """

    identity = str(dataset_identity_sha256 or "").lower()
    parent_identity = str(parent_tournament.get("dataset_identity_sha256") or "").lower()
    parent_stage = str(parent_tournament.get("stage") or "")
    parent_manifest = dict(parent_tournament.get("manifest") or {})
    revalidation_parent = (
        parent_stage == "model_screen"
        and parent_manifest.get("contract_version")
        == "champion-current-identity-revalidation-v1"
    )
    if (
        len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
        or parent_identity != identity
        or (parent_stage != "feature_screen" and not revalidation_parent)
        or parent_tournament.get("status") != "succeeded"
    ):
        raise ValueError("fin_quant requires one completed model tournament identity")
    baseline = dict(baseline_prediction_champion)
    baseline_sha = str(baseline.get("evidence_sha256") or "").lower()
    if len(baseline_sha) != 64 or any(
        character not in "0123456789abcdef" for character in baseline_sha
    ):
        raise ValueError("fin_quant incumbent evidence is not immutable")

    trial_specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in sorted(candidates, key=lambda value: str(value.get("candidate_id") or "")):
        candidate = dict(raw)
        candidate_id = str(candidate.get("candidate_id") or "")
        bundle_sha = str(candidate.get("bundle_manifest_sha256") or "").lower()
        model_candidate_id = str(candidate.get("model_candidate_id") or "")
        feature_sha = str(candidate.get("feature_set_definition_sha256") or "").lower()
        candidate_baseline_sha = str(
            candidate.get("baseline_prediction_champion_sha256") or ""
        ).lower()
        label_binding_sha = str(
            candidate.get("research_label_binding_sha256") or ""
        ).lower()
        factor_bundle_sha = str(
            candidate.get("horizon_factor_bundle_sha256") or ""
        ).lower()
        if (
            not candidate_id
            or candidate_id in seen
            or not model_candidate_id
            or len(bundle_sha) != 64
            or any(character not in "0123456789abcdef" for character in bundle_sha)
            or len(feature_sha) != 64
            or any(character not in "0123456789abcdef" for character in feature_sha)
            or candidate_baseline_sha != baseline_sha
            or (
                bool(label_binding_sha)
                and (
                    len(label_binding_sha) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in label_binding_sha
                    )
                    or len(factor_bundle_sha) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in factor_bundle_sha
                    )
                )
            )
            or (not label_binding_sha and bool(factor_bundle_sha))
        ):
            raise ValueError("fin_quant preregistration candidate is invalid")
        seen.add(candidate_id)
        factor_candidate_ids = sorted(
            str(value) for value in candidate.get("factor_candidate_ids") or []
        )
        if not factor_candidate_ids or len(factor_candidate_ids) != len(
            set(factor_candidate_ids)
        ):
            raise ValueError("fin_quant preregistration factors are invalid")
        trial_specs.append(
            {
                "name": f"fin-quant:{candidate_id}:joint-vs-incumbent",
                "trial_kind": "quant_bundle",
                "candidate_id": candidate_id,
                "spec": {
                    "round": "fin_quant",
                    "candidate_id": candidate_id,
                    "bundle_manifest_sha256": bundle_sha,
                    "model_candidate_id": model_candidate_id,
                    "factor_candidate_ids": factor_candidate_ids,
                    "experiment_family_id": str(
                        candidate.get("experiment_family_id") or ""
                    ),
                    "feature_set_id": str(candidate.get("feature_set_id") or ""),
                    "feature_set_definition_sha256": feature_sha,
                    "baseline_prediction_champion_sha256": baseline_sha,
                    **(
                        {
                            "research_label_binding_sha256": label_binding_sha,
                            "horizon_factor_bundle_sha256": factor_bundle_sha,
                        }
                        if label_binding_sha
                        else {}
                    ),
                    "required_ablations": list(QUANT_SCREENING_ABLATIONS),
                    "profiles": list(FULL_PROFILES),
                    "seeds": list(FULL_SEEDS),
                    "comparison": "joint_vs_frozen_incumbent",
                    **RESEARCH_SCREENING_MARKERS,
                },
                "resource": {
                    "cpu_only": True,
                    "evaluation_concurrency_cap": 3,
                },
            }
        )
    if not trial_specs:
        raise ValueError("fin_quant preregistration requires at least one candidate")
    return {
        "contract_version": "fin-quant-research-tournament-v1",
        "stage": "quant",
        "parent_tournament_id": str(parent_tournament.get("id") or ""),
        "parent_tournament_manifest_sha256": str(
            parent_tournament.get("manifest_sha256") or ""
        ),
        "parent_selection_evidence_sha256": str(
            parent_tournament.get("multiple_testing_sha256") or ""
        ),
        "dataset_identity_sha256": identity,
        "baseline_prediction_champion": baseline,
        "baseline_prediction_champion_sha256": baseline_sha,
        "trial_count": len(trial_specs),
        "batch_statistics": {
            "holm": "independent_evaluator_shared_family",
            "pbo": "independent_evaluator_shared_family",
            "failed_candidate_raw_p_value": 1.0,
            "dsr": "deferred_to_pre_final_portfolio_with_full_prior_trial_count",
        },
        "trials": trial_specs,
        **RESEARCH_SCREENING_MARKERS,
    }


def build_quant_screening_evidence(
    *,
    tournament_id: str,
    parent_tournament_id: str,
    dataset_identity_sha256: str,
    outcomes: Sequence[Mapping[str, Any]],
    run_multiple_testing: Mapping[str, Any] | None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Build an immutable, non-capital fin_quant batch settlement envelope."""

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in sorted(outcomes, key=lambda value: str(value.get("candidate_id") or "")):
        item = dict(raw)
        candidate_id = str(item.get("candidate_id") or "")
        status = str(item.get("status") or "")
        if not candidate_id or candidate_id in seen or status not in {
            "passed",
            "rejected",
            "failed",
        }:
            raise ValueError("fin_quant screening outcome is invalid")
        seen.add(candidate_id)
        normalized.append(
            {
                "candidate_id": candidate_id,
                "status": status,
                "evidence_sha256": str(item.get("evidence_sha256") or ""),
                "reason": str(item.get("reason") or "")[:3000],
            }
        )
    if not normalized:
        raise ValueError("fin_quant screening settlement has no outcomes")
    multiple = dict(run_multiple_testing) if isinstance(run_multiple_testing, Mapping) else None
    evidence = {
        "contract_version": "fin-quant-research-screening-ledger-v1",
        "tournament_id": str(tournament_id),
        "parent_tournament_id": str(parent_tournament_id),
        "dataset_identity_sha256": str(dataset_identity_sha256),
        "outcomes": normalized,
        "passed_candidate_ids": [
            item["candidate_id"] for item in normalized if item["status"] == "passed"
        ],
        "failed_and_rejected_trials_retained": True,
        "failed_candidate_raw_p_value": 1.0,
        "run_multiple_testing": multiple,
        "run_multiple_testing_evidence_sha256": (
            str(multiple.get("evidence_sha256") or "") if multiple else None
        ),
        "statistical_settlement": (
            "within_batch_holm_pbo"
            if multiple is not None
            else "unavailable_batch_execution_failure_no_candidate_admitted"
        ),
        "dsr_accounting": "carried_to_pre_final_portfolio_with_full_prior_trial_count",
        "failure_reason": str(failure_reason or "")[:3000] or None,
        **RESEARCH_SCREENING_MARKERS,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def build_champion_revalidation_manifest(
    *,
    dataset_identity_sha256: str,
    source_champion: Mapping[str, Any],
    source_selection_evidence_sha256: str,
    source_model_evidence_sha256: str,
    components: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze an exact current-vintage revalidation, not a new model search.

    A monthly tournament can choose a model or an equal-rank ensemble.  On a
    later daily Qlib vintage we must not silently reuse its old predictions,
    nor launch a new adaptive model contest just to obtain a current model.
    This manifest fixes the prior winner's code recipe, feature definition and
    (for an ensemble) all members before any current-vintage training starts.
    """

    identity = str(dataset_identity_sha256 or "").lower()
    if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
        raise ValueError("champion revalidation requires a sealed dataset identity")
    champion = dict(source_champion)
    kind = str(champion.get("kind") or "")
    if kind not in {"model", "ensemble"}:
        raise ValueError("champion revalidation source kind is invalid")
    if not str(champion.get("candidate_id") or ""):
        raise ValueError("champion revalidation source candidate is missing")
    source_hashes = {
        "source_selection_evidence_sha256": str(source_selection_evidence_sha256 or "").lower(),
        "source_model_evidence_sha256": str(source_model_evidence_sha256 or "").lower(),
    }
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in source_hashes.values()
    ):
        raise ValueError("champion revalidation source evidence is invalid")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_families: set[str] = set()
    for raw in components:
        item = dict(raw)
        source_id = str(item.get("source_model_candidate_id") or "")
        family = str(item.get("model_family") or "")
        feature_set_id = str(item.get("feature_set_id") or "")
        weight = float(item.get("weight") or 0.0)
        hashes = {
            "source_model_manifest_sha256": str(
                item.get("source_model_manifest_sha256") or ""
            ).lower(),
            "source_recipe_sha256": str(item.get("source_recipe_sha256") or "").lower(),
            "source_code_sha256": str(item.get("source_code_sha256") or "").lower(),
            "feature_set_definition_sha256": str(
                item.get("feature_set_definition_sha256") or ""
            ).lower(),
        }
        if (
            not source_id
            or not family
            or not feature_set_id
            or source_id in seen_ids
            or family in seen_families
            or weight <= 0.0
            or any(
                len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
                for value in hashes.values()
            )
        ):
            raise ValueError("champion revalidation component is invalid")
        seen_ids.add(source_id)
        seen_families.add(family)
        normalized.append(
            {
                "source_model_candidate_id": source_id,
                "model_family": family,
                "feature_set_id": feature_set_id,
                "weight": weight,
                **hashes,
            }
        )
    if not normalized or (kind == "model" and len(normalized) != 1):
        raise ValueError("champion revalidation component count is invalid")
    if kind == "ensemble" and not 2 <= len(normalized) <= ENSEMBLE_MAX_MEMBERS:
        raise ValueError("champion revalidation ensemble members are invalid")
    expected_weight = 1.0 / len(normalized)
    if any(abs(float(item["weight"]) - expected_weight) > 1e-12 for item in normalized):
        raise ValueError("champion revalidation must retain equal-rank ensemble weights")
    normalized.sort(key=lambda item: (item["model_family"], item["source_model_candidate_id"]))

    trials = [
        {
            "name": f"champion-revalidation:{item['source_model_candidate_id']}",
            "trial_kind": "model",
            "feature_set_id": item["feature_set_id"],
            "feature_set_definition_sha256": item["feature_set_definition_sha256"],
            "model_family": item["model_family"],
            "spec": {
                "round": "champion_revalidation",
                "source_model_candidate_id": item["source_model_candidate_id"],
                "source_model_manifest_sha256": item["source_model_manifest_sha256"],
                "source_recipe_sha256": item["source_recipe_sha256"],
                "source_code_sha256": item["source_code_sha256"],
                "feature_set_id": item["feature_set_id"],
                "feature_set_definition_sha256": item["feature_set_definition_sha256"],
                "weight": item["weight"],
                "profiles": list(FULL_PROFILES),
                "seeds": list(FULL_SEEDS),
                "fixed_recipe": True,
                "final_oos_opened": False,
                **RESEARCH_SCREENING_MARKERS,
            },
            "resource": {
                "cpu_only": True,
                "evaluation_concurrency_cap": 3,
                "reserved_service_fraction": 0.25,
            },
        }
        for item in normalized
    ]
    manifest = {
        "contract_version": "champion-current-identity-revalidation-v1",
        "stage": "model_screen",
        "dataset_identity_sha256": identity,
        "source_champion": {
            "kind": kind,
            "candidate_id": str(champion["candidate_id"]),
            "manifest_sha256": str(champion.get("manifest_sha256") or "").lower(),
            "admission_evidence_sha256": str(
                champion.get("admission_evidence_sha256") or ""
            ).lower(),
        },
        **source_hashes,
        "components": normalized,
        "trial_count": len(trials),
        "trials": trials,
        "selection": {
            "kind": "fixed_prior_champion_only",
            "same_recipe_only": True,
            "same_feature_set_only": True,
            "same_ensemble_members_only": kind == "ensemble",
            "final_oos_opened": False,
        },
        **RESEARCH_SCREENING_MARKERS,
    }
    for key in ("manifest_sha256", "admission_evidence_sha256"):
        value = str(manifest["source_champion"].get(key) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("champion revalidation source champion is invalid")
    return manifest


class ResearchTournamentStore:
    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    @staticmethod
    def _settle_operational_source(
        connection: Any,
        *,
        source_tournament_id: str,
        operational_trial_ids: Sequence[str],
    ) -> None:
        """Lock every owner and freeze a quiescent operational source.

        The source ledger is not immutable merely because one trial failed.
        Other source branches may still be executing and may legitimately
        append their terminal evidence.  Lock the tournament first, then each
        bound branch, ResearchRun, and job so a concurrent retry cannot reopen
        old work between this check and successor preregistration.  A live
        ``planned``/``running`` tournament whose owners are all terminal is
        atomically settled as ``blocked``; this is the normal recovery shape
        after an evaluator returned an operational result.
        """

        source = connection.execute(
            select(research_tournaments)
            .where(research_tournaments.c.id == source_tournament_id)
            .with_for_update()
        ).first()
        if source is None:
            raise KeyError(source_tournament_id)
        source_status = str(source.status)
        if source_status not in _REMEDIABLE_TOURNAMENT_STATUSES:
            raise ValueError(
                "operational remediation source tournament is not remediable"
            )
        normalized_operational_ids = sorted(
            {str(item) for item in operational_trial_ids if str(item)}
        )
        if not normalized_operational_ids:
            raise ValueError(
                "operational remediation source has no frozen operational trials"
            )
        frozen_operational_rows = connection.execute(
            select(
                research_tournament_trials.c.id,
                research_tournament_trials.c.status,
            )
            .where(
                research_tournament_trials.c.tournament_id
                == source_tournament_id,
                research_tournament_trials.c.id.in_(normalized_operational_ids),
            )
            .order_by(research_tournament_trials.c.id)
            .with_for_update()
        ).all()
        if (
            [str(item.id) for item in frozen_operational_rows]
            != normalized_operational_ids
            or any(str(item.status) != "failed" for item in frozen_operational_rows)
        ):
            raise ValueError(
                "operational remediation source trial settlement changed"
            )

        cycle_branches = connection.execute(
            select(autopilot_branches)
            .where(autopilot_branches.c.cycle_id == str(source.cycle_id))
            .order_by(autopilot_branches.c.id)
            .with_for_update()
        ).all()
        bound_branches = []
        for branch in cycle_branches:
            details = dict(branch.details_json or {})
            bound_tournament_ids = {
                str(details.get("tournament_id") or ""),
                str(details.get("research_tournament_id") or ""),
            }
            if source_tournament_id in bound_tournament_ids:
                bound_branches.append(branch)
        if not bound_branches:
            raise ValueError(
                "operational remediation source tournament has no bound branches"
            )

        for branch in bound_branches:
            if str(branch.status) not in _TERMINAL_BRANCH_STATUSES:
                raise ValueError(
                    "operational remediation source branch is not terminal"
                )
            run_id = str(branch.research_run_id or "")
            branch_job_id = str(branch.job_id or "")
            if not run_id or not branch_job_id:
                raise ValueError(
                    "operational remediation source branch ownership is incomplete"
                )
            run = connection.execute(
                select(research_runs)
                .where(research_runs.c.id == run_id)
                .with_for_update()
            ).first()
            if run is None:
                raise ValueError(
                    "operational remediation source research run is missing"
                )
            if str(run.status) not in _TERMINAL_RESEARCH_RUN_STATUSES:
                raise ValueError(
                    "operational remediation source research run is not terminal"
                )
            current_job_id = str(run.job_id or "")
            if not current_job_id:
                raise ValueError(
                    "operational remediation source research run has no bound job"
                )
            required_job_ids = {branch_job_id, current_job_id}
            related_jobs = connection.execute(
                select(jobs)
                .where(
                    or_(
                        jobs.c.id.in_(sorted(required_job_ids)),
                        jobs.c.payload_json["research_run_id"].as_string() == run_id,
                    )
                )
                .order_by(jobs.c.id)
                .with_for_update()
            ).all()
            observed_job_ids = {str(item.id) for item in related_jobs}
            if not required_job_ids.issubset(observed_job_ids):
                raise ValueError("operational remediation source job is missing")
            if any(
                str(item.status) not in _TERMINAL_JOB_STATUSES
                for item in related_jobs
            ):
                raise ValueError(
                    "operational remediation source job is not terminal"
                )

        if source_status in {"planned", "running"}:
            prior_evidence = dict(source.multiple_testing_json or {})
            prior_evidence_sha256 = str(source.multiple_testing_sha256 or "") or None
            if prior_evidence and (
                prior_evidence_sha256 != canonical_sha256(prior_evidence)
            ):
                raise ValueError(
                    "operational remediation source tournament evidence changed"
                )
            settlement = {
                "contract_version": "model-tournament-operational-settlement-v1",
                "reason_code": "operational_execution_failure",
                "source_status": source_status,
                "operational_trial_ids": normalized_operational_ids,
                "all_bound_branches_runs_and_jobs_terminal": True,
                "performance_triggered": False,
                "prior_tournament_evidence": prior_evidence or None,
                "prior_tournament_evidence_sha256": prior_evidence_sha256,
            }
            connection.execute(
                update(research_tournaments)
                .where(
                    research_tournaments.c.id == source_tournament_id,
                    research_tournaments.c.status == source_status,
                )
                .values(
                    status="blocked",
                    multiple_testing_json=settlement,
                    multiple_testing_sha256=canonical_sha256(settlement),
                    updated_at=_utcnow(),
                    finished_at=_utcnow(),
                )
            )

    def ensure_preregistered(
        self,
        *,
        cycle_id: str,
        dataset_identity_sha256: str,
        active_sota_feature_set_id: str | None = None,
    ) -> dict[str, Any]:
        manifest = build_preregistered_manifest(
            active_sota_feature_set_id=active_sota_feature_set_id
        )
        manifest_sha = canonical_sha256(manifest)
        now = _utcnow()
        tournament_id = uuid.uuid4().hex
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(research_tournaments).values(
                        id=tournament_id,
                        cycle_id=cycle_id,
                        stage="feature_screen",
                        dataset_identity_sha256=dataset_identity_sha256,
                        status="planned",
                        manifest_json=manifest,
                        manifest_sha256=manifest_sha,
                        max_trials=(
                            len(manifest["trials"])
                            + int(manifest["dynamic_model_trial_capacity"])
                            + int(manifest["ensemble"]["adaptive_trial_capacity"])
                        ),
                        selected_trial_ids_json=[],
                        created_at=now,
                        updated_at=now,
                    )
                )
                for item in manifest["trials"]:
                    spec = dict(item["spec"])
                    resource = dict(item["resource"])
                    connection.execute(
                        insert(research_tournament_trials).values(
                            id=uuid.uuid4().hex,
                            tournament_id=tournament_id,
                            trial_kind=item["trial_kind"],
                            name=item["name"],
                            feature_set_id=item.get("feature_set_id"),
                            feature_set_definition_sha256=item.get(
                                "feature_set_definition_sha256"
                            ),
                            model_family=item.get("model_family"),
                            status="preregistered",
                            spec_json=spec,
                            spec_sha256=canonical_sha256(spec),
                            resource_json=resource,
                            created_at=now,
                            updated_at=now,
                        )
                    )
        except IntegrityError:
            pass
        return self.get_for_cycle(cycle_id)

    def ensure_operational_remediation_preregistered(
        self,
        *,
        source_tournament_id: str,
        source_executor_version: str,
        target_executor_version: str,
    ) -> dict[str, Any]:
        """Create one immutable, current-contract successor experiment family.

        An execution or resource failure is not an investment result.  The old
        tournament and all of its evidence remain immutable, while every static
        hypothesis is preregistered again with a new trial ID.  No old candidate,
        metric, or selection is inherited because the successor uses the current
        common-window contract and is a new multiple-testing family.
        """

        source_executor = str(source_executor_version).strip()
        target_executor = str(target_executor_version).strip()
        if not source_executor or not target_executor or source_executor == target_executor:
            raise ValueError("operational remediation requires a changed executor version")
        source = self.get_tournament(source_tournament_id)
        if str(source.get("stage") or "") != "feature_screen":
            raise ValueError("only the primary model tournament may open remediation")
        operational_trials = [
            item
            for item in source["trials"]
            if str(item.get("status") or "") == "failed"
            and (
                str((item.get("metrics") or {}).get("reason_code") or "")
                in {
                    "resource_blocked",
                    "operational_failure",
                    "screen_execution_failed",
                    "execution_failed_after_retry",
                }
                or (item.get("evidence") or {}).get("investment_hypothesis_rejected")
                is False
            )
        ]
        if not operational_trials:
            raise ValueError("tournament has no operational trial eligible for remediation")
        operational_ids = sorted(str(item["id"]) for item in operational_trials)
        source_manifest = dict(source.get("manifest") or {})
        frozen_trial_names = {
            str(item.get("name") or "") for item in source_manifest.get("trials") or []
        }
        source_trials = [
            item for item in source["trials"] if str(item.get("name") or "") in frozen_trial_names
        ]
        if not source_trials or len(source_trials) != len(frozen_trial_names):
            raise ValueError("source tournament preregistered family is incomplete")
        remediation = {
            "contract_version": "model-operational-remediation-v1",
            "source_tournament_id": str(source["id"]),
            "source_tournament_manifest_sha256": str(source["manifest_sha256"]),
            "source_executor_version": source_executor,
            "target_executor_version": target_executor,
            "operational_source_trial_ids": operational_ids,
            "performance_triggered": False,
            "same_hypotheses_and_frozen_specs": True,
            "contract_upgrade": True,
            "same_frozen_evaluation_contract": False,
            "all_trials_re_preregistered": True,
            "new_multiple_testing_family": True,
            "final_oos_opened": False,
        }
        manifest = {
            **source_manifest,
            "contract_version": "model-tournament-operational-remediation-v1",
            "feature_set_window_policy": dict(FEATURE_SET_COMMON_WINDOW_POLICY),
            "operational_remediation": remediation,
        }
        manifest_sha = canonical_sha256(manifest)
        now = _utcnow()
        tournament_id = uuid.uuid4().hex
        try:
            with self.engine.begin() as connection:
                self._settle_operational_source(
                    connection,
                    source_tournament_id=str(source["id"]),
                    operational_trial_ids=operational_ids,
                )
                connection.execute(
                    insert(research_tournaments).values(
                        id=tournament_id,
                        cycle_id=str(source["cycle_id"]),
                        # The primary tournament owns ``feature_screen`` and
                        # champion revalidation owns ``model_screen``.  The
                        # complete operational successor therefore uses the
                        # otherwise-unused ``model_full`` ledger stage.
                        stage="model_full",
                        dataset_identity_sha256=str(source["dataset_identity_sha256"]),
                        status="planned",
                        manifest_json=manifest,
                        manifest_sha256=manifest_sha,
                        max_trials=int(source["max_trials"]),
                        selected_trial_ids_json=[],
                        created_at=now,
                        updated_at=now,
                    )
                )
                for source_trial in source_trials:
                    connection.execute(
                        insert(research_tournament_trials).values(
                            id=uuid.uuid4().hex,
                            tournament_id=tournament_id,
                            branch_id=None,
                            trial_kind=str(source_trial["trial_kind"]),
                            name=str(source_trial["name"]),
                            feature_set_id=source_trial.get("feature_set_id"),
                            feature_set_definition_sha256=source_trial.get(
                                "feature_set_definition_sha256"
                            ),
                            model_family=source_trial.get("model_family"),
                            candidate_id=None,
                            status="preregistered",
                            spec_json=dict(source_trial.get("spec") or {}),
                            spec_sha256=str(source_trial["spec_sha256"]),
                            metrics_json=None,
                            evidence_json=None,
                            evidence_sha256=None,
                            resource_json=dict(source_trial.get("resource") or {}),
                            created_at=now,
                            updated_at=now,
                        )
                    )
        except IntegrityError:
            pass
        existing = self.get_for_cycle_stage(str(source["cycle_id"]), "model_full")
        if (
            str(existing.get("manifest_sha256") or "") != manifest_sha
            or str(existing.get("dataset_identity_sha256") or "")
            != str(source["dataset_identity_sha256"])
        ):
            raise ValueError(
                "another operational model remediation is already frozen for this cycle"
            )
        return existing

    def ensure_champion_revalidation_preregistered(
        self,
        *,
        cycle_id: str,
        dataset_identity_sha256: str,
        source_champion: Mapping[str, Any],
        source_selection_evidence_sha256: str,
        source_model_evidence_sha256: str,
        components: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Create the one fixed-recipe daily revalidation ledger idempotently."""

        manifest = build_champion_revalidation_manifest(
            dataset_identity_sha256=dataset_identity_sha256,
            source_champion=source_champion,
            source_selection_evidence_sha256=source_selection_evidence_sha256,
            source_model_evidence_sha256=source_model_evidence_sha256,
            components=components,
        )
        manifest_sha = canonical_sha256(manifest)
        now = _utcnow()
        tournament_id = uuid.uuid4().hex
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(research_tournaments).values(
                        id=tournament_id,
                        cycle_id=cycle_id,
                        # model_screen is intentionally unused by the ordinary
                        # four-library tournament.  It gives revalidation one
                        # immutable ledger without changing the DB enum.
                        stage="model_screen",
                        dataset_identity_sha256=dataset_identity_sha256,
                        status="planned",
                        manifest_json=manifest,
                        manifest_sha256=manifest_sha,
                        max_trials=len(manifest["trials"]) + (
                            1 if manifest["source_champion"]["kind"] == "ensemble" else 0
                        ),
                        selected_trial_ids_json=[],
                        created_at=now,
                        updated_at=now,
                    )
                )
                for item in manifest["trials"]:
                    connection.execute(
                        insert(research_tournament_trials).values(
                            id=uuid.uuid4().hex,
                            tournament_id=tournament_id,
                            trial_kind="model",
                            name=str(item["name"]),
                            feature_set_id=str(item["feature_set_id"]),
                            feature_set_definition_sha256=str(
                                item["feature_set_definition_sha256"]
                            ),
                            model_family=str(item["model_family"]),
                            status="preregistered",
                            spec_json=dict(item["spec"]),
                            spec_sha256=canonical_sha256(dict(item["spec"])),
                            resource_json=dict(item["resource"]),
                            created_at=now,
                            updated_at=now,
                        )
                    )
        except IntegrityError:
            pass
        existing = self.get_for_cycle_stage(cycle_id, "model_screen")
        if (
            str(existing.get("manifest_sha256") or "") != manifest_sha
            or str(existing.get("dataset_identity_sha256") or "")
            != str(dataset_identity_sha256)
        ):
            raise ValueError("champion revalidation family changed after preregistration")
        return existing

    def ensure_quant_preregistered(
        self,
        *,
        parent_tournament_id: str,
        dataset_identity_sha256: str,
        baseline_prediction_champion: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        parent = self.get_tournament(parent_tournament_id)
        manifest = build_quant_preregistered_manifest(
            parent_tournament=parent,
            dataset_identity_sha256=dataset_identity_sha256,
            baseline_prediction_champion=baseline_prediction_champion,
            candidates=candidates,
        )
        manifest_sha = canonical_sha256(manifest)
        now = _utcnow()
        tournament_id = uuid.uuid4().hex
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(research_tournaments).values(
                        id=tournament_id,
                        cycle_id=str(parent["cycle_id"]),
                        stage="quant",
                        dataset_identity_sha256=dataset_identity_sha256,
                        status="running",
                        manifest_json=manifest,
                        manifest_sha256=manifest_sha,
                        max_trials=len(manifest["trials"]),
                        selected_trial_ids_json=[],
                        created_at=now,
                        updated_at=now,
                    )
                )
                for item in manifest["trials"]:
                    spec = dict(item["spec"])
                    resource = dict(item["resource"])
                    connection.execute(
                        insert(research_tournament_trials).values(
                            id=uuid.uuid4().hex,
                            tournament_id=tournament_id,
                            trial_kind="quant_bundle",
                            name=str(item["name"]),
                            feature_set_id=str(spec["feature_set_id"]),
                            feature_set_definition_sha256=str(
                                spec["feature_set_definition_sha256"]
                            ),
                            model_family="fin_quant_challenger",
                            candidate_id=str(item["candidate_id"]),
                            status="queued",
                            spec_json=spec,
                            spec_sha256=canonical_sha256(spec),
                            resource_json=resource,
                            created_at=now,
                            updated_at=now,
                        )
                    )
        except IntegrityError:
            pass
        existing = self.get_for_cycle_stage(str(parent["cycle_id"]), "quant")
        if (
            str(existing["manifest_sha256"]) != manifest_sha
            or str(existing["dataset_identity_sha256"]) != dataset_identity_sha256
        ):
            raise ValueError("fin_quant research family changed after preregistration")
        return existing

    def get_for_cycle(self, cycle_id: str) -> dict[str, Any]:
        return self.get_for_cycle_stage(cycle_id, "feature_screen")

    def get_for_cycle_stage(self, cycle_id: str, stage: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(research_tournaments).where(
                    research_tournaments.c.cycle_id == cycle_id,
                    research_tournaments.c.stage == stage,
                )
            ).first()
            if row is None:
                raise KeyError(f"{cycle_id}:{stage}")
            result = self._decode_tournament(row_dict(row))
            result["trials"] = [
                self._decode_trial(row_dict(item))
                for item in connection.execute(
                    select(research_tournament_trials)
                    .where(research_tournament_trials.c.tournament_id == row.id)
                    .order_by(research_tournament_trials.c.created_at)
                )
            ]
        return result

    def get_tournament(self, tournament_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(research_tournaments).where(
                    research_tournaments.c.id == tournament_id
                )
            ).first()
            if row is None:
                raise KeyError(tournament_id)
            result = self._decode_tournament(row_dict(row))
            result["trials"] = [
                self._decode_trial(row_dict(item))
                for item in connection.execute(
                    select(research_tournament_trials)
                    .where(research_tournament_trials.c.tournament_id == row.id)
                    .order_by(research_tournament_trials.c.created_at)
                )
            ]
        return result

    def list_trials(self, cycle_id: str) -> list[dict[str, Any]]:
        return self.get_for_cycle(cycle_id)["trials"]

    def mark_running(self, tournament_id: str) -> None:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_tournaments)
                .where(research_tournaments.c.id == tournament_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(tournament_id)
            if str(row.status) in {"succeeded", "failed", "blocked"}:
                return
            connection.execute(
                update(research_tournaments)
                .where(research_tournaments.c.id == tournament_id)
                .values(status="running", updated_at=_utcnow())
            )

    def complete_selection(
        self,
        tournament_id: str,
        *,
        selected_trial_ids: list[str],
        multiple_testing: dict[str, Any],
    ) -> None:
        if not selected_trial_ids or len(set(selected_trial_ids)) != len(
            selected_trial_ids
        ):
            raise ValueError("tournament selection must contain unique trial ids")
        now = _utcnow()
        with self.engine.begin() as connection:
            tournament = connection.execute(
                select(research_tournaments)
                .where(research_tournaments.c.id == tournament_id)
                .with_for_update()
            ).first()
            if tournament is None:
                raise KeyError(tournament_id)
            selected = connection.execute(
                select(
                    research_tournament_trials.c.id,
                    research_tournament_trials.c.status,
                ).where(
                    research_tournament_trials.c.tournament_id == tournament_id,
                    research_tournament_trials.c.id.in_(selected_trial_ids),
                )
            ).all()
            if len(selected) != len(selected_trial_ids) or any(
                str(item.status) not in {"passed", "selected"} for item in selected
            ):
                raise ValueError("only passed preregistered trials may be selected")
            evidence_sha = canonical_sha256(multiple_testing)
            if str(tournament.status) == "succeeded":
                if (
                    list(tournament.selected_trial_ids_json or [])
                    != selected_trial_ids
                    or str(tournament.multiple_testing_sha256) != evidence_sha
                ):
                    raise ValueError("completed tournament selection is immutable")
                return
            connection.execute(
                update(research_tournaments)
                .where(research_tournaments.c.id == tournament_id)
                .values(
                    status="succeeded",
                    selected_trial_ids_json=selected_trial_ids,
                    multiple_testing_json=multiple_testing,
                    multiple_testing_sha256=evidence_sha,
                    updated_at=now,
                    finished_at=now,
                )
            )

    def complete_quant_screening(
        self,
        tournament_id: str,
        *,
        outcomes: Sequence[Mapping[str, Any]],
        screening_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Settle every preregistered fin_quant trial, including failures."""

        evidence = dict(screening_evidence)
        evidence_sha = canonical_sha256(
            {key: value for key, value in evidence.items() if key != "evidence_sha256"}
        )
        if (
            evidence.get("contract_version")
            != "fin-quant-research-screening-ledger-v1"
            or evidence.get("evidence_sha256") != evidence_sha
            or evidence.get("research_screening_only") is not True
            or evidence.get("not_capital_confirmation") is not True
            or evidence.get("cross_cycle_fwer_claimed") is not False
            or evidence.get("final_oos_opened") is not False
        ):
            raise ValueError("fin_quant screening evidence is invalid")
        outcome_by_candidate: dict[str, dict[str, Any]] = {}
        for raw in outcomes:
            outcome = dict(raw)
            candidate_id = str(outcome.get("candidate_id") or "")
            status = str(outcome.get("status") or "")
            if (
                not candidate_id
                or candidate_id in outcome_by_candidate
                or status not in {"passed", "rejected", "failed"}
            ):
                raise ValueError("fin_quant terminal trial outcome is invalid")
            outcome_by_candidate[candidate_id] = outcome
        now = _utcnow()
        with self.engine.begin() as connection:
            tournament = connection.execute(
                select(research_tournaments)
                .where(research_tournaments.c.id == tournament_id)
                .with_for_update()
            ).first()
            if tournament is None:
                raise KeyError(tournament_id)
            if str(tournament.stage) != "quant":
                raise ValueError("only a quant tournament may settle quant trials")
            trials = connection.execute(
                select(research_tournament_trials)
                .where(research_tournament_trials.c.tournament_id == tournament_id)
                .order_by(research_tournament_trials.c.created_at)
                .with_for_update()
            ).all()
            trial_by_candidate = {str(item.candidate_id): item for item in trials}
            if (
                not trials
                or len(trial_by_candidate) != len(trials)
                or set(trial_by_candidate) != set(outcome_by_candidate)
                or {
                    str(item.get("candidate_id") or "")
                    for item in evidence.get("outcomes") or []
                }
                != set(outcome_by_candidate)
            ):
                raise ValueError("fin_quant settlement changed the preregistered family")
            passed_trial_ids = sorted(
                str(trial_by_candidate[candidate_id].id)
                for candidate_id, outcome in outcome_by_candidate.items()
                if outcome["status"] == "passed"
            )
            if str(tournament.status) == "succeeded":
                if (
                    str(tournament.multiple_testing_sha256) != evidence_sha
                    or sorted(tournament.selected_trial_ids_json or [])
                    != passed_trial_ids
                ):
                    raise ValueError("completed fin_quant screening is immutable")
            else:
                if str(tournament.status) not in {"planned", "running"}:
                    raise ValueError("fin_quant tournament is already terminal")
                for candidate_id, outcome in outcome_by_candidate.items():
                    trial = trial_by_candidate[candidate_id]
                    if str(trial.status) in {"passed", "rejected", "failed"}:
                        raise ValueError("fin_quant trial was already settled")
                    trial_evidence = dict(outcome.get("evidence") or {})
                    if (
                        trial_evidence.get("research_screening_only") is not True
                        or trial_evidence.get("not_capital_confirmation") is not True
                        or trial_evidence.get("cross_cycle_fwer_claimed") is not False
                        or trial_evidence.get("final_oos_opened") is not False
                    ):
                        raise ValueError("fin_quant trial evidence has capital semantics")
                    connection.execute(
                        update(research_tournament_trials)
                        .where(research_tournament_trials.c.id == trial.id)
                        .values(
                            status=str(outcome["status"]),
                            metrics_json=dict(outcome.get("metrics") or {}),
                            evidence_json=trial_evidence,
                            evidence_sha256=canonical_sha256(trial_evidence),
                            updated_at=now,
                        )
                    )
                connection.execute(
                    update(research_tournaments)
                    .where(research_tournaments.c.id == tournament_id)
                    .values(
                        status="succeeded",
                        selected_trial_ids_json=passed_trial_ids,
                        multiple_testing_json=evidence,
                        multiple_testing_sha256=evidence_sha,
                        updated_at=now,
                        finished_at=now,
                    )
                )
        return self.get_tournament(tournament_id)

    def block(self, tournament_id: str, *, reason: str) -> None:
        reason_value = str(reason).strip()
        if not reason_value:
            raise ValueError("tournament block reason is required")
        now = _utcnow()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_tournaments)
                .where(research_tournaments.c.id == tournament_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(tournament_id)
            if str(row.status) == "succeeded":
                raise ValueError("a completed tournament cannot be blocked")
            evidence = {
                "contract_version": "model-tournament-block-v1",
                "reason": reason_value,
            }
            connection.execute(
                update(research_tournaments)
                .where(research_tournaments.c.id == tournament_id)
                .values(
                    status="blocked",
                    multiple_testing_json=evidence,
                    multiple_testing_sha256=canonical_sha256(evidence),
                    updated_at=now,
                    finished_at=now,
                )
            )

    def transition_trial(
        self,
        trial_id: str,
        status: str,
        *,
        candidate_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        allowed = {
            "preregistered": {"queued", "rejected"},
            "queued": {"running", "failed", "rejected"},
            "running": {"passed", "failed", "rejected"},
            "passed": {"selected", "rejected"},
            "failed": set(),
            "rejected": set(),
            "selected": set(),
        }
        now = _utcnow()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(research_tournament_trials)
                .where(research_tournament_trials.c.id == trial_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(trial_id)
            if status not in allowed[str(row.status)]:
                raise ValueError(f"invalid trial transition {row.status} -> {status}")
            values: dict[str, Any] = {"status": status, "updated_at": now}
            if candidate_id is not None:
                values["candidate_id"] = candidate_id
            if metrics is not None:
                values["metrics_json"] = metrics
            if evidence is not None:
                values["evidence_json"] = evidence
                values["evidence_sha256"] = canonical_sha256(evidence)
            connection.execute(
                update(research_tournament_trials)
                .where(research_tournament_trials.c.id == trial_id)
                .values(**values)
            )
        return self.get_trial(trial_id)

    def get_trial(self, trial_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(research_tournament_trials).where(
                    research_tournament_trials.c.id == trial_id
                )
            ).first()
        if row is None:
            raise KeyError(trial_id)
        return self._decode_trial(row_dict(row))

    def register_dynamic_model_trial(
        self,
        *,
        tournament_id: str,
        name: str,
        feature_set_id: str,
        feature_set_definition_sha256: str,
        model_family: str,
        candidate_id: str,
        spec: Mapping[str, Any],
        resource: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind a generated RD-Agent model before independent evaluation.

        The outer tournament preregisters a bounded number of dynamic slots.
        A slot becomes a counted hypothesis as soon as executable code exists,
        before Qlib evaluates it.  A failed result therefore cannot disappear
        from the family after the fact.
        """

        trial_name = str(name).strip()
        family = str(model_family).strip()
        if not trial_name or not family:
            raise ValueError("dynamic model trial name and family are required")
        spec_value = dict(spec)
        resource_value = dict(resource)
        now = _utcnow()
        trial_id = uuid.uuid4().hex
        try:
            with self.engine.begin() as connection:
                tournament = connection.execute(
                    select(research_tournaments)
                    .where(research_tournaments.c.id == tournament_id)
                    .with_for_update()
                ).first()
                if tournament is None:
                    raise KeyError(tournament_id)
                if str(tournament.status) in {"succeeded", "failed", "blocked"}:
                    raise ValueError("completed tournament cannot accept a new trial")
                feature_rows = {
                    str(item.get("id")): str(item.get("definition_sha256"))
                    for item in (tournament.manifest_json or {}).get(
                        "feature_sets", []
                    )
                }
                if feature_rows.get(feature_set_id) != str(
                    feature_set_definition_sha256
                ):
                    raise ValueError(
                        "dynamic model trial is outside the preregistered feature sets"
                    )
                count = len(
                    connection.execute(
                        select(research_tournament_trials.c.id).where(
                            research_tournament_trials.c.tournament_id
                            == tournament_id
                        )
                    ).all()
                )
                if count >= int(tournament.max_trials):
                    raise ValueError("dynamic model trial capacity is exhausted")
                connection.execute(
                    insert(research_tournament_trials).values(
                        id=trial_id,
                        tournament_id=tournament_id,
                        trial_kind="model",
                        name=trial_name,
                        feature_set_id=feature_set_id,
                        feature_set_definition_sha256=feature_set_definition_sha256,
                        model_family=family,
                        candidate_id=candidate_id,
                        status="queued",
                        spec_json=spec_value,
                        spec_sha256=canonical_sha256(spec_value),
                        resource_json=resource_value,
                        created_at=now,
                        updated_at=now,
                    )
                )
        except IntegrityError as exc:
            with self.engine.connect() as connection:
                row = connection.execute(
                    select(research_tournament_trials).where(
                        research_tournament_trials.c.tournament_id
                        == tournament_id,
                        research_tournament_trials.c.name == trial_name,
                    )
                ).first()
            if row is None:
                raise
            existing = self._decode_trial(row_dict(row))
            if (
                str(existing.get("candidate_id")) != candidate_id
                or str(existing.get("feature_set_definition_sha256"))
                != feature_set_definition_sha256
                or str(existing.get("spec_sha256"))
                != canonical_sha256(spec_value)
            ):
                raise ValueError("dynamic model trial identity changed") from exc
            return existing
        return self.get_trial(trial_id)

    def create_ensemble(
        self,
        *,
        tournament_id: str,
        name: str,
        dataset: str,
        dataset_identity_sha256: str,
        components: list[dict[str, Any]],
        prediction_correlations: list[float],
        correlation_evidence: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        families = [str(item.get("model_family") or "") for item in components]
        if not 2 <= len(components) <= ENSEMBLE_MAX_MEMBERS:
            raise ValueError("an ensemble must contain two or three models")
        if len(set(families)) != len(families) or "" in families:
            raise ValueError("ensemble members must come from distinct model families")
        weights = [float(item.get("weight") or 0.0) for item in components]
        expected = 1.0 / len(components)
        if any(abs(value - expected) > 1e-12 for value in weights):
            raise ValueError("only equal-rank ensemble weights are allowed")
        if any(abs(float(value)) > ENSEMBLE_CORRELATION_LIMIT for value in prediction_correlations):
            raise ValueError("ensemble members are too highly correlated")
        component_ids = [str(item.get("model_candidate_id") or "") for item in components]
        if "" in component_ids or len(set(component_ids)) != len(component_ids):
            raise ValueError("ensemble member candidate identities are invalid")
        for item in components:
            for key in (
                "model_manifest_sha256",
                "model_admission_evidence_sha256",
                "prediction_grid_sha256",
            ):
                value = str(item.get(key) or "").lower()
                if len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise ValueError(f"ensemble member {key} is invalid")
        correlations = [dict(item) for item in (correlation_evidence or [])]
        if correlations and len(correlations) != len(prediction_correlations):
            raise ValueError("ensemble correlation evidence is incomplete")
        if any(
            item.get("passed") is not True
            or float(item.get("maximum_mean_absolute_daily_rank_correlation", 2.0))
            > ENSEMBLE_CORRELATION_LIMIT
            for item in correlations
        ):
            raise ValueError("ensemble correlation evidence did not pass")
        manifest = {
            "contract_version": "quantlab-model-ensemble-v2",
            "dataset_identity_sha256": dataset_identity_sha256,
            "components": components,
            "combiner": "equal_rank",
            "prediction_correlations": prediction_correlations,
            "correlation_evidence": correlations,
            "stacking": False,
        }
        manifest_sha = canonical_sha256(manifest)
        now = _utcnow()
        ensemble_id = uuid.uuid4().hex
        existing_id: str | None = None
        try:
            with self.engine.begin() as connection:
                tournament = connection.execute(
                    select(research_tournaments)
                    .where(research_tournaments.c.id == tournament_id)
                    .with_for_update()
                ).first()
                if tournament is None:
                    raise KeyError(tournament_id)
                if str(tournament.dataset_identity_sha256) != dataset_identity_sha256:
                    raise ValueError("ensemble belongs to another tournament dataset")
                if str(tournament.status) in {"failed", "blocked"}:
                    raise ValueError("a failed or blocked tournament cannot add ensembles")
                member_trials = connection.execute(
                    select(
                        research_tournament_trials.c.candidate_id,
                        research_tournament_trials.c.model_family,
                        research_tournament_trials.c.status,
                    ).where(
                        research_tournament_trials.c.tournament_id == tournament_id,
                        research_tournament_trials.c.candidate_id.in_(component_ids),
                    )
                ).all()
                admitted_members = {
                    (str(item.candidate_id), str(item.model_family))
                    for item in member_trials
                    if str(item.status) in {"passed", "selected"}
                }
                if admitted_members != set(zip(component_ids, families, strict=True)):
                    raise ValueError(
                        "ensemble members are not passed preregistered model trials"
                    )
                existing_named = connection.execute(
                    select(model_ensemble_candidates).where(
                        model_ensemble_candidates.c.tournament_id == tournament_id,
                        model_ensemble_candidates.c.name == name,
                    )
                ).first()
                if existing_named is not None:
                    if str(existing_named.manifest_sha256) != manifest_sha:
                        raise ValueError(
                            "ensemble name is bound to another immutable manifest"
                        )
                    existing_id = str(existing_named.id)
                # Counting rows via result avoids a backend-specific COUNT form and
                # keeps the hard cap close to the immutable insert.
                existing = connection.execute(
                    select(model_ensemble_candidates.c.id).where(
                        model_ensemble_candidates.c.tournament_id == tournament_id
                    )
                ).all()
                if existing_id is None and len(existing) >= ENSEMBLE_MAX_CANDIDATES:
                    raise ValueError("ensemble candidate limit is exhausted")
                if existing_id is None:
                    trial_count = len(
                        connection.execute(
                            select(research_tournament_trials.c.id).where(
                                research_tournament_trials.c.tournament_id == tournament_id
                            )
                        ).all()
                    )
                    if trial_count >= int(tournament.max_trials):
                        raise ValueError("tournament adaptive trial capacity is exhausted")
                    connection.execute(
                        insert(model_ensemble_candidates).values(
                            id=ensemble_id,
                            tournament_id=tournament_id,
                            name=name,
                            status="awaiting_evaluation",
                            dataset=dataset,
                            dataset_identity_sha256=dataset_identity_sha256,
                            components_json=components,
                            combiner="equal_rank",
                            manifest_json=manifest,
                            manifest_sha256=manifest_sha,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    trial_spec = {
                        "round": "model_ensemble",
                        "generation_rule": "bounded_cross_family_equal_rank",
                        "ensemble_manifest_sha256": manifest_sha,
                        "components": components,
                        "correlation_evidence_sha256": canonical_sha256(correlations),
                        "profiles": list(FULL_PROFILES),
                        "seeds": list(FULL_SEEDS),
                        "stacking": False,
                    }
                    connection.execute(
                        insert(research_tournament_trials).values(
                            id=uuid.uuid4().hex,
                            tournament_id=tournament_id,
                            trial_kind="model_ensemble",
                            name=f"ensemble:{name}",
                            candidate_id=ensemble_id,
                            status="preregistered",
                            spec_json=trial_spec,
                            spec_sha256=canonical_sha256(trial_spec),
                            resource_json={
                                "cpu_only": True,
                                "evaluation_concurrency_cap": 3,
                                "training_required": False,
                            },
                            created_at=now,
                            updated_at=now,
                        )
                    )
            return self.get_ensemble(existing_id or ensemble_id)
        except IntegrityError as exc:
            raise ValueError("model ensemble already exists") from exc

    def mark_ensemble_evaluating(self, ensemble_id: str) -> dict[str, Any]:
        now = _utcnow()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(model_ensemble_candidates)
                .where(model_ensemble_candidates.c.id == ensemble_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(ensemble_id)
            if str(row.status) not in {"research_admitted", "rejected", "invalidated"}:
                connection.execute(
                    update(model_ensemble_candidates)
                    .where(model_ensemble_candidates.c.id == ensemble_id)
                    .values(status="evaluating", updated_at=now)
                )
                trial = connection.execute(
                    select(research_tournament_trials).where(
                        research_tournament_trials.c.tournament_id == row.tournament_id,
                        research_tournament_trials.c.candidate_id == ensemble_id,
                        research_tournament_trials.c.trial_kind == "model_ensemble",
                    )
                ).first()
                if trial is not None and str(trial.status) == "preregistered":
                    connection.execute(
                        update(research_tournament_trials)
                        .where(research_tournament_trials.c.id == trial.id)
                        .values(status="queued", updated_at=now)
                    )
        return self.get_ensemble(ensemble_id)

    def mark_ensemble_failed(
        self,
        ensemble_id: str,
        *,
        reason: str,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        failure = {
            "contract_version": "model-ensemble-failure-v1",
            "reason": str(reason or "ensemble evaluation failed")[:3000],
            "evidence": dict(evidence or {}),
            **RESEARCH_SCREENING_MARKERS,
        }
        failure["evidence_sha256"] = canonical_sha256(failure)
        now = _utcnow()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(model_ensemble_candidates)
                .where(model_ensemble_candidates.c.id == ensemble_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(ensemble_id)
            if str(row.status) == "research_admitted":
                raise ValueError("an admitted ensemble cannot be rejected")
            if str(row.status) == "rejected":
                recorded = dict(row.admission_evidence_json or {})
                if recorded != failure:
                    raise ValueError("ensemble failure evidence is immutable")
            else:
                connection.execute(
                    update(model_ensemble_candidates)
                    .where(model_ensemble_candidates.c.id == ensemble_id)
                    .values(
                        status="rejected",
                        admission_evidence_json=failure,
                        admission_evidence_sha256=failure["evidence_sha256"],
                        updated_at=now,
                    )
                )
                trial = connection.execute(
                    select(research_tournament_trials).where(
                        research_tournament_trials.c.tournament_id == row.tournament_id,
                        research_tournament_trials.c.candidate_id == ensemble_id,
                        research_tournament_trials.c.trial_kind == "model_ensemble",
                    )
                ).first()
                if trial is not None and str(trial.status) in {
                    "preregistered",
                    "queued",
                    "running",
                }:
                    connection.execute(
                        update(research_tournament_trials)
                        .where(research_tournament_trials.c.id == trial.id)
                        .values(
                            status="failed",
                            evidence_json=failure,
                            evidence_sha256=failure["evidence_sha256"],
                            updated_at=now,
                        )
                    )
        return self.get_ensemble(ensemble_id)

    def record_ensemble_evaluation(
        self,
        ensemble_id: str,
        *,
        profile_id: str,
        seed: int,
        metrics: dict[str, Any],
        passed: bool,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        now = _utcnow()
        with self.engine.begin() as connection:
            connection.execute(
                insert(model_ensemble_evaluations).values(
                    id=uuid.uuid4().hex,
                    model_ensemble_candidate_id=ensemble_id,
                    profile_id=profile_id,
                    seed=seed,
                    metrics_json=metrics,
                    gate_status="passed" if passed else "failed",
                    evidence_json=evidence,
                    evidence_sha256=canonical_sha256(evidence),
                    created_at=now,
                )
            )
            evaluations = connection.execute(
                select(model_ensemble_evaluations.c.gate_status).where(
                    model_ensemble_evaluations.c.model_ensemble_candidate_id == ensemble_id
                )
            ).all()
            status = "evaluating"
            if len(evaluations) >= len(FULL_PROFILES) * len(FULL_SEEDS):
                status = (
                    "research_admitted"
                    if all(str(item.gate_status) == "passed" for item in evaluations)
                    else "rejected"
                )
            connection.execute(
                update(model_ensemble_candidates)
                .where(model_ensemble_candidates.c.id == ensemble_id)
                .values(status=status, updated_at=now)
            )
        return self.get_ensemble(ensemble_id)

    def ingest_ensemble_evaluation_result(
        self,
        ensemble_id: str,
        *,
        evaluation: Mapping[str, Any],
        result_artifact_path: str | Path,
    ) -> dict[str, Any]:
        """Ingest one immutable three-profile/three-seed ensemble envelope."""

        result_path = Path(result_artifact_path).resolve()
        if not result_path.is_file():
            raise ValueError("ensemble evaluation result artifact is missing")
        result_sha = file_sha256(result_path)
        reported_status = str(evaluation.get("status") or "")
        if reported_status == "failed":
            return self.mark_ensemble_failed(
                ensemble_id,
                reason=str(evaluation.get("error") or "ensemble evaluation failed"),
                evidence={
                    "result_artifact_path": str(result_path),
                    "result_artifact_sha256": result_sha,
                },
            )
        if reported_status not in {"passed", "rejected"}:
            raise ValueError("ensemble evaluation has no terminal result")
        evidence = evaluation.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("ensemble evaluation evidence is missing")
        evidence_value = dict(evidence)
        expected_evidence_sha = canonical_sha256(
            {key: value for key, value in evidence_value.items() if key != "evidence_sha256"}
        )
        if (
            evidence_value.get("evidence_sha256") != expected_evidence_sha
            or evaluation.get("evidence_sha256") != expected_evidence_sha
        ):
            raise ValueError("ensemble evaluation evidence hash is invalid")
        now = _utcnow()
        with self.engine.begin() as connection:
            row = connection.execute(
                select(model_ensemble_candidates)
                .where(model_ensemble_candidates.c.id == ensemble_id)
                .with_for_update()
            ).first()
            if row is None:
                raise KeyError(ensemble_id)
            existing_admission = dict(row.admission_evidence_json or {})
            if str(row.status) in {"research_admitted", "rejected"}:
                if (
                    existing_admission.get("result_artifact_sha256") == result_sha
                    and existing_admission.get("independent_evidence_sha256")
                    == expected_evidence_sha
                ):
                    return self.get_ensemble(ensemble_id)
                raise ValueError("ensemble terminal evidence is immutable")
            manifest = dict(row.manifest_json or {})
            if (
                canonical_sha256(manifest) != str(row.manifest_sha256)
                or manifest.get("contract_version") != "quantlab-model-ensemble-v2"
                or manifest.get("dataset_identity_sha256")
                != str(row.dataset_identity_sha256)
                or manifest.get("combiner") != "equal_rank"
                or manifest.get("stacking") is not False
            ):
                raise ValueError("ensemble candidate manifest is invalid")
            if (
                evidence_value.get("contract_version")
                != MODEL_ENSEMBLE_EVALUATION_CONTRACT_VERSION
                or evidence_value.get("source") != "independent_qlib_recompute"
                or evidence_value.get("ensemble_id") != ensemble_id
                or evidence_value.get("ensemble_manifest_sha256")
                != str(row.manifest_sha256)
                or evidence_value.get("dataset_identity_sha256")
                != str(row.dataset_identity_sha256)
                or evidence_value.get("combiner") != "equal_rank"
                or evidence_value.get("stacking") is not False
                or evidence_value.get("final_oos_opened") is not False
            ):
                raise ValueError("ensemble independent evidence contract is invalid")
            environment_path = Path(
                str(evidence_value.get("execution_environment_path") or "")
            ).resolve()
            if (
                not environment_path.is_file()
                or file_sha256(environment_path)
                != str(evidence_value.get("execution_environment_file_sha256") or "")
                or canonical_sha256(
                    dict(evidence_value.get("execution_environment") or {})
                )
                != str(evidence_value.get("execution_environment_sha256") or "")
            ):
                raise ValueError("ensemble execution environment evidence changed")
            component_grids: dict[str, dict[str, Any]] = {}
            component_label_bindings: dict[str, dict[str, Any]] = {}
            for component in row.components_json or []:
                model = connection.execute(
                    select(model_candidates).where(
                        model_candidates.c.id == component["model_candidate_id"]
                    )
                ).first()
                if model is None or str(model.status) != "research_admitted":
                    raise ValueError("ensemble component is no longer research-admitted")
                model_value = row_dict(model)
                grid = prediction_grid_from_admission(model_value)
                if (
                    str(model.manifest_sha256)
                    != str(component.get("model_manifest_sha256") or "")
                    or str(model.admission_evidence_sha256)
                    != str(component.get("model_admission_evidence_sha256") or "")
                    or str(grid["prediction_grid_sha256"])
                    != str(component.get("prediction_grid_sha256") or "")
                ):
                    raise ValueError("ensemble component evidence drifted")
                model_manifest = dict(model.manifest_json or {})
                label_binding = model_manifest.get("research_label_binding")
                if (
                    not isinstance(label_binding, Mapping)
                    or model_manifest.get("research_label_binding_sha256")
                    != label_binding.get("binding_sha256")
                ):
                    raise ValueError("ensemble component has no frozen label binding")
                member_id = str(model.id)
                component_grids[member_id] = grid
                component_label_bindings[member_id] = dict(label_binding)
            label_contract = validate_model_ensemble_label_contract(
                evidence_value.get("ensemble_label_contract") or {},
                member_bindings=component_label_bindings,
            )
            label_identity = dict(label_contract["label_identity"])
            expected_cadence = build_research_execution_cadence_contract(
                str(label_identity["horizon_profile"])
            )
            observed_cadence = validate_research_execution_cadence_contract(
                evidence_value.get("research_execution_cadence") or {},
                expected_horizon_profile=str(label_identity["horizon_profile"]),
            )
            reference_grid = next(iter(component_grids.values()), None)
            if (
                evidence_value.get("ensemble_label_contract_sha256")
                != label_contract["evidence_sha256"]
                or reference_grid is None
                or label_identity["dataset_name"] != str(row.dataset)
                or label_identity["dataset_identity_sha256"]
                != str(row.dataset_identity_sha256)
                or dict(label_identity["periods"])
                != dict(
                    reference_grid["profiles"]["recent_3y"]["periods"]
                )
                or observed_cadence != expected_cadence
                or evidence_value.get("research_execution_cadence_sha256")
                != expected_cadence["evidence_sha256"]
            ):
                raise ValueError("ensemble independent label evidence is invalid")
            profiles = evidence_value.get("profiles")
            if not isinstance(profiles, Mapping) or set(profiles) != set(
                REQUIRED_RESEARCH_PROFILES
            ):
                raise ValueError("ensemble evidence profile grid is incomplete")
            all_cells_passed = True
            cell_rows: list[tuple[str, int, dict[str, Any], bool]] = []
            for profile_id in REQUIRED_RESEARCH_PROFILES:
                profile = profiles[profile_id]
                if not isinstance(profile, Mapping):
                    raise ValueError("ensemble profile evidence is invalid")
                periods = dict(profile.get("periods") or {})
                seeds = profile.get("seeds")
                if not isinstance(seeds, Mapping) or {int(seed) for seed in seeds} != set(
                    REQUIRED_MODEL_SEEDS
                ):
                    raise ValueError("ensemble evidence seed grid is incomplete")
                for component_grid in component_grids.values():
                    if dict(component_grid["profiles"][profile_id]["periods"]) != periods:
                        raise ValueError("ensemble evaluation changed component periods")
                for seed in REQUIRED_MODEL_SEEDS:
                    cell = seeds.get(str(seed), seeds.get(seed))
                    if not isinstance(cell, Mapping):
                        raise ValueError("ensemble cell evidence is missing")
                    cell_value = dict(cell)
                    predictions_path = Path(
                        str(cell_value.get("predictions_path") or "")
                    ).resolve()
                    report_path = Path(
                        str(cell_value.get("portfolio_report_path") or "")
                    ).resolve()
                    if (
                        not predictions_path.is_file()
                        or file_sha256(predictions_path)
                        != str(cell_value.get("predictions_sha256") or "")
                        or not report_path.is_file()
                        or file_sha256(report_path)
                        != str(cell_value.get("portfolio_report_sha256") or "")
                        or cell_value.get("final_oos_opened") is not False
                        or cell_value.get("execution_environment_sha256")
                        != evidence_value.get("execution_environment_sha256")
                    ):
                        raise ValueError("ensemble cell artifacts are missing or changed")
                    coverage = cell_value.get("coverage")
                    if (
                        not isinstance(coverage, Mapping)
                        or coverage.get("coverage_gate_passed") is not True
                        or coverage.get("artifact_sha256")
                        != cell_value.get("predictions_sha256")
                        or coverage.get("test_start") != periods.get("valid_start")
                        or coverage.get("test_end") != periods.get("valid_end")
                    ):
                        raise ValueError("ensemble prediction coverage proof is invalid")
                    member_artifacts = {
                        str(item.get("model_candidate_id") or ""): item
                        for item in cell_value.get("member_prediction_artifacts") or []
                    }
                    if set(member_artifacts) != set(component_grids):
                        raise ValueError("ensemble cell member evidence is incomplete")
                    for member_id, grid in component_grids.items():
                        expected_member = grid["profiles"][profile_id]["seeds"][str(seed)]
                        observed_member = member_artifacts[member_id]
                        if (
                            str(observed_member.get("predictions_path") or "")
                            != str(expected_member["predictions_path"])
                            or str(observed_member.get("predictions_sha256") or "")
                            != str(expected_member["predictions_sha256"])
                        ):
                            raise ValueError("ensemble member prediction artifact changed")
                        member_path = Path(
                            str(observed_member.get("predictions_path") or "")
                        ).resolve()
                        if (
                            not member_path.is_file()
                            or file_sha256(member_path)
                            != str(observed_member.get("predictions_sha256") or "")
                        ):
                            raise ValueError(
                                "ensemble member prediction artifact is missing or changed"
                            )
                    metric_passed = True
                    try:
                        require_model_metric_gate(
                            cell_value.get("metrics") or {},
                            context=f"ensemble {ensemble_id}/{profile_id}/{seed}",
                        )
                    except ValueError:
                        metric_passed = False
                    if (cell_value.get("status") == "passed") != metric_passed:
                        raise ValueError("ensemble metric gate status is inconsistent")
                    all_cells_passed = all_cells_passed and metric_passed
                    cell_rows.append((profile_id, seed, cell_value, metric_passed))
            multiple = evidence_value.get("multiple_testing")
            if not isinstance(multiple, Mapping):
                raise ValueError("ensemble shared multiple-testing evidence is missing")
            returns_path = Path(str(multiple.get("returns_path") or "")).resolve()
            if (
                not returns_path.is_file()
                or file_sha256(returns_path)
                != str(multiple.get("returns_sha256") or "")
                or canonical_sha256(
                    {key: value for key, value in multiple.items() if key != "evidence_sha256"}
                )
                != str(multiple.get("evidence_sha256") or "")
            ):
                raise ValueError("ensemble multiple-testing evidence changed")
            selected_by_multiple = False
            try:
                validate_run_multiple_testing_evidence(
                    multiple, selected_trial_name=ensemble_id
                )
                selected_by_multiple = True
            except ValueError:
                selected_by_multiple = False
            should_pass = all_cells_passed and selected_by_multiple
            if (reported_status == "passed") != should_pass:
                raise ValueError("ensemble terminal gate status is inconsistent")
            existing_cells = connection.execute(
                select(model_ensemble_evaluations).where(
                    model_ensemble_evaluations.c.model_ensemble_candidate_id
                    == ensemble_id
                )
            ).all()
            if existing_cells:
                raise ValueError("ensemble evaluation grid was already partially ingested")
            for profile_id, seed, cell, metric_passed in cell_rows:
                cell_evidence = {
                    **cell,
                    "contract_version": "model-ensemble-cell-evidence-v1",
                    "ensemble_id": ensemble_id,
                    "ensemble_manifest_sha256": str(row.manifest_sha256),
                    "aggregate_evidence_sha256": expected_evidence_sha,
                    "result_artifact_sha256": result_sha,
                    "profile_id": profile_id,
                    "seed": seed,
                }
                connection.execute(
                    insert(model_ensemble_evaluations).values(
                        id=uuid.uuid4().hex,
                        model_ensemble_candidate_id=ensemble_id,
                        profile_id=profile_id,
                        seed=seed,
                        metrics_json=dict(cell["metrics"]),
                        gate_status="passed" if metric_passed else "failed",
                        evidence_json=cell_evidence,
                        evidence_sha256=canonical_sha256(cell_evidence),
                        created_at=now,
                    )
                )
            admission = {
                "contract_version": "model-ensemble-admission-v1",
                "status": reported_status,
                "independent_evidence": evidence_value,
                "independent_evidence_sha256": expected_evidence_sha,
                "result_artifact_path": str(result_path),
                "result_artifact_sha256": result_sha,
                **RESEARCH_SCREENING_MARKERS,
            }
            admission["evidence_sha256"] = canonical_sha256(admission)
            connection.execute(
                update(model_ensemble_candidates)
                .where(model_ensemble_candidates.c.id == ensemble_id)
                .values(
                    status=(
                        "research_admitted" if reported_status == "passed" else "rejected"
                    ),
                    admission_evidence_json=admission,
                    admission_evidence_sha256=admission["evidence_sha256"],
                    updated_at=now,
                )
            )
            trial = connection.execute(
                select(research_tournament_trials).where(
                    research_tournament_trials.c.tournament_id == row.tournament_id,
                    research_tournament_trials.c.candidate_id == ensemble_id,
                    research_tournament_trials.c.trial_kind == "model_ensemble",
                )
            ).first()
            if trial is None or str(trial.status) not in {
                "preregistered",
                "queued",
                "running",
            }:
                raise ValueError("ensemble preregistered trial is missing or closed")
            connection.execute(
                update(research_tournament_trials)
                .where(research_tournament_trials.c.id == trial.id)
                .values(
                    status="passed" if reported_status == "passed" else "rejected",
                    metrics_json={
                        "all_metric_cells_passed": all_cells_passed,
                        "multiple_testing_passed": selected_by_multiple,
                    },
                    evidence_json=admission,
                    evidence_sha256=admission["evidence_sha256"],
                    updated_at=now,
                )
            )
        return self.get_ensemble(ensemble_id)

    def get_ensemble(self, ensemble_id: str, *, verify: bool = False) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(model_ensemble_candidates).where(
                    model_ensemble_candidates.c.id == ensemble_id
                )
            ).first()
            if row is None:
                raise KeyError(ensemble_id)
            result = self._decode_ensemble(row_dict(row))
            result["evaluations"] = [
                self._decode_ensemble_evaluation(row_dict(item))
                for item in connection.execute(
                    select(model_ensemble_evaluations)
                    .where(
                        model_ensemble_evaluations.c.model_ensemble_candidate_id == ensemble_id
                    )
                    .order_by(
                        model_ensemble_evaluations.c.profile_id,
                        model_ensemble_evaluations.c.seed,
                    )
                )
            ]
        if verify:
            if canonical_sha256(result["manifest"]) != str(result["manifest_sha256"]):
                raise ValueError("model ensemble immutable manifest is invalid")
            for evaluation in result["evaluations"]:
                if canonical_sha256(evaluation["evidence"]) != str(
                    evaluation["evidence_sha256"]
                ):
                    raise ValueError("model ensemble cell evidence was changed")
            if result["status"] in {"research_admitted", "rejected"}:
                admission = result.get("admission_evidence")
                if not isinstance(admission, Mapping) or canonical_sha256(
                    {key: value for key, value in admission.items() if key != "evidence_sha256"}
                ) != str(result.get("admission_evidence_sha256") or ""):
                    raise ValueError("model ensemble terminal evidence was changed")
                result_path = Path(str(admission.get("result_artifact_path") or ""))
                if result["status"] == "research_admitted" and (
                    not result_path.is_file()
                    or file_sha256(result_path)
                    != str(admission.get("result_artifact_sha256") or "")
                ):
                    raise ValueError("model ensemble result artifact was changed")
        return result

    def list_ensembles(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(model_ensemble_candidates)
                .order_by(model_ensemble_candidates.c.created_at.desc())
                .limit(min(max(limit, 1), 500))
            ).all()
        return [self._decode_ensemble(row_dict(item)) for item in rows]

    @staticmethod
    def _decode_tournament(value: dict[str, Any]) -> dict[str, Any]:
        value["manifest"] = value.pop("manifest_json")
        value["selected_trial_ids"] = value.pop("selected_trial_ids_json")
        value["multiple_testing"] = value.pop("multiple_testing_json")
        return value

    @staticmethod
    def _decode_trial(value: dict[str, Any]) -> dict[str, Any]:
        value["spec"] = value.pop("spec_json")
        value["metrics"] = value.pop("metrics_json")
        value["evidence"] = value.pop("evidence_json")
        value["resource"] = value.pop("resource_json")
        return value

    @staticmethod
    def _decode_ensemble(value: dict[str, Any]) -> dict[str, Any]:
        value["components"] = value.pop("components_json")
        value["manifest"] = value.pop("manifest_json")
        value["admission_evidence"] = value.pop("admission_evidence_json")
        return value

    @staticmethod
    def _decode_ensemble_evaluation(value: dict[str, Any]) -> dict[str, Any]:
        value["metrics"] = value.pop("metrics_json")
        value["evidence"] = value.pop("evidence_json")
        return value


def spearman_from_rank_vectors(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("prediction rank vectors must have equal non-trivial length")
    n = len(left)
    mean_left = sum(left) / n
    mean_right = sum(right) / n
    numerator = sum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(left, right, strict=True)
    )
    denominator_left = sum((a - mean_left) ** 2 for a in left) ** 0.5
    denominator_right = sum((b - mean_right) ** 2 for b in right) ** 0.5
    if denominator_left == 0 or denominator_right == 0:
        return 1.0
    return numerator / (denominator_left * denominator_right)


def build_equal_rank_ensemble_specs(
    champions: Sequence[Mapping[str, Any]],
    pairwise_correlations: Mapping[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """Build bounded cross-family ensembles without looking at OOS results."""

    by_family: dict[str, Mapping[str, Any]] = {}
    for item in champions:
        family = str(item["model_family"])
        if family in by_family:
            raise ValueError("ensemble input must contain at most one champion per family")
        by_family[family] = item
    ordered = [by_family[key] for key in sorted(by_family)]
    result: list[dict[str, Any]] = []
    for size in range(2, min(ENSEMBLE_MAX_MEMBERS, len(ordered)) + 1):
        for members in itertools.combinations(ordered, size):
            ids = [str(item["id"]) for item in members]
            admissible = True
            correlations: list[float] = []
            for left, right in itertools.combinations(ids, 2):
                value = pairwise_correlations.get((left, right))
                if value is None:
                    value = pairwise_correlations.get((right, left))
                if value is None:
                    raise ValueError(f"missing prediction correlation for {left}/{right}")
                correlations.append(float(value))
                if abs(float(value)) > ENSEMBLE_CORRELATION_LIMIT:
                    admissible = False
            if not admissible:
                continue
            result.append(
                {
                    "name": "equal-rank-" + "-".join(ids),
                    "combiner": "equal_rank",
                    "components": [
                        {
                            "model_candidate_id": str(item["id"]),
                            "model_family": str(item["model_family"]),
                            "weight": 1.0 / len(members),
                        }
                        for item in members
                    ],
                    "prediction_correlations": correlations,
                    "stacking": False,
                }
            )
            if len(result) >= ENSEMBLE_MAX_CANDIDATES:
                return result
    return result
