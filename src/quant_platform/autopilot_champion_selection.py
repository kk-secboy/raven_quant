from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from .autopilot_completion import AutopilotCompletionService
from .factor_library_store import (
    FactorLibraryStore,
    validate_factor_definition_immutability,
    validate_incremental_evidence,
)
from .factor_score_champion import (
    build_factor_score_champion_contract,
    factor_score_champion_feature_set,
    factor_score_champion_signal_config,
)
from .feature_set_registry import register_feature_set
from .research_horizon import (
    LEGACY_AMBIGUOUS,
    canonical_sha256,
    primary_label_horizon_sessions,
)
from .research_store import FactorGatePolicy, ResearchStore, _evaluation_evidence

_NO_ADMITTED_SIGNAL = "no independently admitted signal matches this dataset identity"


def _validate_factor_sota_member_evaluation(
    *,
    candidate: dict[str, Any],
    evaluation: dict[str, Any],
    dataset: str,
    dataset_identity_sha256: str,
) -> None:
    """Revalidate the terminal factor gate and immutable evaluation seal at use."""

    candidate_id = str(candidate.get("id") or "")
    admission_path = str(candidate.get("admission_path") or "standalone")
    policy = FactorGatePolicy()
    metrics = evaluation.get("metrics")
    expected_policy = asdict(policy)
    if (
        not candidate_id
        or admission_path not in {"standalone", "incremental"}
        or evaluation.get("factor_candidate_id") != candidate_id
        or evaluation.get("dataset") != dataset
        or evaluation.get("dataset_identity_sha256") != dataset_identity_sha256
        or evaluation.get("is_legacy") is not False
        or evaluation.get("evaluator_version") != policy.version
        or evaluation.get("candidate_code_sha256") != candidate.get("code_sha256")
        or evaluation.get("candidate_values_sha256") != candidate.get("values_sha256")
        or evaluation.get("recomputed_values_sha256") != candidate.get("values_sha256")
        or not isinstance(metrics, dict)
        or evaluation.get("metrics_sha256") != canonical_sha256(metrics)
        or evaluation.get("policy_json") != expected_policy
        or evaluation.get("policy_sha256") != canonical_sha256(expected_policy)
    ):
        raise ValueError("exact factor champion member evaluation binding is invalid")

    gate_status, gate_reasons = policy.evaluate(metrics)
    layers = policy.evaluate_layers(metrics)
    admitted = (
        gate_status == "passed"
        if admission_path == "standalone"
        else layers["hard_status"] == "passed" and layers["effect_status"] == "failed"
    )
    if (
        not admitted
        or evaluation.get("gate_status") != gate_status
        or evaluation.get("gate_reasons") != gate_reasons
    ):
        raise ValueError("exact factor champion member evaluation gate is invalid")

    try:
        periods = {
            key: evaluation[key].isoformat()
            for key in (
                "train_start",
                "train_end",
                "valid_start",
                "valid_end",
                "test_start",
                "test_end",
            )
        }
        evidence = _evaluation_evidence(
            candidate_id=candidate_id,
            dataset=dataset,
            dataset_identity_sha256=dataset_identity_sha256,
            periods=periods,
            gate_status=gate_status,
            gate_reasons=gate_reasons,
            evaluator_version=str(evaluation["evaluator_version"]),
            candidate_code_sha256=str(evaluation["candidate_code_sha256"]),
            candidate_values_sha256=str(evaluation["candidate_values_sha256"]),
            submitted_values_sha256=str(evaluation["submitted_values_sha256"]),
            recompute_evidence_sha256=canonical_sha256(
                evaluation["recompute_evidence"]
            ),
            artifact_sha256=str(evaluation["artifact_sha256"] or ""),
            metrics_sha256=str(evaluation["metrics_sha256"]),
            policy_sha256=str(evaluation["policy_sha256"]),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "exact factor champion member evaluation evidence is incomplete"
        ) from exc
    evidence_sha256 = canonical_sha256(evidence)
    if (
        evaluation.get("evidence_sha256") != evidence_sha256
        or evaluation.get("execution_contract_hash") != evidence_sha256
    ):
        raise ValueError("exact factor champion member evaluation evidence is invalid")


class _FactorScoreChampionSelection:
    """Read and validate the exact active factor SOTA without granting authority."""

    def __init__(self, database_url: str) -> None:
        self.research = ResearchStore(database_url)
        self.library = FactorLibraryStore(self.research.engine)

    def select(
        self,
        *,
        dataset: str,
        dataset_identity_sha256: str,
        horizon_profile: str,
        allowed_candidate_ids: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        label_horizon = primary_label_horizon_sessions(horizon_profile)
        matches = [
            item
            for item in self.library.list_sota(limit=1000)
            if item.get("status") == "active"
            and item.get("dataset") == dataset
            and item.get("dataset_identity_sha256") == dataset_identity_sha256
            and item.get("universe") == "cn_all"
            and int(item.get("label_horizon_days") or 0) == label_horizon
        ]
        if not matches:
            raise ValueError(_NO_ADMITTED_SIGNAL)
        if len(matches) != 1:
            raise ValueError("multiple exact factor champions match fin_strategy")
        detail = self.library.get_sota(str(matches[0]["id"]))
        if (
            allowed_candidate_ids is not None
            and str(detail["id"]) not in allowed_candidate_ids
        ):
            raise ValueError(_NO_ADMITTED_SIGNAL)
        evidence = detail.get("evidence")
        policy = detail.get("policy")
        if (
            not isinstance(evidence, dict)
            or not isinstance(policy, dict)
            or canonical_sha256(evidence) != detail.get("evidence_sha256")
            or canonical_sha256(policy) != detail.get("policy_sha256")
            or evidence.get("dataset_identity_sha256") != dataset_identity_sha256
            or evidence.get("research_screening_only") is not True
            or evidence.get("not_capital_confirmation") is not True
            or evidence.get("final_oos_opened") is not False
        ):
            raise ValueError("exact factor champion SOTA evidence is invalid")

        members: list[dict[str, Any]] = []
        for expected_rank, member in enumerate(detail.get("members") or []):
            candidate = self.research.get_candidate(
                str(member["factor_candidate_id"])
            )
            definition = self.library.get_definition(
                str(member["factor_definition_id"])
            )
            evaluation = self.research.get_evaluation(
                str(member["factor_evaluation_id"])
            )
            incremental = dict(member.get("incremental_evidence") or {})
            validate_incremental_evidence(incremental)
            definition_sha256 = validate_factor_definition_immutability(definition)
            _validate_factor_sota_member_evaluation(
                candidate=candidate,
                evaluation=evaluation,
                dataset=dataset,
                dataset_identity_sha256=dataset_identity_sha256,
            )
            if (
                int(member.get("member_rank", -1)) != expected_rank
                or candidate.get("status") != "promoted"
                or candidate.get("factor_definition_id") != definition["id"]
                or candidate.get("promoted_evaluation_id") != evaluation["id"]
                or evaluation.get("factor_candidate_id") != candidate["id"]
                or int(candidate.get("label_horizon_days") or 0) != label_horizon
                or incremental.get("factor_candidate_id") != candidate["id"]
                or incremental.get("candidate_code_sha256")
                != candidate.get("code_sha256")
                or incremental.get("candidate_values_sha256")
                != candidate.get("values_sha256")
                or incremental.get("dataset_identity_sha256")
                != dataset_identity_sha256
                or (incremental.get("evaluation_ids") or {}).get("recent_3y")
                != evaluation["id"]
                or (
                    (incremental.get("profiles") or {}).get("recent_3y") or {}
                ).get("evaluation_evidence_sha256")
                != evaluation.get("evidence_sha256")
                or canonical_sha256(incremental)
                != member.get("incremental_evidence_sha256")
            ):
                raise ValueError("exact factor champion member evidence is invalid")
            members.append(
                {
                    "member_rank": expected_rank,
                    "factor_candidate_id": str(candidate["id"]),
                    "factor_definition_id": str(definition["id"]),
                    "factor_definition_sha256": definition_sha256,
                    "factor_evaluation_id": str(evaluation["id"]),
                    "factor_evaluation_evidence_sha256": str(
                        evaluation.get("evidence_sha256") or ""
                    ),
                    "factor_evaluation_dataset_identity_sha256": str(
                        evaluation.get("dataset_identity_sha256") or ""
                    ),
                    "incremental_evidence_sha256": str(
                        member["incremental_evidence_sha256"]
                    ),
                    "candidate_code_sha256": str(candidate["code_sha256"]),
                    "expression": str(definition["expression"]),
                    "direction": (
                        -1
                        if (evaluation.get("metrics") or {}).get("direction")
                        == "inverted"
                        else 1
                    ),
                    "weight": member.get("weight"),
                }
            )
        contract = build_factor_score_champion_contract(
            dataset=dataset,
            dataset_identity_sha256=dataset_identity_sha256,
            horizon_profile=horizon_profile,
            sota_version_id=str(detail["id"]),
            sota_evidence_sha256=str(detail["evidence_sha256"]),
            sota_policy_sha256=str(detail["policy_sha256"]),
            members=members,
        )
        feature_set = register_feature_set(
            factor_score_champion_feature_set(contract)
        )
        selected_config = factor_score_champion_signal_config(contract)
        selection_evidence = {
            "contract_version": "autopilot-factor-score-champion-selection-v1",
            "selection_policy_version": "exact-dataset-horizon-sota-v1",
            "dataset": dataset,
            "dataset_identity_sha256": dataset_identity_sha256,
            "horizon_profile": horizon_profile,
            "selected_kind": "factor",
            "selected_candidate_id": str(detail["id"]),
            "selected_strategy_config": selected_config,
            "selected_research_feature_set": feature_set,
            "factor_score_champion_contract_sha256": contract[
                "contract_sha256"
            ],
            "final_oos_opened": False,
            "research_screening_only": True,
            "not_capital_confirmation": True,
            "cross_cycle_fwer_claimed": False,
        }
        return {
            "champion_selection_evidence": selection_evidence,
            "champion_selection_evidence_sha256": canonical_sha256(
                selection_evidence
            ),
        }


class AutopilotResearchChampionSelector:
    """Expose the read-only Autopilot champion comparison to ``fin_strategy``.

    The historical completion service also contains the retired Autopilot
    capital workflow.  Runtime callers must not receive that wider service:
    this facade deliberately exposes only research-screening selection.  A
    selected factor/model/ensemble/joint signal remains research evidence and
    gains no StrategyVersion, formal-OOS, approval, or paper authority here.
    """

    def __init__(self, database_url: str) -> None:
        self._selection = AutopilotCompletionService(database_url)
        self._factor_selection = _FactorScoreChampionSelection(database_url)

    def select_champion(
        self,
        *,
        dataset: str,
        dataset_identity_sha256: str,
        allowed_candidate_ids: Iterable[str] | None = None,
        horizon_profile: str = LEGACY_AMBIGUOUS,
    ) -> dict[str, Any]:
        allowed = (
            None
            if allowed_candidate_ids is None
            else frozenset(str(value) for value in allowed_candidate_ids)
        )
        try:
            return self._selection.select_champion(
                dataset=dataset,
                dataset_identity_sha256=dataset_identity_sha256,
                allowed_candidate_ids=allowed,
                horizon_profile=horizon_profile,
            )
        except ValueError as exc:
            if str(exc) != _NO_ADMITTED_SIGNAL:
                raise
        # Factor SOTA and model/bundle trials do not share a comparable score
        # grid.  Preserve the existing model-family selection policy and use
        # the exact admitted factor champion only when that family has no
        # independently admitted signal.  It still enters the same downstream
        # fin_strategy competition and capital gates.
        return self._factor_selection.select(
            dataset=dataset,
            dataset_identity_sha256=dataset_identity_sha256,
            horizon_profile=horizon_profile,
            allowed_candidate_ids=allowed,
        )
