"""Immutable score-grid contract for an independently admitted factor SOTA.

The factor library SOTA is research evidence, not a capital strategy.  This
module only projects that evidence into a deterministic Qlib score definition
that ``fin_strategy`` may challenge through its existing policy-only,
full-stack, formal-OOS and paper gates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .research_horizon import canonical_sha256, primary_label_horizon_sessions

FACTOR_SCORE_CHAMPION_CONTRACT_VERSION = "factor-score-champion-v1"
FACTOR_SCORE_FEATURE_SET_CONTRACT_VERSION = (
    "governed-feature-set-v1-factor-score-champion"
)
FACTOR_SCORE_BASELINE_CONTRACT_VERSION = "qlib-six-factor-baseline-v1"


def _sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def build_factor_score_champion_contract(
    *,
    dataset: str,
    dataset_identity_sha256: str,
    horizon_profile: str,
    sota_version_id: str,
    sota_evidence_sha256: str,
    sota_policy_sha256: str,
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze oriented expressions and admitted weights into one score grid."""

    identity = _sha256(dataset_identity_sha256, field="factor champion dataset")
    evidence_sha256 = _sha256(
        sota_evidence_sha256, field="factor champion SOTA evidence"
    )
    policy_sha256 = _sha256(
        sota_policy_sha256, field="factor champion SOTA policy"
    )
    dataset_name = str(dataset or "").strip()
    version_id = str(sota_version_id or "").strip()
    if not dataset_name or not version_id or not members:
        raise ValueError("factor champion identity or members are incomplete")

    raw_weights = [item.get("weight") for item in members]
    if all(value is None for value in raw_weights):
        weights = [1.0 / len(members)] * len(members)
        weight_policy = "equal_weight_when_sota_unspecified"
    elif any(value is None for value in raw_weights):
        raise ValueError("factor champion weights must be all present or all absent")
    else:
        try:
            weights = [float(value) for value in raw_weights]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("factor champion weights must be finite numeric values") from exc
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in weights)
            or abs(sum(weights) - 1.0) > 1e-8
        ):
            raise ValueError("factor champion weights must be positive and sum to one")
        weight_policy = "admitted_sota_weights"

    frozen_members: list[dict[str, Any]] = []
    for expected_rank, (raw, weight) in enumerate(
        zip(members, weights, strict=True)
    ):
        rank = int(raw.get("member_rank", -1))
        candidate_id = str(raw.get("factor_candidate_id") or "").strip()
        definition_id = str(raw.get("factor_definition_id") or "").strip()
        evaluation_id = str(raw.get("factor_evaluation_id") or "").strip()
        expression = str(raw.get("expression") or "").strip()
        direction = int(raw.get("direction") or 0)
        if (
            rank != expected_rank
            or not candidate_id
            or not definition_id
            or not evaluation_id
            or not expression
            or direction not in {-1, 1}
        ):
            raise ValueError("factor champion member identity is invalid")
        feature_id = f"SOTA_{rank:03d}_{definition_id[-8:]}"
        oriented_expression = expression if direction == 1 else f"(-1)*({expression})"
        frozen_members.append(
            {
                "member_rank": rank,
                "feature_id": feature_id,
                "factor_candidate_id": candidate_id,
                "factor_definition_id": definition_id,
                "factor_definition_sha256": _sha256(
                    raw.get("factor_definition_sha256"),
                    field="factor champion definition",
                ),
                "factor_evaluation_id": evaluation_id,
                "factor_evaluation_evidence_sha256": _sha256(
                    raw.get("factor_evaluation_evidence_sha256"),
                    field="factor champion evaluation evidence",
                ),
                "factor_evaluation_dataset_identity_sha256": _sha256(
                    raw.get("factor_evaluation_dataset_identity_sha256"),
                    field="factor champion evaluation dataset",
                ),
                "incremental_evidence_sha256": _sha256(
                    raw.get("incremental_evidence_sha256"),
                    field="factor champion incremental evidence",
                ),
                "candidate_code_sha256": _sha256(
                    raw.get("candidate_code_sha256"),
                    field="factor champion code",
                ),
                "expression_sha256": canonical_sha256(expression),
                "qlib_expression": oriented_expression,
                "direction": direction,
                "weight": float(weight),
            }
        )
    candidate_ids = [item["factor_candidate_id"] for item in frozen_members]
    feature_ids = [item["feature_id"] for item in frozen_members]
    if len(set(candidate_ids)) != len(candidate_ids) or len(set(feature_ids)) != len(
        feature_ids
    ):
        raise ValueError("factor champion members are not unique")

    body = {
        "contract_version": FACTOR_SCORE_CHAMPION_CONTRACT_VERSION,
        "dataset": dataset_name,
        "dataset_identity_sha256": identity,
        "horizon_profile": str(horizon_profile),
        "label_horizon_sessions": primary_label_horizon_sessions(horizon_profile),
        "sota_version_id": version_id,
        "sota_evidence_sha256": evidence_sha256,
        "sota_policy_sha256": policy_sha256,
        "weight_policy": weight_policy,
        "members": frozen_members,
        "authority": "research_only",
        "research_screening_only": True,
        "not_capital_confirmation": True,
        "final_oos_opened": False,
    }
    body["contract_sha256"] = canonical_sha256(body)
    return body


def validate_factor_score_champion_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("factor champion score contract must be an object")
    candidate = deepcopy(dict(value))
    supplied_sha256 = _sha256(
        candidate.pop("contract_sha256", None), field="factor champion contract"
    )
    if (
        candidate.get("contract_version")
        != FACTOR_SCORE_CHAMPION_CONTRACT_VERSION
        or candidate.get("authority") != "research_only"
        or candidate.get("research_screening_only") is not True
        or candidate.get("not_capital_confirmation") is not True
        or candidate.get("final_oos_opened") is not False
        or canonical_sha256(candidate) != supplied_sha256
    ):
        raise ValueError("factor champion score contract is invalid")
    members = candidate.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("factor champion score contract has no members")
    profile = str(candidate.get("horizon_profile") or "")
    if (
        not str(candidate.get("dataset") or "").strip()
        or not str(candidate.get("sota_version_id") or "").strip()
        or int(candidate.get("label_horizon_sessions") or 0)
        != primary_label_horizon_sessions(profile)
        or candidate.get("weight_policy")
        not in {"equal_weight_when_sota_unspecified", "admitted_sota_weights"}
    ):
        raise ValueError("factor champion score contract identity drifted")
    _sha256(candidate.get("dataset_identity_sha256"), field="factor champion dataset")
    _sha256(candidate.get("sota_evidence_sha256"), field="factor champion evidence")
    _sha256(candidate.get("sota_policy_sha256"), field="factor champion policy")
    weights: list[float] = []
    candidate_ids: list[str] = []
    feature_ids: list[str] = []
    required_member_fields = {
        "member_rank",
        "feature_id",
        "factor_candidate_id",
        "factor_definition_id",
        "factor_definition_sha256",
        "factor_evaluation_id",
        "factor_evaluation_evidence_sha256",
        "factor_evaluation_dataset_identity_sha256",
        "incremental_evidence_sha256",
        "candidate_code_sha256",
        "expression_sha256",
        "qlib_expression",
        "direction",
        "weight",
    }
    for expected_rank, member in enumerate(members):
        if not isinstance(member, Mapping) or set(member) != required_member_fields:
            raise ValueError("factor champion member contract drifted")
        definition_id = str(member.get("factor_definition_id") or "")
        expression = str(member.get("qlib_expression") or "").strip()
        if (
            int(member.get("member_rank", -1)) != expected_rank
            or member.get("feature_id")
            != f"SOTA_{expected_rank:03d}_{definition_id[-8:]}"
            or not expression
            or int(member.get("direction") or 0) not in {-1, 1}
        ):
            raise ValueError("factor champion member identity drifted")
        for field in (
            "factor_definition_sha256",
            "factor_evaluation_evidence_sha256",
            "factor_evaluation_dataset_identity_sha256",
            "incremental_evidence_sha256",
            "candidate_code_sha256",
            "expression_sha256",
        ):
            _sha256(member.get(field), field=f"factor champion {field}")
        try:
            weight = float(member["weight"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "factor champion member weight must be a finite numeric value"
            ) from exc
        weights.append(weight)
        candidate_ids.append(str(member.get("factor_candidate_id") or ""))
        feature_ids.append(str(member.get("feature_id") or ""))
    if (
        any(not math.isfinite(value) or value <= 0 for value in weights)
        or abs(sum(weights) - 1.0) > 1e-8
        or "" in candidate_ids
        or len(candidate_ids) != len(set(candidate_ids))
        or len(feature_ids) != len(set(feature_ids))
    ):
        raise ValueError("factor champion member weights or identities are invalid")
    return {**candidate, "contract_sha256": supplied_sha256}


def factor_score_champion_feature_set(contract: Mapping[str, Any]) -> dict[str, Any]:
    frozen = validate_factor_score_champion_contract(contract)
    identity = str(frozen["contract_sha256"])
    definition = {
        "contract_version": FACTOR_SCORE_FEATURE_SET_CONTRACT_VERSION,
        "id": (
            f"factor-score-champion:{frozen['horizon_profile']}:{identity[:24]}"
        ),
        "name": f"{frozen['horizon_profile']} admitted factor score champion",
        "source": str(frozen["sota_version_id"]),
        "features": {
            str(item["feature_id"]): str(item["qlib_expression"])
            for item in frozen["members"]
        },
        "factor_score_weights": {
            str(item["feature_id"]): float(item["weight"])
            for item in frozen["members"]
        },
        "factor_score_champion_contract_sha256": identity,
        "sota_evidence_sha256": str(frozen["sota_evidence_sha256"]),
    }
    definition["definition_sha256"] = canonical_sha256(definition)
    return definition


def factor_score_champion_baseline_definition(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = validate_factor_score_champion_contract(contract)
    return {
        "contract_version": FACTOR_SCORE_BASELINE_CONTRACT_VERSION,
        "frequency": "day",
        "evaluation_api": "qlib.data.D.features",
        "factors": [
            {
                "id": str(item["feature_id"]),
                "weight": float(item["weight"]),
                "qlib_expression": str(item["qlib_expression"]),
            }
            for item in frozen["members"]
        ],
        "preprocessing": [
            "cross_sectional_winsorize_1_99",
            "cross_sectional_zscore",
            "pit_tradability_filter",
        ],
        "neutralization_stage": "build_governed_signal",
    }


def factor_score_champion_signal_config(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = validate_factor_score_champion_contract(contract)
    feature_set = factor_score_champion_feature_set(frozen)
    baseline = factor_score_champion_baseline_definition(frozen)
    return {
        "signal_source": "factor_score",
        "factor_source_mode": "qlib_baseline",
        "challenger_weight": 0.0,
        "feature_set_id": feature_set["id"],
        "feature_set_definition_sha256": feature_set["definition_sha256"],
        "baseline_definition": baseline,
        "baseline_definition_sha256": canonical_sha256(baseline),
        "factor_score_champion_contract": frozen,
        "factor_score_champion_contract_sha256": frozen["contract_sha256"],
    }


__all__ = [
    "FACTOR_SCORE_CHAMPION_CONTRACT_VERSION",
    "build_factor_score_champion_contract",
    "factor_score_champion_baseline_definition",
    "factor_score_champion_feature_set",
    "factor_score_champion_signal_config",
    "validate_factor_score_champion_contract",
]
