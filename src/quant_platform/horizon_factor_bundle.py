"""Immutable factor-package identity for one governed research horizon.

The existing ``fin_quant`` pipeline already evaluates a frozen Qlib feature
set together with RD-Agent factor additions.  This module gives that exact
composition a first-class identity; it does not introduce another research
or promotion path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .research_horizon import (
    LEGACY_AMBIGUOUS,
    canonical_sha256,
    primary_label_horizon_sessions,
    primary_label_policy_contract,
    research_horizon_contract,
)

HORIZON_FACTOR_BUNDLE_CONTRACT_VERSION = "horizon-factor-bundle-v1"
HORIZON_FACTOR_BUNDLE_COMPOSITION_POLICY = (
    "frozen-base-feature-set-plus-incremental-rdagent-v1"
)


def _sha256(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 digest") from exc
    return normalized


def build_horizon_factor_bundle(
    *,
    feature_set: Mapping[str, Any],
    incremental_factors: Sequence[Mapping[str, Any]],
    research_label_binding: Mapping[str, Any],
    incumbent_prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the complete base-plus-incremental factor package.

    ``incremental_factors`` are proposals until the existing independent
    ``fin_quant`` evaluator admits their enclosing quant bundle.  The contract
    therefore carries research-only authority and explicitly requires the
    incumbent comparison; selecting it later into a StrategyVersion does not
    mutate this identity.
    """

    horizon_profile = str(research_label_binding.get("horizon_profile") or "")
    if not horizon_profile or horizon_profile == LEGACY_AMBIGUOUS:
        raise ValueError("a horizon factor bundle requires an active horizon")
    horizon = research_horizon_contract(horizon_profile)
    try:
        label_horizon_sessions = int(
            research_label_binding["label_horizon_sessions"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("horizon factor bundle label is missing") from exc
    if label_horizon_sessions != primary_label_horizon_sessions(horizon_profile):
        raise ValueError("horizon factor bundle must use the primary horizon label")

    feature_set_id = str(feature_set.get("id") or "").strip()
    feature_set_sha256 = _sha256(
        feature_set.get("definition_sha256"), "feature-set definition"
    )
    features = feature_set.get("features")
    if not feature_set_id or not isinstance(features, Mapping) or not features:
        raise ValueError("horizon factor bundle base feature set is incomplete")
    if (
        research_label_binding.get("feature_set_id") != feature_set_id
        or research_label_binding.get("feature_set_sha256")
        != feature_set_sha256
    ):
        raise ValueError("horizon factor bundle changed its research-window features")

    frozen_factors: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_code: set[str] = set()
    for raw in incremental_factors:
        candidate_id = str(raw.get("candidate_id") or "").strip()
        code_sha256 = _sha256(raw.get("code_sha256"), "incremental factor code")
        if (
            not candidate_id
            or candidate_id in seen_ids
            or code_sha256 in seen_code
        ):
            raise ValueError("horizon factor bundle incremental factors are not unique")
        seen_ids.add(candidate_id)
        seen_code.add(code_sha256)
        frozen_factors.append(
            {"candidate_id": candidate_id, "code_sha256": code_sha256}
        )
    if frozen_factors:
        incumbent = dict(incumbent_prediction or {})
        incumbent_kind = str(incumbent.get("kind") or "").strip()
        incumbent_id = str(incumbent.get("candidate_id") or "").strip()
        incumbent_sha256 = _sha256(
            incumbent.get("evidence_sha256"), "incumbent prediction evidence"
        )
        if incumbent_kind not in {"model", "ensemble"} or not incumbent_id:
            raise ValueError("horizon factor bundle incumbent identity is invalid")
        challenge = {
            "mode": "incremental_challenger",
            "incumbent_kind": incumbent_kind,
            "incumbent_candidate_id": incumbent_id,
            "incumbent_evidence_sha256": incumbent_sha256,
            "comparison": "joint_vs_frozen_incumbent",
            "required_ablations": ["factor_only", "model_only", "joint"],
        }
    else:
        challenge = {
            "mode": "baseline_seed",
            "incumbent_kind": None,
            "incumbent_candidate_id": None,
            "incumbent_evidence_sha256": None,
            "comparison": None,
            "required_ablations": [],
        }

    base_feature_set: dict[str, Any] = {
        "id": feature_set_id,
        "definition_sha256": feature_set_sha256,
        "contract_version": str(feature_set.get("contract_version") or ""),
        "source": str(feature_set.get("source") or ""),
        "feature_count": len(features),
        "feature_names_sha256": canonical_sha256(sorted(map(str, features))),
    }
    if feature_set.get("foundation_feature_set_id") is not None:
        base_feature_set["foundation"] = {
            "feature_set_id": str(feature_set["foundation_feature_set_id"]),
            "definition_sha256": _sha256(
                feature_set.get("foundation_feature_set_sha256"),
                "foundation feature set",
            ),
        }
        base_feature_set["admitted_sota"] = {
            "version_id": str(feature_set.get("sota_version_id") or ""),
            "feature_set_sha256": _sha256(
                feature_set.get("sota_feature_set_sha256"),
                "SOTA feature set",
            ),
            "evidence_sha256": _sha256(
                feature_set.get("sota_evidence_sha256"),
                "SOTA evidence",
            ),
            "composition_sha256": _sha256(
                feature_set.get("composition_sha256"),
                "factor champion composition",
            ),
        }
        if not base_feature_set["foundation"]["feature_set_id"] or not (
            base_feature_set["admitted_sota"]["version_id"]
        ):
            raise ValueError("horizon factor champion provenance is incomplete")

    body: dict[str, Any] = {
        "contract_version": HORIZON_FACTOR_BUNDLE_CONTRACT_VERSION,
        "composition_policy": HORIZON_FACTOR_BUNDLE_COMPOSITION_POLICY,
        "horizon_profile": horizon_profile,
        "horizon_contract_sha256": horizon.sha256,
        "label_horizon_sessions": label_horizon_sessions,
        "primary_label_policy": primary_label_policy_contract(),
        "research_label_binding_sha256": _sha256(
            research_label_binding.get("binding_sha256"),
            "research label binding",
        ),
        "dataset_identity_sha256": _sha256(
            research_label_binding.get("dataset_identity_sha256"),
            "dataset identity",
        ),
        "base_feature_set": base_feature_set,
        "incremental_factors": frozen_factors,
        "incremental_challenge": challenge,
        "authority": "research_only",
        "final_oos_opened": False,
    }
    identity_sha256 = canonical_sha256(body)
    contract = {
        **body,
        "id": f"horizon-factor-bundle:{horizon_profile}:{identity_sha256[:24]}",
    }
    contract["bundle_sha256"] = canonical_sha256(contract)
    return contract


def validate_horizon_factor_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a self-contained bundle identity without consulting mutable state."""

    if not isinstance(value, Mapping):
        raise ValueError("horizon factor bundle must be an object")
    candidate = dict(value)
    recorded_sha256 = _sha256(
        candidate.pop("bundle_sha256", None), "horizon factor bundle"
    )
    if canonical_sha256(candidate) != recorded_sha256:
        raise ValueError("horizon factor bundle digest is invalid")
    if (
        candidate.get("contract_version")
        != HORIZON_FACTOR_BUNDLE_CONTRACT_VERSION
        or candidate.get("composition_policy")
        != HORIZON_FACTOR_BUNDLE_COMPOSITION_POLICY
        or candidate.get("authority") != "research_only"
        or candidate.get("final_oos_opened") is not False
    ):
        raise ValueError("horizon factor bundle policy is invalid")

    profile = str(candidate.get("horizon_profile") or "")
    horizon = research_horizon_contract(profile)
    if (
        profile == LEGACY_AMBIGUOUS
        or candidate.get("horizon_contract_sha256") != horizon.sha256
        or int(candidate.get("label_horizon_sessions") or 0)
        != primary_label_horizon_sessions(profile)
    ):
        raise ValueError("horizon factor bundle horizon identity is invalid")
    if candidate.get("primary_label_policy") != primary_label_policy_contract():
        raise ValueError("horizon factor bundle primary-label policy is invalid")
    expected_id_prefix = f"horizon-factor-bundle:{profile}:"
    identity_body = {
        key: item
        for key, item in candidate.items()
        if key != "id"
    }
    expected_id = expected_id_prefix + canonical_sha256(identity_body)[:24]
    if candidate.get("id") != expected_id:
        raise ValueError("horizon factor bundle id is invalid")

    base = candidate.get("base_feature_set")
    factors = candidate.get("incremental_factors")
    challenge = candidate.get("incremental_challenge")
    if (
        not isinstance(base, Mapping)
        or not str(base.get("id") or "")
        or not _sha256(base.get("definition_sha256"), "base feature set")
        or int(base.get("feature_count") or 0) <= 0
        or not _sha256(base.get("feature_names_sha256"), "base feature names")
        or not isinstance(factors, list)
        or not isinstance(challenge, Mapping)
    ):
        raise ValueError("horizon factor bundle composition is invalid")
    foundation = base.get("foundation")
    admitted_sota = base.get("admitted_sota")
    if bool(foundation is not None) != bool(admitted_sota is not None):
        raise ValueError("horizon factor bundle champion provenance is incomplete")
    if foundation is not None:
        if (
            not isinstance(foundation, Mapping)
            or not str(foundation.get("feature_set_id") or "")
            or not _sha256(
                foundation.get("definition_sha256"), "foundation feature set"
            )
            or not isinstance(admitted_sota, Mapping)
            or not str(admitted_sota.get("version_id") or "")
            or not _sha256(
                admitted_sota.get("feature_set_sha256"), "SOTA feature set"
            )
            or not _sha256(
                admitted_sota.get("evidence_sha256"), "SOTA evidence"
            )
            or not _sha256(
                admitted_sota.get("composition_sha256"),
                "factor champion composition",
            )
        ):
            raise ValueError("horizon factor bundle champion provenance is invalid")
    ids: list[str] = []
    codes: list[str] = []
    for factor in factors:
        if not isinstance(factor, Mapping):
            raise ValueError("horizon factor bundle factor is invalid")
        ids.append(str(factor.get("candidate_id") or ""))
        codes.append(_sha256(factor.get("code_sha256"), "incremental factor code"))
    if "" in ids or len(set(ids)) != len(ids) or len(set(codes)) != len(codes):
        raise ValueError("horizon factor bundle factor identity is invalid")
    if factors:
        if (
            challenge.get("mode") != "incremental_challenger"
            or challenge.get("comparison") != "joint_vs_frozen_incumbent"
            or challenge.get("required_ablations")
            != ["factor_only", "model_only", "joint"]
            or challenge.get("incumbent_kind") not in {"model", "ensemble"}
            or not str(challenge.get("incumbent_candidate_id") or "")
        ):
            raise ValueError("horizon factor bundle challenge is invalid")
        _sha256(challenge.get("incumbent_evidence_sha256"), "incumbent evidence")
    elif dict(challenge) != {
        "mode": "baseline_seed",
        "incumbent_kind": None,
        "incumbent_candidate_id": None,
        "incumbent_evidence_sha256": None,
        "comparison": None,
        "required_ablations": [],
    }:
        raise ValueError("baseline horizon factor bundle cannot claim a challenge")
    _sha256(candidate.get("research_label_binding_sha256"), "label binding")
    _sha256(candidate.get("dataset_identity_sha256"), "dataset identity")
    return {**candidate, "bundle_sha256": recorded_sha256}


__all__ = [
    "HORIZON_FACTOR_BUNDLE_COMPOSITION_POLICY",
    "HORIZON_FACTOR_BUNDLE_CONTRACT_VERSION",
    "build_horizon_factor_bundle",
    "validate_horizon_factor_bundle",
]
