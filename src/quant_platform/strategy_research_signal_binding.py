from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from quant_platform.model_strategy_contract import normalize_model_signal_config
from quant_platform.strategy_rule_ir import canonical_sha256

STRATEGY_RESEARCH_SIGNAL_BINDING_VERSION = "fin-strategy-signal-binding-v1"

_MODEL_SIGNAL_FIELDS = (
    "signal_source",
    "model_signal_contract_version",
    "model_candidate_id",
    "model_evaluation_id",
    "model_code_sha256",
    "model_recipe_sha256",
    "model_evidence_sha256",
    "feature_set_id",
    "feature_set_definition_sha256",
    "model_primary_profile_id",
    "model_primary_seed",
    "model_refit_policy",
    "model_refit_policy_sha256",
    "quant_bundle_candidate_id",
    "quant_bundle_evaluation_id",
    "quant_bundle_sha256",
    "quant_bundle_factor_contract",
    "quant_bundle_factor_contract_sha256",
    "model_ensemble_candidate_id",
    "model_ensemble_evaluation_id",
    "model_ensemble_manifest_sha256",
    "model_ensemble_evidence_sha256",
    "model_ensemble_combiner",
    "model_ensemble_stacking",
    "model_component_candidate_ids",
    "model_component_families",
)

_MODEL_FACTOR_SOURCE_BINDING = {
    "factor_source_mode": "not_applicable_model_prediction",
    "challenger_weight": 0.0,
    "baseline_definition": None,
    "baseline_definition_sha256": None,
}


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _project_model_signal_config(value: Mapping[str, Any]) -> dict[str, Any]:
    projected = {
        field: deepcopy(value[field])
        for field in _MODEL_SIGNAL_FIELDS
        if value.get(field) is not None
    }
    # A public recipe starts life as a factor-score baseline. Rebinding only
    # the model ids would leave that baseline attached and StrategyStore would
    # (correctly) reject the mixed signal. Freeze the model-only factor-source
    # contract together with the admitted model identity.
    projected.update(_MODEL_FACTOR_SOURCE_BINDING)
    normalized = normalize_model_signal_config(projected)
    if normalized.get("signal_source") != "model_prediction":
        raise ValueError("fin_strategy champion is not a model-prediction signal")
    # ``normalize_model_signal_config`` may add the immutable contract version
    # and default primary-cell/refit policy. Freeze those additions too so a
    # later release cannot reinterpret the same proposal.
    return {
        field: deepcopy(normalized[field])
        for field in _MODEL_SIGNAL_FIELDS
        if normalized.get(field) is not None
    } | deepcopy(_MODEL_FACTOR_SOURCE_BINDING)


def build_strategy_research_signal_binding(
    *,
    horizon_profile: str,
    dataset: str,
    dataset_identity_sha256: str,
    research_feature_set: Mapping[str, Any],
    champion_selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Freeze the score source used by a fin_strategy proposal.

    The control remains the transparent recipe when no independently admitted
    model/ensemble/fin_quant champion exists. A governed champion is accepted
    only through ``AutopilotCompletionService`` selection evidence; RD-Agent
    never supplies candidate ids or evidence hashes itself.
    """

    identity = _require_sha256(
        dataset_identity_sha256, field="fin_strategy dataset identity"
    )
    dataset_name = str(dataset or "").strip()
    feature_set_id = str(research_feature_set.get("id") or "").strip()
    feature_set_sha256 = _require_sha256(
        research_feature_set.get("definition_sha256"),
        field="fin_strategy research feature-set definition",
    )
    features = research_feature_set.get("features")
    if not dataset_name or not feature_set_id or not isinstance(features, Mapping) or not features:
        raise ValueError("fin_strategy research feature set is incomplete")

    if champion_selection is None:
        signal_config = {
            "signal_source": "factor_score",
            "factor_source_mode": "qlib_baseline",
            "challenger_weight": 0.0,
            "feature_set_id": feature_set_id,
            "feature_set_definition_sha256": feature_set_sha256,
        }
        source = "transparent_public_baseline"
        champion_kind = "factor_control"
        champion_candidate_id = None
        selection_sha256 = None
    else:
        evidence = champion_selection.get("champion_selection_evidence")
        selection_sha256 = _require_sha256(
            champion_selection.get("champion_selection_evidence_sha256"),
            field="fin_strategy champion selection evidence",
        )
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("contract_version") != "autopilot-completion-v1"
            or evidence.get("selection_policy_version")
            != "pre-final-incumbent-challenge-equal-profile-v2"
            or canonical_sha256(dict(evidence)) != selection_sha256
            or evidence.get("dataset") != dataset_name
            or evidence.get("dataset_identity_sha256") != identity
            or evidence.get("horizon_profile") != horizon_profile
            or evidence.get("final_oos_opened") is not False
            or evidence.get("research_screening_only") is not True
            or evidence.get("not_capital_confirmation") is not True
            or evidence.get("cross_cycle_fwer_claimed") is not False
        ):
            raise ValueError("fin_strategy champion selection evidence is invalid")
        champion_kind = str(evidence.get("selected_kind") or "")
        if champion_kind not in {"model", "ensemble", "joint"}:
            raise ValueError("fin_strategy champion kind is unsupported")
        champion_candidate_id = str(evidence.get("selected_candidate_id") or "").strip()
        selected_config = evidence.get("selected_strategy_config")
        if not champion_candidate_id or not isinstance(selected_config, Mapping):
            raise ValueError("fin_strategy champion selection is incomplete")
        signal_config = _project_model_signal_config(selected_config)
        selected_signal_id = str(
            signal_config.get("quant_bundle_candidate_id")
            or signal_config.get("model_ensemble_candidate_id")
            or signal_config.get("model_candidate_id")
            or ""
        )
        if selected_signal_id != champion_candidate_id:
            raise ValueError("fin_strategy champion signal identity was substituted")
        source = "autopilot_governed_champion"

    binding = {
        "contract_version": STRATEGY_RESEARCH_SIGNAL_BINDING_VERSION,
        "source": source,
        "horizon_profile": str(horizon_profile),
        "dataset": dataset_name,
        "dataset_identity_sha256": identity,
        "research_feature_set_id": feature_set_id,
        "research_feature_set_definition_sha256": feature_set_sha256,
        "signal_source": str(signal_config["signal_source"]),
        "champion_kind": champion_kind,
        "champion_candidate_id": champion_candidate_id,
        "champion_selection_evidence_sha256": selection_sha256,
        "signal_config": signal_config,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def validate_strategy_research_signal_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("fin_strategy signal binding must be an object")
    fields = {
        "contract_version",
        "source",
        "horizon_profile",
        "dataset",
        "dataset_identity_sha256",
        "research_feature_set_id",
        "research_feature_set_definition_sha256",
        "signal_source",
        "champion_kind",
        "champion_candidate_id",
        "champion_selection_evidence_sha256",
        "signal_config",
        "binding_sha256",
    }
    if set(value) != fields:
        raise ValueError("fin_strategy signal binding contract drifted")
    normalized = deepcopy(dict(value))
    binding_sha256 = _require_sha256(
        normalized.pop("binding_sha256", None), field="fin_strategy signal binding"
    )
    if (
        normalized.get("contract_version")
        != STRATEGY_RESEARCH_SIGNAL_BINDING_VERSION
        or canonical_sha256(normalized) != binding_sha256
    ):
        raise ValueError("fin_strategy signal binding changed after freezing")
    _require_sha256(
        normalized.get("dataset_identity_sha256"), field="fin_strategy dataset identity"
    )
    _require_sha256(
        normalized.get("research_feature_set_definition_sha256"),
        field="fin_strategy research feature-set definition",
    )
    signal_config = normalized.get("signal_config")
    if not isinstance(signal_config, Mapping):
        raise ValueError("fin_strategy signal config is missing")
    source = str(normalized.get("source") or "")
    if source == "transparent_public_baseline":
        expected_signal = {
            "signal_source": "factor_score",
            "factor_source_mode": "qlib_baseline",
            "challenger_weight": 0.0,
            "feature_set_id": normalized.get("research_feature_set_id"),
            "feature_set_definition_sha256": normalized.get(
                "research_feature_set_definition_sha256"
            ),
        }
        if (
            dict(signal_config) != expected_signal
            or normalized.get("signal_source") != "factor_score"
            or normalized.get("champion_kind") != "factor_control"
            or normalized.get("champion_candidate_id") is not None
            or normalized.get("champion_selection_evidence_sha256") is not None
        ):
            raise ValueError("fin_strategy transparent signal binding is invalid")
    elif source == "autopilot_governed_champion":
        projected = _project_model_signal_config(signal_config)
        if (
            projected != dict(signal_config)
            or normalized.get("signal_source") != "model_prediction"
            or normalized.get("champion_kind") not in {"model", "ensemble", "joint"}
            or not str(normalized.get("champion_candidate_id") or "").strip()
        ):
            raise ValueError("fin_strategy governed champion binding is invalid")
        _require_sha256(
            normalized.get("champion_selection_evidence_sha256"),
            field="fin_strategy champion selection evidence",
        )
        signal_id = str(
            projected.get("quant_bundle_candidate_id")
            or projected.get("model_ensemble_candidate_id")
            or projected.get("model_candidate_id")
            or ""
        )
        if signal_id != str(normalized["champion_candidate_id"]):
            raise ValueError("fin_strategy governed champion identity changed")
    else:
        raise ValueError("fin_strategy signal binding source is unsupported")
    normalized["binding_sha256"] = binding_sha256
    return normalized


def signal_config_from_strategy_research_binding(value: Any) -> dict[str, Any]:
    binding = validate_strategy_research_signal_binding(value)
    return deepcopy(dict(binding["signal_config"]))


def require_strategy_research_signal_config(
    value: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a materialized StrategySpec to retain its frozen score source."""

    binding = validate_strategy_research_signal_binding(value)
    expected = signal_config_from_strategy_research_binding(binding)
    # This also rejects a substituted single/ensemble/joint identity even if
    # the attacker leaves every expected field untouched and adds another
    # model identity beside it.
    normalize_model_signal_config(dict(config))
    if any(config.get(field) != expected_value for field, expected_value in expected.items()):
        raise ValueError("StrategySpec differs from its frozen research signal")
    return binding
