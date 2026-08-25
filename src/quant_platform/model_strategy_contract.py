from __future__ import annotations

from pathlib import Path
from typing import Any

from .model_research_governance import (
    MODEL_PREDICTION_CONTRACT_VERSION,
    MODEL_REFIT_POLICY,
    MODEL_REFIT_POLICY_SHA256,
    PRIMARY_MODEL_PROFILE,
    PRIMARY_MODEL_SEED,
    REQUIRED_MODEL_SEEDS,
    REQUIRED_QUANT_ABLATIONS,
    REQUIRED_RESEARCH_PROFILES,
    RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION,
    canonical_sha256,
    file_sha256,
    is_sha256,
    validate_independent_model_evidence,
    validate_quant_bundle_evidence,
)

MODEL_SIGNAL_CONTRACT_VERSION = "strategy-model-signal-v1"
MODEL_FORMAL_ADMISSION_CONTRACT_VERSION = "formal-model-admission-binding-v1"
MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_STATUS = "not_applicable_model_prediction_independent_grid"
MODEL_FACTOR_VALIDATION_NOT_APPLICABLE_REASON = (
    "model predictions are validated by the sealed independent pre-final grid; "
    "the final OOS is not reopened to fabricate factor-score walk-forward or "
    "component-ablation evidence"
)
QUANT_BUNDLE_FACTOR_CONTRACT_VERSION = "quant-bundle-model-features-v1"
QUANT_BUNDLE_FACTOR_WEIGHT_POLICY = "unit-scale-model-learned-v1"


def normalize_model_signal_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    signal_source = str(result.get("signal_source") or "factor_score")
    if signal_source not in {"factor_score", "model_prediction"}:
        raise ValueError("strategy signal_source must be factor_score or model_prediction")
    result["signal_source"] = signal_source
    if signal_source == "factor_score":
        forbidden = [
            key
            for key in (
                "model_candidate_id",
                "model_evaluation_id",
                "model_code_sha256",
                "model_recipe_sha256",
                "model_evidence_sha256",
                "quant_bundle_candidate_id",
                "quant_bundle_evaluation_id",
                "quant_bundle_sha256",
                "quant_bundle_factor_contract",
                "quant_bundle_factor_contract_sha256",
            )
            if result.get(key) is not None
        ]
        if forbidden:
            raise ValueError("factor-score strategy cannot bind model candidate fields")
        return result

    required_text = ("model_candidate_id", "model_evaluation_id", "feature_set_id")
    if any(not str(result.get(key) or "").strip() for key in required_text):
        raise ValueError("model strategy must bind its candidate, evaluation and feature set")
    for key in (
        "model_code_sha256",
        "model_recipe_sha256",
        "model_evidence_sha256",
        "feature_set_definition_sha256",
    ):
        if not is_sha256(result.get(key)):
            raise ValueError(f"model strategy {key} is missing or invalid")
    result.setdefault("model_primary_profile_id", PRIMARY_MODEL_PROFILE)
    result.setdefault("model_primary_seed", PRIMARY_MODEL_SEED)
    result.setdefault("model_refit_policy", dict(MODEL_REFIT_POLICY))
    result.setdefault("model_refit_policy_sha256", MODEL_REFIT_POLICY_SHA256)
    if (
        result.get("model_primary_profile_id") != PRIMARY_MODEL_PROFILE
        or int(result.get("model_primary_seed") or 0) != PRIMARY_MODEL_SEED
        or result.get("model_refit_policy") != MODEL_REFIT_POLICY
        or result.get("model_refit_policy_sha256") != MODEL_REFIT_POLICY_SHA256
        or canonical_sha256(result["model_refit_policy"])
        != result["model_refit_policy_sha256"]
    ):
        raise ValueError("model strategy primary cell/refit policy is not governed")
    bundle_id = result.get("quant_bundle_candidate_id")
    bundle_evaluation_id = result.get("quant_bundle_evaluation_id")
    bundle_sha256 = result.get("quant_bundle_sha256")
    bundle_factor_contract = result.get("quant_bundle_factor_contract")
    bundle_factor_contract_sha256 = result.get("quant_bundle_factor_contract_sha256")
    if any(value is not None for value in (bundle_id, bundle_evaluation_id, bundle_sha256)):
        if not str(bundle_id or "").strip() or not str(bundle_evaluation_id or "").strip():
            raise ValueError("joint strategy must bind its quant bundle and evaluation")
        if not is_sha256(bundle_sha256):
            raise ValueError("joint strategy quant bundle hash is invalid")
        if bundle_factor_contract is not None or bundle_factor_contract_sha256 is not None:
            if (
                not isinstance(bundle_factor_contract, dict)
                or bundle_factor_contract.get("contract_version")
                != QUANT_BUNDLE_FACTOR_CONTRACT_VERSION
                or bundle_factor_contract.get("bundle_candidate_id") != bundle_id
                or bundle_factor_contract.get("bundle_manifest_sha256") != bundle_sha256
                or bundle_factor_contract.get("weight_policy") != QUANT_BUNDLE_FACTOR_WEIGHT_POLICY
                or not is_sha256(bundle_factor_contract_sha256)
                or canonical_sha256(bundle_factor_contract) != bundle_factor_contract_sha256
            ):
                raise ValueError("joint strategy bundle factor contract is invalid")
    elif bundle_factor_contract is not None or bundle_factor_contract_sha256 is not None:
        raise ValueError("non-joint model strategy cannot carry bundle factors")
    result["model_signal_contract_version"] = MODEL_SIGNAL_CONTRACT_VERSION
    return result


def model_signal_identity(config: dict[str, Any]) -> dict[str, Any] | None:
    normalized = normalize_model_signal_config(config)
    if normalized["signal_source"] != "model_prediction":
        return None
    fields = (
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
        "model_refit_policy_sha256",
        "quant_bundle_candidate_id",
        "quant_bundle_evaluation_id",
        "quant_bundle_sha256",
        "quant_bundle_factor_contract_sha256",
    )
    identity = {key: normalized.get(key) for key in fields if normalized.get(key) is not None}
    return {**identity, "identity_sha256": canonical_sha256(identity)}


def build_model_formal_admission_binding(
    *,
    config: dict[str, Any],
    candidate_manifest_sha256: str,
    dataset_identity_sha256: str,
    pre_final_end: str,
    model_admission_evidence: dict[str, Any],
    model_admission_evidence_sha256: str,
    quant_bundle_manifest_sha256: str | None = None,
    quant_bundle_admission_evidence: dict[str, Any] | None = None,
    quant_bundle_admission_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Reduce independently validated DB evidence to an immutable formal binding.

    The full research grid remains in the governed candidate tables.  A formal
    backtest carries this deterministic digest summary so approval can rebuild
    it from those tables and reject a changed or substituted admission.
    """

    identity = model_signal_identity(config)
    if identity is None:
        raise ValueError("formal model admission requires model_prediction")
    for value, label in (
        (candidate_manifest_sha256, "candidate manifest"),
        (dataset_identity_sha256, "dataset identity"),
        (model_admission_evidence_sha256, "model admission evidence"),
    ):
        if not is_sha256(value):
            raise ValueError(f"formal model {label} SHA-256 is invalid")
    validated_model = validate_independent_model_evidence(
        model_admission_evidence,
        candidate_id=str(identity["model_candidate_id"]),
        dataset_identity_sha256=dataset_identity_sha256,
        pre_final_end=pre_final_end,
    )
    model_multiple_testing = validated_model["multiple_testing"]
    if (
        validated_model["evidence_sha256"] != model_admission_evidence_sha256
        or identity["model_evidence_sha256"] != model_admission_evidence_sha256
    ):
        raise ValueError("formal model admission hash does not match the StrategySpec")

    payload: dict[str, Any] = {
        "contract_version": MODEL_FORMAL_ADMISSION_CONTRACT_VERSION,
        "source": "independent_qlib_recompute",
        "final_oos_opened": False,
        "model_signal_identity_sha256": identity["identity_sha256"],
        "model_candidate_id": identity["model_candidate_id"],
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "dataset_identity_sha256": dataset_identity_sha256,
        "pre_final_end": pre_final_end,
        "model_admission_evidence_sha256": model_admission_evidence_sha256,
        "model_grid": {
            "profiles": list(REQUIRED_RESEARCH_PROFILES),
            "seeds": list(REQUIRED_MODEL_SEEDS),
            "cell_count": len(REQUIRED_RESEARCH_PROFILES) * len(REQUIRED_MODEL_SEEDS),
            "profiles_sha256": canonical_sha256(validated_model["profiles"]),
            "execution_environment_sha256": validated_model[
                "execution_environment_sha256"
            ],
            "multiple_testing_evidence_sha256": model_multiple_testing[
                "evidence_sha256"
            ],
            "selected_trial_name": str(
                validated_model.get("multiple_testing_trial_name")
                or identity["model_candidate_id"]
            ),
            "multiple_testing": {
                key: model_multiple_testing[key]
                for key in (
                    "contract_version",
                    "source",
                    "final_oos_opened",
                    "trial_definitions",
                    "trial_names",
                    "trial_count",
                    "holm_adjusted_p_values",
                    "eligible_trial_names",
                    "pbo",
                    "trial_daily_sharpes",
                    "gate_passed",
                )
            },
        },
        "quant_bundle": None,
    }

    bundle_id = identity.get("quant_bundle_candidate_id")
    if bundle_id is None:
        if any(
            value is not None
            for value in (
                quant_bundle_manifest_sha256,
                quant_bundle_admission_evidence,
                quant_bundle_admission_evidence_sha256,
            )
        ):
            raise ValueError("non-joint model admission cannot carry quant bundle evidence")
    else:
        if (
            not is_sha256(quant_bundle_manifest_sha256)
            or not isinstance(quant_bundle_admission_evidence, dict)
            or not is_sha256(quant_bundle_admission_evidence_sha256)
        ):
            raise ValueError("joint formal admission is missing immutable bundle evidence")
        if canonical_sha256(quant_bundle_admission_evidence) != (
            quant_bundle_admission_evidence_sha256
        ):
            raise ValueError("joint formal admission wrapper SHA-256 is invalid")
        independent_bundle = quant_bundle_admission_evidence.get("independent_bundle")
        if not isinstance(independent_bundle, dict):
            raise ValueError("joint formal admission has no independent bundle")
        validated_bundle = validate_quant_bundle_evidence(
            independent_bundle,
            dataset_identity_sha256=dataset_identity_sha256,
        )
        multiple_testing = validated_bundle["multiple_testing"]
        if (
            str(quant_bundle_admission_evidence.get("candidate_id") or "") != bundle_id
            or quant_bundle_admission_evidence.get("bundle_manifest_sha256")
            != quant_bundle_manifest_sha256
            or quant_bundle_admission_evidence.get("independent_bundle_sha256")
            != validated_bundle["bundle_sha256"]
            or quant_bundle_admission_evidence.get("final_oos_opened") is not False
            or validated_bundle.get("id") != bundle_id
            or identity.get("quant_bundle_sha256") != quant_bundle_manifest_sha256
        ):
            raise ValueError("joint formal admission does not match the frozen bundle")
        payload["quant_bundle"] = {
            "candidate_id": bundle_id,
            "bundle_manifest_sha256": quant_bundle_manifest_sha256,
            "admission_evidence_sha256": quant_bundle_admission_evidence_sha256,
            "independent_bundle_sha256": validated_bundle["bundle_sha256"],
            "ablations": list(REQUIRED_QUANT_ABLATIONS),
            "profiles": list(REQUIRED_RESEARCH_PROFILES),
            "seeds": list(REQUIRED_MODEL_SEEDS),
            "cell_count": (
                len(REQUIRED_QUANT_ABLATIONS)
                * len(REQUIRED_RESEARCH_PROFILES)
                * len(REQUIRED_MODEL_SEEDS)
            ),
            "ablations_sha256": canonical_sha256(validated_bundle["ablations"]),
            "multiple_testing_evidence_sha256": multiple_testing["evidence_sha256"],
            "execution_environment_sha256": validated_bundle[
                "execution_environment_sha256"
            ],
            "multiple_testing": {
                key: multiple_testing[key]
                for key in (
                    "contract_version",
                    "source",
                    "final_oos_opened",
                    "trial_definitions",
                    "trial_names",
                    "trial_count",
                    "holm_adjusted_p_values",
                    "eligible_trial_names",
                    "pbo",
                    "trial_daily_sharpes",
                    "gate_passed",
                )
            },
        }

    payload["binding_sha256"] = canonical_sha256(payload)
    return payload


def validate_model_formal_admission_binding(
    binding: Any,
    *,
    config: dict[str, Any],
    dataset_identity_sha256: str,
    pre_final_end: str,
) -> dict[str, Any]:
    """Validate the digest-only binding carried by a formal backtest manifest."""

    if not isinstance(binding, dict):
        raise ValueError("formal model admission binding is missing")
    identity = model_signal_identity(config)
    if identity is None:
        raise ValueError("factor-score strategy cannot carry model admission evidence")
    payload = {key: value for key, value in binding.items() if key != "binding_sha256"}
    model_grid = binding.get("model_grid")
    model_multiple = model_grid.get("multiple_testing") if isinstance(model_grid, dict) else None
    expected_model_cells = len(REQUIRED_RESEARCH_PROFILES) * len(REQUIRED_MODEL_SEEDS)
    if (
        binding.get("contract_version") != MODEL_FORMAL_ADMISSION_CONTRACT_VERSION
        or binding.get("source") != "independent_qlib_recompute"
        or binding.get("final_oos_opened") is not False
        or binding.get("model_signal_identity_sha256") != identity["identity_sha256"]
        or binding.get("model_candidate_id") != identity["model_candidate_id"]
        or binding.get("dataset_identity_sha256") != dataset_identity_sha256
        or binding.get("pre_final_end") != pre_final_end
        or binding.get("model_admission_evidence_sha256") != identity["model_evidence_sha256"]
        or not is_sha256(binding.get("candidate_manifest_sha256"))
        or not isinstance(model_grid, dict)
        or model_grid.get("profiles") != list(REQUIRED_RESEARCH_PROFILES)
        or model_grid.get("seeds") != list(REQUIRED_MODEL_SEEDS)
        or model_grid.get("cell_count") != expected_model_cells
        or not is_sha256(model_grid.get("profiles_sha256"))
        or not is_sha256(model_grid.get("execution_environment_sha256"))
        or not is_sha256(model_grid.get("multiple_testing_evidence_sha256"))
        or not isinstance(model_multiple, dict)
        or model_multiple.get("contract_version")
        != RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION
        or model_multiple.get("source") != "independent_qlib_recompute"
        or model_multiple.get("final_oos_opened") is not False
        or model_multiple.get("trial_count")
        != len(model_multiple.get("trial_names") or [])
        or len(model_multiple.get("holm_adjusted_p_values") or [])
        != model_multiple.get("trial_count")
        or len(model_multiple.get("trial_daily_sharpes") or [])
        != model_multiple.get("trial_count")
        or not str(model_grid.get("selected_trial_name") or "")
        or model_grid.get("selected_trial_name")
        not in (model_multiple.get("eligible_trial_names") or [])
        or model_multiple.get("gate_passed") is not True
        or binding.get("binding_sha256") != canonical_sha256(payload)
    ):
        raise ValueError("formal model admission binding is invalid or inconsistent")

    bundle = binding.get("quant_bundle")
    bundle_id = identity.get("quant_bundle_candidate_id")
    if bundle_id is None:
        if bundle is not None:
            raise ValueError("non-joint formal admission contains bundle evidence")
    elif (
        not isinstance(bundle, dict)
        or bundle.get("candidate_id") != bundle_id
        or bundle.get("bundle_manifest_sha256") != identity.get("quant_bundle_sha256")
        or not is_sha256(bundle.get("admission_evidence_sha256"))
        or not is_sha256(bundle.get("independent_bundle_sha256"))
        or bundle.get("ablations") != list(REQUIRED_QUANT_ABLATIONS)
        or bundle.get("profiles") != list(REQUIRED_RESEARCH_PROFILES)
        or bundle.get("seeds") != list(REQUIRED_MODEL_SEEDS)
        or bundle.get("cell_count")
        != len(REQUIRED_QUANT_ABLATIONS)
        * len(REQUIRED_RESEARCH_PROFILES)
        * len(REQUIRED_MODEL_SEEDS)
        or not is_sha256(bundle.get("ablations_sha256"))
        or not is_sha256(bundle.get("multiple_testing_evidence_sha256"))
        or bundle.get("execution_environment_sha256")
        != model_grid.get("execution_environment_sha256")
        or not isinstance(bundle.get("multiple_testing"), dict)
        or bundle["multiple_testing"].get("contract_version")
        != RUN_GROUP_MULTIPLE_TESTING_CONTRACT_VERSION
        or bundle["multiple_testing"].get("source") != "independent_qlib_recompute"
        or bundle["multiple_testing"].get("final_oos_opened") is not False
        or bundle["multiple_testing"].get("trial_count")
        != len(bundle["multiple_testing"].get("trial_names") or [])
        or f"{bundle_id}:joint"
        not in (bundle["multiple_testing"].get("eligible_trial_names") or [])
        or len(bundle["multiple_testing"].get("holm_adjusted_p_values") or [])
        != bundle["multiple_testing"].get("trial_count")
        or len(bundle["multiple_testing"].get("trial_daily_sharpes") or [])
        != bundle["multiple_testing"].get("trial_count")
        or bundle["multiple_testing"].get("gate_passed") is not True
    ):
        raise ValueError("joint formal admission bundle binding is invalid")
    return dict(binding)


def formal_model_artifact_failures(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    artifact_root: Path,
    dataset_identity_sha256: str,
    test_start: str,
    test_end: str,
) -> list[str]:
    if str(config.get("signal_source") or "factor_score") != "model_prediction":
        return []
    failures: list[str] = []
    try:
        expected_identity = model_signal_identity(config)
    except ValueError as exc:
        return [str(exc)]
    if manifest.get("model_signal") != expected_identity:
        failures.append("formal model manifest does not match the frozen StrategySpec")
    artifact = manifest.get("formal_model_artifact")
    if not isinstance(artifact, dict):
        return [*failures, "formal model artifact is missing"]
    try:
        predictions_path = (artifact_root / str(artifact["predictions_path"])).resolve()
        checkpoint_path = (artifact_root / str(artifact["checkpoint_path"])).resolve()
        predictions_path.relative_to(artifact_root.resolve())
        checkpoint_path.relative_to(artifact_root.resolve())
    except (KeyError, ValueError):
        return [*failures, "formal model artifact paths are invalid"]
    for path, key in (
        (predictions_path, "predictions_sha256"),
        (checkpoint_path, "checkpoint_sha256"),
    ):
        if (
            not path.is_file()
            or not is_sha256(artifact.get(key))
            or file_sha256(path) != artifact[key]
        ):
            failures.append(f"formal model {key} failed immutable verification")
    additional_path_value = artifact.get("additional_factors_path")
    additional_sha256 = artifact.get("additional_factors_sha256")
    if config.get("quant_bundle_candidate_id") is not None:
        try:
            additional_path = (artifact_root / str(additional_path_value)).resolve()
            additional_path.relative_to(artifact_root.resolve())
        except ValueError:
            additional_path = None
        if (
            additional_path is None
            or not additional_path.is_file()
            or not is_sha256(additional_sha256)
            or file_sha256(additional_path) != additional_sha256
        ):
            failures.append("formal model additional factor matrix is missing or changed")
    elif additional_path_value is not None or additional_sha256 is not None:
        failures.append("non-joint formal model contains additional factor data")
    evidence = artifact.get("evidence")
    if not isinstance(evidence, dict):
        return [*failures, "formal model execution evidence is missing"]
    coverage = evidence.get("oos_coverage")
    admission = metrics.get("formal_validation")
    admission = (
        admission.get("model_admission") if isinstance(admission, dict) else None
    )
    admission_grid = (
        admission.get("model_grid") if isinstance(admission, dict) else None
    )
    admitted_environment_sha256 = (
        admission_grid.get("execution_environment_sha256")
        if isinstance(admission_grid, dict)
        else None
    )
    if (
        evidence.get("sandbox_mode") != "docker-isolated"
        or evidence.get("network_mode") != "none"
        or evidence.get("root_filesystem_read_only") is not True
        or evidence.get("final_oos_opened") is not True
        or evidence.get("dataset_identity_sha256") != dataset_identity_sha256
        or evidence.get("test_start") != test_start
        or evidence.get("test_end") != test_end
        or not is_sha256(evidence.get("execution_environment_sha256"))
        or evidence.get("execution_environment_sha256")
        != admitted_environment_sha256
        or not isinstance(coverage, dict)
        or coverage.get("contract_version") != MODEL_PREDICTION_CONTRACT_VERSION
        or coverage.get("coverage_gate_passed") is not True
    ):
        failures.append("formal model execution/OOS evidence is invalid")
    provenance = metrics.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    if provenance.get("formal_model_predictions_sha256") != artifact.get(
        "predictions_sha256"
    ) or provenance.get("formal_model_evidence_sha256") != canonical_sha256(evidence):
        failures.append("formal model provenance does not match its artifact")
    if provenance.get("formal_model_execution_environment_sha256") != evidence.get(
        "execution_environment_sha256"
    ):
        failures.append("formal model provenance has another execution environment")
    if provenance.get("formal_model_additional_factors_sha256") != additional_sha256:
        failures.append("formal model additional-factor provenance is inconsistent")
    return failures
