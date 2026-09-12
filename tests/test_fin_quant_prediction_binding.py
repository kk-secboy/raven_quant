from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

import quant_platform.rdagent_candidate_store as candidate_store_module
from quant_platform.model_research_governance import (
    REQUIRED_MODEL_SEEDS,
    REQUIRED_RESEARCH_PROFILES,
    canonical_sha256,
    resolve_model_label_contract,
)
from quant_platform.prediction_label_binding import (
    BASELINE_LABEL_BINDING_VERSION,
    LEGACY_BASELINE_LABEL_SHAPE,
    baseline_label_shape_for_replay,
)
from quant_platform.rdagent_candidate_store import RDAGentCandidateStore
from quant_platform.research_horizon import SHORT_1_5D, research_horizon_contract
from quant_platform.research_label_binding import resolve_research_label_binding
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def _binding(*, feature_id: str = "alpha158", feature_sha: str = "a" * 64) -> dict[str, Any]:
    horizon = research_horizon_contract(SHORT_1_5D)
    window = {
        "contract_version": "research-window-v1",
        "horizon_profile": SHORT_1_5D,
        "horizon_contract_sha256": horizon.sha256,
        "dataset_name": "cn-test",
        "dataset_identity_sha256": "d" * 64,
        "dataset_lineage_id": "e" * 64,
        "dataset_contract_sha256": "c" * 64,
        "field_contract_version": "daily-test-v1",
        "field_coverage_sha256": "f" * 64,
        "feature_set_id": feature_id,
        "feature_set_sha256": feature_sha,
        "calendar_start": "2016-01-04",
        "calendar_end": "2026-09-03",
        "requested_data_cutoff_session": "2026-09-03",
        "effective_field_cutoff_session": "2026-09-03",
        "data_cutoff_session": "2026-09-03",
        "signal_time_semantics": "complete_daily_bar_after_exchange_close",
        "earliest_execution_semantics": "next_trading_session_open_or_conservative_daily_fill",
        "execution_lag_sessions": 1,
        "label_horizons_sessions": list(horizon.label_horizons_sessions),
        "purge_sessions": horizon.purge_sessions,
        "embargo_sessions": horizon.embargo_sessions,
        "label_maturity_enforced": True,
        "label_maturity_tail_sessions": 5,
        "latest_mature_label_sessions": {
            "1": "2026-09-02", "2": "2026-09-01", "3": "2026-08-31", "5": "2026-08-27"
        },
        "periods": {
            "train_start": "2016-01-04", "train_end": "2022-05-26",
            "valid_start": "2022-06-07", "valid_end": "2025-07-16",
            "test_start": "2025-08-14", "test_end": "2026-08-27",
        },
        "sealed_oos_sessions": 252,
        "universe": "cn_all_governed_ashare_and_etf",
        "cost_schedule_versions": ["cost-v1"],
        "cost_schedule_sha256": "b" * 64,
        "execution_contract": "daily_close_signal_d_plus_1_open_t_plus_1_lot100_v1",
        "random_seed": 42,
    }
    return _rebind(window)


def _rebind(window: dict[str, Any], *, label: int = 5) -> dict[str, Any]:
    value = resolve_research_label_binding({
        "horizon_profile": window["horizon_profile"],
        "dataset": window["dataset_name"],
        "dataset_identity_sha256": window["dataset_identity_sha256"],
        "periods": deepcopy(window["periods"]),
        "feature_set": {
            "id": window["feature_set_id"], "definition_sha256": window["feature_set_sha256"]
        },
        "research_window_contract": deepcopy(window),
        "research_window_contract_sha256": canonical_sha256(window),
        "label_horizon_sessions": label,
    })
    assert value is not None
    return value


def _member(binding: dict[str, Any], *, candidate_id: str = "model-1") -> dict[str, Any]:
    contract = resolve_model_label_contract(
        research_window_contract=binding["research_window_contract"],
        research_window_contract_sha256=binding["research_window_contract_sha256"],
        label_horizon_sessions=binding["label_horizon_sessions"],
    )
    return {
        "kind": "model",
        "contract_version": "fin-quant-baseline-prediction-v1",
        "source_label_binding_contract_version": BASELINE_LABEL_BINDING_VERSION,
        "candidate_id": candidate_id,
        "model_candidate_id": candidate_id,
        "dataset": binding["dataset_name"],
        "dataset_identity_sha256": binding["dataset_identity_sha256"],
        "feature_set_id": binding["feature_set_id"],
        "feature_set_definition_sha256": binding["feature_set_sha256"],
        "research_label_binding": deepcopy(binding),
        "research_label_binding_sha256": binding["binding_sha256"],
        "profiles": {
            profile_id: {
                "periods": deepcopy(binding["periods"]),
                "seeds": {
                    str(seed): {
                        "model_label_contract": deepcopy(contract),
                        "model_label_contract_sha256": canonical_sha256(contract),
                    }
                    for seed in REQUIRED_MODEL_SEEDS
                },
            }
            for profile_id in REQUIRED_RESEARCH_PROFILES
        },
    }


def test_requested_cutoff_drift_preserves_all_original_model_evidence() -> None:
    source = _binding(feature_id="platform-seed-v1")
    window = deepcopy(source["research_window_contract"])
    window["requested_data_cutoff_session"] = "2026-09-07"
    target = _rebind(window)
    frozen = _member(source)
    original = deepcopy(frozen)

    LocalJobWorker._require_prediction_label_matches_binding(frozen, target)

    assert frozen == original
    assert source["research_window_contract_sha256"] != target["research_window_contract_sha256"]


def _ensemble() -> tuple[dict[str, Any], dict[str, Any]]:
    primary = _binding()
    secondary = _binding(feature_id="platform-seed-v1", feature_sha="f" * 64)
    frozen = {
        "kind": "ensemble",
        "contract_version": "fin-quant-baseline-prediction-v1",
        "source_label_binding_contract_version": BASELINE_LABEL_BINDING_VERSION,
        "dataset": primary["dataset_name"],
        "dataset_identity_sha256": primary["dataset_identity_sha256"],
        "components": [_member(primary), _member(secondary, candidate_id="model-2")],
    }
    return frozen, primary


def test_ensemble_members_retain_distinct_features_and_original_label_digests() -> None:
    frozen, target = _ensemble()
    original = deepcopy(frozen)
    LocalJobWorker._require_prediction_label_matches_binding(
        frozen, target, primary_model_candidate_id="model-1"
    )
    assert frozen == original


@pytest.mark.parametrize("primary_id", ["", "absent", "model-2"])
def test_ensemble_primary_must_match_the_research_feature_set(primary_id: str) -> None:
    frozen, target = _ensemble()
    with pytest.raises(ValueError, match="primary"):
        LocalJobWorker._require_prediction_label_matches_binding(
            frozen, target, primary_model_candidate_id=primary_id
        )


def test_single_model_cannot_substitute_another_feature_binding() -> None:
    target = _binding()
    source = _binding(feature_id="platform-seed-v1", feature_sha="f" * 64)
    with pytest.raises(ValueError, match="another label horizon or window"):
        LocalJobWorker._require_prediction_label_matches_binding(_member(source), target)


@pytest.mark.parametrize("field", [
    "dataset_identity_sha256", "dataset_contract_sha256", "dataset_lineage_id",
    "field_coverage_sha256", "field_contract_version", "effective_field_cutoff_session",
    "data_cutoff_session", "calendar_start", "calendar_end", "universe",
    "cost_schedule_sha256", "cost_schedule_versions", "execution_contract",
    "execution_lag_sessions", "label_maturity_tail_sessions", "latest_mature_label_sessions",
    "sealed_oos_sessions", "random_seed", "unknown_future_boundary",
])
def test_shared_label_cannot_hide_changed_data_or_evaluation_boundaries(field: str) -> None:
    source = _binding()
    window = deepcopy(source["research_window_contract"])
    window[field] = "9" * 64 if "sha256" in field or "lineage" in field else "changed"
    target = _rebind(window)
    with pytest.raises(ValueError):
        LocalJobWorker._require_prediction_label_matches_binding(_member(source), target)


@pytest.mark.parametrize("period", ["train_start", "train_end", "valid_start", "valid_end",
                                    "test_start", "test_end"])
def test_all_period_boundaries_remain_exact(period: str) -> None:
    source = _binding()
    window = deepcopy(source["research_window_contract"])
    window["periods"][period] = "2020-01-01"
    with pytest.raises(ValueError, match="another label horizon or window"):
        LocalJobWorker._require_prediction_label_matches_binding(_member(source), _rebind(window))


def test_different_selected_return_horizon_is_rejected() -> None:
    source = _binding()
    target = _rebind(source["research_window_contract"], label=2)
    with pytest.raises(ValueError, match="another label horizon"):
        LocalJobWorker._require_prediction_label_matches_binding(_member(source), target)


@pytest.mark.parametrize("mutation", ["missing_source", "source_sha", "window_sha", "cell_sha",
                                      "cell_window", "source_feature", "cell_version"])
def test_missing_or_tampered_source_evidence_is_rejected(mutation: str) -> None:
    binding = _binding()
    frozen = _member(binding)
    cell = frozen["profiles"][REQUIRED_RESEARCH_PROFILES[0]]["seeds"][str(REQUIRED_MODEL_SEEDS[0])]
    if mutation == "missing_source":
        frozen.pop("research_label_binding")
    elif mutation == "source_sha":
        frozen["research_label_binding_sha256"] = "0" * 64
    elif mutation == "window_sha":
        frozen["research_label_binding"]["research_window_contract"]["data_cutoff_session"] = (
            "2026-09-04"
        )
    elif mutation == "cell_sha":
        cell["model_label_contract_sha256"] = "0" * 64
    elif mutation == "source_feature":
        frozen["feature_set_id"] = "substituted"
    else:
        key = "research_window_contract_sha256" if mutation == "cell_window" else "contract_version"
        cell["model_label_contract"][key] = "0" * 64
        cell["model_label_contract_sha256"] = canonical_sha256(cell["model_label_contract"])
    with pytest.raises(ValueError):
        LocalJobWorker._require_prediction_label_matches_binding(frozen, binding)


def test_ensemble_cannot_hide_member_without_source_evidence() -> None:
    frozen, target = _ensemble()
    frozen["components"][1].pop("research_label_binding")
    with pytest.raises(ValueError, match="no frozen source label"):
        LocalJobWorker._require_prediction_label_matches_binding(
            frozen, target, primary_model_candidate_id="model-1"
        )


def test_invalid_requested_cutoff_cannot_use_compatibility_exception() -> None:
    source = _binding()
    window = deepcopy(source["research_window_contract"])
    window["requested_data_cutoff_session"] = "2026-09-01"
    with pytest.raises(ValueError, match="precedes"):
        LocalJobWorker._require_prediction_label_matches_binding(_member(source), _rebind(window))


@pytest.mark.parametrize("shape", [BASELINE_LABEL_BINDING_VERSION, LEGACY_BASELINE_LABEL_SHAPE])
def test_store_freezes_verified_manifest_binding_without_rewriting_it(
    monkeypatch: pytest.MonkeyPatch, shape: str,
) -> None:
    binding = _binding()
    member = _member(binding)
    features = {"id": binding["feature_set_id"],
                "definition_sha256": binding["feature_set_sha256"],
                "features": {"alpha": "$close"}}
    candidate = {
        "id": "model-1", "status": "research_admitted", "dataset": binding["dataset_name"],
        "dataset_identity_sha256": binding["dataset_identity_sha256"],
        "pre_final_end": date.fromisoformat(binding["periods"]["valid_end"]),
        "final_oos_start": date.fromisoformat(binding["periods"]["test_start"]),
        "final_oos_end": date.fromisoformat(binding["periods"]["test_end"]),
        "admission_evidence_json": {"profiles": member["profiles"]},
        "admission_evidence_sha256": "3" * 64, "manifest_sha256": "2" * 64,
        "manifest_json": {"research_label_binding": binding,
                          "research_label_binding_sha256": binding["binding_sha256"]},
        "model_hyperparameters_json": {"model_engine": "ridge_baseline"},
        "base_features_manifest_json": {"feature_set_id": features["id"],
                                        "feature_expressions": features["features"]},
        "feature_set_definition_sha256": features["definition_sha256"],
        "code_artifact_id": "code-1", "code_sha256": "4" * 64, "model_type": "Tabular",
    }
    store = object.__new__(RDAGentCandidateStore)
    store.engine = SimpleNamespace(connect=lambda: nullcontext(None))
    verified: list[bool] = []

    def get_candidate(candidate_id: str, *, verify: bool) -> dict[str, Any]:
        assert candidate_id == "model-1"
        verified.append(verify)
        return candidate

    monkeypatch.setattr(store, "get_model_candidate", get_candidate)
    monkeypatch.setattr(store, "_one", lambda *args: SimpleNamespace(storage_path="model.py"))
    monkeypatch.setattr(store, "_verify_artifact_row", lambda _: None)
    monkeypatch.setattr(candidate_store_module, "get_feature_set", lambda _: features)
    monkeypatch.setattr(
        candidate_store_module, "_verify_model_seed_artifacts", lambda *args, **kw: None
    )
    frozen = store.freeze_quant_baseline_prediction(
        candidate_kind="model", candidate_id="model-1", dataset=candidate["dataset"],
        dataset_identity_sha256=candidate["dataset_identity_sha256"],
        pre_final_end=candidate["pre_final_end"], final_oos_start=candidate["final_oos_start"],
        final_oos_end=candidate["final_oos_end"], baseline_label_shape=shape,
    )
    assert verified == [True]
    assert baseline_label_shape_for_replay(frozen) == shape
    if shape == BASELINE_LABEL_BINDING_VERSION:
        assert frozen["research_label_binding"] == binding
        assert frozen["research_label_binding_sha256"] == binding["binding_sha256"]
        LocalJobWorker._require_prediction_label_matches_binding(frozen, binding)
    else:
        assert "research_label_binding" not in frozen
        assert "research_label_binding_sha256" not in frozen
        with pytest.raises(ValueError, match="no versioned source label"):
            LocalJobWorker._require_prediction_label_matches_binding(frozen, binding)


def _envelope(kind: str, *, legacy: bool) -> dict[str, Any]:
    result = _member(_binding()) if kind == "model" else _ensemble()[0]
    if kind == "ensemble":
        result["candidate_id"] = "ensemble-1"
        for member in result["components"]:
            member.pop("source_label_binding_contract_version")
    if legacy:
        result.pop("source_label_binding_contract_version")
        for member in ([result] if kind == "model" else result["components"]):
            member.pop("research_label_binding")
            member.pop("research_label_binding_sha256")
    result["selection_evidence_sha256"] = "5" * 64
    result["evidence_sha256"] = canonical_sha256(result)
    return result


class _Row(SimpleNamespace):
    @property
    def _mapping(self) -> dict[str, Any]:
        return vars(self)


def _historical_bundle_store(
    monkeypatch: pytest.MonkeyPatch, baseline: dict[str, Any],
) -> tuple[RDAGentCandidateStore, list[str]]:
    binding = _binding()
    base_features = {"feature_set_id": binding["feature_set_id"]}
    common = {
        "bundle_artifact_sha256": "1" * 64,
        "base_features_manifest_sha256": canonical_sha256(base_features),
        "feature_set_definition_sha256": binding["feature_set_sha256"],
        "dataset_identity_sha256": binding["dataset_identity_sha256"],
        "pre_final_end": binding["periods"]["valid_end"],
        "final_oos_start": binding["periods"]["test_start"],
        "final_oos_end": binding["periods"]["test_end"],
    }
    manifest = {
        **common,
        "contract_version": "quant-bundle-joint-proposal-v1",
        "baseline_prediction_champion": deepcopy(baseline),
        "model": {"candidate_id": "challenger-1", "code_sha256": "2" * 64},
        "factors": [],
    }
    row = _Row(
        **{key: value for key, value in common.items() if key not in {
            "pre_final_end", "final_oos_start", "final_oos_end"
        }},
        id="bundle-1", bundle_manifest_json=manifest,
        bundle_manifest_sha256=canonical_sha256(manifest), bundle_artifact_id="artifact-1",
        base_features_manifest_json=base_features,
        model_candidate_id="challenger-1", model_ensemble_candidate_id=None,
        dataset=binding["dataset_name"], status="awaiting_independent_evaluation",
        factor_candidate_ids_json=[], capital_eligible=False,
        **{key: date.fromisoformat(common[key]) for key in (
            "pre_final_end", "final_oos_start", "final_oos_end"
        )},
    )
    rows = {
        "bundle-1": row,
        "artifact-1": _Row(content_sha256=common["bundle_artifact_sha256"]),
        "challenger-1": _Row(id="challenger-1", code_sha256="2" * 64,
                             status="awaiting_independent_evaluation"),
    }
    store = object.__new__(RDAGentCandidateStore)
    store.engine = SimpleNamespace(connect=lambda: nullcontext(None))
    monkeypatch.setattr(
        store, "_one", lambda connection, table, identifier, context: rows[identifier]
    )
    monkeypatch.setattr(store, "_verify_artifact_row", lambda _: None)
    monkeypatch.setattr(candidate_store_module, "get_feature_set", lambda _: {
        "id": binding["feature_set_id"], "definition_sha256": binding["feature_set_sha256"]
    })
    requested_shapes: list[str] = []

    def freeze(*, baseline_label_shape: str = BASELINE_LABEL_BINDING_VERSION, **_: Any) -> dict:
        requested_shapes.append(baseline_label_shape)
        rebuilt = _envelope(
            baseline["kind"], legacy=baseline_label_shape == LEGACY_BASELINE_LABEL_SHAPE
        )
        rebuilt.pop("selection_evidence_sha256")
        rebuilt.pop("evidence_sha256")
        rebuilt["evidence_sha256"] = canonical_sha256(rebuilt)
        return rebuilt

    monkeypatch.setattr(store, "freeze_quant_baseline_prediction", freeze)
    return store, requested_shapes


@pytest.mark.parametrize("kind", ["model", "ensemble"])
@pytest.mark.parametrize("legacy", [True, False])
def test_verified_quant_bundle_read_preserves_its_original_baseline_shape_and_hash(
    monkeypatch: pytest.MonkeyPatch, kind: str, legacy: bool,
) -> None:
    baseline = _envelope(kind, legacy=legacy)
    original = deepcopy(baseline)
    store, shapes = _historical_bundle_store(monkeypatch, baseline)
    observed = store.get_quant_bundle_candidate("bundle-1", verify=True)
    assert observed["bundle_manifest_json"]["baseline_prediction_champion"] == original
    assert shapes == [LEGACY_BASELINE_LABEL_SHAPE if legacy else BASELINE_LABEL_BINDING_VERSION]


@pytest.mark.parametrize("kind", ["model", "ensemble"])
@pytest.mark.parametrize("change", ["missing_binding", "missing_marker", "tampered_envelope"])
def test_versioned_baseline_read_never_silently_downgrades_missing_source_evidence(
    monkeypatch: pytest.MonkeyPatch, kind: str, change: str,
) -> None:
    baseline = _envelope(kind, legacy=False)
    members = [baseline] if kind == "model" else baseline["components"]
    if change == "missing_binding":
        members[-1].pop("research_label_binding")
    elif change == "missing_marker":
        baseline.pop("source_label_binding_contract_version")
    else:
        baseline["candidate_id"] = "substituted"
    if change != "tampered_envelope":
        baseline["evidence_sha256"] = canonical_sha256({
            key: value for key, value in baseline.items() if key != "evidence_sha256"
        })
    store, shapes = _historical_bundle_store(monkeypatch, baseline)
    with pytest.raises(ValueError):
        store.get_quant_bundle_candidate("bundle-1", verify=True)
    assert shapes == []


@pytest.mark.parametrize("legacy", [True, False])
def test_non_joint_ensemble_registry_read_keeps_its_explicit_component_shape(
    monkeypatch: pytest.MonkeyPatch, legacy: bool,
) -> None:
    baseline = _envelope("ensemble", legacy=legacy)
    store, shapes = _historical_bundle_store(monkeypatch, baseline)
    original_one = store._one
    row = original_one(None, None, "bundle-1", None)
    registry_components = [{"model_candidate_id": "model-1", "weight": 0.5},
                           {"model_candidate_id": "model-2", "weight": 0.5}]
    ensemble_row = _Row(
        id="ensemble-1", status="research_admitted", dataset=row.dataset,
        dataset_identity_sha256=row.dataset_identity_sha256, manifest_sha256="3" * 64,
        admission_evidence_sha256="4" * 64, components_json=registry_components,
    )
    component = {
        "kind": "ensemble", "candidate_id": "ensemble-1", "manifest_sha256": "3" * 64,
        "admission_evidence_sha256": "4" * 64, "combiner": "equal_rank", "stacking": False,
        "components": registry_components if legacy else baseline["components"],
    }
    if not legacy:
        component["source_label_binding_contract_version"] = BASELINE_LABEL_BINDING_VERSION
    row.model_candidate_id = None
    row.model_ensemble_candidate_id = "ensemble-1"
    row.bundle_manifest_json.update({
        "contract_version": "quant-bundle-candidate-v1", "model_ensemble": component,
        "prediction_component": deepcopy(component),
    })
    row.bundle_manifest_json.pop("baseline_prediction_champion")
    row.bundle_manifest_json.pop("model")
    row.bundle_manifest_sha256 = canonical_sha256(row.bundle_manifest_json)
    monkeypatch.setattr(store, "_one", lambda connection, table, identifier, context: (
        ensemble_row if identifier == "ensemble-1"
        else original_one(connection, table, identifier, context)
    ))
    observed = store.get_quant_bundle_candidate("bundle-1", verify=True)
    assert observed["bundle_manifest_json"]["model_ensemble"] == component
    assert shapes == ([] if legacy else [BASELINE_LABEL_BINDING_VERSION])
