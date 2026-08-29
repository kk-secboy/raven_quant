from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .model_research_governance import canonical_sha256, resolve_model_label_contract
from .research_horizon import LEGACY_AMBIGUOUS

RESEARCH_LABEL_BINDING_VERSION = "research-label-binding-v1"
ACTIVE_RESEARCH_HORIZONS = frozenset({"short_1_5d", "swing_1_6m", "long_1_3y"})


def resolve_research_label_binding(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Bind one executable forward-return label to a verified research window.

    Active horizon research is fail-closed: a horizon name or a bare label value
    is not sufficient.  The immutable ResearchWindowContract and its digest are
    both required, and its dataset, periods, feature set and horizon must match
    the surrounding job payload.  Historical jobs without a window return
    ``None`` so their existing per-candidate label behavior remains readable and
    replayable rather than being reinterpreted as a new horizon.
    """

    window = payload.get("research_window_contract")
    window_sha256 = payload.get("research_window_contract_sha256")
    claimed_profile = str(
        payload.get("horizon_profile")
        or payload.get("strategy_horizon_profile")
        or ""
    ).strip()
    selected = payload.get("label_horizon_sessions")
    if selected is not None and (isinstance(selected, bool) or not isinstance(selected, int)):
        raise ValueError("research label horizon must be an integer number of sessions")

    if window is None:
        if claimed_profile in ACTIVE_RESEARCH_HORIZONS:
            raise ValueError("active horizon research has no verified research window")
        # Do not rewrite an old fin_factor/fin_quant run.  Its candidate-level
        # label_horizon_days remains the historical source of truth.
        return None
    if not isinstance(window, Mapping):
        raise ValueError("research window contract must be an object")

    label = resolve_model_label_contract(
        research_window_contract=window,
        research_window_contract_sha256=str(window_sha256 or ""),
        label_horizon_sessions=selected,
    )
    profile = str(label["horizon_profile"])
    if profile == LEGACY_AMBIGUOUS:
        # A legacy window does not acquire new semantics merely because it has
        # since been wrapped in the ResearchWindowContract envelope.
        return None
    if profile not in ACTIVE_RESEARCH_HORIZONS:
        raise ValueError("research label binding uses an unsupported active horizon")
    if claimed_profile and claimed_profile != profile:
        raise ValueError("research job horizon differs from its research window")

    dataset_identity = str(payload.get("dataset_identity_sha256") or "")
    if dataset_identity != str(window.get("dataset_identity_sha256") or ""):
        raise ValueError("research job dataset identity differs from its research window")
    dataset_name = str(payload.get("dataset") or "")
    if dataset_name and dataset_name != str(window.get("dataset_name") or ""):
        raise ValueError("research job dataset name differs from its research window")
    periods = payload.get("periods")
    if not isinstance(periods, Mapping) or dict(periods) != dict(window.get("periods") or {}):
        raise ValueError("research job periods differ from its research window")

    feature_set = payload.get("feature_set")
    window_feature_id = str(window.get("feature_set_id") or "")
    window_feature_sha256 = str(window.get("feature_set_sha256") or "")
    if feature_set is not None:
        if not isinstance(feature_set, Mapping):
            raise ValueError("research job feature set must be an object")
        if (
            str(feature_set.get("id") or "") != window_feature_id
            or str(feature_set.get("definition_sha256") or "")
            != window_feature_sha256
        ):
            raise ValueError("research job feature set differs from its research window")
    elif window_feature_id or window_feature_sha256:
        raise ValueError("research job omitted the feature set frozen by its window")

    binding: dict[str, Any] = {
        "contract_version": RESEARCH_LABEL_BINDING_VERSION,
        "horizon_profile": profile,
        "legacy": False,
        "allowed_label_horizons_sessions": list(
            label["allowed_label_horizons_sessions"]
        ),
        "label_horizon_sessions": int(label["label_horizon_sessions"]),
        "label_reference_offset_sessions": int(
            label["label_reference_offset_sessions"]
        ),
        "label_expression": str(label["label_expression"]),
        "purge_sessions": int(label["purge_sessions"]),
        "embargo_sessions": int(label["embargo_sessions"]),
        "research_window_contract": dict(window),
        "research_window_contract_sha256": str(window_sha256),
        "dataset_name": dataset_name or str(window["dataset_name"]),
        "dataset_identity_sha256": dataset_identity,
        "periods": dict(periods),
        "feature_set_id": window_feature_id or None,
        "feature_set_sha256": window_feature_sha256 or None,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def validate_research_label_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute and verify a self-contained label binding."""

    if not isinstance(value, Mapping):
        raise ValueError("research label binding must be an object")
    supplied = dict(value)
    binding_sha256 = str(supplied.pop("binding_sha256", "")).lower()
    if (
        supplied.get("contract_version") != RESEARCH_LABEL_BINDING_VERSION
        or supplied.get("legacy") is not False
        or canonical_sha256(supplied) != binding_sha256
    ):
        raise ValueError("research label binding digest is invalid")
    reconstructed = resolve_research_label_binding(
        {
            "horizon_profile": supplied.get("horizon_profile"),
            "dataset": supplied.get("dataset_name"),
            "dataset_identity_sha256": supplied.get("dataset_identity_sha256"),
            "periods": supplied.get("periods"),
            "feature_set": (
                {
                    "id": supplied.get("feature_set_id"),
                    "definition_sha256": supplied.get("feature_set_sha256"),
                }
                if supplied.get("feature_set_id") is not None
                else None
            ),
            "research_window_contract": supplied.get("research_window_contract"),
            "research_window_contract_sha256": supplied.get(
                "research_window_contract_sha256"
            ),
            "label_horizon_sessions": supplied.get("label_horizon_sessions"),
        }
    )
    if reconstructed is None or reconstructed != dict(value):
        raise ValueError("research label binding content is inconsistent")
    return reconstructed


__all__ = [
    "ACTIVE_RESEARCH_HORIZONS",
    "RESEARCH_LABEL_BINDING_VERSION",
    "resolve_research_label_binding",
    "validate_research_label_binding",
]
