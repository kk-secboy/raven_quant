from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from .model_research_governance import canonical_sha256
from .research_label_binding import validate_research_label_binding

LEGACY_BASELINE_LABEL_SHAPE = "legacy-unbound-baseline-v1"
BASELINE_LABEL_BINDING_VERSION = "fin-quant-source-label-binding-v1"


def baseline_label_shape_for_replay(frozen: Mapping[str, Any]) -> str:
    """Select only the shape attested by an already frozen baseline envelope.

    This is a historical-read adapter. New writers always emit the versioned
    source-label extension. A present extension can never fall back to legacy
    merely because a member binding is missing.
    """

    if (
        frozen.get("contract_version") != "fin-quant-baseline-prediction-v1"
        or canonical_sha256({key: value for key, value in frozen.items()
                             if key != "evidence_sha256"}) != frozen.get("evidence_sha256")
    ):
        raise ValueError("frozen baseline envelope digest is invalid")
    return baseline_member_label_shape(frozen)


def baseline_member_label_shape(frozen: Mapping[str, Any]) -> str:
    """Read the source-label extension inside a verified immutable envelope."""

    if frozen.get("kind") == "model":
        members = [frozen]
    elif frozen.get("kind") == "ensemble":
        members = frozen.get("components")
        if not isinstance(members, list) or not members:
            raise ValueError("frozen baseline ensemble members are missing")
    else:
        raise ValueError("frozen baseline kind is invalid")
    version = frozen.get("source_label_binding_contract_version")
    for member in members:
        if not isinstance(member, Mapping):
            raise ValueError("frozen baseline member is invalid")
        if version is None:
            if any(key in member for key in (
                "research_label_binding", "research_label_binding_sha256"
            )):
                raise ValueError("unversioned baseline source label evidence is invalid")
        elif version == BASELINE_LABEL_BINDING_VERSION:
            binding = validate_research_label_binding(member.get("research_label_binding"))
            if member.get("research_label_binding_sha256") != binding["binding_sha256"]:
                raise ValueError("frozen baseline source label binding digest changed")
        else:
            raise ValueError("frozen baseline source label version is unsupported")
    return LEGACY_BASELINE_LABEL_SHAPE if version is None else version


def require_compatible_prediction_label_binding(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    allow_different_features: bool = False,
) -> dict[str, Any]:
    """Verify an incumbent's original binding against the research target.

    A requested cutoff can exceed the available field cutoff without changing
    the actual experiment. Ensemble members can also use distinct immutable
    features. Every other window and label field must still match, and the
    original binding is returned unchanged for exact cell-contract validation.
    """

    verified_source = validate_research_label_binding(source)
    verified_target = validate_research_label_binding(target)
    source_window = dict(verified_source["research_window_contract"])
    target_window = dict(verified_target["research_window_contract"])
    requested = "requested_data_cutoff_session"
    if source_window.get(requested) != target_window.get(requested):
        for window in (source_window, target_window):
            try:
                requested_cutoff = date.fromisoformat(str(window[requested]))
                effective_cutoff = date.fromisoformat(
                    str(window["effective_field_cutoff_session"])
                )
            except (KeyError, ValueError) as exc:
                raise ValueError("prediction label binding cutoff evidence is invalid") from exc
            if requested_cutoff < effective_cutoff:
                raise ValueError("prediction requested cutoff precedes its effective cutoff")
    excluded_window_fields = {requested}
    excluded_binding_fields = {
        "binding_sha256", "research_window_contract", "research_window_contract_sha256"
    }
    if allow_different_features:
        excluded_window_fields.update({"feature_set_id", "feature_set_sha256"})
        excluded_binding_fields.update({"feature_set_id", "feature_set_sha256"})
    if (
        {key: value for key, value in source_window.items() if key not in excluded_window_fields}
        != {key: value for key, value in target_window.items() if key not in excluded_window_fields}
        or {
            key: value for key, value in verified_source.items()
            if key not in excluded_binding_fields
        }
        != {
            key: value for key, value in verified_target.items()
            if key not in excluded_binding_fields
        }
    ):
        raise ValueError("fin_quant incumbent prediction uses another label horizon or window")
    return verified_source
