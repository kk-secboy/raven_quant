from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

JSON_NORMALIZATION_METADATA_KEY = "_quantlab_json_normalization"
_MAX_RECORDED_OCCURRENCES = 100


@dataclass
class _NormalizationState:
    replacement_count: int = 0
    occurrences: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.occurrences is None:
            self.occurrences = []

    def record(
        self,
        path: tuple[str | int, ...],
        *,
        kind: str,
        source_type: str,
    ) -> None:
        self.replacement_count += 1
        assert self.occurrences is not None
        if len(self.occurrences) < _MAX_RECORDED_OCCURRENCES:
            self.occurrences.append(
                {
                    "path": list(path),
                    "kind": kind,
                    "source_type": source_type,
                }
            )


def normalize_jsonb_document(document: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a JSONB-safe copy with non-finite or missing numeric values as null.

    PostgreSQL's JSONB parser rejects the non-standard ``NaN`` and infinity
    tokens accepted by Python's default JSON encoder.  Persisting a job result
    must not turn an otherwise successful computation into a failed job, and a
    missing metric must never be fabricated as zero.  Replacements are therefore
    ``None``/JSON ``null`` and are described by bounded, in-document audit
    metadata.  The caller-owned result object is not mutated.

    NumPy scalar/array containers are converted to their Python equivalents so
    a non-finite NumPy value cannot bypass the same recursive check.  Pandas
    ``NA``/``NaT`` scalars are treated as missing values for the same reason.
    """

    if document is None:
        return None
    state = _NormalizationState()
    normalized = _normalize_value(document, (), state)
    if not isinstance(normalized, dict):  # Defensive: the public contract is a mapping.
        raise TypeError("JSONB job document must normalize to an object")
    if not state.replacement_count:
        return normalized

    metadata_key = JSON_NORMALIZATION_METADATA_KEY
    collision_count = 0
    while metadata_key in normalized:
        collision_count += 1
        metadata_key = f"{JSON_NORMALIZATION_METADATA_KEY}_{collision_count}"

    assert state.occurrences is not None
    metadata: dict[str, Any] = {
        "contract_version": "jsonb-nonfinite-normalization-v1",
        "replacement": "null",
        "replacement_count": state.replacement_count,
        "recorded_occurrences": state.occurrences,
        "recorded_count": len(state.occurrences),
        "truncated_count": state.replacement_count - len(state.occurrences),
    }
    if collision_count:
        metadata["reserved_key_collision_count"] = collision_count
        metadata["metadata_key"] = metadata_key
    normalized[metadata_key] = metadata
    return normalized


def canonical_jsonb_sha256(document: dict[str, Any]) -> str:
    """Hash the exact JSONB-safe projection under the non-finite-null policy.

    ``allow_nan=False`` is deliberate: a future traversal regression must fail
    instead of silently hashing Python's non-standard ``NaN`` token.  Because
    the normalization metadata is included in the projection, a measured NaN
    is distinguishable from an originally missing null and from a real zero.
    """

    normalized = normalize_jsonb_document(document)
    if normalized is None:  # Defensive: the public contract excludes None.
        raise TypeError("canonical JSONB document must be an object")
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_value(
    value: Any,
    path: tuple[str | int, ...],
    state: _NormalizationState,
    *,
    source_type: str | None = None,
) -> Any:
    value_type = type(value)
    qualified_type = source_type or f"{value_type.__module__}.{value_type.__qualname__}"

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        state.record(path, kind=_non_finite_kind(value), source_type=qualified_type)
        return None
    if isinstance(value, Decimal) and not value.is_finite():
        state.record(
            path,
            kind=_non_finite_kind(float(value)),
            source_type=qualified_type,
        )
        return None

    type_root = value_type.__module__.partition(".")[0]
    if type_root == "pandas" and value_type.__name__ in {"NAType", "NaTType"}:
        state.record(path, kind="missing", source_type=qualified_type)
        return None
    if type_root == "numpy":
        if value_type.__name__ == "ndarray":
            return _normalize_value(value.tolist(), path, state, source_type=qualified_type)
        item = getattr(value, "item", None)
        if callable(item):
            converted = item()
            if converted is value:
                return value
            return _normalize_value(converted, path, state, source_type=qualified_type)

    if isinstance(value, Mapping):
        return {
            key: _normalize_value(child, (*path, str(key)), state)
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _normalize_value(child, (*path, index), state)
            for index, child in enumerate(value)
        ]
    return value


def _non_finite_kind(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "positive_infinity" if value > 0 else "negative_infinity"
