from __future__ import annotations

import json
import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quant_platform.jsonb_safety import (
    JSON_NORMALIZATION_METADATA_KEY,
    canonical_jsonb_sha256,
    normalize_jsonb_document,
)

pytestmark = pytest.mark.no_database


def test_nested_non_finite_values_become_null_with_audit_paths() -> None:
    original = {
        "evaluations": [
            {
                "metrics": {
                    "turnover": float("nan"),
                    "gain": float("inf"),
                    "loss": -float("inf"),
                    "finite": 1.25,
                }
            }
        ]
    }

    normalized = normalize_jsonb_document(original)

    assert normalized is not None
    metrics = normalized["evaluations"][0]["metrics"]
    assert metrics == {"turnover": None, "gain": None, "loss": None, "finite": 1.25}
    assert math.isnan(original["evaluations"][0]["metrics"]["turnover"])
    metadata = normalized[JSON_NORMALIZATION_METADATA_KEY]
    assert metadata["replacement"] == "null"
    assert metadata["replacement_count"] == 3
    assert metadata["truncated_count"] == 0
    assert metadata["recorded_occurrences"] == [
        {
            "path": ["evaluations", 0, "metrics", "turnover"],
            "kind": "nan",
            "source_type": "builtins.float",
        },
        {
            "path": ["evaluations", 0, "metrics", "gain"],
            "kind": "positive_infinity",
            "source_type": "builtins.float",
        },
        {
            "path": ["evaluations", 0, "metrics", "loss"],
            "kind": "negative_infinity",
            "source_type": "builtins.float",
        },
    ]
    json.dumps(normalized, allow_nan=False)


def test_numpy_and_pandas_missing_scalars_cannot_bypass_normalization() -> None:
    normalized = normalize_jsonb_document(
        {
            "numpy": [np.float32("nan"), np.float64("inf"), np.int64(7)],
            "array": np.array([1.0, np.nan]),
            "pandas": [pd.NA, pd.NaT],
        }
    )

    assert normalized is not None
    assert normalized["numpy"] == [None, None, 7]
    assert normalized["array"] == [1.0, None]
    assert normalized["pandas"] == [None, None]
    metadata = normalized[JSON_NORMALIZATION_METADATA_KEY]
    assert metadata["replacement_count"] == 5
    assert {item["source_type"] for item in metadata["recorded_occurrences"]} >= {
        "numpy.float32",
        "numpy.float64",
        "pandas._libs.missing.NAType",
        "pandas._libs.tslibs.nattype.NaTType",
    }
    json.dumps(normalized, allow_nan=False)


def test_decimal_non_finite_and_reserved_metadata_collision_are_audited() -> None:
    normalized = normalize_jsonb_document(
        {
            JSON_NORMALIZATION_METADATA_KEY: {"producer_value": True},
            "metric": Decimal("NaN"),
        }
    )

    assert normalized is not None
    assert normalized[JSON_NORMALIZATION_METADATA_KEY] == {"producer_value": True}
    metadata = normalized[f"{JSON_NORMALIZATION_METADATA_KEY}_1"]
    assert metadata["metadata_key"] == f"{JSON_NORMALIZATION_METADATA_KEY}_1"
    assert metadata["reserved_key_collision_count"] == 1
    assert metadata["recorded_occurrences"][0]["source_type"] == "decimal.Decimal"
    json.dumps(normalized, allow_nan=False)


def test_finite_document_is_copied_without_normalization_metadata() -> None:
    original = {"metrics": {"turnover": 0.0}, "values": (1, 2)}

    normalized = normalize_jsonb_document(original)

    assert normalized == {"metrics": {"turnover": 0.0}, "values": [1, 2]}
    assert normalized is not original
    assert JSON_NORMALIZATION_METADATA_KEY not in normalized


def test_canonical_hash_distinguishes_non_finite_missing_and_zero() -> None:
    non_finite = {"turnover": float("nan")}
    normalized_non_finite = normalize_jsonb_document(non_finite)

    assert normalized_non_finite is not None
    assert canonical_jsonb_sha256(non_finite) == canonical_jsonb_sha256(normalized_non_finite)
    assert len(
        {
            canonical_jsonb_sha256(non_finite),
            canonical_jsonb_sha256({"turnover": None}),
            canonical_jsonb_sha256({"turnover": 0.0}),
        }
    ) == 3
