from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CORE_BASELINE_RECIPE_IDS = frozenset(
    {"index_enhancement", "full_market_multifactor"}
)
QLIB_BASELINE_RECIPE_IDS = frozenset(
    {*CORE_BASELINE_RECIPE_IDS, "swing_trend", "minute_mean_reversion"}
)
FACTOR_SOURCE_PROMOTED_ONLY = "promoted_only"
FACTOR_SOURCE_QLIB_BASELINE = "qlib_baseline"
FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER = (
    "qlib_baseline_plus_challenger"
)
FACTOR_SOURCE_QLIB_CHALLENGER_REPLACEMENT = (
    "qlib_challenger_replacement"
)
CORE_FACTOR_SOURCE_MODES = frozenset(
    {
        FACTOR_SOURCE_QLIB_BASELINE,
        FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER,
        FACTOR_SOURCE_QLIB_CHALLENGER_REPLACEMENT,
    }
)
QLIB_BASELINE_CONTRACT_VERSION = "qlib-six-factor-baseline-v1"
QLIB_BASELINE_PREPROCESSING = (
    "cross_sectional_winsorize_1_99",
    "cross_sectional_zscore",
    "pit_industry_and_size_neutralization",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def core_baseline_definition(recipe_id: str) -> dict[str, Any]:
    if recipe_id not in QLIB_BASELINE_RECIPE_IDS:
        raise ValueError("the selected recipe has no governed Qlib factor baseline")
    # Imported lazily to keep recipe serialization independent of this validation module.
    from .strategy_recipes import get_strategy_recipe

    recipe = get_strategy_recipe(recipe_id)
    factors = [
        {
            "id": str(item["id"]),
            "weight": float(item["weight"]),
            "qlib_expression": str(item["qlib_expression"]),
        }
        for item in recipe.get("factor_baseline") or []
    ]
    if recipe_id in CORE_BASELINE_RECIPE_IDS:
        if [item["id"] for item in factors] != [
            "momentum",
            "reversal",
            "value",
            "quality",
            "growth",
            "low_volatility",
        ]:
            raise ValueError("the Qlib six-factor baseline definition is incomplete")
        expected_weights = [0.20, 0.10, 0.20, 0.20, 0.10, 0.20]
        if any(
            abs(item["weight"] - expected) > 1e-12
            for item, expected in zip(factors, expected_weights, strict=True)
        ):
            raise ValueError("the Qlib six-factor baseline weights are not the approved weights")
        preprocessing = list(QLIB_BASELINE_PREPROCESSING)
    else:
        if not factors or abs(sum(item["weight"] for item in factors) - 1.0) > 1e-12:
            raise ValueError("the Qlib recipe baseline weights must sum to one")
        preprocessing = [
            "cross_sectional_winsorize_1_99",
            "cross_sectional_zscore",
            "pit_tradability_filter",
        ]
    if any(not item["qlib_expression"].strip() for item in factors):
        raise ValueError("the Qlib six-factor baseline has an empty expression")
    return {
        "contract_version": QLIB_BASELINE_CONTRACT_VERSION,
        "frequency": str(recipe["config_overrides"].get("signal_frequency") or "day"),
        "evaluation_api": "qlib.data.D.features",
        "factors": factors,
        "preprocessing": preprocessing,
        "neutralization_stage": "build_governed_signal",
    }


def bind_factor_source_config(
    config: dict[str, Any],
    *,
    factor_count: int,
    creating_family: bool,
) -> dict[str, Any]:
    bound = dict(config)
    recipe_id = str(bound.get("recipe_id") or "custom")
    mode = str(bound.get("factor_source_mode") or FACTOR_SOURCE_PROMOTED_ONLY)
    challenger_weight = float(bound.get("challenger_weight") or 0.0)

    if recipe_id not in QLIB_BASELINE_RECIPE_IDS:
        if mode != FACTOR_SOURCE_PROMOTED_ONLY:
            raise ValueError(
                "the selected recipe does not define a governed Qlib baseline"
            )
        if factor_count < 1:
            raise ValueError("a promoted-only strategy must contain a promoted factor")
        bound["factor_source_mode"] = FACTOR_SOURCE_PROMOTED_ONLY
        bound["challenger_weight"] = 1.0
        bound["baseline_definition"] = None
        bound["baseline_definition_sha256"] = None
        return bound

    if mode not in CORE_FACTOR_SOURCE_MODES:
        raise ValueError(
            "a Qlib-baseline recipe requires an explicit governed Qlib factor source"
        )
    if creating_family and mode != FACTOR_SOURCE_QLIB_BASELINE:
        raise ValueError(
            "a Qlib-baseline strategy family must start with an independent Qlib baseline version"
        )
    if mode == FACTOR_SOURCE_QLIB_BASELINE:
        if factor_count:
            raise ValueError("a Qlib baseline version must not bind RD-Agent candidates")
        challenger_weight = 0.0
    elif mode == FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER:
        if factor_count < 1:
            raise ValueError("a baseline-plus-challenger version requires promoted candidates")
        if not 0.0 < challenger_weight < 1.0:
            raise ValueError(
                "baseline-plus-challenger weight must be strictly between zero and one"
            )
    else:
        if factor_count < 1:
            raise ValueError("a challenger replacement version requires promoted candidates")
        if abs(challenger_weight - 1.0) > 1e-12:
            raise ValueError("a challenger replacement version must use challenger_weight=1")

    definition = core_baseline_definition(recipe_id)
    supplied = bound.get("baseline_definition")
    supplied_hash = bound.get("baseline_definition_sha256")
    expected_hash = canonical_sha256(definition)
    if supplied is not None and supplied != definition:
        raise ValueError("the supplied Qlib baseline definition is not authoritative")
    if supplied_hash is not None and supplied_hash != expected_hash:
        raise ValueError("the supplied Qlib baseline definition hash is invalid")
    bound["factor_source_mode"] = mode
    bound["challenger_weight"] = challenger_weight
    bound["baseline_definition"] = definition
    bound["baseline_definition_sha256"] = expected_hash
    return bound


def normalize_qlib_baseline_values(
    values: pd.DataFrame, definition: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    factors = list(definition.get("factors") or [])
    if definition.get("contract_version") != QLIB_BASELINE_CONTRACT_VERSION:
        raise ValueError("the Qlib baseline contract version is unsupported")
    if values.shape[1] != len(factors):
        raise ValueError("Qlib D.features returned an incomplete baseline factor matrix")
    raw = values.copy()
    raw.columns = [str(item["id"]) for item in factors]
    if not isinstance(raw.index, pd.MultiIndex) or raw.index.nlevels != 2:
        raise ValueError("Qlib baseline values require datetime/instrument indexing")
    raw.index = raw.index.set_names(["instrument", "datetime"])
    raw = raw.reorder_levels(["datetime", "instrument"]).sort_index()
    raw = raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    normalized = pd.DataFrame(index=raw.index)
    for item in factors:
        factor_id = str(item["id"])

        def winsorize_and_zscore(group: pd.Series) -> pd.Series:
            valid = group.dropna()
            if valid.empty:
                return group
            lower, upper = valid.quantile([0.01, 0.99])
            clipped = group.clip(lower, upper)
            standard_deviation = clipped.std(ddof=0)
            if not np.isfinite(standard_deviation) or standard_deviation <= 0:
                return clipped * 0.0
            return (clipped - clipped.mean()) / standard_deviation

        normalized[factor_id] = raw[factor_id].groupby(
            level="datetime", group_keys=False
        ).apply(winsorize_and_zscore)
    normalized = normalized.dropna(how="any")
    if normalized.empty:
        raise ValueError("the Qlib baseline has no complete normalized observations")
    weights = pd.Series(
        {str(item["id"]): float(item["weight"]) for item in factors}, dtype=float
    )
    score = normalized.mul(weights, axis=1).sum(axis=1).rename("score")
    return raw, normalized, score


def combine_factor_sources(
    *,
    mode: str,
    baseline: pd.Series,
    challenger: pd.Series | None,
    challenger_weight: float,
) -> pd.Series:
    if mode == FACTOR_SOURCE_QLIB_BASELINE:
        if challenger is not None:
            raise ValueError("a baseline-only version cannot consume challenger values")
        return baseline.sort_index().rename("score")
    if challenger is None:
        raise ValueError("the selected factor source requires challenger values")
    if mode == FACTOR_SOURCE_QLIB_CHALLENGER_REPLACEMENT:
        return challenger.sort_index().rename("score")
    if mode != FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER:
        raise ValueError("unsupported governed factor source mode")
    frame = pd.concat(
        [baseline.rename("baseline"), challenger.rename("challenger")],
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        raise ValueError("baseline and challenger values have no common observations")
    weight = float(challenger_weight)
    return (
        frame["baseline"] * (1.0 - weight) + frame["challenger"] * weight
    ).rename("score")


def baseline_manifest_failures(
    *,
    config: dict[str, Any],
    factor_count: int,
    artifact_root: Path,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> list[str]:
    recipe_id = str(config.get("recipe_id") or "custom")
    if recipe_id not in QLIB_BASELINE_RECIPE_IDS:
        return []
    failures: list[str] = []
    try:
        bound = bind_factor_source_config(
            config, factor_count=factor_count, creating_family=False
        )
    except ValueError as exc:
        return [str(exc)]
    expected_definition = bound["baseline_definition"]
    expected_hash = bound["baseline_definition_sha256"]
    baseline = manifest.get("baseline")
    if not isinstance(baseline, dict):
        return ["strategy backtest manifest has no Qlib baseline evidence"]
    if baseline.get("definition") != expected_definition:
        failures.append("strategy backtest baseline definition does not match the version")
    if baseline.get("definition_sha256") != expected_hash:
        failures.append("strategy backtest baseline definition hash is inconsistent")
    if baseline.get("computed_by") != "qlib.data.D.features":
        failures.append("strategy baseline was not independently recomputed by Qlib D.features")
    if manifest.get("factor_source_mode") != bound["factor_source_mode"]:
        failures.append("strategy backtest factor source mode does not match the version")
    try:
        recorded_weight = float(manifest.get("challenger_weight"))
    except (TypeError, ValueError):
        recorded_weight = float("nan")
    if not np.isfinite(recorded_weight) or abs(
        recorded_weight - float(bound["challenger_weight"])
    ) > 1e-12:
        failures.append("strategy backtest challenger weight does not match the version")
    if manifest.get("qlib_workflow") != provenance.get("qlib_workflow"):
        failures.append("strategy manifest and provenance use different Qlib Recorder identities")

    artifacts = baseline.get("artifacts")
    if not isinstance(artifacts, dict):
        failures.append("strategy baseline value artifacts are missing")
        return failures
    expected_factor_ids = {
        str(item["id"]) for item in expected_definition.get("factors") or []
    }
    for artifact_kind in ("raw", "normalized"):
        entries = artifacts.get(artifact_kind)
        if not isinstance(entries, dict) or set(entries) != expected_factor_ids:
            failures.append(f"strategy baseline {artifact_kind} artifacts are incomplete")
            continue
        for factor_id, entry in entries.items():
            failures.extend(
                _artifact_failures(
                    artifact_root,
                    entry,
                    f"strategy baseline {artifact_kind} factor {factor_id}",
                )
            )
    failures.extend(
        _artifact_failures(
            artifact_root,
            artifacts.get("composite"),
            "strategy baseline composite",
        )
    )
    raw_hashes = provenance.get("baseline_raw_values_sha256")
    normalized_hashes = provenance.get("baseline_normalized_values_sha256")
    if not isinstance(raw_hashes, dict) or raw_hashes != {
        key: value.get("sha256")
        for key, value in (artifacts.get("raw") or {}).items()
        if isinstance(value, dict)
    }:
        failures.append("strategy baseline raw value provenance is inconsistent")
    if not isinstance(normalized_hashes, dict) or normalized_hashes != {
        key: value.get("sha256")
        for key, value in (artifacts.get("normalized") or {}).items()
        if isinstance(value, dict)
    }:
        failures.append("strategy baseline normalized value provenance is inconsistent")
    if provenance.get("baseline_definition_sha256") != expected_hash:
        failures.append("strategy baseline definition provenance is inconsistent")
    if provenance.get("baseline_qlib_expressions") != {
        str(item["id"]): str(item["qlib_expression"])
        for item in expected_definition.get("factors") or []
    }:
        failures.append("strategy baseline Qlib expression provenance is inconsistent")
    if provenance.get("baseline_preprocessing") != expected_definition.get(
        "preprocessing"
    ):
        failures.append("strategy baseline preprocessing provenance is inconsistent")
    if provenance.get("baseline_composite_values_sha256") != (
        artifacts.get("composite") or {}
    ).get("sha256"):
        failures.append("strategy baseline composite provenance is inconsistent")
    if provenance.get("factor_source_mode") != bound["factor_source_mode"]:
        failures.append("strategy factor source provenance is inconsistent")
    try:
        provenance_weight = float(provenance.get("challenger_weight"))
    except (TypeError, ValueError):
        provenance_weight = float("nan")
    if not np.isfinite(provenance_weight) or abs(
        provenance_weight - float(bound["challenger_weight"])
    ) > 1e-12:
        failures.append("strategy challenger weight provenance is inconsistent")
    return failures


def _artifact_failures(
    artifact_root: Path, entry: Any, label: str
) -> list[str]:
    if not isinstance(entry, dict) or not is_sha256(entry.get("sha256")):
        return [f"{label} artifact identity is missing"]
    try:
        path = (artifact_root / str(entry["path"])).resolve()
        path.relative_to(artifact_root.resolve())
    except (KeyError, ValueError):
        return [f"{label} artifact path is unsafe"]
    if not path.is_file():
        return [f"{label} artifact is missing"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        return [f"{label} artifact does not match its SHA-256"]
    return []
