"""Sealed, reproducible feature-distribution drift observations.

The activity-health loop observes the exact inputs owned by the approved
strategy version.  It never reuses a previous health snapshot and never treats
a missing observation as zero drift.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .factor_library import compile_qlib_expression
from .research_horizon import canonical_sha256

FEATURE_DRIFT_FORMULA_VERSION = "factor-cross-sectional-psi-max-v1"
FEATURE_DRIFT_OBSERVATION_VERSION = "strategy-feature-drift-observation-v1"
STRATEGY_HEALTH_FEATURE_SET_VERSION = "strategy-health-feature-set-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_strategy_health_feature_set(
    version: Mapping[str, Any],
    *,
    candidate_expressions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze the exact Qlib expressions consumed by a factor strategy.

    Candidate expressions are accepted only when the caller resolved them
    from a governed factor definition.  Executable Python factor code cannot
    be replaced by its prose/formulation field here.
    """

    version_id = str(version.get("id") or "")
    rules_sha256 = _require_sha256(
        version.get("strategy_rules_sha256"), field="strategy rules"
    )
    config = dict(version.get("config") or version.get("config_json") or {})
    if str(config.get("signal_source") or "factor_score") != "factor_score":
        raise ValueError("model strategies require a model feature observation source")
    mode = str(config.get("factor_source_mode") or "promoted_only")
    includes_baseline = mode in {
        "qlib_baseline",
        "qlib_baseline_plus_challenger",
    }
    includes_challenger = mode in {
        "promoted_only",
        "qlib_baseline_plus_challenger",
        "qlib_challenger_replacement",
    }
    if not version_id or not (includes_baseline or includes_challenger):
        raise ValueError("strategy feature source mode is unsupported")

    features: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    baseline = config.get("baseline_definition")
    if includes_baseline:
        if not isinstance(baseline, dict):
            raise ValueError("strategy baseline definition is missing")
        expected_baseline_sha256 = _require_sha256(
            config.get("baseline_definition_sha256"),
            field="strategy baseline definition",
        )
        if canonical_sha256(baseline) != expected_baseline_sha256:
            raise ValueError("strategy baseline definition seal is invalid")
        for item in baseline.get("factors") or []:
            factor_id = str(item.get("id") or "")
            expression = str(item.get("qlib_expression") or "")
            if not factor_id or factor_id in features:
                raise ValueError("strategy baseline factor ids are missing or duplicated")
            compiled = compile_qlib_expression(expression)
            features[factor_id] = compiled.expression
            sources[factor_id] = {
                "source": "baseline_definition",
                "expression_sha256": compiled.expression_sha256,
            }

    candidate_values = dict(candidate_expressions or {})
    if includes_challenger:
        expected_candidate_ids = {
            str(item.get("factor_candidate_id") or item.get("candidate_id") or "")
            for item in (version.get("factors") or [])
        }
        if "" in expected_candidate_ids or not expected_candidate_ids:
            raise ValueError("strategy challenger factor bindings are missing")
        if set(candidate_values) != expected_candidate_ids:
            raise ValueError("strategy challenger feature definitions are incomplete")
        for candidate_id in sorted(expected_candidate_ids):
            factor_id = f"candidate__{candidate_id}"
            if factor_id in features:
                raise ValueError("strategy feature ids are duplicated")
            compiled = compile_qlib_expression(candidate_values[candidate_id])
            features[factor_id] = compiled.expression
            sources[factor_id] = {
                "source": "governed_factor_definition",
                "factor_candidate_id": candidate_id,
                "expression_sha256": compiled.expression_sha256,
            }

    if not features:
        raise ValueError("strategy feature set is empty")
    factor_contract = {
        "contract_version": STRATEGY_HEALTH_FEATURE_SET_VERSION,
        "strategy_version_id": version_id,
        "strategy_rules_sha256": rules_sha256,
        "factor_source_mode": mode,
        "features": features,
        "sources": sources,
    }
    factor_set_sha256 = canonical_sha256(factor_contract)
    definition = {
        "contract_version": STRATEGY_HEALTH_FEATURE_SET_VERSION,
        "id": f"strategy-health:{version_id}:{factor_set_sha256[:16]}",
        "name": f"Activity-health inputs for strategy {version_id}",
        "features": features,
        "source": f"strategy-version:{version_id}:{factor_set_sha256}",
    }
    return {
        **definition,
        "definition_sha256": canonical_sha256(definition),
        "factor_set_sha256": factor_set_sha256,
        "factor_contract": factor_contract,
    }


def build_factor_psi_observation(
    *,
    reference_values: Mapping[str, pd.Series | pd.DataFrame],
    current_values: Mapping[str, pd.Series | pd.DataFrame],
    contract: Mapping[str, Any],
    as_of: date,
    current_dataset_identity_sha256: str,
    current_dataset_lineage_id: str,
    materialization_manifest_sha256: str,
    materialized_file_sha256: Mapping[str, str],
    reference_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Compare frozen formal-OOS factors with the latest decision window.

    Each factor owns reference-quantile bins.  The policy-facing drift is the
    maximum normalized PSI, rather than an average that could hide one badly
    shifted strategy input.  At bootstrap the latest window may overlap the
    immutable OOS reference; subsequent observations move beyond it naturally.
    """

    normalized_contract = dict(contract)
    contract_sha256 = str(normalized_contract.pop("contract_sha256", ""))
    if (
        normalized_contract.get("contract_version")
        != "strategy-feature-drift-reference-v1"
        or canonical_sha256(normalized_contract) != contract_sha256
    ):
        raise ValueError("feature drift reference contract seal is invalid")
    for field in (
        "strategy_rules_sha256",
        "factor_set_sha256",
        "reference_dataset_identity_sha256",
        "formal_artifact_manifest_sha256",
        "feature_set_definition_sha256",
    ):
        _require_sha256(normalized_contract.get(field), field=f"feature drift {field}")
    for value, field in (
        (current_dataset_identity_sha256, "current dataset identity"),
        (current_dataset_lineage_id, "current dataset lineage"),
        (materialization_manifest_sha256, "materialization manifest"),
    ):
        _require_sha256(value, field=field)
    try:
        reference_start = date.fromisoformat(str(normalized_contract["reference_start"]))
        reference_end = date.fromisoformat(str(normalized_contract["reference_end"]))
        window_sessions = int(normalized_contract["current_window_sessions"])
        bins = int(normalized_contract["bins"])
        minimum_reference = int(normalized_contract["minimum_reference_observations"])
        minimum_current = int(normalized_contract["minimum_current_observations"])
        minimum_reference_sessions = int(normalized_contract["minimum_reference_sessions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("feature drift reference contract is incomplete") from exc
    if (
        not reference_start <= reference_end <= as_of
        or window_sessions < 1
        or bins < 3
        or minimum_reference < bins * 2
        or minimum_current < bins * 2
        or minimum_reference_sessions < 2
    ):
        raise ValueError("feature drift reference/current periods are invalid")

    factor_ids = tuple(sorted(str(item) for item in reference_values))
    if (
        not factor_ids
        or set(current_values) != set(factor_ids)
        or set(materialized_file_sha256) != set(factor_ids)
        or set(reference_file_sha256) != set(factor_ids)
        or set(normalized_contract.get("factor_ids") or []) != set(factor_ids)
    ):
        raise ValueError("feature drift factor artifacts are incomplete")
    for values in (materialized_file_sha256, reference_file_sha256):
        for digest in values.values():
            _require_sha256(digest, field="feature drift factor artifact")

    factors: dict[str, Any] = {}
    for factor_id in factor_ids:
        reference = _factor_series(reference_values[factor_id])
        current = _factor_series(current_values[factor_id])
        reference_dates = pd.DatetimeIndex(
            reference.index.get_level_values("datetime")
        ).normalize()
        reference_mask = (reference_dates.date >= reference_start) & (
            reference_dates.date <= reference_end
        )
        reference = reference.loc[reference_mask]
        if len(set(reference_dates[reference_mask].date)) < minimum_reference_sessions:
            raise ValueError(f"feature drift reference sessions are insufficient: {factor_id}")

        candidate_dates = sorted(
            {
                item.date()
                for item in pd.DatetimeIndex(
                    current.index.get_level_values("datetime")
                ).normalize()
                if item.date() <= as_of
            }
        )
        current_dates = candidate_dates[-window_sessions:]
        if not current_dates or current_dates[-1] != as_of:
            raise ValueError("feature drift latest decision window does not reach as_of")
        dates = pd.DatetimeIndex(current.index.get_level_values("datetime")).normalize()
        current = current.loc[np.isin(dates.date, current_dates)]
        if len(reference) < minimum_reference or len(current) < minimum_current:
            raise ValueError(
                f"feature drift windows have insufficient samples: {factor_id}"
            )
        factors[factor_id] = _factor_psi(
            reference,
            current,
            bins=bins,
            reference_dates=(reference_start, reference_end),
            current_dates=(current_dates[0], current_dates[-1]),
            materialized_sha256=materialized_file_sha256[factor_id],
            reference_sha256=reference_file_sha256[factor_id],
        )

    drifts = np.asarray(
        [float(item["normalized_psi"]) for item in factors.values()], dtype=float
    )
    max_factor_id = max(factors, key=lambda item: float(factors[item]["normalized_psi"]))
    observation = {
        "contract_version": FEATURE_DRIFT_OBSERVATION_VERSION,
        "formula_version": FEATURE_DRIFT_FORMULA_VERSION,
        "metric": "feature_drift",
        "metric_semantics": "max_per_factor_reference_decile_psi_transformed_1_minus_exp",
        "strategy_version_id": str(normalized_contract["strategy_version_id"]),
        "strategy_rules_sha256": str(normalized_contract["strategy_rules_sha256"]),
        "factor_set_sha256": str(normalized_contract["factor_set_sha256"]),
        "feature_set_definition_sha256": str(
            normalized_contract["feature_set_definition_sha256"]
        ),
        "factor_ids": list(factor_ids),
        "formal_backtest_id": str(normalized_contract["formal_backtest_id"]),
        "formal_artifact_manifest_sha256": str(
            normalized_contract["formal_artifact_manifest_sha256"]
        ),
        "reference_contract_sha256": contract_sha256,
        "reference_dataset_identity_sha256": str(
            normalized_contract["reference_dataset_identity_sha256"]
        ),
        "current_dataset_identity_sha256": current_dataset_identity_sha256,
        "current_dataset_lineage_id": current_dataset_lineage_id,
        "materialization_manifest_sha256": materialization_manifest_sha256,
        "reference_start": reference_start.isoformat(),
        "reference_end": reference_end.isoformat(),
        "current_start": min(item["current_start"] for item in factors.values()),
        "current_end": as_of.isoformat(),
        "current_window_sessions_requested": window_sessions,
        "factor_count": len(factors),
        "factors": factors,
        "aggregation": {
            "method": "maximum",
            "max_factor_id": max_factor_id,
            "maximum": float(drifts.max()),
            "median": float(np.median(drifts)),
            "p75": float(np.quantile(drifts, 0.75)),
        },
        "feature_drift": float(drifts.max()),
        "observed_at": f"{as_of.isoformat()}T15:00:00+08:00",
    }
    return {**observation, "observation_sha256": canonical_sha256(observation)}


def validate_factor_psi_observation(
    value: Any,
    *,
    strategy_version_id: str,
    current_dataset_identity_sha256: str,
    expected_as_of: date | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("feature drift observation must be an object")
    observation = dict(value)
    seal = str(observation.pop("observation_sha256", ""))
    if canonical_sha256(observation) != seal:
        raise ValueError("feature drift observation seal is invalid")
    if (
        observation.get("contract_version") != FEATURE_DRIFT_OBSERVATION_VERSION
        or observation.get("formula_version") != FEATURE_DRIFT_FORMULA_VERSION
        or observation.get("strategy_version_id") != strategy_version_id
        or observation.get("current_dataset_identity_sha256")
        != current_dataset_identity_sha256
        or observation.get("metric") != "feature_drift"
    ):
        raise ValueError("feature drift observation identity is invalid")
    for field in (
        "strategy_rules_sha256",
        "factor_set_sha256",
        "feature_set_definition_sha256",
        "formal_artifact_manifest_sha256",
        "reference_contract_sha256",
        "reference_dataset_identity_sha256",
        "current_dataset_identity_sha256",
        "current_dataset_lineage_id",
        "materialization_manifest_sha256",
    ):
        _require_sha256(observation.get(field), field=f"feature drift {field}")
    observed_at = datetime.fromisoformat(str(observation.get("observed_at") or ""))
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("feature drift observation time is invalid")
    current_end = date.fromisoformat(str(observation.get("current_end") or ""))
    if expected_as_of is not None and current_end != expected_as_of:
        raise ValueError("feature drift observation is stale")
    factors = observation.get("factors")
    factor_ids = observation.get("factor_ids")
    if (
        not isinstance(factors, dict)
        or not factors
        or not isinstance(factor_ids, list)
        or sorted(factors) != sorted(str(item) for item in factor_ids)
    ):
        raise ValueError("feature drift factor evidence is incomplete")
    per_factor = []
    for factor_id, raw in factors.items():
        if not isinstance(raw, dict):
            raise ValueError("feature drift factor evidence is invalid")
        for field in ("materialized_file_sha256", "reference_file_sha256"):
            _require_sha256(raw.get(field), field=f"feature drift {factor_id} {field}")
        drift = float(raw.get("normalized_psi"))
        raw_psi = float(raw.get("raw_psi"))
        if not math.isfinite(drift) or not 0 <= drift < 1 or not math.isfinite(raw_psi):
            raise ValueError("feature drift factor metric is invalid")
        if int(raw.get("reference_observations") or 0) < 1 or int(
            raw.get("current_observations") or 0
        ) < 1:
            raise ValueError("feature drift factor sample counts are invalid")
        per_factor.append(drift)
    aggregate = float(observation.get("feature_drift"))
    aggregation = observation.get("aggregation")
    if (
        not math.isfinite(aggregate)
        or aggregate != max(per_factor)
        or not isinstance(aggregation, dict)
        or aggregation.get("method") != "maximum"
        or float(aggregation.get("maximum")) != aggregate
    ):
        raise ValueError("feature drift aggregation is invalid")
    return {**observation, "observation_sha256": seal}


def _factor_psi(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int,
    reference_dates: tuple[date, date],
    current_dates: tuple[date, date],
    materialized_sha256: str,
    reference_sha256: str,
) -> dict[str, Any]:
    edges = np.unique(
        np.quantile(reference.to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1))
    )
    if len(edges) < 4:
        raise ValueError("feature drift reference distribution has insufficient variation")
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference.to_numpy(dtype=float), bins=edges)
    current_counts, _ = np.histogram(current.to_numpy(dtype=float), bins=edges)
    epsilon = 0.5
    reference_share = (reference_counts + epsilon) / (
        float(reference_counts.sum()) + epsilon * len(reference_counts)
    )
    current_share = (current_counts + epsilon) / (
        float(current_counts.sum()) + epsilon * len(current_counts)
    )
    raw_psi = float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )
    return {
        "reference_start": reference_dates[0].isoformat(),
        "reference_end": reference_dates[1].isoformat(),
        "current_start": current_dates[0].isoformat(),
        "current_end": current_dates[1].isoformat(),
        "reference_observations": int(len(reference)),
        "current_observations": int(len(current)),
        "bin_edges": [
            None if not math.isfinite(float(value)) else float(value) for value in edges
        ],
        "reference_counts": reference_counts.astype(int).tolist(),
        "current_counts": current_counts.astype(int).tolist(),
        "reference_values_sha256": _digest_values(reference),
        "current_values_sha256": _digest_values(current),
        "reference_file_sha256": reference_sha256,
        "materialized_file_sha256": materialized_sha256,
        "raw_psi": raw_psi,
        "normalized_psi": 1.0 - math.exp(-max(0.0, raw_psi)),
    }


def _factor_series(values: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("feature drift factor artifact must have one column")
        series = values.iloc[:, 0]
    else:
        series = values
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError("feature drift factors require datetime/instrument indexing")
    names = [str(name or "").lower() for name in series.index.names]
    if names == ["instrument", "datetime"]:
        series = series.reorder_levels(["datetime", "instrument"])
    elif names != ["datetime", "instrument"]:
        raise ValueError("feature drift factor index names are invalid")
    dates = pd.to_datetime(series.index.get_level_values("datetime")).tz_localize(None)
    normalized = pd.to_numeric(series, errors="coerce")
    normalized.index = pd.MultiIndex.from_arrays(
        [dates, series.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if normalized.empty or normalized.index.has_duplicates:
        raise ValueError("feature drift factor is empty or duplicated")
    return normalized.astype(float)


def _digest_values(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for (raw_date, raw_instrument), raw_value in values.sort_index().items():
        timestamp = pd.Timestamp(raw_date).tz_localize(None).isoformat()
        digest.update(timestamp.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(raw_instrument).encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack("<d", float(raw_value)))
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest
