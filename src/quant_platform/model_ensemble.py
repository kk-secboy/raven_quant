from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_research_governance import (
    REQUIRED_MODEL_SEEDS,
    REQUIRED_RESEARCH_PROFILES,
    canonical_sha256,
    file_sha256,
    normalize_model_predictions,
)
from .research_label_binding import validate_research_label_binding

MODEL_ENSEMBLE_CONTRACT_VERSION = "quantlab-model-ensemble-v2"
MODEL_ENSEMBLE_EVALUATION_CONTRACT_VERSION = (
    "model-ensemble-independent-evaluation-v1"
)
MODEL_ENSEMBLE_LABEL_CONTRACT_VERSION = "model-ensemble-label-contract-v1"
MODEL_ENSEMBLE_CORRELATION_POLICY_VERSION = "daily-cross-sectional-rank-correlation-v1"
MODEL_ENSEMBLE_CORRELATION_LIMIT = 0.90
MODEL_ENSEMBLE_MAX_CANDIDATES = 4
MODEL_ENSEMBLE_MAX_MEMBERS = 3
MODEL_ENSEMBLE_MIN_DAILY_INTERSECTION = 50
MODEL_ENSEMBLE_MIN_INTERSECTION_RATIO = 0.80
MODEL_ENSEMBLE_MIN_GOOD_DAY_RATE = 0.95


class EnsemblePredictionsPending(ValueError):
    """A model has no complete immutable pre-final prediction grid yet."""


_MODEL_ENSEMBLE_LABEL_IDENTITY_FIELDS = (
    "horizon_profile",
    "legacy",
    "allowed_label_horizons_sessions",
    "label_horizon_sessions",
    "label_reference_offset_sessions",
    "label_expression",
    "purge_sessions",
    "embargo_sessions",
    "dataset_name",
    "dataset_identity_sha256",
    "periods",
)


def model_ensemble_label_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Project a feature-specific binding onto its shared prediction target."""

    validated = validate_research_label_binding(binding)
    return {key: validated[key] for key in _MODEL_ENSEMBLE_LABEL_IDENTITY_FIELDS}


def build_model_ensemble_label_contract(
    member_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze one label target shared by every immutable ensemble member."""

    if len(member_bindings) < 2:
        raise ValueError("model ensemble label contract requires at least two members")
    normalized: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for raw_member_id, raw_binding in member_bindings.items():
        member_id = str(raw_member_id or "").strip()
        if not member_id or member_id in normalized:
            raise ValueError("model ensemble label contract has invalid member identities")
        binding = validate_research_label_binding(raw_binding)
        normalized[member_id] = binding
        identities[member_id] = model_ensemble_label_identity(binding)
    ordered_ids = sorted(normalized)
    shared_identity = identities[ordered_ids[0]]
    if any(identities[member_id] != shared_identity for member_id in ordered_ids[1:]):
        raise ValueError("model ensemble members use different frozen label targets")
    contract: dict[str, Any] = {
        "contract_version": MODEL_ENSEMBLE_LABEL_CONTRACT_VERSION,
        "label_identity": shared_identity,
        "member_binding_sha256": {
            member_id: str(normalized[member_id]["binding_sha256"])
            for member_id in ordered_ids
        },
    }
    contract["evidence_sha256"] = canonical_sha256(contract)
    return contract


def validate_model_ensemble_label_contract(
    value: Mapping[str, Any],
    *,
    member_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild an ensemble label contract from its frozen member manifests."""

    if not isinstance(value, Mapping):
        raise ValueError("model ensemble label contract must be an object")
    expected = build_model_ensemble_label_contract(member_bindings)
    if dict(value) != expected:
        raise ValueError("model ensemble label contract changed after preregistration")
    return expected


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def load_sealed_predictions(path_value: Any, expected_sha256: Any) -> pd.DataFrame:
    path = Path(str(path_value or "")).resolve()
    digest = str(expected_sha256 or "").lower()
    if not path.is_file() or not _is_sha256(digest) or file_sha256(path) != digest:
        raise EnsemblePredictionsPending(
            "ensemble member prediction artifact is missing or changed"
        )
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        raw = pd.read_hdf(path)
    else:
        raise EnsemblePredictionsPending(
            "ensemble member predictions must be parquet or HDF5"
        )
    return normalize_model_predictions(raw)


def _finite_scores(frame: pd.DataFrame, name: str) -> pd.Series:
    values = pd.to_numeric(frame["score"], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    return values.loc[finite].rename(name)


def _aligned_prediction_frame(
    members: Sequence[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 2 <= len(members) <= MODEL_ENSEMBLE_MAX_MEMBERS:
        raise ValueError("an ensemble requires two or three prediction members")
    ids = [str(item[0]) for item in members]
    if len(set(ids)) != len(ids):
        raise ValueError("ensemble prediction members must be unique")
    series = [_finite_scores(frame, member_id) for member_id, frame in members]
    dates_by_member = {
        member_id: pd.DatetimeIndex(
            values.index.get_level_values("datetime")
        ).normalize().unique().sort_values()
        for member_id, values in zip(ids, series, strict=True)
    }
    date_sets = {tuple(value) for value in dates_by_member.values()}
    if len(date_sets) != 1:
        raise EnsemblePredictionsPending(
            "ensemble members do not cover the same pre-final trading days"
        )
    aligned = pd.concat(series, axis=1, join="inner").dropna()
    if aligned.empty:
        raise EnsemblePredictionsPending(
            "ensemble members have no common finite prediction sample"
        )
    expected_days = next(iter(dates_by_member.values()))
    aligned_days = pd.DatetimeIndex(
        aligned.index.get_level_values("datetime")
    ).normalize()
    counts = pd.Series(1, index=aligned_days).groupby(level=0).sum().reindex(
        expected_days, fill_value=0
    )
    member_counts = []
    for values in series:
        member_days = pd.DatetimeIndex(
            values.index.get_level_values("datetime")
        ).normalize()
        member_counts.append(
            pd.Series(1, index=member_days).groupby(level=0).sum().reindex(
                expected_days, fill_value=0
            )
        )
    minimum_member_counts = pd.concat(member_counts, axis=1).min(axis=1)
    ratios = counts.div(minimum_member_counts.replace(0, np.nan)).fillna(0.0)
    good = (counts >= MODEL_ENSEMBLE_MIN_DAILY_INTERSECTION) & (
        ratios >= MODEL_ENSEMBLE_MIN_INTERSECTION_RATIO
    )
    good_day_rate = float(good.mean()) if len(good) else 0.0
    if good_day_rate < MODEL_ENSEMBLE_MIN_GOOD_DAY_RATE:
        raise EnsemblePredictionsPending(
            "ensemble member prediction intersection coverage is insufficient"
        )
    return aligned, {
        "contract_version": "model-ensemble-intersection-coverage-v1",
        "trading_day_count": len(expected_days),
        "row_count": len(aligned),
        "minimum_daily_intersection": int(counts.min()),
        "minimum_intersection_ratio": float(ratios.min()),
        "good_day_rate": good_day_rate,
        "minimum_daily_intersection_required": MODEL_ENSEMBLE_MIN_DAILY_INTERSECTION,
        "minimum_intersection_ratio_required": MODEL_ENSEMBLE_MIN_INTERSECTION_RATIO,
        "minimum_good_day_rate_required": MODEL_ENSEMBLE_MIN_GOOD_DAY_RATE,
        "coverage_gate_passed": True,
    }


def daily_rank_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> dict[str, Any]:
    aligned, coverage = _aligned_prediction_frame(
        (("left", left), ("right", right))
    )
    daily = aligned.groupby(level="datetime", sort=True).apply(
        lambda frame: frame["left"].corr(frame["right"], method="spearman"),
        include_groups=False,
    )
    daily = pd.to_numeric(daily, errors="coerce").dropna()
    if len(daily) != int(coverage["trading_day_count"]):
        raise EnsemblePredictionsPending(
            "prediction rank correlation is not finite on every pre-final day"
        )
    mean_correlation = float(daily.mean())
    mean_absolute_correlation = float(daily.abs().mean())
    maximum_absolute_correlation = float(daily.abs().max())
    if not all(
        math.isfinite(value)
        for value in (
            mean_correlation,
            mean_absolute_correlation,
            maximum_absolute_correlation,
        )
    ):
        raise EnsemblePredictionsPending("prediction rank correlation is not finite")
    evidence = {
        "contract_version": MODEL_ENSEMBLE_CORRELATION_POLICY_VERSION,
        "method": "daily_cross_sectional_spearman",
        "aggregation": "mean_absolute_daily_correlation",
        "mean_correlation": mean_correlation,
        "mean_absolute_correlation": mean_absolute_correlation,
        "maximum_absolute_daily_correlation": maximum_absolute_correlation,
        "limit": MODEL_ENSEMBLE_CORRELATION_LIMIT,
        "passed": mean_absolute_correlation <= MODEL_ENSEMBLE_CORRELATION_LIMIT,
        "coverage": coverage,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def equal_rank_predictions(
    members: Sequence[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    aligned, coverage = _aligned_prediction_frame(members)
    ranked = aligned.groupby(level="datetime", sort=False).rank(
        method="average", pct=True
    )
    score = ranked.mean(axis=1).rename("score")
    if not np.isfinite(score.to_numpy(dtype=float)).all():
        raise ValueError("equal-rank ensemble produced non-finite scores")
    result = normalize_model_predictions(score.to_frame())
    evidence = {
        "contract_version": "model-ensemble-equal-rank-combination-v1",
        "combiner": "equal_rank",
        "stacking": False,
        "member_ids": [str(item[0]) for item in members],
        "member_count": len(members),
        "weights": [1.0 / len(members)] * len(members),
        "coverage": coverage,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return result, evidence


def prediction_grid_from_admission(candidate: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate.get("id") or "")
    admission = candidate.get("admission_evidence_json")
    if not candidate_id or not isinstance(admission, Mapping):
        raise EnsemblePredictionsPending(
            "ensemble member has no admitted independent prediction evidence"
        )
    profiles = admission.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != set(
        REQUIRED_RESEARCH_PROFILES
    ):
        raise EnsemblePredictionsPending(
            "ensemble member prediction grid is incomplete"
        )
    result: dict[str, Any] = {}
    for profile_id in REQUIRED_RESEARCH_PROFILES:
        profile = profiles[profile_id]
        if not isinstance(profile, Mapping):
            raise EnsemblePredictionsPending(
                "ensemble member prediction profile is invalid"
            )
        seeds = profile.get("seeds")
        if not isinstance(seeds, Mapping) or {int(seed) for seed in seeds} != set(
            REQUIRED_MODEL_SEEDS
        ):
            raise EnsemblePredictionsPending(
                "ensemble member prediction seed grid is incomplete"
            )
        result[profile_id] = {
            "periods": dict(profile.get("periods") or {}),
            "seeds": {
                str(seed): {
                    "predictions_path": str(
                        (seeds.get(str(seed), seeds.get(seed)) or {}).get(
                            "predictions_path"
                        )
                        or ""
                    ),
                    "predictions_sha256": str(
                        (seeds.get(str(seed), seeds.get(seed)) or {}).get(
                            "predictions_sha256"
                        )
                        or ""
                    ).lower(),
                }
                for seed in REQUIRED_MODEL_SEEDS
            },
        }
    grid_manifest = {
        "candidate_id": candidate_id,
        "candidate_manifest_sha256": str(candidate.get("manifest_sha256") or ""),
        "admission_evidence_sha256": str(
            candidate.get("admission_evidence_sha256") or ""
        ),
        "profiles": result,
    }
    grid_manifest["prediction_grid_sha256"] = canonical_sha256(grid_manifest)
    return grid_manifest


def pairwise_grid_correlation(
    left_grid: Mapping[str, Any],
    right_grid: Mapping[str, Any],
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    maximum = 0.0
    for profile_id in REQUIRED_RESEARCH_PROFILES:
        left_profile = left_grid["profiles"][profile_id]
        right_profile = right_grid["profiles"][profile_id]
        if dict(left_profile.get("periods") or {}) != dict(
            right_profile.get("periods") or {}
        ):
            raise EnsemblePredictionsPending(
                "ensemble members use different governed profile periods"
            )
        for seed in REQUIRED_MODEL_SEEDS:
            left_seed = left_profile["seeds"][str(seed)]
            right_seed = right_profile["seeds"][str(seed)]
            correlation = daily_rank_correlation(
                load_sealed_predictions(
                    left_seed["predictions_path"], left_seed["predictions_sha256"]
                ),
                load_sealed_predictions(
                    right_seed["predictions_path"], right_seed["predictions_sha256"]
                ),
            )
            cells[f"{profile_id}:{seed}"] = correlation
            maximum = max(maximum, float(correlation["mean_absolute_correlation"]))
    evidence = {
        "contract_version": "model-ensemble-grid-correlation-v1",
        "left_candidate_id": str(left_grid["candidate_id"]),
        "right_candidate_id": str(right_grid["candidate_id"]),
        "aggregation": "maximum_over_governed_profile_seed_cells",
        "maximum_mean_absolute_daily_rank_correlation": maximum,
        "limit": MODEL_ENSEMBLE_CORRELATION_LIMIT,
        "passed": maximum <= MODEL_ENSEMBLE_CORRELATION_LIMIT,
        "cells": cells,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def bounded_ensemble_combinations(
    champions: Sequence[Mapping[str, Any]],
    pairwise: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_family: dict[str, Mapping[str, Any]] = {}
    for champion in champions:
        family = str(champion.get("model_family") or "")
        if not family or family in by_family:
            raise ValueError("ensemble champions require one model per distinct family")
        by_family[family] = champion
    ordered = [by_family[key] for key in sorted(by_family)]
    if len(ordered) > MODEL_ENSEMBLE_MAX_CANDIDATES:
        raise ValueError("ensemble champion family limit is exceeded")
    by_size: dict[int, list[dict[str, Any]]] = {2: [], 3: []}
    for size in range(2, min(MODEL_ENSEMBLE_MAX_MEMBERS, len(ordered)) + 1):
        for members in itertools.combinations(ordered, size):
            pair_evidence: list[dict[str, Any]] = []
            admitted = True
            for left, right in itertools.combinations(members, 2):
                left_id = str(left["id"])
                right_id = str(right["id"])
                value = pairwise.get((left_id, right_id)) or pairwise.get(
                    (right_id, left_id)
                )
                if not isinstance(value, Mapping):
                    raise EnsemblePredictionsPending(
                        f"missing prediction correlation for {left_id}/{right_id}"
                    )
                pair_evidence.append(dict(value))
                admitted = admitted and value.get("passed") is True
            if not admitted:
                continue
            components = [
                {
                    "model_candidate_id": str(member["id"]),
                    "model_family": str(member["model_family"]),
                    "weight": 1.0 / len(members),
                    "model_manifest_sha256": str(member["manifest_sha256"]),
                    "model_admission_evidence_sha256": str(
                        member["admission_evidence_sha256"]
                    ),
                    "prediction_grid_sha256": str(member["prediction_grid_sha256"]),
                }
                for member in members
            ]
            by_size[size].append(
                {
                    "name": "equal-rank-"
                    + "-".join(str(member["model_family"]) for member in members),
                    "components": components,
                    "correlation_evidence": pair_evidence,
                    "maximum_pair_correlation": max(
                        float(item["maximum_mean_absolute_daily_rank_correlation"])
                        for item in pair_evidence
                    ),
                }
            )
    for size in by_size:
        by_size[size].sort(
            key=lambda item: (
                float(item["maximum_pair_correlation"]),
                str(item["name"]),
            )
        )
    # Reserve capacity for both pair and three-member hypotheses when both
    # exist.  This prevents lexicographic pair enumeration from silently using
    # all four slots before any three-family ensemble is considered.
    result: list[dict[str, Any]] = []
    for size in (2, 3):
        result.extend(by_size[size][:2])
    if len(result) < MODEL_ENSEMBLE_MAX_CANDIDATES:
        selected_names = {str(item["name"]) for item in result}
        remainder = [
            item
            for size in (2, 3)
            for item in by_size[size]
            if str(item["name"]) not in selected_names
        ]
        remainder.sort(
            key=lambda item: (
                float(item["maximum_pair_correlation"]),
                len(item["components"]),
                str(item["name"]),
            )
        )
        result.extend(
            remainder[: MODEL_ENSEMBLE_MAX_CANDIDATES - len(result)]
        )
    for item in result:
        item.pop("maximum_pair_correlation", None)
    return result
