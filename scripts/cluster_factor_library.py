#!/usr/bin/env python3
"""Build resumable full-library daily Rank-correlation clusters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


CONTRACT_VERSION = "full-library-daily-rank-correlation-v2"
MIN_DAILY_INSTRUMENTS = 20
MIN_ELIGIBLE_DAYS = 60


def _load(path: Path, name: str) -> pd.Series:
    frame = pd.read_hdf(path)
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").rename(name)
    if series.index.names != ["datetime", "instrument"]:
        series = series.reorder_levels(["datetime", "instrument"])
    return series.sort_index()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def _materialization_fingerprint(
    manifest: dict[str, Any], names: list[str]
) -> str:
    completed = dict(manifest.get("completed") or {})
    evidence = {
        "contract_version": CONTRACT_VERSION,
        "dataset_identity_sha256": manifest.get("dataset_identity_sha256"),
        "feature_set_definition_sha256": manifest.get(
            "feature_set_definition_sha256"
        ),
        "factors": [
            {
                "id": name,
                "sha256": completed[name]["sha256"],
                "relative_path": completed[name]["relative_path"],
            }
            for name in names
        ],
    }
    return hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _date_offsets(index: pd.MultiIndex) -> np.ndarray:
    datetimes = pd.DatetimeIndex(index.get_level_values("datetime")).asi8
    if len(datetimes) == 0:
        raise ValueError("factor materialization has no rows")
    changes = np.flatnonzero(datetimes[1:] != datetimes[:-1]) + 1
    return np.concatenate(
        (
            np.array([0], dtype=np.int64),
            changes.astype(np.int64, copy=False),
            np.array([len(datetimes)], dtype=np.int64),
        )
    )


def _build_matrix(
    *,
    materialization: Path,
    output: Path,
    manifest: dict[str, Any],
    names: list[str],
    fingerprint: str,
) -> tuple[Path, np.ndarray, int]:
    """Materialize each HDF factor once into a resumable factor-major memmap."""

    output.mkdir(parents=True, exist_ok=True)
    completed = dict(manifest.get("completed") or {})
    matrix_path = output / "factor_matrix.f64"
    offsets_path = output / "date_offsets.npy"
    metadata_path = output / "matrix_metadata.json"
    reference = _load(
        materialization / completed[names[0]]["relative_path"], names[0]
    )
    reference_index = reference.index
    row_count = len(reference)
    offsets = _date_offsets(reference_index)
    expected_size = len(names) * row_count * np.dtype(np.float64).itemsize
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else None
    )
    reusable = bool(
        isinstance(metadata, dict)
        and metadata.get("contract_version") == CONTRACT_VERSION
        and metadata.get("materialization_fingerprint") == fingerprint
        and int(metadata.get("row_count", -1)) == row_count
        and int(metadata.get("definition_count", -1)) == len(names)
        and matrix_path.is_file()
        and matrix_path.stat().st_size == expected_size
        and offsets_path.is_file()
    )
    if reusable:
        stored_offsets = np.load(offsets_path, allow_pickle=False)
        if not np.array_equal(stored_offsets, offsets):
            raise ValueError("factor matrix date offsets do not match source index")
        next_factor = int(metadata.get("next_factor", 0))
        if not 0 <= next_factor <= len(names):
            raise ValueError("factor matrix checkpoint is invalid")
        matrix = np.memmap(
            matrix_path,
            dtype=np.float64,
            mode="r+",
            shape=(len(names), row_count),
        )
    else:
        matrix = np.memmap(
            matrix_path,
            dtype=np.float64,
            mode="w+",
            shape=(len(names), row_count),
        )
        _atomic_npy(offsets_path, offsets)
        metadata = {
            "contract_version": CONTRACT_VERSION,
            "materialization_fingerprint": fingerprint,
            "row_count": row_count,
            "definition_count": len(names),
            "date_count": len(offsets) - 1,
            "next_factor": 0,
        }
        _atomic_json(metadata_path, metadata)
        next_factor = 0

    for factor_index in range(next_factor, len(names)):
        name = names[factor_index]
        values = (
            reference
            if factor_index == 0
            else _load(materialization / completed[name]["relative_path"], name)
        )
        if not values.index.equals(reference_index):
            raise ValueError(f"factor index mismatch: {name}")
        matrix[factor_index, :] = values.to_numpy(dtype=np.float64, na_value=np.nan)
        matrix.flush()
        metadata["next_factor"] = factor_index + 1
        _atomic_json(metadata_path, metadata)
        print(
            json.dumps(
                {
                    "stage": "matrix",
                    "completed": factor_index + 1,
                    "total": len(names),
                }
            ),
            flush=True,
        )
    del matrix
    return matrix_path, offsets, row_count


def _ranked_pairwise_correlation(
    values: np.ndarray, *, min_periods: int = MIN_DAILY_INSTRUMENTS
) -> np.ndarray:
    """Correlate cross-sectional ranks with pairwise-complete observations.

    Each factor is ranked once in its PIT-available daily universe. Pearson
    correlation is then calculated for every pair over their intersection.
    This is the standard cross-sectional rank-correlation definition and avoids
    re-ranking the same factor hundreds of times for one trading day.
    """

    ranks = (
        pd.DataFrame(values, copy=False)
        .rank(axis=0, method="average", na_option="keep")
        .to_numpy(dtype=np.float64, copy=False)
    )
    mask = np.isfinite(ranks).astype(np.float64)
    ranked = np.nan_to_num(ranks, copy=True, nan=0.0)
    count = mask.T @ mask
    safe_count = np.maximum(count, 1.0)
    sums = ranked.T @ mask
    squared_sums = (ranked * ranked).T @ mask
    cross = ranked.T @ ranked
    covariance = cross - (sums * sums.T) / safe_count
    left_variance = squared_sums - (sums * sums) / safe_count
    right_variance = squared_sums.T - (sums.T * sums.T) / safe_count
    denominator = np.sqrt(
        np.maximum(left_variance, 0.0) * np.maximum(right_variance, 0.0)
    )
    correlations = np.full_like(cross, np.nan)
    np.divide(
        covariance,
        denominator,
        out=correlations,
        where=(count >= min_periods) & (denominator > 0),
    )
    correlations = (correlations + correlations.T) / 2.0
    return np.clip(correlations, -1.0, 1.0)


def _correlation_matrices(
    *,
    matrix_path: Path,
    output: Path,
    offsets: np.ndarray,
    definition_count: int,
    row_count: int,
    fingerprint: str,
    checkpoint_every_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint_path = output / "correlation_checkpoint.npz"
    if checkpoint_path.is_file():
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            checkpoint_fingerprint = str(
                checkpoint["materialization_fingerprint"].item()
            )
            if checkpoint_fingerprint != fingerprint:
                raise ValueError("correlation checkpoint belongs to another materialization")
            next_date = int(checkpoint["next_date"].item())
            absolute_sum = checkpoint["absolute_sum"].copy()
            eligible_days = checkpoint["eligible_days"].copy()
    else:
        next_date = 0
        absolute_sum = np.zeros((definition_count, definition_count), dtype=np.float64)
        eligible_days = np.zeros((definition_count, definition_count), dtype=np.uint32)
    expected_shape = (definition_count, definition_count)
    if absolute_sum.shape != expected_shape or eligible_days.shape != expected_shape:
        raise ValueError("correlation checkpoint matrix shape is invalid")
    date_count = len(offsets) - 1
    if not 0 <= next_date <= date_count:
        raise ValueError("correlation checkpoint date is invalid")
    matrix = np.memmap(
        matrix_path,
        dtype=np.float64,
        mode="r",
        shape=(definition_count, row_count),
    )
    for date_index in range(next_date, date_count):
        start, end = int(offsets[date_index]), int(offsets[date_index + 1])
        if end - start >= MIN_DAILY_INSTRUMENTS:
            daily_values = np.ascontiguousarray(matrix[:, start:end].T)
            daily_correlation = _ranked_pairwise_correlation(daily_values)
            valid = np.isfinite(daily_correlation)
            absolute_sum[valid] += np.abs(daily_correlation[valid])
            eligible_days[valid] += 1
        completed_dates = date_index + 1
        if (
            completed_dates % checkpoint_every_days == 0
            or completed_dates == date_count
        ):
            _atomic_npz(
                checkpoint_path,
                contract_version=np.array(CONTRACT_VERSION),
                materialization_fingerprint=np.array(fingerprint),
                next_date=np.array(completed_dates, dtype=np.int64),
                absolute_sum=absolute_sum,
                eligible_days=eligible_days,
            )
            print(
                json.dumps(
                    {
                        "stage": "correlation",
                        "completed": completed_dates,
                        "total": date_count,
                    }
                ),
                flush=True,
            )
    del matrix
    return absolute_sum, eligible_days


def _governed_cluster_assignments(
    *,
    names: list[str],
    similarity: np.ndarray,
    cluster_threshold: float,
    near_duplicate_threshold: float,
) -> dict[str, str]:
    """Create deterministic clusters without low-threshold single-link chaining.

    Near duplicates are first treated as equivalence components. Complete-linkage
    clustering is then applied to those components, using the weakest pairwise
    similarity between components as their merge criterion. Therefore an A-B and
    B-C chain at 0.75 cannot merge A and C when their direct similarity is low.
    """

    definition_count = len(names)
    if similarity.shape != (definition_count, definition_count):
        raise ValueError("factor similarity matrix shape is invalid")
    parents = list(range(definition_count))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left_index in range(definition_count - 1):
        for right_index in range(left_index + 1, definition_count):
            value = similarity[left_index, right_index]
            if np.isfinite(value) and value >= near_duplicate_threshold:
                union(left_index, right_index)
    components_by_root: dict[int, list[int]] = {}
    for index in range(definition_count):
        components_by_root.setdefault(find(index), []).append(index)
    base_components = sorted(
        components_by_root.values(), key=lambda members: tuple(names[i] for i in members)
    )
    if len(base_components) == 1:
        labels = np.ones(1, dtype=np.int64)
    else:
        component_distance = np.zeros(
            (len(base_components), len(base_components)), dtype=np.float64
        )
        for left_index, left_members in enumerate(base_components[:-1]):
            for right_index in range(left_index + 1, len(base_components)):
                right_members = base_components[right_index]
                cross_values = similarity[np.ix_(left_members, right_members)]
                finite = cross_values[np.isfinite(cross_values)]
                weakest_similarity = float(finite.min()) if finite.size else 0.0
                distance = 1.0 - weakest_similarity
                component_distance[left_index, right_index] = distance
                component_distance[right_index, left_index] = distance
        hierarchy = linkage(
            squareform(component_distance, checks=False),
            method="complete",
            optimal_ordering=True,
        )
        labels = fcluster(
            hierarchy,
            t=1.0 - cluster_threshold,
            criterion="distance",
        )
    grouped_members: dict[int, list[str]] = {}
    for component, label in zip(base_components, labels, strict=True):
        grouped_members.setdefault(int(label), []).extend(names[index] for index in component)
    assignments: dict[str, str] = {}
    for members in grouped_members.values():
        digest = hashlib.sha256(":".join(sorted(members)).encode("utf-8")).hexdigest()
        cluster_id = f"definition-cluster-{digest[:20]}"
        assignments.update({name: cluster_id for name in members})
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cluster-threshold", type=float, default=0.75)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--checkpoint-every-days", type=int, default=25)
    args = parser.parse_args()
    if not 0 < args.cluster_threshold < args.near_duplicate_threshold <= 1:
        raise ValueError("factor clustering thresholds are invalid")
    if args.checkpoint_every_days < 1:
        raise ValueError("checkpoint interval must be positive")

    materialization = Path(args.materialization).resolve(strict=True)
    manifest = json.loads(
        (materialization / "manifest.json").read_text(encoding="utf-8")
    )
    completed = dict(manifest.get("completed") or {})
    names = sorted(completed)
    if len(names) < 2:
        raise ValueError("factor clustering requires at least two materialized factors")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = _materialization_fingerprint(manifest, names)
    matrix_path, offsets, row_count = _build_matrix(
        materialization=materialization,
        output=output,
        manifest=manifest,
        names=names,
        fingerprint=fingerprint,
    )
    absolute_sum, eligible_days = _correlation_matrices(
        matrix_path=matrix_path,
        output=output,
        offsets=offsets,
        definition_count=len(names),
        row_count=row_count,
        fingerprint=fingerprint,
        checkpoint_every_days=args.checkpoint_every_days,
    )

    similarity = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    np.fill_diagonal(similarity, 1.0)
    correlated_pairs: list[dict[str, Any]] = []
    insufficient_pair_count = 0
    for left_index, left_name in enumerate(names[:-1]):
        for right_index in range(left_index + 1, len(names)):
            right_name = names[right_index]
            day_count = int(eligible_days[left_index, right_index])
            if day_count < MIN_ELIGIBLE_DAYS:
                insufficient_pair_count += 1
                continue
            correlation = float(absolute_sum[left_index, right_index] / day_count)
            similarity[left_index, right_index] = correlation
            similarity[right_index, left_index] = correlation
            if correlation > args.cluster_threshold:
                correlated_pairs.append(
                    {
                        "left_factor_definition_id": left_name,
                        "right_factor_definition_id": right_name,
                        "mean_abs_spearman": correlation,
                        "eligible_day_count": day_count,
                        "relationship": (
                            "near_duplicate"
                            if correlation >= args.near_duplicate_threshold
                            else "clustered"
                        ),
                    }
                )
    assignments = _governed_cluster_assignments(
        names=names,
        similarity=similarity,
        cluster_threshold=args.cluster_threshold,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    within_cluster_pairs = [
        edge
        for edge in correlated_pairs
        if assignments[str(edge["left_factor_definition_id"])]
        == assignments[str(edge["right_factor_definition_id"])]
    ]
    cross_cluster_pairs = [
        edge
        for edge in correlated_pairs
        if assignments[str(edge["left_factor_definition_id"])]
        != assignments[str(edge["right_factor_definition_id"])]
    ]
    completed_edges = [
        {
            **edge,
            "cluster_id": assignments[str(edge["left_factor_definition_id"])],
            "evidence": {
                "contract_version": CONTRACT_VERSION,
                "rank_universe": "factor_daily_pit_available_universe",
                "pair_observations": "intersection_after_independent_ranking",
                "left_values_sha256": completed[
                    str(edge["left_factor_definition_id"])
                ]["sha256"],
                "right_values_sha256": completed[
                    str(edge["right_factor_definition_id"])
                ]["sha256"],
            },
        }
        for edge in within_cluster_pairs
    ]
    expected_pair_count = len(names) * (len(names) - 1) // 2
    result = {
        "status": "complete",
        "contract_version": CONTRACT_VERSION,
        "clustering_method": "near_duplicate_components_then_complete_linkage",
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "feature_set_id": manifest["feature_set_id"],
        "feature_set_definition_sha256": manifest[
            "feature_set_definition_sha256"
        ],
        "definition_count": len(names),
        "pair_count": expected_pair_count,
        "expected_pair_count": expected_pair_count,
        "insufficient_pair_count": insufficient_pair_count,
        "cluster_count": len(set(assignments.values())),
        "assignments": assignments,
        "edges": completed_edges,
        "cross_cluster_correlations": cross_cluster_pairs,
    }
    _atomic_json(output / "result.json", result)


if __name__ == "__main__":
    main()
