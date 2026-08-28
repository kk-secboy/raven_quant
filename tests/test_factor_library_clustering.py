from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.no_database


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cluster_factor_library.py"
SPEC = importlib.util.spec_from_file_location("cluster_factor_library", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ranked_pairwise_correlation_matches_spearman_without_missing() -> None:
    rng = np.random.default_rng(20260826)
    values = rng.normal(size=(120, 8))

    actual = MODULE._ranked_pairwise_correlation(values)
    expected = pd.DataFrame(values).corr(method="spearman", min_periods=20).to_numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_ranked_pairwise_correlation_uses_daily_available_rank_universe() -> None:
    values = np.array(
        [
            [1.0, 2.0, 9.0],
            [2.0, np.nan, 7.0],
            [3.0, 1.0, 5.0],
            [4.0, 4.0, np.nan],
            [5.0, 3.0, 1.0],
        ]
    )
    ranked = pd.DataFrame(values).rank(axis=0, method="average", na_option="keep")
    expected = ranked.corr(method="pearson", min_periods=3).to_numpy()

    actual = MODULE._ranked_pairwise_correlation(values, min_periods=3)

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_ranked_pairwise_correlation_rejects_too_little_overlap() -> None:
    values = np.array(
        [
            [1.0, 1.0],
            [2.0, np.nan],
            [3.0, 3.0],
            [4.0, np.nan],
        ]
    )

    actual = MODULE._ranked_pairwise_correlation(values, min_periods=3)

    assert np.isnan(actual[0, 1])
    assert actual[0, 0] == pytest.approx(1.0)


def test_governed_clusters_do_not_single_link_a_correlation_chain() -> None:
    names = ["a", "b", "c"]
    similarity = np.array(
        [
            [1.0, 0.80, 0.20],
            [0.80, 1.0, 0.80],
            [0.20, 0.80, 1.0],
        ]
    )

    assignments = MODULE._governed_cluster_assignments(
        names=names,
        similarity=similarity,
        cluster_threshold=0.75,
        near_duplicate_threshold=0.95,
    )

    assert len(set(assignments.values())) == 2
    assert not (
        assignments["a"] == assignments["b"] == assignments["c"]
    )


def test_governed_clusters_keep_near_duplicate_components_together() -> None:
    names = ["a", "b", "c"]
    similarity = np.array(
        [
            [1.0, 0.96, 0.90],
            [0.96, 1.0, 0.96],
            [0.90, 0.96, 1.0],
        ]
    )

    assignments = MODULE._governed_cluster_assignments(
        names=names,
        similarity=similarity,
        cluster_threshold=0.75,
        near_duplicate_threshold=0.95,
    )

    assert assignments["a"] == assignments["b"] == assignments["c"]


def test_materialization_matrix_and_daily_checkpoint_are_resumable(tmp_path: Path) -> None:
    materialization = tmp_path / "materialization"
    values_dir = materialization / "values"
    output = materialization / "clusters"
    values_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-02", periods=65, freq="B")
    instruments = [f"SH{index:06d}" for index in range(25)]
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    base = np.tile(np.arange(25, dtype=np.float64), len(dates))
    factor_values = {
        "factor-a": base,
        "factor-b": -base,
        "factor-c": np.sin(base),
    }
    completed = {}
    for name, values in factor_values.items():
        relative_path = f"values/{name}.h5"
        pd.DataFrame({name: values}, index=index).to_hdf(
            materialization / relative_path, key="factor", mode="w"
        )
        completed[name] = {
            "relative_path": relative_path,
            "sha256": (name.encode().hex() + "0" * 64)[:64],
        }
    manifest = {
        "dataset_identity_sha256": "a" * 64,
        "feature_set_id": "test",
        "feature_set_definition_sha256": "b" * 64,
        "completed": completed,
    }
    (materialization / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    names = sorted(completed)
    fingerprint = MODULE._materialization_fingerprint(manifest, names)

    matrix_path, offsets, row_count = MODULE._build_matrix(
        materialization=materialization,
        output=output,
        manifest=manifest,
        names=names,
        fingerprint=fingerprint,
    )
    absolute_sum, eligible_days = MODULE._correlation_matrices(
        matrix_path=matrix_path,
        output=output,
        offsets=offsets,
        definition_count=len(names),
        row_count=row_count,
        fingerprint=fingerprint,
        checkpoint_every_days=10,
    )
    resumed_sum, resumed_days = MODULE._correlation_matrices(
        matrix_path=matrix_path,
        output=output,
        offsets=offsets,
        definition_count=len(names),
        row_count=row_count,
        fingerprint=fingerprint,
        checkpoint_every_days=10,
    )

    assert int(eligible_days[0, 1]) == len(dates)
    assert absolute_sum[0, 1] / eligible_days[0, 1] == pytest.approx(1.0)
    np.testing.assert_array_equal(resumed_days, eligible_days)
    np.testing.assert_allclose(resumed_sum, absolute_sum)
