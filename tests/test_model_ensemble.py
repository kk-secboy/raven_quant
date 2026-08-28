from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_platform.model_ensemble import (
    EnsemblePredictionsPending,
    bounded_ensemble_combinations,
    daily_rank_correlation,
    equal_rank_predictions,
)
from quant_platform.model_strategy_contract import (
    MODEL_ENSEMBLE_SIGNAL_CONTRACT_VERSION,
    model_signal_identity,
    normalize_model_signal_config,
)

pytestmark = pytest.mark.no_database


def _predictions(order: list[int], *, days: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    instruments = [f"SH{index:06d}" for index in range(len(order))]
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    values = []
    for day in range(days):
        values.extend(float(value) + day * 0.0001 for value in order)
    return pd.DataFrame({"score": values}, index=index)


def test_daily_rank_correlation_rejects_near_duplicate_predictions() -> None:
    left = _predictions(list(range(60)))
    right = _predictions(list(range(60)))
    evidence = daily_rank_correlation(left, right)
    assert evidence["passed"] is False
    assert evidence["mean_absolute_correlation"] == pytest.approx(1.0)


def test_equal_rank_predictions_are_daily_cross_sectional_not_stacking() -> None:
    left = _predictions(list(range(60)))
    right = _predictions([value * 7 % 60 for value in range(60)])
    correlation = daily_rank_correlation(left, right)
    assert correlation["passed"] is True
    combined, evidence = equal_rank_predictions((("ridge", left), ("gru", right)))
    first_day = combined.xs(combined.index.levels[0][0], level="datetime")["score"]
    expected = (
        left.xs(left.index.levels[0][0], level="datetime")["score"].rank(pct=True)
        + right.xs(right.index.levels[0][0], level="datetime")["score"].rank(pct=True)
    ) / 2.0
    assert first_day.to_dict() == pytest.approx(expected.to_dict())
    assert evidence["combiner"] == "equal_rank"
    assert evidence["stacking"] is False
    assert evidence["weights"] == [0.5, 0.5]


def test_rank_correlation_fails_closed_when_dates_do_not_match() -> None:
    left = _predictions(list(range(60)), days=60)
    right = _predictions(list(range(60)), days=59)
    with pytest.raises(EnsemblePredictionsPending, match="same pre-final trading days"):
        daily_rank_correlation(left, right)


def test_bounded_combinations_use_distinct_families_and_four_candidate_cap() -> None:
    champions = [
        {
            "id": family,
            "model_family": family,
            "manifest_sha256": character * 64,
            "admission_evidence_sha256": character.upper() * 64,
            "prediction_grid_sha256": str(index) * 64,
        }
        for index, (family, character) in enumerate(
            (("ridge", "a"), ("lightgbm", "b"), ("gru", "c"), ("transformer", "d")),
            start=1,
        )
    ]
    pairwise = {}
    for left_index, left in enumerate(champions):
        for right in champions[left_index + 1 :]:
            pairwise[(left["id"], right["id"])] = {
                "passed": True,
                "maximum_mean_absolute_daily_rank_correlation": 0.4,
            }
    specs = bounded_ensemble_combinations(champions, pairwise)
    assert len(specs) == 4
    assert all(2 <= len(item["components"]) <= 3 for item in specs)
    assert {len(item["components"]) for item in specs} == {2, 3}
    assert all(
        math.isclose(
            sum(float(component["weight"]) for component in item["components"]), 1.0
        )
        for item in specs
    )
    assert all(item["name"].startswith("equal-rank-") for item in specs)


def test_strategy_contract_freezes_equal_rank_components_without_single_model_alias() -> None:
    config = {
        "signal_source": "model_prediction",
        "model_ensemble_candidate_id": "ensemble-1",
        "model_ensemble_evaluation_id": "evaluation-1",
        "model_ensemble_manifest_sha256": "a" * 64,
        "model_ensemble_evidence_sha256": "b" * 64,
        "model_ensemble_combiner": "equal_rank",
        "model_ensemble_stacking": False,
        "model_component_candidate_ids": ["ridge-1", "gru-1"],
        "model_component_families": ["ridge", "gru"],
    }
    normalized = normalize_model_signal_config(config)
    identity = model_signal_identity(normalized)
    assert normalized["model_signal_contract_version"] == (
        MODEL_ENSEMBLE_SIGNAL_CONTRACT_VERSION
    )
    assert identity is not None
    assert identity["model_component_candidate_ids"] == ["ridge-1", "gru-1"]
    assert len(identity["identity_sha256"]) == 64

    with pytest.raises(ValueError, match="substituted single model"):
        normalize_model_signal_config({**config, "model_candidate_id": "ridge-1"})
    with pytest.raises(ValueError, match="non-stacking equal-rank"):
        normalize_model_signal_config({**config, "model_ensemble_stacking": True})
