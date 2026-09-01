from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.model_ensemble import (
    EnsemblePredictionsPending,
    bounded_ensemble_combinations,
    build_model_ensemble_label_contract,
    daily_rank_correlation,
    equal_rank_predictions,
    validate_model_ensemble_label_contract,
)
from quant_platform.model_research_governance import canonical_sha256
from quant_platform.model_strategy_contract import (
    MODEL_ENSEMBLE_SIGNAL_CONTRACT_VERSION,
    model_signal_identity,
    normalize_model_signal_config,
)
from quant_platform.research_horizon import (
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    research_horizon_contract,
)
from quant_platform.research_label_binding import resolve_research_label_binding

pytestmark = pytest.mark.no_database


def test_ensemble_materializes_qlib_signal_record_dependencies() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "evaluate_model_ensemble.py"
    ).read_text(encoding="utf-8")
    save_index = source.index('"pred.pkl": combined[["score"]]')
    boundary_index = source.index("resolve_qlib_portfolio_calendar_boundary(")
    portfolio_index = source.index("record = PortAnaRecord(")
    assert '"label.pkl": labels.to_frame("label")' in source
    assert '"signal": "<PRED>"' in source
    assert boundary_index < save_index < portfolio_index
    assert '"class": "GovernedDPlusOneTopkDropoutStrategy"' in source
    assert '"module_path": "quant_platform.qlib_research_strategy"' in source
    assert '"research_execution_cadence": execution_cadence' in source
    assert '"end_time": periods["valid_end"]' in source
    assert "fields=[label_expression]" in source
    assert 'fields=["Ref($close, -2)/Ref($close, -1)-1"]' not in source
    assert "Qlib ensemble portfolio record generation was skipped" in source


def _label_binding(profile: str, *, feature_id: str, feature_sha256: str) -> dict:
    horizon = research_horizon_contract(profile)
    periods = {
        "train_start": "2010-01-04",
        "train_end": "2018-12-28",
        "valid_start": "2019-01-02",
        "valid_end": "2022-12-30",
        "test_start": "2024-01-03",
        "test_end": "2026-08-20",
    }
    feature_set = {
        "id": feature_id,
        "definition_sha256": feature_sha256,
        "features": {"alpha": {"expression": "$close"}},
    }
    window = {
        "contract_version": "research-window-v1",
        "horizon_profile": profile,
        "horizon_contract_sha256": horizon.sha256,
        "dataset_name": "cn-governed-day",
        "dataset_identity_sha256": "d" * 64,
        "feature_set_id": feature_id,
        "feature_set_sha256": feature_sha256,
        "label_horizons_sessions": list(horizon.label_horizons_sessions),
        "purge_sessions": horizon.purge_sessions,
        "embargo_sessions": horizon.embargo_sessions,
        "label_maturity_enforced": True,
        "periods": periods,
    }
    binding = resolve_research_label_binding(
        {
            "horizon_profile": profile,
            "dataset": window["dataset_name"],
            "dataset_identity_sha256": window["dataset_identity_sha256"],
            "periods": periods,
            "feature_set": feature_set,
            "research_window_contract": window,
            "research_window_contract_sha256": canonical_sha256(window),
        }
    )
    assert binding is not None
    return binding


@pytest.mark.parametrize(
    ("profile", "expected_horizon"),
    ((SHORT_1_5D, 5), (SWING_1_6M, 63), (LONG_1_3Y, 252)),
)
def test_ensemble_label_contract_uses_each_horizons_frozen_forward_return(
    profile: str, expected_horizon: int
) -> None:
    bindings = {
        "ridge-1": _label_binding(
            profile, feature_id="alpha158", feature_sha256="a" * 64
        ),
        "gru-1": _label_binding(
            profile, feature_id="alpha360", feature_sha256="b" * 64
        ),
    }

    contract = build_model_ensemble_label_contract(bindings)

    assert contract["label_identity"]["horizon_profile"] == profile
    assert contract["label_identity"]["label_horizon_sessions"] == expected_horizon
    assert contract["label_identity"]["label_expression"] == (
        f"Ref($close,-{expected_horizon + 1})/Ref($close,-1)-1"
    )
    assert validate_model_ensemble_label_contract(
        contract, member_bindings=bindings
    ) == contract


def test_ensemble_label_contract_rejects_members_from_different_horizons() -> None:
    bindings = {
        "short-1": _label_binding(
            SHORT_1_5D, feature_id="alpha158", feature_sha256="a" * 64
        ),
        "swing-1": _label_binding(
            SWING_1_6M, feature_id="alpha158", feature_sha256="a" * 64
        ),
    }

    with pytest.raises(ValueError, match="different frozen label targets"):
        build_model_ensemble_label_contract(bindings)


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
