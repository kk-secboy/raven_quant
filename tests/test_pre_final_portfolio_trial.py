from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.model_research_governance import canonical_sha256
from scripts.run_multifactor_backtest import (
    FORMAL_FINAL_OOS_MODE,
    PRE_FINAL_PORTFOLIO_TRIAL_MODE,
    _evaluation_mode,
    _frozen_model_engine,
    _model_execution_authorization,
    _pre_final_execution_periods,
)

pytestmark = pytest.mark.no_database


def _candidate() -> dict[str, str]:
    return {
        "pre_final_end": "2025-12-31",
        "final_oos_start": "2026-01-08",
        "final_oos_end": "2026-08-25",
    }


def _model_periods() -> dict[str, str | int]:
    return {
        "train_start": "2020-01-02",
        "train_end": "2023-12-29",
        "valid_start": "2024-01-02",
        "valid_end": "2025-12-31",
        "seed": 11,
    }


def test_default_model_execution_mode_remains_formal_final_oos() -> None:
    assert _evaluation_mode({}) == FORMAL_FINAL_OOS_MODE
    authorization = _model_execution_authorization(
        evaluation_mode=FORMAL_FINAL_OOS_MODE,
        candidate_manifest=_candidate(),
        model_periods=_model_periods(),
        periods={"start": "2026-01-08", "end": "2026-08-25"},
        historical_periods={"start": "2014-01-02", "end": "2025-12-31"},
        pre_final_cutoff=None,
    )
    assert authorization == {
        "prediction_segment": "test",
        "final_oos_opened": True,
        "allow_final_oos": True,
        "allow_inference": False,
        "evaluation_scope": "final_oos_once",
    }


def test_pre_final_model_trial_never_authorizes_final_oos() -> None:
    authorization = _model_execution_authorization(
        evaluation_mode=PRE_FINAL_PORTFOLIO_TRIAL_MODE,
        candidate_manifest=_candidate(),
        model_periods=_model_periods(),
        periods={"start": "2025-01-02", "end": "2025-12-31"},
        historical_periods={"start": "2020-01-02", "end": "2023-12-29"},
        pre_final_cutoff="2025-12-31",
    )
    assert authorization["final_oos_opened"] is False
    assert authorization["allow_final_oos"] is False
    assert authorization["allow_inference"] is True
    assert authorization["evaluation_scope"] == "pre_final_only"

    with pytest.raises(ValueError, match="crosses"):
        _model_execution_authorization(
            evaluation_mode=PRE_FINAL_PORTFOLIO_TRIAL_MODE,
            candidate_manifest=_candidate(),
            model_periods=_model_periods(),
            periods={"start": "2025-01-02", "end": "2026-01-08"},
            historical_periods={"start": "2020-01-02", "end": "2023-12-29"},
            pre_final_cutoff="2025-12-31",
        )


def test_pre_final_trial_builds_purged_inner_validation_then_inference() -> None:
    calendar = pd.bdate_range("2024-01-02", "2025-01-02")
    periods = _pre_final_execution_periods(
        _model_periods(),
        {"start": "2025-01-02", "end": "2025-06-30"},
        calendar,
        embargo_sessions=5,
    )

    assert periods["test_start"] == "2025-01-02"
    assert pd.Timestamp(periods["valid_end"]) < pd.Timestamp(periods["test_start"])
    assert len(
        calendar[
            (calendar > pd.Timestamp(periods["valid_end"]))
            & (calendar < pd.Timestamp(periods["test_start"]))
        ]
    ) == 5


def test_model_engine_is_read_only_from_the_frozen_recipe() -> None:
    default_recipe = {"model_hyperparameters": {}}
    assert _frozen_model_engine(
        default_recipe, recipe_sha256=canonical_sha256(default_recipe)
    ) == "rdagent_pytorch"
    lightgbm_recipe = {
        "model_hyperparameters": {"model_engine": "lightgbm_baseline"}
    }
    assert (
        _frozen_model_engine(
            lightgbm_recipe,
            recipe_sha256=canonical_sha256(lightgbm_recipe),
        )
        == "lightgbm_baseline"
    )
    with pytest.raises(ValueError, match="changed after admission"):
        _frozen_model_engine(lightgbm_recipe, recipe_sha256="f" * 64)
    conflicting_recipe = {
        "model_engine": "ridge_baseline",
        "model_hyperparameters": {"model_engine": "lightgbm_baseline"},
    }
    with pytest.raises(ValueError, match="conflicting"):
        _frozen_model_engine(
            conflicting_recipe,
            recipe_sha256=canonical_sha256(conflicting_recipe),
        )
    invalid_recipe = {
        "model_hyperparameters": {"model_engine": "manifest_override"}
    }
    with pytest.raises(ValueError, match="ungoverned"):
        _frozen_model_engine(
            invalid_recipe,
            recipe_sha256=canonical_sha256(invalid_recipe),
        )
