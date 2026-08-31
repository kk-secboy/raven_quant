from __future__ import annotations

from copy import deepcopy

import pytest

from quant_platform.model_research_governance import (
    REQUIRED_MODEL_SEEDS,
    REQUIRED_RESEARCH_PROFILES,
)
from quant_platform.strategy_store import _pre_final_stability_failures

pytestmark = pytest.mark.no_database


def _factor_version() -> dict:
    return {
        "config": {
            "horizon_profile": "short_1_5d",
            "signal_source": "factor_score",
            "min_rolling_windows": 3,
            "min_rolling_pass_rate": 0.60,
        }
    }


def _factor_metrics() -> dict:
    return {
        # A final 252-session result can only contain one 252-session
        # observation. It is deliberately irrelevant to this pre-final gate.
        "rolling_window_count": 1,
        "rolling_pass_rate": 0.0,
        "formal_validation": {
            "outer_walk_forward": {
                "status": "completed",
                "passed": True,
                "fold_count": 3,
                "test_pass_rate": 2.0 / 3.0,
            }
        },
    }


def _model_version() -> dict:
    return {
        "config": {
            "horizon_profile": "short_1_5d",
            "signal_source": "model_prediction",
            "min_rolling_windows": 3,
            "min_rolling_pass_rate": 0.60,
        }
    }


def _model_metrics() -> dict:
    return {
        "rolling_window_count": 1,
        "rolling_pass_rate": 0.0,
        "formal_validation": {
            "model_admission": {
                "final_oos_opened": False,
                "model_grid": {
                    "profiles": list(REQUIRED_RESEARCH_PROFILES),
                    "seeds": list(REQUIRED_MODEL_SEEDS),
                    "cell_count": len(REQUIRED_RESEARCH_PROFILES)
                    * len(REQUIRED_MODEL_SEEDS),
                    "multiple_testing": {"gate_passed": True},
                },
            }
        },
    }


def test_factor_stability_uses_pre_final_outer_folds_not_final_rolling() -> None:
    assert _pre_final_stability_failures(_factor_version(), _factor_metrics()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("fold_count", 2), ("test_pass_rate", 0.50), ("passed", False)],
)
def test_factor_stability_fails_closed(field: str, value: object) -> None:
    metrics = _factor_metrics()
    metrics["formal_validation"]["outer_walk_forward"][field] = value

    assert "pre-final factor stability" in _pre_final_stability_failures(
        _factor_version(), metrics
    )[0]


def test_model_stability_uses_independent_profile_seed_grid() -> None:
    assert _pre_final_stability_failures(_model_version(), _model_metrics()) == []


def test_model_stability_rejects_incomplete_or_nonpassing_grid() -> None:
    incomplete = _model_metrics()
    incomplete["formal_validation"]["model_admission"]["model_grid"]["profiles"].pop()
    assert "pre-final model stability" in _pre_final_stability_failures(
        _model_version(), incomplete
    )[0]

    failed_multiple_testing = deepcopy(_model_metrics())
    failed_multiple_testing["formal_validation"]["model_admission"]["model_grid"][
        "multiple_testing"
    ]["gate_passed"] = False
    assert "pre-final model stability" in _pre_final_stability_failures(
        _model_version(), failed_multiple_testing
    )[0]


def test_legacy_stability_remains_on_legacy_approval_path() -> None:
    version = {
        "config": {
            "horizon_profile": "legacy_ambiguous",
            "min_rolling_windows": 3,
        }
    }
    assert _pre_final_stability_failures(version, {}) == []
