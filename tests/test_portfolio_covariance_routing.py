from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = pytest.mark.no_database

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def covariance_modules() -> tuple[ModuleType, ...]:
    return (
        _load_script(
            "run_multifactor_backtest_covariance_test",
            "scripts/run_multifactor_backtest.py",
        ),
        _load_script(
            "run_recommendation_refresh_covariance_test",
            "scripts/run_recommendation_refresh.py",
        ),
    )


def _incomplete_close_history() -> tuple[pd.DataFrame, pd.Index]:
    dates = pd.bdate_range("2026-01-01", periods=61)
    history = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    history.loc[dates[10], "B"] = float("nan")
    return history, pd.Index(["A", "B"], dtype=str)


def test_topk_policies_do_not_invoke_optimizer_covariance(
    covariance_modules: tuple[ModuleType, ...],
) -> None:
    history, instruments = _incomplete_close_history()

    for module in covariance_modules:
        with patch.object(module, "estimate_covariance") as estimator:
            result = module._portfolio_return_covariance(
                {"portfolio_construction": "topk_equal_weight"},
                history,
                instruments,
            )

        assert result is None
        estimator.assert_not_called()


@pytest.mark.parametrize("construction", ["benchmark_relative_qp", "industry_neutral_qp"])
def test_qp_policies_keep_complete_history_gate(
    covariance_modules: tuple[ModuleType, ...],
    construction: str,
) -> None:
    history, instruments = _incomplete_close_history()

    for module in covariance_modules:
        with patch.object(module, "estimate_covariance") as estimator:
            with pytest.raises(ValueError, match="requires 60 complete"):
                module._portfolio_return_covariance(
                    {"portfolio_construction": construction},
                    history,
                    instruments,
                )

        estimator.assert_not_called()
