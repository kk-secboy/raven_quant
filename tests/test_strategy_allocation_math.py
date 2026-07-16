from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.risk_math import COVARIANCE_MODEL_VERSION
from quant_platform.strategy_allocation import analyze_strategy_allocation

pytestmark = pytest.mark.no_database


def test_risk_parity_handles_strong_negative_correlation_without_negative_contribution() -> None:
    random = np.random.default_rng(7)
    first = random.normal(0.0, 0.01, 252)
    second = -1.8 * first + random.normal(0.0, 0.0005, 252)
    returns = pd.DataFrame(
        {"first": first, "second": second}, index=pd.bdate_range("2024-01-02", periods=252)
    )

    analysis = analyze_strategy_allocation(
        returns,
        method="risk_parity",
        lookback_days=252,
        target_volatility=0.20,
        max_pairwise_correlation=0.90,
        max_strategy_weight=0.90,
    )

    contributions = [item["risk_contribution"] for item in analysis["members"].values()]
    assert min(contributions) >= 0.0
    assert max(contributions) - min(contributions) <= 0.04
    assert analysis["solver"]["success"] is True
    assert analysis["solver"]["maximum_risk_budget_error"] <= 0.02
    assert analysis["covariance_model_version"] == COVARIANCE_MODEL_VERSION
