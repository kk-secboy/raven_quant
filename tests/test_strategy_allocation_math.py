from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qlib_test_doubles import (
    QlibPortfolioOptimizer,
    QlibRiskEstimator,
    qlib_runtime_identity,
)

from quant_platform.allocation_store import AllocationStore
from quant_platform.risk_math import COVARIANCE_MODEL_VERSION
from quant_platform.strategy_allocation import _capped, analyze_strategy_allocation

pytestmark = pytest.mark.no_database


def test_capped_weight_projection_keeps_already_capped_members_fixed() -> None:
    projected = _capped(np.array([0.60, 0.35, 0.05]), 0.40)

    assert projected == pytest.approx([0.40, 0.40, 0.20])
    assert projected.sum() == pytest.approx(1.0)
    assert projected.max() <= 0.40 + 1e-12


def test_hypothesis_group_members_share_one_cap_and_trial_count() -> None:
    analysis = {
        "members": {
            "version-a": {"target_weight": 0.30},
            "version-b": {"target_weight": 0.25},
        },
        "solver": {},
    }
    governance = {
        version_id: {
            "economic_hypothesis_group": "same-idea",
            "hypothesis_group_cap": 0.50,
            "shared_experiment_count": 7,
        }
        for version_id in analysis["members"]
    }
    with pytest.raises(ValueError, match="shared capital cap"):
        AllocationStore._apply_hypothesis_group_governance(
            analysis, governance
        )

    analysis["members"]["version-b"]["target_weight"] = 0.20
    governed = AllocationStore._apply_hypothesis_group_governance(
        analysis, governance
    )
    group = governed["economic_hypothesis_groups"]["same-idea"]
    assert group["target_weight"] == pytest.approx(0.50)
    assert group["shared_experiment_count"] == 7
    assert group["member_strategy_version_ids"] == ["version-a", "version-b"]


def test_risk_parity_fails_closed_on_pathological_negative_risk_contributions() -> None:
    random = np.random.default_rng(7)
    first = random.normal(0.0, 0.01, 252)
    second = -1.8 * first + random.normal(0.0, 0.0005, 252)
    returns = pd.DataFrame(
        {"first": first, "second": second}, index=pd.bdate_range("2024-01-02", periods=252)
    )

    with pytest.raises(ValueError, match="Qlib risk parity result"):
        analyze_strategy_allocation(
            returns,
            method="risk_parity",
            lookback_days=252,
            target_volatility=0.20,
            max_pairwise_correlation=0.90,
            max_strategy_weight=0.90,
            optimizer_factory=QlibPortfolioOptimizer,
            risk_estimator_factory=QlibRiskEstimator,
            runtime_identity=qlib_runtime_identity,
        )


def test_risk_parity_uses_pinned_qlib_optimizer_and_risk_model() -> None:
    random = np.random.default_rng(11)
    returns = pd.DataFrame(
        random.normal(0.0, [0.01, 0.015], size=(252, 2)),
        columns=["first", "second"],
        index=pd.bdate_range("2024-01-02", periods=252),
    )
    analysis = analyze_strategy_allocation(
        returns,
        method="risk_parity",
        lookback_days=252,
        target_volatility=0.20,
        max_pairwise_correlation=0.90,
        max_strategy_weight=0.90,
        optimizer_factory=QlibPortfolioOptimizer,
        risk_estimator_factory=QlibRiskEstimator,
        runtime_identity=qlib_runtime_identity,
    )

    contributions = [item["risk_contribution"] for item in analysis["members"].values()]
    assert min(contributions) >= 0.0
    assert max(contributions) - min(contributions) <= 0.04
    assert analysis["solver"]["success"] is True
    assert analysis["solver"]["maximum_risk_budget_error"] <= 0.02
    assert analysis["covariance_model_version"] == COVARIANCE_MODEL_VERSION
    assert analysis["solver"]["engine"].endswith("PortfolioOptimizer")


def test_risk_parity_maps_governed_member_budgets_from_qlib_solution() -> None:
    random = np.random.default_rng(19)
    returns = pd.DataFrame(
        random.normal(0.0, 0.01, size=(504, 2)),
        columns=["core", "satellite"],
        index=pd.bdate_range("2024-01-02", periods=504),
    )
    analysis = analyze_strategy_allocation(
        returns,
        method="risk_parity",
        lookback_days=252,
        target_volatility=0.20,
        max_pairwise_correlation=0.90,
        max_strategy_weight=0.90,
        risk_budgets={"core": 0.80, "satellite": 0.20},
        optimizer_factory=QlibPortfolioOptimizer,
        risk_estimator_factory=QlibRiskEstimator,
        runtime_identity=qlib_runtime_identity,
    )

    assert analysis["solver"]["engine"].endswith("PortfolioOptimizer")
    assert analysis["solver"]["constraint_wrapper"] == "project_risk_budget_mapping_v1"
    assert analysis["solver"]["maximum_risk_budget_error"] <= 0.02
    assert analysis["members"]["core"]["risk_budget"] == pytest.approx(0.80)
    assert analysis["members"]["satellite"]["risk_budget"] == pytest.approx(0.20)
    assert analysis["members"]["core"]["risk_contribution"] == pytest.approx(0.80, abs=0.02)
