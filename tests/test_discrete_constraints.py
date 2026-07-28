from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.discrete_constraints import validate_discrete_constraints
from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig

pytestmark = pytest.mark.no_database


def test_discrete_validation_records_all_account_hard_constraint_evidence() -> None:
    instruments = pd.Index(["stock", "etf"])
    report = validate_discrete_constraints(
        {"stock": 0.40, "etf": 0.30},
        {"stock": 0.30, "etf": 0.30},
        max_position_weight=0.50,
        max_daily_turnover=0.20,
        min_cash_weight=0.20,
        industries={"stock": "bank", "etf": "broad"},
        max_industry_weight=0.50,
        asset_classes={"stock": "stock", "etf": "etf"},
        max_asset_class_weights={"stock": 0.50, "etf": 0.50},
        average_daily_values={"stock": 10_000_000, "etf": 10_000_000},
        portfolio_value=1_000_000,
        max_volume_participation=0.01,
        prices={"stock": 10.0, "etf": 5.0},
        lot_size=100,
    )

    assert report["status"] == "passed"
    assert report["cash_weight"] == pytest.approx(0.30)
    assert {
        item["name"] for item in report["checks"]
    } >= {
        "cash_weight",
        "max_position_weight",
        "daily_turnover",
        "industry_weight",
        "asset_class_weight",
        "capacity_trade_value",
        "round_lot",
    }
    assert instruments.is_unique


def test_discrete_validation_reports_violations_without_relaxing_target() -> None:
    report = validate_discrete_constraints(
        {"one": 0.70, "two": 0.25},
        {},
        max_position_weight=0.50,
        max_daily_turnover=0.20,
        min_cash_weight=0.10,
    )

    assert report["status"] == "failed"
    assert {item["name"] for item in report["violations"]} == {
        "cash_weight",
        "max_position_weight",
        "daily_turnover",
    }


def test_configured_asset_limit_requires_complete_classification() -> None:
    with pytest.raises(ValueError, match="asset class memberships"):
        validate_discrete_constraints(
            {"one": 0.50},
            {},
            max_position_weight=0.50,
            max_daily_turnover=1.0,
            max_asset_class_weights={"stock": 0.60},
        )


def test_tracking_error_is_rechecked_after_discretization() -> None:
    covariance = pd.DataFrame(
        np.eye(2) * 0.0001,
        index=["one", "two"],
        columns=["one", "two"],
    )
    report = validate_discrete_constraints(
        {"one": 0.60, "two": 0.40},
        {},
        max_position_weight=0.70,
        max_daily_turnover=1.0,
        benchmark_weights={"one": 0.50, "two": 0.50},
        return_covariance=covariance,
        max_tracking_error=0.01,
    )

    assert report["status"] == "failed"
    violation = next(
        item for item in report["violations"] if item["name"] == "tracking_error"
    )
    assert violation["observed"] == pytest.approx(np.sqrt(0.0002 * 252) * 0.1)


def test_policy_persists_post_discrete_validation_evidence() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.40,
            max_daily_turnover=1.0,
            min_cash_weight=0.10,
            max_asset_class_weights={"stock": 0.60, "etf": 0.50},
        )
    )
    decision = policy.decide(
        pd.Series({"stock": 2.0, "etf": 1.0}),
        {},
        asset_classes=pd.Series({"stock": "stock", "etf": "etf"}),
        prices=pd.Series({"stock": 10.0, "etf": 5.0}),
        average_daily_values=pd.Series(
            {"stock": 100_000_000.0, "etf": 100_000_000.0}
        ),
        portfolio_value=1_000_000.0,
    )

    evidence = decision.position_state["discrete_constraint_validation"]
    assert evidence["status"] == "passed"
    assert evidence["cash_weight"] >= 0.10


def test_policy_fails_closed_when_discrete_contract_is_violated() -> None:
    policy = PortfolioPolicy(
        PortfolioPolicyConfig(
            topk=2,
            n_drop=0,
            max_position_weight=0.50,
            max_daily_turnover=1.0,
            min_cash_weight=0.10,
        )
    )

    with pytest.raises(
        ValueError,
        match="post-discretization hard constraint violation: cash_weight",
    ):
        policy.decide(pd.Series({"one": 2.0, "two": 1.0}), {})
