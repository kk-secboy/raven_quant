import pytest

from quant_platform.cost_model import CostModelConfig
from quant_platform.transition_decision import (
    DECISION_FORCED,
    DECISION_HOLD,
    DECISION_SWITCH,
    NoTradeBandConfig,
    compare_transition_paths,
    estimate_transition_cost,
    expected_returns_from_scores,
)

pytestmark = pytest.mark.no_database

PRICES = {"SH600000": 10.0, "SH600001": 20.0, "SH600002": 5.0}


def _kwargs(**overrides):
    base = {
        "prices": PRICES,
        "portfolio_value": 1_000_000.0,
        "cost_model": CostModelConfig(),
    }
    base.update(overrides)
    return base


def test_transition_cost_is_two_sided_and_asset_aware() -> None:
    path = estimate_transition_cost(
        {"SH600000": 0.5},
        {"SH600001": 0.5},
        **_kwargs(),
    )
    legs = {leg["instrument"]: leg for leg in path["legs"]}
    sell = legs["SH600000"]
    buy = legs["SH600001"]
    assert sell["side"] == "sell" and buy["side"] == "buy"
    # The stock sell leg carries stamp duty; the buy leg does not.
    assert sell["breakdown"]["stamp_duty"] > 0
    assert buy["breakdown"]["stamp_duty"] == 0
    # Both legs carry commission, transfer fee, slippage and impact.
    for leg in (sell, buy):
        assert leg["breakdown"]["commission"] > 0
        assert leg["breakdown"]["slippage"] > 0
        assert leg["breakdown"]["market_impact"] > 0
    assert path["total_cny"] == pytest.approx(
        sell["cost_cny"] + buy["cost_cny"]
    )
    assert path["turnover"] == pytest.approx(0.5)
    assert path["cost_version"] == CostModelConfig().version


def test_two_path_comparison_reports_hold_and_switch_path_costs() -> None:
    decision = compare_transition_paths(
        {"SH600000": 0.5},
        {"SH600001": 0.5},
        **_kwargs(),
    )
    assert decision.hold_path["total_cny"] == 0.0
    assert decision.hold_path["action"] == "keep_current_target"
    assert decision.switch_path["total_cny"] > 0
    assert decision.policy_version
    assert decision.to_dict()["switch_path"]["legs"]


def test_cost_above_incremental_benefit_holds() -> None:
    # Tiny weight changes: mapped benefit cannot cover full transition cost.
    decision = compare_transition_paths(
        {"SH600000": 0.500, "SH600001": 0.300},
        {"SH600000": 0.505, "SH600001": 0.295},
        expected_returns={"SH600000": 0.001, "SH600001": 0.001},
        **_kwargs(),
    )
    assert decision.decision == DECISION_HOLD
    assert decision.benefit_mode == "calibrated_expected_returns"
    assert decision.incremental_benefit_cny is not None
    assert decision.incremental_benefit_cny <= decision.switch_path["total_cny"]
    assert any("does not cover" in reason for reason in decision.reasons)


def test_benefit_significantly_above_cost_switches() -> None:
    decision = compare_transition_paths(
        {"SH600000": 0.5},
        {"SH600001": 0.5},
        expected_returns={"SH600000": -0.05, "SH600001": 0.15},
        **_kwargs(),
    )
    assert decision.decision == DECISION_SWITCH
    assert decision.incremental_benefit_cny is not None
    assert decision.incremental_benefit_cny > decision.switch_path["total_cny"]
    # Haircut evidence: raw benefit (0.5*0.15 - 0.5*0.05) * 1e6 = 100k,
    # conservative haircut 0.5 -> 50k.
    assert decision.incremental_benefit_cny == pytest.approx(50_000.0)


def test_comparison_is_deterministic() -> None:
    kwargs = _kwargs(expected_returns={"SH600000": 0.0, "SH600001": 0.02})
    first = compare_transition_paths({"SH600000": 0.5}, {"SH600001": 0.5}, **kwargs)
    second = compare_transition_paths({"SH600000": 0.5}, {"SH600001": 0.5}, **kwargs)
    assert first.to_dict() == second.to_dict()


def test_hard_constraints_bypass_the_band() -> None:
    # A microscopic change that the frozen band would otherwise hold.
    for hard in (
        ["cash_shortfall"],
        ["instrument_delisted"],
        ["permission_tightened:sell_only"],
        ["risk_exit:max_drawdown_liquidate"],
    ):
        decision = compare_transition_paths(
            {"SH600000": 0.500},
            {"SH600000": 0.499, "SH600001": 0.001},
            hard_constraints=hard,
            **_kwargs(),
        )
        assert decision.decision == DECISION_FORCED
        assert decision.benefit_mode == "hard_constraint_bypass"
        assert any("bypasses alpha no-trade band" in reason for reason in decision.reasons)


def test_frozen_band_holds_small_drift_without_cny_benefit() -> None:
    decision = compare_transition_paths(
        {"SH600000": 0.5000},
        {"SH600000": 0.4990, "SH600001": 0.0010},
        **_kwargs(),
    )
    assert decision.decision == DECISION_HOLD
    assert decision.benefit_mode == "frozen_drift_band"
    assert decision.incremental_benefit_cny is None
    assert any("frozen drift band" in reason for reason in decision.reasons)


def test_frozen_band_switches_material_moves() -> None:
    decision = compare_transition_paths(
        {"SH600000": 0.5},
        {"SH600001": 0.5},
        **_kwargs(),
    )
    assert decision.decision == DECISION_SWITCH
    assert decision.benefit_mode == "frozen_drift_band"


def test_identical_targets_never_trade() -> None:
    decision = compare_transition_paths(
        {"SH600000": 0.5},
        {"SH600000": 0.5},
        **_kwargs(),
    )
    assert decision.decision == DECISION_HOLD
    assert decision.switch_path["legs"] == []


def test_raw_scores_are_rejected_as_cny_comparable_returns() -> None:
    # Rank scores (e.g. 100, 99, ...) are not per-horizon returns; feeding
    # them as expected returns must fail closed instead of silently comparing
    # score points against CNY costs.
    with pytest.raises(ValueError, match="raw rank scores"):
        compare_transition_paths(
            {"SH600000": 0.5},
            {"SH600001": 0.5},
            expected_returns={"SH600000": 100.0, "SH600001": 99.0},
            **_kwargs(),
        )
    with pytest.raises(ValueError, match="score_to_return_slope"):
        expected_returns_from_scores(
            {"SH600000": 100.0, "SH600001": 99.0},
            score_to_return_slope=0.0,
            horizon_days=20,
        )
    with pytest.raises(ValueError, match="horizon_days"):
        expected_returns_from_scores(
            {"SH600000": 100.0, "SH600001": 99.0},
            score_to_return_slope=0.002,
            horizon_days=0,
        )


def test_explicit_score_mapping_is_frozen_capped_and_deterministic() -> None:
    scores = {"SH600000": 100.0, "SH600001": 90.0, "SH600002": 80.0}
    first = expected_returns_from_scores(
        scores, score_to_return_slope=0.002, horizon_days=20
    )
    second = expected_returns_from_scores(
        scores, score_to_return_slope=0.002, horizon_days=20
    )
    assert first == second
    # Cross-sectional mapping: top score positive, bottom negative, mean ~0.
    assert first["SH600000"] > 0 > first["SH600002"]
    assert sum(first.values()) == pytest.approx(0.0, abs=1e-12)
    capped = expected_returns_from_scores(
        scores, score_to_return_slope=1.0, horizon_days=20
    )
    assert all(abs(value) <= 0.20 for value in capped.values())
    with pytest.raises(ValueError, match="max_expected_return"):
        expected_returns_from_scores(
            scores,
            score_to_return_slope=0.002,
            horizon_days=20,
            max_expected_return=0.0,
        )


def test_minimum_legal_order_value_holds_dust_trades() -> None:
    band = NoTradeBandConfig(min_turnover=0.0000001, min_weight_change=1e-9)
    decision = compare_transition_paths(
        {"SH600000": 0.500000},
        {"SH600000": 0.499995, "SH600001": 0.000005},
        band=band,
        **_kwargs(),
    )
    # 0.000005 * 1e6 = 5 CNY leg is below the minimum legal order value.
    assert decision.decision == DECISION_HOLD
    assert any("minimum legal order value" in reason for reason in decision.reasons)
