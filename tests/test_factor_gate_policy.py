"""FactorGatePolicy 宽进严出:硬门只防蠢,效应/显著性指标只入档。"""

from __future__ import annotations

import pytest

from quant_platform.research_store import FactorGatePolicy

pytestmark = pytest.mark.no_database


def _metrics(**overrides) -> dict:
    metrics = {
        # 硬门(完整性/覆盖率/冗余)全部达标
        "selection_days": 300,
        "coverage_pass_rate": 0.99,
        "mean_coverage_ratio": 0.95,
        "constant_day_rate": 0.0,
        "max_correlation": 0.20,
        # 效应与统计显著性:全部不达标(旧门禁会否决)
        "ic": 0.005,
        "icir": 0.10,
        "rank_ic": 0.006,
        "rank_icir": 0.12,
        "turnover": 0.90,
        "cost_adjusted_return": -0.01,
        "hac_p_value": 0.90,
        "bh_q_value": 0.95,
        "raw_valid_ic": 0.01,
        "raw_selection_ic": 0.01,
    }
    metrics.update(overrides)
    return metrics


def test_insignificant_but_well_formed_factor_is_admitted_with_report_only_effect() -> (
    None
):
    policy = FactorGatePolicy()
    status, reasons = policy.evaluate(_metrics())
    assert status == "passed"
    assert reasons == []
    layers = policy.evaluate_layers(_metrics())
    assert layers["hard_status"] == "passed"
    # 效应层照算照存(入档),但不再否决。
    assert layers["effect_status"] == "failed"
    assert any("hac_p_value" in reason for reason in layers["effect_reasons"])


def test_coverage_and_runnability_failures_still_veto() -> None:
    policy = FactorGatePolicy()
    status, reasons = policy.evaluate(_metrics(coverage_pass_rate=0.50))
    assert status == "failed"
    assert any("coverage_pass_rate" in reason for reason in reasons)

    status, reasons = policy.evaluate(_metrics(selection_days=50))
    assert status == "failed"
    assert any("selection_days" in reason for reason in reasons)


def test_library_redundancy_remains_a_hard_gate() -> None:
    policy = FactorGatePolicy()
    status, reasons = policy.evaluate(_metrics(max_correlation=0.90))
    assert status == "failed"
    assert any("max_correlation" in reason for reason in reasons)
    layers = policy.evaluate_layers(_metrics(max_correlation=0.90))
    assert layers["hard_status"] == "failed"
    assert any("max_correlation" in reason for reason in layers["hard_reasons"])


def test_missing_metrics_fail_closed_as_incomplete_evidence() -> None:
    policy = FactorGatePolicy()
    metrics = _metrics()
    del metrics["ic"]
    del metrics["hac_p_value"]
    status, reasons = policy.evaluate(metrics)
    assert status == "failed"
    assert any("is missing" in reason for reason in reasons)
