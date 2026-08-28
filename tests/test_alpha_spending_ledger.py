from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quant_data.snapshot_lineage import canonical_sha256
from quant_platform.alpha_spending_integration import (
    capital_oos_family_identity_policy,
    capital_oos_family_manifest,
    capital_oos_family_manifest_sha256,
    capital_oos_family_sha256,
    validate_capital_oos_family_manifest,
)
from quant_platform.alpha_spending_ledger import (
    CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS,
    CAPITAL_OOS_MIN_TRADING_DAYS,
    CAPITAL_OOS_POLICY_SHA256,
    CAPITAL_OOS_TOTAL_ALPHA,
    _bootstrap_gate,
    batch_alpha,
    capital_oos_policy_manifest,
    capital_oos_vintage_link_payload,
    cumulative_batch_alpha,
    paired_hac_capital_test,
    validate_final_oos_window,
)

pytestmark = pytest.mark.no_database


def _dates(start: str, periods: int) -> list[str]:
    return [item.date().isoformat() for item in pd.bdate_range(start, periods=periods)]


def test_policy_is_capital_only_and_research_never_spends_it() -> None:
    policy = capital_oos_policy_manifest()
    assert CAPITAL_OOS_TOTAL_ALPHA == Decimal("0.05")
    assert policy["scope"] == "capital_oos_only"
    assert policy["research_tournaments_spend_alpha"] is False
    assert policy["hypotheses_per_batch"] == 1
    assert policy["minimum_final_oos_trading_days"] == 252
    assert policy["minimum_embargo_trading_days"] == 20
    assert policy["maximum_unsettled_batches_per_family"] == 1
    assert policy["dataset_lineage_resets_family"] is False
    assert policy["error_control_scope"].startswith("stable_investment_mandate")
    assert len(CAPITAL_OOS_POLICY_SHA256) == 64


def test_vintage_link_payload_is_exact_and_hash_only() -> None:
    payload = capital_oos_vintage_link_payload(
        batch_id="1" * 64,
        preregistration_sha256="2" * 64,
        frozen_bundle_manifest_sha256="3" * 64,
        frozen_baseline_manifest_sha256="4" * 64,
    )
    assert payload == {
        "contract_version": "capital-oos-vintage-link-v1",
        "batch_id": "1" * 64,
        "preregistration_sha256": "2" * 64,
        "frozen_bundle_manifest_sha256": "3" * 64,
        "frozen_baseline_manifest_sha256": "4" * 64,
    }
    with pytest.raises(ValueError, match="must be SHA256"):
        capital_oos_vintage_link_payload(
            batch_id="not-a-hash",
            preregistration_sha256="2" * 64,
            frozen_bundle_manifest_sha256="3" * 64,
            frozen_baseline_manifest_sha256="4" * 64,
        )


def test_capital_batch_spending_telescopes_without_exhausting_budget() -> None:
    values = [batch_alpha(ordinal) for ordinal in range(1, 1001)]
    assert values[0] == Decimal("0.025")
    assert abs(sum(values) - cumulative_batch_alpha(1000)) < Decimal("1e-26")
    assert sum(values) < CAPITAL_OOS_TOTAL_ALPHA
    assert cumulative_batch_alpha(1_000_000) < CAPITAL_OOS_TOTAL_ALPHA


def test_final_oos_window_requires_252_days_and_20_day_prior_embargo() -> None:
    embargo = _dates("2019-12-02", CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS)
    final = _dates("2020-01-01", CAPITAL_OOS_MIN_TRADING_DAYS)
    result = validate_final_oos_window(
        research_data_end="2019-11-29",
        final_oos_trading_dates=final,
        embargo_trading_dates=embargo,
    )
    assert result["trading_day_count"] == 252
    assert result["embargo_trading_day_count"] == 20
    assert result["final_oos_start"] == final[0]
    with pytest.raises(ValueError, match="at least 252"):
        validate_final_oos_window(
            research_data_end="2019-11-29",
            final_oos_trading_dates=final[:-1],
            embargo_trading_dates=embargo,
        )
    with pytest.raises(ValueError, match="strictly before"):
        validate_final_oos_window(
            research_data_end="2019-11-29",
            final_oos_trading_dates=final,
            embargo_trading_dates=_dates("2020-01-01", 20),
        )
    with pytest.raises(ValueError, match="start after all research data"):
        validate_final_oos_window(
            research_data_end=embargo[0],
            final_oos_trading_dates=final,
            embargo_trading_dates=embargo,
        )


def test_paired_hac_uses_one_sided_nonzero_p_and_bootstrap_hard_gate() -> None:
    index = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(17)
    baseline = pd.Series(rng.normal(0.0, 0.002, len(index)), index=index)
    candidate = baseline + pd.Series(rng.normal(0.002, 0.001, len(index)), index=index)
    difference, hac = paired_hac_capital_test(candidate, baseline)
    bootstrap = _bootstrap_gate(difference)
    assert 0.0 < hac["one_sided_p_value"] < 0.001
    assert hac["return_definition"].endswith("after_cost")
    assert bootstrap["hard_gate_passed"] is True
    assert bootstrap["samples"] == 2000


def test_paired_hac_negative_or_degenerate_effect_cannot_pass() -> None:
    index = pd.bdate_range("2020-01-01", periods=300)
    baseline = pd.Series(0.0, index=index)
    negative = pd.Series(np.random.default_rng(23).normal(-0.001, 0.002, len(index)), index=index)
    constant_positive = pd.Series(0.001, index=index)
    _, negative_test = paired_hac_capital_test(negative, baseline)
    _, degenerate_test = paired_hac_capital_test(constant_positive, baseline)
    assert negative_test["one_sided_p_value"] == 1.0
    assert degenerate_test["one_sided_p_value"] == 1.0
    assert degenerate_test["status"] == "undefined_zero_hac_variance"


def test_stable_family_excludes_champion_bundle_date_and_daily_identity() -> None:
    args = ("csi300", "SH000300", 5, "a" * 64, "b" * 64, "c" * 64)
    first = capital_oos_family_sha256(*args)
    later_champion_and_snapshot = capital_oos_family_sha256(*args)
    assert first == later_champion_and_snapshot
    manifest = capital_oos_family_manifest(*args)
    identity_policy = capital_oos_family_identity_policy()
    excluded = set(identity_policy["excluded_from_family_identity"])
    assert {
        "dataset_identity_sha256",
        "dataset_lineage_id",
        "calendar_date",
        "incumbent_manifest_sha256",
        "champion_manifest_sha256",
        "sota_version_sha256",
        "frozen_bundle_manifest_sha256",
        "frozen_baseline_manifest_sha256",
        "final_oos_window",
    } <= excluded
    assert not any("incumbent" in key or "bundle" in key for key in manifest)


def test_stable_family_changes_only_for_a_changed_mandate() -> None:
    base = capital_oos_family_sha256("csi300", "SH000300", 5, "a" * 64, "b" * 64, "c" * 64)
    normalized_case = capital_oos_family_sha256(
        "CSI300", "sh000300", 5, "A" * 64, "B" * 64, "C" * 64
    )
    changed_cost = capital_oos_family_sha256("csi300", "SH000300", 5, "d" * 64, "b" * 64, "c" * 64)
    assert normalized_case == base
    assert base != changed_cost
    with pytest.raises(ValueError, match="require SHA256"):
        capital_oos_family_sha256("csi300", "SH000300", 5, "bad", "b" * 64, "c" * 64)


def test_stable_family_validator_rejects_candidate_or_date_smuggling() -> None:
    manifest = capital_oos_family_manifest(
        "csi300",
        "SH000300",
        5,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    assert capital_oos_family_manifest_sha256(manifest) == canonical_sha256(manifest)
    for unstable_key in (
        "incumbent_manifest_sha256",
        "champion_manifest_sha256",
        "frozen_bundle_manifest_sha256",
        "frozen_baseline_manifest_sha256",
        "calendar_date",
        "dataset_identity_sha256",
        "dataset_lineage_id",
    ):
        contaminated = {**manifest, unstable_key: "d" * 64}
        with pytest.raises(ValueError, match="unstable or missing"):
            validate_capital_oos_family_manifest(contaminated)
    with pytest.raises(ValueError, match="invalid field types"):
        validate_capital_oos_family_manifest({**manifest, "label_horizon_days": True})
