from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_data.snapshot_lineage import canonical_sha256
from quant_platform.formal_validation import (
    CONSERVATIVE_BONFERRONI_INCOMPLETE_FAMILY_STATUS,
    FORMAL_VALIDATION_CONTRACT_VERSION,
    FROZEN_STRATEGY_OUTER_SCOPE,
    NOT_COMPUTABLE_INCOMPLETE_FAMILY_STATUS,
    PRE_FINAL_HISTORY_CONTRACT_VERSION,
    build_factor_score_incomplete_family_dsr,
    build_factor_score_incomplete_family_multiple_testing,
    build_outer_walk_forward_folds,
    build_pre_final_history_evidence,
    run_ablation_suite,
    run_outer_walk_forward,
    run_signal_decay_suite,
    validate_factor_score_incomplete_family_dsr,
    validate_factor_score_incomplete_family_multiple_testing,
)
from quant_platform.statistical_validation import (
    deflated_sharpe_probability,
    holm_bonferroni,
    paired_moving_block_bootstrap,
    probability_of_backtest_overfitting,
)
from quant_platform.strategy_store import (
    _formal_validation_failures,
    _incomplete_family_manifest_binding_failures,
    _valid_factor_score_incomplete_family_alternative,
)

pytestmark = pytest.mark.no_database


def test_outer_walk_forward_reruns_inner_selection_for_every_fold() -> None:
    dates = pd.bdate_range("2020-01-02", periods=160)
    calls: list[tuple[str, int]] = []

    def inner(candidate: str, fold):
        calls.append((candidate, fold.fold))
        # Winner deliberately changes by fold: selection is not reused.
        return {"information_ratio": 2.0 if candidate == f"c{fold.fold % 2}" else 1.0}

    result = run_outer_walk_forward(
        dates=dates,
        candidate_ids=["c0", "c1"],
        inner_runner=inner,
        test_runner=lambda candidate, fold: {
            "candidate": candidate,
            "test_start": fold.test_start,
            "information_ratio": 0.5,
        },
        selection_metric="information_ratio",
        train_days=60,
        validation_days=20,
        test_days=20,
        purge_days=5,
        embargo_days=5,
    )

    assert result["fold_count"] == 3
    assert len(calls) == 6
    assert [
        item["selected_candidate_id"] for item in result["folds"]
    ] == ["c0", "c1", "c0"]
    test_ranges = [
        (item["fold"]["test_start"], item["fold"]["test_end"])
        for item in result["folds"]
    ]
    assert len(test_ranges) == len(set(test_ranges))
    assert result["passed"] is True
    assert result["test_pass_rate"] == 1.0


def test_outer_walk_forward_fails_when_oos_windows_do_not_hold_up() -> None:
    dates = pd.bdate_range("2020-01-02", periods=160)
    test_values = iter([0.02, -0.40, -0.10])

    result = run_outer_walk_forward(
        dates=dates,
        candidate_ids=["frozen"],
        inner_runner=lambda _candidate, _fold: {"annualized_excess_return": 0.20},
        test_runner=lambda _candidate, _fold: {
            "annualized_excess_return": next(test_values)
        },
        selection_metric="annualized_excess_return",
        train_days=60,
        validation_days=20,
        test_days=20,
        purge_days=5,
        embargo_days=5,
        minimum_test_metric=0.0,
        minimum_test_pass_rate=0.60,
    )

    assert result["passed"] is False
    assert result["test_pass_rate"] == pytest.approx(1.0 / 3.0)
    assert result["mean_test_metric"] < 0


def test_strategy_formal_gate_rejects_completed_but_failing_oos_evidence() -> None:
    version = {
        "config": {
            "minimum_outer_test_excess_return": 0.0,
            "minimum_outer_test_pass_rate": 0.60,
            "min_pre_final_history_days": 2520,
            "outer_embargo_days": 5,
            "baseline_definition": None,
        },
        "factors": [],
    }
    outer = {
        "status": "completed",
        "passed": True,
        "fold_count": 3,
        "test_pass_rate": 2.0 / 3.0,
        "mean_test_metric": 0.01,
        "candidate_coverage": {
            "required_group_trials": 1,
            "provided_candidates": 1,
        },
        "folds": [
            {"test_metric": 0.02, "test_passed": True},
            {"test_metric": 0.02, "test_passed": True},
            {"test_metric": -0.01, "test_passed": False},
        ],
    }
    metrics = {
        "deflated_sharpe": {"trials": 1},
        "formal_validation_passed": True,
        "formal_validation": {
            "contract_version": FORMAL_VALIDATION_CONTRACT_VERSION,
            "status": "passed",
            "pre_final_history": {
                "status": "completed",
                "contract_version": PRE_FINAL_HISTORY_CONTRACT_VERSION,
                "requested_periods": {
                    "start": "2008-01-01",
                    "end": "2020-12-31",
                },
                "observed_periods": {
                    "start": "2008-01-02",
                    "end": "2020-12-31",
                },
                "final_test_periods": {
                    "start": "2021-01-11",
                    "end": "2026-07-28",
                },
                "trading_days": 3150,
                "minimum_trading_days": 2520,
                "embargo_trading_days": 5,
                "minimum_embargo_trading_days": 5,
                "overlaps_final_test": False,
                "uses_final_test_data": False,
                "execution_model": {
                    "method": "open",
                    "frequency": "day",
                    "minute_execution_claimed": False,
                },
            },
            "outer_walk_forward": outer,
            "ablation": {"status": "passed", "runs": []},
            "signal_decay": {
                "status": "completed",
                "frontier_version": "contiguous-zero-delay-frontier-v2",
                "maximum_supported_delay_bars": 0,
                "runs": [{"delay_bars": 0, "passed": True}],
            },
            "paired_block_bootstrap": {
                "status": "ok",
                "confidence_interval_95": [0.0001, 0.01],
            },
            "multiple_testing": {
                "status": "not_applicable_single_trial",
                "holm_adjusted_p_values": [0.01],
            },
        },
    }

    assert _formal_validation_failures(version, metrics) == []

    outer.update(
        {
            "passed": False,
            "test_pass_rate": 1.0 / 3.0,
            "mean_test_metric": -0.10,
        }
    )
    outer["folds"] = [
        {"test_metric": 0.02, "test_passed": True},
        {"test_metric": -0.20, "test_passed": False},
        {"test_metric": -0.12, "test_passed": False},
    ]

    assert any(
        "outer walk-forward" in failure
        for failure in _formal_validation_failures(version, metrics)
    )


def test_pre_final_history_is_long_and_strictly_before_final_test() -> None:
    dates = pd.bdate_range("2008-01-01", "2021-01-11")
    evidence = build_pre_final_history_evidence(
        dates,
        requested_start="2008-01-01",
        requested_end="2020-12-31",
        final_test_start="2021-01-11",
        final_test_end="2026-07-28",
        minimum_trading_days=2520,
        minimum_embargo_trading_days=5,
    )

    assert evidence["trading_days"] >= 2520
    assert evidence["uses_final_test_data"] is False
    assert evidence["final_test_periods"]["start"] == "2021-01-11"
    assert evidence["embargo_trading_days"] >= 5

    with pytest.raises(ValueError, match="must end before"):
        build_pre_final_history_evidence(
            dates,
            requested_start="2008-01-01",
            requested_end="2021-01-01",
            final_test_start="2021-01-01",
            final_test_end="2026-07-28",
            minimum_trading_days=2520,
            minimum_embargo_trading_days=5,
        )

    with pytest.raises(ValueError, match="trading days"):
        build_pre_final_history_evidence(
            pd.bdate_range("2019-01-01", "2020-12-31"),
            requested_start="2019-01-01",
            requested_end="2020-12-31",
            final_test_start="2021-01-01",
            final_test_end="2026-07-28",
            minimum_trading_days=2520,
            minimum_embargo_trading_days=5,
        )

    with pytest.raises(ValueError, match="embargo trading days"):
        build_pre_final_history_evidence(
            pd.bdate_range("2008-01-01", "2021-01-04"),
            requested_start="2008-01-01",
            requested_end="2020-12-31",
            final_test_start="2021-01-04",
            final_test_end="2026-07-28",
            minimum_trading_days=2520,
            minimum_embargo_trading_days=5,
        )


def test_outer_fold_builder_keeps_purge_and_embargo_gaps() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    folds = build_outer_walk_forward_folds(
        dates,
        train_days=40,
        validation_days=10,
        test_days=10,
        purge_days=3,
        embargo_days=4,
    )
    first = folds[0]
    assert dates.get_loc(first.validation_start) - dates.get_loc(first.train_end) == 4
    assert dates.get_loc(first.test_start) - dates.get_loc(first.validation_end) == 5


def test_ablation_suite_records_incremental_evidence() -> None:
    result = run_ablation_suite(
        component_ids=["momentum", "quality"],
        full_metrics={"information_ratio": 0.80},
        runner=lambda component: {
            "information_ratio": 0.50 if component == "momentum" else 0.75
        },
        metric="information_ratio",
        minimum_increment=0.10,
    )
    assert result["status"] == "failed"
    assert result["runs"][0]["increment"] == pytest.approx(0.30)
    assert result["runs"][1]["passed"] is False


def test_signal_decay_derives_last_supported_delay() -> None:
    values = {0: 1.0, 1: 0.85, 2: 0.55, 3: -0.10}
    result = run_signal_decay_suite(
        delays=[3, 0, 2, 1],
        runner=lambda delay: {"annualized_excess_return": values[delay]},
        metric="annualized_excess_return",
        minimum_retention=0.60,
    )
    assert result["maximum_supported_delay_bars"] == 1
    assert result["frontier_version"] == "contiguous-zero-delay-frontier-v2"
    assert [item["delay_bars"] for item in result["runs"]] == [0, 1, 2, 3]


def test_signal_decay_requires_a_contiguous_supported_frontier() -> None:
    values = {0: 1.0, 1: 0.40, 2: 0.80}

    result = run_signal_decay_suite(
        delays=[0, 1, 2],
        runner=lambda delay: {"annualized_excess_return": values[delay]},
        metric="annualized_excess_return",
        minimum_retention=0.60,
    )

    assert [item["passed"] for item in result["runs"]] == [True, False, True]
    assert result["maximum_supported_delay_bars"] == 0


def test_holm_adjustment_preserves_original_order() -> None:
    adjusted = holm_bonferroni([0.04, 0.01, 0.03])
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_paired_block_bootstrap_is_deterministic_and_paired() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.normal(0.0, 0.01, 120)
    candidate = baseline + 0.001
    first = paired_moving_block_bootstrap(
        candidate,
        baseline,
        block_size=10,
        samples=500,
        seed=17,
    )
    second = paired_moving_block_bootstrap(
        candidate,
        baseline,
        block_size=10,
        samples=500,
        seed=17,
    )
    assert first == second
    assert first["observed_mean_difference"] == pytest.approx(0.001)
    assert first["probability_positive"] == 1.0


def test_pbo_distinguishes_stable_candidate_from_fold_winners() -> None:
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        {
            "stable": rng.normal(0.0010, 0.01, 160),
            "noise_a": rng.normal(0.0, 0.01, 160),
            "noise_b": rng.normal(0.0, 0.01, 160),
        }
    )
    result = probability_of_backtest_overfitting(returns, blocks=8)
    assert result["status"] == "ok"
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["split_count"] == 35


def _incomplete_factor_family_fixture() -> tuple[dict, dict]:
    audit_sha256 = "a" * 64
    eligibility_sha256 = "b" * 64
    trial_count = 4
    bootstrap = {
        "status": "ok",
        "confidence_interval_95": [0.0001, 0.01],
        "one_sided_p_value": 0.01,
    }
    multiple = build_factor_score_incomplete_family_multiple_testing(
        paired_bootstrap=bootstrap,
        trial_count=trial_count,
        trial_count_audit_sha256=audit_sha256,
        eligibility_receipt_sha256=eligibility_sha256,
    )
    blocked_dsr = deflated_sharpe_probability(
        pd.Series(np.linspace(-0.01, 0.02, 120)),
        trials=trial_count,
    )
    deflated = build_factor_score_incomplete_family_dsr(
        blocked_dsr=blocked_dsr,
        trial_count=trial_count,
        trial_count_audit_sha256=audit_sha256,
        eligibility_receipt_sha256=eligibility_sha256,
    )
    version = {
        "config": {
            "signal_source": "factor_score",
            "minimum_outer_test_excess_return": 0.0,
            "minimum_outer_test_pass_rate": 0.60,
            "min_pre_final_history_days": 2520,
            "outer_embargo_days": 5,
            "baseline_definition": None,
        },
        "factors": [],
    }
    metrics = {
        "deflated_sharpe": deflated,
        "deflated_sharpe_probability": None,
        "formal_validation_passed": True,
        "formal_validation": {
            "contract_version": FORMAL_VALIDATION_CONTRACT_VERSION,
            "status": "passed",
            "pre_final_history": {
                "status": "completed",
                "contract_version": PRE_FINAL_HISTORY_CONTRACT_VERSION,
                "requested_periods": {"start": "2008-01-01", "end": "2020-12-31"},
                "observed_periods": {"start": "2008-01-02", "end": "2020-12-31"},
                "final_test_periods": {"start": "2021-01-11", "end": "2026-07-28"},
                "trading_days": 3150,
                "minimum_trading_days": 2520,
                "embargo_trading_days": 5,
                "minimum_embargo_trading_days": 5,
                "overlaps_final_test": False,
                "uses_final_test_data": False,
                "execution_model": {
                    "method": "open",
                    "frequency": "day",
                    "minute_execution_claimed": False,
                },
            },
            "outer_walk_forward": {
                "status": "completed",
                "passed": True,
                "fold_count": 3,
                "test_pass_rate": 1.0,
                "mean_test_metric": 0.02,
                "candidate_ids": ["frozen-strategy"],
                "candidate_coverage": {
                    "required_group_trials": trial_count,
                    "provided_candidates": 1,
                    "scope": FROZEN_STRATEGY_OUTER_SCOPE,
                    "selection_performed": False,
                    "historical_candidate_matrix": "incomplete",
                    "trial_count_audit_sha256": audit_sha256,
                    "eligibility_receipt_sha256": eligibility_sha256,
                },
                "folds": [
                    {"test_metric": 0.01, "test_passed": True},
                    {"test_metric": 0.02, "test_passed": True},
                    {"test_metric": 0.03, "test_passed": True},
                ],
            },
            "ablation": {"status": "passed", "runs": []},
            "signal_decay": {
                "status": "completed",
                "frontier_version": "contiguous-zero-delay-frontier-v2",
                "maximum_supported_delay_bars": 0,
                "runs": [{"delay_bars": 0, "passed": True}],
            },
            "paired_block_bootstrap": bootstrap,
            "multiple_testing": multiple,
        },
    }
    return version, metrics


def test_incomplete_factor_family_uses_exact_bonferroni_and_not_computable_diagnostics() -> None:
    version, metrics = _incomplete_factor_family_fixture()
    multiple = metrics["formal_validation"]["multiple_testing"]
    deflated = metrics["deflated_sharpe"]

    assert multiple["status"] == CONSERVATIVE_BONFERRONI_INCOMPLETE_FAMILY_STATUS
    assert multiple["bonferroni_adjusted_p_value"] == pytest.approx(0.04)
    assert multiple["gate_passed"] is True
    assert multiple["pbo"]["status"] == NOT_COMPUTABLE_INCOMPLETE_FAMILY_STATUS
    assert multiple["pbo"]["pbo"] is None
    assert deflated["status"] == NOT_COMPUTABLE_INCOMPLETE_FAMILY_STATUS
    assert deflated["probability"] is None
    assert _valid_factor_score_incomplete_family_alternative(version, metrics) is True
    assert _formal_validation_failures(version, metrics) == []


def test_incomplete_factor_family_rejects_tampering_and_failed_familywise_alpha() -> None:
    version, metrics = _incomplete_factor_family_fixture()
    bootstrap = metrics["formal_validation"]["paired_block_bootstrap"]
    multiple = metrics["formal_validation"]["multiple_testing"]
    audit_sha256 = multiple["trial_count_audit_sha256"]

    assert validate_factor_score_incomplete_family_multiple_testing(
        multiple,
        paired_bootstrap=bootstrap,
        trial_count=4,
        trial_count_audit_sha256=audit_sha256,
        eligibility_receipt_sha256=multiple["eligibility_receipt_sha256"],
    ) == multiple
    assert validate_factor_score_incomplete_family_dsr(
        metrics["deflated_sharpe"],
        trial_count=4,
        trial_count_audit_sha256=audit_sha256,
        eligibility_receipt_sha256=multiple["eligibility_receipt_sha256"],
    ) == metrics["deflated_sharpe"]

    tampered = deepcopy(metrics)
    tampered["formal_validation"]["multiple_testing"]["pbo"]["status"] = "ok"
    assert _valid_factor_score_incomplete_family_alternative(version, tampered) is False
    assert _formal_validation_failures(version, tampered)

    failed_bootstrap = {**bootstrap, "one_sided_p_value": 0.02}
    failed_multiple = build_factor_score_incomplete_family_multiple_testing(
        paired_bootstrap=failed_bootstrap,
        trial_count=4,
        trial_count_audit_sha256=audit_sha256,
        eligibility_receipt_sha256=multiple["eligibility_receipt_sha256"],
    )
    assert failed_multiple["bonferroni_adjusted_p_value"] == pytest.approx(0.08)
    assert failed_multiple["gate_passed"] is False


def test_incomplete_factor_family_requires_one_frozen_outer_candidate() -> None:
    version, metrics = _incomplete_factor_family_fixture()
    metrics["formal_validation"]["outer_walk_forward"]["candidate_coverage"][
        "provided_candidates"
    ] = 4

    assert any(
        "outer walk-forward" in failure
        for failure in _formal_validation_failures(version, metrics)
    )


def test_incomplete_factor_family_binds_real_trial_count_to_manifest_audit() -> None:
    _version, metrics = _incomplete_factor_family_fixture()
    trial_count_audit = {
        "strategy_version_components": [
            {"component_id": "trial-a"},
            {"component_id": "trial-b"},
            {"component_id": "trial-c"},
            {"component_id": "trial-d"},
        ]
    }
    audit_sha256 = canonical_sha256(trial_count_audit)
    eligibility_sha256 = "b" * 64
    bootstrap = metrics["formal_validation"]["paired_block_bootstrap"]
    metrics["formal_validation"][
        "multiple_testing"
    ] = build_factor_score_incomplete_family_multiple_testing(
        paired_bootstrap=bootstrap,
        trial_count=4,
        trial_count_audit_sha256=audit_sha256,
        eligibility_receipt_sha256=eligibility_sha256,
    )
    metrics["formal_validation"]["outer_walk_forward"]["candidate_coverage"][
        "trial_count_audit_sha256"
    ] = audit_sha256
    metrics["formal_validation"]["outer_walk_forward"]["candidate_coverage"][
        "eligibility_receipt_sha256"
    ] = eligibility_sha256
    metrics["deflated_sharpe"] = build_factor_score_incomplete_family_dsr(
        blocked_dsr=deflated_sharpe_probability(
            pd.Series(np.linspace(-0.01, 0.02, 120)),
            trials=4,
        ),
        trial_count=4,
        trial_count_audit_sha256=audit_sha256,
        eligibility_receipt_sha256=eligibility_sha256,
    )
    strategy_version_id = "target-v18"
    strategy_version_ids = [
        "4414d202dbb641608975e5305bc18da4",
        strategy_version_id,
    ]
    missing_artifacts = [
        {
            "artifact_kind": kind,
            "path": f"artifacts/source/{filename}",
            "status": "missing",
            "sha256": None,
            "bytes": None,
            "observed_at": "2026-08-31T00:00:00+00:00",
        }
        for kind, filename in (
            ("trial_candidate_manifest_matrix", "trial_candidate_manifest_matrix.json"),
            ("trial_daily_returns_matrix", "trial_daily_returns_matrix.parquet"),
            ("trial_score_grid_matrix", "trial_score_grid_matrix.parquet"),
        )
    ]
    eligibility_core = {
        "contract_version": "incomplete-factor-family-eligibility-v1",
        "evidence_mode": "consumed_historical_replay",
        "authority": "conservative_bonferroni_only",
        "source_strategy_version_id": "4414d202dbb641608975e5305bc18da4",
        "source_backtest_id": "0a113fe28ca741b6be9c09ab046c9d02",
        "source_job_id": "858a75a6f1994c359fa9c3567ed09f57",
        "strategy_version_id": strategy_version_id,
        "economic_hypothesis_group": "public-short",
        "eligible_strategy_version_ids": strategy_version_ids,
        "strategy_trial_count": 4,
        "trial_count_audit": trial_count_audit,
        "trial_count_audit_sha256": audit_sha256,
        "missing_artifacts": missing_artifacts,
        "cutoff_at": "2026-08-31T00:00:00+00:00",
    }
    eligibility = {**eligibility_core, "receipt_sha256": canonical_sha256(eligibility_core)}
    assert eligibility["receipt_sha256"] != eligibility_sha256
    eligibility_sha256 = eligibility["receipt_sha256"]
    metrics["formal_validation"][
        "multiple_testing"
    ] = build_factor_score_incomplete_family_multiple_testing(
        paired_bootstrap=bootstrap,
        trial_count=4,
        trial_count_audit_sha256=audit_sha256,
        eligibility_receipt_sha256=eligibility_sha256,
    )
    metrics["formal_validation"]["outer_walk_forward"]["candidate_coverage"][
        "eligibility_receipt_sha256"
    ] = eligibility_sha256
    metrics["deflated_sharpe"] = build_factor_score_incomplete_family_dsr(
        blocked_dsr=deflated_sharpe_probability(
            pd.Series(np.linspace(-0.01, 0.02, 120)), trials=4
        ),
        trial_count=4,
        trial_count_audit_sha256=audit_sha256,
        eligibility_receipt_sha256=eligibility_sha256,
    )
    manifest = {
        "strategy_version_id": strategy_version_id,
        "strategy_trial_count": 4,
        "hypothesis_group_evidence": {
            "economic_hypothesis_group": "public-short",
            "shared_experiment_count": 4,
            "strategy_version_ids": strategy_version_ids,
            "trial_count_audit": trial_count_audit,
        },
        "incomplete_factor_family_eligibility": eligibility,
    }

    assert _incomplete_family_manifest_binding_failures(manifest, metrics) == []

    tampered_manifest = deepcopy(manifest)
    tampered_manifest["strategy_trial_count"] = 3
    assert _incomplete_family_manifest_binding_failures(tampered_manifest, metrics)

    tampered_manifest = deepcopy(manifest)
    tampered_manifest["hypothesis_group_evidence"]["trial_count_audit"][
        "strategy_version_components"
    ].append({"component_id": "post-hoc"})
    assert _incomplete_family_manifest_binding_failures(tampered_manifest, metrics)


def test_complete_multi_trial_factor_family_still_requires_real_pbo() -> None:
    version, metrics = _incomplete_factor_family_fixture()
    metrics["formal_validation"]["multiple_testing"] = {
        "status": "ok",
        "trial_count": 4,
        "holm_adjusted_p_values": [0.01, 0.02, 0.03, 0.04],
        "pbo": {"status": NOT_COMPUTABLE_INCOMPLETE_FAMILY_STATUS, "pbo": None},
    }

    assert any(
        "multiple-testing evidence" in failure
        for failure in _formal_validation_failures(version, metrics)
    )
