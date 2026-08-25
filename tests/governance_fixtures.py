from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date
from pathlib import Path

from qlib_test_doubles import qlib_workflow_identity

from quant_data.execution_contract import DAILY_QLIB_FIELD_CONTRACT_VERSION
from quant_platform.api import StrategyConfigRequest
from quant_platform.cost_model import CostModelConfig
from quant_platform.formal_validation import (
    FORMAL_VALIDATION_CONTRACT_VERSION,
    PRE_FINAL_HISTORY_CONTRACT_VERSION,
)
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.qlib_backtest import (
    COMPONENT_COST_STRESS_MULTIPLIERS,
    QLIB_ENGINE_VERSION,
)
from quant_platform.research_store import ResearchStore
from quant_platform.strategy_artifact_manifest import write_backtest_artifact_manifest
from quant_platform.strategy_store import StrategyStore

DATASET_IDENTITY = "a" * 64
PERIODS = {
    "train_start": date(2008, 1, 1),
    "train_end": date(2017, 12, 31),
    "valid_start": date(2018, 1, 1),
    "valid_end": date(2020, 12, 31),
    "test_start": date(2021, 1, 11),
    "test_end": date(2026, 7, 10),
}


def passing_factor_metrics() -> dict:
    return {
        "ic": 0.035,
        "icir": 0.80,
        "rank_ic": 0.041,
        "rank_icir": 0.76,
        "turnover": 0.32,
        "max_correlation": 0.44,
        "cost_adjusted_return": 0.052,
        "raw_valid_ic": -0.031,
        "raw_selection_ic": -0.035,
        "selection_days": 400,
        "selection_start": "2018-05-28",
        "coverage_pass_rate": 0.99,
        "mean_coverage_ratio": 0.95,
        "constant_day_rate": 0.0,
        "direction": "inverted",
        "hac_p_value": 0.01,
        "bh_q_value": 0.02,
        "statistical_contract_version": "research-statistics-v1-hac-bh-dsr",
    }


def create_promoted_factor(
    database_url: str,
    tmp_path: Path,
    *,
    dataset: str = "snapshot",
    dataset_identity: str = DATASET_IDENTITY,
    periods: dict | None = None,
) -> dict:
    periods = periods or PERIODS
    suffix = uuid.uuid4().hex
    store = ResearchStore(database_url)
    run = store.create_run(
        kind=f"factor-{suffix}",
        objective="Create governed factor fixture.",
        dataset=dataset,
        requested_by="test",
        budget={"loop_n": 1},
        config={},
        artifact_path=tmp_path,
    )
    code_path = tmp_path / f"factor-{suffix}.py"
    values_path = tmp_path / f"factor-{suffix}.h5"
    code_path.write_text("def factor(frame):\n    return frame['close']\n", encoding="utf-8")
    values_path.write_bytes(b"immutable-factor-values")
    candidate = store.add_candidate(
        run["id"],
        name=f"factor-{suffix}",
        description="governed fixture",
        formulation="close",
        variables={},
        source_iteration=0,
        code_path=str(code_path),
        values_path=str(values_path),
        code_sha256=hashlib.sha256(code_path.read_bytes()).hexdigest(),
        rdagent_decision=True,
        rdagent_feedback="ok",
    )
    metrics = passing_factor_metrics()
    artifact = tmp_path / f"evaluation-{suffix}.json"
    artifact.write_text(
        json.dumps(
            {
                "status": "ok",
                "qlib_workflow": qlib_workflow_identity(),
                "evaluations": [
                    {"candidate_id": candidate["id"], "status": "ok", "metrics": metrics}
                ],
            }
        ),
        encoding="utf-8",
    )
    recomputed_path = tmp_path / f"recomputed-{suffix}.h5"
    recomputed_path.write_bytes(b"independently-recomputed-factor-values")
    recomputed_sha256 = hashlib.sha256(recomputed_path.read_bytes()).hexdigest()
    store.record_evaluation(
        candidate["id"],
        dataset=dataset,
        dataset_identity_sha256=dataset_identity,
        **periods,
        metrics=metrics,
        artifact_path=str(artifact),
        recomputed_values_path=str(recomputed_path),
        recomputed_values_sha256=recomputed_sha256,
        recompute_evidence={
            "executor_version": "factor-recompute-v4-pit-prefix-invariance",
            "sandbox_mode": "docker-isolated",
            "sandbox_image_id": "sha256:" + "a" * 64,
            "network_mode": "none",
            "root_filesystem_read_only": True,
            "capabilities_dropped": "ALL",
            "no_new_privileges": True,
            "label_horizon_days": 1,
            "code_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
            "dataset_identity_sha256": dataset_identity,
            "provider_input_sha256": "1" * 64,
            "periods": {key: value.isoformat() for key, value in periods.items()},
            "pit_invariance": {
                "contract_version": "factor-pit-prefix-invariance-v1",
                "status": "passed",
                "cutpoint_count": 3,
                "checks": [{"invariant": True}] * 3,
            },
            "research_data_boundary": {
                "latest_input_date": periods["valid_end"].isoformat(),
                "valid_end": periods["valid_end"].isoformat(),
                "test_start": periods["test_start"].isoformat(),
                "final_oos_observations_exposed": False,
            },
            "submitted_comparison": {
                "available": True,
                "exact_match": True,
                "index_exact_match": True,
                "submitted_sha256": hashlib.sha256(values_path.read_bytes()).hexdigest(),
            },
            "authoritative_values_sha256": recomputed_sha256,
        },
    )
    return store.promote(
        candidate["id"], actor="factor-owner", reason="Approved governed test evidence."
    )


def create_strategy_version(
    database_url: str,
    tmp_path: Path,
    *,
    dataset: str = "snapshot",
    dataset_identity: str = DATASET_IDENTITY,
    config_overrides: dict | None = None,
    periods: dict | None = None,
) -> str:
    factor = create_promoted_factor(
        database_url,
        tmp_path,
        dataset=dataset,
        dataset_identity=dataset_identity,
        periods=periods,
    )
    config = {
        "topk": 50,
        "n_drop": 5,
        "max_tracking_error": 0.12,
        "max_drawdown": 0.25,
        "max_turnover": 0.60,
        "min_information_ratio": 0.0,
        "min_sharpe_ratio": 0.0,
        "min_sortino_ratio": 0.0,
        "min_rolling_pass_rate": 0.60,
        "min_rolling_windows": 3,
        "event_count": 5,
        "min_backtest_days": 504,
        "capacity_notional": 5_000_000,
        "annual_cash_yield_rate": 0.0,
        "cash_yield_source": "none_zero_yield",
        "max_volume_participation": 0.01,
        "min_commission": 5.0,
    }
    config.update(CostModelConfig().to_dict())
    config.update(config_overrides or {})
    if config.get("execution_method") in {"twap", "vwap", "next_bar"}:
        config.setdefault("execution_frequency", "5min")
    config = StrategyConfigRequest.model_validate(config).model_dump()
    strategy = StrategyStore(database_url).create(
        name=f"strategy-{uuid.uuid4().hex}",
        description="Governed strategy fixture for v2 tests.",
        benchmark="SH000300",
        universe="cn_all",
        factors=[{"candidate_id": factor["id"], "weight": 1.0}],
        config=config,
        actor="test",
    )
    return str(strategy["versions"][0]["id"])


def formal_backtest_metrics(
    version: dict,
    manifest: Path,
    *,
    hypothesis_group_evidence: dict,
) -> dict:
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    shared_experiment_count = int(
        hypothesis_group_evidence.get("shared_experiment_count") or 0
    )
    if shared_experiment_count <= 0:
        raise ValueError("hypothesis-group evidence must bind a positive trial count")
    if str(hypothesis_group_evidence.get("economic_hypothesis_group") or "") != str(
        version["economic_hypothesis_group"]
    ):
        raise ValueError("hypothesis-group evidence does not bind the strategy family")
    if str(version["id"]) not in {
        str(item) for item in hypothesis_group_evidence.get("strategy_version_ids", [])
    }:
        raise ValueError("hypothesis-group evidence does not include the strategy version")
    manifest_payload["hypothesis_group_evidence"] = hypothesis_group_evidence
    manifest_payload["strategy_trial_count"] = shared_experiment_count
    history_periods = manifest_payload.setdefault(
        "historical_validation_periods",
        {
            "start": PERIODS["train_start"].isoformat(),
            "end": PERIODS["valid_end"].isoformat(),
        },
    )
    final_periods = manifest_payload["periods"]
    version_factors = {
        str(item["factor_candidate_id"]): item for item in version.get("factors", [])
    }
    formal_factor_hashes: dict[str, str] = {}
    formal_factor_evidence: dict[str, dict] = {}
    for manifest_factor in manifest_payload.get("factors", []):
        candidate_id = str(manifest_factor["candidate_id"])
        version_factor = version_factors[candidate_id]
        execution_mode = (
            "frozen_code_recompute"
            if version_factor.get("source_iteration") is not None
            else "frozen_values"
        )
        relative = Path("formal-factor-values") / candidate_id / "authoritative.h5"
        artifact = manifest.parent / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"formal:{candidate_id}".encode())
        artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        evidence = {
            "executor_version": (
                "factor-recompute-v4-pit-prefix-invariance"
                if execution_mode == "frozen_code_recompute"
                else "frozen-values-index-exact-v1"
            ),
            "code_sha256": version_factor["code_sha256"],
            "dataset_identity_sha256": DATASET_IDENTITY,
            "provider_input_sha256": "1" * 64,
            "periods": {
                "warmup_start": history_periods["start"],
                "test_start": final_periods["start"],
                "test_end": final_periods["end"],
            },
            "oos_coverage": {
                "contract_version": "factor-oos-index-exact-v1",
                "test_start": final_periods["start"],
                "test_end": final_periods["end"],
                "trading_day_count": 252,
                "row_count": 1000,
                "finite_row_count": 900,
                "min_daily_finite_required": 50,
                "min_coverage_ratio_required": 0.8,
                "min_good_day_rate_required": 0.95,
                "minimum_daily_finite_observed": 50,
                "minimum_coverage_ratio_observed": 0.8,
                "mean_coverage_ratio_observed": 0.9,
                "good_day_rate": 0.95,
                "coverage_gate_passed": True,
                "index_exact_match": True,
            },
            "authoritative_values_sha256": artifact_sha256,
        }
        if execution_mode == "frozen_code_recompute":
            evidence.update(
                {
                    "sandbox_mode": "docker-isolated",
                    "sandbox_image_id": "sha256:" + "a" * 64,
                    "network_mode": "none",
                    "root_filesystem_read_only": True,
                    "capabilities_dropped": "ALL",
                    "no_new_privileges": True,
                    "pit_invariance": {
                        "contract_version": "factor-pit-prefix-invariance-v1",
                        "status": "passed",
                        "cutpoint_count": 3,
                        "checks": [{"invariant": True}] * 3,
                    },
                }
            )
        manifest_factor["factor_execution_mode"] = execution_mode
        manifest_factor["formal_factor_artifact"] = {
            "path": relative.as_posix(),
            "sha256": artifact_sha256,
            "execution_mode": execution_mode,
            "evidence": evidence,
        }
        formal_factor_hashes[candidate_id] = artifact_sha256
        formal_factor_evidence[candidate_id] = evidence
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    factor = version["factors"][0] if version["factors"] else None
    config_hash = hashlib.sha256(
        json.dumps(
            version["config"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    robustness_scenarios = {}
    for name in ("double_cost", "turnover_75pct", "topk_80pct", "zero_retention_buffer"):
        scenario_artifacts = {}
        for artifact_name, suffix in (
            ("daily_report", "parquet"),
            ("fills", "parquet"),
            ("metrics", "json"),
        ):
            relative = Path("robustness") / name / f"{artifact_name}.{suffix}"
            path = manifest.parent / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{name}:{artifact_name}".encode())
            scenario_artifacts[artifact_name] = {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        robustness_scenarios[name] = {
            "passed": True,
            "artifacts": scenario_artifacts,
        }
    component_scenarios = {}
    for name in COMPONENT_COST_STRESS_MULTIPLIERS:
        scenario_artifacts = {}
        for artifact_name, suffix in (
            ("daily_report", "parquet"),
            ("fills", "parquet"),
            ("metrics", "json"),
        ):
            relative = Path("component_cost_stress") / name / f"{artifact_name}.{suffix}"
            path = manifest.parent / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{name}:{artifact_name}".encode())
            scenario_artifacts[artifact_name] = {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        component_scenarios[name] = {
            "passed": True,
            "artifacts": scenario_artifacts,
        }
    artifact_manifest = write_backtest_artifact_manifest(manifest.parent)
    return {
        "backtest_engine": "qlib",
        "backtest_engine_version": QLIB_ENGINE_VERSION,
        "qlib_native_backtest": True,
        "policy_version": POLICY_VERSION,
        "execution_model": {
            "method": version["config"].get("execution_method", "open"),
            "frequency": version["config"].get("execution_frequency", "day"),
            "price_assumption": "next-day open",
            "strategy_contract_hash": version["config"]["execution_contract_hash"],
        },
        "cost_model": CostModelConfig().to_dict(),
        "tracking_error": min(
            0.05,
            float(version["config"]["max_tracking_error"]) * 0.5,
        ),
        "max_drawdown": -0.10,
        "average_turnover": 0.10,
        "information_ratio": 1.0,
        "sharpe_ratio": 1.0,
        "sortino_ratio": 1.0,
        "sortino_status": "ok",
        "deflated_sharpe_probability": 0.99,
        "deflated_sharpe": {
            "status": "ok",
            "probability": 0.99,
            "trials": 1,
            "method_version": "bailey-lopez-de-prado-cross-trial-v2",
        },
        "formal_validation_passed": True,
        "formal_validation": {
            "contract_version": FORMAL_VALIDATION_CONTRACT_VERSION,
            "status": "passed",
            "pre_final_history": {
                "status": "completed",
                "contract_version": PRE_FINAL_HISTORY_CONTRACT_VERSION,
                "requested_periods": history_periods,
                "observed_periods": history_periods,
                "final_test_periods": final_periods,
                "trading_days": 3150,
                "minimum_trading_days": int(
                    version["config"].get("min_pre_final_history_days", 2520)
                ),
                "embargo_trading_days": 5,
                "minimum_embargo_trading_days": int(
                    version["config"].get("outer_embargo_days", 5)
                ),
                "overlaps_final_test": False,
                "uses_final_test_data": False,
                "execution_model": {
                    "method": "open",
                    "frequency": "day",
                    "scope": "pre_final_signal_and_portfolio_stability_proxy",
                    "minute_execution_claimed": False,
                },
            },
            "outer_walk_forward": {
                "status": "completed",
                "passed": True,
                "fold_count": 3,
                "minimum_test_metric": 0.0,
                "minimum_test_pass_rate": 0.60,
                "test_pass_rate": 1.0,
                "mean_test_metric": 0.01,
                "candidate_coverage": {
                    "required_group_trials": 1,
                    "provided_candidates": 1,
                    "scope": "frozen_strategy_no_search",
                },
                "folds": [
                    {"test_metric": 0.01, "test_passed": True},
                    {"test_metric": 0.01, "test_passed": True},
                    {"test_metric": 0.01, "test_passed": True},
                ],
            },
            "ablation": {
                "status": "passed",
                "runs": [
                    {
                        "removed_component_id": str(
                            item.get("factor_candidate_id") or item.get("id")
                        ),
                        "passed": True,
                        "metrics": {"annualized_excess_return": 0.01},
                    }
                    for item in version.get("factors", [])
                ]
                + [
                    {
                        "removed_component_id": str(item["id"]),
                        "passed": True,
                        "metrics": {"annualized_excess_return": 0.01},
                    }
                    for item in (
                        (
                            version.get("config", {})
                            .get("baseline_definition", {})
                            .get("factors", [])
                        )
                        if isinstance(
                            version.get("config", {}).get("baseline_definition"),
                            dict,
                        )
                        else []
                    )
                ],
            },
            "signal_decay": {
                "status": "completed",
                "frontier_version": "contiguous-zero-delay-frontier-v2",
                "maximum_supported_delay_bars": 1,
                "runs": [
                    {"delay_bars": 0, "passed": True},
                    {"delay_bars": 1, "passed": True},
                ],
            },
            "paired_block_bootstrap": {
                "status": "ok",
                "confidence_interval_95": [0.0001, 0.001],
                "one_sided_p_value": 0.01,
            },
            "multiple_testing": {
                "status": "not_applicable_single_trial",
                "trial_count": 1,
                "holm_adjusted_p_values": [0.01],
                "pbo": {
                    "status": "not_applicable_single_trial",
                    "pbo": None,
                },
            },
        },
        "robustness_pass_rate": 1.0,
        "robustness": {
            "passed": True,
            "pass_rate": 1.0,
            "scenarios": robustness_scenarios,
        },
        "component_cost_stress_pass_rate": 1.0,
        "component_cost_stress": {
            "passed": True,
            "pass_rate": 1.0,
            "scenarios": component_scenarios,
        },
        "rolling_pass_rate": 1.0,
        "rolling_window_count": 4,
        "event_stress_count": 5,
        "event_stress_pass_rate": 1.0,
        "event_stress_passed": True,
        "event_stress": {
            "state_source": "full_backtest_carried_positions",
            "position_state_method": "formal_fill_ledger_v1",
            "events": [
                {
                    "state_source": "full_backtest_carried_positions",
                    "return_state_source": "full_backtest_report_slice",
                    "start_holdings": {"SH600000": 100.0},
                    "state_fill_count": 1,
                }
                for _ in range(5)
            ],
        },
        "closed_trade_count": 40,
        "win_rate": 0.55,
        "average_win": 100.0,
        "average_loss": -80.0,
        "profit_loss_ratio": 1.25,
        "gross_realized_pnl": 600.0,
        "capacity_curve_points": 3,
        "capacity_curve_passed": True,
        "capacity": {
            "points": [
                {"notional": 5_000_000, "annualized_excess_return": 0.05},
                {"notional": 20_000_000, "annualized_excess_return": 0.04},
                {"notional": 100_000_000, "annualized_excess_return": 0.02},
            ],
            "passed": True,
        },
        "trading_days": 600,
        "eligibility": {
            "contract_version": "ashare-point-in-time-eligibility-v1",
            "rows": 1000,
            "eligible_rows": 800,
            "regulatory_data_available": True,
        },
        "provenance": {
            "frequency": "day",
            "dataset_identity_sha256": DATASET_IDENTITY,
            "snapshot_manifest_sha256": "b" * 64,
            "qlib_builder_sha256": "c" * 64,
            "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
            "source_volume_unit": "hand",
            "qlib_volume_unit": "share",
            "source_amount_unit": "thousand_cny",
            "qlib_amount_unit": "cny",
            "source_hand_size": 100,
            "index_volume_policy": "excluded_non_tradable_benchmark",
            "lineage_verified": True,
            "source_lineage_id": "9" * 64,
            "strategy_config_sha256": config_hash,
            "execution_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "artifact_manifest_version": artifact_manifest["version"],
            "artifact_manifest_sha256": artifact_manifest["sha256"],
            "artifact_manifest_file_count": artifact_manifest["file_count"],
            "factor_values_sha256": (
                {
                    factor["factor_candidate_id"]: hashlib.sha256(
                        Path(factor["values_path"]).read_bytes()
                    ).hexdigest()
                }
                if factor
                else {}
            ),
            "factor_code_sha256": (
                {factor["factor_candidate_id"]: factor["code_sha256"]}
                if factor
                else {}
            ),
            "formal_factor_values_sha256": formal_factor_hashes,
            "formal_factor_recompute_evidence": formal_factor_evidence,
            "qlib_version": "0.9.8",
            "qlib_commit": "d5379c520f66a39953bad76234a7019a72796fd0",
            "backtest_engine_version": QLIB_ENGINE_VERSION,
            "policy_version": POLICY_VERSION,
            "qlib_workflow": qlib_workflow_identity(),
        },
    }
