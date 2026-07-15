from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date
from pathlib import Path

from quant_platform.cost_model import CostModelConfig
from quant_platform.portfolio_policy import POLICY_VERSION
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.research_store import ResearchStore
from quant_platform.strategy_store import StrategyStore

DATASET_IDENTITY = "a" * 64
PERIODS = {
    "train_start": date(2018, 1, 1),
    "train_end": date(2021, 12, 31),
    "valid_start": date(2022, 1, 1),
    "valid_end": date(2023, 12, 31),
    "test_start": date(2024, 1, 1),
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
        "selection_start": "2022-05-27",
        "coverage_pass_rate": 0.99,
        "mean_coverage_ratio": 0.95,
        "constant_day_rate": 0.0,
        "direction": "inverted",
    }


def create_promoted_factor(
    database_url: str,
    tmp_path: Path,
    *,
    dataset: str = "snapshot",
) -> dict:
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
        dataset_identity_sha256=DATASET_IDENTITY,
        **PERIODS,
        metrics=metrics,
        artifact_path=str(artifact),
        recomputed_values_path=str(recomputed_path),
        recomputed_values_sha256=recomputed_sha256,
        recompute_evidence={
            "executor_version": "factor-recompute-v1",
            "code_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
            "dataset_identity_sha256": DATASET_IDENTITY,
            "provider_input_sha256": "1" * 64,
            "periods": {key: value.isoformat() for key, value in PERIODS.items()},
            "submitted_comparison": {
                "available": True,
                "exact_match": True,
                "submitted_sha256": hashlib.sha256(values_path.read_bytes()).hexdigest(),
            },
            "authoritative_values_sha256": recomputed_sha256,
        },
    )
    return store.promote(
        candidate["id"], actor="factor-owner", reason="Approved governed test evidence."
    )


def create_strategy_version(database_url: str, tmp_path: Path, *, dataset: str = "snapshot") -> str:
    factor = create_promoted_factor(database_url, tmp_path, dataset=dataset)
    strategy = StrategyStore(database_url).create(
        name=f"strategy-{uuid.uuid4().hex}",
        description="Governed strategy fixture for v2 tests.",
        benchmark="SH000300",
        universe="cn_all",
        factors=[{"candidate_id": factor["id"], "weight": 1.0}],
        config={
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
            "max_volume_participation": 0.01,
            "min_commission": 5.0,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
        },
        actor="test",
    )
    return str(strategy["versions"][0]["id"])


def formal_backtest_metrics(version: dict, manifest: Path) -> dict:
    factor = version["factors"][0]
    config_hash = hashlib.sha256(
        json.dumps(
            version["config"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "backtest_engine": "qlib",
        "backtest_engine_version": QLIB_ENGINE_VERSION,
        "qlib_native_backtest": True,
        "policy_version": POLICY_VERSION,
        "cost_model": CostModelConfig().to_dict(),
        "tracking_error": 0.05,
        "max_drawdown": -0.10,
        "average_turnover": 0.10,
        "information_ratio": 1.0,
        "sharpe_ratio": 1.0,
        "sortino_ratio": 1.0,
        "robustness_pass_rate": 1.0,
        "rolling_pass_rate": 1.0,
        "rolling_window_count": 4,
        "event_stress_count": 5,
        "event_stress_pass_rate": 1.0,
        "event_stress_passed": True,
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
        "provenance": {
            "dataset_identity_sha256": DATASET_IDENTITY,
            "snapshot_manifest_sha256": "b" * 64,
            "qlib_builder_sha256": "c" * 64,
            "strategy_config_sha256": config_hash,
            "execution_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "factor_values_sha256": {
                factor["factor_candidate_id"]: hashlib.sha256(
                    Path(factor["values_path"]).read_bytes()
                ).hexdigest()
            },
            "factor_code_sha256": {factor["factor_candidate_id"]: factor["code_sha256"]},
            "qlib_version": "0.9.8",
            "backtest_engine_version": QLIB_ENGINE_VERSION,
            "policy_version": POLICY_VERSION,
        },
    }
