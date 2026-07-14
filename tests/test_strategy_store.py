import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from quant_platform.research_store import ResearchStore
from quant_platform.strategy_store import StrategyStore

PERIODS = {
    "train_start": date(2018, 1, 1),
    "train_end": date(2021, 12, 31),
    "valid_start": date(2022, 1, 1),
    "valid_end": date(2023, 12, 31),
    "test_start": date(2024, 1, 1),
    "test_end": date(2026, 7, 10),
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provenance(
    factor_id: str,
    execution_replay: dict | None = None,
    *,
    code_sha256: str | None = None,
    values_sha256: str | None = None,
) -> dict:
    digest = "a" * 64
    return {
        "dataset_identity_sha256": digest,
        "snapshot_manifest_sha256": digest,
        "qlib_builder_sha256": digest,
        "strategy_config_sha256": digest,
        "execution_manifest_sha256": digest,
        "execution_replay_sha256": (
            _canonical_sha256(execution_replay) if execution_replay is not None else digest
        ),
        "factor_values_sha256": {factor_id: values_sha256 or digest},
        "factor_code_sha256": {factor_id: code_sha256 or digest},
        "qlib_version": "0.9.8",
    }


def _execution_risk_evidence() -> dict:
    return {
        "execution_risk_overlay_enforced": True,
        "execution_replay": {
            "execution_risk_overlay_enforced": True,
            "execution_model": "next_open",
            "max_drawdown": -0.14,
            "execution_risk_thresholds": {
                "max_daily_loss": 0.03,
                "stop_loss": 0.07,
                "take_profit_partial": 0.12,
                "take_profit_partial_fraction": 0.50,
                "take_profit": 0.20,
                "max_drawdown_reduce": 0.10,
                "max_drawdown_liquidate": 0.15,
                "drawdown_reduction_exposure": 0.50,
            }
        },
    }


def _factor(store: ResearchStore, tmp_path: Path, *, promoted: bool) -> dict:
    run = store.create_run(
        kind="factor",
        objective="Create a factor for governed strategy testing.",
        dataset="snapshot",
        requested_by="researcher",
        budget={"loop_n": 1, "duration": "30m"},
        config={},
        artifact_path=tmp_path,
    )
    code_path = tmp_path / "factor.py"
    values_path = tmp_path / "factor.h5"
    code_path.write_text("def factor(frame):\n    return frame['close']\n", encoding="utf-8")
    values_path.write_bytes(b"immutable-factor-values")
    candidate = store.add_candidate(
        run["id"],
        name="governed_quality",
        description="Quality factor for strategy tests.",
        formulation="roe - leverage",
        variables={},
        source_iteration=0,
        code_path=str(code_path),
        values_path=str(values_path),
        code_sha256=hashlib.sha256(code_path.read_bytes()).hexdigest(),
        rdagent_decision=True,
        rdagent_feedback="ok",
    )
    if not promoted:
        return candidate
    metrics = {
        "ic": 0.04,
        "icir": 0.8,
        "rank_ic": 0.045,
        "rank_icir": 0.75,
        "turnover": 0.30,
        "max_correlation": 0.40,
        "cost_adjusted_return": 0.06,
        "valid_ic": 0.035,
        "test_days": 500,
        "direction": "inverted",
    }
    evaluation_path = tmp_path / "factor-evaluation.json"
    evaluation_path.write_text(
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
    store.record_evaluation(
        candidate["id"],
        dataset="snapshot",
        **PERIODS,
        metrics=metrics,
        artifact_path=str(evaluation_path),
    )
    return store.promote(
        candidate["id"],
        actor="factor-owner",
        reason="Passed independent factor validation for strategy use.",
    )


def test_strategy_requires_promoted_factor_and_successful_risk_gated_backtest(
    tmp_path: Path, database_url: str
) -> None:
    research = ResearchStore(database_url)
    factor = _factor(research, tmp_path, promoted=True)
    strategies = StrategyStore(database_url)
    strategy = strategies.create(
        name="CSI300 governed quality",
        description="Long-only factor strategy with explicit risk limits.",
        benchmark="SH000300",
        universe="cn_all",
        factors=[{"candidate_id": factor["id"], "weight": 2.0}],
        config={
            "topk": 50,
            "n_drop": 5,
            "max_tracking_error": 0.12,
            "max_drawdown": 0.25,
            "max_turnover": 0.60,
            "min_information_ratio": 0.0,
            "min_sharpe_ratio": 0.0,
            "min_sortino_ratio": 0.0,
            "min_robustness_pass_rate": 0.75,
            "min_backtest_days": 504,
            "capacity_notional": 5_000_000,
            "max_volume_participation": 0.01,
            "min_capacity_fill_ratio": 0.95,
            "max_industry_deviation": 0.05,
            "max_size_deviation": 0.30,
            "min_average_daily_amount": 500_000_000,
            "min_commission": 5.0,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "execution_model": "next_open",
        },
        actor="strategy-owner",
    )
    version = strategy["versions"][0]
    assert version["factors"][0]["weight"] == 1.0
    assert version["factors"][0]["direction"] == -1
    assert version["factors"][0]["factor_evaluation_id"]

    with pytest.raises(ValueError, match="successful backtest"):
        strategies.approve(
            version["id"],
            actor="risk-owner",
            reason="Need a backtest before this can be approved.",
        )

    backtest = strategies.create_backtest(
        version_id=version["id"],
        dataset="snapshot",
        periods={"start": "2024-01-01", "end": "2026-07-10"},
        artifact_path=tmp_path / "backtest",
    )
    strategies.mark_backtest(
        backtest["id"],
        "succeeded",
        metrics={
            "backtest_engine": "qlib",
            "qlib_native_backtest": True,
            "tracking_error": 0.08,
            "max_drawdown": -0.16,
            "average_turnover": 0.40,
            "information_ratio": 0.70,
        },
    )
    with pytest.raises(ValueError, match="provenance"):
        strategies.approve(
            version["id"],
            actor="risk-owner",
            reason="Missing reproducibility evidence must fail closed.",
        )
    execution_evidence = _execution_risk_evidence()
    replay_path = tmp_path / "backtest" / "execution_replay.json"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps(execution_evidence["execution_replay"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "strategy_version_id": version["id"],
        "dataset": backtest["dataset"],
        "benchmark": version["benchmark"],
        "periods": backtest["periods"],
        "config": version["config"],
        "factors": [
            {
                "candidate_id": item["factor_candidate_id"],
                "values_path": item["values_path"],
                "code_sha256": item["code_sha256"],
                "weight": item["weight"],
                "direction": item["direction"],
            }
            for item in version["factors"]
        ],
    }
    manifest_path = tmp_path / "backtest" / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    provenance = _provenance(
        factor["id"],
        execution_evidence["execution_replay"],
        code_sha256=factor["code_sha256"],
        values_sha256=hashlib.sha256(Path(factor["values_path"]).read_bytes()).hexdigest(),
    )
    provenance["strategy_config_sha256"] = _canonical_sha256(version["config"])
    provenance["execution_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    strategies.mark_backtest(
        backtest["id"],
        "succeeded",
        metrics={
            "backtest_engine": "qlib",
            "qlib_native_backtest": True,
            "provenance": provenance,
            "tracking_error": 0.08,
            "max_drawdown": -0.16,
            "average_turnover": 0.40,
            "information_ratio": 0.70,
        },
    )
    with pytest.raises(ValueError, match="sharpe_ratio"):
        strategies.approve(
            version["id"],
            actor="risk-owner",
            reason="Missing robustness and capacity evidence must fail closed.",
        )
    strategies.mark_backtest(
        backtest["id"],
        "succeeded",
        metrics={
            "backtest_engine": "qlib",
            "qlib_native_backtest": True,
            "provenance": provenance,
            "tracking_error": 0.08,
            "max_drawdown": -0.16,
            "average_turnover": 0.40,
            "information_ratio": 0.70,
            "sharpe_ratio": 0.80,
            "sortino_ratio": 1.10,
            "robustness_pass_rate": 0.75,
            "rolling_pass_rate": 0.80,
            "rolling_window_count": 5,
            "event_stress_pass_rate": 0.80,
            "event_stress_count": 5,
            "capacity_fill_ratio": 0.98,
            "trading_days": 600,
            "max_industry_deviation": 0.04,
            "max_size_deviation": 0.20,
            "industry_controls_enforced": True,
            "benchmark_weights_enforced": True,
            "size_neutralization_enforced": True,
            "liquidity_filter_enforced": True,
            "market_controls_enforced": True,
            "min_commission": 5.0,
            **execution_evidence,
        },
    )
    persisted_metrics = strategies.get_backtest(backtest["id"])["metrics"]
    strategies.validate_backtest_artifacts(backtest["id"], persisted_metrics)
    original_manifest = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        strategies.validate_backtest_artifacts(backtest["id"], persisted_metrics)
    manifest_path.write_text(original_manifest, encoding="utf-8")
    approved = strategies.approve(
        version["id"],
        actor="risk-owner",
        reason="Backtest passed tracking error, drawdown, turnover, and IR limits.",
    )
    assert approved["status"] == "approved"

    version_two = strategies.create_version(
        strategy["id"],
        benchmark="SH000300",
        universe="cn_all",
        factors=[{"candidate_id": factor["id"], "weight": 1.0}],
        config=version["config"],
        actor="strategy-owner",
    )
    assert version_two["version"] == 2
    assert version_two["status"] == "draft"
    assert strategies.get_version(version["id"])["status"] == "approved"

    invalid_window = strategies.create_backtest(
        version_id=version["id"],
        dataset="snapshot",
        periods={"start": "2023-01-01", "end": "2026-07-10"},
        artifact_path=tmp_path / "invalid-window",
    )
    strategies.mark_backtest(
        invalid_window["id"],
        "succeeded",
        metrics={
            "tracking_error": 0.08,
            "max_drawdown": -0.16,
            "average_turnover": 0.40,
            "information_ratio": 0.70,
            "sharpe_ratio": 0.80,
            "sortino_ratio": 1.10,
            "robustness_pass_rate": 1.0,
            "rolling_pass_rate": 1.0,
            "rolling_window_count": 5,
            "event_stress_pass_rate": 1.0,
            "event_stress_count": 5,
            "capacity_fill_ratio": 0.98,
            "trading_days": 600,
        },
    )
    with pytest.raises(ValueError, match="outside factor test window"):
        strategies.approve(
            version["id"],
            actor="risk-owner",
            reason="An in-sample backtest must not replace independent evidence.",
        )


def test_strategy_rejects_unpromoted_factor(tmp_path: Path, database_url: str) -> None:
    research = ResearchStore(database_url)
    factor = _factor(research, tmp_path, promoted=False)
    strategies = StrategyStore(database_url)
    with pytest.raises(ValueError, match="promoted factors"):
        strategies.create(
            name="invalid strategy",
            description="Must not accept a raw RD-Agent candidate.",
            benchmark="SH000300",
            universe="cn_all",
            factors=[{"candidate_id": factor["id"], "weight": 1.0}],
            config={},
            actor="strategy-owner",
        )
