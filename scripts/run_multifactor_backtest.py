#!/usr/bin/env python3
"""Run a governed multi-factor Top-K backtest on a selected Qlib snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.strategy_backtest import (
    build_governed_signal,
    compose_factor_scores,
    run_event_stress_suite,
    run_robustness_suite,
    run_rolling_suite,
    simulate_long_only_topk,
)


def _load(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported factor artifact: {source}")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    provider_provenance_path = Path(args.provider_uri) / "metadata" / "provenance.json"
    if not provider_provenance_path.exists():
        raise ValueError("formal Qlib backtest requires dataset provenance metadata")
    provider_provenance = json.loads(provider_provenance_path.read_text(encoding="utf-8"))
    factor_value_hashes = {
        str(item["candidate_id"]): _sha256_file(item["values_path"])
        for item in manifest["factors"]
    }
    factor_code_hashes = {
        str(item["candidate_id"]): item.get("code_sha256") for item in manifest["factors"]
    }

    factors = [
        (_load(item["values_path"]), float(item["weight"]), int(item["direction"]))
        for item in manifest["factors"]
    ]
    scores = compose_factor_scores(factors)
    instruments = sorted(set(scores.index.get_level_values("instrument")))

    import qlib
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data import D

    qlib.init(provider_uri=args.provider_uri, region="cn")
    periods = manifest["periods"]
    returns = D.features(
        instruments,
        ["Ref($open, -2)/Ref($open, -1)-1"],
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    )
    # Qlib stores the Tushare daily amount field in thousands of CNY. Ref(..., -1)
    # aligns the execution-day amount with a signal produced after the prior close.
    amount = D.features(
        instruments,
        ["Ref($amount, -1)"],
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    ).mul(1000.0)
    liquidity_amount = D.features(
        instruments,
        ["$amount"],
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    ).mul(1000.0)
    controls = D.features(
        instruments,
        ["Ref($open, -1)", "Ref($paused, -1)", "Ref($up_limit, -1)", "Ref($down_limit, -1)"],
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    )
    controls.columns = ["open", "paused", "up_limit", "down_limit"]
    industry_path = Path(args.provider_uri) / "metadata" / "industry_memberships.parquet"
    industry_memberships = pd.read_parquet(industry_path) if industry_path.exists() else None
    industry_cap_enabled = float(manifest["config"].get("max_industry_weight", 1.0)) < 1.0
    if industry_cap_enabled and industry_memberships is None:
        raise ValueError("industry-constrained backtest requires point-in-time industry metadata")
    benchmark_weight_path = Path(args.provider_uri) / "metadata" / "benchmark_weights.parquet"
    benchmark_weights = (
        pd.read_parquet(benchmark_weight_path) if benchmark_weight_path.exists() else None
    )
    if benchmark_weights is not None:
        benchmark_weights = benchmark_weights[
            benchmark_weights["benchmark"] == manifest["benchmark"]
        ].drop(columns=["benchmark"])
    style_path = Path(args.provider_uri) / "metadata" / "style_exposures.parquet"
    style_exposures = pd.read_parquet(style_path) if style_path.exists() else None
    if benchmark_weights is None or benchmark_weights.empty:
        raise ValueError("index-enhancement backtest requires historical benchmark weights")
    if style_exposures is None:
        raise ValueError("index-enhancement backtest requires point-in-time size exposures")
    benchmark_frame = D.features(
        [manifest["benchmark"]],
        ["Ref($open, -2)/Ref($open, -1)-1"],
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    )
    benchmark = benchmark_frame.iloc[:, 0].droplevel("instrument")
    config = manifest["config"]
    governed_signal = build_governed_signal(
        scores,
        topk=int(config["topk"]),
        liquidity_amount=liquidity_amount,
        industry_memberships=industry_memberships,
        benchmark_weights=benchmark_weights,
        style_exposures=style_exposures,
        max_industry_weight=float(config.get("max_industry_weight", 1.0)),
        max_industry_deviation=float(config.get("max_industry_deviation", 1.0)),
        min_average_daily_amount=float(config.get("min_average_daily_amount", 0.0)),
        liquidity_lookback_days=int(config.get("liquidity_lookback_days", 20)),
    )
    qlib_strategy = TopkDropoutStrategy(
        signal=governed_signal,
        topk=int(config["topk"]),
        n_drop=int(config["n_drop"]),
        only_tradable=True,
        forbid_all_trade_at_limit=False,
    )
    qlib_report, qlib_positions = backtest_daily(
        start_time=periods["start"],
        end_time=periods["end"],
        strategy=qlib_strategy,
        account=float(config["capacity_notional"]),
        benchmark=manifest["benchmark"],
        exchange_kwargs={
            "freq": "day",
            "deal_price": "open",
            "limit_threshold": (
                "Or($paused, Ge($open, $up_limit))",
                "Or($paused, Le($open, $down_limit))",
            ),
            "volume_threshold": (
                "current",
                f"{float(config['max_volume_participation'])} * $volume",
            ),
            "open_cost": float(config["open_cost"]),
            "close_cost": float(config["close_cost"]),
            "min_cost": float(config.get("min_commission", 5.0)),
            "trade_unit": 100,
        },
    )
    metrics, daily, positions = simulate_long_only_topk(
        scores,
        returns,
        benchmark,
        topk=int(config["topk"]),
        n_drop=int(config["n_drop"]),
        max_position_weight=float(config["max_position_weight"]),
        max_daily_turnover=float(config["max_daily_turnover"]),
        open_cost=float(config["open_cost"]),
        close_cost=float(config["close_cost"]),
        market_amount=amount,
        liquidity_amount=liquidity_amount,
        market_controls=controls,
        industry_memberships=industry_memberships,
        max_industry_weight=float(config.get("max_industry_weight", 1.0)),
        benchmark_weights=benchmark_weights,
        style_exposures=style_exposures,
        max_industry_deviation=float(config.get("max_industry_deviation", 1.0)),
        max_size_deviation=float(config.get("max_size_deviation", 10.0)),
        portfolio_notional=float(config["capacity_notional"]),
        max_volume_participation=float(config["max_volume_participation"]),
        execution_risk_enabled=True,
        max_daily_loss=float(config.get("max_daily_loss", 0.03)),
        stop_loss=float(config.get("stop_loss", 0.07)),
        take_profit_partial=float(config.get("take_profit_partial", 0.12)),
        take_profit_partial_fraction=float(config.get("take_profit_partial_fraction", 0.50)),
        take_profit=float(config.get("take_profit", 0.20)),
        max_drawdown_reduce=float(config.get("max_drawdown_reduce", 0.10)),
        max_drawdown_liquidate=float(config.get("max_drawdown_liquidate", 0.15)),
        drawdown_reduction_exposure=float(config.get("drawdown_reduction_exposure", 0.50)),
    )
    robustness = run_robustness_suite(
        scores,
        returns,
        benchmark,
        config=config,
        market_amount=amount,
        liquidity_amount=liquidity_amount,
        market_controls=controls,
        industry_memberships=industry_memberships,
        benchmark_weights=benchmark_weights,
        style_exposures=style_exposures,
    )
    rolling = run_rolling_suite(
        scores,
        returns,
        benchmark,
        config=config,
        market_amount=amount,
        liquidity_amount=liquidity_amount,
        market_controls=controls,
        industry_memberships=industry_memberships,
        benchmark_weights=benchmark_weights,
        style_exposures=style_exposures,
    )
    event_stress = run_event_stress_suite(
        scores,
        returns,
        benchmark,
        config=config,
        market_amount=amount,
        liquidity_amount=liquidity_amount,
        market_controls=controls,
        industry_memberships=industry_memberships,
        benchmark_weights=benchmark_weights,
        style_exposures=style_exposures,
    )
    execution_replay_metrics = dict(metrics)
    execution_replay_sha256 = _canonical_sha256(execution_replay_metrics)
    metrics.update(
        {
            "robustness_pass_rate": robustness["pass_rate"],
            "robustness_passed": robustness["passed"],
            "worst_scenario_excess_return": robustness["worst_annualized_excess_return"],
            "worst_scenario_drawdown": robustness["worst_max_drawdown"],
            "robustness": robustness,
            "rolling_pass_rate": rolling["pass_rate"],
            "rolling_passed": rolling["passed"],
            "rolling_window_count": rolling["window_count"],
            "rolling": rolling,
            "event_stress_pass_rate": event_stress["pass_rate"],
            "event_stress_passed": event_stress["passed"],
            "event_stress_count": event_stress["event_count"],
            "event_stress": event_stress,
            "execution_replay": execution_replay_metrics,
        }
    )
    qlib_net = qlib_report["return"] - qlib_report["cost"]
    qlib_excess = qlib_net - qlib_report["bench"]
    qlib_nav = (1.0 + qlib_net).cumprod()
    qlib_drawdown = qlib_nav / qlib_nav.cummax() - 1.0
    downside = qlib_net[qlib_net < 0].std(ddof=1)
    metrics.update(
        {
            "backtest_engine": "qlib",
            "qlib_native_backtest": True,
            "annualized_return": float(qlib_nav.iloc[-1] ** (252 / len(qlib_net)) - 1.0),
            "annualized_excess_return": float(qlib_excess.mean() * 252),
            "tracking_error": float(qlib_excess.std(ddof=1) * 252**0.5),
            "information_ratio": float(
                qlib_excess.mean() / qlib_excess.std(ddof=1) * 252**0.5
            ),
            "sharpe_ratio": float(qlib_net.mean() / qlib_net.std(ddof=1) * 252**0.5),
            "sortino_ratio": float(qlib_net.mean() / downside * 252**0.5),
            "max_drawdown": float(qlib_drawdown.min()),
            "average_turnover": float(qlib_report["turnover"].mean()),
            "total_cost": float(qlib_report["cost"].sum()),
            "trading_days": int(len(qlib_report)),
            "provenance": {
                "dataset_identity_sha256": provider_provenance.get(
                    "dataset_identity_sha256"
                ),
                "snapshot_manifest_sha256": provider_provenance.get(
                    "snapshot_manifest_sha256"
                ),
                "qlib_builder_sha256": provider_provenance.get("qlib_builder_sha256"),
                "strategy_config_sha256": _canonical_sha256(config),
                "execution_manifest_sha256": _sha256_file(args.manifest),
                "execution_replay_sha256": execution_replay_sha256,
                "factor_values_sha256": factor_value_hashes,
                "factor_code_sha256": factor_code_hashes,
                "qlib_version": getattr(qlib, "__version__", "unknown"),
            },
        }
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily.reset_index().to_parquet(output / "daily_returns.parquet", index=False)
    positions.to_parquet(output / "positions.parquet", index=False)
    governed_signal.to_frame().to_parquet(output / "governed_signal.parquet")
    qlib_report.to_parquet(output / "qlib_portfolio_report.parquet")
    pd.to_pickle(qlib_positions, output / "qlib_positions.pkl")
    (output / "robustness.json").write_text(
        json.dumps(robustness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "rolling.json").write_text(
        json.dumps(rolling, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "event_stress.json").write_text(
        json.dumps(event_stress, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "execution_replay.json").write_text(
        json.dumps(execution_replay_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "status": "ok",
        "backtest_engine": "qlib",
        "metrics": metrics,
        "periods": periods,
        "benchmark": manifest["benchmark"],
        "artifacts": {
            "daily_returns": str(output / "daily_returns.parquet"),
            "positions": str(output / "positions.parquet"),
            "governed_signal": str(output / "governed_signal.parquet"),
            "qlib_portfolio_report": str(output / "qlib_portfolio_report.parquet"),
            "qlib_positions": str(output / "qlib_positions.pkl"),
            "robustness": str(output / "robustness.json"),
            "rolling": str(output / "rolling.json"),
            "event_stress": str(output / "event_stress.json"),
            "execution_replay": str(output / "execution_replay.json"),
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
