#!/usr/bin/env python3
"""Run a governed multi-factor Top-K backtest on a selected Qlib snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.cost_model import CostModelConfig
from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig
from quant_platform.qlib_backtest import (
    QLIB_ENGINE_VERSION,
    run_formal_qlib_backtest,
    run_qlib_validation_suites,
)
from quant_platform.qlib_policy_strategy import create_qlib_policy_strategy
from quant_platform.strategy_backtest import build_governed_signal, compose_factor_scores


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


def _latest_cross_section(frame: pd.DataFrame, when: pd.Timestamp, column: str) -> pd.Series:
    values = frame.copy()
    values["datetime"] = pd.to_datetime(values["datetime"], errors="coerce").dt.tz_localize(None)
    values = values[values["datetime"] <= when]
    if values.empty:
        raise ValueError(f"point-in-time metadata has no {column} values at {when.date()}")
    latest = values["datetime"].max()
    snapshot = values[values["datetime"] == latest]
    result = pd.to_numeric(snapshot[column], errors="coerce")
    result.index = snapshot["instrument"].astype(str)
    if result.index.has_duplicates or result.isna().any():
        raise ValueError(f"point-in-time metadata {column} is duplicated or incomplete")
    return result.astype(float)


def _latest_style_cross_section(frame: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    values = frame.copy()
    values["datetime"] = pd.to_datetime(values["datetime"], errors="coerce")
    values = values[values["datetime"] <= when]
    if values.empty:
        raise ValueError(f"point-in-time styles have no values at {when.date()}")
    values = values[values["datetime"] == values["datetime"].max()]
    result = values.set_index(values["instrument"].astype(str)).drop(
        columns=["datetime", "instrument"]
    )
    result = result.apply(pd.to_numeric, errors="coerce")
    if (
        result.index.has_duplicates
        or result.isna().any().any()
        or not np.isfinite(result.to_numpy(dtype=float)).all()
    ):
        raise ValueError("point-in-time style exposures are duplicated or incomplete")
    return result.astype(float)


def _qlib_cross_section(frame: pd.DataFrame, when: pd.Timestamp, column: str) -> pd.Series:
    values = frame.copy()
    dates = pd.to_datetime(values.index.get_level_values("datetime")).tz_localize(None)
    values.index = pd.MultiIndex.from_arrays(
        [dates, values.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    values = values.sort_index()
    available = values.loc[(slice(None, when), slice(None)), column]
    if available.empty:
        raise ValueError(f"Qlib has no {column} data at {when.date()}")
    return pd.to_numeric(
        available.xs(available.index.get_level_values("datetime").max()), errors="coerce"
    ).astype(float)


def _metadata_provider(
    memberships: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    styles: pd.DataFrame,
    execution_metadata: pd.DataFrame,
):
    membership = memberships.copy()
    membership["in_date"] = pd.to_datetime(membership["in_date"], errors="coerce")
    membership["out_date"] = pd.to_datetime(membership["out_date"], errors="coerce")

    def provide(when: Any, instruments: pd.Index) -> dict[str, Any]:
        timestamp = pd.Timestamp(when).tz_localize(None)
        active = (
            membership[
                (membership["in_date"] <= timestamp)
                & (membership["out_date"].isna() | (membership["out_date"] >= timestamp))
            ]
            .sort_values("in_date")
            .drop_duplicates("instrument", keep="last")
        )
        industries = active.set_index(active["instrument"].astype(str))["industry"].astype(str)
        benchmark = _latest_cross_section(benchmark_weights, timestamp, "weight")
        style = _latest_style_cross_section(styles, timestamp)
        benchmark_industries = industries.reindex(benchmark.index)
        if benchmark_industries.isna().any():
            raise ValueError("benchmark constituents are missing point-in-time industries")
        return {
            "industries": industries.reindex(instruments.astype(str)),
            "benchmark_weights": benchmark,
            "benchmark_industry_weights": benchmark.groupby(benchmark_industries).sum(),
            "style_exposures": style,
            "benchmark_style_exposure": style.reindex(benchmark.index).mul(
                benchmark, axis=0
            ).sum(),
            "prices": _qlib_cross_section(execution_metadata, timestamp, "$open").reindex(
                instruments.astype(str)
            ),
            "current_prices": _qlib_cross_section(
                execution_metadata, timestamp, "$close"
            ).reindex(instruments.astype(str)),
            "average_daily_values": (
                _qlib_cross_section(
                    execution_metadata, timestamp, "Ref(Mean($amount, 20), 1)"
                ).reindex(instruments.astype(str))
                * 1000.0
            ),
        }

    return provide


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
        str(item["candidate_id"]): _sha256_file(item["values_path"]) for item in manifest["factors"]
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
    from qlib.data import D

    qlib.init(provider_uri=args.provider_uri, region="cn")
    periods = manifest["periods"]
    liquidity_amount = D.features(
        instruments,
        ["$amount"],
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    ).mul(1000.0)
    execution_metadata = D.features(
        instruments,
        ["$open", "$close", "Ref(Mean($amount, 20), 1)"],
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    )
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
    style_fields = {
        "Log($total_mv)": "size",
        "1/$pb": "value",
        "($fund_quarter_revenue_yoy+$fund_quarter_profit_yoy)/2": "growth",
        "Std($close/Ref($close, 1)-1, 60)": "volatility",
    }
    style_exposures = D.features(
        instruments,
        list(style_fields),
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    ).rename(columns=style_fields).reset_index()
    if benchmark_weights is None or benchmark_weights.empty:
        raise ValueError("index-enhancement backtest requires historical benchmark weights")
    if style_exposures.empty:
        raise ValueError("index-enhancement backtest requires point-in-time style exposures")
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
    cost_model = CostModelConfig.from_mapping(config)
    policy = PortfolioPolicy(PortfolioPolicyConfig.from_mapping(config), cost_model)
    metadata = _metadata_provider(
        industry_memberships, benchmark_weights, style_exposures, execution_metadata
    )

    def run(
        start: str, end: str, costs: CostModelConfig, account: float | None = None
    ):
        scenario_policy = PortfolioPolicy(PortfolioPolicyConfig.from_mapping(config), costs)
        strategy = create_qlib_policy_strategy(
            signal=governed_signal,
            policy=scenario_policy,
            metadata_provider=metadata,
        )
        return run_formal_qlib_backtest(
            strategy=strategy,
            start_time=start,
            end_time=end,
            account=float(account if account is not None else config["capacity_notional"]),
            benchmark=manifest["benchmark"],
            cost_model=costs,
            execution_method=str(config.get("execution_method", "open")),
        )

    formal = run(periods["start"], periods["end"], cost_model)
    validation = run_qlib_validation_suites(
        runner=run,
        full_result=formal,
        start_time=periods["start"],
        end_time=periods["end"],
        cost_model=cost_model,
        config=config,
        capacity_runner=lambda notional: (
            formal
            if abs(notional - float(config["capacity_notional"])) < 1e-6
            else run(periods["start"], periods["end"], cost_model, notional)
        ),
    )
    qlib_report = formal.report
    qlib_positions = formal.positions
    metrics = {
        **formal.metrics,
        "policy_version": policy.version,
        "cost_model": cost_model.to_dict(),
        "execution_model": {
            "method": str(config.get("execution_method", "open")),
            "days": int(config.get("execution_days", 1)),
            "price_assumption": (
                "daily OHLC mean proxy"
                if config.get("execution_method") == "twap"
                else "daily vwap"
                if config.get("execution_method") == "vwap"
                else "next-day open"
            ),
        },
        "robustness": {"double_cost": validation["double_cost"]},
        "robustness_passed": validation["double_cost"]["passed"],
        "robustness_pass_rate": 1.0 if validation["double_cost"]["passed"] else 0.0,
        "rolling": validation["rolling"],
        "rolling_pass_rate": validation["rolling"]["pass_rate"],
        "rolling_passed": validation["rolling"]["passed"],
        "rolling_window_count": validation["rolling"]["window_count"],
        "event_stress": validation["event_stress"],
        "event_stress_count": validation["event_stress"]["event_count"],
        "event_stress_pass_rate": validation["event_stress"]["pass_rate"],
        "event_stress_passed": validation["event_stress"]["passed"],
        "capacity": validation["capacity"],
        "capacity_curve_points": len(validation["capacity"]["points"]),
        "capacity_curve_passed": validation["capacity"]["passed"],
        "provenance": {
            "dataset_identity_sha256": provider_provenance.get("dataset_identity_sha256"),
            "snapshot_manifest_sha256": provider_provenance.get("snapshot_manifest_sha256"),
            "qlib_builder_sha256": provider_provenance.get("qlib_builder_sha256"),
            "strategy_config_sha256": _canonical_sha256(config),
            "execution_manifest_sha256": _sha256_file(args.manifest),
            "factor_values_sha256": factor_value_hashes,
            "factor_code_sha256": factor_code_hashes,
            "qlib_version": getattr(qlib, "__version__", "unknown"),
            "backtest_engine_version": QLIB_ENGINE_VERSION,
            "policy_version": policy.version,
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    qlib_report.reset_index().to_parquet(output / "daily_returns.parquet", index=False)
    governed_signal.to_frame().to_parquet(output / "governed_signal.parquet")
    qlib_report.to_parquet(output / "qlib_portfolio_report.parquet")
    pd.to_pickle(qlib_positions, output / "qlib_positions.pkl")
    (output / "robustness.json").write_text(
        json.dumps(metrics["robustness"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "rolling.json").write_text(
        json.dumps(metrics["rolling"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "event_stress.json").write_text(
        json.dumps(metrics["event_stress"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "capacity_curve.json").write_text(
        json.dumps(metrics["capacity"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "status": "ok",
        "backtest_engine": "qlib",
        "metrics": metrics,
        "periods": periods,
        "benchmark": manifest["benchmark"],
        "artifacts": {
            "daily_returns": str(output / "daily_returns.parquet"),
            "governed_signal": str(output / "governed_signal.parquet"),
            "qlib_portfolio_report": str(output / "qlib_portfolio_report.parquet"),
            "qlib_positions": str(output / "qlib_positions.pkl"),
            "robustness": str(output / "robustness.json"),
            "rolling": str(output / "rolling.json"),
            "event_stress": str(output / "event_stress.json"),
            "capacity_curve": str(output / "capacity_curve.json"),
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
