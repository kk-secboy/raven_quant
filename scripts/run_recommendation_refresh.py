#!/usr/bin/env python3
"""Generate a recommendation snapshot from the same PortfolioPolicy used by Qlib."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.cost_model import CostModelConfig
from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.strategy_backtest import build_governed_signal, compose_factor_scores


def _load(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported factor artifact: {source}")


def _latest(frame: pd.DataFrame, when: pd.Timestamp, column: str) -> pd.Series:
    values = frame.copy()
    values["datetime"] = pd.to_datetime(values["datetime"], errors="coerce")
    values = values[values["datetime"] <= when]
    if values.empty:
        raise ValueError(f"metadata has no {column} snapshot at {when.date()}")
    values = values[values["datetime"] == values["datetime"].max()]
    result = pd.to_numeric(values[column], errors="coerce")
    result.index = values["instrument"].astype(str)
    if result.index.has_duplicates or result.isna().any():
        raise ValueError(f"metadata {column} snapshot is incomplete")
    return result.astype(float)


def _latest_styles(frame: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    values = frame.copy()
    values["datetime"] = pd.to_datetime(values["datetime"], errors="coerce")
    values = values[values["datetime"] <= when]
    if values.empty:
        raise ValueError(f"style metadata has no snapshot at {when.date()}")
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
        raise ValueError("style metadata snapshot is incomplete")
    return result.astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    as_of = pd.Timestamp(manifest["as_of_date"]).tz_localize(None)
    factors = [
        (_load(item["values_path"]), float(item["weight"]), int(item["direction"]))
        for item in manifest["factors"]
    ]
    scores = compose_factor_scores(factors)
    dates = pd.to_datetime(scores.index.get_level_values("datetime")).tz_localize(None)
    scores.index = pd.MultiIndex.from_arrays(
        [dates, scores.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    if as_of not in set(dates):
        raise ValueError("factor artifacts do not contain the requested recommendation date")

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=args.provider_uri, region="cn")
    config = manifest["config"]
    instruments = sorted(set(scores.index.get_level_values("instrument")))
    lookback = (as_of - pd.Timedelta(days=60)).date().isoformat()
    liquidity = D.features(
        instruments, ["$amount"], start_time=lookback, end_time=as_of.date().isoformat(), freq="day"
    ).mul(1000.0)
    execution_metadata = D.features(
        instruments,
        ["$open", "$close", "Ref(Mean($amount, 20), 1)"],
        start_time=as_of.date().isoformat(),
        end_time=as_of.date().isoformat(),
        freq="day",
    )
    execution_metadata.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(execution_metadata.index.get_level_values("datetime")).tz_localize(None),
            execution_metadata.index.get_level_values("instrument").astype(str),
        ],
        names=["datetime", "instrument"],
    )
    point_metadata = execution_metadata.xs(as_of, level="datetime")
    metadata_root = Path(args.provider_uri) / "metadata"
    memberships = pd.read_parquet(metadata_root / "industry_memberships.parquet")
    benchmark_frame = pd.read_parquet(metadata_root / "benchmark_weights.parquet")
    benchmark_frame = benchmark_frame[benchmark_frame["benchmark"] == manifest["benchmark"]].drop(
        columns=["benchmark"]
    )
    style_fields = {
        "Log($total_mv)": "size",
        "1/$pb": "value",
        "($fund_quarter_revenue_yoy+$fund_quarter_profit_yoy)/2": "growth",
        "Std($close/Ref($close, 1)-1, 60)": "volatility",
    }
    styles_frame = D.features(
        instruments,
        list(style_fields),
        start_time=lookback,
        end_time=as_of.date().isoformat(),
        freq="day",
    ).rename(columns=style_fields).reset_index()
    governed = build_governed_signal(
        scores.loc[(slice(lookback, as_of), slice(None))],
        topk=int(config["topk"]),
        liquidity_amount=liquidity,
        industry_memberships=memberships,
        benchmark_weights=benchmark_frame,
        style_exposures=styles_frame,
        max_industry_weight=float(config.get("max_industry_weight", 1.0)),
        max_industry_deviation=float(config.get("max_industry_deviation", 1.0)),
        min_average_daily_amount=float(config.get("min_average_daily_amount", 0.0)),
        liquidity_lookback_days=int(config.get("liquidity_lookback_days", 20)),
    )
    signal = governed.xs(as_of, level="datetime")
    memberships["in_date"] = pd.to_datetime(memberships["in_date"], errors="coerce")
    memberships["out_date"] = pd.to_datetime(memberships["out_date"], errors="coerce")
    active = (
        memberships[
            (memberships["in_date"] <= as_of)
            & (memberships["out_date"].isna() | (memberships["out_date"] >= as_of))
        ]
        .sort_values("in_date")
        .drop_duplicates("instrument", keep="last")
    )
    industries = active.set_index(active["instrument"].astype(str))["industry"].astype(str)
    benchmark = _latest(benchmark_frame, as_of, "weight")
    styles = _latest_styles(styles_frame, as_of)
    benchmark_industries = industries.reindex(benchmark.index)
    if benchmark_industries.isna().any() or styles.reindex(benchmark.index).isna().any().any():
        raise ValueError("benchmark metadata is incomplete")
    cost_model = CostModelConfig.from_mapping(config)
    policy = PortfolioPolicy(PortfolioPolicyConfig.from_mapping(config), cost_model)
    previous = {
        item["instrument"]: item["weight"] for item in manifest.get("previous_holdings", [])
    }
    previous_snapshot = manifest.get("previous_snapshot") or {}
    previous_performance = manifest.get("previous_performance") or {}
    prior_value = float(
        previous_performance.get("hypothetical_value") or manifest["portfolio_value"]
    )
    hypothetical_return = 0.0
    benchmark_return = 0.0
    previous_as_of = previous_snapshot.get("effective_date") or previous_snapshot.get("as_of_date")
    previous_holdings = previous_snapshot.get("holdings") or []
    if previous_as_of and previous_holdings:
        held = [str(item["instrument"]) for item in previous_holdings]
        close = D.features(
            held,
            ["$close"],
            start_time=previous_as_of,
            end_time=as_of.date().isoformat(),
            freq="day",
        )["$close"].unstack("instrument")
        close.index = pd.to_datetime(close.index).tz_localize(None)
        returns = close.ffill().iloc[-1] / close.ffill().iloc[0] - 1.0
        hypothetical_return = float(
            sum(
                float(item["weight"]) * float(returns.get(str(item["instrument"]), 0.0))
                for item in previous_holdings
            )
        )
        benchmark_close = D.features(
            [manifest["benchmark"]],
            ["$close"],
            start_time=previous_as_of,
            end_time=as_of.date().isoformat(),
            freq="day",
        )["$close"]
        if len(benchmark_close) >= 2:
            benchmark_return = float(benchmark_close.iloc[-1] / benchmark_close.iloc[0] - 1.0)
    gross_value = max(0.0, prior_value * (1.0 + hypothetical_return))
    previous_peak = float(previous_performance.get("high_water_mark") or prior_value)
    current_peak = max(previous_peak, gross_value)
    current_drawdown = gross_value / current_peak - 1.0 if current_peak > 0 else 0.0
    cost_basis = {
        str(item["instrument"]): float(item.get("average_cost") or 0.0)
        for item in previous_holdings
        if float(item.get("average_cost") or 0.0) > 0
    }
    take_profit_stages = {
        str(item["instrument"]): int(item.get("take_profit_stage") or 0)
        for item in previous_holdings
    }
    decision = policy.decide(
        signal,
        previous,
        industries=industries,
        benchmark_weights=benchmark,
        benchmark_industry_weights=benchmark.groupby(benchmark_industries).sum(),
        style_exposures=styles,
        benchmark_style_exposure=styles.reindex(benchmark.index).mul(
            benchmark, axis=0
        ).sum(),
        prices=pd.to_numeric(point_metadata["$open"], errors="coerce"),
        current_prices=pd.to_numeric(point_metadata["$close"], errors="coerce"),
        cost_basis=cost_basis,
        take_profit_stages=take_profit_stages,
        execution_state=(previous_snapshot.get("position_state") or {}).get("execution"),
        portfolio_drawdown=current_drawdown,
        daily_return=hypothetical_return,
        average_daily_values=(
            pd.to_numeric(point_metadata["Ref(Mean($amount, 20), 1)"], errors="coerce") * 1000.0
        ),
        portfolio_value=max(gross_value, 1.0),
        risk_exposure=float(manifest.get("risk_exposure", 1.0)),
    )
    average_values = (
        pd.to_numeric(point_metadata["Ref(Mean($amount, 20), 1)"], errors="coerce") * 1000.0
    )
    estimated_cost = 0.0
    for change in decision.changes:
        gross = abs(float(change["weight_change"])) * prior_value
        if gross <= 0:
            continue
        adv = float(average_values.get(change["instrument"], 0.0))
        participation = (
            min(cost_model.max_volume_participation, gross / adv)
            if adv > 0
            else cost_model.max_volume_participation
        )
        estimated_cost += cost_model.estimate(
            side="buy" if change["action"] == "increase" else "sell",
            gross_value=gross,
            participation=participation,
        )
    hypothetical_value = max(0.0, gross_value - estimated_cost)
    high_water_mark = max(previous_peak, hypothetical_value)
    drawdown = hypothetical_value / high_water_mark - 1.0 if high_water_mark > 0 else 0.0
    calendar = pd.DatetimeIndex(
        D.calendar(
            start_time=as_of.date().isoformat(),
            end_time=(as_of + pd.Timedelta(days=14)).date().isoformat(),
            freq="day",
        )
    ).tz_localize(None)
    future = calendar[calendar > as_of]
    if not len(future):
        raise ValueError("Qlib calendar has no effective trading date")
    changes = {item["instrument"]: item for item in decision.changes}

    def next_average_cost(instrument: str, target_weight: float) -> float:
        mark = float(point_metadata.loc[instrument, "$close"])
        old_weight = float(previous.get(instrument, 0.0))
        old_cost = float(cost_basis.get(instrument, mark))
        increase = max(0.0, target_weight - old_weight)
        if old_weight <= 0 or increase <= 0:
            return mark if old_weight <= 0 else old_cost
        old_shares = old_weight * max(gross_value, 1.0) / old_cost
        new_shares = increase * max(gross_value, 1.0) / mark
        return (old_shares * old_cost + new_shares * mark) / (old_shares + new_shares)

    result = {
        "status": "ok",
        "portfolio_id": manifest["portfolio_id"],
        "strategy_version_id": manifest["strategy_version_id"],
        "as_of_date": as_of.date().isoformat(),
        "effective_date": future[0].date().isoformat(),
        "policy_version": decision.policy_version,
        "backtest_engine_version": QLIB_ENGINE_VERSION,
        "dataset": manifest["dataset"],
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "cost_model": decision.cost_model,
        "position_state": decision.position_state,
        "risk_summary": {
            "expected_turnover": decision.expected_turnover,
            "drawdown": drawdown,
            "high_water_mark": high_water_mark,
            "events": decision.risk_events,
            "execution_method": config.get("execution_method", "open"),
            "execution_days": int(config.get("execution_days", 1)),
        },
        "hypothetical_observation": {
            "trade_date": as_of.date().isoformat(),
            "hypothetical_value": hypothetical_value,
            "daily_return": hypothetical_return,
            "benchmark_return": benchmark_return,
            "drawdown": drawdown,
            "high_water_mark": high_water_mark,
            "turnover": decision.expected_turnover,
            "estimated_cost": estimated_cost,
        },
        "reasons": decision.reasons,
        "cash_weight": max(0.0, 1.0 - sum(decision.target_weights.values())),
        "holdings": [
            {
                "instrument": instrument,
                "weight": weight,
                "previous_weight": changes.get(instrument, {}).get("previous_weight", weight),
                "weight_change": changes.get(instrument, {}).get("weight_change", 0.0),
                "action": changes.get(instrument, {}).get("action", "hold"),
                "reason": changes.get(instrument, {}).get("reason", "unchanged target"),
                "average_cost": next_average_cost(instrument, weight),
                "take_profit_stage": int(
                    decision.position_state.get("take_profit_stages", {}).get(instrument, 0)
                ),
            }
            for instrument, weight in decision.target_weights.items()
        ],
        "changes": decision.changes,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
