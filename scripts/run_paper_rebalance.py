#!/usr/bin/env python3
"""Generate one governed next-open paper rebalance from an approved strategy."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.paper_trading import build_rebalance_plan
from quant_platform.strategy_backtest import build_governed_signal, compose_factor_scores


def _load(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported factor artifact: {source}")


def _portfolio_metadata(provider_uri: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(provider_uri).resolve() / "metadata"
    paths = {
        "industry": root / "industry_memberships.parquet",
        "benchmark": root / "benchmark_weights.parquet",
        "style": root / "style_exposures.parquet",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "paper rebalance requires Qlib point-in-time metadata: " + ", ".join(missing)
        )
    return (
        pd.read_parquet(paths["industry"]),
        pd.read_parquet(paths["benchmark"]),
        pd.read_parquet(paths["style"]),
    )


def _industry_map(memberships: pd.DataFrame, trade_date: pd.Timestamp) -> dict[str, str]:
    frame = memberships.copy()
    frame["in_date"] = pd.to_datetime(frame["in_date"], errors="coerce")
    frame["out_date"] = pd.to_datetime(frame["out_date"], errors="coerce")
    active = frame[
        (frame["in_date"] <= trade_date)
        & (frame["out_date"].isna() | (frame["out_date"] >= trade_date))
    ]
    active = active.sort_values("in_date").drop_duplicates("instrument", keep="last")
    return dict(
        zip(
            active["instrument"].astype(str),
            active["industry"].astype(str),
            strict=True,
        )
    )


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
    dates = pd.DatetimeIndex(scores.index.get_level_values("datetime")).tz_localize(None)
    scores.index = pd.MultiIndex.from_arrays(
        [dates, scores.index.get_level_values("instrument")],
        names=["datetime", "instrument"],
    )
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=args.provider_uri, region="cn")
    if as_of not in dates:
        raise ValueError("factor artifacts do not contain the requested signal date")
    config = manifest["config"]
    score_instruments = sorted(set(scores.index.get_level_values("instrument").astype(str)))
    signal_lookback_start = (as_of - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    liquidity_amount = D.features(
        score_instruments,
        ["$amount"],
        start_time=signal_lookback_start,
        end_time=as_of.strftime("%Y-%m-%d"),
        freq="day",
    ).mul(1000.0)
    industry_memberships, benchmark_weights, style_exposures = _portfolio_metadata(
        args.provider_uri
    )
    benchmark_weights = benchmark_weights[
        benchmark_weights["benchmark"] == manifest["benchmark"]
    ].drop(columns=["benchmark"])
    recent_scores = scores.loc[
        pd.to_datetime(scores.index.get_level_values("datetime"))
        >= pd.Timestamp(signal_lookback_start)
    ]
    governed_signal = build_governed_signal(
        recent_scores,
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
    try:
        signal = governed_signal.xs(as_of, level="datetime")
    except KeyError as exc:
        raise ValueError("requested signal date has no governed eligible portfolio") from exc
    calendar = pd.DatetimeIndex(
        D.calendar(
            start_time=as_of.strftime("%Y-%m-%d"),
            end_time=(as_of + pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
            freq="day",
        )
    ).tz_localize(None)
    future = calendar[calendar > as_of]
    if not len(future):
        raise ValueError("Qlib calendar has no next trading day after the signal date")
    trade_date = future[0]
    instruments = sorted(
        set(signal.index.astype(str)) | {item["instrument"] for item in manifest["positions"]}
    )
    frame = D.features(
        instruments,
        [
            "$open/$factor",
            "$close/$factor",
            "$paused",
            "$volume*$factor",
            "$amount",
            "$up_limit",
            "$down_limit",
        ],
        start_time=trade_date.strftime("%Y-%m-%d"),
        end_time=trade_date.strftime("%Y-%m-%d"),
        freq="day",
    )
    frame.columns = [
        "open",
        "close",
        "paused",
        "volume",
        "amount",
        "up_limit",
        "down_limit",
    ]
    market = frame.droplevel("datetime")
    lookback_calendar = pd.DatetimeIndex(
        D.calendar(
            start_time=(trade_date - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
            end_time=trade_date.strftime("%Y-%m-%d"),
            freq="day",
        )
    ).tz_localize(None)
    lookback_start = (
        lookback_calendar[-20] if len(lookback_calendar) >= 20 else lookback_calendar[0]
    )
    amount_frame = D.features(
        instruments,
        ["$amount"],
        start_time=lookback_start.strftime("%Y-%m-%d"),
        end_time=trade_date.strftime("%Y-%m-%d"),
        freq="day",
    )
    average_amount = amount_frame.iloc[:, 0].groupby(level="instrument").mean() * 1000.0
    market["amount"] = pd.to_numeric(market["amount"], errors="coerce").fillna(0.0) * 1000.0
    market["average_amount"] = market.index.map(average_amount).astype(float)
    industries = _industry_map(industry_memberships, trade_date)
    market["industry"] = market.index.map(industries)
    benchmark_frame = D.features(
        [manifest["benchmark"]],
        ["$close/$factor"],
        start_time=as_of.strftime("%Y-%m-%d"),
        end_time=trade_date.strftime("%Y-%m-%d"),
        freq="day",
    )
    benchmark_close = benchmark_frame.iloc[:, 0].droplevel("instrument").dropna()
    benchmark_return = (
        float(benchmark_close.iloc[-1] / benchmark_close.iloc[-2] - 1)
        if len(benchmark_close) >= 2
        else None
    )
    # Qlib's supported local WSL runtime is Python 3.10, before datetime.UTC.
    fill_time = datetime.combine(
        trade_date.date(),
        time(1, 30),
        tzinfo=timezone.utc,  # noqa: UP017
    )
    plan = build_rebalance_plan(
        signal,
        market,
        manifest["positions"],
        nav=float(manifest["nav"]),
        cash=float(manifest["cash"]),
        high_water_mark=float(manifest.get("high_water_mark") or manifest["nav"]),
        portfolio_status=str(manifest.get("portfolio_status") or "active"),
        topk=int(config["topk"]),
        n_drop=int(config["n_drop"]),
        max_position_weight=float(config["max_position_weight"]),
        max_daily_turnover=float(config["max_daily_turnover"]),
        max_daily_loss=float(config.get("max_daily_loss", 0.03)),
        stop_loss=float(config.get("stop_loss", 0.07)),
        take_profit_partial=float(config.get("take_profit_partial", 0.12)),
        take_profit_partial_fraction=float(config.get("take_profit_partial_fraction", 0.50)),
        take_profit=float(config.get("take_profit", 0.20)),
        max_drawdown_reduce=float(config.get("max_drawdown_reduce", 0.10)),
        max_drawdown_liquidate=float(config.get("max_drawdown_liquidate", 0.15)),
        drawdown_reduction_exposure=float(config.get("drawdown_reduction_exposure", 0.50)),
        max_industry_weight=float(config.get("max_industry_weight", 0.30)),
        min_average_daily_amount=float(config.get("min_average_daily_amount", 500_000_000)),
        max_volume_participation=float(config.get("max_volume_participation", 0.01)),
        open_cost=float(config["open_cost"]),
        close_cost=float(config["close_cost"]),
        slippage=float(manifest.get("slippage", 0.0005)),
        fill_time=fill_time,
    )
    result = {
        "status": "ok",
        "strategy_version_id": manifest["strategy_version_id"],
        "signal_engine": "qlib_governed_signal",
        "provenance": {
            "daily_dataset_identity_sha256": manifest["daily_provenance"].get(
                "dataset_identity_sha256"
            ),
            "daily_dataset_lineage_id": manifest["daily_provenance"].get(
                "dataset_lineage_id"
            ),
        },
        "as_of_date": as_of.date().isoformat(),
        "trade_date": trade_date.date().isoformat(),
        "benchmark_return": benchmark_return,
        **plan,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
