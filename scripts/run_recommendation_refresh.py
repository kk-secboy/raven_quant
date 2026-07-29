#!/usr/bin/env python3
"""Generate a recommendation snapshot from the same PortfolioPolicy used by Qlib."""

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

from quant_data.availability import filter_available
from quant_data.execution_contract import (
    require_daily_qlib_contract,
    require_minute_execution_contract,
    require_strategy_execution_contract,
)
from quant_platform.cost_model import CostModelConfig
from quant_platform.eligibility import eligibility_statistics
from quant_platform.portfolio_policy import (
    PortfolioPolicy,
    PortfolioPolicyConfig,
    is_rebalance_due,
)
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.qlib_factor_baseline import (
    FACTOR_SOURCE_PROMOTED_ONLY,
    combine_factor_sources,
    normalize_qlib_baseline_values,
)
from quant_platform.qlib_workflow import qlib_workflow_run
from quant_platform.risk_math import estimate_covariance
from quant_platform.strategy_backtest import build_governed_signal, compose_factor_scores


def _load(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported factor artifact: {source}")


def _next_known_trading_date(provider_uri: str | Path, as_of: pd.Timestamp) -> str:
    source = Path(provider_uri) / "metadata" / "known_trading_calendar.parquet"
    if not source.is_file():
        raise ValueError(
            "Qlib dataset has no immutable known trading calendar for next-session execution"
        )
    calendar = pd.to_datetime(
        pd.read_parquet(source)["date"], errors="coerce"
    ).dropna()
    future = calendar[calendar.dt.date > as_of.date()].sort_values()
    if future.empty:
        raise ValueError("known trading calendar has no effective trading date")
    return future.iloc[0].date().isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_qlib_order_plan(
    *,
    manifest: dict[str, Any],
    result: dict[str, Any],
    dataset_provenance: dict[str, Any],
    order_plan_root: Path,
    tracking_uri: str,
) -> dict[str, Any]:
    target_payload = {
        "target_weights": dict(
            sorted(
                (
                    str(item["instrument"]).upper(),
                    float(item["weight"]),
                )
                for item in result["holdings"]
            )
        )
    }
    target_bytes = _canonical_bytes(target_payload)
    target_file_sha256 = _sha256_bytes(target_bytes)
    signal_at = manifest.get("signal_at")
    signal_date = str(manifest.get("signal_date") or result["as_of_date"])
    plan = {
        "format_version": "qlib-order-plan-v1",
        "produced_by": "qlib-workflow-recorder",
        "source_type": "strategy_version",
        "source_id": manifest["strategy_version_id"],
        "formal_backtest_id": manifest["formal_backtest_id"],
        "execution_contract_hash": manifest["config"]["execution_contract_hash"],
        "daily_dataset": manifest["dataset"],
        "signal_date": signal_date,
        "trade_date": result["effective_date"],
        "source_snapshot": {
            "id": dataset_provenance["dataset_identity_sha256"],
            "dataset_identity_sha256": dataset_provenance[
                "dataset_identity_sha256"
            ],
            "dataset_lineage_id": dataset_provenance["dataset_lineage_id"],
        },
        "target_weights_file_sha256": target_file_sha256,
        "target_weights_sha256": _sha256_bytes(target_bytes),
    }
    if signal_at is not None:
        plan["signal_at"] = str(signal_at)
    if manifest.get("execution_not_before") is not None:
        plan["execution_not_before"] = str(manifest["execution_not_before"])
    if isinstance(manifest.get("signal_dataset"), dict):
        plan["signal_snapshot"] = dict(manifest["signal_dataset"])
    run_id = str(manifest["order_plan_job_id"])
    with qlib_workflow_run(
        run_kind="simulation-order-plan",
        run_id=run_id,
        tracking_uri=tracking_uri,
        dataset_identity_sha256=dataset_provenance["dataset_identity_sha256"],
    ) as workflow:
        workflow.log_params(
            {
                "simulation_portfolio_id": manifest["simulation_portfolio_id"],
                "strategy_version_id": manifest["strategy_version_id"],
                "formal_backtest_id": manifest["formal_backtest_id"],
                "dataset": manifest["dataset"],
                "signal_date": signal_date,
                "signal_at": signal_at,
                "execution_contract_hash": manifest["config"][
                    "execution_contract_hash"
                ],
            }
        )
        plan["qlib_workflow"] = workflow.identity_dict()
        manifest_bytes = _canonical_bytes(plan)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        artifact = (order_plan_root / manifest_sha256).resolve()
        allowed_root = order_plan_root.resolve()
        try:
            artifact.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError("Qlib order-plan output path is unsafe") from exc
        artifact.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact / "manifest.json"
        target_path = artifact / "target_weights.json"
        for path, expected in (
            (manifest_path, manifest_bytes),
            (target_path, target_bytes),
        ):
            if path.exists() and path.read_bytes() != expected:
                raise ValueError(
                    "Qlib order-plan retry encountered different immutable content"
                )
            path.write_bytes(expected)
        workflow.log_metrics(
            {
                "target_count": len(target_payload["target_weights"]),
                "target_weight_sum": sum(target_payload["target_weights"].values()),
            }
        )
        workflow.save_artifacts(artifact)
    return {
        **result,
        "order_plan_manifest_sha256": manifest_sha256,
        "order_plan_artifact_path": str(artifact),
        "qlib_workflow": plan["qlib_workflow"],
    }


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
    parser.add_argument("--tracking-uri")
    parser.add_argument("--order-plan-root")
    parser.add_argument("--signal-provider-uri")
    args = parser.parse_args()
    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    provenance_path = Path(args.provider_uri) / "metadata" / "provenance.json"
    if not provenance_path.exists():
        raise ValueError("recommendation refresh requires dataset provenance metadata")
    dataset_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    require_daily_qlib_contract(dataset_provenance)
    as_of = pd.Timestamp(
        manifest.get("signal_at") or manifest["as_of_date"]
    ).tz_localize(None)

    import qlib
    from qlib.data import D

    config = manifest["config"]
    require_strategy_execution_contract(config)
    signal_frequency = str(config.get("signal_frequency") or "day").lower()
    signal_provider_uri = args.signal_provider_uri or args.provider_uri
    if signal_frequency != "day":
        if not args.signal_provider_uri:
            raise ValueError(
                "minute simulation order-plan requires a Qlib signal provider"
            )
        signal_provenance_path = (
            Path(signal_provider_uri) / "metadata" / "provenance.json"
        )
        if not signal_provenance_path.exists():
            raise ValueError("minute Qlib signal dataset has no provenance metadata")
        signal_provenance = json.loads(
            signal_provenance_path.read_text(encoding="utf-8")
        )
        require_minute_execution_contract(
            signal_provenance,
            frequency=signal_frequency,
            simulation_eligible=True,
        )
        expected_signal = dict(manifest.get("signal_dataset") or {})
        if Path(signal_provider_uri).name != str(expected_signal.get("name") or ""):
            raise ValueError(
                "minute Qlib signal provider name does not match the order-plan manifest"
            )
        for field in (
            "dataset_identity_sha256",
            "dataset_lineage_id",
            "source_lineage_id",
            "frequency",
        ):
            if str(signal_provenance.get(field) or "") != str(
                expected_signal.get(field) or ""
            ):
                raise ValueError(
                    "minute Qlib signal dataset does not match the order-plan manifest"
                )
    qlib.init(provider_uri=signal_provider_uri, region="cn")
    challenger = None
    if manifest["factors"]:
        challenger = compose_factor_scores(
            [
                (
                    _load(item["values_path"]),
                    float(item["weight"]),
                    int(item["direction"]),
                )
                for item in manifest["factors"]
            ]
        )
    baseline_definition = config.get("baseline_definition")
    if isinstance(baseline_definition, dict):
        expressions = [
            str(item["qlib_expression"])
            for item in baseline_definition.get("factors") or []
        ]
        baseline_values = D.features(
            D.instruments(manifest.get("universe") or "cn_all"),
            expressions,
            start_time=(as_of - pd.Timedelta(days=400)).isoformat(),
            end_time=as_of.isoformat(),
            freq=str(baseline_definition.get("frequency") or "day"),
        )
        _, _, baseline = normalize_qlib_baseline_values(
            baseline_values, baseline_definition
        )
        scores = combine_factor_sources(
            mode=str(config.get("factor_source_mode") or ""),
            baseline=baseline,
            challenger=challenger,
            challenger_weight=float(config.get("challenger_weight") or 0.0),
        )
    else:
        if str(config.get("factor_source_mode") or FACTOR_SOURCE_PROMOTED_ONLY) != (
            FACTOR_SOURCE_PROMOTED_ONLY
        ) or challenger is None:
            raise ValueError("recommendation source has no governed Qlib factor scores")
        scores = challenger
    dates = pd.to_datetime(scores.index.get_level_values("datetime")).tz_localize(None)
    scores.index = pd.MultiIndex.from_arrays(
        [dates, scores.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    if as_of not in set(dates):
        raise ValueError("factor artifacts do not contain the requested recommendation date")
    if signal_frequency != "day":
        qlib.init(
            provider_uri=args.provider_uri,
            region="cn",
            clear_mem_cache=True,
        )
    market_as_of = as_of.normalize()
    instruments = sorted(set(scores.index.get_level_values("instrument")))
    lookback = (as_of - pd.Timedelta(days=60)).date().isoformat()
    # $amount is CNY yuan under the v3 daily field contract.
    liquidity = D.features(
        instruments, ["$amount"], start_time=lookback, end_time=as_of.date().isoformat(), freq="day"
    )
    execution_metadata = D.features(
        instruments,
        ["$open", "$close", "Ref(Mean($amount, 20), 1)"],
        start_time=as_of.date().isoformat(),
        end_time=as_of.date().isoformat(),
        freq="day",
    )
    close_history = D.features(
        instruments,
        ["$close"],
        start_time=lookback,
        end_time=as_of.date().isoformat(),
        freq="day",
    )["$close"].unstack("instrument").sort_index()
    execution_metadata.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(execution_metadata.index.get_level_values("datetime")).tz_localize(None),
            execution_metadata.index.get_level_values("instrument").astype(str),
        ],
        names=["datetime", "instrument"],
    )
    point_metadata = execution_metadata.xs(market_as_of, level="datetime")
    metadata_root = Path(args.provider_uri) / "metadata"
    memberships = pd.read_parquet(metadata_root / "industry_memberships.parquet")
    if config.get("portfolio_construction") == "industry_neutral_qp":
        benchmark_frame = pd.read_parquet(metadata_root / "full_market_weights.parquet")
    else:
        benchmark_frame = pd.read_parquet(metadata_root / "benchmark_weights.parquet")
        benchmark_frame = benchmark_frame[
            benchmark_frame["benchmark"] == manifest["benchmark"]
        ].drop(columns=["benchmark"])
    style_fields = {
        "Log($total_mv)": "size",
        "1/$pb": "value",
        "($fund_quarter_revenue_yoy+$fund_op_profit_yoy)/2": "growth",
        "Std($close/Ref($close, 1)-1, 60)": "volatility",
    }
    styles_frame = D.features(
        instruments,
        list(style_fields),
        start_time=lookback,
        end_time=as_of.date().isoformat(),
        freq="day",
    ).rename(columns=style_fields).reset_index()
    eligibility_frame = pd.read_parquet(metadata_root / "eligibility_matrix.parquet")
    eligibility_evidence = eligibility_statistics(eligibility_frame)
    if config.get("require_regulatory_events") and not eligibility_evidence[
        "regulatory_data_available"
    ]:
        raise ValueError("strategy requires regulatory events but no reliable source is available")
    neutralize_baseline = config.get("portfolio_construction") in {
        "benchmark_relative_qp",
        "industry_neutral_qp",
    }
    governed = build_governed_signal(
        scores.loc[(slice(lookback, as_of), slice(None))],
        topk=int(config["topk"]),
        liquidity_amount=liquidity,
        industry_memberships=memberships,
        benchmark_weights=benchmark_frame,
        style_exposures=styles_frame,
        eligibility_matrix=eligibility_frame,
        max_industry_weight=float(config.get("max_industry_weight", 1.0)),
        max_industry_deviation=float(config.get("max_industry_deviation", 1.0)),
        min_average_daily_amount=float(config.get("min_average_daily_amount", 0.0)),
        liquidity_lookback_days=int(config.get("liquidity_lookback_days", 20)),
        neutralize_industry=neutralize_baseline,
        neutralize_style_columns=("size",) if neutralize_baseline else (),
    )
    signal = governed.xs(as_of, level="datetime")
    # Read-side availability guard (design draft 3.3): industry membership and
    # benchmark weights are only usable after the versioned conservative
    # publication lag, applied here through the shared registry policy.
    active = (
        filter_available("index_member_all", memberships, as_of)
        .sort_values("in_date")
        .drop_duplicates("instrument", keep="last")
    )
    industries = active.set_index(active["instrument"].astype(str))["industry"].astype(str)
    benchmark = _latest(filter_available("index_weight", benchmark_frame, as_of), as_of, "weight")
    styles = _latest_styles(styles_frame, as_of)
    benchmark_industries = industries.reindex(benchmark.index)
    if benchmark_industries.isna().any() or styles.reindex(benchmark.index).isna().any().any():
        raise ValueError("benchmark metadata is incomplete")
    risk_instruments = signal.index.astype(str).union(benchmark.index.astype(str))
    risk_returns = (
        close_history.reindex(columns=risk_instruments)
        .tail(61)
        .pct_change(fill_method=None)
        .dropna(how="any")
    )
    if len(risk_returns) < 60:
        raise ValueError("recommendation optimizer requires 60 complete return observations")
    cost_model = CostModelConfig.from_mapping(config)
    policy = PortfolioPolicy(PortfolioPolicyConfig.from_mapping(config), cost_model)
    previous = {
        item["instrument"]: item["weight"] for item in manifest.get("previous_holdings", [])
    }
    previous_snapshot = manifest.get("previous_snapshot") or {}
    rebalance_due = is_rebalance_due(
        as_of,
        previous_snapshot.get("as_of_date"),
        str(config.get("rebalance_frequency", "day")),
    )
    construction_notional = float(manifest["construction_notional"])
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
        return_covariance=estimate_covariance(risk_returns),
        prices=pd.to_numeric(point_metadata["$open"], errors="coerce"),
        current_prices=pd.to_numeric(point_metadata["$close"], errors="coerce"),
        portfolio_drawdown=float(manifest["portfolio_drawdown"]),
        daily_return=float(manifest["daily_return"]),
        # $amount is CNY yuan under the v3 daily field contract.
        average_daily_values=(
            pd.to_numeric(point_metadata["Ref(Mean($amount, 20), 1)"], errors="coerce")
        ),
        portfolio_value=construction_notional,
        risk_exposure=float(manifest.get("risk_exposure", 1.0)),
        allow_new_risk=bool(manifest.get("allow_new_risk", True)),
        rebalance_due=rebalance_due,
    )
    if signal_frequency == "day":
        effective_date = _next_known_trading_date(args.provider_uri, market_as_of)
    else:
        if manifest.get("signal_at") is None:
            raise ValueError("minute Qlib order-plan generation requires signal_at")
        effective_date = as_of.date().isoformat()
    changes = {item["instrument"]: item for item in decision.changes}

    result = {
        "status": "ok",
        "portfolio_id": manifest["portfolio_id"],
        "strategy_version_id": manifest["strategy_version_id"],
        "as_of_date": as_of.date().isoformat(),
        "effective_date": effective_date,
        "policy_version": decision.policy_version,
        "backtest_engine_version": QLIB_ENGINE_VERSION,
        "execution_contract_hash": config["execution_contract_hash"],
        "dataset": manifest["dataset"],
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "cost_model": decision.cost_model,
        "position_state": decision.position_state,
        "risk_summary": {
            "expected_turnover": decision.expected_turnover,
            "events": decision.risk_events,
            "execution_method": config.get("execution_method", "open"),
            "execution_days": int(config.get("execution_days", 1)),
            "execution_frequency": config.get("execution_frequency", "day"),
            "execution_contract_hash": config["execution_contract_hash"],
            "rebalance_frequency": config.get("rebalance_frequency", "day"),
            "rebalance_due": rebalance_due,
            "member_risk_state": dict(manifest.get("member_risk_state") or {}),
            "account_risk_state": dict(manifest.get("account_risk_state") or {}),
            "eligibility": eligibility_evidence,
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
            }
            for instrument, weight in decision.target_weights.items()
        ],
        "changes": decision.changes,
    }
    if manifest.get("artifact_kind") == "simulation_order_plan":
        if not args.tracking_uri or not args.order_plan_root:
            raise ValueError(
                "simulation order-plan generation requires tracking URI and artifact root"
            )
        result = _write_qlib_order_plan(
            manifest=manifest,
            result=result,
            dataset_provenance=dataset_provenance,
            order_plan_root=Path(args.order_plan_root),
            tracking_uri=args.tracking_uri,
        )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
