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

from quant_data.availability import filter_available
from quant_data.execution_contract import (
    require_daily_qlib_contract,
    require_minute_execution_contract,
    require_strategy_execution_contract,
)
from quant_platform.cost_model import CostScheduleBook
from quant_platform.eligibility import eligibility_statistics
from quant_platform.execution_algorithms import execution_time_slots
from quant_platform.formal_validation import (
    FORMAL_VALIDATION_CONTRACT_VERSION,
    run_ablation_suite,
    run_outer_walk_forward,
    run_signal_decay_suite,
)
from quant_platform.portfolio_policy import PortfolioPolicy, PortfolioPolicyConfig
from quant_platform.qlib_backtest import (
    QLIB_ENGINE_VERSION,
    run_formal_qlib_backtest,
    run_qlib_validation_suites,
)
from quant_platform.qlib_factor_baseline import (
    FACTOR_SOURCE_PROMOTED_ONLY,
    combine_factor_sources,
    normalize_qlib_baseline_values,
)
from quant_platform.qlib_policy_strategy import create_qlib_policy_strategy
from quant_platform.qlib_workflow import qlib_workflow_run
from quant_platform.risk_math import estimate_covariance
from quant_platform.statistical_validation import (
    deflated_sharpe_probability,
    holm_bonferroni,
    paired_moving_block_bootstrap,
)
from quant_platform.strategy_backtest import build_governed_signal, compose_factor_scores
from quant_platform.upstream_versions import upstream_runtime_identity

GOVERNED_STYLE_COLUMNS = ("size", "value", "growth", "volatility")
MAX_STYLE_CROSS_SECTION_MISSING_RATE = 0.05
STYLE_EXPOSURE_CONTRACT_VERSION = "standardized-neutral-imputation-v1"


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


def _recompute_qlib_baseline(
    data_api: Any,
    *,
    universe: str,
    definition: dict[str, Any],
    start_time: str,
    end_time: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    expressions = [str(item["qlib_expression"]) for item in definition.get("factors") or []]
    values = data_api.features(
        data_api.instruments(universe),
        expressions,
        start_time=start_time,
        end_time=end_time,
        freq=str(definition.get("frequency") or "day"),
    )
    return normalize_qlib_baseline_values(values, definition)


def _write_baseline_artifacts(
    output: Path,
    *,
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
    composite: pd.Series,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"raw": {}, "normalized": {}}
    for artifact_kind, frame in (("raw", raw), ("normalized", normalized)):
        root = output / "baseline" / artifact_kind
        root.mkdir(parents=True, exist_ok=True)
        for factor_id in frame.columns:
            path = root / f"{factor_id}.parquet"
            frame[factor_id].rename("value").to_frame().to_parquet(path, compression="zstd")
            artifacts[artifact_kind][str(factor_id)] = {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sha256": _sha256_file(path),
            }
    composite_path = output / "baseline" / "composite.parquet"
    composite_path.parent.mkdir(parents=True, exist_ok=True)
    composite.rename("score").to_frame().to_parquet(composite_path, compression="zstd")
    artifacts["composite"] = {
        "path": str(composite_path.relative_to(output)).replace("\\", "/"),
        "sha256": _sha256_file(composite_path),
    }
    return artifacts


def _qlib_instruments(provider_uri: str | Path) -> set[str]:
    root = Path(provider_uri) / "instruments"
    candidates = (root / "cn_all.txt", root / "liquid_all.txt", root / "all.txt")
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise ValueError("minute execution Qlib dataset has no instrument universe")
    return {
        line.split("\t", 1)[0].strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _eligible_strategy_instruments(
    scores: pd.Series, eligibility_matrix: pd.DataFrame
) -> list[str]:
    required = {"instrument", "eligible"}
    if not required.issubset(eligibility_matrix.columns):
        raise ValueError("point-in-time eligibility metadata is incomplete")
    score_instruments = set(scores.index.get_level_values("instrument").astype(str))
    eligible_instruments = set(
        eligibility_matrix.loc[
            eligibility_matrix["eligible"].fillna(False).astype(bool), "instrument"
        ].astype(str)
    )
    instruments = sorted(score_instruments & eligible_instruments)
    if not instruments:
        raise ValueError("factor scores have no point-in-time eligible instruments")
    return instruments


def _minute_warmup_window(
    provider_uri: str | Path, frequency: str, start_time: str, lookback_days: int
) -> tuple[str, str]:
    calendar_path = Path(provider_uri) / "calendars" / f"{frequency}.txt"
    try:
        timestamps = pd.to_datetime(
            [
                line.strip()
                for line in calendar_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
            errors="raise",
        )
    except (OSError, ValueError) as exc:
        raise ValueError("minute execution calendar is missing or invalid") from exc
    start_date = pd.Timestamp(start_time).date()
    prior_days = sorted({value.date() for value in timestamps if value.date() < start_date})
    if len(prior_days) < lookback_days:
        raise ValueError(
            f"VWAP execution requires {lookback_days} complete pre-backtest minute trading days"
        )
    selected = prior_days[-lookback_days:]
    return selected[0].isoformat(), selected[-1].isoformat()


def _historical_vwap_profile(
    *,
    instruments: list[str],
    start_time: str,
    end_time: str,
    frequency: str,
    slice_minutes: int,
    max_slices: int,
) -> list[dict[str, Any]]:
    from qlib.data import D

    volume = D.features(
        instruments,
        ["$volume"],
        start_time=start_time,
        end_time=end_time,
        freq=frequency,
    ).reset_index()
    if volume.empty or not {"datetime", "$volume"}.issubset(volume.columns):
        raise ValueError("VWAP warm-up window contains no minute volume evidence")
    volume["datetime"] = pd.to_datetime(volume["datetime"], errors="coerce")
    volume["volume"] = pd.to_numeric(volume["$volume"], errors="coerce")
    volume = volume.dropna(subset=["datetime", "volume"])
    volume = volume[volume["volume"] > 0]
    slots = execution_time_slots(
        trade_date=pd.Timestamp(start_time).date(),
        policy={"slice_minutes": slice_minutes, "max_slices": max_slices},
    )
    slot_names = [item.strftime("%H:%M") for item in slots]
    volume["time"] = volume["datetime"].dt.strftime("%H:%M")
    by_time = volume[volume["time"].isin(slot_names)].groupby("time")["volume"].mean()
    missing = [slot for slot in slot_names if slot not in by_time or not np.isfinite(by_time[slot])]
    if missing:
        raise ValueError(
            "VWAP warm-up evidence is missing configured execution slots: " + ", ".join(missing)
        )
    return [{"time": slot, "weight": float(by_time[slot])} for slot in slot_names]


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
    if result.index.has_duplicates:
        raise ValueError("point-in-time style exposures are duplicated")
    result = result.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_rates = result.isna().mean()
    systemic = missing_rates[missing_rates > MAX_STYLE_CROSS_SECTION_MISSING_RATE]
    if not systemic.empty:
        details = ", ".join(f"{column}={rate:.2%}" for column, rate in systemic.items())
        raise ValueError(
            "point-in-time standardized style exposure missing rate exceeds "
            f"{MAX_STYLE_CROSS_SECTION_MISSING_RATE:.0%}: {details}"
        )
    # The builder writes cross-sectionally standardized exposures. Zero is the
    # neutral exposure, so sparse missing descriptors are conservatively
    # imputed to neutral only after the per-date systemic-missing gate above.
    result = result.fillna(0.0)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("point-in-time style exposures are not finite")
    return result.astype(float)


def _load_governed_style_exposures(
    provider_uri: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(provider_uri) / "metadata" / "style_exposures.parquet"
    if not path.is_file():
        raise ValueError("constrained backtest requires standardized point-in-time style metadata")
    required = ["instrument", "datetime", *GOVERNED_STYLE_COLUMNS]
    try:
        frame = pd.read_parquet(path, columns=required)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "standardized style metadata is missing governed exposure columns"
        ) from exc
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    if frame[["instrument", "datetime"]].isna().any().any():
        raise ValueError("standardized style metadata contains invalid identity fields")
    frame["instrument"] = frame["instrument"].astype(str)
    if frame.duplicated(["datetime", "instrument"]).any():
        raise ValueError("standardized style metadata contains duplicate instrument dates")
    frame[list(GOVERNED_STYLE_COLUMNS)] = (
        frame[list(GOVERNED_STYLE_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    missing_counts = {column: int(frame[column].isna().sum()) for column in GOVERNED_STYLE_COLUMNS}
    return frame, {
        "contract_version": STYLE_EXPOSURE_CONTRACT_VERSION,
        "source": "qlib_builder_standardized_style_exposures",
        "path": str(path),
        "sha256": _sha256_file(path),
        "columns": list(GOVERNED_STYLE_COLUMNS),
        "rows": int(len(frame)),
        "missing_counts": missing_counts,
        "max_cross_section_missing_rate": MAX_STYLE_CROSS_SECTION_MISSING_RATE,
        "missing_imputation": "zero_standardized_neutral_exposure",
    }


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
    close_history: pd.DataFrame,
    *,
    open_field: str = "$open",
    close_field: str = "$close",
    intraday_prices: pd.DataFrame | None = None,
):
    membership = memberships.copy()
    membership["in_date"] = pd.to_datetime(membership["in_date"], errors="coerce")
    membership["out_date"] = pd.to_datetime(membership["out_date"], errors="coerce")
    close_matrix = close_history["$close"].unstack("instrument").sort_index()

    def provide(when: Any, instruments: pd.Index) -> dict[str, Any]:
        timestamp = pd.Timestamp(when).tz_localize(None)
        market_timestamp = (
            timestamp.normalize() - pd.Timedelta(nanoseconds=1)
            if timestamp != timestamp.normalize()
            else timestamp
        )
        # Read-side availability guard (design draft 3.3): membership intervals
        # and weight snapshots become usable only after the versioned
        # conservative publication lag from the shared registry policy.
        active = (
            filter_available("index_member_all", membership, market_timestamp)
            .sort_values("in_date")
            .drop_duplicates("instrument", keep="last")
        )
        industries = active.set_index(active["instrument"].astype(str))["industry"].astype(str)
        benchmark = _latest_cross_section(
            filter_available("index_weight", benchmark_weights, market_timestamp),
            market_timestamp,
            "weight",
        )
        style = _latest_style_cross_section(styles, market_timestamp)
        benchmark_industries = industries.reindex(benchmark.index)
        if benchmark_industries.isna().any():
            raise ValueError("benchmark constituents are missing point-in-time industries")
        risk_instruments = instruments.astype(str).union(benchmark.index.astype(str))
        history = close_matrix.loc[:market_timestamp].reindex(columns=risk_instruments).tail(61)
        returns = history.pct_change(fill_method=None).dropna(how="any")
        if len(returns) < 60:
            raise ValueError("optimizer requires 60 complete point-in-time return observations")
        return {
            "industries": industries.reindex(instruments.astype(str)),
            "benchmark_weights": benchmark,
            "benchmark_industry_weights": benchmark.groupby(benchmark_industries).sum(),
            "style_exposures": style,
            "benchmark_style_exposure": style.reindex(benchmark.index).mul(benchmark, axis=0).sum(),
            "return_covariance": estimate_covariance(returns),
            "prices": _qlib_cross_section(
                intraday_prices if intraday_prices is not None else execution_metadata,
                timestamp if intraday_prices is not None else market_timestamp,
                "$vwap" if intraday_prices is not None else open_field,
            ).reindex(instruments.astype(str)),
            "current_prices": _qlib_cross_section(
                intraday_prices if intraday_prices is not None else execution_metadata,
                timestamp if intraday_prices is not None else market_timestamp,
                "$close" if intraday_prices is not None else close_field,
            ).reindex(instruments.astype(str)),
            # $amount is CNY yuan under the v3 daily field contract.
            "average_daily_values": (
                _qlib_cross_section(
                    execution_metadata,
                    market_timestamp,
                    "Ref(Mean($amount, 20), 1)",
                ).reindex(instruments.astype(str))
            ),
        }

    return provide


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--execution-provider-uri")
    parser.add_argument("--execution-frequency")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    provider_provenance_path = Path(args.provider_uri) / "metadata" / "provenance.json"
    if not provider_provenance_path.exists():
        raise ValueError("formal Qlib backtest requires dataset provenance metadata")
    provider_provenance = json.loads(provider_provenance_path.read_text(encoding="utf-8"))
    require_daily_qlib_contract(provider_provenance)
    factor_value_hashes = {
        str(item["candidate_id"]): _sha256_file(item["values_path"]) for item in manifest["factors"]
    }
    factor_code_hashes = {
        str(item["candidate_id"]): item.get("code_sha256") for item in manifest["factors"]
    }

    challenger_entries = [
        (
            str(item["candidate_id"]),
            _load(item["values_path"]),
            float(item["weight"]),
            int(item["direction"]),
        )
        for item in manifest["factors"]
    ]
    challenger_factors = [
        (values, weight, direction)
        for _candidate_id, values, weight, direction in challenger_entries
    ]
    config = manifest["config"]
    strategy_contract = require_strategy_execution_contract(config)
    factor_source_mode = str(config.get("factor_source_mode") or FACTOR_SOURCE_PROMOTED_ONLY)
    signal_frequency = str(config.get("signal_frequency") or "day")
    execution_method = str(config.get("execution_method", "open"))
    minute_execution = execution_method in {"twap", "vwap", "next_bar"}
    minute_signal = signal_frequency != "day"
    if minute_signal and execution_method != "next_bar":
        raise ValueError("minute signals currently require the Qlib next_bar execution adapter")
    configured_execution_frequency = str(config.get("execution_frequency") or "day")
    if minute_execution and args.execution_frequency != configured_execution_frequency:
        raise ValueError(
            "execution dataset frequency does not match the immutable strategy contract"
        )
    if not minute_execution and configured_execution_frequency != "day":
        raise ValueError("daily open execution requires a day strategy execution frequency")
    execution_provenance: dict[str, Any] = {}
    if minute_execution:
        if not args.execution_provider_uri or not args.execution_frequency:
            raise ValueError("minute formal backtests require a minute Qlib dataset")
        execution_provenance_path = (
            Path(args.execution_provider_uri) / "metadata" / "provenance.json"
        )
        if not execution_provenance_path.exists():
            raise ValueError("minute execution Qlib dataset requires provenance metadata")
        execution_provenance = json.loads(execution_provenance_path.read_text(encoding="utf-8"))
        require_minute_execution_contract(execution_provenance, frequency=args.execution_frequency)
    import qlib
    from qlib.data import D

    qlib_runtime = upstream_runtime_identity("qlib")

    provider_uri: str | dict[str, str] = args.provider_uri
    if minute_execution:
        provider_uri = {
            "day": args.provider_uri,
            str(args.execution_frequency): str(args.execution_provider_uri),
        }
    qlib.init(provider_uri=provider_uri, region="cn")
    periods = manifest["periods"]
    challenger_scores = compose_factor_scores(challenger_factors) if challenger_factors else None
    baseline_artifacts: dict[str, Any] | None = None
    baseline_definition = config.get("baseline_definition")
    baseline_raw: pd.DataFrame | None = None
    baseline_normalized: pd.DataFrame | None = None
    baseline_scores: pd.Series | None = None
    if isinstance(baseline_definition, dict):
        baseline_raw, baseline_normalized, baseline_scores = _recompute_qlib_baseline(
            D,
            universe=str(manifest.get("universe") or "cn_all"),
            definition=baseline_definition,
            start_time=periods["start"],
            end_time=periods["end"],
        )
        baseline_artifacts = _write_baseline_artifacts(
            output,
            raw=baseline_raw,
            normalized=baseline_normalized,
            composite=baseline_scores,
        )
        manifest["baseline"] = {
            "definition": baseline_definition,
            "definition_sha256": config.get("baseline_definition_sha256"),
            "computed_by": "qlib.data.D.features",
            "artifacts": baseline_artifacts,
        }
    if factor_source_mode == FACTOR_SOURCE_PROMOTED_ONLY:
        if challenger_scores is None:
            raise ValueError("a promoted-only strategy has no challenger factor values")
        scores = challenger_scores
    else:
        if baseline_scores is None:
            raise ValueError("a core strategy has no governed Qlib baseline definition")
        scores = combine_factor_sources(
            mode=factor_source_mode,
            baseline=baseline_scores,
            challenger=challenger_scores,
            challenger_weight=float(config.get("challenger_weight") or 0.0),
        )
    eligibility_path = Path(args.provider_uri) / "metadata" / "eligibility_matrix.parquet"
    if not eligibility_path.is_file():
        raise ValueError("Qlib daily dataset has no point-in-time eligibility matrix")
    eligibility_matrix = pd.read_parquet(eligibility_path)
    instruments = _eligible_strategy_instruments(scores, eligibility_matrix)
    if minute_execution:
        available_instruments = _qlib_instruments(args.execution_provider_uri)
        missing_instruments = sorted(set(instruments) - available_instruments)
        if missing_instruments:
            preview = ", ".join(missing_instruments[:10])
            raise ValueError(
                "minute execution dataset is missing "
                f"{len(missing_instruments)} strategy instruments: " + preview
            )
    # $amount is CNY yuan under the v3 daily field contract.
    liquidity_amount = D.features(
        instruments,
        ["$amount"],
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    )
    open_field = "$open/$factor" if minute_execution else "$open"
    close_field = "$close/$factor" if minute_execution else "$close"
    execution_metadata = D.features(
        instruments,
        [open_field, close_field, "Ref(Mean($amount, 20), 1)"],
        start_time=periods["start"],
        end_time=periods["end"],
        freq="day",
    )
    covariance_start = (pd.Timestamp(periods["start"]) - pd.Timedelta(days=120)).date().isoformat()
    close_history = D.features(
        instruments,
        ["$close"],
        start_time=covariance_start,
        end_time=periods["end"],
        freq="day",
    )
    intraday_prices = (
        D.features(
            instruments,
            ["$vwap", "$close"],
            start_time=periods["start"],
            end_time=periods["end"],
            freq=str(args.execution_frequency),
        )
        if minute_signal
        else None
    )
    industry_path = Path(args.provider_uri) / "metadata" / "industry_memberships.parquet"
    industry_memberships = pd.read_parquet(industry_path) if industry_path.exists() else None
    industry_cap_enabled = float(manifest["config"].get("max_industry_weight", 1.0)) < 1.0
    if industry_cap_enabled and industry_memberships is None:
        raise ValueError("industry-constrained backtest requires point-in-time industry metadata")
    if config.get("portfolio_construction") == "industry_neutral_qp":
        target_weight_path = Path(args.provider_uri) / "metadata" / "full_market_weights.parquet"
        benchmark_weights = (
            pd.read_parquet(target_weight_path) if target_weight_path.exists() else None
        )
        target_weight_label = "full-market float-cap"
    else:
        target_weight_path = Path(args.provider_uri) / "metadata" / "benchmark_weights.parquet"
        benchmark_weights = (
            pd.read_parquet(target_weight_path) if target_weight_path.exists() else None
        )
        if benchmark_weights is not None:
            benchmark_weights = benchmark_weights[
                benchmark_weights["benchmark"] == manifest["benchmark"]
            ].drop(columns=["benchmark"])
        target_weight_label = "index benchmark"
    style_exposures, style_exposure_evidence = _load_governed_style_exposures(args.provider_uri)
    if benchmark_weights is None or benchmark_weights.empty:
        raise ValueError(f"constrained backtest requires historical {target_weight_label} weights")
    if style_exposures.empty:
        raise ValueError("index-enhancement backtest requires point-in-time style exposures")
    eligibility_evidence = eligibility_statistics(eligibility_matrix)
    if (
        config.get("require_regulatory_events")
        and not eligibility_evidence["regulatory_data_available"]
    ):
        raise ValueError("strategy requires regulatory events but no reliable source is available")

    def governed_for(
        scenario_config: dict[str, Any],
        signal_scores: pd.Series | None = None,
    ) -> pd.Series:
        neutralize_baseline = scenario_config.get("portfolio_construction") in {
            "benchmark_relative_qp",
            "industry_neutral_qp",
        }
        return build_governed_signal(
            scores if signal_scores is None else signal_scores,
            topk=int(scenario_config["topk"]),
            liquidity_amount=liquidity_amount,
            industry_memberships=industry_memberships,
            benchmark_weights=benchmark_weights,
            style_exposures=style_exposures,
            eligibility_matrix=eligibility_matrix,
            max_industry_weight=float(scenario_config.get("max_industry_weight", 1.0)),
            max_industry_deviation=float(scenario_config.get("max_industry_deviation", 1.0)),
            min_average_daily_amount=float(scenario_config.get("min_average_daily_amount", 0.0)),
            liquidity_lookback_days=int(scenario_config.get("liquidity_lookback_days", 20)),
            neutralize_industry=neutralize_baseline,
            neutralize_style_columns=("size",) if neutralize_baseline else (),
        )

    governed_signal = governed_for(config)
    cost_schedule = CostScheduleBook.from_mapping(config)
    # PortfolioPolicy only consumes date-independent broker assumptions (lot size
    # and participation), resolved here at the backtest start date.
    policy_costs = cost_schedule.as_of(pd.Timestamp(periods["start"]).date())
    policy = PortfolioPolicy(PortfolioPolicyConfig.from_mapping(config), policy_costs)
    metadata = _metadata_provider(
        industry_memberships,
        benchmark_weights,
        style_exposures,
        execution_metadata,
        close_history,
        open_field=open_field,
        close_field=close_field,
        intraday_prices=intraday_prices,
    )
    execution_policy: dict[str, Any] | None = None
    vwap_profile_evidence: dict[str, Any] | None = None
    if minute_execution:
        slice_minutes = int(config.get("execution_slice_minutes", 20))
        max_slices = int(config.get("max_execution_slices", 24))
        volume_profile = None
        if execution_method == "vwap":
            lookback_days = int(config.get("vwap_lookback_days", 20))
            warmup_start, warmup_end = _minute_warmup_window(
                args.execution_provider_uri,
                args.execution_frequency,
                periods["start"],
                lookback_days,
            )
            volume_profile = _historical_vwap_profile(
                instruments=instruments,
                start_time=warmup_start,
                end_time=warmup_end,
                frequency=args.execution_frequency,
                slice_minutes=slice_minutes,
                max_slices=max_slices,
            )
            vwap_profile_evidence = {
                "start": warmup_start,
                "end": warmup_end,
                "trading_days": lookback_days,
                "profile_sha256": _canonical_sha256(volume_profile),
                "future_data_used": False,
            }
        execution_policy = {
            "execution_algorithm": execution_method,
            "slice_minutes": slice_minutes,
            "max_slices": max_slices,
            "max_participation": policy_costs.max_volume_participation,
            "volume_profile": volume_profile,
        }

    def run(
        start: str,
        end: str,
        costs: CostScheduleBook,
        account: float | None = None,
        scenario_config: dict[str, Any] | None = None,
        signal_scores: pd.Series | None = None,
    ):
        effective_config = {**config, **(scenario_config or {})}
        scenario_policy = PortfolioPolicy(
            PortfolioPolicyConfig.from_mapping(effective_config),
            costs.as_of(pd.Timestamp(start).date()),
        )
        strategy = create_qlib_policy_strategy(
            signal=(
                governed_for(effective_config, signal_scores)
                if scenario_config or signal_scores is not None
                else governed_signal
            ),
            policy=scenario_policy,
            metadata_provider=metadata,
        )
        return run_formal_qlib_backtest(
            strategy=strategy,
            start_time=start,
            end_time=end,
            account=float(account if account is not None else config["capacity_notional"]),
            benchmark=manifest["benchmark"],
            cost_schedule=costs,
            execution_method=execution_method,
            signal_frequency=signal_frequency,
            execution_frequency=args.execution_frequency if minute_execution else None,
            execution_policy=execution_policy,
            instruments=instruments,
            annual_minimum_acceptable_return=float(
                effective_config.get("annual_minimum_acceptable_return", 0.0)
            ),
        )

    formal = run(periods["start"], periods["end"], cost_schedule)

    def write_robustness_artifacts(name: str, result: Any) -> dict[str, Any]:
        target = output / "robustness" / name
        target.mkdir(parents=True, exist_ok=True)
        report_path = target / "daily_report.parquet"
        fills_path = target / "fills.parquet"
        metrics_path = target / "metrics.json"
        result.report.to_parquet(report_path, compression="zstd")
        pd.DataFrame(
            result.fills,
            columns=[
                "instrument",
                "date",
                "side",
                "requested_amount",
                "amount",
                "capacity_fill_ratio",
                "trade_price",
                "trade_value",
                "cost",
            ],
        ).to_parquet(fills_path, index=False, compression="zstd")
        metrics_path.write_text(
            json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            key: {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sha256": _sha256_file(path),
            }
            for key, path in {
                "daily_report": report_path,
                "fills": fills_path,
                "metrics": metrics_path,
            }.items()
        }

    validation = run_qlib_validation_suites(
        runner=run,
        full_result=formal,
        start_time=periods["start"],
        end_time=periods["end"],
        cost_schedule=cost_schedule,
        config=config,
        capacity_runner=lambda notional: (
            formal
            if abs(notional - float(config["capacity_notional"])) < 1e-6
            else run(periods["start"], periods["end"], cost_schedule, notional)
        ),
        robustness_runner=lambda overrides, costs: run(
            periods["start"], periods["end"], costs, scenario_config=overrides
        ),
        robustness_artifact_writer=write_robustness_artifacts,
    )
    qlib_report = formal.report
    qlib_positions = formal.positions
    net_daily_returns = pd.to_numeric(qlib_report["return"], errors="coerce") - pd.to_numeric(
        qlib_report.get("cost", 0.0), errors="coerce"
    )
    strategy_trial_count = int(manifest.get("strategy_trial_count") or 1)

    def write_formal_run_artifacts(category: str, name: str, result: Any) -> dict[str, Any]:
        target = output / "formal-validation" / category / name
        target.mkdir(parents=True, exist_ok=True)
        report_path = target / "daily_report.parquet"
        fills_path = target / "fills.parquet"
        metrics_path = target / "metrics.json"
        result.report.to_parquet(report_path, compression="zstd")
        pd.DataFrame(result.fills).to_parquet(fills_path, index=False, compression="zstd")
        metrics_path.write_text(
            json.dumps(result.metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            key: {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "sha256": _sha256_file(path),
            }
            for key, path in {
                "daily_report": report_path,
                "fills": fills_path,
                "metrics": metrics_path,
            }.items()
        }

    baseline_weights = {
        str(item["id"]): float(item["weight"])
        for item in (
            baseline_definition.get("factors") or []
            if isinstance(baseline_definition, dict)
            else []
        )
    }
    ablation_components = [
        *(f"baseline:{factor_id}" for factor_id in sorted(baseline_weights)),
        *(
            f"challenger:{candidate_id}"
            for candidate_id, _values, _weight, _direction in challenger_entries
        ),
    ]

    def ablated_scores(component: str) -> pd.Series | None:
        baseline_variant = baseline_scores
        challenger_variant = challenger_scores
        if component.startswith("baseline:"):
            removed = component.split(":", 1)[1]
            remaining = {
                factor_id: weight
                for factor_id, weight in baseline_weights.items()
                if factor_id != removed
            }
            baseline_variant = (
                baseline_normalized.mul(pd.Series(remaining), axis=1).sum(axis=1).rename("score")
                if remaining and baseline_normalized is not None
                else None
            )
        elif component.startswith("challenger:"):
            removed = component.split(":", 1)[1]
            remaining = [
                (values, weight, direction)
                for candidate_id, values, weight, direction in challenger_entries
                if candidate_id != removed
            ]
            challenger_variant = compose_factor_scores(remaining) if remaining else None
        else:
            raise ValueError(f"unknown ablation component: {component}")
        if factor_source_mode == FACTOR_SOURCE_PROMOTED_ONLY:
            return challenger_variant
        if baseline_variant is None:
            return None
        if challenger_variant is None:
            return baseline_variant
        return combine_factor_sources(
            mode=factor_source_mode,
            baseline=baseline_variant,
            challenger=challenger_variant,
            challenger_weight=float(config.get("challenger_weight") or 0.0),
        )

    def run_ablation(component: str) -> dict[str, Any]:
        signal = ablated_scores(component)
        if signal is None:
            # Removing the final alpha component leaves the declared simple
            # benchmark/no-alpha baseline, not an arbitrary Top-K tie break.
            return {
                "annualized_excess_return": 0.0,
                "baseline_state": "no_alpha_component",
                "artifacts": {},
            }
        result = run(
            periods["start"],
            periods["end"],
            cost_schedule,
            signal_scores=signal,
        )
        return {
            **result.metrics,
            "artifacts": write_formal_run_artifacts(
                "ablation", component.replace(":", "-"), result
            ),
        }

    ablation = run_ablation_suite(
        component_ids=ablation_components,
        full_metrics=formal.metrics,
        runner=run_ablation,
        metric="annualized_excess_return",
        minimum_increment=float(config.get("min_component_increment", 0.0)),
    )

    def delayed_scores(delay: int) -> pd.Series:
        if delay == 0:
            return scores
        shifted = (
            scores.groupby(level="instrument", group_keys=False)
            .shift(delay)
            .dropna()
            .rename("score")
        )
        if shifted.empty:
            raise ValueError("signal decay delay leaves no executable observations")
        return shifted

    def run_delay(delay: int) -> dict[str, Any]:
        if delay == 0:
            return {**formal.metrics, "artifacts": {}}
        result = run(
            periods["start"],
            periods["end"],
            cost_schedule,
            signal_scores=delayed_scores(delay),
        )
        return {
            **result.metrics,
            "artifacts": write_formal_run_artifacts("signal-decay", f"delay-{delay}", result),
        }

    signal_decay = run_signal_decay_suite(
        delays=config.get("signal_decay_delays", [0, 1, 2, 3]),
        runner=run_delay,
        metric="annualized_excess_return",
        minimum_retention=float(config.get("minimum_signal_retention", 0.60)),
    )

    if strategy_trial_count == 1:
        outer_walk_forward = run_outer_walk_forward(
            dates=pd.DatetimeIndex(qlib_report.index).tz_localize(None),
            candidate_ids=["frozen-strategy"],
            inner_runner=lambda _candidate, fold: (
                run(
                    fold.validation_start,
                    fold.validation_end,
                    cost_schedule,
                ).metrics
            ),
            test_runner=lambda _candidate, fold: (
                run(
                    fold.test_start,
                    fold.test_end,
                    cost_schedule,
                ).metrics
            ),
            selection_metric="annualized_excess_return",
            train_days=int(config.get("outer_train_days", 252)),
            validation_days=int(config.get("outer_validation_days", 42)),
            test_days=int(config.get("outer_test_days", 42)),
            purge_days=int(config.get("outer_purge_days", 5)),
            embargo_days=int(config.get("outer_embargo_days", 5)),
        )
        outer_walk_forward["candidate_coverage"] = {
            "required_group_trials": 1,
            "provided_candidates": 1,
            "scope": "frozen_strategy_no_search",
        }
    else:
        outer_walk_forward = {
            "status": "blocked_missing_group_candidate_artifacts",
            "contract_version": FORMAL_VALIDATION_CONTRACT_VERSION,
            "candidate_coverage": {
                "required_group_trials": strategy_trial_count,
                "provided_candidates": 1,
                "scope": "economic_hypothesis_group",
            },
        }

    paired = pd.concat(
        [
            net_daily_returns.rename("candidate"),
            pd.to_numeric(qlib_report["bench"], errors="coerce").rename("baseline"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    paired_bootstrap = paired_moving_block_bootstrap(
        paired["candidate"],
        paired["baseline"],
        block_size=int(config.get("bootstrap_block_days", 20)),
        samples=int(config.get("bootstrap_samples", 2000)),
        seed=int(config.get("validation_seed", 0)),
    )
    if strategy_trial_count == 1:
        multiple_testing = {
            "status": "not_applicable_single_trial",
            "trial_count": 1,
            "holm_adjusted_p_values": holm_bonferroni([paired_bootstrap["one_sided_p_value"]]),
            "pbo": {
                "status": "not_applicable_single_trial",
                "pbo": None,
            },
        }
    else:
        multiple_testing = {
            "status": "blocked_missing_group_candidate_artifacts",
            "trial_count": strategy_trial_count,
            "holm_adjusted_p_values": None,
            "pbo": {
                "status": "blocked_missing_group_candidate_artifacts",
                "pbo": None,
            },
        }
    formal_validation = {
        "contract_version": FORMAL_VALIDATION_CONTRACT_VERSION,
        "status": (
            "passed"
            if outer_walk_forward.get("status") == "completed"
            and ablation["status"] == "passed"
            and signal_decay["maximum_supported_delay_bars"] is not None
            and paired_bootstrap["confidence_interval_95"][0] > 0
            and multiple_testing["status"] == "not_applicable_single_trial"
            else "failed"
        ),
        "outer_walk_forward": outer_walk_forward,
        "ablation": ablation,
        "signal_decay": signal_decay,
        "paired_block_bootstrap": paired_bootstrap,
        "multiple_testing": multiple_testing,
    }
    deflated_sharpe = deflated_sharpe_probability(
        net_daily_returns,
        trials=strategy_trial_count,
    )
    metrics = {
        **formal.metrics,
        "policy_version": policy.version,
        "cost_model": cost_schedule.to_dict(),
        "cash_yield": {
            "annual_rate": float(config.get("annual_cash_yield_rate", 0.0)),
            "source": str(config.get("cash_yield_source") or "none_zero_yield"),
            "accounting": "idle cash earns zero without a governed cash instrument",
        },
        "eligibility": eligibility_evidence,
        "deflated_sharpe": deflated_sharpe,
        "deflated_sharpe_probability": deflated_sharpe["probability"],
        "formal_validation": formal_validation,
        "formal_validation_passed": formal_validation["status"] == "passed",
        "execution_model": {
            "method": execution_method,
            "days": int(config.get("execution_days", 1)),
            "price_assumption": (
                "next eligible minute bar vwap"
                if execution_method == "next_bar"
                else ("minute bar vwap fills" if minute_execution else "next-day open")
            ),
            "signal_frequency": signal_frequency,
            "frequency": args.execution_frequency if minute_execution else "day",
            "dataset": (
                manifest.get("execution_dataset") if minute_execution else manifest["dataset"]
            ),
            "contract_version": (
                execution_provenance.get("execution_contract_version") if minute_execution else None
            ),
            "slice_minutes": execution_policy.get("slice_minutes") if execution_policy else None,
            "max_slices": execution_policy.get("max_slices") if execution_policy else None,
            "vwap_profile": vwap_profile_evidence,
            "strategy_contract": strategy_contract,
            "strategy_contract_hash": config["execution_contract_hash"],
        },
        "robustness": validation["robustness"],
        "robustness_passed": validation["robustness"]["passed"],
        "robustness_pass_rate": validation["robustness"]["pass_rate"],
        "component_cost_stress": validation["component_cost_stress"],
        "component_cost_stress_passed": validation["component_cost_stress"]["passed"],
        "component_cost_stress_pass_rate": validation["component_cost_stress"]["pass_rate"],
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
            "field_contract_version": provider_provenance.get("field_contract_version"),
            "source_volume_unit": provider_provenance.get("source_volume_unit"),
            "qlib_volume_unit": provider_provenance.get("qlib_volume_unit"),
            "source_hand_size": provider_provenance.get("source_hand_size"),
            "lineage_verified": provider_provenance.get("lineage_verified"),
            "source_lineage_id": provider_provenance.get("source_lineage_id"),
            "style_exposure_contract": style_exposure_evidence,
            "execution_dataset_identity_sha256": execution_provenance.get(
                "dataset_identity_sha256"
            ),
            "execution_snapshot_manifest_sha256": execution_provenance.get(
                "snapshot_manifest_sha256"
            ),
            "execution_qlib_builder_sha256": execution_provenance.get("qlib_builder_sha256"),
            "execution_contract_version": execution_provenance.get("execution_contract_version"),
            "execution_fields": execution_provenance.get("fields"),
            "execution_source_datasets": execution_provenance.get("source_datasets"),
            "execution_source_unit_contracts": execution_provenance.get("source_unit_contracts"),
            "execution_source_lineage_id": execution_provenance.get("source_lineage_id"),
            "execution_lineage_verified": execution_provenance.get("lineage_verified"),
            "strategy_config_sha256": _canonical_sha256(config),
            "execution_manifest_sha256": None,
            "factor_values_sha256": factor_value_hashes,
            "factor_code_sha256": factor_code_hashes,
            "factor_source_mode": factor_source_mode,
            "challenger_weight": float(config.get("challenger_weight") or 0.0),
            "baseline_definition_sha256": config.get("baseline_definition_sha256"),
            "baseline_qlib_expressions": (
                {
                    str(item["id"]): str(item["qlib_expression"])
                    for item in baseline_definition.get("factors") or []
                }
                if isinstance(baseline_definition, dict)
                else None
            ),
            "baseline_preprocessing": (
                baseline_definition.get("preprocessing")
                if isinstance(baseline_definition, dict)
                else None
            ),
            "baseline_raw_values_sha256": (
                {
                    factor_id: entry["sha256"]
                    for factor_id, entry in baseline_artifacts["raw"].items()
                }
                if baseline_artifacts
                else None
            ),
            "baseline_normalized_values_sha256": (
                {
                    factor_id: entry["sha256"]
                    for factor_id, entry in baseline_artifacts["normalized"].items()
                }
                if baseline_artifacts
                else None
            ),
            "baseline_composite_values_sha256": (
                baseline_artifacts["composite"]["sha256"] if baseline_artifacts else None
            ),
            "qlib_version": qlib_runtime["version"],
            "qlib_commit": qlib_runtime["commit"],
            "backtest_engine_version": QLIB_ENGINE_VERSION,
            "policy_version": policy.version,
        },
    }
    qlib_report.reset_index().to_parquet(output / "daily_returns.parquet", index=False)
    governed_signal.to_frame().to_parquet(output / "governed_signal.parquet")
    qlib_report.to_parquet(output / "qlib_portfolio_report.parquet")
    pd.to_pickle(qlib_positions, output / "qlib_positions.pkl")
    pd.DataFrame(
        formal.fills,
        columns=[
            "instrument",
            "date",
            "side",
            "requested_amount",
            "amount",
            "capacity_fill_ratio",
            "trade_price",
            "trade_value",
            "cost",
        ],
    ).to_parquet(output / "execution_fills.parquet", index=False)
    (output / "execution_model.json").write_text(
        json.dumps(metrics["execution_model"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    (output / "formal_validation.json").write_text(
        json.dumps(metrics["formal_validation"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Dataset descriptors consumed by the promotion chain: after the formal
    # hard gate approves the version, the isolated paper simulation account is
    # created from these verbatim dataset provenance records (design 6.11).
    (output / "datasets.json").write_text(
        json.dumps(
            {
                "daily": {"name": manifest["dataset"], "provenance": provider_provenance},
                "execution": (
                    {
                        "name": manifest["execution_dataset"],
                        "provenance": execution_provenance,
                    }
                    if minute_execution
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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
            "execution_fills": str(output / "execution_fills.parquet"),
            "execution_model": str(output / "execution_model.json"),
            "robustness": str(output / "robustness.json"),
            "rolling": str(output / "rolling.json"),
            "event_stress": str(output / "event_stress.json"),
            "capacity_curve": str(output / "capacity_curve.json"),
            "formal_validation": str(output / "formal_validation.json"),
        },
    }
    workflow_run_id = str(
        manifest.get("backtest_id")
        or (
            f"{manifest['strategy_version_id']}-"
            f"{_canonical_sha256({'periods': periods, 'config': config})[:16]}"
        )
    )
    with qlib_workflow_run(
        run_kind="formal-backtest",
        run_id=workflow_run_id,
        tracking_uri=args.tracking_uri,
        dataset_identity_sha256=provider_provenance.get("dataset_identity_sha256"),
    ) as workflow:
        workflow.log_params(
            {
                "backtest_id": manifest.get("backtest_id") or workflow_run_id,
                "strategy_version_id": manifest["strategy_version_id"],
                "dataset": manifest["dataset"],
                "execution_dataset": manifest.get("execution_dataset"),
                "benchmark": manifest["benchmark"],
                "start": periods["start"],
                "end": periods["end"],
                "execution_method": execution_method,
                "execution_frequency": (args.execution_frequency if minute_execution else "day"),
                "strategy_config_sha256": metrics["provenance"]["strategy_config_sha256"],
            }
        )
        workflow.log_metrics(metrics)
        recorder_identity = workflow.identity_dict()
        manifest["qlib_workflow"] = recorder_identity
        manifest["factor_source_mode"] = factor_source_mode
        manifest["challenger_weight"] = float(config.get("challenger_weight") or 0.0)
        Path(args.manifest).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metrics["provenance"]["execution_manifest_sha256"] = _sha256_file(args.manifest)
        metrics["provenance"]["qlib_workflow"] = recorder_identity
        result["qlib_workflow"] = recorder_identity
        (output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        workflow.save_artifacts(output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
