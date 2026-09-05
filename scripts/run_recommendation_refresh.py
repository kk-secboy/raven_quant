#!/usr/bin/env python3
"""Generate a recommendation snapshot from the same PortfolioPolicy used by Qlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.availability import filter_available
from quant_data.execution_contract import (
    require_daily_qlib_contract,
    require_minute_execution_contract,
    require_strategy_execution_contract,
)
from quant_data.qlib_builder import verify_qlib_output_manifest
from quant_platform.cost_model import CostModelConfig
from quant_platform.eligibility import (
    eligibility_statistics,
    project_point_in_time_risk_states,
)
from quant_platform.horizon_review import (
    build_financial_review_scope,
    validate_financial_review_scope,
    validate_financial_review_trigger,
)
from quant_platform.investor_profile import (
    investor_profile_permission_map,
    validate_investor_profile_binding,
)
from quant_platform.paper_policy_state import seal_paper_policy_state
from quant_platform.portfolio_policy import (
    PortfolioPolicy,
    PortfolioPolicyConfig,
    is_rebalance_due,
    rebalance_period_key,
)
from quant_platform.promotion import build_horizon_review_evidence
from quant_platform.qlib_backtest import QLIB_ENGINE_VERSION
from quant_platform.qlib_factor_baseline import (
    FACTOR_SOURCE_PROMOTED_ONLY,
    combine_factor_sources,
    normalize_qlib_baseline_values,
)
from quant_platform.qlib_workflow import qlib_workflow_run
from quant_platform.risk_math import estimate_covariance
from quant_platform.strategy_backtest import (
    build_governed_signal,
    compose_factor_scores,
    governed_score_neutralization,
)
from quant_platform.strategy_rule_runtime import (
    apply_strategy_rule_alpha_weights,
    build_portfolio_policy_runtime_metadata,
    build_strategy_rule_runtime_metadata,
    load_market_trend_close_history,
    policy_style_cross_sections,
    required_rule_history_sessions,
)
from quant_platform.strategy_rule_runtime import (
    load_governed_style_exposures as _load_governed_style_exposures,
)

_COVARIANCE_REQUIRED_PORTFOLIO_CONSTRUCTIONS = frozenset(
    {"benchmark_relative_qp", "industry_neutral_qp"}
)


def _apply_investor_profile_permissions(
    risk_projection: pd.DataFrame,
    binding: dict[str, Any],
    *,
    on_date: date,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Block exposure increases outside the paper account's frozen permissions."""

    profile = validate_investor_profile_binding(binding)
    projection = risk_projection.copy()
    permissions = investor_profile_permission_map(
        profile,
        set(projection.index.astype(str)),
        on_date=on_date,
    )
    for instrument, evidence in permissions.items():
        if evidence["allowed"] is True:
            continue
        current_state = str(projection.loc[instrument, "risk_state"])
        if current_state not in {"reduce", "exit"}:
            projection.loc[instrument, "risk_state"] = "restricted"
        projection.loc[instrument, "allow_new_risk"] = False
        raw_reasons = projection.loc[instrument, "risk_reasons"]
        try:
            reasons = json.loads(str(raw_reasons))
        except json.JSONDecodeError:
            reasons = []
        if not isinstance(reasons, list):
            reasons = []
        reason = str(evidence.get("reason") or "investor_permission_denied")
        if reason not in reasons:
            reasons.append(reason)
        projection.loc[instrument, "risk_reasons"] = json.dumps(
            reasons,
            ensure_ascii=False,
            sort_keys=True,
        )
    return projection, permissions


def _load(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported factor artifact: {source}")


def _portfolio_return_covariance(
    strategy_config: dict[str, Any],
    close_history: pd.DataFrame,
    risk_instruments: pd.Index,
) -> pd.DataFrame | None:
    """Build optimizer risk only for policies that actually consume it."""

    if (
        str(strategy_config.get("portfolio_construction") or "")
        not in _COVARIANCE_REQUIRED_PORTFOLIO_CONSTRUCTIONS
    ):
        return None
    risk_returns = (
        close_history.reindex(columns=risk_instruments)
        .tail(61)
        .pct_change(fill_method=None)
        .dropna(how="any")
    )
    if len(risk_returns) < 60:
        raise ValueError("recommendation optimizer requires 60 complete return observations")
    return estimate_covariance(risk_returns)


def _market_trend_close_history(
    data_api: Any,
    *,
    strategy_config: dict[str, Any],
    start_time: str,
    end_time: str,
) -> pd.DataFrame | None:
    return load_market_trend_close_history(
        data_api,
        config=strategy_config,
        start_time=start_time,
        end_time=end_time,
    )


def _recommendation_rule_runtime_metadata(
    data_api: Any,
    *,
    config: dict[str, Any],
    instruments: pd.Index,
    close_history: pd.DataFrame,
    value_exposures: pd.Series | None,
    start_time: str,
    end_time: str,
) -> dict[str, Any]:
    """Build recommendation rules with the same bound benchmark as backtests."""

    benchmark_close_history = _market_trend_close_history(
        data_api,
        strategy_config=config,
        start_time=start_time,
        end_time=end_time,
    )
    return build_strategy_rule_runtime_metadata(
        config,
        instruments=instruments,
        close_history=close_history,
        benchmark_close_history=benchmark_close_history,
        value_exposures=value_exposures,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_prediction_scores(
    artifact: dict[str, Any], *, expected_dataset_identity: str
) -> pd.Series:
    if not isinstance(artifact, dict):
        raise ValueError(
            "model-prediction strategy requires an active governed ModelArtifact"
        )
    path = Path(str(artifact.get("artifact_path") or ""))
    recorded = str(artifact.get("predictions_sha256") or "").lower()
    if (
        not path.is_file()
        or len(recorded) != 64
        or _sha256_file(path) != recorded
        or str(artifact.get("artifact_sha256") or "").lower() != recorded
        or str(artifact.get("dataset_identity_sha256") or "")
        != expected_dataset_identity
    ):
        raise ValueError("active ModelArtifact prediction table failed immutable verification")
    values = _load(str(path))
    if isinstance(values, pd.Series):
        scores = values
    elif values.shape[1] == 1:
        scores = values.iloc[:, 0]
    elif "score" in values:
        scores = values["score"]
    else:
        raise ValueError("ModelArtifact prediction table must contain one score column")
    if not isinstance(scores.index, pd.MultiIndex) or scores.index.nlevels != 2:
        raise ValueError("ModelArtifact predictions require datetime/instrument indexing")
    scores = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if scores.index.has_duplicates or scores.dropna().empty:
        raise ValueError("ModelArtifact predictions are duplicate or empty")
    names = [str(name or "").lower() for name in scores.index.names]
    if names == ["instrument", "datetime"]:
        scores = scores.reorder_levels(["datetime", "instrument"])
    elif names != ["datetime", "instrument"]:
        raise ValueError(
            "ModelArtifact prediction index names must be datetime and instrument"
        )
    scores.index = scores.index.set_names(["datetime", "instrument"])
    return scores.rename("score").sort_index()


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


def _financial_review_for_signal(
    manifest: dict[str, Any], signal_date: date
) -> dict[str, Any] | None:
    raw = manifest.get("financial_review_trigger")
    if raw is None:
        return None
    horizon = str((manifest.get("config") or {}).get("horizon_profile") or "")
    if horizon != "long_1_3y" or not isinstance(raw, dict):
        raise ValueError("financial review trigger is only valid for a long strategy")
    return validate_financial_review_trigger(
        raw,
        expected_signal_date=signal_date,
        expected_dataset_identity_sha256=str(
            manifest.get("dataset_identity_sha256") or ""
        ),
    )


def _review_completed_at(signal_date: date) -> datetime:
    return datetime.combine(
        signal_date,
        time(15, 0),
        tzinfo=ZoneInfo("Asia/Shanghai"),
    ).astimezone(UTC)


def _rebalance_due_for_financial_review(
    *,
    scheduled_rebalance_due: bool,
    financial_review_scope: dict[str, Any] | None,
) -> bool:
    return bool(
        scheduled_rebalance_due
        or (
            financial_review_scope is not None
            and financial_review_scope.get("review_mode") == "decision_review"
        )
    )


def _rebalance_instruments_for_financial_review(
    *,
    scheduled_rebalance_due: bool,
    financial_review_scope: dict[str, Any] | None,
) -> list[str] | None:
    """Return the local decision sleeve for an off-cadence PIT review."""

    if scheduled_rebalance_due or financial_review_scope is None:
        return None
    if financial_review_scope.get("review_mode") != "decision_review":
        return None
    reviewed = financial_review_scope.get("reviewed_instruments")
    if not isinstance(reviewed, list) or not reviewed:
        raise ValueError("financial decision review requires governed instruments")
    return [str(instrument) for instrument in reviewed]


def _horizon_review_for_order_plan(
    *,
    manifest: dict[str, Any],
    result: dict[str, Any],
    dataset_provenance: dict[str, Any],
) -> dict[str, Any] | None:
    config = dict(manifest.get("config") or {})
    horizon = str(config.get("horizon_profile") or "")
    if horizon not in {"swing_1_6m", "long_1_3y"}:
        return None
    signal_date = date.fromisoformat(
        str(manifest.get("signal_date") or result.get("as_of_date") or "")
    )
    dataset_identity = str(dataset_provenance.get("dataset_identity_sha256") or "")
    if dataset_identity != str(manifest.get("dataset_identity_sha256") or ""):
        raise ValueError("horizon review changed its immutable dataset binding")
    rebalance_due = bool((result.get("risk_summary") or {}).get("rebalance_due"))
    financial = _financial_review_for_signal(manifest, signal_date)
    financial_scope = None
    if financial is not None:
        raw_scope = (result.get("risk_summary") or {}).get("financial_review_scope")
        if raw_scope is not None:
            financial_scope = validate_financial_review_scope(
                raw_scope,
                trigger=financial,
            )
    completed_at = _review_completed_at(signal_date)
    strategy_version_id = str(manifest.get("strategy_version_id") or "")
    if financial is not None and financial_scope is not None and (
        financial_scope["review_mode"] == "decision_review"
    ):
        if not rebalance_due:
            raise ValueError(
                "financial review evidence requires an executed rebalance decision"
            )
        source_event_sha256 = str(financial["source_event_sha256"])
        event_id = (
            f"{strategy_version_id}:financial:{financial['report_period']}:"
            f"{signal_date.isoformat()}:{source_event_sha256}"
        )
        return build_horizon_review_evidence(
            event_id=event_id,
            review_type="financial_report_review",
            horizon_profile=horizon,
            completed_at=completed_at,
            strategy_version_id=strategy_version_id,
            signal_date=signal_date,
            dataset_identity_sha256=dataset_identity,
            trigger_source=str(financial["trigger_source"]),
            trigger_effective_date=date.fromisoformat(
                str(financial["trigger_effective_date"])
            ),
            report_period=str(financial["report_period"]),
            report_periods=[
                str(item)
                for item in (
                    financial.get("report_periods") or [financial["report_period"]]
                )
            ],
            announcement_date=date.fromisoformat(str(financial["announcement_date"])),
            previous_signal_date=date.fromisoformat(
                str(financial["previous_signal_date"])
            ),
            source_datasets=[str(item) for item in financial["source_datasets"]],
            source_event_count=int(financial["source_event_count"]),
            source_event_sha256=source_event_sha256,
            reviewed_instruments=[
                str(item) for item in financial_scope["reviewed_instruments"]
            ],
            review_scope_sha256=str(financial_scope["scope_sha256"]),
        )
    if not rebalance_due:
        return None
    frequency = "week" if horizon == "swing_1_6m" else "month"
    period = "-".join(str(item) for item in rebalance_period_key(signal_date, frequency))
    event_id = (
        f"{strategy_version_id}:scheduled:{horizon}:{period}:"
        f"{dataset_identity}"
    )
    return build_horizon_review_evidence(
        event_id=event_id,
        review_type="scheduled_review",
        horizon_profile=horizon,
        completed_at=completed_at,
        strategy_version_id=strategy_version_id,
        signal_date=signal_date,
        dataset_identity_sha256=dataset_identity,
        trigger_source=f"rebalance_calendar:{frequency}",
        trigger_effective_date=signal_date,
    )


def _write_qlib_order_plan(
    *,
    manifest: dict[str, Any],
    result: dict[str, Any],
    dataset_provenance: dict[str, Any],
    order_plan_root: Path,
    tracking_uri: str,
) -> dict[str, Any]:
    target_weights = dict(
        sorted(
            (
                str(item["instrument"]).upper(),
                float(item["weight"]),
            )
            for item in result["holdings"]
        )
    )
    target_payload = {
        "target_weights": target_weights,
        "paper_policy_state": seal_paper_policy_state(result["position_state"]),
    }
    target_bytes = _canonical_bytes(target_payload)
    target_file_sha256 = _sha256_bytes(target_bytes)
    target_weights_sha256 = _sha256_bytes(
        _canonical_bytes({"target_weights": target_weights})
    )
    signal_at = manifest.get("signal_at")
    signal_date = str(manifest.get("signal_date") or result["as_of_date"])
    plan = {
        "format_version": "qlib-order-plan-v1",
        "produced_by": "qlib-workflow-recorder",
        "source_type": "strategy_version",
        "source_id": manifest["strategy_version_id"],
        "formal_backtest_id": manifest["formal_backtest_id"],
        "promotion_stage_id": manifest["promotion_stage_id"],
        "promotion_stage_opened_at": manifest["promotion_stage_opened_at"],
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
        "target_weights_sha256": target_weights_sha256,
    }
    raw_profile_binding = manifest.get("investor_profile_binding")
    if isinstance(raw_profile_binding, dict):
        plan["investor_profile_binding"] = validate_investor_profile_binding(
            raw_profile_binding
        )
    elif str((manifest.get("config") or {}).get("horizon_profile") or "") in {
        "short_1_5d",
        "swing_1_6m",
        "long_1_3y",
    }:
        raise ValueError(
            "explicit-horizon paper order-plan has no investor-profile binding"
        )
    if signal_at is not None:
        plan["signal_at"] = str(signal_at)
    if manifest.get("execution_not_before") is not None:
        plan["execution_not_before"] = str(manifest["execution_not_before"])
    if isinstance(manifest.get("signal_dataset"), dict):
        plan["signal_snapshot"] = dict(manifest["signal_dataset"])
    if isinstance(manifest.get("model_artifact"), dict):
        plan["model_prediction"] = {
            "artifact_id": manifest["model_artifact"].get("id"),
            "artifact_key": manifest["model_artifact"].get("artifact_key"),
            "strategy_spec_sha256": manifest["model_artifact"].get(
                "strategy_spec_sha256"
            ),
            "model_recipe_sha256": manifest["model_artifact"].get(
                "model_recipe_sha256"
            ),
            "predictions_sha256": manifest["model_artifact"].get(
                "predictions_sha256"
            ),
        }
    horizon_review = _horizon_review_for_order_plan(
        manifest=manifest,
        result=result,
        dataset_provenance=dataset_provenance,
    )
    if horizon_review is not None:
        plan["horizon_review"] = horizon_review
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


def _prepare_execution_evidence(
    point_metadata: pd.DataFrame,
    *,
    instruments: pd.Index,
    risk_projection: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    current_closes = pd.to_numeric(point_metadata["$close"], errors="coerce").reindex(
        instruments
    )
    execution_prices = pd.to_numeric(point_metadata["$open"], errors="coerce").reindex(
        instruments
    )
    valid_current_close = current_closes.notna() & np.isfinite(
        current_closes.to_numpy(dtype=float)
    ) & (current_closes > 0)
    execution_prices = execution_prices.where(valid_current_close)
    average_daily_values = pd.to_numeric(
        point_metadata["Ref(Mean($amount, 20), 1)"], errors="coerce"
    ).reindex(instruments)
    non_tradable = ~risk_projection["tradable"].astype(bool)
    if non_tradable.any():
        blocked = risk_projection.index[non_tradable]
        execution_prices.loc[blocked] = np.nan
        average_daily_values.loc[blocked] = np.nan
    return execution_prices, current_closes, average_daily_values


def _resolve_reference_prices(
    target_weights: dict[str, float],
    *,
    current_prices: pd.Series,
    close_history: pd.DataFrame,
    previous_weights: dict[str, float],
    frozen_instruments: set[str],
) -> tuple[pd.Series, dict[str, str]]:
    """Resolve display/valuation prices without turning stale marks into executions.

    A current positive close is always preferred.  Only an already-held,
    execution-frozen instrument may fall back to its latest positive PIT close;
    a new target with no current close is rejected instead of being converted
    into an order using stale data.
    """

    instruments = pd.Index(target_weights, dtype=str)
    prices = pd.to_numeric(current_prices, errors="coerce").reindex(instruments)
    sources = {str(instrument): "current_close" for instrument in instruments}
    invalid = prices.isna() | ~np.isfinite(prices.to_numpy(dtype=float)) | (prices <= 0)
    for instrument in instruments[invalid]:
        name = str(instrument)
        if float(previous_weights.get(name, 0.0)) <= 0 or name not in frozen_instruments:
            raise ValueError(
                f"new or tradable target {name} requires a positive current reference price"
            )
        if name not in close_history.columns:
            raise ValueError(f"held frozen instrument {name} has no PIT close history")
        history = pd.to_numeric(close_history[name], errors="coerce")
        history = history[np.isfinite(history.to_numpy(dtype=float)) & (history > 0)]
        if history.empty:
            raise ValueError(f"held frozen instrument {name} has no positive PIT reference price")
        prices.loc[name] = float(history.iloc[-1])
        sources[name] = "latest_positive_pit_close"
    return prices.astype(float), sources


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
    verify_qlib_output_manifest(Path(args.provider_uri), dataset_provenance)
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
        verify_qlib_output_manifest(Path(signal_provider_uri), signal_provenance)
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
    signal_source = str(config.get("signal_source") or "factor_score")
    challenger = None
    if signal_source == "model_prediction":
        scores = _model_prediction_scores(
            manifest.get("model_artifact"),
            expected_dataset_identity=str(manifest["dataset_identity_sha256"]),
        )
    elif manifest["factors"]:
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
    if signal_source == "model_prediction":
        pass
    elif isinstance(baseline_definition, dict):
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
        _, normalized_baseline, baseline = normalize_qlib_baseline_values(
            baseline_values, baseline_definition
        )
        baseline = apply_strategy_rule_alpha_weights(
            normalized_baseline,
            baseline,
            config,
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
    previous = {
        str(item["instrument"]): float(item["weight"])
        for item in manifest.get("previous_holdings", [])
    }
    score_instruments = sorted(set(scores.index.get_level_values("instrument")))
    data_instruments = sorted(set(score_instruments) | set(previous))
    required_history = required_rule_history_sessions(config)
    lookback = (
        as_of - pd.Timedelta(days=required_history * 2 + 30)
    ).date().isoformat()
    # $amount is CNY yuan under the v3 daily field contract.
    liquidity = D.features(
        data_instruments,
        ["$amount"],
        start_time=lookback,
        end_time=as_of.date().isoformat(),
        freq="day",
    )
    execution_metadata = D.features(
        data_instruments,
        ["$open", "$close", "Ref(Mean($amount, 20), 1)"],
        start_time=as_of.date().isoformat(),
        end_time=as_of.date().isoformat(),
        freq="day",
    )
    close_history = D.features(
        data_instruments,
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
    constrained = (
        str(config.get("portfolio_construction") or "")
        in _COVARIANCE_REQUIRED_PORTFOLIO_CONSTRUCTIONS
    )
    if config.get("portfolio_construction") == "industry_neutral_qp":
        benchmark_frame = pd.read_parquet(metadata_root / "full_market_weights.parquet")
    elif constrained:
        benchmark_frame = pd.read_parquet(metadata_root / "benchmark_weights.parquet")
        benchmark_frame = benchmark_frame[
            benchmark_frame["benchmark"] == manifest["benchmark"]
        ].drop(columns=["benchmark"])
    else:
        benchmark_frame = None
    styles_frame, style_exposure_evidence = _load_governed_style_exposures(
        args.provider_uri
    )
    eligibility_frame = pd.read_parquet(metadata_root / "eligibility_matrix.parquet")
    eligibility_evidence = eligibility_statistics(eligibility_frame)
    if config.get("require_regulatory_events") and not eligibility_evidence[
        "regulatory_data_available"
    ]:
        raise ValueError("strategy requires regulatory events but no reliable source is available")
    neutralize_industry, neutralize_styles = governed_score_neutralization(config)
    policy_config = PortfolioPolicyConfig.from_mapping(config)
    governed = build_governed_signal(
        scores.loc[(slice(lookback, as_of), slice(None))],
        topk=policy_config.topk,
        n_drop=policy_config.n_drop,
        liquidity_amount=liquidity,
        industry_memberships=memberships,
        benchmark_weights=benchmark_frame,
        style_exposures=styles_frame,
        eligibility_matrix=eligibility_frame,
        max_position_weight=policy_config.max_position_weight,
        max_industry_weight=policy_config.max_industry_weight,
        max_industry_deviation=policy_config.max_industry_deviation,
        min_average_daily_amount=float(config.get("min_average_daily_amount", 0.0)),
        liquidity_lookback_days=int(config.get("liquidity_lookback_days", 20)),
        neutralize_industry=neutralize_industry,
        neutralize_style_columns=neutralize_styles,
        benchmark_relative_industry_constraints=(
            config.get("portfolio_construction")
            in {"benchmark_relative_qp", "industry_neutral_qp"}
        ),
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
    benchmark = (
        _latest(
            filter_available("index_weight", benchmark_frame, as_of),
            as_of,
            "weight",
        )
        if constrained and benchmark_frame is not None
        else None
    )
    required_instruments = signal.index.astype(str).union(
        pd.Index(previous, dtype=str)
    )
    raw_styles, styles = policy_style_cross_sections(
        styles_frame, as_of, config=config
    )
    risk_instruments = required_instruments
    if constrained and benchmark is not None:
        risk_instruments = risk_instruments.union(benchmark.index.astype(str))
    return_covariance = _portfolio_return_covariance(
        config,
        close_history,
        risk_instruments,
    )
    portfolio_metadata = build_portfolio_policy_runtime_metadata(
        config,
        instruments=required_instruments,
        industries=industries,
        benchmark_weights=benchmark,
        style_exposures=styles,
        return_covariance=return_covariance,
    )
    cost_model = CostModelConfig.from_mapping(config)
    policy = PortfolioPolicy(policy_config, cost_model)
    previous_snapshot = manifest.get("previous_snapshot") or {}
    scheduled_rebalance_due = is_rebalance_due(
        as_of,
        previous_snapshot.get("as_of_date"),
        str(config.get("rebalance_frequency", "day")),
    )
    financial_review_trigger = _financial_review_for_signal(
        manifest, as_of.date()
    )
    financial_review_scope = (
        build_financial_review_scope(
            financial_review_trigger,
            current_holdings=list(previous),
            qualified_candidates=list(signal.index.astype(str)),
        )
        if financial_review_trigger is not None
        else None
    )
    # Filing season produces announcements almost every trading day.  Only a
    # newly PIT-effective filing for a held name or an already qualified
    # candidate may open an off-cadence decision; all other source events are
    # retained as a governance-only batch.  Hard risk exits still run below
    # with rebalance_due=False inside the shared policy.
    rebalance_due = _rebalance_due_for_financial_review(
        scheduled_rebalance_due=scheduled_rebalance_due,
        financial_review_scope=financial_review_scope,
    )
    rebalance_instruments = _rebalance_instruments_for_financial_review(
        scheduled_rebalance_due=scheduled_rebalance_due,
        financial_review_scope=financial_review_scope,
    )
    construction_notional = float(manifest["construction_notional"])
    previous_position_state = dict(previous_snapshot.get("position_state") or {})
    runtime_rule_metadata = _recommendation_rule_runtime_metadata(
        D,
        config=config,
        instruments=required_instruments,
        close_history=close_history.loc[:market_as_of],
        value_exposures=(
            raw_styles["value"] if "value" in raw_styles.columns else None
        ),
        start_time=lookback,
        end_time=as_of.date().isoformat(),
    )
    previous_holding_rows = {
        str(item["instrument"]): item
        for item in manifest.get("previous_holdings") or []
        if isinstance(item, dict) and item.get("instrument")
    }
    cost_basis = {
        instrument: float(item["average_cost"])
        for instrument, item in previous_holding_rows.items()
        if item.get("average_cost") is not None
    }
    take_profit_stages = {
        instrument: int(item.get("take_profit_stage") or 0)
        for instrument, item in previous_holding_rows.items()
    }
    risk_projection = project_point_in_time_risk_states(
        eligibility_frame,
        as_of=market_as_of,
        instruments=required_instruments,
    ).set_index("instrument")
    investor_permission_evidence: dict[str, dict[str, Any]] = {}
    if manifest.get("artifact_kind") == "simulation_order_plan":
        raw_profile_binding = manifest.get("investor_profile_binding")
        explicit_horizon = str(
            (manifest.get("config") or {}).get("horizon_profile") or ""
        ) in {"short_1_5d", "swing_1_6m", "long_1_3y"}
        if explicit_horizon and not isinstance(raw_profile_binding, dict):
            raise ValueError(
                "simulation order-plan has no frozen investor-profile binding"
            )
        if isinstance(raw_profile_binding, dict):
            risk_projection, investor_permission_evidence = (
                _apply_investor_profile_permissions(
                    risk_projection,
                    raw_profile_binding,
                    on_date=market_as_of.date(),
                )
            )
    # A positive current close is part of the recommendation evidence contract.
    # Without it, an existing position may be frozen and marked from the latest
    # PIT close, but a new instrument must not become an order.
    execution_prices, current_closes, average_daily_values = _prepare_execution_evidence(
        point_metadata,
        instruments=required_instruments,
        risk_projection=risk_projection,
    )
    decision = policy.decide(
        signal,
        previous,
        **portfolio_metadata,
        prices=execution_prices,
        current_prices=current_closes,
        cost_basis=cost_basis,
        take_profit_stages=(
            previous_position_state.get("take_profit_stages") or take_profit_stages
        ),
        execution_state=previous_position_state.get("execution") or {},
        holding_age_sessions=(
            previous_position_state.get("holding_age_sessions")
            or manifest.get("holding_age_sessions")
            or {}
        ),
        portfolio_drawdown=float(manifest["portfolio_drawdown"]),
        daily_return=float(manifest["daily_return"]),
        # $amount is CNY yuan under the v3 daily field contract.
        average_daily_values=average_daily_values,
        instrument_risk_states=risk_projection["risk_state"],
        portfolio_value=construction_notional,
        risk_exposure=float(manifest.get("risk_exposure", 1.0)),
        allow_new_risk=bool(manifest.get("allow_new_risk", True)),
        rebalance_due=rebalance_due,
        rebalance_instruments=rebalance_instruments,
        **runtime_rule_metadata,
    )
    if signal_frequency == "day":
        effective_date = _next_known_trading_date(args.provider_uri, market_as_of)
    else:
        if manifest.get("signal_at") is None:
            raise ValueError("minute Qlib order-plan generation requires signal_at")
        effective_date = as_of.date().isoformat()
    changes = {item["instrument"]: item for item in decision.changes}
    frozen_instruments = {
        str(item) for item in decision.position_state.get("frozen_instruments") or []
    }
    reference_prices, reference_price_sources = _resolve_reference_prices(
        decision.target_weights,
        current_prices=current_closes,
        close_history=close_history.loc[:market_as_of],
        previous_weights=previous,
        frozen_instruments=frozen_instruments,
    )
    evidence_instruments = (
        set(decision.target_weights)
        | set(previous)
        | set(risk_projection.index[risk_projection["risk_state"].ne("normal")])
    )
    instrument_risk_evidence = {
        str(instrument): {
            "risk_state": str(row["risk_state"]),
            "tradable": bool(row["tradable"]),
            "allow_new_risk": bool(row["allow_new_risk"]),
            "risk_reasons": json.loads(str(row["risk_reasons"])),
            "evidence_date": (
                pd.Timestamp(row["evidence_datetime"]).date().isoformat()
                if pd.notna(row["evidence_datetime"])
                else None
            ),
        }
        for instrument, row in risk_projection.iterrows()
        if str(instrument) in evidence_instruments
    }

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
            "scheduled_rebalance_due": scheduled_rebalance_due,
            "financial_review_trigger": financial_review_trigger,
            "financial_review_scope": financial_review_scope,
            "member_risk_state": dict(manifest.get("member_risk_state") or {}),
            "account_risk_state": dict(manifest.get("account_risk_state") or {}),
            "instrument_risk_states": instrument_risk_evidence,
            "investor_profile_permissions": investor_permission_evidence,
            "frozen_instruments": sorted(frozen_instruments),
            "reference_price_sources": reference_price_sources,
            "eligibility": eligibility_evidence,
            "style_exposure_contract": style_exposure_evidence,
        },
        "reasons": decision.reasons,
        "cash_weight": max(0.0, 1.0 - sum(decision.target_weights.values())),
        "reference_prices": {
            str(instrument): float(price)
            for instrument, price in reference_prices.items()
        },
        "industry_memberships": {
            str(instrument): str(industries.loc[instrument])
            for instrument in decision.target_weights
            if instrument in industries.index
        },
        "holdings": [
            {
                "instrument": instrument,
                "weight": weight,
                "industry": (
                    str(industries.loc[instrument])
                    if instrument in industries.index
                    else None
                ),
                "previous_weight": changes.get(instrument, {}).get("previous_weight", weight),
                "weight_change": changes.get(instrument, {}).get("weight_change", 0.0),
                "action": changes.get(instrument, {}).get("action", "hold"),
                "reason": (
                    "execution evidence unavailable; retained at previous weight"
                    if instrument in frozen_instruments
                    else changes.get(instrument, {}).get("reason", "unchanged target")
                ),
                "execution_state": (
                    "WAIT" if instrument in frozen_instruments else "READY"
                ),
                "reference_price_source": reference_price_sources[instrument],
                "average_cost": cost_basis.get(
                    instrument, float(reference_prices[instrument])
                ),
                "take_profit_stage": int(
                    decision.position_state.get("take_profit_stages", {}).get(
                        instrument, 0
                    )
                ),
                "holding_age_sessions": int(
                    decision.position_state.get("holding_age_sessions", {}).get(
                        instrument, 0
                    )
                ),
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
