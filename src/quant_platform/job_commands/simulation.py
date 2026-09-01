"""Simulation, recommendation and replay command builders for LocalJobWorker."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_data.path_utils import to_wsl_path as _to_wsl_path

from ..data_rollover import qlib_trading_date_on_or_before
from ..execution_algorithms import execution_time_slots
from ..horizon_review import resolve_financial_review_trigger
from ..investor_profile import validate_investor_profile_binding
from ..ops_calendar import load_calendar_days
from ..paper_policy_state import bind_current_paper_holdings
from ..services import list_qlib_datasets, resolve_snapshot_dataset
from ._shared import (
    _bind_daily_simulation_settlement_calendar,
    _qlib_workflow_environment,
    _require_supported_simulation_execution,
)


def simulation_order_plan_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    portfolio = worker.simulations.get(payload["simulation_portfolio_id"])
    if (
        portfolio["status"] != "active"
        or portfolio["source_type"] != "strategy_version"
        or portfolio["execution_adapter"] != "long_only"
    ):
        raise ValueError(
            "simulation order-plan generation requires an active long-only "
            "strategy-version simulation"
        )
    version = worker.strategies.get_version(portfolio["source_id"])
    if version["status"] != "approved" or version.get("is_legacy"):
        raise ValueError(
            "simulation order-plan generation requires an approved "
            "non-legacy strategy version"
        )
    formal = next(
        (
            item
            for item in worker.strategies.list_backtests(version["id"])
            if item["status"] == "succeeded" and not item.get("is_legacy")
        ),
        None,
    )
    if formal is None:
        raise ValueError(
            "simulation order-plan generation requires a successful formal Qlib backtest"
        )
    datasets = {
        item["name"]: item
        for item in list_qlib_datasets(worker.settings.data_root)
        if item.get("ready") and item.get("reproducible")
    }
    anchor = datasets.get(portfolio["daily_dataset"])
    if anchor is None:
        raise ValueError("simulation order-plan Qlib daily dataset is unavailable")
    anchor_provenance = dict(anchor.get("provenance") or {})
    if (
        anchor_provenance.get("dataset_identity_sha256")
        != portfolio["daily_dataset_identity_sha256"]
        or anchor_provenance.get("dataset_lineage_id")
        != portfolio["daily_dataset_lineage_id"]
    ):
        raise ValueError(
            "simulation order-plan Qlib dataset no longer matches the "
            "bound account snapshot"
        )
    local_today = datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).date()
    frozen_identity = str(payload.get("dataset_identity_sha256") or "")
    dataset = next(
        (
            item
            for item in datasets.values()
            if str(
                dict(item.get("provenance") or {}).get(
                    "dataset_identity_sha256"
                )
                or ""
            )
            == frozen_identity
            and str(
                dict(item.get("provenance") or {}).get("dataset_lineage_id")
                or ""
            )
            == str(portfolio.get("daily_dataset_lineage_id") or "")
            and dict(item.get("provenance") or {}).get("lineage_verified")
            is True
        ),
        None,
    )
    if dataset is None:
        raise ValueError(
            "paper order-plan frozen Qlib dataset is unavailable or unverified"
        )
    provenance = dict(dataset.get("provenance") or {})
    signal_date = date.fromisoformat(str(payload["signal_date"]))
    if str(payload.get("dataset_identity_sha256") or "") != str(
        provenance.get("dataset_identity_sha256") or ""
    ):
        raise ValueError(
            "paper order-plan job changed its immutable dataset binding"
        )
    current_available_date = qlib_trading_date_on_or_before(dataset, local_today)
    if signal_date != current_available_date:
        raise ValueError(
            "paper signal is not the latest currently available governed trading day"
        )
    promotion_stage = worker.promotions.require_paper_signal(
        str(version["id"]),
        portfolio_id=str(portfolio["id"]),
        signal_date=signal_date,
    )
    worker.simulations.require_order_plan_predecessor_settled(
        str(portfolio["id"]),
        signal_date=signal_date,
    )
    if (
        str(payload.get("promotion_stage_id") or "") != promotion_stage["id"]
        or str(payload.get("promotion_stage_opened_at") or "")
        != promotion_stage["opened_at"]
    ):
        raise ValueError("paper order-plan job changed its promotion-stage binding")
    signal_frequency = str(version.get("signal_frequency") or "day").lower()
    signal_at = payload.get("signal_at")
    execution_not_before: str | None = None
    signal_dataset: dict | None = None
    if signal_frequency != "day":
        if not signal_at:
            raise ValueError("minute simulation order-plan requires signal_at")
        try:
            signal_timestamp = datetime.fromisoformat(str(signal_at).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("minute simulation order-plan signal_at is invalid") from exc
        if signal_timestamp.tzinfo is None or signal_timestamp.utcoffset() is None:
            raise ValueError("minute simulation order-plan signal_at requires a timezone")
        local_signal = signal_timestamp.astimezone(ZoneInfo("Asia/Shanghai"))
        if local_signal.date().isoformat() != str(payload["signal_date"]):
            raise ValueError(
                "minute simulation order-plan signal_at does not match signal_date"
            )
        source_lineage = str(provenance.get("source_lineage_id") or "")
        candidates = [
            item
            for item in datasets.values()
            if item.get("reproducible") is True
            and dict(item.get("provenance") or {}).get("lineage_verified") is True
            and str(dict(item.get("provenance") or {}).get("frequency") or "")
            == signal_frequency
            and str(dict(item.get("provenance") or {}).get("source_lineage_id") or "")
            == source_lineage
        ]
        signal_dataset = next(
            (item for item in candidates if item["name"] == portfolio["execution_dataset"]),
            candidates[0] if candidates else None,
        )
        if signal_dataset is None:
            raise ValueError(
                "minute simulation order-plan requires a ready Qlib signal "
                f"dataset at {signal_frequency} from the bound Tushare lineage"
            )
        first_slot = execution_time_slots(
            trade_date=local_signal.date(),
            policy=dict(portfolio["execution_policy"]),
            signal_at=signal_timestamp,
        )[0]
        execution_not_before = first_slot.isoformat()
    strategy_config = dict(version.get("config") or {})
    horizon_profile = str(
        version.get("horizon_profile")
        or strategy_config.get("horizon_profile")
        or "legacy_ambiguous"
    )
    raw_investor_profile_binding = dict(
        portfolio.get("execution_policy") or {}
    ).get("investor_profile_binding")
    investor_profile_binding = (
        validate_investor_profile_binding(raw_investor_profile_binding)
        if isinstance(raw_investor_profile_binding, dict)
        else None
    )
    if (
        horizon_profile in {"short_1_5d", "swing_1_6m", "long_1_3y"}
        and investor_profile_binding is None
    ):
        raise ValueError(
            "paper order-plan account has no frozen investor-profile binding"
        )
    requires_complete_holding_age = (
        horizon_profile in {"short_1_5d", "swing_1_6m", "long_1_3y"}
        or strategy_config.get("max_holding_sessions") is not None
        or strategy_config.get("thesis_min_holding_sessions") is not None
    )
    positions = worker.simulations.positions_with_holding_age(
        str(portfolio["id"]),
        calendar_days=load_calendar_days(str(dataset["path"])),
        as_of_date=signal_date,
        require_complete_age=requires_complete_holding_age,
    )
    nav = float(portfolio["nav"])
    previous_holdings = [
        {
            "instrument": str(item["instrument"]),
            "weight": max(0.0, float(item.get("market_value") or 0.0)) / nav,
            "average_cost": float(item["average_cost"]),
            "holding_age_sessions": (
                int(item["holding_age_sessions"])
                if item.get("holding_age_sessions") is not None
                else None
            ),
        }
        for item in positions
        if nav > 0
        and str(item.get("position_side") or "long") == "long"
        and float(item.get("market_value") or 0.0) > 0
    ]
    previous_snapshot = None
    if horizon_profile in {"short_1_5d", "swing_1_6m", "long_1_3y"}:
        previous_snapshot = worker.simulations.latest_paper_previous_snapshot(
            str(portfolio["id"]),
            promotion_stage_id=str(promotion_stage["id"]),
            before_signal_date=signal_date,
        )
        previous_snapshot = bind_current_paper_holdings(
            previous_snapshot,
            previous_holdings,
        )
    previous_signal_date = (
        date.fromisoformat(str(previous_snapshot["as_of_date"])[:10])
        if previous_snapshot and previous_snapshot.get("as_of_date")
        else None
    )
    financial_review_trigger = (
        resolve_financial_review_trigger(
            data_root=worker.settings.data_root,
            dataset_provenance=provenance,
            previous_signal_date=previous_signal_date,
            signal_date=signal_date,
        )
        if horizon_profile == "long_1_3y"
        else None
    )
    strategy_risk_state = worker.allocations.strategy_risk_state(str(version["id"]))
    required_nav_date = qlib_trading_date_on_or_before(
        dataset,
        date.fromisoformat(str(payload["signal_date"])),
    )
    account_risk_state = worker.simulations.policy_risk_inputs(
        str(portfolio["id"]),
        required_nav_date=required_nav_date,
    )
    output = worker.settings.data_root / "artifacts" / "order-plan-jobs" / job["id"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str | Path) -> str:
        return _to_wsl_path(Path(value)) if is_wsl else str(value)

    model_artifact = worker._bound_current_model_artifact(
        version=version,
        payload=payload,
        dataset_identity_sha256=str(
            provenance["dataset_identity_sha256"]
        ),
        signal_date=signal_date,
    )
    factor_artifacts = worker._bound_current_factor_artifacts(
        version=version,
        payload=payload,
        dataset_identity_sha256=str(
            provenance["dataset_identity_sha256"]
        ),
        signal_date=signal_date,
    )

    manifest = {
        "artifact_kind": "simulation_order_plan",
        "order_plan_job_id": job["id"],
        "simulation_portfolio_id": portfolio["id"],
        "portfolio_id": portfolio["id"],
        "strategy_version_id": version["id"],
        "formal_backtest_id": formal["id"],
        "dataset": dataset["name"],
        "dataset_identity_sha256": provenance["dataset_identity_sha256"],
        "dataset_lineage_id": provenance["dataset_lineage_id"],
        "promotion_stage_id": promotion_stage["id"],
        "promotion_stage_opened_at": promotion_stage["opened_at"],
        "investor_profile_binding": investor_profile_binding,
        "signal_date": payload["signal_date"],
        "signal_at": signal_at,
        "execution_not_before": execution_not_before,
        "as_of_date": signal_at or payload["signal_date"],
        "benchmark": version["benchmark"],
        "universe": version["universe"],
        "config": version["config"],
        "model_artifact": (
            {
                **model_artifact,
                "artifact_path": runtime_path(model_artifact["artifact_path"]),
            }
            if model_artifact is not None
            else None
        ),
        "construction_notional": nav,
        "risk_exposure": float(strategy_risk_state["risk_exposure_override"]),
        "risk_exposure_override": float(strategy_risk_state["risk_exposure_override"]),
        "allow_new_risk": bool(strategy_risk_state["allow_new_risk"])
        and bool(account_risk_state["allow_new_risk"]),
        "member_risk_state": strategy_risk_state,
        "account_risk_state": account_risk_state,
        "portfolio_drawdown": account_risk_state["portfolio_drawdown"],
        "daily_return": account_risk_state["daily_return"],
        "previous_holdings": previous_holdings,
        "holding_age_sessions": {
            str(item["instrument"]): int(item["holding_age_sessions"])
            for item in previous_holdings
            if item.get("holding_age_sessions") is not None
        },
        "holding_age_evidence": {
            str(item["instrument"]): dict(item["holding_age_evidence"])
            for item in positions
            if item.get("holding_age_evidence") is not None
        },
        "previous_snapshot": previous_snapshot,
        "financial_review_trigger": financial_review_trigger,
        "signal_dataset": (
            {
                "name": signal_dataset["name"],
                "dataset_identity_sha256": dict(signal_dataset.get("provenance") or {}).get(
                    "dataset_identity_sha256"
                ),
                "dataset_lineage_id": dict(signal_dataset.get("provenance") or {}).get(
                    "dataset_lineage_id"
                ),
                "source_lineage_id": dict(signal_dataset.get("provenance") or {}).get(
                    "source_lineage_id"
                ),
                "frequency": signal_frequency,
            }
            if signal_dataset is not None
            else None
        ),
        "factors": [
            {
                "candidate_id": item["candidate_id"],
                "values_path": runtime_path(item["artifact_path"]),
                "weight": item["weight"],
                "direction": item["direction"],
            }
            for item in factor_artifacts
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    script = worker.project_root / "scripts" / "run_recommendation_refresh.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            runtime_path(dataset["path"]),
            "--manifest",
            runtime_path(manifest_path),
            "--output",
            runtime_path(result_path),
            "--tracking-uri",
            worker.settings.mlflow_tracking_uri,
            "--order-plan-root",
            runtime_path(worker.settings.data_root / "artifacts" / "order-plans"),
        ]
    )
    if signal_dataset is not None:
        command.extend(
            [
                "--signal-provider-uri",
                runtime_path(signal_dataset["path"]),
            ]
        )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def recommendation_refresh_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = (
        worker.settings.data_root
        / "artifacts"
        / "recommendations"
        / payload["recommendation_portfolio_id"]
        / payload["recommendation_snapshot_id"]
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    portfolio = worker.recommendations.get(payload["recommendation_portfolio_id"])
    version = worker.strategies.get_version(portfolio["strategy_version_id"])
    if version["status"] != "approved":
        raise ValueError("recommendation refresh requires an approved strategy version")
    recommendation_date = date.fromisoformat(str(payload["as_of_date"]))
    current_available_date = qlib_trading_date_on_or_before(
        {"path": payload["dataset_path"]},
        datetime.now(UTC).astimezone(ZoneInfo("Asia/Shanghai")).date(),
    )
    if recommendation_date != current_available_date:
        raise ValueError(
            "recommendation refresh is not the latest currently available "
            "governed trading day"
        )
    worker.promotions.require_recommendation_signal(
        str(version["id"]), signal_date=recommendation_date
    )
    member_risk_state = worker.allocations.strategy_risk_state(str(version["id"]))
    required_nav_date = qlib_trading_date_on_or_before(
        {"path": payload["dataset_path"]},
        date.fromisoformat(str(payload["as_of_date"])),
    )
    account_risk_state = worker.recommendation_accounts.policy_risk_inputs(
        str(portfolio["id"]),
        required_nav_date=required_nav_date,
    )
    risk_exposure = min(
        float(portfolio.get("risk_exposure_override", 1.0)),
        float(member_risk_state["risk_exposure_override"]),
    )
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str) -> str:
        return _to_wsl_path(Path(value)) if is_wsl else str(value)

    snapshot_history = [
        item
        for item in portfolio.get("snapshots") or []
        if isinstance(item, dict)
        and item.get("status") == "succeeded"
        and str(item.get("id") or "")
        != str(payload.get("recommendation_snapshot_id") or "")
    ]
    latest_snapshot = snapshot_history[0] if snapshot_history else {}
    if not latest_snapshot:
        fallback_snapshot = portfolio.get("latest_snapshot") or {}
        if fallback_snapshot.get("status") in {None, "succeeded"}:
            latest_snapshot = fallback_snapshot
    latest_snapshot_payload = dict(latest_snapshot.get("snapshot") or {})
    previous_position_state = dict(
        latest_snapshot_payload.get("position_state") or {}
    )
    dataset_provenance_path = (
        Path(payload["dataset_path"]) / "metadata" / "provenance.json"
    )
    try:
        dataset_provenance = json.loads(
            dataset_provenance_path.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "recommendation refresh has no immutable dataset provenance"
        ) from exc
    if str(dataset_provenance.get("dataset_identity_sha256") or "") != str(
        payload["dataset_identity_sha256"]
    ):
        raise ValueError("recommendation refresh changed its dataset identity")
    model_artifact = worker._bound_current_model_artifact(
        version=version,
        payload=payload,
        dataset_identity_sha256=str(payload["dataset_identity_sha256"]),
        signal_date=recommendation_date,
    )
    factor_artifacts = worker._bound_current_factor_artifacts(
        version=version,
        payload=payload,
        dataset_identity_sha256=str(payload["dataset_identity_sha256"]),
        signal_date=recommendation_date,
    )
    horizon_profile = str(
        version.get("horizon_profile")
        or version.get("config", {}).get("horizon_profile")
        or "legacy_ambiguous"
    )
    previous_signal_date = (
        date.fromisoformat(str(latest_snapshot["as_of_date"])[:10])
        if latest_snapshot and latest_snapshot.get("as_of_date")
        else None
    )
    financial_review_trigger = (
        resolve_financial_review_trigger(
            data_root=worker.settings.data_root,
            dataset_provenance=dataset_provenance,
            previous_signal_date=previous_signal_date,
            signal_date=recommendation_date,
        )
        if horizon_profile == "long_1_3y"
        else None
    )
    manifest = {
        "portfolio_id": portfolio["id"],
        "strategy_version_id": version["id"],
        "dataset": payload["dataset"],
        "dataset_identity_sha256": payload["dataset_identity_sha256"],
        "as_of_date": payload["as_of_date"],
        "benchmark": version["benchmark"],
        "universe": version["universe"],
        "config": version["config"],
        "model_artifact": (
            {
                **model_artifact,
                "artifact_path": runtime_path(model_artifact["artifact_path"]),
            }
            if model_artifact is not None
            else None
        ),
        "construction_notional": float(portfolio["construction_notional"]),
        "risk_exposure": risk_exposure,
        "risk_exposure_override": risk_exposure,
        "allow_new_risk": bool(member_risk_state["allow_new_risk"])
        and bool(account_risk_state["allow_new_risk"]),
        "member_risk_state": member_risk_state,
        "account_risk_state": account_risk_state,
        "portfolio_drawdown": account_risk_state["portfolio_drawdown"],
        "daily_return": account_risk_state["daily_return"],
        "previous_holdings": latest_snapshot.get("holdings") or [],
        "previous_snapshot": (
            {
                "as_of_date": str(latest_snapshot["as_of_date"]),
                "effective_date": str(latest_snapshot.get("effective_date") or ""),
                "holdings": latest_snapshot.get("holdings") or [],
                "position_state": previous_position_state,
            }
            if latest_snapshot
            else None
        ),
        "financial_review_trigger": financial_review_trigger,
        "factors": [
            {
                "candidate_id": item["candidate_id"],
                "values_path": runtime_path(item["artifact_path"]),
                "weight": item["weight"],
                "direction": item["direction"],
            }
            for item in factor_artifacts
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    script = worker.project_root / "scripts" / "run_recommendation_refresh.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            _to_wsl_path(Path(payload["dataset_path"]))
            if is_wsl
            else str(Path(payload["dataset_path"])),
            "--manifest",
            _to_wsl_path(manifest_path) if is_wsl else str(manifest_path),
            "--output",
            _to_wsl_path(result_path) if is_wsl else str(result_path),
        ]
    )
    return command, result_path, {}


def simulation_replay_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    manifest = worker.simulations.execution_manifest(payload["simulation_batch_id"])
    _require_supported_simulation_execution(
        "simulation_replay",
        execution_adapter=str(manifest.get("execution_adapter") or ""),
    )
    datasets = {
        item["name"]: item
        for item in list_qlib_datasets(worker.settings.data_root)
        if item.get("ready")
    }
    minute_dataset = datasets.get(manifest["execution_dataset"])
    if minute_dataset is None:
        raise ValueError("simulation execution Qlib dataset is unavailable")
    manifest = _bind_daily_simulation_settlement_calendar(
        manifest,
        minute_dataset,
    )
    output = (
        worker.settings.data_root
        / "artifacts"
        / "simulations"
        / manifest["portfolio_id"]
        / manifest["batch_id"]
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")
    script = worker.project_root / "scripts" / "run_simulation_replay.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            _to_wsl_path(Path(minute_dataset["path"]))
            if is_wsl
            else str(minute_dataset["path"]),
            "--manifest",
            _to_wsl_path(manifest_path) if is_wsl else str(manifest_path),
            "--output",
            _to_wsl_path(result_path) if is_wsl else str(result_path),
        ]
    )
    dividend_dataset = None
    daily_dataset = datasets.get(manifest["daily_dataset"])
    daily_provenance = (
        dict(daily_dataset.get("provenance") or {}) if daily_dataset else {}
    )
    dividend_snapshot_name = str(daily_provenance.get("snapshot_name") or "")
    if dividend_snapshot_name:
        try:
            resolved_dividend = resolve_snapshot_dataset(
                worker.settings.data_root,
                snapshot_name=dividend_snapshot_name,
                dataset_name="dividend",
            )
        except (FileNotFoundError, ValueError, KeyError):
            resolved_dividend = None
        if resolved_dividend is not None:
            expected_manifest = str(
                daily_provenance.get("snapshot_manifest_sha256") or ""
            )
            if expected_manifest and expected_manifest != str(
                resolved_dividend["manifest_sha256"]
            ):
                raise ValueError(
                    "dividend snapshot no longer matches the bound daily dataset"
                )
            dividend_dataset = resolved_dividend
    if dividend_dataset is not None:
        dividend_path = Path(dividend_dataset["dataset_path"])
        command.extend(
            [
                "--dividend-path",
                _to_wsl_path(dividend_path) if is_wsl else str(dividend_path),
            ]
        )
    return command, result_path, {}


COMMANDS = {
    "simulation_order_plan": simulation_order_plan_command,
    "recommendation_refresh": recommendation_refresh_command,
    "simulation_replay": simulation_replay_command,
}
