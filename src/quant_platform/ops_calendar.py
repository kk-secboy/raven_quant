"""Operational run-calendar tasks (design draft §10.4).

Four pieces, all reusing the existing scheduling/alert discipline instead of a
parallel channel:

- ``evaluate_recommendation_gate`` — the ordering gate in front of
  ``recommendation_refresh``: the linked simulation account's latest batch
  must have reconciled (batch succeeded, cash conservation is enforced by the
  engine failing the batch otherwise) and its latest NAV must be healthy,
  performance-certified, free of stale prices and fresh against the signal
  date. Failing the gate blocks the new recommendation fail-closed; the
  previous recommendation snapshot stays in place and is explicitly reported
  as retained/stale via an alert (never silently reused as if it were new).
- ``build_weekly_report`` — weekend report: simulation NAV/fills/costs,
  reconciliation anomalies, risk events and data health over the ISO week,
  persisted as an immutable JSON artifact plus a deduped alert.
- ``build_monthly_decision_day`` — first trading day of the month: monthly
  allocation artifacts whose ``valid_until`` (decision_frequency semantics,
  migration 0039) has expired are re-solved through
  ``AllocationStore.refresh`` (the existing frozen-decision-day gate), plus a
  monthly health summary.
- ``build_preopen_check`` — trading-day preopen check: calendar confirmation,
  data readiness, account status, open orders and suspension/risk summary;
  anomalies raise deduped critical alerts.

All three task builders are deterministic in their dedupe keys (per ISO week
/ per date), so reruns of the same schedule slot are idempotent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from quant_data.config import Settings
from quant_data.database import (
    open_database,
    row_dict,
    simulation_batches,
    simulation_events,
    simulation_fills,
    simulation_nav,
    simulation_orders,
    strategy_allocation_events,
)

from .alert_store import AlertStore
from .allocation_store import AllocationStore
from .health_store import OperationalHealthStore
from .recommendation_store import RecommendationStore
from .services import list_qlib_datasets
from .simulation_store import SimulationStore

# Weekly report runs on Saturday local time (§10.4 每周末). trading_days_only
# does not apply to it: Saturday is never a trading day.
WEEKLY_REPORT_WEEKDAY = 5

OPS_TASK_KINDS = ("weekly_report", "monthly_decision_day", "preopen_check")


def is_weekly_report_day(local_date: date) -> bool:
    return local_date.weekday() == WEEKLY_REPORT_WEEKDAY


def is_monthly_decision_day(local_date: date, calendar_days: set[date]) -> bool:
    """First trading day of the month per the persisted Qlib calendar.

    Fail-closed semantics live in the caller: when the calendar does not cover
    ``local_date`` the task must not guess (never weekday rules).
    """

    if local_date not in calendar_days:
        return False
    month_days = [
        day
        for day in calendar_days
        if day.year == local_date.year and day.month == local_date.month
    ]
    return bool(month_days) and local_date == min(month_days)


def load_calendar_days(dataset_path: str) -> set[date]:
    calendar_path = Path(dataset_path) / "calendars" / "day.txt"
    return {
        date.fromisoformat(line.strip())
        for line in calendar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def select_ops_dataset(data_root: Path, anchor_name: str | None = None) -> dict[str, Any]:
    """Pick the ops-calendar Qlib dataset; fail closed when none is usable."""

    datasets = [
        item
        for item in list_qlib_datasets(data_root)
        if item["ready"] and item.get("reproducible")
    ]
    if anchor_name:
        anchor = next((item for item in datasets if item["name"] == anchor_name), None)
        if anchor is None:
            raise ValueError(f"ops calendar Qlib dataset {anchor_name!r} is not usable")
        return anchor
    if not datasets:
        raise ValueError("no ready and reproducible Qlib dataset for the ops calendar")
    # list_qlib_datasets is sorted newest-first by directory name.
    return datasets[0]


@dataclass(slots=True)
class OpsStores:
    alerts: AlertStore
    simulations: SimulationStore
    recommendations: RecommendationStore
    allocations: AllocationStore
    health: OperationalHealthStore


def ops_stores(settings: Settings) -> OpsStores:
    return OpsStores(
        alerts=AlertStore(settings.database_url),
        simulations=SimulationStore(settings.database_url),
        recommendations=RecommendationStore(settings.database_url),
        allocations=AllocationStore(settings.database_url),
        health=OperationalHealthStore(settings),
    )


# ---------------------------------------------------------------------------
# Reconciliation ordering gate (recommendation_refresh)
# ---------------------------------------------------------------------------


def evaluate_recommendation_gate(
    simulations: SimulationStore,
    portfolio: dict[str, Any],
    signal_date: date,
    calendar_days: set[date] | None,
) -> dict[str, Any]:
    """Check simulation reconciliation health before a recommendation refresh.

    Returns {"passed": bool, "reasons": [...], "details": {...}}. A
    recommendation portfolio without a linked simulation account passes with
    an explicit note (nothing to reconcile against); every linked account must
    have a succeeded latest batch and a healthy, certified, fresh latest NAV.
    """

    linked = [
        item
        for item in simulations.list(1000)
        if str(item.get("source_type")) == "recommendation"
        and str(item.get("source_id")) == str(portfolio["id"])
    ]
    if not linked:
        return {
            "passed": True,
            "reasons": [],
            "details": {"note": "no linked simulation account; gate not applicable"},
        }
    account = max(linked, key=lambda item: str(item.get("updated_at")))
    reasons: list[str] = []
    batch = simulations.latest_batch(str(account["id"]))
    nav = simulations.latest_nav(str(account["id"]))
    details: dict[str, Any] = {
        "simulation_portfolio_id": str(account["id"]),
        "simulation_portfolio_name": account.get("name"),
        "latest_batch": (
            {
                "id": batch["id"],
                "trade_date": str(batch["trade_date"]),
                "status": batch["status"],
            }
            if batch
            else None
        ),
        "latest_nav": (
            {
                "trade_date": str(nav["trade_date"]),
                "status": nav["status"],
                "performance_certified": bool(nav["performance_certified"]),
                "has_stale_prices": bool(nav["has_stale_prices"]),
            }
            if nav
            else None
        ),
    }
    if batch is None:
        reasons.append("linked simulation account has no reconciled batch yet")
    elif str(batch["status"]) != "succeeded":
        reasons.append(
            f"latest simulation batch {batch['id']} ({batch['trade_date']}) "
            f"is {batch['status']}: reconciliation did not pass"
        )
    if nav is None:
        reasons.append("linked simulation account has no NAV row yet")
    else:
        if calendar_days is not None:
            covered = [day for day in calendar_days if day <= signal_date]
            if not covered or signal_date > max(calendar_days):
                reasons.append(
                    f"dataset calendar does not cover the signal date {signal_date}"
                )
            else:
                required_date = max(covered)
                details["required_nav_date"] = required_date.isoformat()
                if nav["trade_date"] < required_date:
                    reasons.append(
                        f"latest simulation NAV is {nav['trade_date']}, lagging the "
                        f"required {required_date} (data freshness)"
                    )
        if str(nav["status"]) == "degraded":
            reasons.append("latest simulation NAV is degraded")
        if not bool(nav["performance_certified"]):
            reasons.append("latest simulation NAV is not performance-certified")
        if bool(nav["has_stale_prices"]):
            reasons.append("latest simulation NAV carries stale prices")
    return {"passed": not reasons, "reasons": reasons, "details": details}


# ---------------------------------------------------------------------------
# Shared aggregation helpers
# ---------------------------------------------------------------------------


def _table_counts(
    engine: Any,
    table: Any,
    date_column: Any,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    start_dt = datetime.combine(start, time.min)
    end_dt = datetime.combine(end, time.max)
    column = (
        getattr(table.c, date_column)
        if isinstance(date_column, str)
        else date_column
    )
    with engine.connect() as connection:
        rows = connection.execute(
            select(table).where(column >= start_dt, column <= end_dt)
        ).all()
    return [row_dict(row) for row in rows]


def _data_health_summary(stores: OpsStores) -> dict[str, Any]:
    latest = stores.health.latest()
    if latest is None:
        return {"status": "unknown", "note": "no operational health snapshot recorded"}
    components = latest.get("components") or {}
    degraded = [
        name
        for name, component in components.items()
        if str(component.get("status")) in {"degraded", "unavailable"}
    ]
    return {
        "status": "degraded" if degraded else "healthy",
        "recorded_at": latest.get("recorded_at"),
        "degraded_components": sorted(degraded),
    }


def _write_report_artifact(data_root: Path, name: str, report: dict[str, Any]) -> str:
    directory = data_root / "artifacts" / "ops-reports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return str(path)


# ---------------------------------------------------------------------------
# Weekly report
# ---------------------------------------------------------------------------


def build_weekly_report(
    settings: Settings, stores: OpsStores, local_date: date
) -> dict[str, Any]:
    """ISO-week operations report ending on ``local_date`` (a Saturday)."""

    iso = local_date.isocalendar()
    week_start = local_date - timedelta(days=6)
    engine = open_database(settings.database_url)

    nav_rows = _table_counts(engine, simulation_nav, "trade_date", week_start, local_date)
    batches = _table_counts(engine, simulation_batches, "trade_date", week_start, local_date)
    fills = _table_counts(engine, simulation_fills, "executed_at", week_start, local_date)
    events = _table_counts(engine, simulation_events, "created_at", week_start, local_date)
    risk_events = _table_counts(
        engine, strategy_allocation_events, "created_at", week_start, local_date
    )
    with engine.connect() as connection:
        open_orders = connection.execute(
            select(func.count())
            .select_from(simulation_orders)
            .where(simulation_orders.c.status.in_(("planned", "open")))
        ).scalar_one()

    degraded_nav = [row for row in nav_rows if str(row["status"]) == "degraded"]
    uncertified_nav = [row for row in nav_rows if not row["performance_certified"]]
    failed_batches = [row for row in batches if str(row["status"]) == "failed"]
    severe_events = [
        row for row in events if str(row["severity"]) in {"warning", "critical"}
    ]
    open_risk_events = [row for row in risk_events if str(row["status"]) == "open"]
    total_fee = sum(float(row["fee"]) for row in fills)
    health = _data_health_summary(stores)
    datasets = [
        {"name": item["name"], "end_date": item["end_date"]}
        for item in list_qlib_datasets(settings.data_root)
        if item["ready"]
    ]

    report: dict[str, Any] = {
        "kind": "weekly_report",
        "iso_week": f"{iso[0]}-W{iso[1]:02d}",
        "window": {"start": week_start.isoformat(), "end": local_date.isoformat()},
        "nav": {
            "rows": len(nav_rows),
            "degraded_rows": len(degraded_nav),
            "uncertified_rows": len(uncertified_nav),
            "worst_drawdown": min((float(row["drawdown"]) for row in nav_rows), default=0.0),
        },
        "batches": {
            "total": len(batches),
            "succeeded": len(batches) - len(failed_batches),
            "failed": [
                {"id": row["id"], "trade_date": str(row["trade_date"]), "error": row.get("error")}
                for row in failed_batches
            ],
        },
        "fills": {"count": len(fills), "total_fee": total_fee},
        "open_orders": int(open_orders),
        "simulation_events": {"warning_or_critical": len(severe_events)},
        "allocation_risk_events": {"open": len(open_risk_events)},
        "data_health": health,
        "datasets": datasets,
    }
    artifact_path = _write_report_artifact(
        settings.data_root, f"weekly-{iso[0]}-W{iso[1]:02d}", report
    )
    report["artifact_path"] = artifact_path

    week_key = f"{iso[0]}-W{iso[1]:02d}"
    anomalies = {
        "failed_batches": failed_batches,
        "degraded_nav": degraded_nav,
        "uncertified_nav": uncertified_nav,
        "open_risk_events": open_risk_events,
    }
    for slug, rows in anomalies.items():
        if not rows:
            continue
        stores.alerts.create(
            source_type="ops_task",
            source_id=f"weekly_report:{week_key}",
            severity="critical",
            category=f"weekly_report_{slug}",
            title=f"周报异常({week_key})：{slug} {len(rows)} 项",
            message=f"weekly report {week_key} flagged {len(rows)} {slug}; see {artifact_path}",
            dedupe_key=f"weekly-report:{week_key}:{slug}",
            details={"count": len(rows), "artifact_path": artifact_path},
        )
    stores.alerts.create(
        source_type="ops_task",
        source_id=f"weekly_report:{week_key}",
        severity="info",
        category="weekly_report",
        title=f"周报 {week_key}",
        message=(
            f"NAV 行 {len(nav_rows)}（degraded {len(degraded_nav)}）、批次 "
            f"{len(batches)}（失败 {len(failed_batches)}）、成交 {len(fills)} 笔、"
            f"费用 {total_fee:.2f}、数据健康 {health['status']}"
        ),
        dedupe_key=f"weekly-report:{week_key}",
        details={"artifact_path": artifact_path, "report": report},
    )
    return report


# ---------------------------------------------------------------------------
# Monthly decision day
# ---------------------------------------------------------------------------


def build_monthly_decision_day(
    settings: Settings, stores: OpsStores, local_date: date
) -> dict[str, Any]:
    """First-trading-day monthly pass: due allocation artifacts + health summary."""

    due: list[dict[str, Any]] = []
    refreshed: list[str] = []
    failed: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for allocation in stores.allocations.list(1000):
        if allocation.get("is_legacy") or str(allocation["status"]) != "active":
            skipped.append(
                {"allocation_id": allocation["id"], "reason": str(allocation["status"])}
            )
            continue
        artifacts = allocation.get("artifacts") or []
        latest = artifacts[0] if artifacts else None
        valid_until = (
            date.fromisoformat(str(latest["valid_until"])) if latest else None
        )
        if latest is not None and valid_until is not None and local_date < valid_until:
            continue  # artifact still frozen-valid (decision_frequency semantics)
        due.append(
            {
                "allocation_id": allocation["id"],
                "allocation_name": allocation.get("name"),
                "decision_frequency": allocation.get("decision_frequency"),
                "valid_until": valid_until.isoformat() if valid_until else None,
            }
        )
        try:
            stores.allocations.refresh(
                str(allocation["id"]), actor="monthly-decision-day"
            )
            refreshed.append(str(allocation["id"]))
        except Exception as exc:  # noqa: BLE001 - recorded and alerted, never silent
            failed.append({"allocation_id": str(allocation["id"]), "error": str(exc)[:500]})

    health = _data_health_summary(stores)
    report: dict[str, Any] = {
        "kind": "monthly_decision_day",
        "date": local_date.isoformat(),
        "allocations_due": due,
        "allocations_refreshed": refreshed,
        "allocations_refresh_failed": failed,
        "allocations_skipped": skipped,
        "data_health": health,
    }
    artifact_path = _write_report_artifact(
        settings.data_root, f"monthly-decision-{local_date.isoformat()}", report
    )
    report["artifact_path"] = artifact_path

    day_key = local_date.isoformat()
    for item in failed:
        stores.alerts.create(
            source_type="ops_task",
            source_id=f"monthly_decision_day:{day_key}",
            severity="critical",
            category="monthly_decision_refresh_failed",
            title=f"月度决策日重估失败：{item['allocation_id']}",
            message=item["error"],
            dedupe_key=f"monthly-decision-day:{day_key}:{item['allocation_id']}",
            details={"artifact_path": artifact_path},
        )
    stores.alerts.create(
        source_type="ops_task",
        source_id=f"monthly_decision_day:{day_key}",
        severity="info",
        category="monthly_decision_day",
        title=f"月度决策日 {day_key}",
        message=(
            f"到期 allocation {len(due)} 个，重估成功 {len(refreshed)}、"
            f"失败 {len(failed)}；数据健康 {health['status']}"
        ),
        dedupe_key=f"monthly-decision-day:{day_key}",
        details={"artifact_path": artifact_path, "report": report},
    )
    return report


# ---------------------------------------------------------------------------
# Preopen check
# ---------------------------------------------------------------------------


def build_preopen_check(
    settings: Settings,
    stores: OpsStores,
    local_date: date,
    *,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Trading-day preopen check; anomalies raise deduped critical alerts."""

    calendar_days = load_calendar_days(dataset["path"])
    if local_date not in calendar_days:
        # The scheduler calendar gate should have skipped this slot already;
        # reaching here means the dataset changed underneath — fail closed.
        raise ValueError(f"{local_date} is not a trading day in {dataset['name']}")
    previous_days = [day for day in calendar_days if day < local_date]
    previous_day = max(previous_days) if previous_days else None

    engine = open_database(settings.database_url)
    anomalies: list[dict[str, str]] = []
    dataset_end = (
        date.fromisoformat(str(dataset["end_date"])) if dataset.get("end_date") else None
    )
    if previous_day is None or dataset_end is None or dataset_end < previous_day:
        anomalies.append(
            {
                "slug": "data-not-ready",
                "message": (
                    f"dataset {dataset['name']} ends at {dataset.get('end_date')}, "
                    f"previous trading day {previous_day} is not covered"
                ),
            }
        )

    portfolios = stores.simulations.list(1000)
    inactive_accounts = [
        {"id": item["id"], "name": item.get("name"), "status": item["status"]}
        for item in portfolios
        if str(item["status"]) != "active"
    ]
    with engine.connect() as connection:
        open_orders = connection.execute(
            select(func.count())
            .select_from(simulation_orders)
            .where(simulation_orders.c.status.in_(("planned", "open")))
        ).scalar_one()
        recent_events = connection.execute(
            select(simulation_events).where(
                simulation_events.c.trade_date
                >= (previous_day or local_date) - timedelta(days=3),
                simulation_events.c.severity.in_(("warning", "critical")),
            )
        ).all()
        open_risk = connection.execute(
            select(func.count())
            .select_from(strategy_allocation_events)
            .where(strategy_allocation_events.c.status == "open")
        ).scalar_one()
    event_rows = [row_dict(row) for row in recent_events]
    suspensions = [
        row for row in event_rows if "suspend" in str(row.get("event_type", "")).lower()
    ]
    if inactive_accounts:
        anomalies.append(
            {
                "slug": "inactive-accounts",
                "message": f"{len(inactive_accounts)} simulation account(s) not active",
            }
        )
    if open_risk:
        anomalies.append(
            {
                "slug": "open-risk-events",
                "message": f"{int(open_risk)} open allocation risk event(s)",
            }
        )

    report: dict[str, Any] = {
        "kind": "preopen_check",
        "date": local_date.isoformat(),
        "dataset": {"name": dataset["name"], "end_date": dataset.get("end_date")},
        "previous_trading_day": previous_day.isoformat() if previous_day else None,
        "accounts": {
            "total": len(portfolios),
            "inactive": inactive_accounts,
        },
        "open_orders": int(open_orders),
        "recent_warning_events": len(event_rows),
        "suspension_events": len(suspensions),
        "open_allocation_risk_events": int(open_risk),
        "anomalies": anomalies,
    }
    artifact_path = _write_report_artifact(
        settings.data_root, f"preopen-{local_date.isoformat()}", report
    )
    report["artifact_path"] = artifact_path

    day_key = local_date.isoformat()
    for anomaly in anomalies:
        stores.alerts.create(
            source_type="ops_task",
            source_id=f"preopen_check:{day_key}",
            severity="critical",
            category=f"preopen_check_{anomaly['slug']}",
            title=f"盘前检查异常({day_key})：{anomaly['slug']}",
            message=anomaly["message"],
            dedupe_key=f"preopen-check:{day_key}:{anomaly['slug']}",
            details={"artifact_path": artifact_path},
        )
    stores.alerts.create(
        source_type="ops_task",
        source_id=f"preopen_check:{day_key}",
        severity="info",
        category="preopen_check",
        title=f"盘前检查 {day_key}",
        message=(
            f"数据截至 {dataset.get('end_date')}，账户 {len(portfolios)} 个"
            f"（非活跃 {len(inactive_accounts)}），未完成订单 {int(open_orders)}，"
            f"异常 {len(anomalies)} 项"
        ),
        dedupe_key=f"preopen-check:{day_key}",
        details={"artifact_path": artifact_path, "report": report},
    )
    return report
