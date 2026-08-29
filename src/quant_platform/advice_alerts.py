"""Persistent alerts for governed unified-account action changes.

This is a read-side projection over immutable simulation order-plan batches.
It never creates a target, changes a strategy, or routes a broker order.  The
existing :class:`AlertStore` remains the single Inbox/webhook path.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from quant_data.database import (
    open_database,
    simulation_batches,
    simulation_portfolios,
    strategy_allocations,
)

from .alert_store import AlertStore

ACTIONABLE = frozenset({"BUY", "ADD", "SELL", "REDUCE", "EXIT"})


def visible_account_action(item: dict[str, Any]) -> str:
    """Translate ledger actions into the novice account vocabulary."""

    action = str(item.get("action") or "").strip().upper()
    target = int(item.get("target_quantity") or 0)
    filled = int(item.get("filled_position") or 0)
    if action in {"BUY", "ADD"}:
        return "ADD" if filled > 0 else "BUY"
    if action in {"SELL", "REDUCE"}:
        return "EXIT" if target <= 0 else "REDUCE"
    if action == "EXIT":
        return "EXIT"
    return action if action in {"HOLD", "NO_ACTION"} else "NO_ACTION"


def action_alert_payloads(batch: Any) -> list[dict[str, Any]]:
    """Return actionable per-instrument payloads from one sealed batch.

    HOLD and NO_ACTION intentionally produce no notification.  A later
    actionable batch receives a new immutable batch id and therefore a fresh
    alert; there is no broad action/instrument dedupe that could suppress a
    genuine repeat or escalation.
    """

    payload = dict(batch.target_payload_json or {})
    actions = dict(payload.get("order_plan") or {}).get("actions") or []
    projected: list[dict[str, Any]] = []
    for raw in actions:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        instrument = str(item.get("instrument") or "").strip().upper()
        action = visible_account_action(item)
        if not instrument or action not in ACTIONABLE:
            continue
        target = item.get("target_quantity")
        projected.append(
            {
                "instrument": instrument,
                "action": action,
                "target_quantity": int(target) if target is not None else None,
                "filled_position": int(item.get("filled_position") or 0),
                "projected_position": int(item.get("projected_position") or 0),
                "execution_state": str(item.get("execution_state") or "unknown"),
                "blocked_reason": item.get("blocked_reason"),
                "wait_reason": item.get("wait_reason"),
                "order_plan": list(item.get("order_plan") or []),
            }
        )
    return projected


class UnifiedAccountAdviceAlertProjector:
    """Project the primary paper account's actions into Alert Inbox/webhooks."""

    def __init__(
        self,
        database_url: str,
        *,
        alerts: AlertStore | None = None,
    ) -> None:
        self.engine = open_database(database_url)
        self.alerts = alerts or AlertStore(database_url)

    def project(self, *, limit: int = 500) -> int:
        with self.engine.connect() as connection:
            batches = connection.execute(
                select(simulation_batches)
                .join(
                    simulation_portfolios,
                    simulation_portfolios.c.id == simulation_batches.c.portfolio_id,
                )
                .join(
                    strategy_allocations,
                    strategy_allocations.c.id == simulation_portfolios.c.source_id,
                )
                .where(
                    strategy_allocations.c.status == "active",
                    simulation_portfolios.c.source_type == "allocation",
                    simulation_batches.c.account_netting_plan_id.is_not(None),
                )
                .order_by(simulation_batches.c.created_at.desc())
                .limit(limit)
            ).all()
        projected = 0
        labels = {
            "BUY": "买入",
            "ADD": "加仓",
            "REDUCE": "减仓",
            "EXIT": "退出",
        }
        for batch in batches:
            for action in action_alert_payloads(batch):
                verb = labels[action["action"]]
                target = action["target_quantity"]
                quantity_text = "等待整手换算" if target is None else f"目标 {target} 股"
                self.alerts.create(
                    source_type="simulation_batch",
                    source_id=str(batch.id),
                    severity=(
                        "warning"
                        if action["action"] in {"REDUCE", "EXIT"}
                        else "info"
                    ),
                    category="unified_account_action",
                    title=f"统一模拟账户{verb}提醒：{action['instrument']}",
                    message=(
                        f"{batch.signal_date.isoformat()} 收盘信号，"
                        f"{batch.trade_date.isoformat()} 模拟执行；{quantity_text}；"
                        f"执行状态 {action['execution_state']}。"
                    ),
                    dedupe_key=(
                        "unified-account-action:"
                        f"{batch.id}:{action['instrument']}:{action['action']}:{target}"
                    ),
                    details={
                        **action,
                        "simulation_only": True,
                        "real_broker_order": False,
                        "account_netting_plan_id": str(
                            batch.account_netting_plan_id
                        ),
                        "signal_date": batch.signal_date.isoformat(),
                        "trade_date": batch.trade_date.isoformat(),
                    },
                )
                projected += 1
        return projected
