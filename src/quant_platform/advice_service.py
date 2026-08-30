"""Novice-facing three-horizon advice projection.

This module is intentionally a projection over the existing governed stores.
It cannot create research candidates, promote strategies, or place orders.  In
particular, paper targets remain visibly labelled as simulation evidence and
only ``recommendation_enabled`` versions may produce investment advice.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import case, select

from quant_data.cninfo_announcements import load_trade_calendar_open_days
from quant_data.database import (
    account_netting_plans,
    backtest_runs,
    jobs,
    open_database,
    recommendation_holdings,
    recommendation_portfolios,
    recommendation_snapshots,
    row_dict,
    simulation_batches,
    simulation_portfolios,
    strategies,
    strategy_allocation_members,
    strategy_allocations,
    strategy_forward_gates,
    strategy_promotion_stages,
    strategy_versions,
)

from .advice_freshness import (
    advice_session_requirement,
    assess_netting_plan_freshness,
    assess_signal_freshness,
)
from .data_rollover import select_qlib_dataset
from .ops_calendar import load_calendar_days
from .promotion import (
    PROMOTION_CONTRACT_VERSION,
    PromotionStore,
    _require_forward_gate_criteria,
)
from .research_horizon import LEGACY_AMBIGUOUS, LONG_1_3Y, SHORT_1_5D, SWING_1_6M
from .simulation_store import SimulationStore
from .strategy_health_authority import load_production_health_gate
from .strategy_store import StrategyStore
from .three_horizon_account import THREE_HORIZON_PRIMARY_SIMULATION_ACTOR

ADVICE_TODAY_CONTRACT_VERSION = "three-horizon-advice-v4"
HORIZONS = (SHORT_1_5D, SWING_1_6M, LONG_1_3Y)

_LONG_MATURITY_THRESHOLDS = {
    "forward_trading_days": 756,
    "review_events": 36,
    "financial_report_reviews": 12,
}

_MARKET_PERMISSION_LABELS = {
    "main_board": "沪深主板",
    "star_market": "科创板",
    "chi_next": "创业板",
    "beijing_exchange": "北交所",
    "etf": "境内ETF",
}

_HORIZON_UI = {
    SHORT_1_5D: {
        "title": "短线",
        "holding": "1～5个交易日",
        "research_cadence": "每周或漂移触发",
        "review_sessions": 1,
        "validity_sessions": 5,
    },
    SWING_1_6M: {
        "title": "中线",
        "holding": "1～6个月",
        "research_cadence": "每月或漂移触发",
        "review_sessions": 5,
        "validity_sessions": 126,
    },
    LONG_1_3Y: {
        "title": "长线",
        "holding": "1～3年以上",
        "research_cadence": "每季度、财报季后或论点失效触发",
        "review_sessions": 21,
        "validity_sessions": 756,
    },
}

_HEALTH_PHASE = {
    "healthy": "verified",
    "watch": "verified",
    "restricted": "restricted",
    "suspended": "suspended",
    "retired": "retired",
    "missing": "restricted",
    "invalid": "restricted",
    "insufficient_evidence": "restricted",
    # Compatibility projections for pre-0072 health evidence remain
    # conservative until a current snapshot replaces them.
    "degraded": "restricted",
    "unhealthy": "suspended",
    "frozen": "suspended",
}

_BACKTEST_FAILURE_STATUSES = frozenset({"failed", "cancelled"})


def _project_backtest_status(
    run_status: Any,
    job_status: Any,
) -> str:
    """Combine the durable backtest and worker-job state without optimistic claims."""

    run = str(run_status or "").strip().lower()
    job = str(job_status or "").strip().lower()
    if run in _BACKTEST_FAILURE_STATUSES or job in _BACKTEST_FAILURE_STATUSES:
        return "failed"
    # Job completion is the outer execution boundary.  A worker may persist the
    # backtest result shortly before it marks the job terminal, so do not expose
    # that transient interval as completed.
    if job == "running" or run == "running":
        return "running"
    if job == "queued" or run == "queued":
        return "queued"
    if run == "succeeded" and job in {"", "succeeded"}:
        return "succeeded"
    if not run and not job:
        return "not_started"
    return "unknown"


def _project_stage(
    *,
    version_status: Any,
    promotion_stage: Any,
    health_status: str,
    backtest_status: str,
    forward_gate_passed: bool,
) -> tuple[str, str]:
    """Return the novice stage while failing closed on unfinished evidence."""

    if backtest_status == "failed":
        return "backtest", "回测失败"
    if backtest_status == "running":
        return "backtest", "回测运行中"
    if backtest_status == "queued":
        return "backtest", "回测排队"

    status = str(version_status or "").strip().lower()
    promotion = str(promotion_stage or "").strip().lower()
    if status != "approved":
        return "research", "研究中"
    if promotion in {"paper", "recommendation_enabled"} and backtest_status != "succeeded":
        return "backtest", "回测证据不完整"
    if promotion == "paper":
        return "simulation_validation", "模拟验证中"
    if promotion == "recommendation_enabled":
        if not forward_gate_passed:
            return "restricted", "受限"
        stage = _HEALTH_PHASE.get(health_status, "restricted")
        return stage, {
            "verified": "已验证",
            "restricted": "受限",
            "suspended": "暂停",
            "retired": "已退役",
        }[stage]
    return "research", "研究中"


def _insufficient_evidence(reasons: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "status": "insufficient_evidence",
        "passed": False,
        "reasons": reasons,
        **extra,
    }


def _forward_gate_checks(
    *,
    horizon: str,
    gate: Any,
    evidence: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    values: dict[str, tuple[float | int, float | int, str]] = {
        "governed_batch_integrity": (
            1.0 if evidence["ungoverned_batches"] == 0 else 0.0,
            1.0,
            "min",
        ),
        "data_completeness": (
            evidence["data_completeness"],
            float(gate.min_data_completeness),
            "min",
        ),
        "reconciliation_rate": (
            evidence["reconciliation_rate"],
            float(gate.min_reconciliation_rate),
            "min",
        ),
        "cost_deviation": (
            evidence["cost_deviation"],
            float(gate.max_cost_deviation),
            "max",
        ),
    }
    if horizon == LEGACY_AMBIGUOUS:
        values.update(
            {
                "forward_calendar_days": (
                    evidence["forward_calendar_days"],
                    int(gate.min_forward_calendar_days),
                    "min",
                ),
                "decision_batches": (
                    evidence["decision_batches"],
                    int(gate.min_decision_batches),
                    "min",
                ),
                "completed_cycles": (
                    evidence["completed_cycles"],
                    int(gate.min_completed_cycles),
                    "min",
                ),
            }
        )
    elif horizon == SHORT_1_5D:
        values.update(
            {
                "forward_trading_days": (
                    evidence["forward_trading_days"],
                    int(gate.min_forward_trading_days),
                    "min",
                ),
                "decision_batches": (
                    evidence["decision_batches"],
                    int(gate.min_decision_batches),
                    "min",
                ),
                "closed_round_trips": (
                    evidence["closed_round_trips"],
                    int(gate.min_closed_round_trips),
                    "min",
                ),
            }
        )
    elif horizon == SWING_1_6M:
        values.update(
            {
                "forward_trading_days": (
                    evidence["forward_trading_days"],
                    int(gate.min_forward_trading_days),
                    "min",
                ),
                "review_events": (
                    evidence["review_events"],
                    int(gate.min_review_events),
                    "min",
                ),
                "closed_round_trips": (
                    evidence["closed_round_trips"],
                    int(gate.min_closed_round_trips),
                    "min",
                ),
            }
        )
    elif horizon == LONG_1_3Y:
        values.update(
            {
                "forward_trading_days": (
                    evidence["forward_trading_days"],
                    int(gate.min_forward_trading_days),
                    "min",
                ),
                "review_events": (
                    evidence["review_events"],
                    int(gate.min_review_events),
                    "min",
                ),
                "financial_report_reviews": (
                    evidence["financial_report_reviews"],
                    int(gate.min_financial_report_reviews),
                    "min",
                ),
            }
        )
    else:
        raise ValueError(f"unsupported horizon profile: {horizon}")
    return {
        name: {
            "observed": observed,
            "threshold": threshold,
            "passed": observed >= threshold if mode == "min" else observed <= threshold,
        }
        for name, (observed, threshold, mode) in values.items()
    }


def _date_value(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None
    return None


def _action(value: Any, *, previous_weight: float = 0.0, weight: float = 0.0) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"BUY", "ADD", "HOLD", "REDUCE", "EXIT", "NO_ACTION"}:
        return normalized
    if normalized in {"INCREASE", "NEW"}:
        return "BUY" if previous_weight <= 0 else "ADD"
    if normalized in {"DECREASE", "SELL"}:
        return "EXIT" if weight <= 0 else "REDUCE"
    return "HOLD" if weight > 0 else "NO_ACTION"


def _remaining_trade_quantity(item: dict[str, Any]) -> int:
    """Return the still-actionable order quantity, never the target position."""

    if str(item.get("execution_state") or "").upper() in {"BLOCKED", "WAIT"}:
        return 0
    total = 0
    for raw in item.get("order_plan") or []:
        if not isinstance(raw, dict):
            continue
        order = dict(raw)
        if str(order.get("op") or "").lower() not in {"new", "keep", "replace"}:
            continue
        quantity = int(order.get("quantity") or 0)
        if quantity > 0:
            total += quantity
    return total


def _unified_account_action(trades: list[dict[str, Any]]) -> str:
    """Summarize only executable account trades, never continuous weight intent."""

    for item in trades:
        action = str(item.get("action") or "").upper()
        quantity = int(item.get("trade_quantity") or 0)
        state = str(item.get("execution_state") or "").upper()
        if (
            action in {"BUY", "ADD", "REDUCE", "EXIT"}
            and quantity > 0
            and state not in {"BLOCKED", "WAIT"}
        ):
            return "REBALANCE"
    return "NO_ACTION"


def _unified_no_action_reason(instrument_facts: Mapping[str, Any]) -> str:
    """Explain why a sealed net target produced no executable board-lot trade."""

    reasons: list[str] = []
    for raw in instrument_facts.values():
        if not isinstance(raw, Mapping):
            continue
        reasons.extend(str(item) for item in raw.get("cannot_buy_reasons") or [])
    unique = list(dict.fromkeys(reason for reason in reasons if reason))
    if unique:
        return "本次没有可执行的整手交易：" + "；".join(unique)
    return "净额目标与当前持仓一致，本次不需要买卖，资金继续保留在现金或原持仓中"


def _long_maturity_projection(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Project the optional three-year maturity badge, never an advice gate."""

    checks = {
        name: {
            "observed": int(evidence.get(name) or 0),
            "threshold": threshold,
            "passed": int(evidence.get(name) or 0) >= threshold,
        }
        for name, threshold in _LONG_MATURITY_THRESHOLDS.items()
    }
    mature = all(item["passed"] for item in checks.values())
    return {
        "status": "mature" if mature else "accumulating",
        "label": "三年前向成熟" if mature else "三年成熟度积累中",
        "passed": mature,
        "checks": checks,
        "recommendation_blocking": False,
    }


def _cannot_buy_reasons(item: Mapping[str, Any]) -> list[str]:
    """Translate sealed account-plan constraints into novice-facing reasons."""

    reasons: list[str] = []
    permission_key = str(item.get("investor_permission_key") or "")
    if item.get("new_risk_blocked") is True or permission_key:
        market = _MARKET_PERMISSION_LABELS.get(permission_key, "该证券市场")
        reasons.append(f"你尚未确认{market}交易权限，本次不新增买入")

    raw_reasons = [
        str(item.get("blocked_reason") or ""),
        str(item.get("wait_reason") or ""),
    ]
    for raw in raw_reasons:
        if not raw:
            continue
        if raw == "fresh_reference_price_unavailable":
            reasons.append("缺少当日有效参考价格，暂不能计算可买数量")
        elif raw.startswith("account_risk_off_new_risk_blocked"):
            reasons.append("账户风险状态已禁止新增买入")
        elif raw.startswith("account_risk_caution_manual_review"):
            reasons.append("账户风险状态需要复核，本次暂不新增买入")
        elif raw.startswith("active_recommendation_account_unavailable"):
            reasons.append("活动模拟账户不可用，暂不能计算买入")
        elif raw.startswith("manual_shadow_valuation_incomplete"):
            reasons.append("模拟账户估值不完整，暂不能计算买入")
        elif raw == "sellable_quantity_unavailable":
            reasons.append("可卖数量受T+1或未完成委托限制，等待下一次复核")
        else:
            reasons.append("账户执行门槛尚未满足，本次暂不交易")

    notes = {str(note) for note in item.get("notes") or []}
    if "new_buy_below_min_lot" in notes:
        reasons.append("模拟本金不足以买入最小整手，资金保留为现金")

    action = AdviceService._display_account_action(dict(item))
    state = str(item.get("execution_state") or "").upper()
    if action in {"BUY", "ADD"} and state in {"BLOCKED", "WAIT"} and not reasons:
        reasons.append("账户执行条件尚未满足，本次暂不买入")
    return list(dict.fromkeys(reasons))


class AdviceService:
    """Read-only product projection for simple and advanced UI modes."""

    def __init__(self, database_url: str, *, data_root: Path | None = None) -> None:
        self.engine = open_database(database_url)
        self.data_root = data_root
        self.strategies = StrategyStore(database_url)
        self.promotions = PromotionStore(database_url)
        self.simulations = SimulationStore(database_url)

    def today(
        self,
        *,
        investor_profile: dict[str, Any] | None = None,
        now: datetime | None = None,
        platform_safe_mode: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        freshness_requirement = advice_session_requirement(
            self.data_root,
            now=now,
        )
        cards = [
            self._horizon_card(
                horizon,
                now=now,
                freshness_requirement=freshness_requirement,
            )
            for horizon in HORIZONS
        ]
        cutoffs = [
            _date_value(card.get("data_cutoff"))
            for card in cards
            if card.get("data_cutoff")
        ]
        verified = [card for card in cards if card["is_investment_advice"]]
        onboarding_required = investor_profile is None
        unified = self._unified_account(
            verified_cards=verified,
            formal_cards=[card for card in cards if card["stage"] == "verified"],
            onboarding_required=onboarding_required,
            freshness_requirement=freshness_requirement,
        )
        self._attach_unified_account_facts(cards, unified)
        result = {
            "contract_version": ADVICE_TODAY_CONTRACT_VERSION,
            "generated_at": now.isoformat(),
            "data_cutoff": max(cutoffs).isoformat() if cutoffs else None,
            "onboarding_required": onboarding_required,
            "investor_profile": investor_profile,
            "cards": cards,
            "unified_account": unified,
            "freshness": freshness_requirement,
            "advice_available": (
                bool(verified)
                and not onboarding_required
                and unified.get("status") == "ready"
            ),
            "execution_contract": {
                "signal": "D日收盘后",
                "earliest_fill": "D+1开盘或保守日线成交模型",
                "intraday_claims": False,
                "real_broker_orders": False,
            },
            "disclaimer": "系统仅运行模拟盘；历史和模拟表现不保证未来收益。",
        }
        return self._apply_platform_safe_mode(result, platform_safe_mode)

    @staticmethod
    def _apply_platform_safe_mode(
        projection: dict[str, Any],
        state: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Make a platform emergency stop visible and non-executable.

        Existing EXIT/REDUCE observations remain available as explicitly
        blocked risk information.  They are not presented as executable
        advice because safe mode is reserved for data, ledger or version
        anomalies and the simulation order path is stopped as well.
        """

        raw_state = dict(state or {})
        triggered_at = raw_state.get("triggered_at")
        gate = {
            "status": "safe_mode" if raw_state.get("active") is True else "open",
            "active": raw_state.get("active") is True,
            "reason": str(raw_state.get("reason") or "") or None,
            "source": str(raw_state.get("source") or "") or None,
            "triggered_at": (
                triggered_at.isoformat()
                if isinstance(triggered_at, datetime)
                else str(triggered_at or "") or None
            ),
        }
        result = {**projection, "platform_gate": gate}
        if not gate["active"]:
            return result

        reason = (
            "平台安全模式已开启：暂停正式荐股和模拟订单；"
            "已有减仓/退出信号仅保留为风险信息。"
        )
        cards: list[dict[str, Any]] = []
        for raw_card in projection.get("cards") or []:
            card = dict(raw_card)
            risk_exits = [
                dict(signal)
                for signal in card.get("signals") or []
                if str(signal.get("action") or "").upper() in {"REDUCE", "EXIT"}
            ]
            card.update(
                {
                    "is_investment_advice": False,
                    "action": "NO_ACTION",
                    "simulation_action": None,
                    "signals": [],
                    "safe_mode_risk_exits": risk_exits,
                    "platform_gate": gate,
                    "veto_reasons": [
                        reason,
                        *[
                            str(item)
                            for item in card.get("veto_reasons") or []
                            if str(item) != reason
                        ],
                    ],
                }
            )
            cards.append(card)

        raw_unified = dict(projection.get("unified_account") or {})
        blocked_reductions = [
            dict(trade)
            for trade in raw_unified.get("trades") or []
            if str(trade.get("side") or "").lower() == "sell"
        ]
        unified = {
            **raw_unified,
            "status": "safe_mode",
            "action": "NO_ACTION",
            "reason": reason,
            "targets": [],
            "trades": [],
            "safe_mode_risk_reductions": blocked_reductions,
            "platform_gate": gate,
        }
        return {
            **result,
            "cards": cards,
            "unified_account": unified,
            "advice_available": False,
        }

    def _review_projection(
        self,
        *,
        effective_date: date | None,
        review_sessions: int,
        dataset: str | None,
    ) -> dict[str, Any]:
        """Project a review date only from the governed exchange calendar.

        The immutable strategy contract defines the interval in trading
        sessions.  A Monday-Friday approximation is not an exchange calendar
        (holidays and exceptional closures matter), so an exact date is only
        exposed when the persisted SSE calendar contains the future review
        session.  The current Qlib dataset ends at the latest published bar
        and therefore cannot be used to schedule a future review date.
        """

        result = {
            "review_sessions": int(review_sessions),
            "review_date": None,
            # Compatibility keys consumed by the current novice UI.  The old
            # value was a weekday guess; keeping it null makes that UI say it
            # is waiting for an exchange calendar instead of showing fiction.
            "review_date_estimate": None,
            "review_date_is_exchange_calendar": False,
            "review_date_source": "strategy_horizon_contract",
            "review_date_status": "awaiting_governed_exchange_calendar",
        }
        data_root = getattr(self, "data_root", None)
        if effective_date is None or data_root is None:
            return result
        try:
            calendar = load_trade_calendar_open_days(data_root)
            effective_index = calendar.index(effective_date)
            review_index = effective_index + int(review_sessions)
            if review_index >= len(calendar):
                return result
            review_date = calendar[review_index].isoformat()
        except (OSError, RuntimeError, TypeError, ValueError):
            return result
        return {
            **result,
            "review_date": review_date,
            "review_date_estimate": review_date,
            "review_date_is_exchange_calendar": True,
            "review_date_source": "persisted_sse_trade_calendar",
            "review_date_status": "scheduled",
        }

    @staticmethod
    def _display_account_action(item: dict[str, Any]) -> str:
        action = str(item.get("action") or "").strip().upper()
        target = int(item.get("target_quantity") or 0)
        filled = int(item.get("filled_position") or 0)
        if action == "BUY":
            return "ADD" if filled > 0 else "BUY"
        if action in {"SELL", "REDUCE"}:
            return "EXIT" if target <= 0 else "REDUCE"
        if action in {"EXIT", "HOLD", "NO_ACTION"}:
            return action
        return "NO_ACTION"

    def _unified_execution_facts(
        self,
        plan_id: str,
        *,
        account_id: str,
        portfolio_id: str,
    ) -> dict[str, Any]:
        """Read quantities and lot ages from the one authoritative paper ledger."""

        with self.engine.connect() as connection:
            portfolios = connection.execute(
                select(simulation_portfolios)
                .where(
                    simulation_portfolios.c.id == portfolio_id,
                    simulation_portfolios.c.status == "active",
                    simulation_portfolios.c.source_type == "allocation",
                    simulation_portfolios.c.source_id == account_id,
                    simulation_portfolios.c.created_by
                    == THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                )
                .limit(1)
            ).all()
            if len(portfolios) != 1:
                return {
                    "status": "primary_ledger_missing" if not portfolios else "ambiguous",
                    "items": {},
                    "portfolio_id": None,
                    "batch_id": None,
                    "reason": (
                        "统一模拟账户尚未建立"
                        if not portfolios
                        else "统一模拟账户存在多个活动主账本"
                    ),
                }
            portfolio = portfolios[0]
            batch = connection.execute(
                select(simulation_batches)
                .where(
                    simulation_batches.c.portfolio_id == portfolio.id,
                    simulation_batches.c.account_netting_plan_id == plan_id,
                    simulation_batches.c.created_by
                    == THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                )
                .order_by(
                    simulation_batches.c.trade_date.desc(),
                    simulation_batches.c.created_at.desc(),
                )
                .limit(1)
            ).first()
            age_as_of = connection.scalar(
                select(simulation_batches.c.trade_date)
                .where(
                    simulation_batches.c.portfolio_id == portfolio.id,
                    simulation_batches.c.status == "succeeded",
                )
                .order_by(
                    simulation_batches.c.trade_date.desc(),
                    simulation_batches.c.created_at.desc(),
                )
                .limit(1)
            )

        decision_record = None
        if batch is None:
            try:
                decision_record = self.simulations.get_account_order_plan_decision(
                    str(portfolio.id),
                    account_netting_plan_id=plan_id,
                )
            except ValueError as exc:
                return {
                    "status": "decision_evidence_invalid",
                    "items": {},
                    "portfolio_id": str(portfolio.id),
                    "batch_id": None,
                    "decision_event_id": None,
                    "reason": str(exc),
                }

        facts: dict[str, dict[str, Any]] = {}
        actions: list[Any] = []
        quantity_source = "unified_account_order_plan"
        if batch is not None:
            payload = dict(batch.target_payload_json or {})
            actions = dict(payload.get("order_plan") or {}).get("actions") or []
        elif decision_record is not None:
            actions = list(
                dict(decision_record.get("decision") or {}).get("actions") or []
            )
            quantity_source = "unified_account_decision_only"
        for raw in actions:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            instrument = str(item.get("instrument") or "").upper()
            if not instrument:
                continue
            target = item.get("target_quantity")
            cannot_buy_reasons = _cannot_buy_reasons(item)
            desired_action = self._display_account_action(item)
            execution_state = str(item.get("execution_state") or "").upper()
            trade_quantity = _remaining_trade_quantity(item)
            executable_action = desired_action
            if desired_action in {"BUY", "ADD", "REDUCE", "EXIT"} and (
                trade_quantity <= 0 or execution_state in {"BLOCKED", "WAIT"}
            ):
                executable_action = "NO_ACTION"
            facts[instrument] = {
                "action": executable_action,
                "desired_action": desired_action,
                "target_position_quantity": (
                    int(target) if target is not None else None
                ),
                "trade_quantity": trade_quantity,
                "filled_position": int(item.get("filled_position") or 0),
                "projected_position": int(item.get("projected_position") or 0),
                "execution_state": execution_state,
                "quantity_source": quantity_source,
                "cannot_buy_reasons": cannot_buy_reasons,
                "execution_constraints": {
                    "blocked_reason": item.get("blocked_reason"),
                    "wait_reason": item.get("wait_reason"),
                    "notes": list(item.get("notes") or []),
                    "new_risk_blocked": item.get("new_risk_blocked") is True,
                    "new_risk_blocked_reason": item.get(
                        "new_risk_blocked_reason"
                    ),
                    "investor_permission_key": item.get(
                        "investor_permission_key"
                    ),
                },
                "holding_age_sessions": None,
                "holding_age_source": "awaiting_authoritative_account_lots",
            }

        data_root = getattr(self, "data_root", None)
        if data_root is not None and isinstance(age_as_of, date):
            try:
                selected = select_qlib_dataset(
                    data_root,
                    anchor_name=str(portfolio.daily_dataset),
                    roll_policy="pinned",
                    lineage_id=None,
                    required_date=age_as_of,
                )
                positions = self.simulations.positions_with_holding_age(
                    str(portfolio.id),
                    calendar_days=load_calendar_days(str(selected["path"])),
                    as_of_date=age_as_of,
                )
            except (FileNotFoundError, KeyError, ValueError):
                positions = []
            for position in positions:
                instrument = str(position.get("instrument") or "").upper()
                if not instrument:
                    continue
                fact = facts.setdefault(
                    instrument,
                    {
                        "action": "HOLD",
                        "target_position_quantity": int(
                            position.get("quantity") or 0
                        ),
                        "trade_quantity": 0,
                        "filled_position": int(position.get("quantity") or 0),
                        "projected_position": int(position.get("quantity") or 0),
                        "execution_state": "ready",
                        "quantity_source": "unified_account_position_ledger",
                    },
                )
                age = position.get("holding_age_sessions")
                fact["holding_age_sessions"] = int(age) if age is not None else None
                fact["holding_age_source"] = (
                    "simulation_position_lots_qlib_calendar"
                    if age is not None
                    else "unproven_authoritative_account_lots"
                )
                fact["holding_age_evidence"] = position.get("holding_age_evidence")
        has_decision = batch is not None or decision_record is not None
        return {
            "status": "ready" if has_decision else "execution_plan_missing",
            "items": facts,
            "portfolio_id": str(portfolio.id),
            "batch_id": str(batch.id) if batch is not None else None,
            "decision_event_id": (
                str(decision_record["id"]) if decision_record is not None else None
            ),
            "holding_age_as_of": age_as_of.isoformat() if isinstance(age_as_of, date) else None,
            "reason": (
                None
                if has_decision
                else "最新净额计划尚未生成统一账户执行批次或零订单决策"
            ),
        }

    def _primary_account_binding(self) -> dict[str, Any]:
        """Resolve the sole active three-horizon ledger and its exact allocation."""

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    simulation_portfolios.c.id.label("portfolio_id"),
                    simulation_portfolios.c.source_id.label("account_id"),
                )
                .join(
                    strategy_allocations,
                    strategy_allocations.c.id == simulation_portfolios.c.source_id,
                )
                .where(
                    strategy_allocations.c.status == "active",
                    simulation_portfolios.c.status == "active",
                    simulation_portfolios.c.source_type == "allocation",
                    simulation_portfolios.c.created_by
                    == THREE_HORIZON_PRIMARY_SIMULATION_ACTOR,
                )
                .order_by(simulation_portfolios.c.created_at)
                .limit(2)
            ).all()
        if len(rows) != 1:
            return {
                "status": "missing" if not rows else "ambiguous",
                "reason": (
                    "统一模拟账户尚未绑定活动三周期配置"
                    if not rows
                    else "存在多个活动三周期主账本，已阻止账户建议"
                ),
            }
        return {
            "status": "ready",
            "portfolio_id": str(rows[0].portfolio_id),
            "account_id": str(rows[0].account_id),
        }

    @staticmethod
    def _validate_netting_plan_seal(
        row: Any,
        *,
        plan: dict[str, Any],
        account_id: str,
        portfolio_id: str,
    ) -> str | None:
        """Validate durable row bindings and the plan's primary-ledger evidence."""

        bindings = (
            ("account_id", account_id),
            ("allocation_artifact_id", str(row.allocation_artifact_id)),
            ("plan_key", str(row.plan_key)),
            ("plan_hash", str(row.plan_hash)),
        )
        if any(str(plan.get(field) or "") != expected for field, expected in bindings):
            return "统一账户净额计划的数据库封存字段不一致"
        raw_evidence = plan.get("input_evidence")
        if not isinstance(raw_evidence, dict):
            return "统一账户净额计划缺少封存输入证据"
        raw_primary = raw_evidence.get("primary_account")
        if not isinstance(raw_primary, dict):
            return "统一账户净额计划未封存三周期主账本证据"
        primary = dict(raw_primary)
        if (
            str(primary.get("portfolio_id") or "") != portfolio_id
            or str(primary.get("source_id") or "") != account_id
        ):
            return "统一账户净额计划未绑定当前三周期主账本"
        return None

    @staticmethod
    def _attach_unified_account_facts(
        cards: list[dict[str, Any]], unified: dict[str, Any]
    ) -> None:
        facts = dict(unified.get("instrument_facts") or {})
        if not facts and unified.get("status") != "ready":
            reason = str(
                unified.get("reason")
                or "统一账户执行计划尚未就绪，本次暂不买入"
            )
            for card in cards:
                for signal in card.get("signals") or []:
                    signal["account_action"] = "NO_ACTION"
                    signal["target_position_quantity"] = 0
                    signal["trade_quantity"] = 0
                    signal["execution_state"] = "BLOCKED"
                    signal["quantity_source"] = "unified_account_not_ready"
                    signal["cannot_buy_reasons"] = [reason]
                if card.get("is_investment_advice") is True:
                    card["action"] = "NO_ACTION"
            return
        for card in cards:
            for signal in card.get("signals") or []:
                fact = facts.get(str(signal.get("instrument") or "").upper())
                if not isinstance(fact, dict):
                    if unified.get("status") == "ready":
                        signal["account_action"] = "NO_ACTION"
                        signal["target_position_quantity"] = 0
                        signal["trade_quantity"] = 0
                        signal["quantity_source"] = "unified_account_netted_out"
                        desired_action = str(signal.get("action") or "").upper()
                        if desired_action in {"SELL", "REDUCE", "EXIT"}:
                            netted_reason = (
                                "三个周期净额后该股票没有账户级可卖数量，本次无需卖出"
                            )
                        elif desired_action == "HOLD":
                            netted_reason = "三个周期净额后该股票无需账户级交易，本次继续观察"
                        else:
                            netted_reason = (
                                "三个周期净额后该股票没有账户级买入量，本次无需买入"
                            )
                        signal["cannot_buy_reasons"] = [
                            netted_reason
                        ]
                    continue
                if fact.get("target_position_quantity") is not None:
                    signal["target_position_quantity"] = int(
                        fact["target_position_quantity"]
                    )
                if fact.get("trade_quantity") is not None:
                    signal["trade_quantity"] = int(fact["trade_quantity"])
                signal["account_action"] = fact.get("action")
                signal["execution_state"] = fact.get("execution_state")
                signal["quantity_source"] = fact.get("quantity_source")
                signal["cannot_buy_reasons"] = list(
                    fact.get("cannot_buy_reasons") or []
                )
                signal["execution_constraints"] = dict(
                    fact.get("execution_constraints") or {}
                )
                if fact.get("holding_age_sessions") is not None:
                    signal["holding_age_sessions"] = int(
                        fact["holding_age_sessions"]
                    )
                    signal["holding_age_source"] = fact.get("holding_age_source")
                    signal["holding_age_evidence"] = fact.get(
                        "holding_age_evidence"
                    )
            if card.get("is_investment_advice") is True:
                account_actions = [
                    str(signal.get("account_action") or "NO_ACTION").upper()
                    for signal in card.get("signals") or []
                ]
                if any(action in {"BUY", "ADD"} for action in account_actions):
                    card["action"] = "BUY"
                elif any(
                    action in {"REDUCE", "EXIT"} for action in account_actions
                ):
                    card["action"] = "REDUCE"
                elif any(action == "HOLD" for action in account_actions):
                    card["action"] = "HOLD"
                else:
                    card["action"] = "NO_ACTION"

    def _authoritative_member_health_snapshots(
        self, account_id: str
    ) -> dict[str, dict[str, Any] | None]:
        """Return the latest health seal for every exact active-allocation member."""

        with self.engine.connect() as connection:
            members = connection.execute(
                select(
                    strategy_allocation_members.c.strategy_version_id,
                    strategy_versions.c.status,
                    strategy_versions.c.promotion_stage,
                )
                .join(
                    strategy_versions,
                    strategy_versions.c.id
                    == strategy_allocation_members.c.strategy_version_id,
                )
                .where(strategy_allocation_members.c.allocation_id == account_id)
                .order_by(strategy_allocation_members.c.strategy_version_id)
            ).all()
            result: dict[str, dict[str, Any] | None] = {
                str(member.strategy_version_id): None for member in members
            }
            for member in members:
                version_id = str(member.strategy_version_id)
                if (
                    str(member.status) != "approved"
                    or str(member.promotion_stage) != "recommendation_enabled"
                ):
                    continue
                gate = load_production_health_gate(connection, version_id)
                if gate.get("ready") is True:
                    result[version_id] = {
                        "snapshot_id": str(gate["snapshot_id"]),
                        "as_of": str(gate["as_of"]),
                    }
        return result

    def _latest_version(self, horizon: str) -> dict[str, Any] | None:
        serving_incumbent = self.promotions.serving_incumbent_for_pending_cutover(
            horizon
        )
        if serving_incumbent is not None:
            with self.engine.connect() as connection:
                row = connection.execute(
                    select(
                        strategy_versions.c.id,
                        strategies.c.name.label("strategy_name"),
                    )
                    .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
                    .where(strategy_versions.c.id == serving_incumbent)
                ).first()
            if row is not None:
                version = self.strategies.get_version(str(row.id))
                version["strategy_name"] = str(row.strategy_name)
                version["activation_cutover_fallback"] = True
                return version
        priority = case(
            (
                strategy_versions.c.promotion_stage == "recommendation_enabled",
                0,
            ),
            (strategy_versions.c.promotion_stage == "paper", 1),
            else_=2,
        )
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    strategy_versions.c.id,
                    strategies.c.name.label("strategy_name"),
                )
                .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
                .where(
                    strategy_versions.c.horizon_profile == horizon,
                    strategy_versions.c.status.in_(("draft", "candidate", "approved")),
                )
                .order_by(priority, strategy_versions.c.created_at.desc())
                .limit(1)
            ).first()
        if row is None:
            return None
        version = self.strategies.get_version(str(row.id))
        version["strategy_name"] = str(row.strategy_name)
        return version

    def _latest_backtest(self, version_id: str) -> dict[str, Any]:
        """Read the latest formal backtest and its outer worker state."""

        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    backtest_runs.c.id.label("run_id"),
                    backtest_runs.c.status.label("run_status"),
                    backtest_runs.c.created_at.label("created_at"),
                    backtest_runs.c.started_at.label("started_at"),
                    backtest_runs.c.finished_at.label("finished_at"),
                    jobs.c.id.label("job_id"),
                    jobs.c.status.label("job_status"),
                )
                .outerjoin(jobs, jobs.c.id == backtest_runs.c.job_id)
                .where(backtest_runs.c.strategy_version_id == version_id)
                .order_by(backtest_runs.c.created_at.desc(), backtest_runs.c.id.desc())
                .limit(1)
            ).first()
        if row is None:
            return {
                "status": "not_started",
                "run_id": None,
                "run_status": None,
                "job_id": None,
                "job_status": None,
            }
        result = row_dict(row)
        result["status"] = _project_backtest_status(
            result.get("run_status"),
            result.get("job_status"),
        )
        return result

    def _latest_health(self, version_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            return load_production_health_gate(connection, version_id)

    def _forward_evidence(self, version: dict[str, Any]) -> dict[str, Any]:
        if version.get("promotion_stage") not in {"paper", "recommendation_enabled"}:
            return {
                "status": "not_started",
                "passed": False,
                "reasons": ["策略尚未进入隔离模拟盘"],
            }
        # The product endpoint is a read-only projection.  PromotionStore's
        # operational evaluator intentionally freezes and replaces a drifted
        # paper stage; calling it from a GET would make page refresh mutate the
        # lifecycle.  Reuse only its read-only evidence collector and project
        # contract drift as blocked for the scheduler to reconcile.
        try:
            with self.engine.connect() as connection:
                version_row = connection.execute(
                    select(strategy_versions).where(
                        strategy_versions.c.id == str(version["id"])
                    )
                ).first()
                if version_row is None:
                    raise KeyError(str(version["id"]))
                gate = connection.execute(
                    select(strategy_forward_gates).where(
                        strategy_forward_gates.c.strategy_version_id
                        == str(version["id"])
                    )
                ).first()
                if gate is None:
                    return _insufficient_evidence(
                        ["forward evidence gate is not pre-registered"]
                    )
                criteria = _require_forward_gate_criteria(version_row, gate)
                stage = connection.execute(
                    select(strategy_promotion_stages)
                    .where(
                        strategy_promotion_stages.c.strategy_version_id
                        == str(version["id"]),
                        strategy_promotion_stages.c.status.in_(
                            ("active", "awaiting_simulation")
                        ),
                    )
                    .order_by(strategy_promotion_stages.c.stage_index.desc())
                    .limit(1)
                ).first()
                if stage is None:
                    return _insufficient_evidence(["no active paper stage exists"])
                if stage.simulation_portfolio_id is None:
                    return _insufficient_evidence(
                        ["paper stage has no isolated simulation account; evidence is zero"]
                    )
                portfolio = connection.execute(
                    select(simulation_portfolios).where(
                        simulation_portfolios.c.id == stage.simulation_portfolio_id
                    )
                ).first()
                if portfolio is None:
                    return _insufficient_evidence(
                        ["paper stage simulation account does not exist"]
                    )
                try:
                    SimulationStore._require_current_source_contract(
                        connection,
                        portfolio,
                    )
                except ValueError as exc:
                    return _insufficient_evidence(
                        [
                            "paper stage source contract drift requires lifecycle reconciliation",
                            str(exc),
                        ],
                        contract_drift=True,
                    )
                evidence = self.promotions._collect_evidence(
                    connection,
                    stage,
                    portfolio,
                    version_row,
                )
            checks = _forward_gate_checks(
                horizon=str(version_row.horizon_profile),
                gate=gate,
                evidence=evidence,
            )
        except (KeyError, ValueError) as exc:
            return {
                "status": "blocked",
                "passed": False,
                "reasons": [str(exc)],
            }
        failures = [name for name, result in checks.items() if not result["passed"]]
        common = {
            "checks": checks,
            "evidence": evidence,
            "stage_id": str(stage.id),
            "criteria_json": criteria,
            "criteria_sha256": str(gate.criteria_sha256),
        }
        if str(version_row.horizon_profile) == LONG_1_3Y:
            common["maturity"] = _long_maturity_projection(evidence)
        if failures:
            return _insufficient_evidence(
                [
                    f"{name} below/above the pre-registered threshold"
                    for name in failures
                ],
                **common,
            )
        return {
            "status": "ok",
            "passed": True,
            "reasons": [],
            **common,
            "contract_version": PROMOTION_CONTRACT_VERSION,
        }

    def _horizon_card(
        self,
        horizon: str,
        *,
        now: datetime,
        freshness_requirement: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = _HORIZON_UI[horizon]
        freshness_requirement = freshness_requirement or advice_session_requirement(
            getattr(self, "data_root", None),
            now=now,
        )
        required_signal_date = _date_value(
            freshness_requirement.get("required_signal_date")
        )
        version = self._latest_version(horizon)
        if version is None:
            return {
                "horizon": horizon,
                **meta,
                "stage": "research",
                "stage_label": "研究中",
                "strategy": None,
                "health": "insufficient_evidence",
                "evidence": {"status": "not_started", "passed": False},
                "backtest": {"status": "not_started"},
                "is_investment_advice": False,
                "recommendation_freshness": {
                    "status": "not_authorized",
                    "passed": False,
                    "required_signal_date": (
                        required_signal_date.isoformat()
                        if required_signal_date
                        else None
                    ),
                    "observed_signal_date": None,
                    "reason": "该周期尚无可发布的正式策略",
                },
                "data_cutoff": None,
                "signals": [],
                "action": "NO_ACTION",
                "veto_reasons": ["该周期尚未生成策略版本"],
            }

        health = self._latest_health(str(version["id"]))
        health_status = str((health or {}).get("health_status") or "insufficient_evidence")
        promotion_stage = str(version.get("promotion_stage") or "")
        backtest = self._latest_backtest(str(version["id"]))
        evidence = self._forward_evidence(version)
        stage, stage_label = _project_stage(
            version_status=version.get("status"),
            promotion_stage=promotion_stage,
            health_status=health_status,
            backtest_status=str(backtest["status"]),
            forward_gate_passed=evidence.get("passed") is True,
        )
        governance_authorized = (
            version.get("status") == "approved"
            and promotion_stage == "recommendation_enabled"
            and stage == "verified"
            and backtest["status"] == "succeeded"
            and evidence.get("passed") is True
        )
        signals: list[dict[str, Any]] = []
        data_cutoff: str | None = None
        review_sessions = int(
            version.get("review_interval_sessions")
            or _HORIZON_UI[horizon]["review_sessions"]
        )
        if stage == "simulation_validation":
            projection = self.simulations.paper_target_for_strategy_version(str(version["id"]))
            data_cutoff = projection.get("signal_date")
            signals = [
                self._paper_signal(
                    item,
                    projection=projection,
                    horizon=horizon,
                    review_sessions=review_sessions,
                )
                for item in projection.get("targets") or []
            ]
        elif governance_authorized:
            signals, data_cutoff = self._verified_signals(
                str(version["id"]),
                horizon=horizon,
                review_sessions=review_sessions,
                allow_paused=bool(version.get("activation_cutover_fallback")),
            )

        if governance_authorized:
            recommendation_freshness = assess_signal_freshness(
                required_signal_date=required_signal_date,
                observed_signal_date=data_cutoff,
                requirement_reason=freshness_requirement.get("reason"),
            )
            if recommendation_freshness["passed"] is not True:
                signals = []
        else:
            recommendation_freshness = {
                "status": "not_authorized",
                "passed": False,
                "required_signal_date": (
                    required_signal_date.isoformat() if required_signal_date else None
                ),
                "observed_signal_date": data_cutoff,
                "reason": "策略尚未通过正式荐股治理门槛",
            }
        investment_authorized = (
            governance_authorized
            and recommendation_freshness["passed"] is True
        )

        if not signals:
            card_action = "NO_ACTION"
        elif any(item["action"] in {"BUY", "ADD"} for item in signals):
            card_action = "BUY"
        elif any(item["action"] in {"EXIT", "REDUCE"} for item in signals):
            card_action = "REDUCE"
        else:
            card_action = "HOLD"
        veto_reasons = []
        if not governance_authorized:
            veto_reasons.append(
                "前向证据未成熟，仅展示模拟验证" if stage == "simulation_validation"
                else f"当前阶段为{stage_label}"
            )
        elif recommendation_freshness["passed"] is not True:
            veto_reasons.append(str(recommendation_freshness["reason"]))
        if backtest["status"] == "failed":
            veto_reasons.append("最近一次回测或执行任务失败，不能作为荐股依据")
        elif backtest["status"] == "succeeded" and version.get("status") != "approved":
            veto_reasons.append("历史回测已完成，仍需治理审批和严格前向模拟")
        if promotion_stage == "recommendation_enabled" and evidence.get("passed") is not True:
            veto_reasons.append("前向门槛证据不可用，已按失败关闭原则停止荐股")
        if stage in {"restricted", "suspended", "retired"}:
            veto_reasons.append("策略健康状态禁止新增买入")
        if not signals and (
            not governance_authorized
            or recommendation_freshness["passed"] is True
        ):
            veto_reasons.append("当前没有满足成本、风险和有效性门槛的新机会")
        return {
            "horizon": horizon,
            **meta,
            "stage": stage,
            "stage_label": stage_label,
            "strategy": {
                "id": str(version["id"]),
                "name": version["strategy_name"],
                "version": int(version["version"]),
                "status": str(version.get("status") or ""),
                "promotion_stage": promotion_stage or None,
                "rules_sha256": (version.get("config") or {}).get(
                    "strategy_rules_sha256"
                ),
                "activation_cutover_fallback": bool(
                    version.get("activation_cutover_fallback")
                ),
            },
            "health": health_status,
            "health_snapshot": health,
            "evidence": evidence,
            "backtest": backtest,
            "is_investment_advice": investment_authorized,
            "recommendation_freshness": recommendation_freshness,
            "data_cutoff": data_cutoff,
            "signals": signals,
            "action": card_action if investment_authorized else "NO_ACTION",
            "simulation_action": card_action if stage == "simulation_validation" else None,
            "veto_reasons": veto_reasons,
        }

    def _paper_signal(
        self,
        item: dict[str, Any],
        *,
        projection: dict[str, Any],
        horizon: str,
        review_sessions: int,
    ) -> dict[str, Any]:
        weight = float(item.get("target_weight") or item.get("weight") or 0.0)
        previous = float(item.get("previous_target_weight") or 0.0)
        effective = _date_value(projection.get("trade_date"))
        portfolio = dict(projection.get("simulation_portfolio") or {})
        review = self._review_projection(
            effective_date=effective,
            review_sessions=review_sessions,
            dataset=str(portfolio.get("daily_dataset") or "") or None,
        )
        return {
            "instrument": str(item.get("instrument") or ""),
            "action": _action(item.get("action"), previous_weight=previous, weight=weight),
            "target_weight": weight,
            "target_position_quantity": None,
            "trade_quantity": None,
            "quantity_source": "awaiting_account_order_plan",
            "effective_date": effective.isoformat() if effective else None,
            "validity_sessions": _HORIZON_UI[horizon]["validity_sessions"],
            **review,
            "holding_age_sessions": None,
            "holding_age_source": "awaiting_authoritative_account_lots",
            "reason": {
                "summary": "冻结策略目标相对上一模拟目标发生变化",
                "signals": [str(item.get("reason") or "模拟目标变化")],
            },
            "risks": ["尚未达到该周期的严格前向荐股门槛"],
            "invalidation": ["排名、交易资格、成本后优势或策略规则失效"],
            "evidence_state": "simulation_validation",
        }

    def _verified_signals(
        self,
        version_id: str,
        *,
        horizon: str,
        review_sessions: int,
        allow_paused: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        with self.engine.connect() as connection:
            snapshot = connection.execute(
                select(recommendation_snapshots)
                .join(
                    recommendation_portfolios,
                    recommendation_portfolios.c.id == recommendation_snapshots.c.portfolio_id,
                )
                .where(
                    recommendation_portfolios.c.strategy_version_id == version_id,
                    recommendation_portfolios.c.status.in_(
                        ("active", "paused") if allow_paused else ("active",)
                    ),
                    recommendation_snapshots.c.status == "succeeded",
                )
                .order_by(
                    recommendation_snapshots.c.as_of_date.desc(),
                    recommendation_snapshots.c.created_at.desc(),
                )
                .limit(1)
            ).first()
            if snapshot is None:
                return [], None
            holdings = connection.execute(
                select(recommendation_holdings)
                .where(recommendation_holdings.c.snapshot_id == snapshot.id)
                .order_by(recommendation_holdings.c.weight.desc())
            ).all()
        effective = _date_value(snapshot.effective_date)
        account_actions = {
            str(item.get("instrument") or "").upper(): dict(item)
            for item in (
                dict(snapshot.account_actions_json or {}).get("items") or []
            )
            if isinstance(item, dict) and str(item.get("instrument") or "")
        }
        changes = {
            str(item.get("instrument") or "").upper(): dict(item)
            for item in (
                dict(getattr(snapshot, "snapshot_json", None) or {}).get("changes")
                or []
            )
            if isinstance(item, dict) and str(item.get("instrument") or "")
        }
        holding_by_instrument = {
            str(row.instrument).upper(): row for row in holdings
        }
        removed_instruments = {
            instrument
            for instrument in set(account_actions) | set(changes)
            if _action(
                (account_actions.get(instrument) or {}).get("action")
                or (changes.get(instrument) or {}).get("action"),
                previous_weight=float(
                    (changes.get(instrument) or {}).get("previous_weight")
                    or (account_actions.get(instrument) or {}).get("previous_weight")
                    or 0.0
                ),
                weight=float(
                    (changes.get(instrument) or {}).get("target_weight")
                    or (changes.get(instrument) or {}).get("weight")
                    or (account_actions.get(instrument) or {}).get("target_weight")
                    or (account_actions.get(instrument) or {}).get("weight")
                    or 0.0
                ),
            )
            in {"REDUCE", "EXIT"}
        }
        projected_instruments = [
            *holding_by_instrument,
            *sorted(removed_instruments - set(holding_by_instrument)),
        ]
        review = self._review_projection(
            effective_date=effective,
            review_sessions=review_sessions,
            dataset=str(snapshot.dataset or "") or None,
        )
        signals = []
        for instrument in projected_instruments:
            row = holding_by_instrument.get(instrument)
            change = changes.get(instrument) or {}
            account_action = account_actions.get(instrument) or {}
            weight = float(
                row.weight
                if row is not None
                else change.get("target_weight")
                or change.get("weight")
                or account_action.get("target_weight")
                or account_action.get("weight")
                or 0.0
            )
            previous = float(
                row.previous_weight
                if row is not None
                else change.get("previous_weight")
                or account_action.get("previous_weight")
                or 0.0
            )
            action = _action(
                account_action.get("action")
                or change.get("action")
                or (row.action if row is not None else None),
                previous_weight=previous,
                weight=weight,
            )
            if row is None and action not in {"REDUCE", "EXIT"}:
                continue
            reason = str(
                row.reason
                if row is not None
                else change.get("reason")
                or account_action.get("reason")
                or "策略目标已减仓或退出"
            )
            signals.append(
                {
                    "instrument": str(row.instrument) if row is not None else instrument,
                    "action": action,
                    "target_weight": weight,
                    "target_position_quantity": None,
                    "trade_quantity": None,
                    "quantity_source": "awaiting_unified_account_order_plan",
                    "effective_date": effective.isoformat() if effective else None,
                    "validity_sessions": _HORIZON_UI[horizon]["validity_sessions"],
                    **review,
                    "holding_age_sessions": None,
                    "holding_age_source": "awaiting_authoritative_account_lots",
                    "reason": {
                        "summary": reason,
                        "signals": [reason],
                    },
                    "risks": ["市场、流动性、模型漂移与成本均可能使信号失效"],
                    "invalidation": ["策略排名、基本面、估值或风险规则失效"],
                    "evidence_state": "verified_forward",
                }
            )
        return signals, snapshot.as_of_date.isoformat()

    def _unified_account(
        self,
        *,
        verified_cards: list[dict[str, Any]],
        formal_cards: list[dict[str, Any]],
        onboarding_required: bool,
        freshness_requirement: dict[str, Any],
    ) -> dict[str, Any]:
        if onboarding_required:
            return {
                "status": "onboarding_required",
                "action": "NO_ACTION",
                "reason": "请先填写模拟本金和证券权限",
                "targets": [],
                "trades": [],
            }
        if not verified_cards and formal_cards:
            stale_horizons = [str(item["horizon"]) for item in formal_cards]
            reasons = [
                str(dict(item.get("recommendation_freshness") or {}).get("reason"))
                for item in formal_cards
                if dict(item.get("recommendation_freshness") or {}).get("reason")
            ]
            return {
                "status": "waiting_for_current_recommendations",
                "action": "NO_ACTION",
                "reason": (
                    "正式策略等待最新闭市交易日推荐快照："
                    + "；".join(reasons)
                ),
                "verified_horizons": [],
                "stale_horizons": stale_horizons,
                "freshness": freshness_requirement,
                "targets": [],
                "trades": [],
            }
        if not verified_cards:
            return {
                "status": "waiting_for_verified_horizons",
                "action": "NO_ACTION",
                "reason": "三个周期均仍处于研究或模拟验证，全部预算保留现金",
                "verified_horizons": [],
                "freshness": freshness_requirement,
                "targets": [],
                "trades": [],
            }
        verified_horizons = [str(item["horizon"]) for item in verified_cards]
        missing_horizons = [item for item in HORIZONS if item not in verified_horizons]
        primary = self._primary_account_binding()
        if primary["status"] != "ready":
            return {
                "status": "waiting_for_primary_account",
                "action": "NO_ACTION",
                "reason": str(primary["reason"]),
                "verified_horizons": verified_horizons,
                "missing_horizons": missing_horizons,
                "targets": [],
                "trades": [],
            }
        account_id = str(primary["account_id"])
        portfolio_id = str(primary["portfolio_id"])
        with self.engine.connect() as connection:
            row = connection.execute(
                select(account_netting_plans)
                .where(account_netting_plans.c.account_id == account_id)
                .order_by(
                    account_netting_plans.c.decision_date.desc(),
                    account_netting_plans.c.created_at.desc(),
                )
                .limit(1)
            ).first()
        if row is None:
            return {
                "status": "waiting_for_netting",
                "action": "NO_ACTION",
                "reason": "已验证周期等待生成统一账户净额计划；其余预算保留现金",
                "verified_horizons": verified_horizons,
                "missing_horizons": missing_horizons,
                "targets": [],
                "trades": [],
            }
        plan = dict(row.plan_json or {})
        seal_error = self._validate_netting_plan_seal(
            row,
            plan=plan,
            account_id=account_id,
            portfolio_id=portfolio_id,
        )
        if seal_error is not None:
            return {
                "status": "waiting_for_current_netting",
                "action": "NO_ACTION",
                "reason": seal_error,
                "verified_horizons": verified_horizons,
                "missing_horizons": missing_horizons,
                "plan_id": str(row.id),
                "targets": [],
                "trades": [],
            }
        plan_freshness = assess_netting_plan_freshness(
            required_signal_date=_date_value(
                freshness_requirement.get("required_signal_date")
            ),
            inputs_as_of=row.inputs_as_of,
            plan=plan,
            requirement_reason=freshness_requirement.get("reason"),
        )
        if plan_freshness["status"] == "member_health_authority_missing":
            plan_freshness = assess_netting_plan_freshness(
                required_signal_date=_date_value(
                    freshness_requirement.get("required_signal_date")
                ),
                inputs_as_of=row.inputs_as_of,
                plan=plan,
                authoritative_health_snapshots=(
                    self._authoritative_member_health_snapshots(
                        account_id
                    )
                ),
                requirement_reason=freshness_requirement.get("reason"),
            )
        if plan_freshness["passed"] is not True:
            return {
                "status": "waiting_for_current_netting",
                "action": "NO_ACTION",
                "reason": str(plan_freshness["reason"]),
                "verified_horizons": verified_horizons,
                "missing_horizons": missing_horizons,
                "plan_id": str(row.id),
                "decision_date": row.decision_date.isoformat(),
                "inputs_as_of": row.inputs_as_of.isoformat(),
                "freshness": plan_freshness,
                "targets": [],
                "trades": [],
            }
        execution = self._unified_execution_facts(
            str(row.id),
            account_id=account_id,
            portfolio_id=portfolio_id,
        )
        if execution.get("status") != "ready":
            return {
                "status": "waiting_for_account_execution",
                "action": "NO_ACTION",
                "reason": str(
                    execution.get("reason")
                    or "统一模拟账户执行证据尚未就绪"
                ),
                "verified_horizons": verified_horizons,
                "missing_horizons": missing_horizons,
                "plan_id": str(row.id),
                "decision_date": row.decision_date.isoformat(),
                "inputs_as_of": row.inputs_as_of.isoformat(),
                "freshness": plan_freshness,
                "targets": [],
                "trades": [],
            }
        instrument_facts = dict(execution.get("items") or {})
        required_fact_instruments = {
            str(instrument).upper()
            for instrument in set(plan.get("net_targets") or {})
            | set(plan.get("net_trades") or {})
        }
        missing_execution_facts = sorted(
            required_fact_instruments - set(instrument_facts)
        )
        if missing_execution_facts:
            return {
                "status": "waiting_for_account_execution",
                "action": "NO_ACTION",
                "reason": "统一账户执行批次缺少净额目标明细",
                "verified_horizons": verified_horizons,
                "missing_horizons": missing_horizons,
                "plan_id": str(row.id),
                "decision_date": row.decision_date.isoformat(),
                "inputs_as_of": row.inputs_as_of.isoformat(),
                "freshness": plan_freshness,
                "missing_execution_fact_instruments": missing_execution_facts,
                "targets": [],
                "trades": [],
            }
        trades = [
            {
                "instrument": instrument,
                **dict(value),
                **dict(instrument_facts.get(str(instrument).upper()) or {}),
            }
            for instrument, value in sorted((plan.get("net_trades") or {}).items())
        ]
        targets = [
            {
                "instrument": instrument,
                **dict(value),
                **dict(instrument_facts.get(str(instrument).upper()) or {}),
            }
            for instrument, value in sorted((plan.get("net_targets") or {}).items())
        ]
        account_action = _unified_account_action(trades)
        return {
            "status": "ready",
            "action": account_action,
            "reason": (
                _unified_no_action_reason(instrument_facts)
                if account_action == "NO_ACTION"
                else None
            ),
            "verified_horizons": verified_horizons,
            "missing_horizons": missing_horizons,
            "missing_horizon_budget_policy": "remain_in_cash",
            "plan_id": str(row.id),
            "decision_date": row.decision_date.isoformat(),
            "inputs_as_of": row.inputs_as_of.isoformat(),
            "freshness": plan_freshness,
            "simulation_portfolio_id": execution.get("portfolio_id"),
            "execution_batch_id": execution.get("batch_id"),
            "decision_event_id": execution.get("decision_event_id"),
            "holding_age_as_of": execution.get("holding_age_as_of"),
            "cash_weight": float(plan.get("cash_weight") or 0.0),
            "targets": targets,
            "trades": trades,
            "instrument_facts": instrument_facts,
            "strategy_contributions": plan.get("strategy_contributions") or {},
            "accounting_rule": "三周期先独立出目标，再在一个账户层净额化",
        }
