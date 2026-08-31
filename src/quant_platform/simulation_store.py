from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from math import isfinite
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import delete, func, insert, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

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
    simulation_cash_event_allocations,
    simulation_cash_events,
    simulation_cash_flows,
    simulation_cash_lots,
    simulation_cash_reservations,
    simulation_corporate_events,
    simulation_day_attributions,
    simulation_dividend_actions,
    simulation_dividend_entitlements,
    simulation_events,
    simulation_external_flows,
    simulation_fee_adjustments,
    simulation_fills,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    simulation_position_lots,
    simulation_position_reservations,
    simulation_positions,
    simulation_security_events,
    strategy_allocation_members,
    strategy_allocations,
    strategy_forward_gates,
    strategy_pairs,
    strategy_promotion_stages,
    strategy_versions,
)
from quant_data.execution_contract import (
    QLIB_ORDER_PLAN_FORMAT_VERSION,
    QLIB_OUTPUT_MANIFEST_VERSION,
    require_daily_qlib_contract,
    require_minute_execution_contract,
    require_native_daily_execution_controls,
    require_next_bar_execution,
    require_strategy_execution_contract,
)

from .account_netting import primary_position_inventory_evidence
from .account_risk_state import (
    LEDGER_POLICY_RISK_VERSION,
    RISK_SCOPE,
    ledger_policy_risk_inputs,
)
from .allocation_store import THREE_HORIZON_ALLOCATION_CUTOVER_LOCK_KEY
from .corporate_actions import corporate_actions_sha256
from .cost_model import (
    KNOWN_COST_SCHEDULE_VERSIONS,
    CostModelConfig,
    CostScheduleBook,
    infer_cn_asset_type,
)
from .execution_algorithms import execution_time_slots, normalize_execution_policy
from .forward_only_rehabilitation import EVIDENCE_MODE_REPLAY, require_qualification
from .member_risk_gate import (
    load_allocation_risk_state,
    load_strategy_risk_state,
)
from .paper_policy_state import (
    previous_snapshot_from_paper_batch,
    validate_paper_policy_state,
)
from .qlib_workflow import require_qlib_workflow_identity
from .safe_mode import SafeModeStore
from .simulation_engine import (
    SIMULATION_ENGINE_VERSION,
    execute_atomic_pair_day,
    execute_simulation_day,
)
from .simulation_order_state import (
    OPEN_STATUSES,
    ORDER_PLAN_MODEL_VERSION,
    STATUS_CANCELLED,
    STATUS_EXPIRED,
    STATUS_OPEN,
    STATUS_PLANNED,
    apply_order_plan,
)
from .unitized_performance import (
    UNITIZED_PERFORMANCE_VERSION,
    benchmark_relative_statistics,
    chain_unitized_day,
    unitized_drawdown_recovery,
    unitized_return_statistics,
    xirr,
)

# Batch-failure messages that indicate ledger-integrity corruption (as
# opposed to data/capacity rejections); they engage the platform safe mode.
LEDGER_INTEGRITY_ERROR_MARKERS = (
    "cash conservation failed",
    "cash flows do not reconcile",
    "would create negative cash",
)

ALLOCATION_SOURCE_REBIND_VERSION = "simulation-allocation-source-rebind-v1"
ACCOUNT_ORDER_DECISION_VERSION = "simulation-account-order-decision-v1"
ACCOUNT_ORDER_DECISION_EVENT_TYPE = "account_order_plan_decision_only"
ACCOUNT_DECISION_VALUATION_VERSION = "simulation-account-decision-valuation-v1"


def _now() -> datetime:
    return datetime.now(UTC)


def _benchmark_chain_day(
    evidence: dict[str, Any] | None,
    *,
    benchmark: str,
    signal_date: date,
    trade_date: date,
    execution_frequency: str,
    dataset_identity_sha256: str,
    dataset_lineage_id: str,
    prior_nav: Any | None,
) -> dict[str, Any]:
    if benchmark == "CASH":
        prior_wealth = (
            float(prior_nav.benchmark_wealth)
            if prior_nav is not None and prior_nav.benchmark_wealth is not None
            else 1.0
        )
        return {
            "status": "cash_baseline",
            "benchmark_close": 1.0,
            "benchmark_return": 0.0,
            "benchmark_wealth": prior_wealth,
            "evidence_sha256": _canonical_hash(
                {
                    "benchmark": "CASH",
                    "trade_date": trade_date.isoformat(),
                    "return": 0.0,
                }
            ),
        }
    if evidence is None:
        return {
            "status": "benchmark_evidence_missing",
            "benchmark_close": None,
            "benchmark_return": None,
            "benchmark_wealth": None,
            "evidence_sha256": None,
        }
    payload = {
        key: evidence.get(key)
        for key in (
            "contract_version",
            "instrument",
            "baseline_date",
            "baseline_close",
            "trade_date",
            "close",
            "dataset_identity_sha256",
            "dataset_lineage_id",
            "frequency",
        )
    }
    expected = {
        "contract_version": SIMULATION_BENCHMARK_EVIDENCE_VERSION,
        "instrument": benchmark,
        "baseline_date": signal_date.isoformat(),
        "trade_date": trade_date.isoformat(),
        "dataset_identity_sha256": dataset_identity_sha256,
        "dataset_lineage_id": dataset_lineage_id,
        "frequency": execution_frequency,
    }
    mismatches = [
        key for key, value in expected.items() if str(payload.get(key) or "") != value
    ]
    if mismatches:
        raise ValueError(
            "simulation benchmark evidence does not match the bound contract: "
            + ", ".join(mismatches)
        )
    baseline_close = float(payload["baseline_close"])
    current_close = float(payload["close"])
    if (
        not isfinite(baseline_close)
        or baseline_close <= 0
        or not isfinite(current_close)
        or current_close <= 0
    ):
        raise ValueError("simulation benchmark evidence requires positive finite closes")
    observed_hash = str(evidence.get("evidence_sha256") or "")
    if not _is_sha256(observed_hash) or observed_hash != _canonical_hash(payload):
        raise ValueError("simulation benchmark evidence hash is invalid")

    if prior_nav is not None:
        if (
            prior_nav.benchmark_close is None
            or prior_nav.benchmark_wealth is None
            or prior_nav.trade_date >= trade_date
        ):
            return {
                "status": "unavailable_broken_benchmark_chain",
                "benchmark_close": current_close,
                "benchmark_return": None,
                "benchmark_wealth": None,
                "evidence_sha256": observed_hash,
            }
        prior_close = float(prior_nav.benchmark_close)
        prior_wealth = float(prior_nav.benchmark_wealth)
        if (
            not isfinite(prior_close)
            or prior_close <= 0
            or not isfinite(prior_wealth)
            or prior_wealth < 0
        ):
            raise ValueError("persisted simulation benchmark chain is invalid")
    else:
        prior_close = baseline_close
        prior_wealth = 1.0
    daily_return = current_close / prior_close - 1.0
    wealth = prior_wealth * (1.0 + daily_return)
    if not isfinite(daily_return) or daily_return < -1.0 or not isfinite(wealth):
        raise ValueError("simulation benchmark return chain is invalid")
    return {
        "status": "ok",
        "benchmark_close": current_close,
        "benchmark_return": daily_return,
        "benchmark_wealth": wealth,
        "evidence_sha256": observed_hash,
    }


SIMULATION_SOURCE_TYPES = frozenset({"recommendation", "strategy_version", "allocation"})
SIMULATION_EXECUTION_ADAPTERS = frozenset({"long_only", "pair"})
SIMULATION_EXECUTION_FREQUENCIES = frozenset({"day", "1min", "5min"})
SIMULATION_EXECUTION_SEMANTICS_VERSION = "simulation-execution-semantics-v1"
SIMULATION_BENCHMARK_EVIDENCE_VERSION = "simulation-benchmark-evidence-v1"
SIMULATION_SETTLEMENT_CALENDAR_EVIDENCE_VERSION = (
    "simulation-settlement-calendar-evidence-v1"
)
SIMULATION_SETTLEMENT_CALENDAR_BINDING_VERSION = (
    "simulation-settlement-calendar-binding-v1"
)
SETTLEMENT_CALENDAR_RELATIVE_PATH = "metadata/known_trading_calendar.parquet"


class ExecutionDataNotReadyError(ValueError):
    """The sealed signal exists, but its immutable execution day does not yet."""

    def __init__(self, trade_date: date) -> None:
        self.trade_date = trade_date
        super().__init__(
            "governed execution data is awaiting publication for "
            f"{trade_date.isoformat()}"
        )


def _annotate_position_holding_ages(
    positions: list[dict[str, Any]],
    lots: list[dict[str, Any]],
    *,
    calendar_days: set[date],
    as_of_date: date,
    require_complete_age: bool,
) -> list[dict[str, Any]]:
    """Project durable lot acquisition dates into Qlib trading-session ages.

    This is deliberately a pure read-side projection.  Re-running a paper
    order-plan after a worker restart therefore derives the same age from the
    simulation ledger instead of trusting ephemeral policy state.  A live
    position is proven only when its positive lots account for the full
    position quantity and every lot has an acquisition date covered by the
    selected daily Qlib calendar.
    """

    normalized_calendar = sorted(set(calendar_days))
    if as_of_date not in normalized_calendar:
        raise ValueError(
            "paper holding-age projection requires the signal date in the "
            "bound Qlib trading calendar"
        )
    calendar_index = {day: index for index, day in enumerate(normalized_calendar)}
    lots_by_instrument: dict[str, list[dict[str, Any]]] = {}
    for raw_lot in lots:
        lot = dict(raw_lot)
        instrument = str(lot.get("instrument") or "")
        if instrument and int(lot.get("quantity") or 0) > 0:
            lots_by_instrument.setdefault(instrument, []).append(lot)

    projected: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for raw_position in positions:
        position = dict(raw_position)
        instrument = str(position.get("instrument") or "")
        quantity = int(position.get("quantity") or 0)
        is_live_long = (
            str(position.get("position_side") or "long") == "long" and quantity > 0
        )
        if not is_live_long:
            projected.append(position)
            continue

        instrument_lots = lots_by_instrument.get(instrument, [])
        reasons: list[str] = []
        if not instrument_lots:
            reasons.append("no_position_lots")
        elif sum(int(item.get("quantity") or 0) for item in instrument_lots) != quantity:
            reasons.append("lot_quantity_mismatch")

        acquired_dates: list[date] = []
        for lot in instrument_lots:
            acquired_at = lot.get("acquired_at")
            if isinstance(acquired_at, datetime):
                acquired_at = acquired_at.date()
            elif isinstance(acquired_at, str):
                try:
                    acquired_at = date.fromisoformat(acquired_at)
                except ValueError:
                    acquired_at = None
            if not isinstance(acquired_at, date):
                reasons.append("acquired_at_unknown")
                continue
            if acquired_at > as_of_date:
                reasons.append("acquired_after_signal_date")
            elif acquired_at not in calendar_index:
                reasons.append("acquired_at_outside_qlib_calendar")
            acquired_dates.append(acquired_at)

        earliest = min(acquired_dates) if acquired_dates else None
        if reasons or earliest is None or earliest not in calendar_index:
            evidence = {
                "status": "unproven",
                "calendar": "qlib_day",
                "as_of_date": as_of_date.isoformat(),
                "earliest_acquired_at": earliest.isoformat() if earliest else None,
                "reasons": sorted(set(reasons or ["acquired_at_unproven"])),
            }
            position["holding_age_sessions"] = None
            position["holding_age_evidence"] = evidence
            incomplete.append(f"{instrument}({','.join(evidence['reasons'])})")
        else:
            age = calendar_index[as_of_date] - calendar_index[earliest]
            position["holding_age_sessions"] = age
            position["holding_age_evidence"] = {
                "status": "proven",
                "calendar": "qlib_day",
                "as_of_date": as_of_date.isoformat(),
                "earliest_acquired_at": earliest.isoformat(),
                "lot_count": len(instrument_lots),
                "position_quantity": quantity,
            }
        projected.append(position)

    if require_complete_age and incomplete:
        raise ValueError(
            "active holding-age exit policy cannot prove simulation lot acquisition "
            "dates from the bound Qlib calendar: " + "; ".join(sorted(incomplete))
        )
    return projected
PAPER_TARGET_PROJECTION_VERSION = "paper-target-projection-v1"
VWAP_PROFILE_METHOD = "qlib-historical-average-volume-v1"
SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
CASH_OPENING_BALANCE_AT = datetime(1970, 1, 1, tzinfo=UTC)


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_account_netting_primary_evidence(
    connection: Any,
    *,
    portfolio: Any,
    plan_json: dict[str, Any],
) -> None:
    """Reject a netting plan if its authoritative account snapshot changed."""

    raw_evidence = plan_json.get("input_evidence") or {}
    if not isinstance(raw_evidence, dict):
        raise ValueError("account netting input evidence is invalid")
    primary = raw_evidence.get("primary_account")
    if primary is None:
        return
    if not isinstance(primary, dict):
        raise ValueError("account netting primary-capital evidence is invalid")
    expected_nav = float(primary.get("nav") or 0.0)
    if (
        str(primary.get("portfolio_id") or "") != str(portfolio.id)
        or str(primary.get("source_id") or "") != str(portfolio.source_id)
        or not isfinite(expected_nav)
        or expected_nav <= 0
        or abs(expected_nav - float(portfolio.nav)) > 1e-6
    ):
        raise ValueError("account NAV changed after netting; rebuild the order plan")

    expected_positions_sha256 = str(primary.get("positions_sha256") or "")
    if not expected_positions_sha256:
        # Legacy primary-capital plans predate exact sleeve inventory.  Their
        # existing NAV seal remains valid, but no position identity is invented.
        return
    if not _is_sha256(expected_positions_sha256):
        raise ValueError("account netting position-inventory evidence is invalid")
    raw_inventory = raw_evidence.get("primary_position_inventory")
    if not isinstance(raw_inventory, dict):
        raise ValueError("account netting position-inventory evidence is missing")
    stored_inventory = dict(raw_inventory)
    stored_sha256 = str(stored_inventory.pop("positions_sha256", ""))
    if (
        stored_sha256 != expected_positions_sha256
        or _canonical_hash(stored_inventory) != expected_positions_sha256
    ):
        raise ValueError("account netting position-inventory seal is invalid")
    policy_sha256 = str(primary.get("sleeve_inventory_policy_sha256") or "")
    if not _is_sha256(policy_sha256):
        raise ValueError("account netting sleeve-inventory policy seal is invalid")
    raw_sleeve = raw_evidence.get("sleeve_inventory_allocation")
    if not isinstance(raw_sleeve, dict):
        raise ValueError("account netting sleeve-inventory allocation is missing")
    sleeve = dict(raw_sleeve)
    allocation_sha256 = str(sleeve.pop("allocation_sha256", ""))
    if (
        str(sleeve.get("policy_sha256") or "") != policy_sha256
        or not _is_sha256(allocation_sha256)
        or _canonical_hash(sleeve) != allocation_sha256
    ):
        raise ValueError("account netting sleeve-inventory allocation seal is invalid")
    expected_account_weights = {
        str(instrument): float(values.get("account_weight") or 0.0)
        for instrument, values in dict(stored_inventory.get("positions") or {}).items()
        if float(values.get("account_weight") or 0.0) > 1e-9
    }
    if dict(sleeve.get("actual_account_weights") or {}) != expected_account_weights:
        raise ValueError("account netting sleeve inventory differs from primary positions")
    member_inventory = plan_json.get("member_current_account_weights") or {}
    if not isinstance(member_inventory, dict):
        raise ValueError("account netting exact member inventory is invalid")
    attributed_account_weights: dict[str, float] = {}
    for raw_book in member_inventory.values():
        if not isinstance(raw_book, dict):
            raise ValueError("account netting exact member inventory is invalid")
        for instrument, weight in raw_book.items():
            attributed_account_weights[str(instrument)] = (
                attributed_account_weights.get(str(instrument), 0.0) + float(weight)
            )
    if set(attributed_account_weights) != set(expected_account_weights) or any(
        abs(attributed_account_weights[instrument] - expected_account_weights[instrument])
        > 1e-9
        for instrument in expected_account_weights
    ):
        raise ValueError("account netting member inventory does not reconcile to positions")

    live_rows = connection.execute(
        select(simulation_positions)
        .where(simulation_positions.c.portfolio_id == portfolio.id)
        .order_by(simulation_positions.c.instrument)
    ).all()
    live_inventory, _actual_weights = primary_position_inventory_evidence(
        portfolio_id=str(portfolio.id),
        nav=float(portfolio.nav),
        positions=[dict(row._mapping) for row in live_rows],
    )
    if live_inventory["positions_sha256"] != expected_positions_sha256:
        raise ValueError(
            "account positions changed after netting; rebuild the order plan"
        )


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def build_account_order_decision_event(
    *,
    portfolio_id: str,
    account_netting_plan_id: str,
    target_version: str,
    actor: str,
    signal_date: date,
    trade_date: date,
    actions: list[dict[str, Any]],
    limit_prices: dict[str, float] | None = None,
    not_before: datetime | None = None,
    not_after: datetime | None = None,
) -> dict[str, Any]:
    """Seal an account decision that creates no new order-plan operations.

    A cash-only target, a target below the exchange minimum lot, a permission
    veto, or an already-satisfied HOLD is still a real account decision.  It is
    persisted separately from :func:`create_order_plan_batch` so an empty
    decision can never be mistaken for an executable batch or fake order.
    """

    identifiers = {
        "portfolio_id": str(portfolio_id).strip(),
        "account_netting_plan_id": str(account_netting_plan_id).strip(),
        "target_version": str(target_version).strip(),
        "actor": str(actor).strip(),
    }
    if any(not value for value in identifiers.values()):
        raise ValueError("account order decision identifiers are required")
    if not isinstance(signal_date, date) or not isinstance(trade_date, date):
        raise ValueError("account order decision dates are required")
    if signal_date > trade_date:
        raise ValueError("account order decision signal date follows trade date")
    if not isinstance(actions, list) or any(not isinstance(item, dict) for item in actions):
        raise ValueError("account order decision actions must be a list of objects")
    for action in actions:
        order_plan = action.get("order_plan")
        if not isinstance(order_plan, list):
            raise ValueError("account order decision actions require order_plan lists")
        if order_plan:
            raise ValueError("decision-only evidence cannot contain order operations")
        if str(action.get("action") or "").upper() not in {
            "BUY",
            "SELL",
            "EXIT",
            "HOLD",
            "NO_ACTION",
        }:
            raise ValueError("decision-only evidence contains an unknown account action")
    prices = {
        str(key).upper(): float(value) for key, value in (limit_prices or {}).items()
    }
    if any(not isfinite(value) or value <= 0 for value in prices.values()):
        raise ValueError("account order decision prices must be positive and finite")
    payload = {
        "contract_version": ACCOUNT_ORDER_DECISION_VERSION,
        **identifiers,
        "signal_date": signal_date.isoformat(),
        "trade_date": trade_date.isoformat(),
        "actions": actions,
        "limit_prices": prices,
        "not_before": not_before.isoformat() if not_before else None,
        "not_after": not_after.isoformat() if not_after else None,
        "decision_outcome": "no_new_order_plan_operations",
        "execution_order_plan_batch_created": False,
        "simulation_orders_created": 0,
        "prior_plan_continuation_allowed": any(
            "no_valid_target_previous_retained" in list(action.get("notes") or [])
            for action in actions
        ),
        "valuation_replay_required": True,
    }
    return {**payload, "event_sha256": _canonical_hash(payload)}


def validate_account_order_decision_event(
    value: dict[str, Any],
    *,
    portfolio_id: str,
    account_netting_plan_id: str,
) -> dict[str, Any]:
    """Validate decision-only evidence before Advice treats it as authoritative."""

    if not isinstance(value, dict):
        raise ValueError("account order decision evidence must be an object")
    payload = dict(value)
    event_sha256 = str(payload.pop("event_sha256", "")).lower()
    if not _is_sha256(event_sha256) or event_sha256 != _canonical_hash(payload):
        raise ValueError("account order decision evidence seal is invalid")
    try:
        signal_date = date.fromisoformat(str(payload.get("signal_date") or ""))
        trade_date = date.fromisoformat(str(payload.get("trade_date") or ""))
    except ValueError as exc:
        raise ValueError("account order decision evidence dates are invalid") from exc
    if (
        payload.get("contract_version") != ACCOUNT_ORDER_DECISION_VERSION
        or str(payload.get("portfolio_id") or "") != str(portfolio_id)
        or str(payload.get("account_netting_plan_id") or "")
        != str(account_netting_plan_id)
        or not str(payload.get("target_version") or "").strip()
        or not str(payload.get("actor") or "").strip()
        or signal_date > trade_date
        or payload.get("decision_outcome") != "no_new_order_plan_operations"
        or payload.get("execution_order_plan_batch_created") is not False
        or payload.get("simulation_orders_created") != 0
        or not isinstance(payload.get("prior_plan_continuation_allowed"), bool)
        or payload.get("valuation_replay_required") is not True
    ):
        raise ValueError("account order decision evidence identity is invalid")
    actions = payload.get("actions")
    if not isinstance(actions, list):
        raise ValueError("account order decision evidence actions are invalid")
    expected_prior_continuation = False
    for action in actions:
        if (
            not isinstance(action, dict)
            or action.get("order_plan") != []
            or str(action.get("action") or "").upper()
            not in {"BUY", "SELL", "EXIT", "HOLD", "NO_ACTION"}
        ):
            raise ValueError("account order decision evidence contains order operations")
        expected_prior_continuation = expected_prior_continuation or (
            "no_valid_target_previous_retained"
            in list(action.get("notes") or [])
        )
    if (
        payload.get("prior_plan_continuation_allowed")
        is not expected_prior_continuation
    ):
        raise ValueError(
            "account order decision continuation authority is inconsistent"
        )
    prices = payload.get("limit_prices")
    if not isinstance(prices, dict) or any(
        not isfinite(float(value)) or float(value) <= 0 for value in prices.values()
    ):
        raise ValueError("account order decision evidence prices are invalid")
    return {**payload, "event_sha256": event_sha256}


def build_settlement_calendar_evidence(
    *,
    trade_date: date,
    next_trade_date: date,
    dataset_identity_sha256: str,
    dataset_lineage_id: str,
    calendar_file_sha256: str,
) -> dict[str, Any]:
    """Seal the next-session decision to one immutable execution dataset."""

    identity = str(dataset_identity_sha256 or "").strip().lower()
    lineage = str(dataset_lineage_id or "").strip().lower()
    calendar_hash = str(calendar_file_sha256 or "").strip().lower()
    if not _is_sha256(identity) or not _is_sha256(lineage):
        raise ValueError("settlement calendar evidence requires sealed dataset lineage")
    if not _is_sha256(calendar_hash):
        raise ValueError("settlement calendar evidence requires a sealed calendar file")
    if not trade_date < next_trade_date <= trade_date + timedelta(days=31):
        raise ValueError("settlement calendar next session is outside its horizon")
    payload = {
        "contract_version": SIMULATION_SETTLEMENT_CALENDAR_EVIDENCE_VERSION,
        "calendar_path": SETTLEMENT_CALENDAR_RELATIVE_PATH,
        "calendar_file_sha256": calendar_hash,
        "dataset_identity_sha256": identity,
        "dataset_lineage_id": lineage,
        "trade_date": trade_date.isoformat(),
        "next_trade_date": next_trade_date.isoformat(),
    }
    return {**payload, "evidence_sha256": _canonical_hash(payload)}


def validate_settlement_calendar_binding(
    value: Any,
    *,
    trade_date: date,
    dataset_identity_sha256: str,
    dataset_lineage_id: str,
) -> dict[str, Any]:
    """Validate the batch-persisted calendar identity independently of replay."""

    if not isinstance(value, dict):
        raise ValueError("daily simulation batch has no settlement calendar binding")
    identity = str(dataset_identity_sha256 or "").strip().lower()
    lineage = str(dataset_lineage_id or "").strip().lower()
    calendar_hash = str(value.get("calendar_file_sha256") or "").strip().lower()
    calendar_bytes = value.get("calendar_file_bytes")
    try:
        bound_trade_date = date.fromisoformat(str(value.get("trade_date") or ""))
        next_trade_date = date.fromisoformat(
            str(value.get("next_trade_date") or "")
        )
    except ValueError as exc:
        raise ValueError("daily settlement calendar binding dates are invalid") from exc
    if (
        not _is_sha256(identity)
        or not _is_sha256(lineage)
        or not _is_sha256(calendar_hash)
        or isinstance(calendar_bytes, bool)
        or not isinstance(calendar_bytes, int)
        or calendar_bytes < 1
        or bound_trade_date != trade_date
        or not trade_date < next_trade_date <= trade_date + timedelta(days=31)
    ):
        raise ValueError("daily settlement calendar binding identity is invalid")
    payload = {
        "contract_version": SIMULATION_SETTLEMENT_CALENDAR_BINDING_VERSION,
        "calendar_path": SETTLEMENT_CALENDAR_RELATIVE_PATH,
        "calendar_file_sha256": calendar_hash,
        "calendar_file_bytes": calendar_bytes,
        "dataset_identity_sha256": identity,
        "dataset_lineage_id": lineage,
        "trade_date": trade_date.isoformat(),
        "next_trade_date": next_trade_date.isoformat(),
    }
    expected = {**payload, "binding_sha256": _canonical_hash(payload)}
    if value != expected:
        raise ValueError("daily settlement calendar binding does not match its dataset")
    return expected


def build_settlement_calendar_binding(
    dataset: dict[str, Any], *, trade_date: date
) -> dict[str, Any]:
    """Resolve and verify the calendar hash from one sealed Qlib dataset."""

    if not isinstance(dataset, dict) or dataset.get("output_files_verified") is not True:
        raise ValueError("daily settlement dataset outputs are not independently verified")
    provider = Path(str(dataset.get("path") or ""))
    provenance = dict(dataset.get("provenance") or {})
    output_manifest = provenance.get("output_manifest")
    if (
        not isinstance(output_manifest, dict)
        or output_manifest.get("version") != QLIB_OUTPUT_MANIFEST_VERSION
        or not isinstance(output_manifest.get("files"), list)
    ):
        raise ValueError("daily settlement dataset has no sealed output manifest")
    entries = [
        item
        for item in output_manifest["files"]
        if isinstance(item, dict)
        and item.get("path") == SETTLEMENT_CALENDAR_RELATIVE_PATH
    ]
    if len(entries) != 1:
        raise ValueError("daily settlement dataset has no unique sealed calendar")
    entry = entries[0]
    calendar_bytes = entry.get("bytes")
    calendar_hash = str(entry.get("sha256") or "").strip().lower()
    calendar_path = provider / Path(SETTLEMENT_CALENDAR_RELATIVE_PATH)
    if (
        isinstance(calendar_bytes, bool)
        or not isinstance(calendar_bytes, int)
        or calendar_bytes < 1
        or not _is_sha256(calendar_hash)
        or not calendar_path.is_file()
        or calendar_path.stat().st_size != calendar_bytes
        or _sha256_file(calendar_path) != calendar_hash
    ):
        raise ValueError("daily settlement calendar does not match the sealed output manifest")
    try:
        frame = pd.read_parquet(calendar_path)
    except Exception as exc:
        raise ValueError("daily settlement calendar cannot be read") from exc
    if (
        calendar_path.stat().st_size != calendar_bytes
        or _sha256_file(calendar_path) != calendar_hash
    ):
        raise ValueError("daily settlement calendar changed while being bound")
    if "date" not in frame.columns or frame.empty:
        raise ValueError("daily settlement calendar has no sessions")
    timestamps = pd.to_datetime(frame["date"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("daily settlement calendar contains invalid sessions")
    sessions = [value.date() for value in timestamps]
    if sessions != sorted(set(sessions)) or trade_date not in sessions:
        raise ValueError("daily settlement calendar sessions are not governed")
    later_sessions = [value for value in sessions if value > trade_date]
    if not later_sessions:
        raise ValueError("daily settlement calendar has no next session")
    next_trade_date = later_sessions[0]
    if next_trade_date > trade_date + timedelta(days=31):
        raise ValueError("daily settlement calendar next session is outside its horizon")
    payload = {
        "contract_version": SIMULATION_SETTLEMENT_CALENDAR_BINDING_VERSION,
        "calendar_path": SETTLEMENT_CALENDAR_RELATIVE_PATH,
        "calendar_file_sha256": calendar_hash,
        "calendar_file_bytes": calendar_bytes,
        "dataset_identity_sha256": str(
            provenance.get("dataset_identity_sha256") or ""
        ).lower(),
        "dataset_lineage_id": str(provenance.get("dataset_lineage_id") or "").lower(),
        "trade_date": trade_date.isoformat(),
        "next_trade_date": next_trade_date.isoformat(),
    }
    return validate_settlement_calendar_binding(
        {**payload, "binding_sha256": _canonical_hash(payload)},
        trade_date=trade_date,
        dataset_identity_sha256=payload["dataset_identity_sha256"],
        dataset_lineage_id=payload["dataset_lineage_id"],
    )


def validate_settlement_calendar_evidence(
    value: Any,
    *,
    trade_date: date,
    next_trade_date: date,
    dataset_identity_sha256: str,
    dataset_lineage_id: str,
    calendar_file_sha256: str,
) -> dict[str, Any]:
    """Reject settlement dates not sealed to the batch execution snapshot."""

    if not isinstance(value, dict):
        raise ValueError("simulation execution evidence has no settlement calendar proof")
    expected = build_settlement_calendar_evidence(
        trade_date=trade_date,
        next_trade_date=next_trade_date,
        dataset_identity_sha256=dataset_identity_sha256,
        dataset_lineage_id=dataset_lineage_id,
        calendar_file_sha256=calendar_file_sha256,
    )
    if value != expected:
        raise ValueError(
            "simulation settlement calendar evidence does not match the bound dataset"
        )
    return expected


def build_allocation_source_rebind_event(
    *,
    portfolio_id: str,
    old_allocation_id: str,
    new_allocation_id: str,
    actor: str,
    old_execution_contract_hash: str,
    new_execution_contract_hash: str,
    old_simulation_semantics_sha256: str,
    new_simulation_semantics_sha256: str,
) -> dict[str, Any]:
    """Seal one in-place allocation-source replacement for the primary ledger."""

    identifiers = {
        "portfolio_id": str(portfolio_id).strip(),
        "old_allocation_id": str(old_allocation_id).strip(),
        "new_allocation_id": str(new_allocation_id).strip(),
        "actor": str(actor).strip(),
    }
    if any(not value for value in identifiers.values()):
        raise ValueError("allocation source replacement identifiers are required")
    if identifiers["old_allocation_id"] == identifiers["new_allocation_id"]:
        raise ValueError("allocation source replacement requires a new allocation")
    hashes = {
        "old_execution_contract_hash": str(old_execution_contract_hash).lower(),
        "new_execution_contract_hash": str(new_execution_contract_hash).lower(),
        "old_simulation_semantics_sha256": str(
            old_simulation_semantics_sha256
        ).lower(),
        "new_simulation_semantics_sha256": str(
            new_simulation_semantics_sha256
        ).lower(),
    }
    if any(not _is_sha256(value) for value in hashes.values()):
        raise ValueError("allocation source replacement requires sealed contracts")
    payload = {
        "contract_version": ALLOCATION_SOURCE_REBIND_VERSION,
        **identifiers,
        **hashes,
        "ledger_action": "in_place_source_rebind",
        "ledger_rows_copied": False,
    }
    return {**payload, "event_sha256": _canonical_hash(payload)}


def validate_allocation_source_rebind_event(
    value: dict[str, Any],
    *,
    portfolio_id: str,
    old_allocation_id: str,
    new_allocation_id: str,
) -> dict[str, Any]:
    """Validate a persisted cutover event before treating a retry as complete."""

    if not isinstance(value, dict):
        raise ValueError("allocation source replacement event must be an object")
    payload = dict(value)
    event_sha256 = str(payload.pop("event_sha256", "")).lower()
    if not _is_sha256(event_sha256) or event_sha256 != _canonical_hash(payload):
        raise ValueError("allocation source replacement event seal is invalid")
    if (
        payload.get("contract_version") != ALLOCATION_SOURCE_REBIND_VERSION
        or str(payload.get("portfolio_id") or "") != str(portfolio_id)
        or str(payload.get("old_allocation_id") or "") != str(old_allocation_id)
        or str(payload.get("new_allocation_id") or "") != str(new_allocation_id)
        or not str(payload.get("actor") or "").strip()
        or payload.get("ledger_action") != "in_place_source_rebind"
        or payload.get("ledger_rows_copied") is not False
        or any(
            not _is_sha256(payload.get(field))
            for field in (
                "old_execution_contract_hash",
                "new_execution_contract_hash",
                "old_simulation_semantics_sha256",
                "new_simulation_semantics_sha256",
            )
        )
    ):
        raise ValueError("allocation source replacement event identity is invalid")
    return {**payload, "event_sha256": event_sha256}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paper_target_adjustments(
    target_weights: dict[str, float],
    previous_target_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Describe a frozen paper order-plan delta without inventing a thesis.

    The immutable Qlib order-plan artifact deliberately stores executable target
    weights, not an LLM-written investment rationale.  This projection therefore
    exposes only the factual reason available at this boundary: how the current
    frozen target changed from the preceding frozen target.  It must never be
    confused with the separately gated, human-facing RecommendationStore.
    """

    current = {str(key).upper(): float(value) for key, value in target_weights.items()}
    previous = {
        str(key).upper(): float(value)
        for key, value in (previous_target_weights or {}).items()
    }
    rows: list[dict[str, Any]] = []
    for instrument in sorted(set(current) | set(previous)):
        target_weight = current.get(instrument, 0.0)
        previous_weight = previous.get(instrument, 0.0)
        delta = target_weight - previous_weight
        if previous_weight <= 0.0 < target_weight:
            action = "add"
            reason = "new frozen Qlib order-plan target"
        elif target_weight <= 0.0 < previous_weight:
            action = "remove"
            reason = "removed by frozen Qlib order-plan target"
        elif delta > 1e-12:
            action = "increase"
            reason = "frozen Qlib order-plan target weight increased"
        elif delta < -1e-12:
            action = "decrease"
            reason = "frozen Qlib order-plan target weight decreased"
        else:
            action = "hold"
            reason = "frozen Qlib order-plan target unchanged"
        rows.append(
            {
                "instrument": instrument,
                "target_weight": target_weight,
                "previous_weight": previous_weight,
                "weight_change": delta,
                "action": action,
                "reason": reason,
                "reason_basis": "frozen_qlib_order_plan_weight_delta",
            }
        )
    return rows


def _parse_aware_timestamp(value: Any, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Qlib order-plan {field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Qlib order-plan {field} must include a timezone")
    return parsed.astimezone(UTC)


def _pair_source_contract_hash(config: dict[str, Any]) -> str:
    return _canonical_hash(
        {
            "strategy_type": "pair",
            "signal_frequency": "day",
            "signal_horizon": "1d",
            "execution_frequency": "1min",
            "config": config,
        }
    )


def _simulation_semantics_payload(
    *,
    source_type: str,
    source_id: str,
    source_execution_contract_hash: str,
    execution_adapter: str,
    execution_frequency: str,
    daily_dataset: str,
    daily_dataset_identity_sha256: str,
    daily_dataset_lineage_id: str,
    execution_dataset: str,
    execution_dataset_identity_sha256: str,
    execution_dataset_lineage_id: str,
    execution_field_contract_version: str,
    execution_engine_version: str,
    execution_policy: dict[str, Any],
) -> dict[str, Any]:
    policy = dict(execution_policy)
    policy.pop("simulation_semantics_sha256", None)
    return {
        "version": SIMULATION_EXECUTION_SEMANTICS_VERSION,
        "source_type": source_type,
        "source_id": source_id,
        "source_execution_contract_hash": source_execution_contract_hash,
        "execution_adapter": execution_adapter,
        "execution_frequency": execution_frequency,
        "daily_dataset": daily_dataset,
        "daily_dataset_identity_sha256": daily_dataset_identity_sha256,
        "daily_dataset_lineage_id": daily_dataset_lineage_id,
        "execution_dataset": execution_dataset,
        "execution_dataset_identity_sha256": execution_dataset_identity_sha256,
        "execution_dataset_lineage_id": execution_dataset_lineage_id,
        "execution_field_contract_version": execution_field_contract_version,
        "execution_engine_version": execution_engine_version,
        "execution_policy": policy,
    }


class SimulationStore:
    """Transactional T+1 ledger for governed recommendation, strategy and allocation targets."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = open_database(database_url)
        self.safe_mode = SafeModeStore(database_url)

    @staticmethod
    def _cash_amount(value: Any) -> Decimal:
        amount = Decimal(str(value)).quantize(Decimal("0.000001"))
        if not amount.is_finite():
            raise ValueError("cash ledger amount must be finite")
        return amount

    @staticmethod
    def _session_time(day: date, hour: int, minute: int = 0) -> datetime:
        return datetime.combine(
            day,
            time(hour, minute),
            SHANGHAI_TIMEZONE,
        )

    def _insert_cash_event(
        self,
        connection: Any,
        *,
        portfolio_id: str,
        event_key: str,
        event_type: str,
        amount: Decimal,
        occurred_at: datetime,
        details: dict[str, Any],
        allocations: list[tuple[str, str, Decimal]],
        batch_id: str | None = None,
        order_id: str | None = None,
        now: datetime,
    ) -> str:
        normalized_amount = self._cash_amount(amount)
        if normalized_amount < 0:
            raise ValueError("cash event amount must be non-negative")
        existing = connection.execute(
            select(simulation_cash_events).where(
                simulation_cash_events.c.portfolio_id == portfolio_id,
                simulation_cash_events.c.event_key == event_key,
            )
        ).first()
        if existing is not None:
            identity = {
                "event_type": str(existing.event_type),
                "amount": self._cash_amount(existing.amount),
                "batch_id": str(existing.batch_id) if existing.batch_id else None,
                "order_id": str(existing.order_id) if existing.order_id else None,
            }
            expected = {
                "event_type": event_type,
                "amount": normalized_amount,
                "batch_id": str(batch_id) if batch_id else None,
                "order_id": str(order_id) if order_id else None,
            }
            if identity != expected:
                raise ValueError("cash event key was reused with a different payload")
            return str(existing.id)
        event_id = uuid.uuid4().hex
        connection.execute(
            insert(simulation_cash_events).values(
                id=event_id,
                portfolio_id=portfolio_id,
                batch_id=batch_id,
                order_id=order_id,
                event_key=event_key,
                event_type=event_type,
                amount=normalized_amount,
                details_json=details,
                occurred_at=occurred_at,
                created_at=now,
            )
        )
        for lot_id, action, allocated in allocations:
            value = self._cash_amount(allocated)
            if value <= 0:
                continue
            connection.execute(
                insert(simulation_cash_event_allocations).values(
                    id=uuid.uuid4().hex,
                    event_id=event_id,
                    cash_lot_id=lot_id,
                    action=action,
                    amount=value,
                    created_at=now,
                )
            )
        return event_id

    def _create_cash_lot(
        self,
        connection: Any,
        *,
        portfolio_id: str,
        event_key: str,
        source_type: str,
        amount: Decimal,
        tradable_at: datetime,
        withdrawable_at: datetime,
        occurred_at: datetime,
        now: datetime,
        source_reference_id: str | None = None,
        batch_id: str | None = None,
        order_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        value = self._cash_amount(amount)
        if value <= 0:
            raise ValueError("cash lot amount must be positive")
        existing = connection.execute(
            select(simulation_cash_lots).where(
                simulation_cash_lots.c.portfolio_id == portfolio_id,
                simulation_cash_lots.c.lot_key == event_key,
            )
        ).first()
        if existing is not None:
            if (
                str(existing.source_type) != source_type
                or self._cash_amount(existing.free_amount)
                + self._cash_amount(existing.frozen_amount)
                != value
            ):
                raise ValueError("cash lot key was reused with a different payload")
            return str(existing.id)
        lot_id = uuid.uuid4().hex
        connection.execute(
            insert(simulation_cash_lots).values(
                id=lot_id,
                portfolio_id=portfolio_id,
                lot_key=event_key,
                source_type=source_type,
                source_reference_id=source_reference_id,
                free_amount=value,
                frozen_amount=Decimal("0"),
                tradable_at=tradable_at,
                withdrawable_at=withdrawable_at,
                created_at=now,
                updated_at=now,
            )
        )
        self._insert_cash_event(
            connection,
            portfolio_id=portfolio_id,
            event_key=event_key,
            event_type="create",
            amount=value,
            occurred_at=occurred_at,
            details={"source_type": source_type, **(details or {})},
            allocations=[(lot_id, "create", value)],
            batch_id=batch_id,
            order_id=order_id,
            now=now,
        )
        return lot_id

    def _allocate_free_cash(
        self,
        connection: Any,
        *,
        portfolio_id: str,
        event_key: str,
        event_type: str,
        amount: Decimal,
        as_of: datetime,
        now: datetime,
        require_withdrawable: bool = False,
        batch_id: str | None = None,
        order_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> list[tuple[str, Decimal]]:
        value = self._cash_amount(amount)
        if value <= 0:
            raise ValueError("cash allocation amount must be positive")
        if event_type not in {"freeze", "consume_free"}:
            raise ValueError("cash allocation event type is invalid")
        existing = connection.execute(
            select(simulation_cash_events).where(
                simulation_cash_events.c.portfolio_id == portfolio_id,
                simulation_cash_events.c.event_key == event_key,
            )
        ).first()
        if existing is not None:
            if (
                str(existing.event_type) != event_type
                or self._cash_amount(existing.amount) != value
            ):
                raise ValueError("cash event key was reused with a different payload")
            return []
        availability = (
            simulation_cash_lots.c.withdrawable_at
            if require_withdrawable
            else simulation_cash_lots.c.tradable_at
        )
        rows = connection.execute(
            select(simulation_cash_lots)
            .where(
                simulation_cash_lots.c.portfolio_id == portfolio_id,
                simulation_cash_lots.c.free_amount > 0,
                availability <= as_of,
            )
            .order_by(
                availability,
                simulation_cash_lots.c.created_at,
                simulation_cash_lots.c.id,
            )
            .with_for_update()
        ).all()
        remaining = value
        allocations: list[tuple[str, Decimal]] = []
        for row in rows:
            if remaining <= 0:
                break
            available = self._cash_amount(row.free_amount)
            used = min(available, remaining)
            if used <= 0:
                continue
            updates: dict[str, Any] = {
                "free_amount": available - used,
                "updated_at": now,
            }
            if event_type == "freeze":
                updates["frozen_amount"] = self._cash_amount(row.frozen_amount) + used
            connection.execute(
                update(simulation_cash_lots)
                .where(simulation_cash_lots.c.id == row.id)
                .values(**updates)
            )
            allocations.append((str(row.id), used))
            remaining -= used
        if remaining > Decimal("0.000001"):
            permission = "withdrawable" if require_withdrawable else "tradable"
            raise RuntimeError(
                f"simulation cash ledger has insufficient {permission} free cash"
            )
        self._insert_cash_event(
            connection,
            portfolio_id=portfolio_id,
            event_key=event_key,
            event_type=event_type,
            amount=value,
            occurred_at=as_of,
            details=details or {},
            allocations=[
                (lot_id, event_type, allocated)
                for lot_id, allocated in allocations
            ],
            batch_id=batch_id,
            order_id=order_id,
            now=now,
        )
        return allocations

    def _freeze_order_cash(
        self,
        connection: Any,
        *,
        portfolio_id: str,
        order_id: str,
        event_key: str,
        amount: Decimal,
        as_of: datetime,
        now: datetime,
        batch_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        allocations = self._allocate_free_cash(
            connection,
            portfolio_id=portfolio_id,
            event_key=event_key,
            event_type="freeze",
            amount=amount,
            as_of=as_of,
            now=now,
            batch_id=batch_id,
            order_id=order_id,
            details=details,
        )
        for lot_id, allocated in allocations:
            connection.execute(
                insert(simulation_cash_reservations).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio_id,
                    order_id=order_id,
                    cash_lot_id=lot_id,
                    reserved_amount=allocated,
                    remaining_amount=allocated,
                    created_at=now,
                    updated_at=now,
                )
            )

    def _reservation_total(
        self,
        connection: Any,
        order_id: str,
    ) -> Decimal:
        value = connection.scalar(
            select(
                func.coalesce(
                    func.sum(simulation_cash_reservations.c.remaining_amount),
                    0,
                )
            ).where(simulation_cash_reservations.c.order_id == order_id)
        )
        return self._cash_amount(value or 0)

    def _move_order_reservation(
        self,
        connection: Any,
        *,
        portfolio_id: str,
        order_id: str,
        event_key: str,
        event_type: str,
        amount: Decimal | None,
        occurred_at: datetime,
        now: datetime,
        batch_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> Decimal:
        if event_type not in {"consume_frozen", "release"}:
            raise ValueError("reservation movement type is invalid")
        existing = connection.execute(
            select(simulation_cash_events).where(
                simulation_cash_events.c.portfolio_id == portfolio_id,
                simulation_cash_events.c.event_key == event_key,
            )
        ).first()
        if existing is not None:
            if str(existing.event_type) != event_type:
                raise ValueError("cash event key was reused with a different payload")
            return self._cash_amount(existing.amount)
        reservations = connection.execute(
            select(simulation_cash_reservations)
            .where(
                simulation_cash_reservations.c.order_id == order_id,
                simulation_cash_reservations.c.remaining_amount > 0,
            )
            .order_by(
                simulation_cash_reservations.c.created_at,
                simulation_cash_reservations.c.id,
            )
            .with_for_update()
        ).all()
        available = sum(
            (self._cash_amount(row.remaining_amount) for row in reservations),
            Decimal("0"),
        )
        requested = available if amount is None else self._cash_amount(amount)
        if requested < 0 or requested - available > Decimal("0.000001"):
            raise RuntimeError("simulation order cash reservation is insufficient")
        if requested == 0:
            return Decimal("0")
        remaining = requested
        allocations: list[tuple[str, str, Decimal]] = []
        for reservation in reservations:
            if remaining <= 0:
                break
            reserved = self._cash_amount(reservation.remaining_amount)
            used = min(reserved, remaining)
            if used <= 0:
                continue
            lot = connection.execute(
                select(simulation_cash_lots)
                .where(simulation_cash_lots.c.id == reservation.cash_lot_id)
                .with_for_update()
            ).one()
            frozen = self._cash_amount(lot.frozen_amount)
            if used - frozen > Decimal("0.000001"):
                raise RuntimeError("simulation frozen cash lot is inconsistent")
            lot_updates: dict[str, Any] = {
                "frozen_amount": frozen - used,
                "updated_at": now,
            }
            if event_type == "release":
                lot_updates["free_amount"] = self._cash_amount(lot.free_amount) + used
            connection.execute(
                update(simulation_cash_lots)
                .where(simulation_cash_lots.c.id == lot.id)
                .values(**lot_updates)
            )
            connection.execute(
                update(simulation_cash_reservations)
                .where(simulation_cash_reservations.c.id == reservation.id)
                .values(remaining_amount=reserved - used, updated_at=now)
            )
            allocations.append((str(lot.id), event_type, used))
            remaining -= used
        if remaining > Decimal("0.000001"):
            raise RuntimeError("simulation order reservation movement did not reconcile")
        self._insert_cash_event(
            connection,
            portfolio_id=portfolio_id,
            event_key=event_key,
            event_type=event_type,
            amount=requested,
            occurred_at=occurred_at,
            details=details or {},
            allocations=allocations,
            batch_id=batch_id,
            order_id=order_id,
            now=now,
        )
        return requested

    def _cash_view_in_connection(
        self,
        connection: Any,
        portfolio_id: str,
        *,
        as_of: datetime,
    ) -> dict[str, Any]:
        rows = connection.execute(
            select(simulation_cash_lots)
            .where(simulation_cash_lots.c.portfolio_id == portfolio_id)
            .order_by(simulation_cash_lots.c.created_at, simulation_cash_lots.c.id)
        ).all()
        free = sum(
            (self._cash_amount(row.free_amount) for row in rows),
            Decimal("0"),
        )
        frozen = sum(
            (self._cash_amount(row.frozen_amount) for row in rows),
            Decimal("0"),
        )
        tradable = sum(
            (
                self._cash_amount(row.free_amount)
                for row in rows
                if row.tradable_at <= as_of
            ),
            Decimal("0"),
        )
        withdrawable = sum(
            (
                self._cash_amount(row.free_amount)
                for row in rows
                if row.withdrawable_at <= as_of
            ),
            Decimal("0"),
        )
        return {
            "as_of": as_of,
            "total_cash": free + frozen,
            "free_cash": free,
            "frozen_cash": frozen,
            "tradable_cash": tradable,
            "withdrawable_cash": withdrawable,
            "unsettled_cash": free - tradable,
            "tradable_not_withdrawable_cash": tradable - withdrawable,
            "lot_count": len(rows),
        }

    def _assert_cash_lots_reconcile(
        self,
        connection: Any,
        portfolio_id: str,
        *,
        expected_cash: Any,
        as_of: datetime,
    ) -> dict[str, Any]:
        view = self._cash_view_in_connection(
            connection,
            portfolio_id,
            as_of=as_of,
        )
        difference = view["total_cash"] - self._cash_amount(expected_cash)
        if abs(difference) > Decimal("0.000001"):
            raise RuntimeError("simulation cash lots do not reconcile with portfolio cash")
        return view

    def cash_view(
        self,
        portfolio_id: str,
        *,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        observed_at = as_of or _now()
        if observed_at.tzinfo is None:
            raise ValueError("cash view as_of must be timezone-aware")
        with self.engine.connect() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            view = self._assert_cash_lots_reconcile(
                connection,
                portfolio_id,
                expected_cash=portfolio.cash,
                as_of=observed_at,
            )
            return {
                key: float(value) if isinstance(value, Decimal) else value
                for key, value in view.items()
            }

    def _insert_security_event(
        self,
        connection: Any,
        *,
        portfolio_id: str,
        event_key: str,
        event_type: str,
        instrument: str,
        quantity: int,
        occurred_at: datetime,
        now: datetime,
        batch_id: str | None = None,
        order_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if quantity <= 0:
            raise ValueError("security event quantity must be positive")
        existing = connection.execute(
            select(simulation_security_events).where(
                simulation_security_events.c.portfolio_id == portfolio_id,
                simulation_security_events.c.event_key == event_key,
            )
        ).first()
        if existing is not None:
            identity = (
                str(existing.event_type),
                str(existing.instrument),
                int(existing.quantity),
                str(existing.order_id) if existing.order_id else None,
            )
            expected = (
                event_type,
                instrument,
                quantity,
                str(order_id) if order_id else None,
            )
            if identity != expected:
                raise ValueError(
                    "security event key was reused with a different payload"
                )
            return
        connection.execute(
            insert(simulation_security_events).values(
                id=uuid.uuid4().hex,
                portfolio_id=portfolio_id,
                batch_id=batch_id,
                order_id=order_id,
                event_key=event_key,
                event_type=event_type,
                instrument=instrument,
                quantity=quantity,
                details_json=details or {},
                occurred_at=occurred_at,
                created_at=now,
            )
        )

    def _freeze_sell_quantity(
        self,
        connection: Any,
        *,
        portfolio_id: str,
        order_id: str,
        instrument: str,
        quantity: int,
        trade_date: date,
        event_key: str,
        occurred_at: datetime,
        now: datetime,
        batch_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if quantity <= 0:
            raise ValueError("sell reservation quantity must be positive")
        existing = connection.execute(
            select(simulation_position_reservations).where(
                simulation_position_reservations.c.order_id == order_id
            )
        ).first()
        if existing is not None:
            if (
                str(existing.instrument) != instrument
                or int(existing.reserved_quantity) != quantity
            ):
                raise ValueError(
                    "sell order reservation is already bound to another payload"
                )
            return
        position = connection.execute(
            select(simulation_positions)
            .where(
                simulation_positions.c.portfolio_id == portfolio_id,
                simulation_positions.c.instrument == instrument,
            )
            .with_for_update()
        ).first()
        if position is None:
            raise RuntimeError("sell order has no economic position to freeze")
        available = int(position.available_quantity)
        if position.last_trade_date is None or position.last_trade_date < trade_date:
            settled = int(position.quantity)
            if settled > available:
                delta = settled - available
                available = settled
                connection.execute(
                    update(simulation_positions)
                    .where(
                        simulation_positions.c.portfolio_id == portfolio_id,
                        simulation_positions.c.instrument == instrument,
                    )
                    .values(available_quantity=available, updated_at=now)
                )
                self._insert_security_event(
                    connection,
                    portfolio_id=portfolio_id,
                    event_key=(
                        f"position:{portfolio_id}:{instrument}:"
                        f"reclassify:{trade_date.isoformat()}"
                    ),
                    event_type="reclassify",
                    instrument=instrument,
                    quantity=delta,
                    occurred_at=occurred_at,
                    now=now,
                    batch_id=batch_id,
                    details={"reason": "t_plus_one_sellable"},
                )
        frozen = int(position.frozen_quantity)
        if quantity > available - frozen:
            raise RuntimeError(
                "sell order exceeds free sellable quantity after existing freezes"
            )
        connection.execute(
            update(simulation_positions)
            .where(
                simulation_positions.c.portfolio_id == portfolio_id,
                simulation_positions.c.instrument == instrument,
            )
            .values(frozen_quantity=frozen + quantity, updated_at=now)
        )
        connection.execute(
            insert(simulation_position_reservations).values(
                id=uuid.uuid4().hex,
                portfolio_id=portfolio_id,
                order_id=order_id,
                instrument=instrument,
                reserved_quantity=quantity,
                remaining_quantity=quantity,
                created_at=now,
                updated_at=now,
            )
        )
        self._insert_security_event(
            connection,
            portfolio_id=portfolio_id,
            event_key=event_key,
            event_type="freeze",
            instrument=instrument,
            quantity=quantity,
            occurred_at=occurred_at,
            now=now,
            batch_id=batch_id,
            order_id=order_id,
            details=details,
        )

    def _security_reservation_total(
        self,
        connection: Any,
        order_id: str,
    ) -> int:
        return int(
            connection.scalar(
                select(
                    func.coalesce(
                        func.sum(
                            simulation_position_reservations.c.remaining_quantity
                        ),
                        0,
                    )
                ).where(simulation_position_reservations.c.order_id == order_id)
            )
            or 0
        )

    def _move_sell_reservation(
        self,
        connection: Any,
        *,
        portfolio_id: str,
        order_id: str,
        event_key: str,
        event_type: str,
        quantity: int | None,
        occurred_at: datetime,
        now: datetime,
        batch_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> int:
        if event_type not in {"consume", "release"}:
            raise ValueError("sell reservation movement type is invalid")
        existing_event = connection.execute(
            select(simulation_security_events).where(
                simulation_security_events.c.portfolio_id == portfolio_id,
                simulation_security_events.c.event_key == event_key,
            )
        ).first()
        if existing_event is not None:
            if str(existing_event.event_type) != event_type:
                raise ValueError(
                    "security event key was reused with a different payload"
                )
            return int(existing_event.quantity)
        reservation = connection.execute(
            select(simulation_position_reservations)
            .where(simulation_position_reservations.c.order_id == order_id)
            .with_for_update()
        ).first()
        if reservation is None:
            if quantity in {None, 0}:
                return 0
            raise RuntimeError("sell order has no frozen security reservation")
        available = int(reservation.remaining_quantity)
        moved = available if quantity is None else int(quantity)
        if moved < 0 or moved > available:
            raise RuntimeError("sell order frozen quantity is insufficient")
        if moved == 0:
            return 0
        position = connection.execute(
            select(simulation_positions)
            .where(
                simulation_positions.c.portfolio_id == portfolio_id,
                simulation_positions.c.instrument == reservation.instrument,
            )
            .with_for_update()
        ).one()
        frozen = int(position.frozen_quantity)
        if moved > frozen:
            raise RuntimeError("simulation frozen security position is inconsistent")
        connection.execute(
            update(simulation_positions)
            .where(
                simulation_positions.c.portfolio_id == portfolio_id,
                simulation_positions.c.instrument == reservation.instrument,
            )
            .values(frozen_quantity=frozen - moved, updated_at=now)
        )
        connection.execute(
            update(simulation_position_reservations)
            .where(simulation_position_reservations.c.id == reservation.id)
            .values(remaining_quantity=available - moved, updated_at=now)
        )
        self._insert_security_event(
            connection,
            portfolio_id=portfolio_id,
            event_key=event_key,
            event_type=event_type,
            instrument=str(reservation.instrument),
            quantity=moved,
            occurred_at=occurred_at,
            now=now,
            batch_id=batch_id,
            order_id=order_id,
            details=details,
        )
        return moved

    @staticmethod
    def _governed_execution_policy(
        source: dict[str, Any],
        supplied: dict[str, Any],
        *,
        adapter: str,
    ) -> dict[str, Any]:
        provided = dict(supplied or {})
        if adapter != "long_only" or source.get("policy_mode") == "self_contained":
            return normalize_execution_policy(provided)
        config = dict(source.get("config") or {})
        method = str(source.get("execution_method") or "").lower()
        if method not in {"open", "twap", "vwap", "next_bar"}:
            raise ValueError(
                "long-only simulation requires a governed execution method in its "
                "approved source contract"
            )
        governed = {
            "execution_algorithm": method,
            "execution_frequency": str(
                source.get("execution_frequency")
                or config.get("execution_frequency")
                or ("day" if method == "open" else "5min")
            ),
            "slice_minutes": int(config.get("execution_slice_minutes") or 20),
            "max_slices": int(config.get("max_execution_slices") or 24),
            "max_participation": float(config.get("max_volume_participation") or 0.0),
            "volume_profile": None,
        }
        aliases = {
            "execution_algorithm": "execution_algorithm",
            "execution_frequency": "execution_frequency",
            "slice_minutes": "slice_minutes",
            "max_slices": "max_slices",
            "max_participation": "max_participation",
        }
        for supplied_key, governed_key in aliases.items():
            if supplied_key not in provided or provided[supplied_key] is None:
                continue
            observed = provided[supplied_key]
            expected = governed[governed_key]
            if isinstance(expected, float):
                matches = abs(float(observed) - expected) <= 1e-12
            elif isinstance(expected, int):
                matches = int(observed) == expected
            else:
                matches = str(observed).lower() == str(expected).lower()
            if not matches:
                raise ValueError(
                    f"simulation {supplied_key} must be derived from the approved "
                    "source contract"
                )
        if provided.get("volume_profile") is not None:
            raise ValueError(
                "simulation VWAP profile is derived from the bound Qlib execution "
                "dataset and cannot be supplied by an operator"
            )
        policy = normalize_execution_policy(governed)
        policy["volume_profile_method"] = (
            VWAP_PROFILE_METHOD if method == "vwap" else "none"
        )
        policy["volume_profile_lookback_days"] = (
            int(config.get("vwap_lookback_days") or 20) if method == "vwap" else 0
        )
        return policy

    @staticmethod
    def _bind_execution_semantics(
        *,
        source: dict[str, Any],
        source_type: str,
        source_id: str,
        adapter: str,
        frequency: str,
        daily_dataset: dict[str, Any],
        execution_dataset: dict[str, Any],
        policy: dict[str, Any],
        cost_model: CostModelConfig,
    ) -> dict[str, Any]:
        daily_provenance = dict(daily_dataset["provenance"])
        execution_provenance = dict(execution_dataset["provenance"])
        bound = {
            **policy,
            "simulation_contract_version": SIMULATION_EXECUTION_SEMANTICS_VERSION,
            "source_execution_contract_hash": str(source["execution_contract_hash"]),
            "execution_frequency": frequency,
            "cost_model": cost_model.to_dict(),
        }
        payload = _simulation_semantics_payload(
            source_type=source_type,
            source_id=source_id,
            source_execution_contract_hash=str(source["execution_contract_hash"]),
            execution_adapter=adapter,
            execution_frequency=frequency,
            daily_dataset=str(daily_dataset["name"]),
            daily_dataset_identity_sha256=str(
                daily_provenance["dataset_identity_sha256"]
            ),
            daily_dataset_lineage_id=str(daily_provenance["dataset_lineage_id"]),
            execution_dataset=str(execution_dataset["name"]),
            execution_dataset_identity_sha256=str(
                execution_provenance["dataset_identity_sha256"]
            ),
            execution_dataset_lineage_id=str(
                execution_provenance["dataset_lineage_id"]
            ),
            execution_field_contract_version=str(
                execution_provenance[
                    "field_contract_version"
                    if frequency == "day"
                    else "execution_contract_version"
                ]
            ),
            execution_engine_version=SIMULATION_ENGINE_VERSION,
            execution_policy=bound,
        )
        bound["simulation_semantics_sha256"] = _canonical_hash(payload)
        return bound

    def create(
        self,
        *,
        name: str,
        recommendation_portfolio_id: str | None = None,
        source_type: str | None = None,
        source_id: str | None = None,
        daily_dataset: dict[str, Any],
        execution_dataset: dict[str, Any],
        initial_cash: float,
        execution_policy: dict[str, Any],
        cost_schedule_version: str,
        actor: str,
        execution_adapter: str | None = None,
        execution_contract_hash: str | None = None,
        daily_roll_policy: str = "pinned",
        execution_roll_policy: str = "pinned",
        promotion_stage_id: str | None = None,
    ) -> dict[str, Any]:
        self.safe_mode.assert_inactive(action="simulation account creation")
        if not isfinite(float(initial_cash)) or float(initial_cash) <= 0:
            raise ValueError("simulation initial cash must be positive")
        if cost_schedule_version not in KNOWN_COST_SCHEDULE_VERSIONS:
            raise ValueError("simulation cost schedule version is unavailable")
        daily_provenance = dict(daily_dataset.get("provenance") or {})
        execution_provenance = dict(execution_dataset.get("provenance") or {})
        require_daily_qlib_contract(daily_provenance)
        execution_frequency = str(execution_provenance.get("frequency") or "")
        if execution_frequency not in SIMULATION_EXECUTION_FREQUENCIES:
            raise ValueError("simulation execution frequency must be day, 1min, or 5min")
        if execution_frequency == "day":
            require_daily_qlib_contract(execution_provenance)
            require_native_daily_execution_controls(
                execution_provenance,
                start=str(execution_dataset.get("end_date") or date.today().isoformat()),
            )
        else:
            require_minute_execution_contract(
                execution_provenance,
                frequency=execution_frequency,
                simulation_eligible=True,
            )
        daily_source = str(daily_provenance.get("source_lineage_id") or "")
        execution_source = str(execution_provenance.get("source_lineage_id") or "")
        if len(daily_source) != 64 or daily_source != execution_source:
            raise ValueError("daily and execution datasets must share one verified source lineage")
        required_hashes = {
            "daily identity": daily_provenance.get("dataset_identity_sha256"),
            "daily lineage": daily_provenance.get("dataset_lineage_id"),
            "execution identity": execution_provenance.get("dataset_identity_sha256"),
            "execution lineage": execution_provenance.get("dataset_lineage_id"),
        }
        if any(len(str(value or "")) != 64 for value in required_hashes.values()):
            raise ValueError("simulation datasets require immutable identities and lineages")
        if daily_roll_policy not in {"pinned", "latest_compatible"}:
            raise ValueError("simulation daily roll policy is invalid")
        if execution_roll_policy not in {"pinned", "latest_compatible"}:
            raise ValueError("simulation execution roll policy is invalid")
        normalized_source_type = str(
            source_type or ("recommendation" if recommendation_portfolio_id else "")
        )
        normalized_source_id = str(source_id or recommendation_portfolio_id or "").strip()
        if normalized_source_type not in SIMULATION_SOURCE_TYPES or not normalized_source_id:
            raise ValueError("simulation source_type and source_id are required")
        normalized_stage_id = str(promotion_stage_id or "").strip() or None
        with self.engine.connect() as connection:
            source = self._resolve_source(
                connection, normalized_source_type, normalized_source_id
            )
            if normalized_stage_id is not None:
                if normalized_source_type != "strategy_version":
                    raise ValueError(
                        "only strategy-version paper accounts may bind a promotion stage"
                    )
                stage = connection.execute(
                    select(strategy_promotion_stages).where(
                        strategy_promotion_stages.c.id == normalized_stage_id
                    )
                ).first()
                if (
                    stage is None
                    or str(stage.strategy_version_id) != normalized_source_id
                    or str(stage.status) != "awaiting_simulation"
                    or stage.simulation_portfolio_id is not None
                ):
                    raise ValueError(
                        "simulation promotion stage is not the exact awaiting source stage"
                    )
        if str(source["dataset"]) != str(daily_dataset.get("name") or ""):
            raise ValueError("simulation daily dataset must match the governed source")
        normalized_adapter = str(execution_adapter or source["execution_adapter"])
        if normalized_adapter not in SIMULATION_EXECUTION_ADAPTERS:
            raise ValueError("simulation execution adapter must be long_only or pair")
        if normalized_adapter != source["execution_adapter"]:
            raise ValueError("simulation adapter does not match the governed strategy source")
        if normalized_adapter != "long_only":
            raise ValueError(
                "pair simulation writes are retired; historical pair ledgers are read-only"
            )
        benchmark = str(source.get("benchmark") or "").strip().upper()
        if not benchmark or (benchmark == "CASH" and normalized_adapter != "pair"):
            raise ValueError(
                "long-only simulation sources must bind one non-cash performance benchmark"
            )
        governed_frequency = str(source.get("execution_frequency") or "")
        if governed_frequency and governed_frequency != execution_frequency:
            raise ValueError("simulation frequency does not match the governed strategy source")
        source_contract_hash = str(source.get("execution_contract_hash") or "")
        if not _is_sha256(source_contract_hash):
            raise ValueError(
                "simulation source uses a legacy or incomplete execution contract; "
                "create a new approved source"
            )
        contract_hash = source_contract_hash
        if execution_contract_hash and execution_contract_hash != contract_hash:
            raise ValueError("simulation execution contract does not match its source")
        policy = self._governed_execution_policy(
            source,
            execution_policy,
            adapter=normalized_adapter,
        )
        source_cost_model = CostModelConfig.from_mapping(source.get("config"))
        if cost_schedule_version != source_cost_model.version:
            raise ValueError("simulation cost schedule must match the approved source contract")
        policy = self._bind_execution_semantics(
            source=source,
            source_type=normalized_source_type,
            source_id=normalized_source_id,
            adapter=normalized_adapter,
            frequency=execution_frequency,
            daily_dataset=daily_dataset,
            execution_dataset=execution_dataset,
            policy=policy,
            cost_model=source_cost_model,
        )
        portfolio_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(simulation_portfolios).values(
                        id=portfolio_id,
                        name=name.strip(),
                        recommendation_portfolio_id=(
                            normalized_source_id
                            if normalized_source_type == "recommendation"
                            else None
                        ),
                        source_type=normalized_source_type,
                        source_id=normalized_source_id,
                        promotion_stage_id=normalized_stage_id,
                        status="paused",
                        base_currency="CNY",
                        benchmark=benchmark,
                        initial_cash=Decimal(str(initial_cash)),
                        cash=Decimal(str(initial_cash)),
                        nav=Decimal(str(initial_cash)),
                        high_water_mark=Decimal(str(initial_cash)),
                        execution_algorithm=policy["execution_algorithm"],
                        execution_adapter=normalized_adapter,
                        execution_frequency=execution_frequency,
                        execution_contract_hash=contract_hash,
                        execution_dataset=str(execution_dataset["name"]),
                        daily_dataset=str(daily_dataset["name"]),
                        daily_roll_policy=daily_roll_policy,
                        execution_roll_policy=execution_roll_policy,
                        daily_dataset_identity_sha256=daily_provenance[
                            "dataset_identity_sha256"
                        ],
                        daily_dataset_lineage_id=daily_provenance["dataset_lineage_id"],
                        daily_field_contract_version=daily_provenance["field_contract_version"],
                        execution_dataset_identity_sha256=execution_provenance[
                            "dataset_identity_sha256"
                        ],
                        execution_dataset_lineage_id=execution_provenance["dataset_lineage_id"],
                        execution_field_contract_version=execution_provenance[
                            "field_contract_version"
                            if execution_frequency == "day"
                            else "execution_contract_version"
                        ],
                        execution_engine_version=SIMULATION_ENGINE_VERSION,
                        cost_schedule_version=source_cost_model.version,
                        execution_policy_json=policy,
                        investment_wealth=1.0,
                        twr_high_water_mark=1.0,
                        created_by=actor.strip(),
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    insert(simulation_cash_flows).values(
                        id=uuid.uuid4().hex,
                        portfolio_id=portfolio_id,
                        batch_id=None,
                        trade_date=now.date(),
                        flow_type="initial_deposit",
                        amount=Decimal(str(initial_cash)),
                        balance_after=Decimal(str(initial_cash)),
                        created_at=now,
                    )
                )
                self._create_cash_lot(
                    connection,
                    portfolio_id=portfolio_id,
                    event_key=f"initial-deposit:{portfolio_id}",
                    source_type="initial_deposit",
                    source_reference_id=portfolio_id,
                    amount=Decimal(str(initial_cash)),
                    tradable_at=CASH_OPENING_BALANCE_AT,
                    withdrawable_at=CASH_OPENING_BALANCE_AT,
                    occurred_at=now,
                    now=now,
                    details={"account_created_by": actor.strip()},
                )
        except IntegrityError as exc:
            raise ValueError(
                "simulation name or source/execution dataset is already in use"
            ) from exc
        return self.get(portfolio_id)

    def record_external_flow(
        self,
        portfolio_id: str,
        *,
        trade_date: date,
        timing: str,
        amount: float,
        actor: str,
        flow_key: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Record a confirmed external cash flow (design 4.4/12.1).

        Deposits are positive, withdrawals negative. ``timing`` is ``open``
        (confirmed before the open, investable the same day) or ``close``
        (confirmed after the open, investable from the next day). The flow is
        idempotent on ``flow_key`` (default: a hash of the payload, so a retry
        of the same request replays cleanly); a settled trade date (one that
        already has a NAV row) rejects new flows — the ledger must be rebuilt
        from the last verified state instead.
        """

        self.safe_mode.assert_inactive(action="simulation external cash flow")
        day = trade_date if isinstance(trade_date, date) else date.fromisoformat(
            str(trade_date)
        )
        normalized_timing = str(timing).strip().lower()
        if normalized_timing not in {"open", "close"}:
            raise ValueError("external cash flow timing must be open or close")
        value = float(amount)
        if not isfinite(value) or value == 0.0:
            raise ValueError("external cash flow amount must be finite and non-zero")
        payload = {
            "portfolio_id": str(portfolio_id),
            "trade_date": day.isoformat(),
            "timing": normalized_timing,
            "amount": value,
        }
        key = str(flow_key).strip() if flow_key else _canonical_hash(payload)
        now = _now()
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if portfolio is None:
                raise ValueError("simulation portfolio does not exist")
            if str(portfolio.execution_adapter) == "pair":
                raise ValueError("pair research ledgers do not accept external cash flows")
            settled = connection.execute(
                select(func.count())
                .select_from(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.trade_date == day,
                )
            ).scalar_one()
            if settled:
                raise ValueError(
                    "external cash flows cannot be recorded for a settled trade date; "
                    "rebuild the ledger from the last verified state"
                )
            row = connection.execute(
                select(simulation_external_flows).where(
                    simulation_external_flows.c.portfolio_id == portfolio_id,
                    simulation_external_flows.c.flow_key == key,
                )
            ).first()
            created = row is None
            if created:
                connection.execute(
                    pg_insert(simulation_external_flows)
                    .values(
                        id=uuid.uuid4().hex,
                        portfolio_id=portfolio_id,
                        flow_key=key,
                        trade_date=day,
                        timing=normalized_timing,
                        amount=Decimal(str(value)),
                        note=(str(note).strip() or None) if note else None,
                        created_by=actor.strip(),
                        created_at=now,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[
                            simulation_external_flows.c.portfolio_id,
                            simulation_external_flows.c.flow_key,
                        ]
                    )
                )
                row = connection.execute(
                    select(simulation_external_flows).where(
                        simulation_external_flows.c.portfolio_id == portfolio_id,
                        simulation_external_flows.c.flow_key == key,
                    )
                ).first()
            if row is None:  # pragma: no cover - defensive
                raise RuntimeError("external cash flow ledger write did not land")
            stored = {
                "portfolio_id": str(row.portfolio_id),
                "trade_date": row.trade_date.isoformat(),
                "timing": str(row.timing),
                "amount": float(row.amount),
            }
            if stored != payload:
                raise ValueError(
                    "external cash flow key was reused with a different payload"
                )
            if created:
                connection.execute(
                    insert(simulation_events).values(
                        id=uuid.uuid4().hex,
                        portfolio_id=portfolio_id,
                        batch_id=None,
                        trade_date=day,
                        severity="info",
                        event_type="external_cash_flow_recorded",
                        instrument=None,
                        reason=f"external_{normalized_timing}_flow",
                        details_json={
                            "flow_id": str(row.id),
                            "flow_key": key,
                            "timing": normalized_timing,
                            "amount": value,
                        },
                        created_at=now,
                    )
                )
        return {
            "id": str(row.id),
            "portfolio_id": str(portfolio_id),
            "flow_key": key,
            "trade_date": day.isoformat(),
            "timing": normalized_timing,
            "amount": value,
            "created": created,
        }

    def record_final_fee(
        self,
        portfolio_id: str,
        *,
        fill_id: str,
        final_fee: float,
        evidence_sha256: str,
        actor: str,
        source: str = "user_import",
        adjustment_key: str | None = None,
    ) -> dict[str, Any]:
        """Append a final fee confirmation and apply only its economic delta.

        The original fill is immutable.  A confirmation applies
        ``final_fee - (fill fee + prior adjustments)`` once, updates the
        latest unreviewed day-end NAV, and writes an explicit cash-flow and
        event trail.  Older or independently reviewed NAV cannot be silently
        restated; those cases require rebuilding from the last verified
        ledger state.
        """

        self.safe_mode.assert_inactive(action="simulation final fee confirmation")
        normalized_source = str(source).strip().lower()
        if normalized_source not in {"end_of_day", "user_import"}:
            raise ValueError("final fee source must be end_of_day or user_import")
        creator = str(actor).strip()
        if len(creator) < 2:
            raise ValueError("a responsible final fee actor is required")
        evidence = str(evidence_sha256).strip().lower()
        if not _is_sha256(evidence):
            raise ValueError("final fee evidence must be a SHA-256 digest")
        try:
            confirmed_final = Decimal(str(final_fee)).quantize(Decimal("0.000001"))
        except Exception as exc:
            raise ValueError("final fee must be a finite non-negative amount") from exc
        if not confirmed_final.is_finite() or confirmed_final < 0:
            raise ValueError("final fee must be a finite non-negative amount")
        payload = {
            "portfolio_id": str(portfolio_id),
            "fill_id": str(fill_id),
            "final_fee": format(confirmed_final, "f"),
            "evidence_sha256": evidence,
            "source": normalized_source,
        }
        key = (
            str(adjustment_key).strip()
            if adjustment_key
            else _canonical_hash(payload)
        )
        if not key:
            raise ValueError("final fee adjustment key is required")
        now = _now()

        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            fill = connection.execute(
                select(
                    simulation_fills,
                    simulation_batches.c.portfolio_id.label("fill_portfolio_id"),
                    simulation_batches.c.trade_date.label("fill_trade_date"),
                    simulation_batches.c.status.label("batch_status"),
                )
                .join(
                    simulation_batches,
                    simulation_batches.c.id == simulation_fills.c.batch_id,
                )
                .where(simulation_fills.c.id == fill_id)
            ).first()
            if fill is None or str(fill.fill_portfolio_id) != str(portfolio_id):
                raise KeyError(fill_id)
            if str(fill.batch_status) != "succeeded":
                raise ValueError("final fee confirmation requires a succeeded fill batch")

            existing = connection.execute(
                select(simulation_fee_adjustments).where(
                    simulation_fee_adjustments.c.portfolio_id == portfolio_id,
                    simulation_fee_adjustments.c.adjustment_key == key,
                )
            ).first()
            if existing is not None:
                stored = {
                    "portfolio_id": str(existing.portfolio_id),
                    "fill_id": str(existing.fill_id),
                    "final_fee": format(Decimal(existing.final_fee), "f"),
                    "evidence_sha256": str(existing.evidence_sha256),
                    "source": str(existing.source),
                }
                if stored != payload:
                    raise ValueError(
                        "final fee adjustment key was reused with a different payload"
                    )
                result = row_dict(existing)
                result["created"] = False
                return result

            latest_nav = connection.execute(
                select(simulation_nav)
                .where(simulation_nav.c.portfolio_id == portfolio_id)
                .order_by(simulation_nav.c.trade_date.desc())
                .limit(1)
                .with_for_update()
            ).first()
            trade_date = fill.fill_trade_date
            if latest_nav is None or latest_nav.trade_date != trade_date:
                raise ValueError(
                    "final fee can only adjust the latest settled trade date; "
                    "rebuild older ledger state"
                )
            if latest_nav.reviewed_at is not None:
                raise ValueError(
                    "independently reviewed NAV is immutable; rebuild from the "
                    "last verified state"
                )

            previous_adjustments = connection.scalar(
                select(func.coalesce(func.sum(simulation_fee_adjustments.c.adjustment_amount), 0))
                .where(simulation_fee_adjustments.c.fill_id == fill_id)
            )
            previously_confirmed = (
                Decimal(fill.fee) + Decimal(previous_adjustments or 0)
            ).quantize(Decimal("0.000001"))
            delta = (confirmed_final - previously_confirmed).quantize(
                Decimal("0.000001")
            )
            new_cash = (Decimal(latest_nav.cash) - delta).quantize(
                Decimal("0.000001")
            )
            new_nav = (Decimal(latest_nav.nav) - delta).quantize(
                Decimal("0.000001")
            )
            if new_cash < 0:
                raise RuntimeError("final fee adjustment would create negative cash")

            prior_rows = connection.execute(
                select(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.trade_date < trade_date,
                )
                .order_by(simulation_nav.c.trade_date)
            ).all()
            if prior_rows:
                prior_nav = float(prior_rows[-1].nav)
                prior_wealth = (
                    float(prior_rows[-1].investment_wealth)
                    if prior_rows[-1].investment_wealth is not None
                    else None
                )
                previous_cny_peak = max(float(item.nav) for item in prior_rows)
                wealth_points = [
                    float(item.investment_wealth)
                    for item in prior_rows
                    if item.investment_wealth is not None
                ]
                previous_twr_peak = max(wealth_points) if wealth_points else None
            else:
                initial_deposit = connection.scalar(
                    select(simulation_cash_flows.c.amount)
                    .where(
                        simulation_cash_flows.c.portfolio_id == portfolio_id,
                        simulation_cash_flows.c.flow_type == "initial_deposit",
                    )
                    .order_by(simulation_cash_flows.c.created_at)
                    .limit(1)
                )
                if initial_deposit is None:
                    raise RuntimeError("simulation initial deposit is missing")
                prior_nav = float(initial_deposit)
                prior_wealth = 1.0
                previous_cny_peak = prior_nav
                previous_twr_peak = 1.0
            cny_peak = max(previous_cny_peak, float(new_nav))
            unitized = chain_unitized_day(
                prior_nav=prior_nav,
                nav=float(new_nav),
                flow_open=float(latest_nav.external_flow_open),
                flow_close=float(latest_nav.external_flow_close),
                prior_wealth=prior_wealth,
                prior_high_water_mark=previous_twr_peak,
            )

            adjustment_id = uuid.uuid4().hex
            connection.execute(
                insert(simulation_fee_adjustments).values(
                    id=adjustment_id,
                    portfolio_id=portfolio_id,
                    fill_id=fill_id,
                    batch_id=fill.batch_id,
                    adjustment_key=key,
                    trade_date=trade_date,
                    source=normalized_source,
                    previously_confirmed_fee=previously_confirmed,
                    final_fee=confirmed_final,
                    adjustment_amount=delta,
                    evidence_sha256=evidence,
                    created_by=creator,
                    created_at=now,
                )
            )
            if delta:
                connection.execute(
                    insert(simulation_cash_flows).values(
                        id=uuid.uuid4().hex,
                        portfolio_id=portfolio_id,
                        batch_id=fill.batch_id,
                        trade_date=trade_date,
                        flow_type="fee_adjustment",
                        amount=-delta,
                        balance_after=new_cash,
                        reference_id=adjustment_id,
                        created_at=now,
                    )
                )
                cash_event_key = f"final-fee:{adjustment_id}"
                if delta > 0:
                    self._allocate_free_cash(
                        connection,
                        portfolio_id=str(portfolio_id),
                        event_key=cash_event_key,
                        event_type="consume_free",
                        amount=delta,
                        as_of=now,
                        now=now,
                        batch_id=str(fill.batch_id),
                        order_id=str(fill.order_id),
                        details={
                            "adjustment_id": adjustment_id,
                            "fill_id": str(fill_id),
                        },
                    )
                else:
                    self._create_cash_lot(
                        connection,
                        portfolio_id=str(portfolio_id),
                        event_key=cash_event_key,
                        source_type="fee_adjustment_refund",
                        source_reference_id=adjustment_id,
                        amount=-delta,
                        tradable_at=now,
                        withdrawable_at=now,
                        occurred_at=now,
                        now=now,
                        batch_id=str(fill.batch_id),
                        order_id=str(fill.order_id),
                        details={"fill_id": str(fill_id)},
                    )
                self._assert_cash_lots_reconcile(
                    connection,
                    str(portfolio_id),
                    expected_cash=new_cash,
                    as_of=now,
                )
            connection.execute(
                update(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.trade_date == trade_date,
                )
                .values(
                    cash=new_cash,
                    nav=new_nav,
                    daily_return=float(new_nav) / prior_nav - 1.0,
                    drawdown=float(new_nav) / cny_peak - 1.0,
                    twr_daily_return=unitized["daily_return"],
                    investment_wealth=unitized["investment_wealth"],
                    twr_drawdown=unitized["drawdown"],
                    twr_status=unitized["status"],
                )
            )
            twr_peak = unitized["high_water_mark"]
            connection.execute(
                update(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .values(
                    cash=new_cash,
                    nav=new_nav,
                    high_water_mark=Decimal(str(cny_peak)),
                    investment_wealth=unitized["investment_wealth"],
                    twr_high_water_mark=twr_peak,
                    updated_at=now,
                )
            )
            all_adjustments = connection.execute(
                select(
                    func.count(simulation_fee_adjustments.c.id),
                    func.coalesce(
                        func.sum(simulation_fee_adjustments.c.adjustment_amount), 0
                    ),
                ).where(simulation_fee_adjustments.c.batch_id == fill.batch_id)
            ).one()
            batch_summary = connection.scalar(
                select(simulation_batches.c.summary_json).where(
                    simulation_batches.c.id == fill.batch_id
                )
            )
            summary = dict(batch_summary or {})
            summary.update(
                {
                    "cash": float(new_cash),
                    "nav": float(new_nav),
                    "fee_adjustments": {
                        "count": int(all_adjustments[0]),
                        "net_amount": float(all_adjustments[1]),
                        "latest_adjustment_id": adjustment_id,
                    },
                }
            )
            connection.execute(
                update(simulation_batches)
                .where(simulation_batches.c.id == fill.batch_id)
                .values(summary_json=summary)
            )
            connection.execute(
                insert(simulation_events).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio_id,
                    batch_id=fill.batch_id,
                    trade_date=trade_date,
                    severity="info",
                    event_type="final_fee_adjustment",
                    instrument=str(fill.instrument),
                    reason="final_fee_minus_previously_confirmed_fee",
                    details_json={
                        "adjustment_id": adjustment_id,
                        "adjustment_key": key,
                        "fill_id": str(fill_id),
                        "source": normalized_source,
                        "previously_confirmed_fee": float(previously_confirmed),
                        "final_fee": float(confirmed_final),
                        "adjustment_amount": float(delta),
                        "evidence_sha256": evidence,
                    },
                    created_at=now,
                )
            )

        return {
            "id": adjustment_id,
            **payload,
            "adjustment_key": key,
            "trade_date": trade_date,
            "previously_confirmed_fee": previously_confirmed,
            "adjustment_amount": delta,
            "created_by": creator,
            "created_at": now,
            "created": True,
        }

    @staticmethod
    def _resolve_source(connection: Any, source_type: str, source_id: str) -> dict[str, Any]:
        if source_type == "recommendation":
            row = connection.execute(
                select(recommendation_portfolios).where(
                    recommendation_portfolios.c.id == source_id
                )
            ).first()
            if row is None:
                raise KeyError(source_id)
            if row.status != "active":
                raise ValueError("simulation requires an active recommendation portfolio")
            version = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == row.strategy_version_id
                )
            ).one()
            if version.status != "approved" or version.is_legacy:
                raise ValueError(
                    "simulation recommendation must reference an approved non-legacy "
                    "strategy version"
                )
            version_config = dict(version.config_json or {})
            contract_hash = SimulationStore._validated_version_contract_hash(
                version, version_config
            )
            return {
                "dataset": str(row.dataset),
                "execution_adapter": "pair" if version.strategy_type == "pair" else "long_only",
                "execution_contract_hash": contract_hash,
                "signal_frequency": str(
                    version.signal_frequency
                    or version_config.get("signal_frequency")
                    or "day"
                ),
                "signal_horizon": str(version.signal_horizon or "1d"),
                "execution_frequency": str(
                    version.execution_frequency
                    or version_config.get("execution_frequency")
                    or ""
                ),
                "execution_method": str(version_config.get("execution_method") or ""),
                "benchmark": str(version.benchmark or ""),
                "config": version_config,
            }
        if source_type == "strategy_version":
            row = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == source_id)
            ).first()
            if row is None:
                raise KeyError(source_id)
            if row.status != "approved" or row.is_legacy:
                raise ValueError("simulation requires an approved non-legacy strategy version")
            backtest = connection.execute(
                select(backtest_runs)
                .where(
                    backtest_runs.c.strategy_version_id == source_id,
                    backtest_runs.c.status == "succeeded",
                    backtest_runs.c.is_legacy.is_(False),
                )
                .order_by(backtest_runs.c.finished_at.desc())
                .limit(1)
            ).first()
            if backtest is None:
                raise ValueError("strategy simulation requires a successful formal Qlib backtest")
            version_config = dict(row.config_json or {})
            contract_hash = SimulationStore._validated_version_contract_hash(
                row, version_config
            )
            if str(backtest.execution_contract_hash) != contract_hash:
                raise ValueError(
                    "strategy formal backtest contract does not match the approved version"
                )
            if str(row.evidence_mode) == EVIDENCE_MODE_REPLAY:
                require_qualification(
                    connection,
                    version=row,
                    backtest=backtest,
                )
            return {
                "dataset": str(backtest.dataset),
                "execution_adapter": "pair" if row.strategy_type == "pair" else "long_only",
                "execution_contract_hash": contract_hash,
                "signal_frequency": str(
                    row.signal_frequency
                    or version_config.get("signal_frequency")
                    or "day"
                ),
                "signal_horizon": str(row.signal_horizon or "1d"),
                "execution_frequency": str(
                    row.execution_frequency
                    or version_config.get("execution_frequency")
                    or ""
                ),
                "execution_method": str(version_config.get("execution_method") or ""),
                "benchmark": str(row.benchmark or ""),
                "config": version_config,
                "formal_backtest_id": str(backtest.id),
            }
        if source_type == "allocation":
            row = connection.execute(
                select(strategy_allocations).where(strategy_allocations.c.id == source_id)
            ).first()
            if row is None:
                raise KeyError(source_id)
            if row.status != "active" or row.is_legacy:
                raise ValueError("simulation requires an approved active allocation")
            member_benchmarks = {
                str(value).strip().upper()
                for value in connection.scalars(
                    select(strategy_versions.c.benchmark)
                    .select_from(
                        strategy_allocation_members.join(
                            strategy_versions,
                            strategy_allocation_members.c.strategy_version_id
                            == strategy_versions.c.id,
                        )
                    )
                    .where(
                        strategy_allocation_members.c.allocation_id == source_id
                    )
                )
                if str(value or "").strip()
            }
            return {
                "dataset": str(row.dataset),
                "execution_adapter": "long_only",
                "execution_contract_hash": _canonical_hash(
                    {
                        "source_type": "allocation",
                        "source_id": str(row.id),
                        "dataset": str(row.dataset),
                        "approval_simulation_evidence": dict(row.analysis_json or {}).get(
                            "approval_simulation_evidence"
                        ),
                    }
                ),
                "signal_frequency": "day",
                "signal_horizon": "1d",
                "execution_frequency": "",
                "execution_method": "",
                "benchmark": (
                    next(iter(member_benchmarks))
                    if len(member_benchmarks) == 1
                    else ""
                ),
                "config": CostModelConfig().to_dict(),
                "policy_mode": "self_contained",
            }
        raise ValueError("unsupported simulation source type")

    @staticmethod
    def _validated_version_contract_hash(
        version: Any, config: dict[str, Any]
    ) -> str:
        if str(version.strategy_type) == "pair":
            expected = _pair_source_contract_hash(config)
        else:
            require_strategy_execution_contract(config)
            expected = str(config["execution_contract_hash"])
        stored = str(version.execution_contract_hash or "")
        if not _is_sha256(stored) or stored != expected:
            raise ValueError(
                "approved strategy version execution contract is missing or inconsistent"
            )
        return stored

    @classmethod
    def _require_current_source_contract(cls, connection: Any, portfolio: Any) -> dict[str, Any]:
        source = cls._resolve_source(
            connection, str(portfolio.source_type), str(portfolio.source_id)
        )
        source_hash = str(source.get("execution_contract_hash") or "")
        if not _is_sha256(source_hash):
            raise ValueError(
                "simulation source uses a legacy or incomplete execution contract; "
                "create a new approved strategy version and simulation account"
            )
        if source_hash != str(portfolio.execution_contract_hash):
            raise ValueError("simulation execution contract no longer matches its governed source")
        policy = dict(portfolio.execution_policy_json or {})
        if (
            policy.get("simulation_contract_version")
            != SIMULATION_EXECUTION_SEMANTICS_VERSION
            or policy.get("source_execution_contract_hash") != source_hash
            or policy.get("execution_frequency") != str(portfolio.execution_frequency)
        ):
            raise ValueError("simulation execution semantics are incomplete or inconsistent")
        if str(policy.get("execution_algorithm") or "") != str(
            portfolio.execution_algorithm
        ):
            raise ValueError("simulation execution algorithm has drifted from its policy")
        try:
            cost_model = CostModelConfig.from_mapping(policy.get("cost_model"))
        except (TypeError, ValueError) as exc:
            raise ValueError("simulation cost contract is missing or invalid") from exc
        if cost_model.version != str(portfolio.cost_schedule_version):
            raise ValueError("simulation cost schedule has drifted from its policy")
        if str(portfolio.execution_adapter) == "long_only" and source.get(
            "policy_mode"
        ) != "self_contained":
            expected_policy = cls._governed_execution_policy(
                source, {}, adapter="long_only"
            )
            for field in (
                "execution_algorithm",
                "slice_minutes",
                "max_slices",
                "max_participation",
                "volume_profile_method",
                "volume_profile_lookback_days",
            ):
                if policy.get(field) != expected_policy.get(field):
                    raise ValueError(
                        "simulation execution policy no longer matches its approved "
                        f"source contract: {field}"
                    )
            governed_cost_model = CostModelConfig.from_mapping(
                source.get("config")
            ).to_dict()
            if cost_model.to_dict() != governed_cost_model:
                raise ValueError(
                    "simulation cost parameters no longer match the approved source contract"
                )
        semantics = _simulation_semantics_payload(
            source_type=str(portfolio.source_type),
            source_id=str(portfolio.source_id),
            source_execution_contract_hash=source_hash,
            execution_adapter=str(portfolio.execution_adapter),
            execution_frequency=str(portfolio.execution_frequency),
            daily_dataset=str(portfolio.daily_dataset),
            daily_dataset_identity_sha256=str(
                portfolio.daily_dataset_identity_sha256
            ),
            daily_dataset_lineage_id=str(portfolio.daily_dataset_lineage_id),
            execution_dataset=str(portfolio.execution_dataset),
            execution_dataset_identity_sha256=str(
                portfolio.execution_dataset_identity_sha256
            ),
            execution_dataset_lineage_id=str(
                portfolio.execution_dataset_lineage_id
            ),
            execution_field_contract_version=str(
                portfolio.execution_field_contract_version
            ),
            execution_engine_version=str(portfolio.execution_engine_version),
            execution_policy=policy,
        )
        if str(policy.get("simulation_semantics_sha256") or "") != _canonical_hash(
            semantics
        ):
            raise ValueError("simulation execution semantics failed immutable verification")
        return source

    @staticmethod
    def _allocation_dataset_lineage(allocation: Any) -> str:
        lineage = str(
            dict(allocation.analysis_json or {}).get("allocation_dataset_lineage_id")
            or ""
        ).lower()
        if lineage and not _is_sha256(lineage):
            raise ValueError("allocation dataset lineage evidence is invalid")
        return lineage

    @staticmethod
    def _validated_allocation_rebind_datasets(
        daily_dataset: dict[str, Any] | None,
        execution_dataset: dict[str, Any] | None,
    ) -> dict[str, str] | None:
        if daily_dataset is None and execution_dataset is None:
            return None
        if daily_dataset is None or execution_dataset is None:
            raise ValueError(
                "allocation source replacement requires both daily and execution datasets"
            )
        daily_provenance = dict(daily_dataset.get("provenance") or {})
        execution_provenance = dict(execution_dataset.get("provenance") or {})
        require_daily_qlib_contract(daily_provenance)
        execution_frequency = str(execution_provenance.get("frequency") or "")
        if execution_frequency == "day":
            require_daily_qlib_contract(execution_provenance)
            require_native_daily_execution_controls(
                execution_provenance,
                start=str(execution_dataset.get("end_date") or date.today().isoformat()),
            )
            execution_field_contract = str(
                execution_provenance.get("field_contract_version") or ""
            )
        else:
            require_minute_execution_contract(
                execution_provenance,
                frequency=execution_frequency,
                simulation_eligible=True,
            )
            execution_field_contract = str(
                execution_provenance.get("execution_contract_version") or ""
            )
        daily_source = str(daily_provenance.get("source_lineage_id") or "")
        execution_source = str(execution_provenance.get("source_lineage_id") or "")
        if not _is_sha256(daily_source) or daily_source != execution_source:
            raise ValueError(
                "allocation replacement datasets must share one verified source lineage"
            )
        bindings = {
            "daily_dataset": str(daily_dataset.get("name") or ""),
            "daily_dataset_identity_sha256": str(
                daily_provenance.get("dataset_identity_sha256") or ""
            ),
            "daily_dataset_lineage_id": str(
                daily_provenance.get("dataset_lineage_id") or ""
            ),
            "daily_field_contract_version": str(
                daily_provenance.get("field_contract_version") or ""
            ),
            "execution_dataset": str(execution_dataset.get("name") or ""),
            "execution_dataset_identity_sha256": str(
                execution_provenance.get("dataset_identity_sha256") or ""
            ),
            "execution_dataset_lineage_id": str(
                execution_provenance.get("dataset_lineage_id") or ""
            ),
            "execution_field_contract_version": execution_field_contract,
            "execution_frequency": execution_frequency,
        }
        required_hashes = (
            "daily_dataset_identity_sha256",
            "daily_dataset_lineage_id",
            "execution_dataset_identity_sha256",
            "execution_dataset_lineage_id",
        )
        if (
            not bindings["daily_dataset"]
            or not bindings["execution_dataset"]
            or any(not _is_sha256(bindings[field]) for field in required_hashes)
        ):
            raise ValueError(
                "allocation replacement datasets require immutable identities and lineages"
            )
        return bindings

    @staticmethod
    def _require_allocation_rebind_quiescent(
        connection: Any, portfolio_id: str
    ) -> None:
        queued_batches = connection.scalars(
            select(simulation_batches.c.id)
            .where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_batches.c.status == "queued",
            )
            .limit(5)
        ).all()
        working_orders = connection.scalars(
            select(simulation_orders.c.id)
            .select_from(
                simulation_orders.join(
                    simulation_batches,
                    simulation_batches.c.id == simulation_orders.c.batch_id,
                )
            )
            .where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_orders.c.status.in_(OPEN_STATUSES),
            )
            .limit(5)
        ).all()
        if queued_batches or working_orders:
            raise ValueError(
                "three-horizon account cutover is waiting for all queued batches "
                "and working orders to settle"
            )

    def prepare_allocation_source_replacement(
        self,
        portfolio_id: str,
        *,
        old_allocation_id: str,
        new_allocation_id: str,
        actor: str,
        daily_dataset: dict[str, Any] | None = None,
        execution_dataset: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Pause one quiescent primary ledger before AllocationStore replacement.

        Locking the portfolio before inspecting batches/orders serializes this
        preflight with every order-plan writer.  Once paused, no execution can
        start while AllocationStore atomically replaces the allocation version.
        """

        self.safe_mode.assert_inactive(action="simulation allocation source replacement")
        responsible = str(actor).strip()
        if len(responsible) < 2:
            raise ValueError("a responsible allocation cutover actor is required")
        old_id = str(old_allocation_id).strip()
        new_id = str(new_allocation_id).strip()
        if not old_id or not new_id or old_id == new_id:
            raise ValueError("allocation cutover requires distinct old and new sources")
        replacement_bindings = self._validated_allocation_rebind_datasets(
            daily_dataset, execution_dataset
        )
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(:lock_key)"),
                {"lock_key": THREE_HORIZON_ALLOCATION_CUTOVER_LOCK_KEY},
            )
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if str(portfolio.source_type) != "allocation":
                raise ValueError("only an allocation simulation may replace its source")
            allocations = connection.execute(
                select(strategy_allocations)
                .where(strategy_allocations.c.id.in_([old_id, new_id]))
                .order_by(strategy_allocations.c.id)
                .with_for_update()
            ).all()
            by_id = {str(item.id): item for item in allocations}
            if set(by_id) != {old_id, new_id}:
                raise ValueError("allocation cutover source versions are unavailable")
            if str(portfolio.source_id) == new_id:
                self._require_current_source_contract(connection, portfolio)
                return self._portfolio_dict(portfolio)
            if (
                str(portfolio.source_id) != old_id
                or str(by_id[old_id].status) != "active"
                or str(by_id[new_id].status) != "draft"
            ):
                raise ValueError(
                    "allocation cutover preflight does not match the serving source states"
                )
            if (
                abs(float(by_id[old_id].total_capital) - float(by_id[new_id].total_capital))
                > 1e-6
                or abs(float(portfolio.initial_cash) - float(by_id[new_id].total_capital))
                > 1e-6
            ):
                raise ValueError(
                    "allocation source replacement cannot change account principal"
                )
            old_lineage = self._allocation_dataset_lineage(by_id[old_id])
            new_lineage = self._allocation_dataset_lineage(by_id[new_id])
            account_lineage = str(portfolio.daily_dataset_lineage_id)
            if old_lineage and old_lineage != account_lineage:
                raise ValueError(
                    "serving allocation no longer matches the primary account dataset lineage"
                )
            if new_lineage != account_lineage:
                if replacement_bindings is None:
                    raise ValueError(
                        "allocation dataset-lineage rollover requires a verified "
                        "replacement dataset"
                    )
                if (
                    replacement_bindings["daily_dataset_lineage_id"] != new_lineage
                    or replacement_bindings["execution_dataset_lineage_id"] != new_lineage
                    or replacement_bindings["daily_dataset"] != str(by_id[new_id].dataset)
                ):
                    raise ValueError(
                        "replacement allocation and dataset-lineage evidence do not match"
                    )
            self._require_current_source_contract(connection, portfolio)
            self._require_allocation_rebind_quiescent(connection, str(portfolio.id))
            connection.execute(
                update(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio.id)
                .values(status="paused", updated_at=_now())
            )
        return self.get(portfolio_id)

    def abort_allocation_source_replacement(
        self,
        portfolio_id: str,
        *,
        old_allocation_id: str,
        new_allocation_id: str,
    ) -> dict[str, Any]:
        """Restore the old service when allocation approval did not commit."""

        old_id = str(old_allocation_id).strip()
        new_id = str(new_allocation_id).strip()
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(:lock_key)"),
                {"lock_key": THREE_HORIZON_ALLOCATION_CUTOVER_LOCK_KEY},
            )
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if str(portfolio.source_id) == new_id:
                return self._portfolio_dict(portfolio)
            old = connection.execute(
                select(strategy_allocations)
                .where(strategy_allocations.c.id == old_id)
                .with_for_update()
            ).first()
            new = connection.execute(
                select(strategy_allocations)
                .where(strategy_allocations.c.id == new_id)
                .with_for_update()
            ).first()
            if (
                str(portfolio.source_type) != "allocation"
                or str(portfolio.source_id) != old_id
                or old is None
                or str(old.status) != "active"
                or new is None
                or str(new.status) != "draft"
            ):
                raise ValueError(
                    "the previous three-horizon account can no longer be resumed safely"
                )
            self._require_current_source_contract(connection, portfolio)
            self._require_allocation_rebind_quiescent(connection, str(portfolio.id))
            connection.execute(
                update(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio.id)
                .values(status="active", updated_at=_now())
            )
        return self.get(portfolio_id)

    def replace_allocation_source(
        self,
        portfolio_id: str,
        *,
        old_allocation_id: str,
        new_allocation_id: str,
        actor: str,
        daily_dataset: dict[str, Any] | None = None,
        execution_dataset: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Rebind one allocation ledger in place without copying financial rows."""

        self.safe_mode.assert_inactive(action="simulation allocation source replacement")
        responsible = str(actor).strip()
        if len(responsible) < 2:
            raise ValueError("a responsible allocation cutover actor is required")
        old_id = str(old_allocation_id).strip()
        new_id = str(new_allocation_id).strip()
        if not old_id or not new_id or old_id == new_id:
            raise ValueError("allocation cutover requires distinct old and new sources")
        replacement_bindings = self._validated_allocation_rebind_datasets(
            daily_dataset, execution_dataset
        )
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_key)"),
                    {"lock_key": THREE_HORIZON_ALLOCATION_CUTOVER_LOCK_KEY},
                )
                portfolio = connection.execute(
                    select(simulation_portfolios)
                    .where(simulation_portfolios.c.id == portfolio_id)
                    .with_for_update()
                ).first()
                if portfolio is None:
                    raise KeyError(portfolio_id)
                if str(portfolio.source_type) != "allocation":
                    raise ValueError("only an allocation simulation may replace its source")
                if str(portfolio.source_id) == new_id:
                    self._require_current_source_contract(connection, portfolio)
                    if replacement_bindings is not None and any(
                        str(getattr(portfolio, field)) != str(value)
                        for field, value in replacement_bindings.items()
                        if field != "execution_frequency"
                    ):
                        raise ValueError(
                            "completed allocation cutover does not match its replacement dataset"
                        )
                    prior_event = connection.execute(
                        select(simulation_events.c.details_json).where(
                            simulation_events.c.portfolio_id == portfolio.id,
                            simulation_events.c.event_type
                            == "allocation_source_rebound",
                        )
                    ).all()
                    governed_retry = False
                    for row in prior_event:
                        try:
                            validate_allocation_source_rebind_event(
                                dict(row.details_json or {}),
                                portfolio_id=str(portfolio.id),
                                old_allocation_id=old_id,
                                new_allocation_id=new_id,
                            )
                        except ValueError:
                            continue
                        governed_retry = True
                        break
                    if not governed_retry:
                        raise ValueError(
                            "allocation source already changed without a governed cutover event"
                        )
                    if str(portfolio.status) != "active":
                        connection.execute(
                            update(simulation_portfolios)
                            .where(simulation_portfolios.c.id == portfolio.id)
                            .values(status="active", updated_at=_now())
                        )
                    replayed = self._portfolio_dict(portfolio)
                    replayed["status"] = "active"
                    return replayed

                allocations = connection.execute(
                    select(strategy_allocations)
                    .where(strategy_allocations.c.id.in_([old_id, new_id]))
                    .order_by(strategy_allocations.c.id)
                    .with_for_update()
                ).all()
                by_id = {str(item.id): item for item in allocations}
                if set(by_id) != {old_id, new_id}:
                    raise ValueError("allocation cutover source versions are unavailable")
                old = by_id[old_id]
                new = by_id[new_id]
                if (
                    str(portfolio.source_id) != old_id
                    or str(portfolio.status) != "paused"
                    or str(old.status) != "paused"
                    or str(new.status) != "active"
                ):
                    raise ValueError(
                        "allocation source replacement is outside its safe paused cutover"
                    )
                self._require_allocation_rebind_quiescent(connection, str(portfolio.id))
                if (
                    abs(float(old.total_capital) - float(new.total_capital)) > 1e-6
                    or abs(float(portfolio.initial_cash) - float(new.total_capital)) > 1e-6
                ):
                    raise ValueError(
                        "allocation source replacement cannot change account principal"
                    )
                old_lineage = self._allocation_dataset_lineage(old)
                new_lineage = self._allocation_dataset_lineage(new)
                if old_lineage and old_lineage != str(
                    portfolio.daily_dataset_lineage_id
                ):
                    raise ValueError(
                        "serving allocation no longer matches the primary account dataset lineage"
                    )
                if replacement_bindings is not None:
                    if (
                        replacement_bindings["daily_dataset_lineage_id"] != new_lineage
                        or replacement_bindings["execution_dataset_lineage_id"] != new_lineage
                        or replacement_bindings["daily_dataset"] != str(new.dataset)
                        or replacement_bindings["execution_frequency"]
                        != str(portfolio.execution_frequency)
                    ):
                        raise ValueError(
                            "replacement allocation and dataset evidence do not match"
                        )
                elif new_lineage:
                    if new_lineage not in {
                        str(portfolio.daily_dataset_lineage_id),
                        str(portfolio.execution_dataset_lineage_id),
                    } or str(portfolio.daily_dataset_lineage_id) != str(
                        portfolio.execution_dataset_lineage_id
                    ):
                        raise ValueError(
                            "allocation dataset-lineage rollover requires verified datasets"
                        )
                elif not (
                    str(old.dataset)
                    == str(new.dataset)
                    == str(portfolio.daily_dataset)
                ):
                    raise ValueError(
                        "allocation source replacement has no lineage proof for dataset rollover"
                    )
                target_execution_dataset = (
                    replacement_bindings["execution_dataset"]
                    if replacement_bindings is not None
                    else str(portfolio.execution_dataset)
                )
                conflict = connection.execute(
                    select(simulation_portfolios.c.id).where(
                        simulation_portfolios.c.source_type == "allocation",
                        simulation_portfolios.c.source_id == new_id,
                        simulation_portfolios.c.execution_dataset
                        == target_execution_dataset,
                        simulation_portfolios.c.id != portfolio.id,
                    )
                ).first()
                if conflict is not None:
                    raise ValueError(
                        "replacement allocation already owns another simulation ledger"
                    )
                source = self._resolve_source(connection, "allocation", new_id)
                if replacement_bindings is not None and str(source.get("dataset") or "") != str(
                    replacement_bindings["daily_dataset"]
                ):
                    raise ValueError(
                        "replacement dataset does not match the active allocation source"
                    )
                benchmark = str(source.get("benchmark") or "").strip().upper()
                if not benchmark or benchmark != str(portfolio.benchmark or "").upper():
                    raise ValueError(
                        "allocation source replacement changed the account benchmark"
                    )
                new_contract_hash = str(source.get("execution_contract_hash") or "")
                if not _is_sha256(new_contract_hash):
                    raise ValueError("replacement allocation has no sealed execution contract")
                cost_model = CostModelConfig.from_mapping(source.get("config"))
                if cost_model.version != str(portfolio.cost_schedule_version):
                    raise ValueError(
                        "replacement allocation changed the account cost schedule"
                    )
                current_policy = dict(portfolio.execution_policy_json or {})
                old_semantics_sha256 = str(
                    current_policy.get("simulation_semantics_sha256") or ""
                )
                normalized_policy = self._governed_execution_policy(
                    source,
                    current_policy,
                    adapter=str(portfolio.execution_adapter),
                )
                if (
                    normalized_policy.get("execution_algorithm")
                    != str(portfolio.execution_algorithm)
                    or normalized_policy.get("execution_frequency")
                    != str(portfolio.execution_frequency)
                ):
                    raise ValueError(
                        "replacement allocation changed the account execution policy"
                    )
                bound_daily_dataset = daily_dataset or {
                    "name": str(portfolio.daily_dataset),
                    "provenance": {
                        "dataset_identity_sha256": str(
                            portfolio.daily_dataset_identity_sha256
                        ),
                        "dataset_lineage_id": str(
                            portfolio.daily_dataset_lineage_id
                        ),
                    },
                }
                bound_execution_dataset = execution_dataset or {
                    "name": str(portfolio.execution_dataset),
                    "provenance": {
                        "dataset_identity_sha256": str(
                            portfolio.execution_dataset_identity_sha256
                        ),
                        "dataset_lineage_id": str(
                            portfolio.execution_dataset_lineage_id
                        ),
                        "field_contract_version": str(
                            portfolio.execution_field_contract_version
                        ),
                        "execution_contract_version": str(
                            portfolio.execution_field_contract_version
                        ),
                    },
                }
                rebound_policy = self._bind_execution_semantics(
                    source=source,
                    source_type="allocation",
                    source_id=new_id,
                    adapter=str(portfolio.execution_adapter),
                    frequency=str(portfolio.execution_frequency),
                    daily_dataset=bound_daily_dataset,
                    execution_dataset=bound_execution_dataset,
                    policy=normalized_policy,
                    cost_model=cost_model,
                )
                event = build_allocation_source_rebind_event(
                    portfolio_id=str(portfolio.id),
                    old_allocation_id=old_id,
                    new_allocation_id=new_id,
                    actor=responsible,
                    old_execution_contract_hash=str(
                        portfolio.execution_contract_hash
                    ),
                    new_execution_contract_hash=new_contract_hash,
                    old_simulation_semantics_sha256=old_semantics_sha256,
                    new_simulation_semantics_sha256=str(
                        rebound_policy["simulation_semantics_sha256"]
                    ),
                )
                now = _now()
                dataset_updates = (
                    {
                        field: value
                        for field, value in replacement_bindings.items()
                        if field != "execution_frequency"
                    }
                    if replacement_bindings is not None
                    else {}
                )
                connection.execute(
                    update(simulation_portfolios)
                    .where(simulation_portfolios.c.id == portfolio.id)
                    .values(
                        source_id=new_id,
                        status="active",
                        benchmark=benchmark,
                        execution_contract_hash=new_contract_hash,
                        execution_policy_json=rebound_policy,
                        **dataset_updates,
                        updated_at=now,
                    )
                )
                connection.execute(
                    pg_insert(simulation_events)
                    .values(
                        id=event["event_sha256"],
                        portfolio_id=portfolio.id,
                        batch_id=None,
                        trade_date=now.astimezone(SHANGHAI_TIMEZONE).date(),
                        severity="info",
                        event_type="allocation_source_rebound",
                        instrument=None,
                        reason=(
                            "progressive_horizon_activation_preserved_primary_ledger"
                        ),
                        details_json=event,
                        created_at=now,
                    )
                    .on_conflict_do_nothing(index_elements=[simulation_events.c.id])
                )
                stored_event = connection.scalar(
                    select(simulation_events.c.details_json).where(
                        simulation_events.c.id == event["event_sha256"]
                    )
                )
                if dict(stored_event or {}) != event:
                    raise ValueError(
                        "allocation source replacement event identity was reused"
                    )
                rebound = connection.execute(
                    select(simulation_portfolios).where(
                        simulation_portfolios.c.id == portfolio.id
                    )
                ).one()
                self._require_current_source_contract(connection, rebound)
        except IntegrityError as exc:
            raise ValueError(
                "allocation source replacement conflicts with another simulation ledger"
            ) from exc
        return self.get(portfolio_id)

    def set_status(self, portfolio_id: str, status: str) -> dict[str, Any]:
        if status not in {"active", "paused"}:
            raise ValueError("simulation status must be active or paused")
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if status == "active":
                if str(portfolio.execution_adapter) != "long_only":
                    raise ValueError(
                        "pair simulation writes are retired; historical pair ledgers are read-only"
                    )
                self._require_current_source_contract(connection, portfolio)
            result = connection.execute(
                update(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .values(status=status, updated_at=_now())
            )
            if not result.rowcount:
                raise KeyError(portfolio_id)
        return self.get(portfolio_id)

    def create_batch_for_snapshot(
        self,
        snapshot_id: str,
        *,
        actor: str = "recommendation-worker",
        data_root: Path | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        batches = self.create_batches_for_snapshot(
            snapshot_id, actor=actor, data_root=data_root
        )
        return batches[0] if batches else (None, False)

    @staticmethod
    def _portfolio_batch_dataset_bindings(portfolio: Any) -> dict[str, str]:
        return {
            "daily_dataset": str(portfolio.daily_dataset),
            "daily_dataset_identity_sha256": str(
                portfolio.daily_dataset_identity_sha256
            ),
            "daily_dataset_lineage_id": str(portfolio.daily_dataset_lineage_id),
            "execution_dataset": str(portfolio.execution_dataset),
            "execution_dataset_identity_sha256": str(
                portfolio.execution_dataset_identity_sha256
            ),
            "execution_dataset_lineage_id": str(
                portfolio.execution_dataset_lineage_id
            ),
        }

    @classmethod
    def _account_order_plan_dataset_bindings(
        cls,
        *,
        portfolio: Any,
        signal_date: date,
        trade_date: date,
        data_root: Path | None,
    ) -> dict[str, str]:
        bindings = cls._portfolio_batch_dataset_bindings(portfolio)
        daily_roll = str(portfolio.daily_roll_policy or "pinned")
        execution_roll = str(portfolio.execution_roll_policy or "pinned")
        if daily_roll == "pinned" and execution_roll == "pinned":
            return bindings
        if data_root is None:
            raise ValueError(
                "latest-compatible account order plan is waiting for governed datasets"
            )
        from .data_rollover import next_qlib_trading_date, select_qlib_dataset

        daily = select_qlib_dataset(
            data_root,
            anchor_name=str(portfolio.daily_dataset),
            roll_policy=daily_roll,
            lineage_id=str(portfolio.daily_dataset_lineage_id),
            required_date=signal_date,
        )
        execution = select_qlib_dataset(
            data_root,
            anchor_name=str(portfolio.execution_dataset),
            roll_policy=execution_roll,
            lineage_id=str(portfolio.execution_dataset_lineage_id),
            required_date=trade_date,
        )
        # The replay contract settles T+1 cash on the following Qlib session.
        # Do not freeze a batch until that calendar evidence is already present.
        next_qlib_trading_date(execution, trade_date)
        daily_provenance = dict(daily.get("provenance") or {})
        execution_provenance = dict(execution.get("provenance") or {})
        require_daily_qlib_contract(daily_provenance)
        if str(portfolio.execution_frequency) == "day":
            require_daily_qlib_contract(execution_provenance)
        else:
            require_minute_execution_contract(
                execution_provenance,
                frequency=str(portfolio.execution_frequency),
                simulation_eligible=True,
            )
        if str(daily_provenance.get("source_lineage_id") or "") != str(
            execution_provenance.get("source_lineage_id") or ""
        ):
            raise ValueError(
                "account daily and execution datasets do not share source lineage"
            )
        resolved = {
            "daily_dataset": str(daily["name"]),
            "daily_dataset_identity_sha256": str(
                daily_provenance["dataset_identity_sha256"]
            ),
            "daily_dataset_lineage_id": str(daily_provenance["dataset_lineage_id"]),
            "execution_dataset": str(execution["name"]),
            "execution_dataset_identity_sha256": str(
                execution_provenance["dataset_identity_sha256"]
            ),
            "execution_dataset_lineage_id": str(
                execution_provenance["dataset_lineage_id"]
            ),
        }
        if (
            resolved["daily_dataset_lineage_id"]
            != str(portfolio.daily_dataset_lineage_id)
            or resolved["execution_dataset_lineage_id"]
            != str(portfolio.execution_dataset_lineage_id)
        ):
            raise ValueError(
                "latest-compatible account datasets are outside the primary ledger lineage"
            )
        return resolved

    @staticmethod
    def _batch_simulation_semantics_sha256(
        portfolio: Any,
        bindings: dict[str, str],
    ) -> str:
        return _canonical_hash(
            _simulation_semantics_payload(
                source_type=str(portfolio.source_type),
                source_id=str(portfolio.source_id),
                source_execution_contract_hash=str(
                    portfolio.execution_contract_hash
                ),
                execution_adapter=str(portfolio.execution_adapter),
                execution_frequency=str(portfolio.execution_frequency),
                daily_dataset=bindings["daily_dataset"],
                daily_dataset_identity_sha256=bindings[
                    "daily_dataset_identity_sha256"
                ],
                daily_dataset_lineage_id=bindings["daily_dataset_lineage_id"],
                execution_dataset=bindings["execution_dataset"],
                execution_dataset_identity_sha256=bindings[
                    "execution_dataset_identity_sha256"
                ],
                execution_dataset_lineage_id=bindings[
                    "execution_dataset_lineage_id"
                ],
                execution_field_contract_version=str(
                    portfolio.execution_field_contract_version
                ),
                execution_engine_version=str(portfolio.execution_engine_version),
                execution_policy=dict(portfolio.execution_policy_json or {}),
            )
        )

    def _snapshot_batch_dataset_bindings(
        self,
        *,
        portfolio: Any,
        snapshot: Any,
        trade_date: date,
        data_root: Path | None,
    ) -> dict[str, str]:
        bindings = self._portfolio_batch_dataset_bindings(portfolio)
        daily_roll = str(portfolio.daily_roll_policy or "pinned")
        execution_roll = str(portfolio.execution_roll_policy or "pinned")
        if daily_roll == "pinned":
            if (
                str(snapshot.dataset) != bindings["daily_dataset"]
                or str(snapshot.dataset_identity_sha256)
                != bindings["daily_dataset_identity_sha256"]
            ):
                raise ValueError(
                    "pinned simulation daily dataset does not match the recommendation"
                )
        elif daily_roll != "latest_compatible":
            raise ValueError("simulation daily roll policy is invalid")
        if execution_roll not in {"pinned", "latest_compatible"}:
            raise ValueError("simulation execution roll policy is invalid")
        if daily_roll == "latest_compatible" or execution_roll == "latest_compatible":
            if data_root is None:
                raise ValueError(
                    "compatible forward simulation datasets are not available yet"
                )
            from .data_rollover import select_qlib_dataset
            from .services import list_qlib_datasets

            datasets = {
                str(item["name"]): item for item in list_qlib_datasets(data_root)
            }
            daily = datasets.get(str(snapshot.dataset))
            if daily is None or not daily.get("ready"):
                raise ValueError(
                    "recommendation daily dataset is not ready for forward simulation"
                )
            daily_provenance = dict(daily.get("provenance") or {})
            if (
                str(daily_provenance.get("dataset_identity_sha256") or "")
                != str(snapshot.dataset_identity_sha256)
                or str(daily_provenance.get("dataset_lineage_id") or "")
                != str(portfolio.daily_dataset_lineage_id)
                or str(snapshot.dataset_lineage_id or "")
                != str(portfolio.daily_dataset_lineage_id)
            ):
                raise ValueError(
                    "recommendation daily dataset is outside the simulation lineage"
                )
            execution = select_qlib_dataset(
                data_root,
                anchor_name=str(portfolio.execution_dataset),
                roll_policy=execution_roll,
                lineage_id=str(portfolio.execution_dataset_lineage_id),
                required_date=trade_date,
            )
            execution_provenance = dict(execution.get("provenance") or {})
            if (
                str(daily_provenance.get("source_lineage_id") or "")
                != str(execution_provenance.get("source_lineage_id") or "")
            ):
                raise ValueError(
                    "forward daily and execution datasets do not share source lineage"
                )
            bindings = {
                "daily_dataset": str(daily["name"]),
                "daily_dataset_identity_sha256": str(
                    daily_provenance["dataset_identity_sha256"]
                ),
                "daily_dataset_lineage_id": str(
                    daily_provenance["dataset_lineage_id"]
                ),
                "execution_dataset": str(execution["name"]),
                "execution_dataset_identity_sha256": str(
                    execution_provenance["dataset_identity_sha256"]
                ),
                "execution_dataset_lineage_id": str(
                    execution_provenance["dataset_lineage_id"]
                ),
            }
        return bindings

    def _strategy_order_plan_dataset_bindings(
        self,
        *,
        portfolio: Any,
        daily_dataset: str,
        source_snapshot: dict[str, Any],
        signal_date: date,
        trade_date: date,
        data_root: Path,
    ) -> dict[str, str]:
        """Resolve exact immutable descendants for one forward paper batch."""

        bindings = self._portfolio_batch_dataset_bindings(portfolio)
        daily_roll = str(portfolio.daily_roll_policy or "pinned")
        execution_roll = str(portfolio.execution_roll_policy or "pinned")
        from .services import list_qlib_datasets

        datasets = {str(item["name"]): item for item in list_qlib_datasets(data_root)}
        selected_daily = datasets.get(daily_dataset)
        if daily_roll == "pinned":
            if (
                daily_dataset != bindings["daily_dataset"]
                or source_snapshot.get("dataset_identity_sha256")
                != bindings["daily_dataset_identity_sha256"]
                or source_snapshot.get("dataset_lineage_id")
                != bindings["daily_dataset_lineage_id"]
            ):
                raise ValueError(
                    "pinned paper order-plan does not match its account dataset"
                )
        elif daily_roll == "latest_compatible":
            daily = selected_daily
            provenance = dict((daily or {}).get("provenance") or {})
            if (
                daily is None
                or not daily.get("ready")
                or not daily.get("reproducible")
                or provenance.get("dataset_identity_sha256")
                != source_snapshot.get("dataset_identity_sha256")
                or provenance.get("dataset_lineage_id")
                != str(portfolio.daily_dataset_lineage_id)
                or source_snapshot.get("dataset_lineage_id")
                != str(portfolio.daily_dataset_lineage_id)
            ):
                raise ValueError(
                    "paper order-plan daily dataset is outside the verified account lineage"
                )
            bindings.update(
                {
                    "daily_dataset": daily_dataset,
                    "daily_dataset_identity_sha256": str(
                        provenance["dataset_identity_sha256"]
                    ),
                    "daily_dataset_lineage_id": str(provenance["dataset_lineage_id"]),
                }
            )
        else:
            raise ValueError("paper simulation daily roll policy is invalid")

        from .data_rollover import qlib_trading_date_on_or_before

        selected_daily = datasets.get(bindings["daily_dataset"])
        if selected_daily is None or qlib_trading_date_on_or_before(
            selected_daily,
            _now().astimezone(ZoneInfo("Asia/Shanghai")).date(),
        ) != signal_date:
            raise ValueError(
                "paper order-plan is not the latest currently available trading day"
            )

        if execution_roll == "latest_compatible":
            from .data_rollover import select_qlib_dataset

            try:
                execution = select_qlib_dataset(
                    data_root,
                    anchor_name=str(portfolio.execution_dataset),
                    roll_policy=execution_roll,
                    lineage_id=str(portfolio.execution_dataset_lineage_id),
                    required_date=trade_date,
                )
            except ValueError as exc:
                if str(exc) == (
                    "no verified latest-compatible Qlib descendant covers the "
                    "requested date"
                ):
                    raise ExecutionDataNotReadyError(trade_date) from exc
                raise
            execution_provenance = dict(execution.get("provenance") or {})
            selected_daily = datasets.get(bindings["daily_dataset"])
            daily_source_lineage = str(
                dict((selected_daily or {}).get("provenance") or {}).get(
                    "source_lineage_id"
                )
                or ""
            )
            if daily_source_lineage != str(
                execution_provenance.get("source_lineage_id") or ""
            ):
                raise ValueError(
                    "forward paper daily and execution datasets do not share source lineage"
                )
            bindings.update(
                {
                    "execution_dataset": str(execution["name"]),
                    "execution_dataset_identity_sha256": str(
                        execution_provenance["dataset_identity_sha256"]
                    ),
                    "execution_dataset_lineage_id": str(
                        execution_provenance["dataset_lineage_id"]
                    ),
                }
            )
        elif execution_roll != "pinned":
            raise ValueError("paper simulation execution roll policy is invalid")
        return bindings

    def _pair_replay_dataset_bindings(
        self,
        *,
        portfolio: Any,
        governed_plan: dict[str, Any],
        trade_date: date,
        data_root: Path,
    ) -> dict[str, str]:
        """Bind a pair shadow day to verified descendants for that exact date."""

        from .data_rollover import select_qlib_dataset
        daily = select_qlib_dataset(
            data_root,
            anchor_name=str(portfolio.daily_dataset),
            roll_policy=str(portfolio.daily_roll_policy or "pinned"),
            lineage_id=str(portfolio.daily_dataset_lineage_id),
            required_date=trade_date,
        )
        execution = select_qlib_dataset(
            data_root,
            anchor_name=str(portfolio.execution_dataset),
            roll_policy=str(portfolio.execution_roll_policy or "pinned"),
            lineage_id=str(portfolio.execution_dataset_lineage_id),
            required_date=trade_date,
        )
        daily_provenance = dict(daily.get("provenance") or {})
        execution_provenance = dict(execution.get("provenance") or {})
        minute_binding = dict(governed_plan.get("minute_dataset") or {})
        if (
            daily_provenance.get("source_lineage_id")
            != execution_provenance.get("source_lineage_id")
        ):
            raise ValueError("pair forward daily and minute datasets do not share source lineage")
        if (
            str(execution_provenance.get("snapshot_name") or "")
            != str(governed_plan.get("execution_snapshot") or "")
            or str(minute_binding.get("dataset_name") or "")
            not in set(execution_provenance.get("source_datasets") or [])
            or str(execution_provenance.get("snapshot_manifest_sha256") or "")
            != str(minute_binding.get("manifest_sha256") or "")
        ):
            raise ValueError(
                "pair forward minute dataset is not derived from the governed execution snapshot"
            )
        return {
            "daily_dataset": str(daily["name"]),
            "daily_dataset_identity_sha256": str(
                daily_provenance["dataset_identity_sha256"]
            ),
            "daily_dataset_lineage_id": str(daily_provenance["dataset_lineage_id"]),
            "execution_dataset": str(execution["name"]),
            "execution_dataset_identity_sha256": str(
                execution_provenance["dataset_identity_sha256"]
            ),
            "execution_dataset_lineage_id": str(
                execution_provenance["dataset_lineage_id"]
            ),
        }

    def create_batches_for_snapshot(
        self,
        snapshot_id: str,
        *,
        actor: str = "recommendation-worker",
        data_root: Path | None = None,
    ) -> list[tuple[dict[str, Any], bool]]:
        self.safe_mode.assert_inactive(action="simulation batch creation")
        producer = actor.strip()
        if len(producer) < 2:
            raise ValueError("simulation batch producer is required")
        now = _now()
        batch_refs: list[tuple[str, bool]] = []
        with self.engine.begin() as connection:
            snapshot = connection.execute(
                select(recommendation_snapshots).where(
                    recommendation_snapshots.c.id == snapshot_id
                )
            ).first()
            if snapshot is None:
                raise KeyError(snapshot_id)
            if snapshot.status != "succeeded" or snapshot.effective_date is None:
                raise ValueError("simulation batch requires a successful recommendation snapshot")
            portfolios = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.source_type == "recommendation",
                    simulation_portfolios.c.source_id == snapshot.portfolio_id,
                    simulation_portfolios.c.status == "active",
                )
                .order_by(simulation_portfolios.c.created_at)
            ).all()
            for portfolio in portfolios:
                self._require_current_source_contract(connection, portfolio)
                dataset_bindings = self._snapshot_batch_dataset_bindings(
                    portfolio=portfolio,
                    snapshot=snapshot,
                    trade_date=snapshot.effective_date,
                    data_root=data_root,
                )
                target_payload = None
                if str(portfolio.execution_frequency) == "day":
                    if data_root is None:
                        raise ValueError(
                            "daily recommendation simulation requires the sealed "
                            "execution calendar dataset"
                        )
                    from .services import list_qlib_datasets

                    execution_datasets = {
                        str(item["name"]): item
                        for item in list_qlib_datasets(Path(data_root))
                    }
                    execution_dataset = execution_datasets.get(
                        dataset_bindings["execution_dataset"]
                    )
                    if (
                        execution_dataset is None
                        or execution_dataset.get("ready") is not True
                        or execution_dataset.get("reproducible") is not True
                    ):
                        raise ValueError(
                            "daily recommendation execution dataset is not ready or reproducible"
                        )
                    execution_provenance = dict(
                        execution_dataset.get("provenance") or {}
                    )
                    require_daily_qlib_contract(execution_provenance)
                    require_native_daily_execution_controls(
                        execution_provenance,
                        start=snapshot.effective_date,
                    )
                    settlement_binding = build_settlement_calendar_binding(
                        execution_dataset,
                        trade_date=snapshot.effective_date,
                    )
                    validate_settlement_calendar_binding(
                        settlement_binding,
                        trade_date=snapshot.effective_date,
                        dataset_identity_sha256=dataset_bindings[
                            "execution_dataset_identity_sha256"
                        ],
                        dataset_lineage_id=dataset_bindings[
                            "execution_dataset_lineage_id"
                        ],
                    )
                    target_payload = {
                        "governed_order_plan": {
                            "format_version": "recommendation-snapshot-daily-v1",
                            "source_snapshot_id": str(snapshot.id),
                            "settlement_calendar_binding": settlement_binding,
                        }
                    }
                simulation_semantics_sha256 = (
                    self._batch_simulation_semantics_sha256(
                        portfolio, dataset_bindings
                    )
                )
                batch_id = uuid.uuid4().hex
                inserted_id = connection.scalar(
                    pg_insert(simulation_batches)
                    .values(
                        id=batch_id,
                        portfolio_id=portfolio.id,
                        recommendation_snapshot_id=snapshot_id,
                        source_snapshot_id=snapshot_id,
                        target_payload_json=target_payload,
                        execution_adapter=portfolio.execution_adapter,
                        execution_contract_hash=portfolio.execution_contract_hash,
                        **dataset_bindings,
                        simulation_semantics_sha256=simulation_semantics_sha256,
                        signal_date=snapshot.as_of_date,
                        trade_date=snapshot.effective_date,
                        status="queued",
                        idempotency_key=f"simulation:{portfolio.id}:{snapshot_id}",
                        created_by=producer,
                        created_at=now,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[simulation_batches.c.idempotency_key]
                    )
                    .returning(simulation_batches.c.id)
                )
                if inserted_id is None:
                    existing_id = connection.scalar(
                        select(simulation_batches.c.id).where(
                            simulation_batches.c.idempotency_key
                            == f"simulation:{portfolio.id}:{snapshot_id}"
                        )
                    )
                    if existing_id is None:
                        raise RuntimeError("simulation batch idempotency lookup failed")
                    batch_refs.append((str(existing_id), False))
                else:
                    batch_refs.append((batch_id, True))
        return [(self.get_batch(batch_id), created) for batch_id, created in batch_refs]

    def create_batch_for_targets(
        self,
        portfolio_id: str,
        *,
        source_snapshot_id: str,
        signal_date: date,
        trade_date: date,
        target_payload: dict[str, Any],
        execution_contract_hash: str,
        idempotency_key: str,
        actor: str = "simulation-operator",
    ) -> tuple[dict[str, Any], bool]:
        """Reject the retired client-authored long-only target path."""

        del (
            portfolio_id,
            source_snapshot_id,
            signal_date,
            trade_date,
            target_payload,
            execution_contract_hash,
            idempotency_key,
            actor,
        )
        raise ValueError(
            "direct simulation target payloads are forbidden; use an immutable "
            "Qlib order-plan artifact or the recommendation snapshot path"
        )

    def order_plan_predecessor_state(
        self,
        portfolio_id: str,
        *,
        signal_date: date,
    ) -> dict[str, Any]:
        """Prove the prior sealed plan has a certified ledger day for this account."""

        with self.engine.connect() as connection:
            portfolio_exists = connection.scalar(
                select(simulation_portfolios.c.id).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            )
            if portfolio_exists is None:
                raise KeyError(portfolio_id)
            prior = connection.execute(
                select(
                    jobs.c.id,
                    jobs.c.status,
                    jobs.c.payload_json,
                    jobs.c.progress_json,
                    jobs.c.created_at,
                )
                .where(
                    jobs.c.kind == "simulation_order_plan",
                    jobs.c.payload_json["simulation_portfolio_id"].as_string()
                    == portfolio_id,
                    jobs.c.payload_json["signal_date"].as_string()
                    < signal_date.isoformat(),
                )
                .order_by(
                    jobs.c.payload_json["signal_date"].as_string().desc(),
                    jobs.c.created_at.desc(),
                )
                .limit(1)
            ).first()
            if prior is None:
                return {
                    "ready": True,
                    "status": "first_plan",
                    "portfolio_id": portfolio_id,
                    "signal_date": signal_date.isoformat(),
                }
            prior_payload = dict(prior.payload_json or {})
            prior_progress = dict(prior.progress_json or {})
            prior_signal_date = str(prior_payload.get("signal_date") or "")
            base = {
                "ready": False,
                "portfolio_id": portfolio_id,
                "signal_date": signal_date.isoformat(),
                "predecessor_job_id": str(prior.id),
                "predecessor_signal_date": prior_signal_date,
            }
            if str(prior.status) != "succeeded":
                return {
                    **base,
                    "status": "predecessor_plan_not_succeeded",
                    "predecessor_job_status": str(prior.status),
                    "recovery": "retry_frozen_simulation_order_plan",
                }
            batch_id = str(prior_progress.get("simulation_batch_id") or "")
            if not batch_id:
                return {
                    **base,
                    "status": str(
                        prior_progress.get("simulation_batch_materialization_status")
                        or "predecessor_batch_not_materialized"
                    ),
                }
            batch = connection.execute(
                select(simulation_batches).where(simulation_batches.c.id == batch_id)
            ).first()
            if (
                batch is None
                or str(batch.portfolio_id) != portfolio_id
                or batch.signal_date.isoformat() != prior_signal_date
                or batch.trade_date > signal_date
            ):
                return {**base, "status": "predecessor_batch_identity_invalid"}
            if str(batch.status) != "succeeded":
                return {
                    **base,
                    "status": "predecessor_batch_not_succeeded",
                    "predecessor_batch_id": batch_id,
                    "predecessor_batch_status": str(batch.status),
                }
            nav = connection.execute(
                select(simulation_nav).where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.trade_date == batch.trade_date,
                )
            ).first()
            if (
                nav is None
                or not bool(nav.performance_certified)
                or bool(nav.has_stale_prices)
                or str(nav.status) != "healthy"
                or nav.market_date != batch.trade_date
            ):
                return {
                    **base,
                    "status": "predecessor_nav_not_certified",
                    "predecessor_batch_id": batch_id,
                    "predecessor_trade_date": batch.trade_date.isoformat(),
                }
            return {
                **base,
                "ready": True,
                "status": "predecessor_settled",
                "predecessor_batch_id": batch_id,
                "predecessor_trade_date": batch.trade_date.isoformat(),
            }

    def require_order_plan_predecessor_settled(
        self,
        portfolio_id: str,
        *,
        signal_date: date,
    ) -> dict[str, Any]:
        state = self.order_plan_predecessor_state(
            portfolio_id,
            signal_date=signal_date,
        )
        if not bool(state["ready"]):
            raise ValueError(
                "paper order-plan is waiting for its account predecessor: "
                f"{state['status']}"
            )
        return state

    def create_batch_from_order_plan(
        self,
        portfolio_id: str,
        *,
        order_plan_manifest_sha256: str,
        data_root: Path,
        actor: str,
    ) -> tuple[dict[str, Any], bool]:
        """Queue one long-only replay from an immutable Qlib order-plan artifact."""

        self.safe_mode.assert_inactive(action="simulation batch creation")
        manifest_sha256 = str(order_plan_manifest_sha256 or "").lower()
        if not _is_sha256(manifest_sha256):
            raise ValueError("Qlib order-plan manifest requires a SHA-256 identity")
        producer = actor.strip()
        if len(producer) < 2:
            raise ValueError("simulation batch producer is required")
        artifact_root = (
            Path(data_root) / "artifacts" / "order-plans" / manifest_sha256
        ).resolve()
        allowed_root = (Path(data_root) / "artifacts" / "order-plans").resolve()
        try:
            artifact_root.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError("Qlib order-plan artifact path is unsafe") from exc
        manifest_path = artifact_root / "manifest.json"
        target_path = artifact_root / "target_weights.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            target_payload = json.loads(target_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Qlib order-plan artifact is missing or invalid") from exc
        if not isinstance(manifest, dict) or not isinstance(target_payload, dict):
            raise ValueError("Qlib order-plan artifacts must be JSON objects")
        if _sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("Qlib order-plan manifest failed immutable verification")
        target_file_sha256 = str(manifest.get("target_weights_file_sha256") or "")
        if not _is_sha256(target_file_sha256) or _sha256_file(
            target_path
        ) != target_file_sha256:
            raise ValueError("Qlib order-plan target weights failed immutable verification")
        normalized_targets = self._normalize_target_payload(
            target_payload, adapter="long_only"
        )
        raw_policy_state = target_payload.get("paper_policy_state")
        if not isinstance(raw_policy_state, dict):
            raise ValueError("Qlib paper order-plan has no sealed policy state")
        policy_state = validate_paper_policy_state(raw_policy_state)
        target_weights_sha256 = _canonical_hash(normalized_targets)
        if manifest.get("target_weights_sha256") != target_weights_sha256:
            raise ValueError("Qlib order-plan normalized targets do not match its manifest")
        if manifest.get("format_version") != QLIB_ORDER_PLAN_FORMAT_VERSION:
            raise ValueError("Qlib order-plan artifact format is unsupported")
        if manifest.get("produced_by") != "qlib-workflow-recorder":
            raise ValueError("Qlib order-plan was not produced by the governed research path")
        require_qlib_workflow_identity(manifest.get("qlib_workflow"))
        try:
            signal_date = date.fromisoformat(str(manifest["signal_date"]))
            trade_date = date.fromisoformat(str(manifest["trade_date"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Qlib order-plan dates are missing or invalid") from exc
        source_snapshot = manifest.get("source_snapshot")
        if not isinstance(source_snapshot, dict):
            raise ValueError("Qlib order-plan source snapshot is missing")
        source_snapshot_id = str(source_snapshot.get("id") or "")
        if not _is_sha256(source_snapshot_id):
            raise ValueError("Qlib order-plan source snapshot identity is invalid")
        now = _now()
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if portfolio.status != "active":
                raise ValueError("simulation portfolio is not active")
            self._require_current_source_contract(connection, portfolio)
            if str(portfolio.source_type) == "recommendation":
                raise ValueError(
                    "recommendation simulations are queued only from successful "
                    "recommendation snapshots"
                )
            if str(portfolio.source_type) == "allocation":
                raise ValueError(
                    "allocation simulation NAV is derived from certified member simulations"
                )
            if str(portfolio.execution_adapter) != "long_only":
                raise ValueError(
                    "pair simulation batches must be derived from an approved immutable "
                    "formal backtest artifact"
                )
            version = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == portfolio.source_id
                )
            ).one()
            promotion_stage = connection.execute(
                select(strategy_promotion_stages).where(
                    strategy_promotion_stages.c.strategy_version_id == version.id,
                    strategy_promotion_stages.c.simulation_portfolio_id == portfolio.id,
                    strategy_promotion_stages.c.status == "active",
                )
            ).first()
            forward_gate = connection.execute(
                select(strategy_forward_gates.c.strategy_version_id).where(
                    strategy_forward_gates.c.strategy_version_id == version.id
                )
            ).first()
            if promotion_stage is None or forward_gate is None:
                raise ValueError(
                    "Qlib paper order-plan is not bound to an active gated promotion stage"
                )
            opened_date = promotion_stage.opened_at.astimezone(
                ZoneInfo("Asia/Shanghai")
            ).date()
            if (
                signal_date <= opened_date
                or signal_date > now.astimezone(ZoneInfo("Asia/Shanghai")).date()
                or manifest.get("promotion_stage_id") != str(promotion_stage.id)
                or manifest.get("promotion_stage_opened_at")
                != promotion_stage.opened_at.isoformat()
            ):
                raise ValueError(
                    "Qlib paper order-plan is outside its genuine forward promotion period"
                )
            signal_at, execution_not_before = self._validate_order_plan_timing(
                manifest=manifest,
                version=version,
                portfolio=portfolio,
                signal_date=signal_date,
                trade_date=trade_date,
            )
            backtest_id = str(manifest.get("formal_backtest_id") or "")
            backtest = connection.execute(
                select(backtest_runs).where(backtest_runs.c.id == backtest_id)
            ).first()
            if (
                backtest is None
                or str(backtest.strategy_version_id) != str(version.id)
                or backtest.status != "succeeded"
                or backtest.is_legacy
            ):
                raise ValueError(
                    "Qlib order-plan does not reference the approved source formal backtest"
                )
            required_matches = (
                (manifest.get("source_type"), "strategy_version"),
                (manifest.get("source_id"), str(version.id)),
                (
                    manifest.get("execution_contract_hash"),
                    str(portfolio.execution_contract_hash),
                ),
                (
                    source_snapshot.get("dataset_lineage_id"),
                    str(portfolio.daily_dataset_lineage_id),
                ),
                (
                    source_snapshot_id,
                    str(source_snapshot.get("dataset_identity_sha256") or ""),
                ),
            )
            if any(observed != expected for observed, expected in required_matches):
                raise ValueError(
                    "Qlib order-plan does not match the simulation source contract "
                    "or immutable snapshot"
                )
            dataset_bindings = self._strategy_order_plan_dataset_bindings(
                portfolio=portfolio,
                daily_dataset=str(manifest.get("daily_dataset") or ""),
                source_snapshot=source_snapshot,
                signal_date=signal_date,
                trade_date=trade_date,
                data_root=Path(data_root),
            )
            settlement_calendar_binding = None
            if str(portfolio.execution_frequency) == "day":
                from .services import list_qlib_datasets

                execution_datasets = {
                    str(item["name"]): item
                    for item in list_qlib_datasets(Path(data_root))
                }
                execution_dataset = execution_datasets.get(
                    dataset_bindings["execution_dataset"]
                )
                if execution_dataset is None:
                    raise ValueError(
                        "daily paper execution dataset disappeared before batch binding"
                    )
                settlement_calendar_binding = build_settlement_calendar_binding(
                    execution_dataset,
                    trade_date=trade_date,
                )
                validate_settlement_calendar_binding(
                    settlement_calendar_binding,
                    trade_date=trade_date,
                    dataset_identity_sha256=dataset_bindings[
                        "execution_dataset_identity_sha256"
                    ],
                    dataset_lineage_id=dataset_bindings[
                        "execution_dataset_lineage_id"
                    ],
                )
            horizon_review = manifest.get("horizon_review")
            if horizon_review is not None:
                if (
                    not isinstance(horizon_review, dict)
                    or str(horizon_review.get("strategy_version_id") or "")
                    != str(version.id)
                    or str(horizon_review.get("signal_date") or "")
                    != signal_date.isoformat()
                    or str(horizon_review.get("dataset_identity_sha256") or "")
                    != source_snapshot_id
                    or len(str(horizon_review.get("evidence_sha256") or "")) != 64
                ):
                    raise ValueError(
                        "Qlib order-plan horizon review does not match its paper batch"
                    )
            plan = {
                "format_version": QLIB_ORDER_PLAN_FORMAT_VERSION,
                "manifest_sha256": manifest_sha256,
                "target_weights_file_sha256": target_file_sha256,
                "target_weights_sha256": target_weights_sha256,
                "formal_backtest_id": backtest_id,
                "promotion_stage_id": str(promotion_stage.id),
                "promotion_stage_opened_at": promotion_stage.opened_at.isoformat(),
                "source_snapshot": source_snapshot,
                "execution_contract_hash": str(portfolio.execution_contract_hash),
                "signal_at": signal_at.isoformat() if signal_at else None,
                "execution_not_before": (
                    execution_not_before.isoformat()
                    if execution_not_before
                    else None
                ),
                "signal_snapshot": manifest.get("signal_snapshot"),
                "qlib_workflow": require_qlib_workflow_identity(
                    manifest.get("qlib_workflow")
                ),
            }
            if horizon_review is not None:
                plan["horizon_review"] = dict(horizon_review)
            if settlement_calendar_binding is not None:
                plan["settlement_calendar_binding"] = settlement_calendar_binding
            payload = {
                **normalized_targets,
                "paper_policy_state": policy_state,
                "governed_order_plan": plan,
            }
            batch_id = uuid.uuid4().hex
            idempotency_key = (
                f"qlib-order-plan:{portfolio.id}:{manifest_sha256}"
            )
            inserted_id = connection.scalar(
                pg_insert(simulation_batches)
                .values(
                    id=batch_id,
                    portfolio_id=portfolio.id,
                    recommendation_snapshot_id=None,
                    source_snapshot_id=source_snapshot_id,
                    target_payload_json=payload,
                    execution_adapter="long_only",
                    execution_contract_hash=portfolio.execution_contract_hash,
                    **dataset_bindings,
                    simulation_semantics_sha256=self._batch_simulation_semantics_sha256(
                        portfolio, dataset_bindings
                    ),
                    signal_date=signal_date,
                    trade_date=trade_date,
                    signal_at=signal_at,
                    execution_not_before=execution_not_before,
                    status="queued",
                    idempotency_key=idempotency_key,
                    created_by=producer,
                    created_at=now,
                )
                .on_conflict_do_nothing(
                    index_elements=[simulation_batches.c.idempotency_key]
                )
                .returning(simulation_batches.c.id)
            )
            if inserted_id is None:
                existing = connection.execute(
                    select(simulation_batches).where(
                        simulation_batches.c.idempotency_key == idempotency_key
                    )
                ).one()
                if (
                    str(existing.portfolio_id) != str(portfolio.id)
                    or str(existing.source_snapshot_id) != source_snapshot_id
                    or dict(existing.target_payload_json or {}) != payload
                    or str(existing.execution_contract_hash)
                    != str(portfolio.execution_contract_hash)
                    or any(
                        str(getattr(existing, key)) != str(value)
                        for key, value in dataset_bindings.items()
                    )
                    or existing.signal_at != signal_at
                    or existing.execution_not_before != execution_not_before
                ):
                    raise ValueError(
                        "Qlib order-plan idempotency identity is already bound "
                        "to different targets"
                    )
                return self._batch_dict(existing), False
        return self.get_batch(batch_id), True

    def record_account_order_plan_decision(
        self,
        portfolio_id: str,
        *,
        trade_date: date,
        actions: list[dict[str, Any]],
        target_version: str,
        actor: str,
        account_netting_plan_id: str,
        limit_prices: dict[str, float] | None = None,
        not_before: datetime | None = None,
        not_after: datetime | None = None,
        signal_date: date | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Persist an intentional zero-order account decision.

        This is deliberately not a ``simulation_batch``.  It cannot cancel,
        replace, create, reserve, or execute an order; it only seals why the
        exact account-netting plan resulted in HOLD/NO_ACTION.  Locking the
        primary ledger makes one plan id bind to exactly one decision even
        when scheduler replicas or restarts race.
        """

        self.safe_mode.assert_inactive(action="simulation decision-only plan creation")
        normalized_signal_date = signal_date or trade_date
        event = build_account_order_decision_event(
            portfolio_id=portfolio_id,
            account_netting_plan_id=account_netting_plan_id,
            target_version=target_version,
            actor=actor,
            signal_date=normalized_signal_date,
            trade_date=trade_date,
            actions=actions,
            limit_prices=limit_prices,
            not_before=not_before,
            not_after=not_after,
        )
        now = _now()
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if portfolio.status != "active":
                raise ValueError("simulation portfolio is not active")
            self._require_current_source_contract(connection, portfolio)
            if str(portfolio.execution_adapter) != "long_only":
                raise ValueError("decision-only account plans require the long_only adapter")
            plan_row = connection.execute(
                select(account_netting_plans).where(
                    account_netting_plans.c.id == account_netting_plan_id
                )
            ).first()
            if plan_row is None:
                raise KeyError(account_netting_plan_id)
            if str(portfolio.source_type) != "allocation" or str(
                portfolio.source_id
            ) != str(plan_row.account_id):
                raise ValueError(
                    "account netting plan does not match the simulation account"
                )
            _validate_account_netting_primary_evidence(
                connection,
                portfolio=portfolio,
                plan_json=dict(plan_row.plan_json or {}),
            )
            existing_batch = connection.execute(
                select(simulation_batches).where(
                    simulation_batches.c.portfolio_id == portfolio.id,
                    simulation_batches.c.account_netting_plan_id
                    == account_netting_plan_id,
                )
            ).first()
            if existing_batch is not None:
                raise ValueError(
                    "account netting plan is already bound to an execution batch"
                )
            existing_rows = connection.execute(
                select(simulation_events).where(
                    simulation_events.c.portfolio_id == portfolio.id,
                    simulation_events.c.event_type
                    == ACCOUNT_ORDER_DECISION_EVENT_TYPE,
                )
            ).all()
            for row in existing_rows:
                details = dict(row.details_json or {})
                if str(details.get("account_netting_plan_id") or "") != str(
                    account_netting_plan_id
                ):
                    continue
                existing = validate_account_order_decision_event(
                    details,
                    portfolio_id=str(portfolio.id),
                    account_netting_plan_id=account_netting_plan_id,
                )
                if existing != event or str(row.id) != event["event_sha256"]:
                    raise ValueError(
                        "account netting plan is already bound to a different "
                        "decision-only record"
                    )
                return existing, False
            inserted_id = connection.scalar(
                pg_insert(simulation_events)
                .values(
                    id=event["event_sha256"],
                    portfolio_id=portfolio.id,
                    batch_id=None,
                    trade_date=trade_date,
                    severity="info",
                    event_type=ACCOUNT_ORDER_DECISION_EVENT_TYPE,
                    instrument=None,
                    reason="account_netting_plan_created_no_new_order_operations",
                    details_json=event,
                    created_at=now,
                )
                .on_conflict_do_nothing(index_elements=[simulation_events.c.id])
                .returning(simulation_events.c.id)
            )
            if inserted_id is None:
                stored = connection.execute(
                    select(simulation_events).where(
                        simulation_events.c.id == event["event_sha256"]
                    )
                ).first()
                if (
                    stored is None
                    or str(stored.portfolio_id) != str(portfolio.id)
                    or str(stored.event_type) != ACCOUNT_ORDER_DECISION_EVENT_TYPE
                    or dict(stored.details_json or {}) != event
                ):
                    raise ValueError("account order decision event identity was reused")
                return event, False
        return event, True

    def get_account_order_plan_decision(
        self,
        portfolio_id: str,
        *,
        account_netting_plan_id: str,
    ) -> dict[str, Any] | None:
        """Read one exact decision-only record for an exact primary ledger."""

        with self.engine.connect() as connection:
            rows = connection.execute(
                select(simulation_events)
                .where(
                    simulation_events.c.portfolio_id == portfolio_id,
                    simulation_events.c.event_type
                    == ACCOUNT_ORDER_DECISION_EVENT_TYPE,
                )
                .order_by(
                    simulation_events.c.trade_date.desc(),
                    simulation_events.c.created_at.desc(),
                )
            ).all()
        matches: list[tuple[Any, dict[str, Any]]] = []
        for row in rows:
            details = dict(row.details_json or {})
            if str(details.get("account_netting_plan_id") or "") != str(
                account_netting_plan_id
            ):
                continue
            validated = validate_account_order_decision_event(
                details,
                portfolio_id=portfolio_id,
                account_netting_plan_id=account_netting_plan_id,
            )
            if (
                str(row.id) != validated["event_sha256"]
                or row.batch_id is not None
                or row.trade_date.isoformat() != validated["trade_date"]
            ):
                raise ValueError("account order decision database binding is invalid")
            matches.append((row, validated))
        if len(matches) > 1:
            raise ValueError("account netting plan has multiple decision-only records")
        if not matches:
            return None
        row, decision = matches[0]
        return {
            "id": str(row.id),
            "portfolio_id": str(row.portfolio_id),
            "trade_date": row.trade_date.isoformat(),
            "event_type": str(row.event_type),
            "decision": decision,
        }

    def create_account_decision_valuation_batch(
        self,
        portfolio_id: str,
        *,
        decision_event_sha256: str,
        actor: str,
        data_root: Path,
    ) -> tuple[dict[str, Any], bool]:
        """Queue a zero-order daily mark for one sealed decision-only event.

        The batch has no ``account_netting_plan_id`` and creates no new order.
        It runs the existing corporate-action, closing-price, NAV, benchmark,
        and reconciliation engine on a HOLD/NO_ACTION day.  When NO_ACTION
        explicitly retains a previously frozen target, the normal engine may
        continue that already-authorized open order; the decision event still
        remains the only object bound to the new netting plan.
        """

        self.safe_mode.assert_inactive(action="simulation decision valuation")
        responsible = str(actor).strip()
        event_sha256 = str(decision_event_sha256).strip().lower()
        if len(responsible) < 2:
            raise ValueError("simulation decision valuation actor is required")
        if not _is_sha256(event_sha256):
            raise ValueError("simulation decision valuation event identity is invalid")
        if data_root is None:
            raise ValueError("simulation decision valuation requires governed datasets")
        now = _now()
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if portfolio.status != "active":
                raise ValueError("simulation portfolio is not active")
            self._require_current_source_contract(connection, portfolio)
            if (
                str(portfolio.execution_adapter) != "long_only"
                or str(portfolio.execution_frequency) != "day"
            ):
                raise ValueError(
                    "decision-only valuation requires a daily long_only account"
                )
            event_row = connection.execute(
                select(simulation_events).where(
                    simulation_events.c.id == event_sha256,
                    simulation_events.c.portfolio_id == portfolio.id,
                    simulation_events.c.event_type
                    == ACCOUNT_ORDER_DECISION_EVENT_TYPE,
                    simulation_events.c.batch_id.is_(None),
                )
            ).first()
            if event_row is None:
                raise KeyError(event_sha256)
            event = dict(event_row.details_json or {})
            plan_id = str(event.get("account_netting_plan_id") or "")
            validate_account_order_decision_event(
                event,
                portfolio_id=str(portfolio.id),
                account_netting_plan_id=plan_id,
            )
            existing_execution = connection.execute(
                select(simulation_batches.c.id).where(
                    simulation_batches.c.portfolio_id == portfolio.id,
                    simulation_batches.c.account_netting_plan_id == plan_id,
                )
            ).first()
            if existing_execution is not None:
                raise ValueError(
                    "decision-only valuation cannot coexist with an execution batch"
                )
            signal_date = date.fromisoformat(event["signal_date"])
            trade_date = date.fromisoformat(event["trade_date"])
            dataset_bindings = self._account_order_plan_dataset_bindings(
                portfolio=portfolio,
                signal_date=signal_date,
                trade_date=trade_date,
                data_root=Path(data_root),
            )
            from .services import list_qlib_datasets

            datasets = {
                str(item["name"]): item
                for item in list_qlib_datasets(Path(data_root))
            }
            execution_dataset = datasets.get(dataset_bindings["execution_dataset"])
            if execution_dataset is None:
                raise ValueError(
                    "decision-only valuation execution dataset is unavailable"
                )
            settlement_binding = build_settlement_calendar_binding(
                execution_dataset,
                trade_date=trade_date,
            )
            target_payload = {
                "order_plan": {
                    "plan_version": ORDER_PLAN_MODEL_VERSION,
                    "target_version": str(event["target_version"]),
                    "account_netting_plan_id": None,
                    "actions": [],
                    "decision_valuation_replay": True,
                    "decision_event_sha256": event_sha256,
                },
                "decision_actions": list(event.get("actions") or []),
                "governed_order_plan": {
                    "format_version": ACCOUNT_DECISION_VALUATION_VERSION,
                    "decision_event_sha256": event_sha256,
                    "settlement_calendar_binding": settlement_binding,
                },
            }
            idempotency_key = (
                f"account-decision-valuation:{portfolio.id}:{event_sha256}"
            )
            existing = connection.execute(
                select(simulation_batches).where(
                    simulation_batches.c.idempotency_key == idempotency_key
                )
            ).first()
            if existing is not None:
                if (
                    str(existing.portfolio_id) != str(portfolio.id)
                    or existing.account_netting_plan_id is not None
                    or str(existing.source_snapshot_id) != event_sha256
                    or dict(existing.target_payload_json or {}) != target_payload
                    or existing.signal_date != signal_date
                    or existing.trade_date != trade_date
                ):
                    raise ValueError(
                        "decision-only valuation identity is already bound differently"
                    )
                return self._batch_dict(existing), False
            batch_id = uuid.uuid4().hex
            connection.execute(
                insert(simulation_batches).values(
                    id=batch_id,
                    portfolio_id=portfolio.id,
                    recommendation_snapshot_id=None,
                    source_snapshot_id=event_sha256,
                    target_payload_json=target_payload,
                    execution_adapter="long_only",
                    execution_contract_hash=portfolio.execution_contract_hash,
                    **dataset_bindings,
                    simulation_semantics_sha256=self._batch_simulation_semantics_sha256(
                        portfolio, dataset_bindings
                    ),
                    signal_date=signal_date,
                    trade_date=trade_date,
                    signal_at=None,
                    execution_not_before=None,
                    status="queued",
                    idempotency_key=idempotency_key,
                    account_netting_plan_id=None,
                    created_by=responsible,
                    created_at=now,
                )
            )
        return self.get_batch(batch_id), True

    @staticmethod
    def _validate_account_decision_valuation_batch(
        connection: Any,
        *,
        batch: Any,
        target_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Rebind valuation replay to its sealed decision at execution time."""

        order_plan = target_payload.get("order_plan")
        governed_plan = target_payload.get("governed_order_plan")
        source_event_id = str(batch.source_snapshot_id or "")
        event_row = None
        if source_event_id:
            event_row = connection.execute(
                select(simulation_events).where(
                    simulation_events.c.id == source_event_id
                )
            ).first()
        valuation_marker_present = (
            str(batch.idempotency_key or "").startswith(
                "account-decision-valuation:"
            )
            or (
                isinstance(order_plan, dict)
                and "decision_valuation_replay" in order_plan
            )
            or (
                isinstance(governed_plan, dict)
                and governed_plan.get("format_version")
                == ACCOUNT_DECISION_VALUATION_VERSION
            )
            or (
                event_row is not None
                and str(event_row.event_type) == ACCOUNT_ORDER_DECISION_EVENT_TYPE
            )
        )
        if not valuation_marker_present:
            return None
        if (
            event_row is None
            or str(event_row.portfolio_id) != str(batch.portfolio_id)
            or str(event_row.event_type) != ACCOUNT_ORDER_DECISION_EVENT_TYPE
            or event_row.batch_id is not None
        ):
            raise ValueError(
                "decision-only valuation source event binding is invalid"
            )
        event_payload = dict(event_row.details_json or {})
        plan_id = str(event_payload.get("account_netting_plan_id") or "")
        event = validate_account_order_decision_event(
            event_payload,
            portfolio_id=str(batch.portfolio_id),
            account_netting_plan_id=plan_id,
        )
        expected_order_plan = {
            "plan_version": ORDER_PLAN_MODEL_VERSION,
            "target_version": str(event["target_version"]),
            "account_netting_plan_id": None,
            "actions": [],
            "decision_valuation_replay": True,
            "decision_event_sha256": source_event_id,
        }
        if not isinstance(governed_plan, dict):
            raise ValueError(
                "decision-only valuation governed replay binding is missing"
            )
        expected_governed_plan = {
            "format_version": ACCOUNT_DECISION_VALUATION_VERSION,
            "decision_event_sha256": source_event_id,
            "settlement_calendar_binding": governed_plan.get(
                "settlement_calendar_binding"
            ),
        }
        expected_payload = {
            "order_plan": expected_order_plan,
            "decision_actions": list(event.get("actions") or []),
            "governed_order_plan": expected_governed_plan,
        }
        expected_idempotency_key = (
            f"account-decision-valuation:{batch.portfolio_id}:{source_event_id}"
        )
        if (
            str(event["event_sha256"]) != source_event_id
            or str(batch.idempotency_key) != expected_idempotency_key
            or batch.recommendation_snapshot_id is not None
            or batch.account_netting_plan_id is not None
            or batch.signal_date
            != date.fromisoformat(str(event["signal_date"]))
            or batch.trade_date != date.fromisoformat(str(event["trade_date"]))
            or event_row.trade_date != batch.trade_date
            or target_payload != expected_payload
        ):
            raise ValueError(
                "decision-only valuation replay identity is invalid"
            )
        return event

    def create_order_plan_batch(
        self,
        portfolio_id: str,
        *,
        trade_date: date,
        actions: list[dict[str, Any]],
        target_version: str,
        actor: str,
        account_netting_plan_id: str | None = None,
        limit_prices: dict[str, float] | None = None,
        not_before: datetime | None = None,
        not_after: datetime | None = None,
        signal_date: date | None = None,
        data_root: Path | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Commit a keep/cancel/replace/new order plan and queue its execution batch.

        One transaction is the design 8.1 step-8 commit point: the execution
        batch, the cancel/replace mutations on the live order book, the new
        ``planned`` orders and the cancel/replace events land together.
        ``actions`` are per-instrument outputs of
        :mod:`quant_platform.recommendation_actions` (each carries an
        ``order_plan`` list). Retrying with identical inputs replays the stored
        batch via the idempotency key — no order is re-created and no remainder
        is released twice. ``account_netting_plan_id`` optionally binds the
        batch (and its new orders) to a persisted account netting plan; the
        plan's ``strategy_contributions`` land on the order rows.
        """

        self.safe_mode.assert_inactive(action="simulation order plan creation")
        producer = actor.strip()
        if len(producer) < 2:
            raise ValueError("simulation batch producer is required")
        normalized_target_version = str(target_version or "").strip()
        if not normalized_target_version:
            raise ValueError("order-plan batches require the final target version")
        if not isinstance(actions, list) or not all(isinstance(item, dict) for item in actions):
            raise ValueError("order-plan actions must be a list of instrument plans")
        plan_entries: list[dict[str, Any]] = []
        for action in actions:
            entries = action.get("order_plan")
            if not isinstance(entries, list):
                raise ValueError("order-plan actions must carry an order_plan list")
            for entry in entries:
                normalized = dict(entry)
                if not str(normalized.get("instrument") or ""):
                    normalized["instrument"] = str(action.get("instrument") or "")
                plan_entries.append(normalized)
        if not plan_entries:
            raise ValueError("order-plan actions contain no keep/cancel/replace/new entry")
        prices = {
            str(key).upper(): float(value) for key, value in (limit_prices or {}).items()
        }
        if any(value <= 0 for value in prices.values()):
            raise ValueError("order-plan limit prices must be positive")
        normalized_signal_date = signal_date or trade_date
        if normalized_signal_date > trade_date:
            raise ValueError("order-plan signal date must not follow the trade date")
        now = _now()
        plan_hash = _canonical_hash(
            {
                "plan_version": ORDER_PLAN_MODEL_VERSION,
                "portfolio_id": str(portfolio_id),
                "trade_date": trade_date.isoformat(),
                "target_version": normalized_target_version,
                "actions": actions,
                "account_netting_plan_id": account_netting_plan_id,
                "limit_prices": prices,
                "not_before": not_before.isoformat() if not_before else None,
                "not_after": not_after.isoformat() if not_after else None,
            }
        )
        idempotency_key = f"order-plan:{portfolio_id}:{plan_hash}"
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if portfolio.status != "active":
                raise ValueError("simulation portfolio is not active")
            self._require_current_source_contract(connection, portfolio)
            if str(portfolio.execution_adapter) != "long_only":
                raise ValueError("order-plan batches require the long_only adapter")
            contributions: dict[str, Any] = {}
            if account_netting_plan_id is not None:
                plan_row = connection.execute(
                    select(account_netting_plans).where(
                        account_netting_plans.c.id == account_netting_plan_id
                    )
                ).first()
                if plan_row is None:
                    raise KeyError(account_netting_plan_id)
                if str(portfolio.source_type) != "allocation" or str(
                    portfolio.source_id
                ) != str(plan_row.account_id):
                    raise ValueError(
                        "account netting plan does not match the simulation account"
                    )
                _validate_account_netting_primary_evidence(
                    connection,
                    portfolio=portfolio,
                    plan_json=dict(plan_row.plan_json or {}),
                )
                contributions = {
                    str(instrument): entry
                    for instrument, entry in (
                        dict(plan_row.plan_json or {}).get("strategy_contributions") or {}
                    ).items()
                }
                decision_rows = connection.execute(
                    select(simulation_events).where(
                        simulation_events.c.portfolio_id == portfolio.id,
                        simulation_events.c.event_type
                        == ACCOUNT_ORDER_DECISION_EVENT_TYPE,
                    )
                ).all()
                for decision_row in decision_rows:
                    details = dict(decision_row.details_json or {})
                    if str(details.get("account_netting_plan_id") or "") != str(
                        account_netting_plan_id
                    ):
                        continue
                    validate_account_order_decision_event(
                        details,
                        portfolio_id=str(portfolio.id),
                        account_netting_plan_id=account_netting_plan_id,
                    )
                    raise ValueError(
                        "account netting plan is already bound to a decision-only record"
                    )
            existing = connection.execute(
                select(simulation_batches).where(
                    simulation_batches.c.idempotency_key == idempotency_key
                )
            ).first()
            if existing is not None:
                # 幂等重放：计划已提交过，不重复建单、不重复释放余量。
                return self._batch_dict(existing), False
            order_rows = connection.execute(
                select(simulation_orders)
                .where(simulation_orders.c.portfolio_id == portfolio.id)
                .with_for_update()
            ).all()
            book = {
                str(row.id): row
                for row in order_rows
                if str(row.status) in OPEN_STATUSES
            }
            outcome = apply_order_plan(
                open_orders=[dict(row_dict(row)) for row in order_rows],
                plan_entries=plan_entries,
            )
            cost_model = CostScheduleBook.from_mapping(
                dict(portfolio.execution_policy_json or {}).get("cost_model")
            ).as_of(trade_date)
            dataset_bindings = self._account_order_plan_dataset_bindings(
                portfolio=portfolio,
                signal_date=normalized_signal_date,
                trade_date=trade_date,
                data_root=data_root,
            )
            batch_id = uuid.uuid4().hex
            connection.execute(
                insert(simulation_batches).values(
                    id=batch_id,
                    portfolio_id=portfolio.id,
                    recommendation_snapshot_id=None,
                    source_snapshot_id=None,
                    target_payload_json={
                        "order_plan": {
                            "plan_version": ORDER_PLAN_MODEL_VERSION,
                            "target_version": normalized_target_version,
                            "account_netting_plan_id": account_netting_plan_id,
                            "actions": actions,
                        }
                    },
                    execution_adapter=str(portfolio.execution_adapter),
                    execution_contract_hash=portfolio.execution_contract_hash,
                    **dataset_bindings,
                    simulation_semantics_sha256=self._batch_simulation_semantics_sha256(
                        portfolio, dataset_bindings
                    ),
                    signal_date=normalized_signal_date,
                    trade_date=trade_date,
                    signal_at=None,
                    execution_not_before=None,
                    status="queued",
                    idempotency_key=idempotency_key,
                    account_netting_plan_id=account_netting_plan_id,
                    created_by=producer,
                    created_at=now,
                )
            )

            def _order_event(
                *, event_type: str, instrument: str | None, reason: str, details: dict
            ) -> None:
                connection.execute(
                    insert(simulation_events).values(
                        id=uuid.uuid4().hex,
                        portfolio_id=portfolio.id,
                        batch_id=batch_id,
                        trade_date=trade_date,
                        severity="info",
                        event_type=event_type,
                        instrument=instrument,
                        reason=reason,
                        details_json=details,
                        created_at=now,
                    )
                )

            for cancel in outcome["cancels"]:
                row = book[cancel["order_id"]]
                updated = connection.execute(
                    update(simulation_orders)
                    .where(
                        simulation_orders.c.id == cancel["order_id"],
                        simulation_orders.c.status.in_(OPEN_STATUSES),
                    )
                    .values(
                        status=STATUS_CANCELLED,
                        cancel_reason=cancel["reason"],
                        updated_at=now,
                    )
                )
                if int(updated.rowcount or 0) != 1:
                    raise ValueError(
                        f"order {cancel['order_id']} changed state while the plan committed"
                    )
                if str(row.side) == "buy":
                    self._move_order_reservation(
                        connection,
                        portfolio_id=str(portfolio.id),
                        order_id=str(row.id),
                        event_key=f"order:{row.id}:release:{batch_id}:cancel",
                        event_type="release",
                        amount=None,
                        occurred_at=now,
                        now=now,
                        batch_id=batch_id,
                        details={"reason": cancel["reason"]},
                    )
                else:
                    self._move_sell_reservation(
                        connection,
                        portfolio_id=str(portfolio.id),
                        order_id=str(row.id),
                        event_key=f"order:{row.id}:security-release:{batch_id}:cancel",
                        event_type="release",
                        quantity=None,
                        occurred_at=now,
                        now=now,
                        batch_id=batch_id,
                        details={"reason": cancel["reason"]},
                    )
                _order_event(
                    event_type="order_cancelled",
                    instrument=str(row.instrument),
                    reason=cancel["reason"],
                    details={
                        "order_id": cancel["order_id"],
                        "released_quantity": cancel["released_quantity"],
                    },
                )
            for replace in outcome["replaces"]:
                row = book[replace["order_id"]]
                if str(row.side) == "buy":
                    open_quantity = int(row.requested_quantity) - int(
                        row.filled_quantity
                    )
                    reservation = self._reservation_total(connection, str(row.id))
                    release_amount = (
                        reservation
                        * Decimal(str(replace["released_quantity"]))
                        / Decimal(str(open_quantity))
                    ).quantize(Decimal("0.000001"))
                    self._move_order_reservation(
                        connection,
                        portfolio_id=str(portfolio.id),
                        order_id=str(row.id),
                        event_key=f"order:{row.id}:release:{batch_id}:replace",
                        event_type="release",
                        amount=release_amount,
                        occurred_at=now,
                        now=now,
                        batch_id=batch_id,
                        details={
                            "previous_requested_quantity": int(
                                row.requested_quantity
                            ),
                            "new_requested_quantity": replace[
                                "new_requested_quantity"
                            ],
                        },
                    )
                else:
                    self._move_sell_reservation(
                        connection,
                        portfolio_id=str(portfolio.id),
                        order_id=str(row.id),
                        event_key=f"order:{row.id}:security-release:{batch_id}:replace",
                        event_type="release",
                        quantity=int(replace["released_quantity"]),
                        occurred_at=now,
                        now=now,
                        batch_id=batch_id,
                        details={
                            "previous_requested_quantity": int(
                                row.requested_quantity
                            ),
                            "new_requested_quantity": replace[
                                "new_requested_quantity"
                            ],
                        },
                    )
                updated = connection.execute(
                    update(simulation_orders)
                    .where(
                        simulation_orders.c.id == replace["order_id"],
                        simulation_orders.c.status.in_(OPEN_STATUSES),
                    )
                    .values(
                        requested_quantity=replace["new_requested_quantity"],
                        plan_op="replace",
                        updated_at=now,
                    )
                )
                if int(updated.rowcount or 0) != 1:
                    raise ValueError(
                        f"order {replace['order_id']} changed state while the plan committed"
                    )
                _order_event(
                    event_type="order_replaced",
                    instrument=str(row.instrument),
                    reason=replace["reason"],
                    details={
                        "order_id": replace["order_id"],
                        "previous_requested_quantity": int(row.requested_quantity),
                        "new_requested_quantity": replace["new_requested_quantity"],
                        "released_quantity": replace["released_quantity"],
                    },
                )
            for new in outcome["news"]:
                instrument = new["instrument"]
                supplied_price = new.get("limit_price")
                limit_price = prices.get(
                    instrument,
                    float(supplied_price) if supplied_price is not None else None,
                )
                if new["side"] == "buy" and limit_price is None:
                    raise ValueError(
                        "persistent buy orders require a frozen positive limit price "
                        "so their maximum cash and fee reservation is deterministic"
                    )
                if limit_price is not None and limit_price <= 0:
                    raise ValueError("persistent order limit price must be positive")
                order_id = uuid.uuid4().hex
                connection.execute(
                    insert(simulation_orders).values(
                        id=order_id,
                        batch_id=batch_id,
                        portfolio_id=portfolio.id,
                        instrument=instrument,
                        side=new["side"],
                        target_weight=0.0,
                        requested_quantity=int(new["quantity"]),
                        filled_quantity=0,
                        status=STATUS_PLANNED,
                        reject_reason=None,
                        requested_value=Decimal(
                            str(int(new["quantity"]) * limit_price)
                            if limit_price is not None
                            else "0"
                        ),
                        filled_value=Decimal("0"),
                        capacity_fill_ratio=0.0,
                        expires_at=not_after
                        or datetime.combine(trade_date, time(15, 0), ZoneInfo("Asia/Shanghai")),
                        limit_price=(
                            Decimal(str(limit_price)) if limit_price is not None else None
                        ),
                        not_before=not_before,
                        not_after=not_after,
                        target_version=normalized_target_version,
                        account_netting_plan_id=account_netting_plan_id,
                        strategy_contributions_json=contributions.get(instrument),
                        plan_op="new",
                        created_at=now,
                    )
                )
                if new["side"] == "buy":
                    gross = Decimal(
                        str(int(new["quantity"]) * float(limit_price))
                    ).quantize(Decimal("0.000001"))
                    # Fees are confirmed per fill slice; minimum commission can
                    # therefore be charged more than once. Reserve a
                    # conservative all-slices upper bound, not one aggregate
                    # order commission.
                    maximum_slices = int(
                        dict(portfolio.execution_policy_json or {}).get(
                            "max_slices",
                            24,
                        )
                        or 24
                    )
                    slice_gross = float(gross) / maximum_slices
                    estimated_fee = self._cash_amount(
                        maximum_slices
                        * cost_model.estimate(
                            side="buy",
                            gross_value=slice_gross,
                            participation=cost_model.max_volume_participation,
                            asset_type=infer_cn_asset_type(instrument),
                            trade_date=trade_date,
                        )
                    )
                    self._freeze_order_cash(
                        connection,
                        portfolio_id=str(portfolio.id),
                        order_id=order_id,
                        event_key=f"order:{order_id}:freeze",
                        amount=gross + estimated_fee,
                        as_of=now,
                        now=now,
                        batch_id=batch_id,
                        details={
                            "instrument": instrument,
                            "quantity": int(new["quantity"]),
                            "limit_price": float(limit_price),
                            "estimated_fee": float(estimated_fee),
                        },
                    )
                else:
                    self._freeze_sell_quantity(
                        connection,
                        portfolio_id=str(portfolio.id),
                        order_id=order_id,
                        instrument=instrument,
                        quantity=int(new["quantity"]),
                        trade_date=trade_date,
                        event_key=f"order:{order_id}:security-freeze",
                        occurred_at=now,
                        now=now,
                        batch_id=batch_id,
                        details={"target_version": normalized_target_version},
                    )
        return self.get_batch(batch_id), True

    @staticmethod
    def _validate_order_plan_timing(
        *,
        manifest: dict[str, Any],
        version: Any,
        portfolio: Any,
        signal_date: date,
        trade_date: date,
    ) -> tuple[datetime | None, datetime | None]:
        signal_frequency = str(
            version.signal_frequency
            or dict(version.config_json or {}).get("signal_frequency")
            or "day"
        ).lower()
        raw_signal_at = manifest.get("signal_at")
        raw_execution_not_before = manifest.get("execution_not_before")
        if signal_frequency == "day":
            if raw_signal_at is not None or raw_execution_not_before is not None:
                raise ValueError("daily Qlib order-plans must not contain intraday timestamps")
            if trade_date <= signal_date:
                raise ValueError("daily Qlib order-plan violates next-session execution")
            return None, None
        if raw_signal_at is None or raw_execution_not_before is None:
            raise ValueError(
                "minute Qlib order-plans require signal_at and execution_not_before"
            )
        signal_snapshot = manifest.get("signal_snapshot")
        if not isinstance(signal_snapshot, dict):
            raise ValueError("minute Qlib order-plan signal snapshot is missing")
        if (
            str(signal_snapshot.get("frequency") or "") != signal_frequency
            or not str(signal_snapshot.get("name") or "")
            or any(
                not _is_sha256(signal_snapshot.get(field))
                for field in (
                    "dataset_identity_sha256",
                    "dataset_lineage_id",
                    "source_lineage_id",
                )
            )
        ):
            raise ValueError("minute Qlib order-plan signal snapshot is invalid")
        signal_at = _parse_aware_timestamp(raw_signal_at, field="signal_at")
        execution_not_before = _parse_aware_timestamp(
            raw_execution_not_before,
            field="execution_not_before",
        )
        if signal_at.astimezone(SHANGHAI_TIMEZONE).date() != signal_date:
            raise ValueError("Qlib order-plan signal_at does not match signal_date")
        if execution_not_before.astimezone(SHANGHAI_TIMEZONE).date() != trade_date:
            raise ValueError(
                "Qlib order-plan execution_not_before does not match trade_date"
            )
        policy = dict(portfolio.execution_policy_json or {})
        if str(policy.get("execution_algorithm") or "") != "next_bar":
            raise ValueError("minute Qlib order-plans require next_bar execution")
        execution_frequency = str(portfolio.execution_frequency)
        require_next_bar_execution(
            signal_at,
            execution_not_before,
            signal_frequency=signal_frequency,
            execution_frequency=execution_frequency,
        )
        expected = execution_time_slots(
            trade_date=trade_date,
            policy=policy,
            signal_at=signal_at,
        )[0].astimezone(UTC)
        if execution_not_before != expected:
            raise ValueError(
                "Qlib order-plan execution_not_before is not the first eligible bar"
            )
        return signal_at, execution_not_before

    def create_pair_batch_from_backtest(
        self,
        portfolio_id: str,
        *,
        backtest_id: str,
        trade_date: date,
        data_root: Path,
        actor: str,
    ) -> tuple[dict[str, Any], bool]:
        """Derive an atomic pair target from one immutable approved formal backtest trade."""

        self.safe_mode.assert_inactive(action="simulation batch creation")
        producer = actor.strip()
        if len(producer) < 2:
            raise ValueError("simulation batch producer is required")
        now = _now()
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if portfolio.status != "active":
                raise ValueError("simulation portfolio is not active")
            if (
                str(portfolio.source_type) != "strategy_version"
                or str(portfolio.execution_adapter) != "pair"
            ):
                raise ValueError(
                    "pair replay requires a pair simulation sourced from a strategy version"
                )
            self._require_current_source_contract(connection, portfolio)
            version = connection.execute(
                select(strategy_versions).where(
                    strategy_versions.c.id == portfolio.source_id
                )
            ).one()
            if version.status != "approved" or version.is_legacy:
                raise ValueError("pair replay requires an approved non-legacy strategy version")
            backtest = connection.execute(
                select(backtest_runs).where(backtest_runs.c.id == backtest_id)
            ).first()
            if backtest is None:
                raise KeyError(backtest_id)
            if (
                str(backtest.strategy_version_id) != str(version.id)
                or backtest.status != "succeeded"
                or backtest.is_legacy
            ):
                raise ValueError(
                    "pair replay requires a successful formal backtest belonging to "
                    "the approved source version"
                )
            if str(backtest.execution_contract_hash) != str(
                portfolio.execution_contract_hash
            ):
                raise ValueError("pair backtest execution contract does not match simulation")
            pair = connection.execute(
                select(strategy_pairs).where(
                    strategy_pairs.c.strategy_version_id == version.id
                )
            ).first()
            if pair is None:
                raise ValueError("approved pair strategy has no immutable pair definition")
            target_payload, signal_date, plan = self._derive_pair_replay_target(
                portfolio=portfolio,
                version=version,
                pair=pair,
                backtest=backtest,
                trade_date=trade_date,
                data_root=data_root,
            )
            batch_id = uuid.uuid4().hex
            idempotency_key = (
                f"pair-replay:{portfolio.id}:{backtest.id}:{trade_date.isoformat()}:"
                f"{plan['pair_plan_sha256']}"
            )
            dataset_bindings = self._pair_replay_dataset_bindings(
                portfolio=portfolio,
                governed_plan=plan,
                trade_date=trade_date,
                data_root=data_root,
            )
            inserted_id = connection.scalar(
                pg_insert(simulation_batches)
                .values(
                    id=batch_id,
                    portfolio_id=portfolio.id,
                    recommendation_snapshot_id=None,
                    source_snapshot_id=plan["pair_plan_sha256"],
                    target_payload_json={
                        **target_payload,
                        "governed_pair_plan": plan,
                    },
                    execution_adapter="pair",
                    execution_contract_hash=portfolio.execution_contract_hash,
                    **dataset_bindings,
                    simulation_semantics_sha256=self._batch_simulation_semantics_sha256(
                        portfolio, dataset_bindings
                    ),
                    signal_date=signal_date,
                    trade_date=trade_date,
                    status="queued",
                    idempotency_key=idempotency_key,
                    created_by=producer,
                    created_at=now,
                )
                .on_conflict_do_nothing(
                    index_elements=[simulation_batches.c.idempotency_key]
                )
                .returning(simulation_batches.c.id)
            )
            if inserted_id is None:
                existing_id = connection.scalar(
                    select(simulation_batches.c.id).where(
                        simulation_batches.c.idempotency_key == idempotency_key
                    )
                )
                if existing_id is None:
                    raise RuntimeError("pair replay idempotency lookup failed")
                return self.get_batch(str(existing_id)), False
        return self.get_batch(batch_id), True

    @classmethod
    def _derive_pair_replay_target(
        cls,
        *,
        portfolio: Any,
        version: Any,
        pair: Any,
        backtest: Any,
        trade_date: date,
        data_root: Path,
    ) -> tuple[dict[str, Any], date, dict[str, Any]]:
        artifact_root = Path(str(backtest.artifact_path)).resolve()
        allowed_root = (Path(data_root) / "artifacts" / "backtests").resolve()
        try:
            artifact_root.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(
                "pair backtest artifact is outside the governed artifact root"
            ) from exc
        main_manifest_path = artifact_root / "manifest.json"
        pair_manifest_path = artifact_root / "pair_artifact_manifest.json"
        try:
            main_manifest = json.loads(main_manifest_path.read_text(encoding="utf-8"))
            pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("pair replay artifact manifest is missing or invalid") from exc
        if not isinstance(main_manifest, dict) or not isinstance(pair_manifest, dict):
            raise ValueError("pair replay artifact manifests must be JSON objects")
        metrics = dict(backtest.metrics_json or {})
        provenance = metrics.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        if (
            provenance.get("execution_manifest_sha256") != _sha256_file(main_manifest_path)
            or provenance.get("pair_artifact_manifest_sha256")
            != _sha256_file(pair_manifest_path)
        ):
            raise ValueError("pair replay artifact manifest does not match backtest provenance")
        expected_pair = {
            "leg_y": str(pair.leg_y),
            "leg_x": str(pair.leg_x),
            "asset_class": str(pair.asset_class),
            "shorting_mode": str(pair.shorting_mode),
        }
        main_pair = {
            key: dict(main_manifest.get("pair") or {}).get(key)
            for key in expected_pair
        }
        artifact_pair = {
            key: dict(pair_manifest.get("pair") or {}).get(key)
            for key in expected_pair
        }
        expected_config_sha256 = _canonical_hash(dict(version.config_json or {}))
        required_matches = (
            (main_manifest.get("backtest_id"), str(backtest.id)),
            (main_manifest.get("strategy_version_id"), str(version.id)),
            (main_manifest.get("execution_contract_hash"), str(version.execution_contract_hash)),
            (main_manifest.get("dataset"), str(backtest.dataset)),
            (main_manifest.get("periods"), dict(backtest.periods_json or {})),
            (main_pair, expected_pair),
            (_canonical_hash(dict(main_manifest.get("config") or {})), expected_config_sha256),
            (pair_manifest.get("format_version"), "pair-replay-artifact-v1"),
            (pair_manifest.get("backtest_id"), str(backtest.id)),
            (pair_manifest.get("strategy_version_id"), str(version.id)),
            (
                pair_manifest.get("execution_contract_hash"),
                str(version.execution_contract_hash),
            ),
            (pair_manifest.get("dataset"), str(backtest.dataset)),
            (pair_manifest.get("periods"), dict(backtest.periods_json or {})),
            (artifact_pair, expected_pair),
            (pair_manifest.get("strategy_config_sha256"), expected_config_sha256),
        )
        if any(observed != expected for observed, expected in required_matches):
            raise ValueError(
                "pair replay artifact does not belong to the approved strategy/backtest contract"
            )
        files = pair_manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("pair replay artifact file manifest is missing")
        for name in (
            "daily_returns.parquet",
            "daily_ledger.parquet",
            "kalman_spread.parquet",
            "trades.json",
            "rejections.json",
        ):
            evidence = files.get(name)
            path = (artifact_root / name).resolve()
            try:
                path.relative_to(artifact_root)
            except ValueError as exc:
                raise ValueError("pair replay artifact path is unsafe") from exc
            if (
                not isinstance(evidence, dict)
                or not path.is_file()
                or path.stat().st_size != int(evidence.get("bytes") or -1)
                or _sha256_file(path) != str(evidence.get("sha256") or "")
            ):
                raise ValueError(f"pair replay artifact {name} failed immutable verification")
        try:
            trades = json.loads((artifact_root / "trades.json").read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("pair replay trades artifact is invalid") from exc
        matches = [
            item
            for item in trades
            if isinstance(item, dict)
            and str(item.get("trade_date") or "")[:10] == trade_date.isoformat()
        ]
        if len(matches) > 1:
            raise ValueError(
                "selected trade date identifies multiple governed pair backtest trades"
            )
        ledger = pd.read_parquet(artifact_root / "daily_ledger.parquet").reset_index()
        datetime_field = next(
            (name for name in ("datetime", "trade_date", "date") if name in ledger),
            None,
        )
        if datetime_field is None:
            raise ValueError("pair replay daily ledger has no trade date")
        ledger[datetime_field] = pd.to_datetime(ledger[datetime_field], errors="coerce")
        rows = ledger[ledger[datetime_field].dt.date == trade_date]
        if len(rows) != 1 or not {"quantity_y", "quantity_x"}.issubset(rows.columns):
            raise ValueError("pair replay daily ledger has no unique governed target")
        row = rows.iloc[0]
        prior_dates = sorted(
            value
            for value in ledger[datetime_field].dt.date.dropna().unique().tolist()
            if value < trade_date
        )
        if not prior_dates:
            raise ValueError("pair replay requires a prior governed signal date")
        if matches:
            trade = matches[0]
            try:
                signal_date = date.fromisoformat(str(trade["signal_date"])[:10])
                direction = int(trade["direction"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("governed pair trade has invalid signal metadata") from exc
            action = str(trade.get("action") or "")
            if action not in {"entry", "exit"}:
                raise ValueError("pair replay artifact contains an unsupported trade action")
        else:
            # A persistent ledger must mark and accrue every holding day, not
            # only entry/exit days.  Reuse the last governed entry direction
            # and the immediately preceding immutable ledger date as signal.
            historical = [
                item
                for item in trades
                if isinstance(item, dict)
                and str(item.get("trade_date") or "")[:10] < trade_date.isoformat()
                and str(item.get("action") or "") == "entry"
            ]
            historical.sort(key=lambda item: str(item.get("trade_date") or ""))
            direction = int(historical[-1]["direction"]) if historical else 1
            signal_date = prior_dates[-1]
            action = "hold"
            trade = {
                "action": action,
                "direction": direction,
                "signal_date": signal_date.isoformat(),
                "trade_date": trade_date.isoformat(),
                "source": "immutable_daily_ledger",
            }
        if signal_date >= trade_date or direction not in {-1, 1}:
            raise ValueError("governed pair trade violates next-session execution")
        artifact_quantities = {
            str(pair.leg_y): int(row["quantity_y"]),
            str(pair.leg_x): int(row["quantity_x"]),
        }
        if any(value and (value > 0) != (direction > 0) for value in [
            artifact_quantities[str(pair.leg_y)]
        ]):
            raise ValueError("pair replay ledger direction does not match the governed trade")
        if any(value and (value < 0) != (direction > 0) for value in [
            artifact_quantities[str(pair.leg_x)]
        ]):
            raise ValueError("pair replay ledger hedge direction does not match the governed trade")
        config = dict(version.config_json or {})
        reference_capital = float(config.get("initial_capital") or 0.0)
        if reference_capital <= 0:
            raise ValueError("pair strategy has no governed reference capital")
        scale = float(portfolio.nav) / reference_capital
        scaled = {
            instrument: int(abs(quantity) * scale // 100) * 100
            for instrument, quantity in artifact_quantities.items()
        }
        if action == "entry" and (not all(scaled.values())):
            raise ValueError("simulation capital is too small for the governed pair board lots")
        if action == "exit" and any(scaled.values()):
            raise ValueError("governed pair exit artifact must target zero quantities")
        annual_borrow_rate = float(config.get("annual_borrow_rate") or 0.0)
        if not 0 < annual_borrow_rate <= 1:
            raise ValueError("pair strategy has no governed annual borrow rate")
        sides = {
            str(pair.leg_y): "long" if direction > 0 else "short",
            str(pair.leg_x): "short" if direction > 0 else "long",
        }
        pair_artifact_sha256 = _sha256_file(pair_manifest_path)
        plan_identity = {
            "format_version": "governed-pair-plan-v1",
            "portfolio_id": str(portfolio.id),
            "portfolio_nav": float(portfolio.nav),
            "backtest_id": str(backtest.id),
            "strategy_version_id": str(version.id),
            "trade_date": trade_date.isoformat(),
            "signal_date": signal_date.isoformat(),
            "action": action,
            "direction": direction,
            "execution_contract_hash": str(version.execution_contract_hash),
            "pair_artifact_manifest_sha256": pair_artifact_sha256,
            "trade_sha256": _canonical_hash(trade),
            "execution_snapshot": pair_manifest.get("execution_snapshot"),
            "minute_dataset": pair_manifest.get("minute_dataset"),
            "shortability_dataset": pair_manifest.get("shortability_dataset"),
        }
        plan = {
            **plan_identity,
            "pair_plan_sha256": _canonical_hash(plan_identity),
        }
        group_id = f"pair-{plan['pair_plan_sha256'][:24]}"
        legs = [
            {
                "instrument": instrument,
                "leg_no": leg_no,
                "position_side": sides[instrument],
                "target_quantity": scaled[instrument],
                "annual_borrow_rate": (
                    annual_borrow_rate if sides[instrument] == "short" else 0.0
                ),
            }
            for leg_no, instrument in enumerate(
                (str(pair.leg_y), str(pair.leg_x)), start=1
            )
        ]
        return cls._normalize_target_payload(
            {"atomic_group_id": group_id, "legs": legs}, adapter="pair"
        ), signal_date, plan

    @staticmethod
    def _normalize_target_payload(
        target_payload: dict[str, Any], *, adapter: str
    ) -> dict[str, Any]:
        payload = dict(target_payload or {})
        if adapter == "long_only":
            values = payload.get("target_weights")
            if not isinstance(values, dict):
                raise ValueError("long-only simulation requires target_weights")
            targets = {str(key).upper(): float(value) for key, value in values.items()}
            if any(not isfinite(value) or value < 0 for value in targets.values()):
                raise ValueError("simulation target weights must be finite and non-negative")
            if sum(targets.values()) > 1.0 + 1e-8:
                raise ValueError("simulation target weights exceed one")
            return {"target_weights": dict(sorted(targets.items()))}
        if adapter != "pair":
            raise ValueError("unsupported simulation execution adapter")
        group_id = str(payload.get("atomic_group_id") or "").strip()
        legs = payload.get("legs")
        if not group_id or not isinstance(legs, list) or len(legs) != 2:
            raise ValueError("pair simulation requires one atomic group with exactly two legs")
        normalized: list[dict[str, Any]] = []
        for item in legs:
            leg = dict(item or {})
            quantity = int(leg.get("target_quantity") or 0)
            rate = float(leg.get("annual_borrow_rate") or 0.0)
            normalized.append(
                {
                    "instrument": str(leg.get("instrument") or "").strip().upper(),
                    "leg_no": int(leg.get("leg_no") or 0),
                    "position_side": str(leg.get("position_side") or "").strip(),
                    "target_quantity": quantity,
                    "annual_borrow_rate": rate,
                }
            )
        if {item["leg_no"] for item in normalized} != {1, 2}:
            raise ValueError("pair simulation leg numbers must be 1 and 2")
        if len({item["instrument"] for item in normalized}) != 2 or any(
            not item["instrument"] for item in normalized
        ):
            raise ValueError("pair simulation instruments must be distinct")
        if {item["position_side"] for item in normalized} != {"long", "short"}:
            raise ValueError("pair simulation requires one long and one short leg")
        if any(
            item["target_quantity"] < 0 or item["target_quantity"] % 100
            for item in normalized
        ):
            raise ValueError("pair target quantities must be non-negative board lots")
        short_leg = next(item for item in normalized if item["position_side"] == "short")
        if short_leg["target_quantity"] > 0 and not 0 < short_leg["annual_borrow_rate"] <= 1:
            raise ValueError("pair short leg requires a positive governed borrow rate")
        return {
            "atomic_group_id": group_id,
            "legs": sorted(normalized, key=lambda item: item["leg_no"]),
        }

    @staticmethod
    def _source_strategy_version_id(connection: Any, portfolio: Any) -> str | None:
        source_type = str(portfolio.source_type)
        if source_type == "strategy_version":
            return str(portfolio.source_id)
        if source_type != "recommendation":
            return None
        return connection.scalar(
            select(recommendation_portfolios.c.strategy_version_id).where(
                recommendation_portfolios.c.id == portfolio.source_id
            )
        )

    def source_risk_state(self, portfolio_id: str) -> dict[str, Any]:
        """Expose the unified member/allocation gate used by every simulation adapter."""

        with self.engine.connect() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            if str(portfolio.source_type) == "allocation":
                state = load_allocation_risk_state(
                    connection,
                    str(portfolio.source_id),
                )
                return {
                    "strategy_version_id": None,
                    "allow_new_risk": state["risk_exposure_override"] >= 1.0,
                    **state,
                }
            strategy_version_id = self._source_strategy_version_id(connection, portfolio)
            if not strategy_version_id:
                return {
                    "strategy_version_id": None,
                    "state": "active",
                    "allow_new_risk": True,
                    "risk_exposure_override": 1.0,
                    "event_ids": [],
                    "allocation_ids": [],
                }
            return load_strategy_risk_state(connection, str(strategy_version_id))

    def policy_risk_inputs(
        self,
        portfolio_id: str,
        *,
        required_nav_date: date | None = None,
    ) -> dict[str, Any]:
        """Read the selected simulation ledger facts consumed by PortfolioPolicy."""

        with self.engine.connect() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            latest_nav = connection.execute(
                select(simulation_nav)
                .where(simulation_nav.c.portfolio_id == portfolio_id)
                .order_by(simulation_nav.c.trade_date.desc())
                .limit(1)
            ).first()
            position_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(simulation_positions)
                    .where(
                        simulation_positions.c.portfolio_id == portfolio_id,
                        simulation_positions.c.quantity != 0,
                    )
                )
                or 0
            )
            open_order_count = int(
                connection.scalar(
                    select(func.count())
                    .select_from(simulation_orders)
                    .where(
                        simulation_orders.c.portfolio_id == portfolio_id,
                        simulation_orders.c.status.in_(OPEN_STATUSES),
                    )
                )
                or 0
            )
        return ledger_policy_risk_inputs(
            portfolio_id=str(portfolio.id),
            portfolio_status=str(portfolio.status),
            latest_nav=row_dict(latest_nav) if latest_nav is not None else None,
            position_count=position_count + open_order_count,
            required_nav_date=required_nav_date,
        )

    def recommendation_policy_risk_inputs(
        self,
        recommendation_portfolio_id: str,
        *,
        required_nav_date: date | None = None,
    ) -> dict[str, Any]:
        """Resolve the sole selected account for a recommendation, fail-closed."""

        with self.engine.connect() as connection:
            account_ids = list(
                connection.scalars(
                    select(simulation_portfolios.c.id)
                    .where(
                        simulation_portfolios.c.source_type == "recommendation",
                        simulation_portfolios.c.source_id
                        == recommendation_portfolio_id,
                        simulation_portfolios.c.status.in_(("active", "paused")),
                    )
                    .order_by(simulation_portfolios.c.created_at)
                )
            )
        if not account_ids:
            # Cold start is needed to create the first recommendation before
            # its paper account exists. The status remains explicit.
            return {
                "contract_version": LEDGER_POLICY_RISK_VERSION,
                "risk_scope": RISK_SCOPE,
                "portfolio_id": None,
                "status": "initial_unbound_recommendation",
                "portfolio_drawdown": 0.0,
                "daily_return": 0.0,
                "allow_new_risk": True,
                "reasons": [],
            }
        if len(account_ids) > 1:
            return {
                "contract_version": LEDGER_POLICY_RISK_VERSION,
                "risk_scope": RISK_SCOPE,
                "portfolio_id": None,
                "status": "blocked_ambiguous_selected_account",
                "portfolio_drawdown": 0.0,
                "daily_return": 0.0,
                "allow_new_risk": False,
                "reasons": ["multiple_simulation_accounts_require_selection"],
                "candidate_portfolio_ids": [str(value) for value in account_ids],
            }
        return self.policy_risk_inputs(
            str(account_ids[0]),
            required_nav_date=required_nav_date,
        )

    @staticmethod
    def _apply_risk_exposure_override(
        *,
        adapter: str,
        target_payload: dict[str, Any],
        risk_exposure_override: float,
    ) -> dict[str, Any]:
        override = float(risk_exposure_override)
        if not isfinite(override) or not 0.0 <= override <= 1.0:
            raise ValueError("strategy risk exposure override must be between zero and one")
        payload = dict(target_payload)
        if adapter == "long_only":
            payload["target_weights"] = {
                instrument: float(weight) * override
                for instrument, weight in dict(
                    payload.get("target_weights") or {}
                ).items()
            }
            return payload
        if adapter != "pair":
            raise ValueError("unsupported simulation execution adapter")
        scaled_legs: list[dict[str, Any]] = []
        for raw_leg in payload.get("legs") or []:
            leg = dict(raw_leg)
            quantity = int(leg.get("target_quantity") or 0)
            leg["target_quantity"] = int(quantity * override) // 100 * 100
            scaled_legs.append(leg)
        payload["legs"] = scaled_legs
        return payload

    @staticmethod
    def _apply_member_new_risk_gate(
        *,
        adapter: str,
        target_payload: dict[str, Any],
        positions: dict[str, dict[str, Any]],
        portfolio_nav: float,
    ) -> dict[str, Any]:
        payload = dict(target_payload)
        if adapter == "long_only":
            if portfolio_nav <= 0:
                raise ValueError("simulation NAV must be positive for the member risk gate")
            current_weights = {
                instrument: max(0.0, float(position.get("market_value") or 0.0))
                / portfolio_nav
                for instrument, position in positions.items()
                if str(position.get("position_side") or "long") == "long"
            }
            payload["target_weights"] = {
                instrument: min(float(weight), current_weights.get(instrument, 0.0))
                for instrument, weight in dict(payload.get("target_weights") or {}).items()
            }
            return payload
        if adapter != "pair":
            raise ValueError("unsupported simulation execution adapter")
        gated_legs: list[dict[str, Any]] = []
        for raw_leg in payload.get("legs") or []:
            leg = dict(raw_leg)
            current = positions.get(str(leg.get("instrument") or "").upper()) or {}
            same_side = str(current.get("position_side") or "") == str(
                leg.get("position_side") or ""
            )
            current_quantity = int(current.get("quantity") or 0) if same_side else 0
            leg["target_quantity"] = min(
                int(leg.get("target_quantity") or 0),
                current_quantity,
            )
            gated_legs.append(leg)
        payload["legs"] = gated_legs
        return payload

    @staticmethod
    def _require_governed_long_only_target(
        *,
        batch: Any,
        portfolio: Any,
        target_payload: dict[str, Any],
    ) -> None:
        plan = target_payload.get("governed_order_plan")
        if not isinstance(plan, dict):
            raise ValueError(
                "long-only strategy simulation batch has no governed Qlib order-plan"
            )
        normalized = SimulationStore._normalize_target_payload(
            {"target_weights": target_payload.get("target_weights")},
            adapter="long_only",
        )
        source_snapshot = plan.get("source_snapshot")
        if not isinstance(source_snapshot, dict):
            raise ValueError("governed Qlib order-plan source snapshot is missing")
        required_matches = (
            (plan.get("format_version"), QLIB_ORDER_PLAN_FORMAT_VERSION),
            (
                plan.get("execution_contract_hash"),
                str(portfolio.execution_contract_hash),
            ),
            (
                plan.get("target_weights_sha256"),
                _canonical_hash(normalized),
            ),
            (
                source_snapshot.get("id"),
                str(batch.source_snapshot_id),
            ),
            (
                source_snapshot.get("dataset_identity_sha256"),
                str(batch.daily_dataset_identity_sha256),
            ),
            (
                source_snapshot.get("dataset_lineage_id"),
                str(batch.daily_dataset_lineage_id),
            ),
        )
        if any(observed != expected for observed, expected in required_matches):
            raise ValueError("governed Qlib order-plan failed batch-time verification")
        for field in ("signal_at", "execution_not_before"):
            expected_timestamp = getattr(batch, field)
            planned_timestamp = plan.get(field)
            if expected_timestamp is None:
                if planned_timestamp is not None:
                    raise ValueError(
                        "governed Qlib order-plan timing failed batch-time verification"
                    )
                continue
            if planned_timestamp is None or _parse_aware_timestamp(
                planned_timestamp,
                field=field,
            ) != expected_timestamp.astimezone(UTC):
                raise ValueError(
                    "governed Qlib order-plan timing failed batch-time verification"
                )
        signal_snapshot = plan.get("signal_snapshot")
        if batch.signal_at is not None and (
            not isinstance(signal_snapshot, dict)
            or not str(signal_snapshot.get("name") or "")
            or any(
                not _is_sha256(signal_snapshot.get(field))
                for field in (
                    "dataset_identity_sha256",
                    "dataset_lineage_id",
                    "source_lineage_id",
                )
            )
        ):
            raise ValueError(
                "governed Qlib order-plan signal snapshot failed verification"
            )
        if not _is_sha256(plan.get("manifest_sha256")) or not _is_sha256(
            plan.get("target_weights_file_sha256")
        ):
            raise ValueError("governed Qlib order-plan artifact identities are invalid")
        require_qlib_workflow_identity(plan.get("qlib_workflow"))

    @staticmethod
    def _runtime_execution_policy(
        *,
        portfolio: Any,
        batch: Any,
        execution_evidence: dict[str, Any],
        requires_volume_profile: bool,
    ) -> dict[str, Any]:
        stored = dict(portfolio.execution_policy_json or {})
        stored["simulation_semantics_sha256"] = str(
            batch.simulation_semantics_sha256
        )
        if str(portfolio.execution_adapter) == "pair":
            return stored
        algorithm = str(stored.get("execution_algorithm") or "")
        if algorithm != "vwap":
            if execution_evidence.get("execution_volume_profile") is not None:
                raise ValueError(
                    "non-VWAP simulation cannot accept execution volume-profile evidence"
                )
            return stored
        if stored.get("volume_profile_method") != VWAP_PROFILE_METHOD:
            raise ValueError("simulation VWAP profile method is not governed")
        if not requires_volume_profile:
            if any(
                execution_evidence.get(field) is not None
                for field in (
                    "execution_volume_profile",
                    "execution_volume_profile_evidence",
                    "execution_volume_profile_sha256",
                )
            ):
                raise ValueError(
                    "empty VWAP simulation cannot accept execution volume-profile evidence"
                )
            return stored
        lookback_days = int(stored.get("volume_profile_lookback_days") or 0)
        profile = execution_evidence.get("execution_volume_profile")
        evidence = execution_evidence.get("execution_volume_profile_evidence")
        if not isinstance(profile, list) or not profile or not isinstance(evidence, dict):
            raise ValueError(
                "VWAP simulation requires a historical profile from the bound "
                "Qlib execution dataset"
            )
        try:
            profile_start = date.fromisoformat(str(evidence["start"]))
            profile_end = date.fromisoformat(str(evidence["end"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("VWAP execution profile dates are invalid") from exc
        identity = {
            "method": VWAP_PROFILE_METHOD,
            "lookback_days": lookback_days,
            "start": profile_start.isoformat(),
            "end": profile_end.isoformat(),
            "trade_date": batch.trade_date.isoformat(),
            "dataset_identity_sha256": str(
                batch.execution_dataset_identity_sha256
            ),
            "dataset_lineage_id": str(batch.execution_dataset_lineage_id),
            "simulation_semantics_sha256": stored.get(
                "simulation_semantics_sha256"
            ),
            "profile": profile,
        }
        if (
            evidence.get("method") != VWAP_PROFILE_METHOD
            or int(evidence.get("lookback_days") or 0) != lookback_days
            or evidence.get("future_data_used") is not False
            or profile_end >= batch.trade_date
            or str(evidence.get("dataset_identity_sha256") or "")
            != str(batch.execution_dataset_identity_sha256)
            or str(evidence.get("dataset_lineage_id") or "")
            != str(batch.execution_dataset_lineage_id)
            or str(evidence.get("simulation_semantics_sha256") or "")
            != str(stored.get("simulation_semantics_sha256") or "")
            or execution_evidence.get("execution_volume_profile_sha256")
            != _canonical_hash(identity)
        ):
            raise ValueError(
                "VWAP execution profile does not match the immutable simulation contract"
            )
        return {
            **stored,
            "volume_profile": profile,
        }

    def _open_order_book(
        self, connection: Any, portfolio: Any, batch: Any, now: datetime
    ) -> dict[str, Any]:
        """Activate this batch's planned orders and load the working order book.

        planned -> open for every portfolio order whose window allows today
        (a superseding plan re-activates orders created by an earlier queued
        batch); still-open orders from earlier batches carry over; orders
        whose window lapsed before today expire (fail closed, never
        executed). Returns the executable rows keyed by order id.
        """

        day_start = datetime.combine(
            batch.trade_date, time(0, 0), ZoneInfo("Asia/Shanghai")
        )
        day_end = datetime.combine(
            batch.trade_date, time(15, 0), ZoneInfo("Asia/Shanghai")
        )
        connection.execute(
            update(simulation_orders)
            .where(
                simulation_orders.c.portfolio_id == portfolio.id,
                simulation_orders.c.status == STATUS_PLANNED,
                or_(
                    simulation_orders.c.not_before.is_(None),
                    simulation_orders.c.not_before <= day_end,
                ),
            )
            .values(status=STATUS_OPEN, updated_at=now)
        )
        stale = connection.execute(
            select(simulation_orders).where(
                simulation_orders.c.portfolio_id == portfolio.id,
                simulation_orders.c.status.in_(OPEN_STATUSES),
                func.coalesce(simulation_orders.c.not_after, simulation_orders.c.expires_at)
                < day_start,
            )
        ).all()
        for row in stale:
            if str(row.side) == "buy":
                self._move_order_reservation(
                    connection,
                    portfolio_id=str(portfolio.id),
                    order_id=str(row.id),
                    event_key=f"order:{row.id}:release:{batch.id}:expired",
                    event_type="release",
                    amount=None,
                    occurred_at=now,
                    now=now,
                    batch_id=str(batch.id),
                    details={"reason": "execution_window_elapsed"},
                )
            else:
                self._move_sell_reservation(
                    connection,
                    portfolio_id=str(portfolio.id),
                    order_id=str(row.id),
                    event_key=f"order:{row.id}:security-release:{batch.id}:expired",
                    event_type="release",
                    quantity=None,
                    occurred_at=now,
                    now=now,
                    batch_id=str(batch.id),
                    details={"reason": "execution_window_elapsed"},
                )
            connection.execute(
                update(simulation_orders)
                .where(simulation_orders.c.id == row.id)
                .values(status=STATUS_EXPIRED, updated_at=now)
            )
            connection.execute(
                insert(simulation_events).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio.id,
                    batch_id=batch.id,
                    trade_date=batch.trade_date,
                    severity="info",
                    event_type="order_expired",
                    instrument=str(row.instrument),
                    reason="execution_window_elapsed",
                    details_json={"order_id": str(row.id)},
                    created_at=now,
                )
            )
        rows = connection.execute(
            select(simulation_orders).where(
                simulation_orders.c.portfolio_id == portfolio.id,
                simulation_orders.c.status == STATUS_OPEN,
            )
        ).all()
        working: dict[str, Any] = {}
        for row in rows:
            if int(row.requested_quantity) - int(row.filled_quantity) <= 0:
                continue
            if row.not_before is not None and row.not_before > day_end:
                continue
            if row.not_after is not None and row.not_after < day_start:
                continue
            if (
                str(row.side) == "buy"
                and self._reservation_total(connection, str(row.id)) <= 0
            ):
                raise RuntimeError(
                    f"working buy order {row.id} has no frozen cash reservation"
                )
            if (
                str(row.side) == "sell"
                and self._security_reservation_total(connection, str(row.id))
                != int(row.requested_quantity) - int(row.filled_quantity)
            ):
                raise RuntimeError(
                    f"working sell order {row.id} does not match its frozen "
                    "security reservation"
                )
            working[str(row.id)] = row
        return working

    def process_batch(
        self,
        batch_id: str,
        *,
        minute_bars: pd.DataFrame,
        closing_prices: dict[str, dict[str, Any]],
        execution_evidence: dict[str, Any],
        corporate_actions: list[dict[str, Any]] | None = None,
        corporate_events: list[dict[str, Any]] | None = None,
        industry_snapshot: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compute, book and value a simulation day in one database transaction."""

        now = _now()
        with self.engine.begin() as connection:
            batch = connection.execute(
                select(simulation_batches)
                .where(simulation_batches.c.id == batch_id)
                .with_for_update()
            ).first()
            if batch is None:
                raise KeyError(batch_id)
            if batch.status == "succeeded":
                return self._batch_dict(batch)
            if batch.status != "queued":
                raise ValueError("only queued simulation batches may execute")
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == batch.portfolio_id)
                .with_for_update()
            ).one()
            if portfolio.status != "active":
                raise ValueError("simulation portfolio is not active")
            self._require_current_source_contract(connection, portfolio)
            if (
                str(batch.execution_contract_hash)
                != str(portfolio.execution_contract_hash)
                or str(batch.execution_adapter) != str(portfolio.execution_adapter)
            ):
                raise ValueError("simulation batch contract no longer matches its portfolio")
            expected_execution = {
                "dataset_identity_sha256": str(
                    batch.execution_dataset_identity_sha256
                ),
                "dataset_lineage_id": str(batch.execution_dataset_lineage_id),
                "execution_contract_version": str(
                    portfolio.execution_field_contract_version
                ),
                "execution_contract_hash": str(portfolio.execution_contract_hash),
                "simulation_semantics_sha256": str(
                    batch.simulation_semantics_sha256
                ),
            }
            mismatches = [
                field
                for field, expected in expected_execution.items()
                if str(execution_evidence.get(field) or "") != expected
            ]
            if mismatches:
                raise ValueError(
                    "simulation execution evidence does not match the bound dataset: "
                    + ", ".join(mismatches)
                )
            if str(execution_evidence.get("batch_id") or "") != str(batch.id):
                raise ValueError("simulation execution evidence does not match the batch")
            try:
                next_trade_date = date.fromisoformat(
                    str(execution_evidence["next_trade_date"])
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "simulation execution evidence requires the bound Qlib "
                    "next trading session"
                ) from exc
            if not (
                batch.trade_date < next_trade_date
                <= batch.trade_date + timedelta(days=31)
            ):
                raise ValueError(
                    "simulation next trading session evidence is outside its "
                    "settlement horizon"
                )
            batch_target_payload = dict(batch.target_payload_json or {})
            decision_valuation_event = (
                self._validate_account_decision_valuation_batch(
                    connection,
                    batch=batch,
                    target_payload=batch_target_payload,
                )
            )
            if str(portfolio.execution_frequency) == "day":
                governed_plan = batch_target_payload.get("governed_order_plan")
                if not isinstance(governed_plan, dict):
                    raise ValueError(
                        "daily simulation batch has no governed settlement binding"
                    )
                settlement_binding = validate_settlement_calendar_binding(
                    governed_plan.get("settlement_calendar_binding"),
                    trade_date=batch.trade_date,
                    dataset_identity_sha256=str(
                        batch.execution_dataset_identity_sha256
                    ),
                    dataset_lineage_id=str(batch.execution_dataset_lineage_id),
                )
                if next_trade_date.isoformat() != settlement_binding["next_trade_date"]:
                    raise ValueError(
                        "simulation next trading session differs from its batch binding"
                    )
                validate_settlement_calendar_evidence(
                    execution_evidence.get("settlement_calendar_evidence"),
                    trade_date=batch.trade_date,
                    next_trade_date=next_trade_date,
                    dataset_identity_sha256=str(
                        batch.execution_dataset_identity_sha256
                    ),
                    dataset_lineage_id=str(batch.execution_dataset_lineage_id),
                    calendar_file_sha256=str(
                        settlement_binding["calendar_file_sha256"]
                    ),
                )
            normalized_actions = [dict(item) for item in (corporate_actions or [])]
            if corporate_actions is not None:
                # 公司行动与行情一样属于不可变执行输入：必须与证据哈希绑定。
                actions_hash = corporate_actions_sha256(normalized_actions)
                if (
                    str(execution_evidence.get("corporate_actions_sha256") or "")
                    != actions_hash
                ):
                    raise ValueError(
                        "simulation corporate actions do not match the execution evidence"
                    )
            normalized_events = [dict(item) for item in (corporate_events or [])]
            if corporate_events is not None:
                # 非分红类公司行动（公告/拆并股/代码变更等）同样绑定证据哈希。
                events_hash = corporate_actions_sha256(normalized_events)
                if (
                    str(execution_evidence.get("corporate_events_sha256") or "")
                    != events_hash
                ):
                    raise ValueError(
                        "simulation corporate events do not match the execution evidence"
                    )
            normalized_industries: dict[str, str] = {}
            if industry_snapshot is None:
                if execution_evidence.get("industry_snapshot_sha256") is not None:
                    raise ValueError(
                        "simulation industry snapshot evidence has no payload"
                    )
            else:
                normalized_industries = {
                    str(instrument).strip().upper(): str(industry).strip()
                    for instrument, industry in industry_snapshot.items()
                }
                if any(
                    not instrument or not industry
                    for instrument, industry in normalized_industries.items()
                ):
                    raise ValueError(
                        "simulation industry snapshot contains an empty key or value"
                    )
                industry_identity = {
                    "trade_date": batch.trade_date.isoformat(),
                    "values": dict(sorted(normalized_industries.items())),
                }
                if str(
                    execution_evidence.get("industry_snapshot_sha256") or ""
                ) != _canonical_hash(industry_identity):
                    raise ValueError(
                        "simulation industry snapshot does not match its "
                        "immutable execution evidence"
                    )
            for field in ("signal_at", "execution_not_before"):
                expected_timestamp = getattr(batch, field)
                observed_timestamp = execution_evidence.get(field)
                if expected_timestamp is None:
                    if observed_timestamp is not None:
                        raise ValueError(
                            "simulation execution evidence contains unexpected timing"
                        )
                    continue
                if observed_timestamp is None or _parse_aware_timestamp(
                    observed_timestamp,
                    field=field,
                ) != expected_timestamp.astimezone(UTC):
                    raise ValueError(
                        "simulation execution evidence does not match order-plan timing"
                    )
            target_payload = dict(batch_target_payload)
            order_plan_mode = False
            working_orders: dict[str, Any] = {}
            if batch.recommendation_snapshot_id:
                snapshot = connection.execute(
                    select(recommendation_snapshots).where(
                        recommendation_snapshots.c.id == batch.recommendation_snapshot_id
                    )
                ).one()
                if snapshot.status != "succeeded":
                    raise ValueError("simulation recommendation snapshot is no longer valid")
                target_payload = {
                    "target_weights": {
                        str(item.instrument): float(item.weight)
                        for item in connection.execute(
                            select(recommendation_holdings).where(
                                recommendation_holdings.c.snapshot_id == snapshot.id
                            )
                        )
                    }
                }
            elif target_payload.get("order_plan") is not None:
                order_plan = dict(target_payload["order_plan"])
                if order_plan.get("plan_version") != ORDER_PLAN_MODEL_VERSION:
                    raise ValueError("simulation order-plan batch has an unsupported version")
                if str(portfolio.execution_adapter) != "long_only":
                    raise ValueError("order-plan batches require the long_only adapter")
                if not str(order_plan.get("target_version") or ""):
                    raise ValueError("simulation order-plan batch has no target version")
                bound_plan = order_plan.get("account_netting_plan_id")
                if str(bound_plan or "") != str(batch.account_netting_plan_id or ""):
                    raise ValueError("order-plan batch netting binding has drifted")
                order_plan_mode = True
                working_orders = self._open_order_book(connection, portfolio, batch, now)
                if (
                    decision_valuation_event is not None
                    and decision_valuation_event[
                        "prior_plan_continuation_allowed"
                    ]
                    is False
                    and working_orders
                ):
                    raise ValueError(
                        "decision-only valuation forbids unapproved working-order "
                        "continuation"
                    )
            elif not target_payload:
                raise ValueError("simulation batch has no governed target payload")
            if (
                not order_plan_mode
                and not batch.recommendation_snapshot_id
                and str(portfolio.execution_adapter) == "long_only"
            ):
                self._require_governed_long_only_target(
                    batch=batch,
                    portfolio=portfolio,
                    target_payload=target_payload,
                )
            position_state = {
                str(item.instrument): row_dict(item)
                for item in connection.execute(
                    select(simulation_positions).where(
                        simulation_positions.c.portfolio_id == portfolio.id
                    )
                )
            }
            # 持仓批次、股息权利与未结应收随持仓一起加载；无批次的旧持仓
            # 不附加 lots 键，由引擎按遗留语义合成（行为不变）。
            lots_by_instrument: dict[str, list[dict[str, Any]]] = {}
            for lot_row in connection.execute(
                select(simulation_position_lots).where(
                    simulation_position_lots.c.portfolio_id == portfolio.id
                )
            ):
                lots_by_instrument.setdefault(str(lot_row.instrument), []).append(
                    {
                        "lot_key": str(lot_row.lot_key),
                        "acquired_at": lot_row.acquired_at,
                        "sellable_from": lot_row.sellable_from,
                        "quantity": int(lot_row.quantity),
                        "cost_basis_total": float(lot_row.cost_basis_total),
                        "origin": str(lot_row.origin),
                        "entitlements": [],
                    }
                )
            for entitlement_row in connection.execute(
                select(simulation_dividend_entitlements).where(
                    simulation_dividend_entitlements.c.portfolio_id == portfolio.id
                )
            ):
                for lot in lots_by_instrument.get(str(entitlement_row.instrument), []):
                    if lot["lot_key"] == str(entitlement_row.lot_key):
                        lot["entitlements"].append(
                            {
                                "record_date": entitlement_row.record_date,
                                "kind": str(entitlement_row.kind),
                                "income_per_share": float(entitlement_row.income_per_share),
                                "untaxed_quantity": int(entitlement_row.untaxed_quantity),
                                "liability_per_share": float(
                                    entitlement_row.liability_per_share
                                ),
                            }
                        )
            for instrument, lots in lots_by_instrument.items():
                if instrument in position_state:
                    position_state[instrument]["lots"] = lots
            applied_ex_dates: dict[str, set[str]] = {}
            open_receivables: list[dict[str, Any]] = []
            for action_row in connection.execute(
                select(simulation_dividend_actions).where(
                    simulation_dividend_actions.c.portfolio_id == portfolio.id
                )
            ):
                instrument = str(action_row.instrument)
                applied_ex_dates.setdefault(instrument, set()).add(
                    action_row.ex_date.isoformat()
                )
                if str(action_row.status) == "accrued":
                    open_receivables.append(
                        {
                            "instrument": instrument,
                            "ex_date": action_row.ex_date,
                            "record_date": action_row.record_date,
                            "pay_date": action_row.pay_date,
                            "quantity": int(action_row.eligible_quantity),
                            "cash_per_share": float(action_row.cash_per_share),
                            "amount": float(action_row.receivable_amount),
                            "tax_rule_version": str(action_row.tax_rule_version),
                            "valuation_uncertain": bool(action_row.valuation_uncertain),
                        }
                    )
            for instrument, ex_dates in applied_ex_dates.items():
                if instrument in position_state:
                    position_state[instrument]["_applied_ca_ex_dates"] = tuple(
                        sorted(ex_dates)
                    )
            # 非分红类公司行动台账：已应用事件键是幂等状态；持有人选择事件
            # 重新推导持仓的 choice_pending 标记（处置前卖出仅提示不阻断）。
            applied_event_keys: list[str] = []
            for event_row in connection.execute(
                select(simulation_corporate_events).where(
                    simulation_corporate_events.c.portfolio_id == portfolio.id
                )
            ):
                event_key = str(event_row.event_key)
                applied_event_keys.append(event_key)
                if str(event_row.event_type) == "choice_required":
                    instrument = str(event_row.instrument)
                    if instrument in position_state:
                        position_state[instrument]["choice_pending"] = {
                            "event_key": event_key,
                            "since": event_row.effective_date.isoformat(),
                            "title": (event_row.details_json or {}).get("title"),
                            "manual_action_required": True,
                        }
            strategy_version_id = self._source_strategy_version_id(connection, portfolio)
            strategy_risk_state = (
                load_strategy_risk_state(connection, str(strategy_version_id))
                if strategy_version_id
                else {
                    "strategy_version_id": None,
                    "state": "active",
                    "allow_new_risk": True,
                    "risk_exposure_override": 1.0,
                    "event_ids": [],
                    "allocation_ids": [],
                }
            )
            risk_exposure_override = float(
                strategy_risk_state["risk_exposure_override"]
            )
            if risk_exposure_override < 1.0:
                target_payload = self._apply_risk_exposure_override(
                    adapter=str(portfolio.execution_adapter),
                    target_payload=target_payload,
                    risk_exposure_override=risk_exposure_override,
                )
            if not strategy_risk_state["allow_new_risk"]:
                target_payload = self._apply_member_new_risk_gate(
                    adapter=str(portfolio.execution_adapter),
                    target_payload=target_payload,
                    positions=position_state,
                    portfolio_nav=float(portfolio.nav),
                )
            execution_policy = self._runtime_execution_policy(
                portfolio=portfolio,
                batch=batch,
                execution_evidence=execution_evidence,
                requires_volume_profile=(
                    str(portfolio.execution_adapter) == "pair"
                    or any(
                        int(position.get("quantity") or 0) != 0
                        for position in position_state.values()
                    )
                    or any(
                        float(weight) > 0
                        for weight in dict(
                            target_payload.get("target_weights") or {}
                        ).values()
                    )
                    or bool(working_orders)
                ),
            )
            cost_schedule = CostScheduleBook.from_mapping(
                dict(portfolio.execution_policy_json or {}).get("cost_model")
            )
            # 当日已确认外部现金流（设计 4.4/12.1）：open 当日可投资，close 次日起。
            flow_open = 0.0
            flow_close = 0.0
            for flow_row in connection.execute(
                select(simulation_external_flows).where(
                    simulation_external_flows.c.portfolio_id == portfolio.id,
                    simulation_external_flows.c.trade_date == batch.trade_date,
                )
            ):
                if str(flow_row.timing) == "open":
                    flow_open += float(flow_row.amount)
                else:
                    flow_close += float(flow_row.amount)
            cash_as_of = self._session_time(batch.trade_date, 9, 30)
            cash_view = self._assert_cash_lots_reconcile(
                connection,
                str(portfolio.id),
                expected_cash=portfolio.cash,
                as_of=cash_as_of,
            )
            if (
                flow_open < 0
                and abs(Decimal(str(flow_open)))
                > cash_view["withdrawable_cash"] + Decimal("0.000001")
            ):
                raise RuntimeError(
                    "external open withdrawal exceeds withdrawable cash"
                )
            if (
                flow_close < 0
                and abs(Decimal(str(flow_close)))
                > cash_view["withdrawable_cash"] + Decimal("0.000001")
            ):
                raise RuntimeError(
                    "external close withdrawal exceeds withdrawable cash"
                )
            prior_investment_wealth = (
                float(portfolio.investment_wealth)
                if portfolio.investment_wealth is not None
                else None
            )
            twr_high_water_mark = (
                float(portfolio.twr_high_water_mark)
                if portfolio.twr_high_water_mark is not None
                else None
            )
            if prior_investment_wealth is None:
                settled_days = connection.execute(
                    select(func.count())
                    .select_from(simulation_nav)
                    .where(simulation_nav.c.portfolio_id == portfolio.id)
                ).scalar_one()
                if not settled_days:
                    # 迁移前创建但尚无 NAV 行的账户：单位化链从 1.0 起。
                    prior_investment_wealth = 1.0
                    twr_high_water_mark = 1.0
            if order_plan_mode:
                blocked_buys: list[str] = []
                specs: list[dict[str, Any]] = []
                working_reserved_cash = Decimal("0")
                for row in working_orders.values():
                    side = str(row.side)
                    if side == "buy" and not strategy_risk_state["allow_new_risk"]:
                        # 风险硬门：只阻断新增风险，工作买单保留待风险解除。
                        blocked_buys.append(str(row.id))
                        continue
                    reserved_cash = (
                        self._reservation_total(connection, str(row.id))
                        if side == "buy"
                        else Decimal("0")
                    )
                    working_reserved_cash += reserved_cash
                    specs.append(
                        {
                            "instrument": str(row.instrument),
                            "side": side,
                            "requested_quantity": int(row.requested_quantity)
                            - int(row.filled_quantity),
                            "target_weight": float(row.target_weight),
                            "order_ref": str(row.id),
                            "limit_price": (
                                float(row.limit_price) if row.limit_price is not None else None
                            ),
                            "not_before": row.not_before,
                            "not_after": row.not_after,
                            "reserved_cash": float(reserved_cash),
                        }
                    )
                result = execute_simulation_day(
                    trade_date=batch.trade_date,
                    cash=float(portfolio.cash),
                    prior_nav=float(portfolio.nav),
                    high_water_mark=float(portfolio.high_water_mark),
                    positions=position_state,
                    target_weights={},
                    minute_bars=minute_bars,
                    closing_prices=closing_prices,
                    cost_schedule=cost_schedule,
                    execution_policy=execution_policy,
                    signal_at=batch.signal_at,
                    corporate_actions=normalized_actions,
                    dividend_receivables=open_receivables,
                    corporate_events=normalized_events,
                    applied_event_keys=applied_event_keys,
                    order_specs_override=specs,
                    tradable_cash=float(
                        cash_view["tradable_cash"] + working_reserved_cash
                    ),
                    external_flow_open=flow_open,
                    external_flow_close=flow_close,
                    prior_investment_wealth=prior_investment_wealth,
                    twr_high_water_mark=twr_high_water_mark,
                )
                result.setdefault("dividend_receivables", list(open_receivables))
                result.setdefault("corporate_actions_applied", [])
                day_end = datetime.combine(
                    batch.trade_date, time(15, 0), ZoneInfo("Asia/Shanghai")
                )
                for order in result["orders"]:
                    row = working_orders[str(order["order_ref"])]
                    requested_total = int(row.requested_quantity)
                    cumulative_filled = int(row.filled_quantity) + int(
                        order["filled_quantity"]
                    )
                    order["filled_quantity"] = cumulative_filled
                    order["filled_value"] = float(row.filled_value) + float(
                        order["filled_value"]
                    )
                    order["requested_quantity"] = requested_total
                    order["requested_value"] = float(row.requested_value)
                    order["capacity_fill_ratio"] = cumulative_filled / requested_total
                    window_end = row.not_after or row.expires_at
                    window_closed = window_end is not None and window_end <= day_end
                    if cumulative_filled >= requested_total:
                        order["status"] = "filled"
                    elif window_closed:
                        order["status"] = (
                            "partial_filled_expired" if cumulative_filled else "expired"
                        )
                    else:
                        # 执行窗口未结束：工作单保持 open，余量留给后续批次。
                        order["status"] = "open"
                if blocked_buys:
                    result["events"].append(
                        {
                            "severity": "warning",
                            "event_type": "order_plan_buys_blocked",
                            "instrument": None,
                            "reason": "strategy_risk_gate_pauses_new_risk",
                            "details": {"order_ids": sorted(blocked_buys)},
                        }
                    )
            elif str(portfolio.execution_adapter) == "pair":
                governed_plan = target_payload.get("governed_pair_plan")
                if not isinstance(governed_plan, dict):
                    raise ValueError("pair simulation batch has no governed artifact plan")
                shortability_binding = dict(
                    governed_plan.get("shortability_dataset") or {}
                )
                if not (
                    execution_evidence.get("pair_plan_sha256")
                    == governed_plan.get("pair_plan_sha256")
                    and execution_evidence.get("pair_artifact_manifest_sha256")
                    == governed_plan.get("pair_artifact_manifest_sha256")
                    and execution_evidence.get("shortability_source_sha256")
                    == shortability_binding.get("source_sha256")
                    and execution_evidence.get(
                        "shortability_snapshot_manifest_sha256"
                    )
                    == shortability_binding.get("manifest_sha256")
                ):
                    raise ValueError(
                        "pair execution evidence does not match the governed backtest plan"
                    )
                shortability = execution_evidence.get("shortability")
                if not isinstance(shortability, dict):
                    raise ValueError("pair simulation requires dated shortability evidence")
                if str(execution_evidence.get("shortability_trade_date") or "") != (
                    batch.trade_date.isoformat()
                ):
                    raise ValueError("pair shortability evidence is not valid for the trade date")
                shortability_sha256 = execution_evidence.get(
                    "shortability_evidence_sha256"
                )
                if not _is_sha256(shortability_sha256):
                    raise ValueError("pair shortability evidence requires an immutable identity")
                result = execute_atomic_pair_day(
                    trade_date=batch.trade_date,
                    cash=float(portfolio.cash),
                    prior_nav=float(portfolio.nav),
                    high_water_mark=float(portfolio.high_water_mark),
                    positions=position_state,
                    target_payload=target_payload,
                    minute_bars=minute_bars,
                    closing_prices=closing_prices,
                    shortability={
                        str(key).upper(): value is True
                        for key, value in shortability.items()
                    },
                    cost_schedule=cost_schedule,
                    execution_policy=execution_policy,
                    external_flow_open=flow_open,
                    external_flow_close=flow_close,
                    prior_investment_wealth=prior_investment_wealth,
                    twr_high_water_mark=twr_high_water_mark,
                )
                result["shortability_evidence_sha256"] = str(shortability_sha256)
            else:
                result = execute_simulation_day(
                    trade_date=batch.trade_date,
                    cash=float(portfolio.cash),
                    prior_nav=float(portfolio.nav),
                    high_water_mark=float(portfolio.high_water_mark),
                    positions=position_state,
                    target_weights=dict(target_payload["target_weights"]),
                    minute_bars=minute_bars,
                    closing_prices=closing_prices,
                    cost_schedule=cost_schedule,
                    execution_policy=execution_policy,
                    signal_at=batch.signal_at,
                    corporate_actions=normalized_actions,
                    dividend_receivables=open_receivables,
                    corporate_events=normalized_events,
                    applied_event_keys=applied_event_keys,
                    tradable_cash=float(cash_view["tradable_cash"]),
                    external_flow_open=flow_open,
                    external_flow_close=flow_close,
                    prior_investment_wealth=prior_investment_wealth,
                    twr_high_water_mark=twr_high_water_mark,
                )
                result.setdefault("dividend_receivables", list(open_receivables))
                result.setdefault("corporate_actions_applied", [])
            result["next_trade_date"] = next_trade_date
            if str(portfolio.execution_adapter) == "pair":
                # 配对账户按设计只作离线研究：公司行动不入账，但必须留痕。
                pair_relevant = [
                    action
                    for action in normalized_actions
                    if str(action.get("instrument") or "").upper() in position_state
                ]
                pair_relevant += [
                    item
                    for item in normalized_events
                    if str(item.get("instrument") or "").upper() in position_state
                ]
                if pair_relevant:
                    result["events"].append(
                        {
                            "severity": "warning",
                            "event_type": "corporate_action_unhandled_pair",
                            "instrument": None,
                            "reason": "pair_adapter_does_not_book_corporate_actions",
                            "details": {
                                "instruments": sorted(
                                    str(item["instrument"]).upper()
                                    for item in pair_relevant
                                ),
                            },
                        }
                    )
            if (
                not strategy_risk_state["allow_new_risk"]
                or risk_exposure_override < 1.0
            ):
                result["events"].append(
                    {
                        "severity": "warning",
                        "event_type": "strategy_risk_gate_applied",
                        "reason": str(strategy_risk_state["state"]),
                        "details": strategy_risk_state,
                    }
                )
                result["strategy_risk_state"] = strategy_risk_state
            result["industry_snapshot"] = normalized_industries
            result["industry_snapshot_sha256"] = (
                str(execution_evidence["industry_snapshot_sha256"])
                if industry_snapshot is not None
                else None
            )
            benchmark = str(portfolio.benchmark or "").strip().upper()
            prior_benchmark_nav = connection.execute(
                select(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio.id,
                    simulation_nav.c.trade_date < batch.trade_date,
                )
                .order_by(simulation_nav.c.trade_date.desc())
                .limit(1)
            ).first()
            benchmark_day = (
                _benchmark_chain_day(
                    execution_evidence.get("benchmark_evidence"),
                    benchmark=benchmark,
                    signal_date=batch.signal_date,
                    trade_date=batch.trade_date,
                    execution_frequency=str(portfolio.execution_frequency),
                    dataset_identity_sha256=str(
                        batch.execution_dataset_identity_sha256
                    ),
                    dataset_lineage_id=str(batch.execution_dataset_lineage_id),
                    prior_nav=prior_benchmark_nav,
                )
                if benchmark
                else {
                    "status": "benchmark_not_bound_legacy",
                    "benchmark_close": None,
                    "benchmark_return": None,
                    "benchmark_wealth": None,
                    "evidence_sha256": None,
                }
            )
            result["benchmark"] = benchmark or None
            result["benchmark_day"] = benchmark_day
            result["nav_row"].update(
                {
                    "benchmark_close": benchmark_day["benchmark_close"],
                    "benchmark_return": benchmark_day["benchmark_return"],
                    "benchmark_wealth": benchmark_day["benchmark_wealth"],
                }
            )
            self._persist_result(connection, batch, portfolio, result, now)
        return self.get_batch(batch_id)

    def mark_batch_failed(self, batch_id: str, error: str) -> None:
        with self.engine.begin() as connection:
            result = connection.execute(
                update(simulation_batches)
                .where(
                    simulation_batches.c.id == batch_id,
                    simulation_batches.c.status == "queued",
                )
                .values(status="failed", error=error, finished_at=_now())
            )
            if not result.rowcount:
                existing = connection.execute(
                    select(simulation_batches.c.id).where(
                        simulation_batches.c.id == batch_id
                    )
                ).first()
                if existing is None:
                    raise KeyError(batch_id)
        # Design 11.3: a ledger-integrity failure is a severe anomaly and
        # engages the platform safe mode (new recommendations and simulation
        # orders stop until a manual release).
        if any(marker in error for marker in LEDGER_INTEGRITY_ERROR_MARKERS):
            SafeModeStore(self.database_url).activate(
                reason=f"simulation batch {batch_id} ledger integrity failure: {error[:500]}",
                source="simulation_ledger",
                actor="system",
                details={"batch_id": batch_id, "error": error[:2000]},
            )

    @staticmethod
    def _build_day_attribution(
        connection: Any,
        *,
        batch: Any,
        portfolio: Any,
        result: dict[str, Any],
        fill_order_ids_by_seq: dict[int, str],
    ) -> dict[str, Any]:
        """Build the immutable §9.4 activity attribution without inventing P&L."""

        order_ids = sorted(set(fill_order_ids_by_seq.values()))
        order_rows = {
            str(row.id): row
            for row in (
                connection.execute(
                    select(simulation_orders).where(
                        simulation_orders.c.id.in_(order_ids)
                    )
                ).all()
                if order_ids
                else []
            )
        }
        strategy_groups: dict[str, dict[str, Any]] = {}
        strategy_exact = True
        cost_components: dict[str, float] = {}
        asset_groups: dict[str, dict[str, Any]] = {}
        execution_groups: dict[str, dict[str, Any]] = {}

        for instrument, position in result["positions"].items():
            instrument = str(instrument)
            asset_type = infer_cn_asset_type(instrument)
            asset = asset_groups.setdefault(
                asset_type,
                {
                    "closing_market_value": 0.0,
                    "gross_turnover": 0.0,
                    "fees": 0.0,
                    "position_count": 0,
                },
            )
            asset["closing_market_value"] += float(
                position.get("market_value") or 0.0
            )
            asset["position_count"] += 1

        for fill_seq, fill in enumerate(result["fills"]):
            instrument = str(fill["instrument"])
            quantity = int(fill["quantity"])
            gross = float(fill["gross_value"])
            fee = float(fill["fee"])
            side = str(fill["side"])
            asset_type = infer_cn_asset_type(instrument)
            asset = asset_groups.setdefault(
                asset_type,
                {
                    "closing_market_value": 0.0,
                    "gross_turnover": 0.0,
                    "fees": 0.0,
                    "position_count": 0,
                },
            )
            asset["gross_turnover"] += gross
            asset["fees"] += fee
            for name, value in dict(fill.get("cost_breakdown") or {}).items():
                if isinstance(value, (int, float)) and isfinite(float(value)):
                    cost_components[str(name)] = (
                        cost_components.get(str(name), 0.0) + float(value)
                    )

            execution = execution_groups.setdefault(
                instrument,
                {
                    "side": side,
                    "filled_quantity": 0,
                    "gross_value": 0.0,
                    "fees": 0.0,
                    "capacity_quantity": 0,
                    "minute_volume": 0,
                },
            )
            execution["filled_quantity"] += quantity
            execution["gross_value"] += gross
            execution["fees"] += fee
            execution["capacity_quantity"] += int(fill["capacity_quantity"])
            execution["minute_volume"] += int(fill["minute_volume"])

            order_id = fill_order_ids_by_seq[fill_seq]
            order = order_rows[order_id]
            contribution = dict(order.strategy_contributions_json or {})
            members = dict(contribution.get("members") or {})
            same_side = {
                str(member): abs(float(values.get("net_contribution") or 0.0))
                for member, values in members.items()
                if (
                    float(values.get("net_contribution") or 0.0) > 0
                    if side == "buy"
                    else float(values.get("net_contribution") or 0.0) < 0
                )
            }
            denominator = sum(same_side.values())
            if denominator <= 0:
                strategy_exact = False
                same_side = {str(portfolio.source_id): 1.0}
                denominator = 1.0
            for strategy_id, weight in same_side.items():
                share = weight / denominator
                strategy = strategy_groups.setdefault(
                    strategy_id,
                    {
                        "filled_quantity": 0.0,
                        "gross_value": 0.0,
                        "fees": 0.0,
                        "instruments": {},
                    },
                )
                strategy["filled_quantity"] += quantity * share
                strategy["gross_value"] += gross * share
                strategy["fees"] += fee * share
                instrument_row = strategy["instruments"].setdefault(
                    instrument,
                    {
                        "side": side,
                        "filled_quantity": 0.0,
                        "gross_value": 0.0,
                        "fees": 0.0,
                    },
                )
                instrument_row["filled_quantity"] += quantity * share
                instrument_row["gross_value"] += gross * share
                instrument_row["fees"] += fee * share

        requested_by_instrument: dict[str, int] = {}
        cumulative_filled_by_instrument: dict[str, int] = {}
        rejected_by_instrument: dict[str, list[str]] = {}
        for order in result["orders"]:
            instrument = str(order["instrument"])
            requested_by_instrument[instrument] = (
                requested_by_instrument.get(instrument, 0)
                + int(order["requested_quantity"])
            )
            cumulative_filled_by_instrument[instrument] = (
                cumulative_filled_by_instrument.get(instrument, 0)
                + int(order["filled_quantity"])
            )
            if order.get("reject_reason"):
                rejected_by_instrument.setdefault(instrument, []).append(
                    str(order["reject_reason"])
                )
        for instrument in sorted(
            set(requested_by_instrument) | set(execution_groups)
        ):
            execution = execution_groups.setdefault(
                instrument,
                {
                    "side": None,
                    "filled_quantity": 0,
                    "gross_value": 0.0,
                    "fees": 0.0,
                    "capacity_quantity": 0,
                    "minute_volume": 0,
                },
            )
            requested = requested_by_instrument.get(instrument, 0)
            day_filled = int(execution["filled_quantity"])
            cumulative_filled = cumulative_filled_by_instrument.get(
                instrument, day_filled
            )
            execution["requested_quantity"] = requested
            execution["day_filled_quantity"] = day_filled
            execution["cumulative_filled_quantity"] = cumulative_filled
            execution["unfilled_quantity"] = max(
                requested - cumulative_filled, 0
            )
            execution["fill_ratio"] = (
                cumulative_filled / requested if requested else None
            )
            execution["average_fill_price"] = (
                execution["gross_value"] / day_filled if day_filled else None
            )
            position = result["positions"].get(instrument) or {}
            closing_price = position.get("market_price")
            execution["closing_reference_price"] = (
                float(closing_price) if closing_price is not None else None
            )
            if (
                day_filled
                and closing_price is not None
                and float(closing_price) > 0
            ):
                average_price = float(execution["average_fill_price"])
                close = float(closing_price)
                execution["adverse_price_deviation_bps"] = (
                    (average_price / close - 1.0) * 10_000
                    if execution["side"] == "buy"
                    else (close / average_price - 1.0) * 10_000
                )
                execution["price_deviation_status"] = "available"
            else:
                execution["adverse_price_deviation_bps"] = None
                execution["price_deviation_status"] = (
                    "no_fill"
                    if not day_filled
                    else "blocked_missing_closing_reference"
                )
            execution["rejection_reasons"] = sorted(
                set(rejected_by_instrument.get(instrument, []))
            )

        instruments = sorted(
            set(result["positions"])
            | {str(fill["instrument"]) for fill in result["fills"]}
        )
        industry_snapshot = {
            str(instrument).upper(): str(industry)
            for instrument, industry in dict(
                result.get("industry_snapshot") or {}
            ).items()
        }
        missing_industries = [
            instrument
            for instrument in instruments
            if instrument.upper() not in industry_snapshot
        ]
        industry_groups: dict[str, dict[str, Any]] = {}
        for instrument in instruments:
            industry_name = industry_snapshot.get(instrument.upper())
            if industry_name is None:
                continue
            group = industry_groups.setdefault(
                industry_name,
                {
                    "closing_market_value": 0.0,
                    "gross_turnover": 0.0,
                    "fees": 0.0,
                    "instruments": [],
                },
            )
            position = result["positions"].get(instrument) or {}
            group["closing_market_value"] += float(
                position.get("market_value") or 0.0
            )
            for fill in result["fills"]:
                if str(fill["instrument"]) != instrument:
                    continue
                group["gross_turnover"] += float(fill["gross_value"])
                group["fees"] += float(fill["fee"])
            group["instruments"].append(instrument)
        blocker_reasons = []
        if missing_industries:
            blocker_reasons.append(
                "blocked_missing_bound_industry_snapshot"
                if not industry_snapshot
                else "blocked_incomplete_bound_industry_snapshot"
            )
        strategy = {
            "status": (
                "exact_frozen_same_side_pro_rata"
                if strategy_exact
                else "source_level_fallback_no_frozen_member_contributions"
            ),
            "groups": strategy_groups,
        }
        industry = {
            "status": (
                (
                    "blocked_missing_bound_industry_snapshot"
                    if not industry_snapshot
                    else "blocked_incomplete_bound_industry_snapshot"
                )
                if missing_industries
                else (
                    "available"
                    if instruments
                    else "not_applicable_no_instruments"
                )
            ),
            "snapshot_sha256": result.get("industry_snapshot_sha256"),
            "groups": industry_groups,
            "unclassified_instruments": missing_industries,
        }
        asset = {
            "status": "available",
            "scope": "closing_exposure_and_trading_activity",
            "groups": asset_groups,
        }
        cost = {
            "status": "available",
            "total_fee": sum(float(fill["fee"]) for fill in result["fills"]),
            "components": cost_components,
        }
        execution = {
            "status": "available",
            "reference": "same_day_closing_price_when_position_remains",
            "instruments": execution_groups,
        }
        payload = {
            "batch_id": str(batch.id),
            "portfolio_id": str(portfolio.id),
            "trade_date": batch.trade_date.isoformat(),
            "strategy": strategy,
            "industry": industry,
            "asset": asset,
            "cost": cost,
            "execution": execution,
        }
        return {
            **payload,
            "coverage_status": "partial" if blocker_reasons else "complete",
            "blocker_reasons": blocker_reasons,
            "input_sha256": _canonical_hash(payload),
        }

    def _persist_result(
        self, connection: Any, batch: Any, portfolio: Any, result: dict[str, Any], now: datetime
    ) -> None:
        order_ids: dict[Any, str] = {}
        for order in result["orders"]:
            ref = order.get("order_ref")
            if ref is not None:
                # 持久订单：更新既有行（累计成交/终态或继续 open），不新建行。
                order_ids[str(ref)] = str(ref)
                connection.execute(
                    update(simulation_orders)
                    .where(simulation_orders.c.id == str(ref))
                    .values(
                        filled_quantity=int(order["filled_quantity"]),
                        status=order["status"],
                        reject_reason=order.get("reject_reason"),
                        filled_value=Decimal(str(order["filled_value"])),
                        capacity_fill_ratio=float(order["capacity_fill_ratio"]),
                        updated_at=now,
                    )
                )
                continue
            order_id = uuid.uuid4().hex
            key = (str(order["instrument"]), str(order["side"]))
            order_ids[key] = order_id
            expires_at = order.get("expires_at") or datetime.combine(
                batch.trade_date, datetime.max.time(), tzinfo=UTC
            )
            connection.execute(
                insert(simulation_orders).values(
                    id=order_id,
                    batch_id=batch.id,
                    portfolio_id=portfolio.id,
                    instrument=order["instrument"],
                    side=order["side"],
                    atomic_group_id=order.get("atomic_group_id"),
                    leg_no=order.get("leg_no"),
                    position_side=order.get("position_side", "long"),
                    borrow_cost=Decimal(str(order.get("borrow_cost", 0.0))),
                    target_weight=float(order["target_weight"]),
                    requested_quantity=int(order["requested_quantity"]),
                    filled_quantity=int(order["filled_quantity"]),
                    status=order["status"],
                    reject_reason=order.get("reject_reason"),
                    requested_value=Decimal(str(order["requested_value"])),
                    filled_value=Decimal(str(order["filled_value"])),
                    capacity_fill_ratio=float(order["capacity_fill_ratio"]),
                    expires_at=expires_at,
                    created_at=now,
                    updated_at=now,
                )
            )
        fill_ids: list[str] = []
        fill_ids_by_seq: dict[int, str] = {}
        fill_order_ids_by_seq: dict[int, str] = {}
        for fill_seq, fill in enumerate(result["fills"]):
            fill_id = uuid.uuid4().hex
            fill_ids.append(fill_id)
            fill_ids_by_seq[fill_seq] = fill_id
            fill_key = fill.get("order_ref") or (
                str(fill["instrument"]),
                str(fill["side"]),
            )
            fill_order_ids_by_seq[fill_seq] = order_ids[fill_key]
            connection.execute(
                insert(simulation_fills).values(
                    id=fill_id,
                    order_id=order_ids[fill_key],
                    batch_id=batch.id,
                    instrument=fill["instrument"],
                    side=fill["side"],
                    atomic_group_id=fill.get("atomic_group_id"),
                    leg_no=fill.get("leg_no"),
                    position_side=fill.get("position_side", "long"),
                    borrow_cost=Decimal(str(fill.get("borrow_cost", 0.0))),
                    executed_at=fill["executed_at"],
                    quantity=int(fill["quantity"]),
                    price=Decimal(str(fill["price"])),
                    gross_value=Decimal(str(fill["gross_value"])),
                    fee=Decimal(str(fill["fee"])),
                    cost_breakdown_json=fill["cost_breakdown"],
                    minute_volume=int(fill["minute_volume"]),
                    capacity_quantity=int(fill["capacity_quantity"]),
                )
            )
        immediate_sell_orders: dict[str, dict[str, Any]] = {}
        for fill_seq, fill in enumerate(result["fills"]):
            if (
                str(batch.execution_adapter) != "long_only"
                or str(fill["side"]) != "sell"
            ):
                continue
            order_id = fill_order_ids_by_seq[fill_seq]
            entry = immediate_sell_orders.setdefault(
                order_id,
                {
                    "instrument": str(fill["instrument"]),
                    "quantity": 0,
                    "occurred_at": fill["executed_at"],
                },
            )
            entry["quantity"] += int(fill["quantity"])
            entry["occurred_at"] = min(entry["occurred_at"], fill["executed_at"])
        for order_id, entry in immediate_sell_orders.items():
            if self._security_reservation_total(connection, order_id) > 0:
                continue
            self._freeze_sell_quantity(
                connection,
                portfolio_id=str(portfolio.id),
                order_id=order_id,
                instrument=str(entry["instrument"]),
                quantity=int(entry["quantity"]),
                trade_date=batch.trade_date,
                event_key=f"order:{order_id}:security-freeze:immediate",
                occurred_at=entry["occurred_at"],
                now=now,
                batch_id=str(batch.id),
                details={"reason": "immediate_order_fill"},
            )
        for index, flow in enumerate(result["cash_flows"]):
            raw_fill_seq = flow.get("fill_seq")
            fill_seq = int(raw_fill_seq) if raw_fill_seq is not None else None
            if fill_seq is None and str(flow["flow_type"]).startswith("pair_"):
                fill_seq = index if index in fill_ids_by_seq else None
            reference_id = (
                fill_ids_by_seq.get(fill_seq) if fill_seq is not None else None
            )
            order_id = (
                fill_order_ids_by_seq.get(fill_seq)
                if fill_seq is not None
                else None
            )
            occurred_at = (
                result["fills"][fill_seq]["executed_at"]
                if fill_seq is not None
                else self._session_time(
                    batch.trade_date,
                    15
                    if str(flow["flow_type"]).endswith("_close")
                    else 9,
                    30,
                )
            )
            amount = self._cash_amount(flow["amount"])
            connection.execute(
                insert(simulation_cash_flows).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio.id,
                    batch_id=batch.id,
                    trade_date=batch.trade_date,
                    flow_type=flow["flow_type"],
                    amount=amount,
                    balance_after=Decimal(str(flow["balance_after"])),
                    reference_id=reference_id,
                    created_at=now,
                )
            )
            event_key = (
                f"batch:{batch.id}:cash-flow:{index}:{flow['flow_type']}"
            )
            if amount > 0:
                flow_type = str(flow["flow_type"])
                if flow_type == "sell_settlement":
                    if order_id is None or fill_seq is None:
                        raise RuntimeError(
                            "sell settlement is missing its order/fill identity"
                        )
                    self._move_sell_reservation(
                        connection,
                        portfolio_id=str(portfolio.id),
                        order_id=order_id,
                        event_key=f"order:{order_id}:security-consume:{reference_id}",
                        event_type="consume",
                        quantity=int(result["fills"][fill_seq]["quantity"]),
                        occurred_at=occurred_at,
                        now=now,
                        batch_id=str(batch.id),
                        details={"fill_id": reference_id},
                    )
                    tradable_at = occurred_at
                    withdrawable_at = self._session_time(
                        result["next_trade_date"],
                        9,
                    )
                elif flow_type == "external_deposit_close":
                    tradable_at = self._session_time(
                        result["next_trade_date"],
                        9,
                    )
                    withdrawable_at = tradable_at
                else:
                    tradable_at = occurred_at
                    withdrawable_at = occurred_at
                self._create_cash_lot(
                    connection,
                    portfolio_id=str(portfolio.id),
                    event_key=event_key,
                    source_type=flow_type,
                    source_reference_id=reference_id,
                    amount=amount,
                    tradable_at=tradable_at,
                    withdrawable_at=withdrawable_at,
                    occurred_at=occurred_at,
                    now=now,
                    batch_id=str(batch.id),
                    order_id=order_id,
                    details={"cash_flow_index": index},
                )
            elif amount < 0:
                debit = -amount
                if str(flow["flow_type"]) == "buy_settlement" and order_id:
                    if self._reservation_total(connection, order_id) <= 0:
                        # Immediate target-weight orders have no pre-existing
                        # working window. Freeze the exact confirmed fill
                        # amount in the same atomic booking transaction before
                        # consuming it.
                        self._freeze_order_cash(
                            connection,
                            portfolio_id=str(portfolio.id),
                            order_id=order_id,
                            event_key=f"{event_key}:freeze",
                            amount=debit,
                            as_of=occurred_at,
                            now=now,
                            batch_id=str(batch.id),
                            details={
                                "reason": "immediate_order_fill",
                                "cash_flow_index": index,
                            },
                        )
                    self._move_order_reservation(
                        connection,
                        portfolio_id=str(portfolio.id),
                        order_id=order_id,
                        event_key=event_key,
                        event_type="consume_frozen",
                        amount=debit,
                        occurred_at=occurred_at,
                        now=now,
                        batch_id=str(batch.id),
                        details={
                            "fill_id": reference_id,
                            "cash_flow_index": index,
                        },
                    )
                else:
                    self._allocate_free_cash(
                        connection,
                        portfolio_id=str(portfolio.id),
                        event_key=event_key,
                        event_type="consume_free",
                        amount=debit,
                        as_of=occurred_at,
                        now=now,
                        require_withdrawable=str(flow["flow_type"]).startswith(
                            "external_withdrawal"
                        ),
                        batch_id=str(batch.id),
                        order_id=order_id,
                        details={
                            "flow_type": str(flow["flow_type"]),
                            "reference_id": reference_id,
                            "cash_flow_index": index,
                        },
                    )
            self._assert_cash_lots_reconcile(
                connection,
                str(portfolio.id),
                expected_cash=flow["balance_after"],
                as_of=occurred_at,
            )
        for order in result["orders"]:
            ref = order.get("order_ref")
            if ref is None:
                continue
            if str(order.get("status")) in OPEN_STATUSES:
                continue
            if str(order.get("side")) == "buy":
                if self._reservation_total(connection, str(ref)) <= 0:
                    continue
                self._move_order_reservation(
                    connection,
                    portfolio_id=str(portfolio.id),
                    order_id=str(ref),
                    event_key=f"order:{ref}:release:{batch.id}:terminal",
                    event_type="release",
                    amount=None,
                    occurred_at=now,
                    now=now,
                    batch_id=str(batch.id),
                    details={"terminal_status": str(order.get("status"))},
                )
            else:
                if self._security_reservation_total(connection, str(ref)) <= 0:
                    continue
                self._move_sell_reservation(
                    connection,
                    portfolio_id=str(portfolio.id),
                    order_id=str(ref),
                    event_key=(
                        f"order:{ref}:security-release:{batch.id}:terminal"
                    ),
                    event_type="release",
                    quantity=None,
                    occurred_at=now,
                    now=now,
                    batch_id=str(batch.id),
                    details={"terminal_status": str(order.get("status"))},
                )
        cash_view = self._assert_cash_lots_reconcile(
            connection,
            str(portfolio.id),
            expected_cash=result["cash"],
            as_of=now,
        )
        frozen_by_instrument = {
            str(row.instrument): int(row.quantity)
            for row in connection.execute(
                select(
                    simulation_position_reservations.c.instrument,
                    func.sum(
                        simulation_position_reservations.c.remaining_quantity
                    ).label("quantity"),
                )
                .where(
                    simulation_position_reservations.c.portfolio_id == portfolio.id,
                    simulation_position_reservations.c.remaining_quantity > 0,
                )
                .group_by(simulation_position_reservations.c.instrument)
            )
        }
        connection.execute(
            delete(simulation_positions).where(
                simulation_positions.c.portfolio_id == portfolio.id
            )
        )
        for instrument, position in result["positions"].items():
            connection.execute(
                insert(simulation_positions).values(
                    portfolio_id=portfolio.id,
                    instrument=instrument,
                    atomic_group_id=position.get("atomic_group_id"),
                    leg_no=position.get("leg_no"),
                    position_side=position.get("position_side", "long"),
                    borrow_cost=Decimal(str(position.get("borrow_cost", 0.0))),
                    quantity=int(position["quantity"]),
                    available_quantity=int(position["available_quantity"]),
                    frozen_quantity=frozen_by_instrument.get(instrument, 0),
                    average_cost=Decimal(str(position["average_cost"])),
                    last_trade_date=position.get("last_trade_date"),
                    market_price=(
                        Decimal(str(position["market_price"]))
                        if position.get("market_price") is not None
                        else None
                    ),
                    market_date=position.get("market_date"),
                    stale=bool(position.get("stale", True)),
                    market_value=Decimal(str(position.get("market_value", 0.0))),
                    updated_at=now,
                )
            )
        # 批次与股息权利与持仓同生命周期：随结果全量重写（与 positions 同事务）。
        connection.execute(
            delete(simulation_position_lots).where(
                simulation_position_lots.c.portfolio_id == portfolio.id
            )
        )
        connection.execute(
            delete(simulation_dividend_entitlements).where(
                simulation_dividend_entitlements.c.portfolio_id == portfolio.id
            )
        )
        for instrument, position in result["positions"].items():
            for lot in position.get("lots") or []:
                connection.execute(
                    insert(simulation_position_lots).values(
                        id=uuid.uuid4().hex,
                        portfolio_id=portfolio.id,
                        instrument=instrument,
                        lot_key=str(lot["lot_key"]),
                        acquired_at=lot.get("acquired_at"),
                        sellable_from=lot.get("sellable_from") or date.min,
                        quantity=int(lot["quantity"]),
                        cost_basis_total=Decimal(str(lot.get("cost_basis_total", 0.0))),
                        origin=str(lot.get("origin") or "buy"),
                        updated_at=now,
                    )
                )
                for entitlement in lot.get("entitlements") or []:
                    connection.execute(
                        insert(simulation_dividend_entitlements).values(
                            id=uuid.uuid4().hex,
                            portfolio_id=portfolio.id,
                            instrument=instrument,
                            lot_key=str(lot["lot_key"]),
                            record_date=entitlement["record_date"],
                            kind=str(entitlement["kind"]),
                            income_per_share=Decimal(str(entitlement["income_per_share"])),
                            untaxed_quantity=int(entitlement["untaxed_quantity"]),
                            liability_per_share=Decimal(
                                str(entitlement.get("liability_per_share", 0.0))
                            ),
                            updated_at=now,
                        )
                    )
        # 公司行动是追加式历史：除权确认用唯一键幂等插入，到账只做状态重分类。
        for applied in result.get("corporate_actions_applied") or []:
            if applied["kind"] == "ex":
                connection.execute(
                    pg_insert(simulation_dividend_actions)
                    .values(
                        id=uuid.uuid4().hex,
                        portfolio_id=portfolio.id,
                        instrument=str(applied["instrument"]),
                        ex_date=date.fromisoformat(str(applied["ex_date"])),
                        record_date=(
                            date.fromisoformat(str(applied["record_date"]))
                            if applied.get("record_date")
                            else None
                        ),
                        pay_date=(
                            date.fromisoformat(str(applied["pay_date"]))
                            if applied.get("pay_date")
                            else None
                        ),
                        eligible_quantity=int(applied["eligible_quantity"]),
                        cash_per_share=Decimal(str(applied["cash_per_share"])),
                        receivable_amount=Decimal(str(applied["receivable_amount"])),
                        tax_liability_amount=Decimal(
                            str(applied.get("tax_liability", 0.0))
                        ),
                        bonus_share_ratio=float(applied.get("bonus_share_ratio") or 0.0),
                        conversion_ratio=float(applied.get("conversion_ratio") or 0.0),
                        new_shares=int(applied.get("new_shares") or 0),
                        div_listdate=(
                            date.fromisoformat(str(applied["list_date"]))
                            if applied.get("list_date")
                            else None
                        ),
                        status="accrued",
                        tax_rule_version=str(applied["tax_rule_version"]),
                        valuation_uncertain=bool(applied.get("valuation_uncertain")),
                        payload_sha256=str(applied["payload_sha256"]),
                        batch_id=batch.id,
                        created_at=now,
                        updated_at=now,
                    )
                    .on_conflict_do_nothing(
                        constraint="uq_simulation_dividend_actions_key"
                    )
                )
            elif applied["kind"] == "pay":
                connection.execute(
                    update(simulation_dividend_actions)
                    .where(
                        simulation_dividend_actions.c.portfolio_id == portfolio.id,
                        simulation_dividend_actions.c.instrument
                        == str(applied["instrument"]),
                        simulation_dividend_actions.c.ex_date
                        == date.fromisoformat(str(applied["ex_date"])),
                        simulation_dividend_actions.c.status == "accrued",
                    )
                    .values(status="paid", paid_batch_id=batch.id, updated_at=now)
                )
        # 非分红类公司行动台账：事件键唯一幂等插入，重放不产生第二行。
        for applied in result.get("corporate_events_applied") or []:
            connection.execute(
                pg_insert(simulation_corporate_events)
                .values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio.id,
                    event_key=str(applied["event_key"]),
                    event_type=str(applied["event_type"]),
                    instrument=str(applied["instrument"]),
                    effective_date=date.fromisoformat(str(applied["effective_date"])),
                    payload_sha256=str(applied.get("payload_sha256") or ""),
                    details_json=applied.get("details") or {},
                    batch_id=batch.id,
                    created_at=now,
                )
                .on_conflict_do_nothing(
                    constraint="uq_simulation_corporate_events_key"
                )
            )
        nav = result["nav_row"]
        connection.execute(
            insert(simulation_nav).values(
                portfolio_id=portfolio.id,
                trade_date=batch.trade_date,
                cash=Decimal(str(nav["cash"])),
                market_value=Decimal(str(nav["market_value"])),
                corporate_receivables=Decimal(str(nav.get("corporate_receivables", 0.0))),
                corporate_tax_liabilities=Decimal(
                    str(nav.get("corporate_tax_liabilities", 0.0))
                ),
                nav=Decimal(str(nav["nav"])),
                daily_return=float(nav["daily_return"]),
                benchmark_close=(
                    Decimal(str(nav["benchmark_close"]))
                    if nav.get("benchmark_close") is not None
                    else None
                ),
                benchmark_return=(
                    float(nav["benchmark_return"])
                    if nav.get("benchmark_return") is not None
                    else None
                ),
                benchmark_wealth=(
                    float(nav["benchmark_wealth"])
                    if nav.get("benchmark_wealth") is not None
                    else None
                ),
                drawdown=float(nav["drawdown"]),
                external_flow_open=Decimal(str(nav.get("external_flow_open", 0.0))),
                external_flow_close=Decimal(str(nav.get("external_flow_close", 0.0))),
                twr_daily_return=(
                    float(nav["twr_daily_return"])
                    if nav.get("twr_daily_return") is not None
                    else None
                ),
                investment_wealth=(
                    float(nav["investment_wealth"])
                    if nav.get("investment_wealth") is not None
                    else None
                ),
                twr_drawdown=(
                    float(nav["twr_drawdown"])
                    if nav.get("twr_drawdown") is not None
                    else None
                ),
                twr_status=str(nav.get("twr_status") or "unavailable_legacy"),
                market_date=nav["market_date"],
                has_stale_prices=bool(nav["has_stale_prices"]),
                status=nav["status"],
                performance_certified=bool(nav["performance_certified"]),
                nav_scope=(
                    "aggregate_view"
                    if str(portfolio.source_type) == "allocation"
                    else "member_ledger"
                ),
                produced_by=str(batch.created_by),
                created_at=now,
            )
        )
        for event in result["events"]:
            connection.execute(
                insert(simulation_events).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio.id,
                    batch_id=batch.id,
                    trade_date=batch.trade_date,
                    severity=event["severity"],
                    event_type=event["event_type"],
                    instrument=event.get("instrument"),
                    reason=event["reason"],
                    details_json=event.get("details") or {},
                    created_at=now,
                )
            )
        attribution = self._build_day_attribution(
            connection,
            batch=batch,
            portfolio=portfolio,
            result=result,
            fill_order_ids_by_seq=fill_order_ids_by_seq,
        )
        connection.execute(
            insert(simulation_day_attributions).values(
                id=uuid.uuid4().hex,
                portfolio_id=portfolio.id,
                batch_id=batch.id,
                trade_date=batch.trade_date,
                strategy_json=attribution["strategy"],
                industry_json=attribution["industry"],
                asset_json=attribution["asset"],
                cost_json=attribution["cost"],
                execution_json=attribution["execution"],
                coverage_status=attribution["coverage_status"],
                blocker_reasons_json=attribution["blocker_reasons"],
                input_sha256=attribution["input_sha256"],
                created_at=now,
            )
        )
        summary = {
            "engine_version": result["engine_version"],
            "execution_adapter": str(batch.execution_adapter),
            "execution_contract_hash": str(batch.execution_contract_hash),
            "orders": len(result["orders"]),
            "fills": len(result["fills"]),
            "rejections": sum(order["status"] != "filled" for order in result["orders"]),
            "cash": result["cash"],
            "cash_classification": {
                key: float(value)
                for key, value in cash_view.items()
                if isinstance(value, Decimal)
            },
            "nav": result["nav"],
            "benchmark": {
                "instrument": result.get("benchmark"),
                **dict(result.get("benchmark_day") or {}),
            },
            "conservation": result["conservation"],
            "corporate_actions": len(result.get("corporate_actions_applied") or []),
            "corporate_events": len(result.get("corporate_events_applied") or []),
            "day_attribution": {
                "coverage_status": attribution["coverage_status"],
                "blocker_reasons": attribution["blocker_reasons"],
                "input_sha256": attribution["input_sha256"],
            },
        }
        if result.get("shortability_evidence_sha256"):
            summary["shortability_evidence_sha256"] = result[
                "shortability_evidence_sha256"
            ]
        if result.get("strategy_risk_state"):
            summary["strategy_risk_state"] = result["strategy_risk_state"]
        connection.execute(
            update(simulation_portfolios)
            .where(simulation_portfolios.c.id == portfolio.id)
            .values(
                cash=Decimal(str(result["cash"])),
                nav=Decimal(str(result["nav"])),
                high_water_mark=Decimal(str(result["high_water_mark"])),
                investment_wealth=(
                    float(result["investment_wealth"])
                    if result.get("investment_wealth") is not None
                    else None
                ),
                twr_high_water_mark=(
                    float(result["twr_high_water_mark"])
                    if result.get("twr_high_water_mark") is not None
                    else None
                ),
                updated_at=now,
            )
        )
        connection.execute(
            update(simulation_batches)
            .where(simulation_batches.c.id == batch.id)
            .values(
                status="succeeded",
                summary_json=summary,
                started_at=now,
                finished_at=now,
                error=None,
            )
        )

    def review_nav(
        self,
        portfolio_id: str,
        trade_date: date,
        *,
        actor: str,
        evidence_sha256: str,
        note: str,
    ) -> dict[str, Any]:
        """Attach one immutable four-eyes review to a certified simulation NAV row."""

        reviewer = actor.strip()
        evidence = evidence_sha256.strip().lower()
        review_note = note.strip()
        if len(reviewer) < 2:
            raise ValueError("a responsible simulation NAV reviewer is required")
        if not _is_sha256(evidence):
            raise ValueError("simulation NAV review evidence must be a SHA-256 digest")
        if len(review_note) < 10:
            raise ValueError("simulation NAV review note must be meaningful")
        now = _now()
        with self.engine.begin() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            nav_row = connection.execute(
                select(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.trade_date == trade_date,
                )
                .with_for_update()
            ).first()
            if nav_row is None:
                raise KeyError(f"{portfolio_id}:{trade_date.isoformat()}")
            if not nav_row.performance_certified:
                raise ValueError("only performance-certified simulation NAV may be reviewed")
            if nav_row.reviewed_at is not None:
                raise ValueError("simulation NAV review is immutable and already recorded")
            if reviewer == str(nav_row.produced_by):
                raise ValueError("simulation NAV reviewer must differ from its producer")
            reviewed = connection.execute(
                update(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.trade_date == trade_date,
                    simulation_nav.c.reviewed_at.is_(None),
                )
                .values(
                    reviewed_by=reviewer,
                    reviewed_at=now,
                    review_evidence_sha256=evidence,
                    review_note=review_note,
                )
                .returning(simulation_nav)
            ).one()
            batch_id = connection.scalar(
                select(simulation_batches.c.id)
                .where(
                    simulation_batches.c.portfolio_id == portfolio_id,
                    simulation_batches.c.trade_date == trade_date,
                )
                .order_by(simulation_batches.c.created_at.desc())
                .limit(1)
            )
            connection.execute(
                insert(simulation_events).values(
                    id=uuid.uuid4().hex,
                    portfolio_id=portfolio_id,
                    batch_id=batch_id,
                    trade_date=trade_date,
                    severity="info",
                    event_type="simulation_nav_reviewed",
                    instrument=None,
                    reason="certified simulation NAV independently reviewed",
                    details_json={
                        "reviewed_by": reviewer,
                        "review_evidence_sha256": evidence,
                        "nav_scope": str(nav_row.nav_scope),
                        "source_type": str(portfolio.source_type),
                    },
                    created_at=now,
                )
            )
        result = row_dict(reviewed)
        result["review_subject"] = (
            "aggregate_simulation_view"
            if result["nav_scope"] == "aggregate_view"
            else "member_simulation_ledger"
        )
        return result

    def get(self, portfolio_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if row is None:
                raise KeyError(portfolio_id)
            result = self._portfolio_dict(row)
            result["latest_nav"] = self._first_dict(
                connection.execute(
                    select(simulation_nav)
                    .where(simulation_nav.c.portfolio_id == portfolio_id)
                    .order_by(simulation_nav.c.trade_date.desc())
                    .limit(1)
                ).first()
            )
            reviewed_rows = connection.execute(
                select(simulation_nav)
                .where(
                    simulation_nav.c.portfolio_id == portfolio_id,
                    simulation_nav.c.performance_certified.is_(True),
                    simulation_nav.c.reviewed_at.is_not(None),
                )
                .order_by(simulation_nav.c.trade_date.desc())
                .limit(5)
            ).all()
            review_evidence = [
                {
                    "trade_date": item.trade_date.isoformat(),
                    "reviewed_by": str(item.reviewed_by),
                    "reviewed_at": item.reviewed_at.isoformat(),
                    "review_evidence_sha256": str(item.review_evidence_sha256),
                }
                for item in reversed(reviewed_rows)
            ]
            nav_scope = (
                str(reviewed_rows[0].nav_scope)
                if reviewed_rows
                else (
                    "aggregate_view"
                    if str(row.source_type) == "allocation"
                    else "member_ledger"
                )
            )
            result["review_readiness"] = {
                "nav_scope": nav_scope,
                "view_semantics": (
                    "aggregate view derived from member simulation NAV"
                    if nav_scope == "aggregate_view"
                    else "persistent member simulation ledger"
                ),
                "required_reviewed_days": 5,
                "reviewed_days": len(reviewed_rows),
                "ready": len(reviewed_rows) == 5,
                "evidence_sha256": (
                    _canonical_hash({"reviews": review_evidence})
                    if review_evidence
                    else None
                ),
                "reviews": review_evidence,
            }
            current_cash = self._assert_cash_lots_reconcile(
                connection,
                portfolio_id,
                expected_cash=row.cash,
                as_of=_now(),
            )
            result["cash_view"] = {
                key: float(value) if isinstance(value, Decimal) else value
                for key, value in current_cash.items()
            }
        return result

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(simulation_portfolios)
                .order_by(simulation_portfolios.c.updated_at.desc())
                .limit(limit)
            )
            return [self._portfolio_dict(row) for row in rows]

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(simulation_batches).where(simulation_batches.c.id == batch_id)
            ).first()
            if row is None:
                raise KeyError(batch_id)
            return self._batch_dict(row)

    def latest_batch(self, portfolio_id: str) -> dict[str, Any] | None:
        """Most recent batch by trade_date (reconciliation status source)."""

        with self.engine.connect() as connection:
            row = connection.execute(
                select(simulation_batches)
                .where(simulation_batches.c.portfolio_id == portfolio_id)
                .order_by(
                    simulation_batches.c.trade_date.desc(),
                    simulation_batches.c.created_at.desc(),
                )
                .limit(1)
            ).first()
        return self._batch_dict(row) if row is not None else None

    def latest_paper_previous_snapshot(
        self,
        portfolio_id: str,
        *,
        promotion_stage_id: str,
        before_signal_date: date,
    ) -> dict[str, Any] | None:
        """Read the last successful policy state from the active paper lane.

        The stage and portfolio are checked in the same read transaction as
        the batch selection.  A latest successful row with a malformed or
        missing state is an integrity error; it never silently falls back to
        an older batch and changes cadence after a restart.
        """

        with self.engine.connect() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id,
                    simulation_portfolios.c.status == "active",
                    simulation_portfolios.c.source_type == "strategy_version",
                    simulation_portfolios.c.execution_adapter == "long_only",
                    simulation_portfolios.c.promotion_stage_id == promotion_stage_id,
                )
            ).first()
            if portfolio is None:
                raise ValueError(
                    "paper policy state requires the active stage-owned account"
                )
            stage = connection.execute(
                select(strategy_promotion_stages.c.id).where(
                    strategy_promotion_stages.c.id == promotion_stage_id,
                    strategy_promotion_stages.c.strategy_version_id
                    == portfolio.source_id,
                    strategy_promotion_stages.c.simulation_portfolio_id
                    == portfolio.id,
                    strategy_promotion_stages.c.status == "active",
                )
            ).first()
            if stage is None:
                raise ValueError(
                    "paper policy state is not bound to the active promotion stage"
                )
            batch = connection.execute(
                select(simulation_batches)
                .where(
                    simulation_batches.c.portfolio_id == portfolio.id,
                    simulation_batches.c.status == "succeeded",
                    simulation_batches.c.signal_date < before_signal_date,
                )
                .order_by(
                    simulation_batches.c.signal_date.desc(),
                    simulation_batches.c.created_at.desc(),
                )
                .limit(1)
            ).first()
        if batch is None:
            return None
        return previous_snapshot_from_paper_batch(
            self._batch_dict(batch),
            expected_portfolio_id=str(portfolio.id),
            expected_promotion_stage_id=promotion_stage_id,
        )

    def current_autopilot_paper_target(self) -> dict[str, Any]:
        """Return the one read-only daily target projection for Autopilot.

        This is intentionally *not* a recommendation endpoint.  It only
        projects the immutable Qlib order-plan already attached to the active
        isolated Autopilot paper account.  In particular, a manual simulation,
        a historical frozen account, an allocation account, or a human-enabled
        recommendation portfolio can never be selected here.
        """

        with self.engine.connect() as connection:
            candidates = connection.execute(
                select(
                    simulation_portfolios,
                    strategy_versions.c.config_json.label("_strategy_config"),
                    strategy_promotion_stages.c.status.label("_paper_stage_status"),
                    strategy_promotion_stages.c.opened_at.label("_paper_stage_opened_at"),
                )
                .join(
                    strategy_versions,
                    strategy_versions.c.id == simulation_portfolios.c.source_id,
                )
                .join(
                    strategy_promotion_stages,
                    strategy_promotion_stages.c.id
                    == simulation_portfolios.c.promotion_stage_id,
                )
                .where(
                    simulation_portfolios.c.status == "active",
                    simulation_portfolios.c.source_type == "strategy_version",
                    simulation_portfolios.c.execution_adapter == "long_only",
                )
                .order_by(simulation_portfolios.c.updated_at.desc())
            ).all()
            selected = None
            for candidate in candidates:
                metadata = candidate._mapping
                config = dict(metadata["_strategy_config"] or {})
                if (
                    str(config.get("autopilot_completion_contract_version") or "")
                    .startswith("autopilot-completion-")
                    and str(metadata["_paper_stage_status"]) == "active"
                    and config.get("recommendation_enabled") is False
                    and config.get("broker_connection_enabled") is False
                    and config.get("real_trading_eligible") is False
                ):
                    selected = candidate
                    break
            if selected is None:
                return {
                    "contract_version": PAPER_TARGET_PROJECTION_VERSION,
                    "status": "waiting_for_paper_account",
                    "message": "等待通过最终 OOS 的自动驾驶策略创建隔离多头模拟账户",
                    "mode": "paper_only",
                    "recommendation_enabled": False,
                    "real_trading_eligible": False,
                    "simulation_portfolio": None,
                    "signal_date": None,
                    "trade_date": None,
                    "targets": [],
                }
            batches = connection.execute(
                select(simulation_batches)
                .where(simulation_batches.c.portfolio_id == selected.id)
                .order_by(
                    simulation_batches.c.trade_date.desc(),
                    simulation_batches.c.created_at.desc(),
                )
            ).all()

        portfolio = selected
        account = self._portfolio_dict(portfolio)
        if not batches:
            return {
                "contract_version": PAPER_TARGET_PROJECTION_VERSION,
                "status": "waiting_for_order_plan",
                "message": "模拟账户已创建，等待首份冻结 Qlib 订单计划",
                "mode": "paper_only",
                "recommendation_enabled": False,
                "real_trading_eligible": False,
                "simulation_portfolio": {
                    "id": account["id"],
                    "name": account["name"],
                    "status": account["status"],
                    "strategy_version_id": str(portfolio.source_id),
                },
                "signal_date": None,
                "trade_date": None,
                "targets": [],
            }
        return self._paper_target_projection_from_batches(
            portfolio=portfolio,
            account=account,
            paper_stage_opened_at=selected._mapping["_paper_stage_opened_at"],
            batches=batches,
        )

    def paper_target_for_strategy_version(self, version_id: str) -> dict[str, Any]:
        """Return the latest governed target for one horizon's paper account.

        Unlike ``current_autopilot_paper_target`` this selector is explicit and
        therefore supports three simultaneously validating horizon accounts.
        It remains a read-only simulation projection and never confers
        recommendation permission.
        """

        with self.engine.connect() as connection:
            selected = connection.execute(
                select(
                    simulation_portfolios,
                    strategy_promotion_stages.c.status.label("_paper_stage_status"),
                    strategy_promotion_stages.c.opened_at.label("_paper_stage_opened_at"),
                )
                .join(
                    strategy_promotion_stages,
                    strategy_promotion_stages.c.id
                    == simulation_portfolios.c.promotion_stage_id,
                )
                .where(
                    simulation_portfolios.c.status == "active",
                    simulation_portfolios.c.source_type == "strategy_version",
                    simulation_portfolios.c.source_id == version_id,
                    simulation_portfolios.c.execution_adapter == "long_only",
                    strategy_promotion_stages.c.status == "active",
                )
                .order_by(simulation_portfolios.c.updated_at.desc())
                .limit(1)
            ).first()
            if selected is None:
                return {
                    "contract_version": PAPER_TARGET_PROJECTION_VERSION,
                    "status": "waiting_for_paper_account",
                    "mode": "paper_only",
                    "recommendation_enabled": False,
                    "real_trading_eligible": False,
                    "simulation_portfolio": None,
                    "signal_date": None,
                    "trade_date": None,
                    "targets": [],
                }
            batches = connection.execute(
                select(simulation_batches)
                .where(simulation_batches.c.portfolio_id == selected.id)
                .order_by(
                    simulation_batches.c.trade_date.desc(),
                    simulation_batches.c.created_at.desc(),
                )
            ).all()

        account = self._portfolio_dict(selected)
        if not batches:
            return {
                "contract_version": PAPER_TARGET_PROJECTION_VERSION,
                "status": "waiting_for_order_plan",
                "mode": "paper_only",
                "recommendation_enabled": False,
                "real_trading_eligible": False,
                "simulation_portfolio": {
                    "id": account["id"],
                    "name": account["name"],
                    "status": account["status"],
                    "strategy_version_id": version_id,
                },
                "signal_date": None,
                "trade_date": None,
                "targets": [],
            }
        return self._paper_target_projection_from_batches(
            portfolio=selected,
            account=account,
            paper_stage_opened_at=selected._mapping["_paper_stage_opened_at"],
            batches=batches,
        )

    @classmethod
    def _paper_target_projection_from_batches(
        cls,
        *,
        portfolio: Any,
        account: dict[str, Any],
        paper_stage_opened_at: datetime,
        batches: list[Any],
    ) -> dict[str, Any]:
        """Build a factual target-weight projection from verified batch rows."""

        batch = batches[0]
        payload = dict(batch.target_payload_json or {})
        try:
            cls._require_governed_long_only_target(
                batch=batch,
                portfolio=portfolio,
                target_payload=payload,
            )
            plan = dict(payload["governed_order_plan"])
            if (
                str(plan.get("promotion_stage_id") or "")
                != str(portfolio.promotion_stage_id or "")
                or str(plan.get("promotion_stage_opened_at") or "")
                != paper_stage_opened_at.isoformat()
            ):
                raise ValueError("Qlib order-plan is not bound to the active paper stage")
            current_weights = cls._normalize_target_payload(
                {"target_weights": payload.get("target_weights")}, adapter="long_only"
            )["target_weights"]
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "contract_version": PAPER_TARGET_PROJECTION_VERSION,
                "status": "blocked_invalid_order_plan",
                "message": "最新模拟目标未通过不可变订单计划校验，已停止展示候选",
                "blocker": str(exc),
                "mode": "paper_only",
                "recommendation_enabled": False,
                "real_trading_eligible": False,
                "simulation_portfolio": {
                    "id": account["id"],
                    "name": account["name"],
                    "status": account["status"],
                    "strategy_version_id": str(portfolio.source_id),
                },
                "signal_date": batch.signal_date.isoformat(),
                "trade_date": batch.trade_date.isoformat(),
                "targets": [],
            }

        previous_weights: dict[str, float] | None = None
        previous_batch_id: str | None = None
        if len(batches) > 1:
            previous = batches[1]
            previous_payload = dict(previous.target_payload_json or {})
            try:
                cls._require_governed_long_only_target(
                    batch=previous,
                    portfolio=portfolio,
                    target_payload=previous_payload,
                )
                previous_weights = cls._normalize_target_payload(
                    {"target_weights": previous_payload.get("target_weights")},
                    adapter="long_only",
                )["target_weights"]
                previous_batch_id = str(previous.id)
            except (KeyError, TypeError, ValueError):
                # A corrupted predecessor must never change today's valid
                # target.  We simply make the delta baseline unavailable.
                previous_weights = None

        targets = paper_target_adjustments(current_weights, previous_weights)
        return {
            "contract_version": PAPER_TARGET_PROJECTION_VERSION,
            "status": "ready",
            "message": "仅展示已冻结 Qlib 订单计划的模拟目标，不构成正式荐股或实盘指令",
            "mode": "paper_only",
            "recommendation_enabled": False,
            "real_trading_eligible": False,
            "simulation_portfolio": {
                "id": account["id"],
                "name": account["name"],
                "status": account["status"],
                "strategy_version_id": str(portfolio.source_id),
                "daily_dataset": str(portfolio.daily_dataset),
                "daily_dataset_lineage_id": str(
                    portfolio.daily_dataset_lineage_id
                ),
            },
            "batch": {
                "id": str(batch.id),
                "status": str(batch.status),
                "order_plan_manifest_sha256": str(plan["manifest_sha256"]),
                "target_weights_sha256": str(plan["target_weights_sha256"]),
                "previous_batch_id": previous_batch_id,
            },
            "signal_date": batch.signal_date.isoformat(),
            "trade_date": batch.trade_date.isoformat(),
            "cash_weight": max(0.0, 1.0 - sum(current_weights.values())),
            "targets": targets,
        }

    def latest_nav(self, portfolio_id: str) -> dict[str, Any] | None:
        """Most recent NAV row by trade_date (health/certification source)."""

        with self.engine.connect() as connection:
            row = connection.execute(
                select(simulation_nav)
                .where(simulation_nav.c.portfolio_id == portfolio_id)
                .order_by(simulation_nav.c.trade_date.desc())
                .limit(1)
            ).first()
        return row_dict(row) if row is not None else None

    def performance_summary(self, portfolio_id: str) -> dict[str, Any]:
        """Unitized TWR curve metrics plus the money-weighted XIRR companion.

        The CNY NAV series stays the ledger reconciliation view; this report
        is built on the unitized ``investment_wealth`` curve (design 4.4/8.3):
        TWR, max drawdown and recovery time come from that curve, external
        cash flows never manufacture return or drawdown (design 12.1). XIRR
        uses the investor-perspective flows (initial deposit and recorded
        external flows out of pocket, latest NAV as the terminal value) and
        reports an explicit status on degenerate inputs.
        """

        with self.engine.connect() as connection:
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == portfolio_id
                )
            ).first()
            if portfolio is None:
                raise KeyError(portfolio_id)
            nav_rows = [
                row_dict(row)
                for row in connection.execute(
                    select(simulation_nav)
                    .where(simulation_nav.c.portfolio_id == portfolio_id)
                    .order_by(simulation_nav.c.trade_date)
                )
            ]
            flow_rows = [
                row_dict(row)
                for row in connection.execute(
                    select(simulation_external_flows)
                    .where(simulation_external_flows.c.portfolio_id == portfolio_id)
                    .order_by(simulation_external_flows.c.trade_date)
                )
            ]
            first_batch_signal_date = connection.scalar(
                select(simulation_batches.c.signal_date)
                .where(
                    simulation_batches.c.portfolio_id == portfolio_id,
                    simulation_batches.c.status == "succeeded",
                )
                .order_by(
                    simulation_batches.c.trade_date,
                    simulation_batches.c.signal_date,
                )
                .limit(1)
            )
        chain_broken = any(row["investment_wealth"] is None for row in nav_rows)
        statistics_chain_broken = any(
            row["investment_wealth"] is None or row["twr_daily_return"] is None
            for row in nav_rows
        )
        points = [
            (row["trade_date"], float(row["investment_wealth"]))
            for row in nav_rows
            if row["investment_wealth"] is not None
        ]
        unitized = unitized_drawdown_recovery(points)
        statistics_points = [
            (
                row["trade_date"],
                float(row["investment_wealth"]),
                float(row["twr_daily_return"]),
            )
            for row in nav_rows
            if row["investment_wealth"] is not None
            and row["twr_daily_return"] is not None
        ]
        statistics = unitized_return_statistics(
            statistics_points,
            inception_date=first_batch_signal_date,
        )
        if chain_broken:
            unitized = {
                **unitized,
                "status": "unavailable_broken_chain",
                "broken_from": next(
                    row["trade_date"].isoformat()
                    for row in nav_rows
                    if row["investment_wealth"] is None
                ),
            }
        if statistics_chain_broken:
            statistics = {
                **statistics,
                "status": "unavailable_broken_chain",
                "broken_from": next(
                    row["trade_date"].isoformat()
                    for row in nav_rows
                    if row["investment_wealth"] is None
                    or row["twr_daily_return"] is None
                ),
            }
        benchmark = str(portfolio.benchmark or "").strip().upper()
        if not benchmark:
            relative_performance = {
                "status": "benchmark_not_configured",
                "benchmark": None,
                "information_ratio": None,
                "tracking_error": None,
                "annualized_excess_return": None,
            }
        else:
            benchmark_chain_broken = any(
                row["twr_daily_return"] is None
                or row["benchmark_return"] is None
                or row["benchmark_wealth"] is None
                for row in nav_rows
            )
            relative_points = [
                (
                    row["trade_date"],
                    float(row["twr_daily_return"]),
                    float(row["benchmark_return"]),
                    float(row["benchmark_wealth"]),
                )
                for row in nav_rows
                if row["twr_daily_return"] is not None
                and row["benchmark_return"] is not None
                and row["benchmark_wealth"] is not None
            ]
            relative_performance = {
                **benchmark_relative_statistics(relative_points),
                "benchmark": benchmark,
            }
            if benchmark_chain_broken:
                relative_performance = {
                    **relative_performance,
                    "status": "unavailable_broken_benchmark_chain",
                    "broken_from": next(
                        row["trade_date"].isoformat()
                        for row in nav_rows
                        if row["twr_daily_return"] is None
                        or row["benchmark_return"] is None
                        or row["benchmark_wealth"] is None
                    ),
                }
        money_weighted: dict[str, Any]
        xirr_flow_count = 0
        xirr_inception_date: date | None = None
        if nav_rows:
            terminal_date = nav_rows[-1]["trade_date"]
            realized_flows = [
                row for row in flow_rows if row["trade_date"] <= terminal_date
            ]
            # Wall-clock creation is not an economic inception date for a
            # deterministic historical replay. Anchor initial capital no later
            # than the first booked NAV/flow; forward simulations retain their
            # actual creation date.
            economic_dates = [
                portfolio.created_at.date(),
                *([first_batch_signal_date] if first_batch_signal_date else []),
                nav_rows[0]["trade_date"],
                *[row["trade_date"] for row in realized_flows],
            ]
            xirr_inception_date = min(economic_dates)
            money_flows = [
                (xirr_inception_date, -float(portfolio.initial_cash)),
                *[
                    (row["trade_date"], -float(row["amount"]))
                    for row in realized_flows
                ],
            ]
            xirr_flow_count = len(realized_flows)
            money_weighted = xirr(
                money_flows,
                terminal=(terminal_date, float(nav_rows[-1]["nav"])),
            )
        else:
            money_weighted = {"status": "insufficient_evidence", "rate": None}
        return {
            "portfolio_id": str(portfolio_id),
            "contract_version": UNITIZED_PERFORMANCE_VERSION,
            "nav_days": len(nav_rows),
            "external_flow_count": len(flow_rows),
            "xirr_external_flow_count": xirr_flow_count,
            "xirr_inception_date": (
                xirr_inception_date.isoformat() if xirr_inception_date else None
            ),
            "unitized": unitized,
            "statistics": statistics,
            "relative_performance": relative_performance,
            "xirr": money_weighted,
            "cny_nav_latest": (float(nav_rows[-1]["nav"]) if nav_rows else None),
        }

    def execution_manifest(self, batch_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            batch = connection.execute(
                select(simulation_batches).where(simulation_batches.c.id == batch_id)
            ).first()
            if batch is None:
                raise KeyError(batch_id)
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == batch.portfolio_id
                )
            ).one()
            payload = dict(batch.target_payload_json or {})
            governed_pair_plan = payload.get("governed_pair_plan")
            governed_order_plan = payload.get("governed_order_plan")
            snapshot = None
            if batch.recommendation_snapshot_id:
                snapshot = connection.execute(
                    select(recommendation_snapshots).where(
                        recommendation_snapshots.c.id == batch.recommendation_snapshot_id
                    )
                ).one()
                target_instruments = set(
                    str(value)
                    for value in connection.scalars(
                        select(recommendation_holdings.c.instrument).where(
                            recommendation_holdings.c.snapshot_id == snapshot.id
                        )
                    )
                )
            else:
                target_instruments = set(payload.get("target_weights") or {})
                target_instruments.update(
                    str(item.get("instrument"))
                    for item in (payload.get("legs") or [])
                    if item.get("instrument")
                )
            if batch.recommendation_snapshot_id:
                governed_pair_plan = None
            held_instruments = set(
                str(value)
                for value in connection.scalars(
                    select(simulation_positions.c.instrument).where(
                        simulation_positions.c.portfolio_id == portfolio.id
                    )
                )
            )
        settlement_calendar_binding = (
            governed_order_plan.get("settlement_calendar_binding")
            if isinstance(governed_order_plan, dict)
            else None
        )
        if str(portfolio.execution_frequency) == "day":
            settlement_calendar_binding = validate_settlement_calendar_binding(
                settlement_calendar_binding,
                trade_date=batch.trade_date,
                dataset_identity_sha256=str(
                    batch.execution_dataset_identity_sha256
                ),
                dataset_lineage_id=str(batch.execution_dataset_lineage_id),
            )
        return {
            "batch_id": str(batch.id),
            "portfolio_id": str(portfolio.id),
            "source_type": str(portfolio.source_type),
            "source_id": str(portfolio.source_id),
            "benchmark": str(portfolio.benchmark or ""),
            "source_snapshot_id": str(batch.source_snapshot_id),
            "recommendation_portfolio_id": (
                str(portfolio.recommendation_portfolio_id)
                if portfolio.recommendation_portfolio_id
                else None
            ),
            "recommendation_snapshot_id": str(snapshot.id) if snapshot else None,
            "signal_date": batch.signal_date.isoformat(),
            "trade_date": batch.trade_date.isoformat(),
            "signal_at": (
                batch.signal_at.isoformat() if batch.signal_at is not None else None
            ),
            "execution_not_before": (
                batch.execution_not_before.isoformat()
                if batch.execution_not_before is not None
                else None
            ),
            "daily_dataset": str(batch.daily_dataset),
            "execution_dataset": str(batch.execution_dataset),
            "execution_algorithm": str(portfolio.execution_algorithm),
            "execution_policy": {
                **dict(portfolio.execution_policy_json or {}),
                "simulation_semantics_sha256": str(
                    batch.simulation_semantics_sha256
                ),
            },
            "execution_adapter": str(portfolio.execution_adapter),
            "execution_frequency": str(portfolio.execution_frequency),
            "execution_contract_hash": str(portfolio.execution_contract_hash),
            "instruments": sorted(target_instruments | held_instruments),
            "governed_pair_plan": governed_pair_plan,
            "settlement_calendar_binding": settlement_calendar_binding,
        }

    def positions_with_holding_age(
        self,
        portfolio_id: str,
        *,
        calendar_days: set[date],
        as_of_date: date,
        require_complete_age: bool = False,
    ) -> list[dict[str, Any]]:
        """Return current positions with ledger-proven Qlib-session ages.

        Position and lot rows are read through one store operation; no age is
        written back, so retries remain deterministic and cannot double-count
        a session.  ``require_complete_age`` is used by bounded-holding and
        thesis-exit policies to fail closed for legacy or incomplete lot data.
        """

        with self.engine.connect() as connection:
            positions = [
                row_dict(row)
                for row in connection.execute(
                    select(simulation_positions)
                    .where(simulation_positions.c.portfolio_id == portfolio_id)
                    .order_by(simulation_positions.c.instrument)
                )
            ]
            lots = [
                row_dict(row)
                for row in connection.execute(
                    select(simulation_position_lots)
                    .where(simulation_position_lots.c.portfolio_id == portfolio_id)
                    .order_by(
                        simulation_position_lots.c.instrument,
                        simulation_position_lots.c.lot_key,
                    )
                )
            ]
        projected = _annotate_position_holding_ages(
            positions,
            lots,
            calendar_days=calendar_days,
            as_of_date=as_of_date,
            require_complete_age=require_complete_age,
        )
        for row in projected:
            row["free_sellable_quantity"] = int(
                row["available_quantity"]
            ) - int(row["frozen_quantity"])
        return projected

    def rows(
        self,
        portfolio_id: str,
        resource: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        resources = {
            "batches": (
                simulation_batches,
                simulation_batches,
                simulation_batches.c.created_at,
            ),
            "orders": (
                simulation_orders.join(
                    simulation_batches,
                    simulation_orders.c.batch_id == simulation_batches.c.id,
                ),
                simulation_orders,
                simulation_orders.c.created_at,
            ),
            "fills": (
                simulation_fills.join(
                    simulation_batches,
                    simulation_fills.c.batch_id == simulation_batches.c.id,
                ),
                simulation_fills,
                simulation_fills.c.executed_at,
            ),
            "positions": (
                simulation_positions,
                simulation_positions,
                simulation_positions.c.instrument,
            ),
            "nav": (simulation_nav, simulation_nav, simulation_nav.c.trade_date),
            "events": (simulation_events, simulation_events, simulation_events.c.created_at),
            "external_flows": (
                simulation_external_flows,
                simulation_external_flows,
                simulation_external_flows.c.trade_date,
            ),
            "cash_flows": (
                simulation_cash_flows,
                simulation_cash_flows,
                simulation_cash_flows.c.created_at,
            ),
            "corporate_events": (
                simulation_corporate_events,
                simulation_corporate_events,
                simulation_corporate_events.c.created_at,
            ),
            "dividend_actions": (
                simulation_dividend_actions,
                simulation_dividend_actions,
                simulation_dividend_actions.c.created_at,
            ),
            "dividend_entitlements": (
                simulation_dividend_entitlements,
                simulation_dividend_entitlements,
                simulation_dividend_entitlements.c.updated_at,
            ),
            "fee_adjustments": (
                simulation_fee_adjustments,
                simulation_fee_adjustments,
                simulation_fee_adjustments.c.created_at,
            ),
            "cash_lots": (
                simulation_cash_lots,
                simulation_cash_lots,
                simulation_cash_lots.c.created_at,
            ),
            "cash_events": (
                simulation_cash_events,
                simulation_cash_events,
                simulation_cash_events.c.occurred_at,
            ),
            "cash_reservations": (
                simulation_cash_reservations,
                simulation_cash_reservations,
                simulation_cash_reservations.c.created_at,
            ),
            "position_reservations": (
                simulation_position_reservations,
                simulation_position_reservations,
                simulation_position_reservations.c.created_at,
            ),
            "security_events": (
                simulation_security_events,
                simulation_security_events,
                simulation_security_events.c.occurred_at,
            ),
            "day_attributions": (
                simulation_day_attributions,
                simulation_day_attributions,
                simulation_day_attributions.c.trade_date,
            ),
        }
        if resource not in resources:
            raise ValueError("unknown simulation resource")
        source, table, ordering = resources[resource]
        portfolio_column = (
            simulation_batches.c.portfolio_id
            if resource in {"orders", "fills"}
            else table.c.portfolio_id
        )
        statement = (
            select(table)
            .select_from(source)
            .where(portfolio_column == portfolio_id)
        )
        if limit is None:
            statement = statement.order_by(ordering)
        else:
            statement = statement.order_by(ordering.desc()).limit(limit)
        with self.engine.connect() as connection:
            rows = [
                row_dict(row)
                for row in connection.execute(statement)
            ]
        if limit is not None:
            rows.reverse()
        if resource == "positions":
            for row in rows:
                row["free_sellable_quantity"] = int(
                    row["available_quantity"]
                ) - int(row["frozen_quantity"])
        return rows

    @staticmethod
    def _portfolio_dict(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["execution_policy"] = result.pop("execution_policy_json")
        is_pair = result["execution_adapter"] == "pair"
        result["simulation_mode"] = "shadow_pair" if is_pair else "paper"
        result["synthetic_short_exposure"] = is_pair
        result["financing_enabled"] = False
        result["real_trading_eligible"] = False
        result["provenance"] = {
            "daily": {
                "dataset": result["daily_dataset"],
                "dataset_identity_sha256": result["daily_dataset_identity_sha256"],
                "dataset_lineage_id": result["daily_dataset_lineage_id"],
                "field_contract_version": result["daily_field_contract_version"],
            },
            "minute": {
                "dataset": result["execution_dataset"],
                "dataset_identity_sha256": result[
                    "execution_dataset_identity_sha256"
                ],
                "dataset_lineage_id": result["execution_dataset_lineage_id"],
                "field_contract_version": result["execution_field_contract_version"],
            },
            "execution_engine_version": result["execution_engine_version"],
            "cost_schedule_version": result["cost_schedule_version"],
        }
        return result

    @staticmethod
    def _batch_dict(row: Any) -> dict[str, Any]:
        result = row_dict(row)
        result["summary"] = result.pop("summary_json")
        result["target_payload"] = result.pop("target_payload_json")
        return result

    @staticmethod
    def _first_dict(row: Any | None) -> dict[str, Any] | None:
        return row_dict(row) if row is not None else None
