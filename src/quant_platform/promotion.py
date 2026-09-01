"""Strategy promotion chain: paper stage and forward evidence gate.

Design 4.5/6.11/7.4/9.5: the lifecycle is
``research -> candidate -> paper -> recommendation_enabled``.

- ``candidate -> paper`` happens automatically when the formal hard gate
  (``StrategyStore.approve``) passes: an isolated forward paper stage is
  opened with its own simulation account, capital and evidence counters —
  historical backtest evidence never substitutes forward evidence.
- ``paper -> recommendation_enabled`` requires the version's pre-registered
  forward gate (minimum natural time, independent decision batches, completed
  holding/rebalance cycles, data completeness, ledger reconciliation rate and
  cost deviation). The scheduler invokes the same atomic promotion transaction
  as ``system:auto-promotion``; no human confirmation can replace or weaken the
  gate. Insufficient evidence keeps the version in paper and is reported as
  ``insufficient_evidence``.
- A substantive source-contract drift (design 9.5, detected through the
  simulation source-contract guard) freezes the old stage read-only and opens
  a new one whose evidence starts from zero; stages are never concatenated.

Evidence is derived from the simulation ledger of the stage's own paper
account: succeeded batches are independent decisions, batch conservation
results feed the reconciliation rate, fills feed the realized cost rate, and
certified NAV dates measure trading time. Legacy versions retain their old
sell-batch cycle proxy. Explicit horizons count a closed round trip only when
governed long fills take one instrument from a positive position back to zero;
partial exits, oversells and sells without a prior buy never count.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, insert, select, text, update

from quant_data.database import (
    account_netting_plans,
    backtest_runs,
    open_database,
    recommendation_portfolios,
    recommendation_snapshots,
    row_dict,
    simulation_batches,
    simulation_fills,
    simulation_nav,
    simulation_orders,
    simulation_portfolios,
    strategy_allocation_events,
    strategy_allocation_members,
    strategy_allocations,
    strategy_events,
    strategy_forward_gates,
    strategy_promotion_stages,
    strategy_versions,
)
from quant_platform.cost_model import (
    CN_COST_SCHEDULE_VERSIONS,
    COST_SCHEDULE_VERSION,
    CostModelConfig,
)
from quant_platform.forward_only_rehabilitation import (
    EVIDENCE_MODE_REPLAY,
    require_qualification,
)
from quant_platform.forward_only_rehabilitation import (
    canonical_sha256 as rehabilitation_canonical_sha256,
)
from quant_platform.investor_profile import (
    InvestorSimulationProfileStore,
    bind_investor_profile,
    validate_investor_profile_binding,
)
from quant_platform.research_horizon import (
    LEGACY_AMBIGUOUS,
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
)
from quant_platform.simulation_store import (
    QLIB_ORDER_PLAN_FORMAT_VERSION,
    SimulationStore,
)
from quant_platform.strategy_health_authority import load_production_health_gate

PROMOTION_CONTRACT_VERSION = "promotion-chain-v1"
FORWARD_GATE_CRITERIA_VERSION = "strategy-forward-gate-v2"
HORIZON_REVIEW_CONTRACT_VERSION = "strategy-horizon-review-v3"
LEGACY_HORIZON_REVIEW_CONTRACT_VERSION = "strategy-horizon-review-v2"

STAGE_PAPER = "paper"
STAGE_RECOMMENDATION_ENABLED = "recommendation_enabled"

ACTIVATION_PENDING_EVENT = "strategy.activation_cutover_pending"
ACTIVATION_COMPLETED_EVENT = "strategy.activation_cutover_completed"
ACTIVATION_ROLLED_BACK_EVENT = "strategy.activation_cutover_rolled_back"

_STAGE_ACTIVE = "active"
_STAGE_AWAITING = "awaiting_simulation"
_STAGE_FROZEN = "frozen"

_REFERENCE_ORDER_VALUE = 100_000.0
_DEFAULT_PAPER_INITIAL_CASH = 100_000.0
MAX_FORWARD_CHALLENGERS_PER_HORIZON = 2
_RECONCILIATION_TOLERANCE = 1e-6
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_horizon_challenger_capacity(
    connection: Any,
    *,
    horizon_profile: str,
    version_id: str,
) -> dict[str, Any]:
    """Keep one production incumbent and at most two forward challengers.

    The production incumbent is already protected by the unique active-horizon
    database index.  This gate bounds the expensive, isolated paper side of
    the same lifecycle.  It runs under a horizon advisory lock both before
    approval and while reconciling the paper stage, so concurrent research
    completions cannot each observe the last free slot.
    """

    profile = str(horizon_profile or "").strip()
    candidate_id = str(version_id or "").strip()
    if not candidate_id:
        raise ValueError("forward challenger version id is required")
    if profile == LEGACY_AMBIGUOUS:
        return {
            "horizon_profile": profile,
            "challenger_count": 0,
            "challenger_limit": MAX_FORWARD_CHALLENGERS_PER_HORIZON,
            "legacy_unbounded": True,
        }
    if profile not in {SHORT_1_5D, SWING_1_6M, LONG_1_3Y}:
        raise ValueError("forward challenger uses an unsupported horizon")
    connection.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:scope))"),
        {"scope": f"forward-challenger-capacity:{profile}"},
    )
    # A StrategyVersion is immutable history.  Its ``promotion_stage=paper``
    # marker deliberately remains after a paper stage is frozen, so counting
    # version rows would permanently consume capacity.  The stage lifecycle is
    # the authority for whether a challenger is actually occupying a forward
    # slot; recommendation-enabled and retired versions are excluded as well.
    challengers = {
        str(value)
        for value in connection.scalars(
            select(strategy_promotion_stages.c.strategy_version_id)
            .join(
                strategy_versions,
                strategy_versions.c.id
                == strategy_promotion_stages.c.strategy_version_id,
            )
            .where(
                strategy_versions.c.horizon_profile == profile,
                strategy_versions.c.status == "approved",
                strategy_versions.c.promotion_stage == STAGE_PAPER,
                strategy_promotion_stages.c.status.in_(
                    [_STAGE_ACTIVE, _STAGE_AWAITING]
                ),
            )
            .distinct()
        )
    }
    projected_count = len(challengers | {candidate_id})
    if projected_count > MAX_FORWARD_CHALLENGERS_PER_HORIZON:
        raise ValueError(
            f"{profile} already has {len(challengers)} forward challengers; "
            f"the governed limit is {MAX_FORWARD_CHALLENGERS_PER_HORIZON}"
        )
    return {
        "horizon_profile": profile,
        "challenger_count": projected_count,
        "challenger_limit": MAX_FORWARD_CHALLENGERS_PER_HORIZON,
        "challenger_version_ids": sorted(challengers | {candidate_id}),
        "legacy_unbounded": False,
    }


def _is_autopilot_config(value: Any) -> bool:
    return isinstance(value, dict) and str(
        value.get("autopilot_completion_contract_version") or ""
    ).startswith("autopilot-completion-")


def _autopilot_paper_accounts_compete(
    current_horizon: str | None, prior_horizon: str | None
) -> bool:
    """Return whether two Autopilot paper accounts occupy the same evidence lane.

    Explicit short, swing and long horizons may each keep two governed shadow
    challengers active in parallel.  Capacity is enforced before their stages
    open, so those accounts do not supersede one another.  Legacy strategies
    retain the historical single-primary behaviour because their ambiguous
    holding period cannot be assigned to an independent lane safely.
    """

    current = str(current_horizon or LEGACY_AMBIGUOUS)
    prior = str(prior_horizon or LEGACY_AMBIGUOUS)
    if current == LEGACY_AMBIGUOUS or prior == LEGACY_AMBIGUOUS:
        return True
    return False


def resolve_paper_initial_cash(
    config: dict[str, Any], *, requested_initial_cash: float | None = None
) -> float:
    """Resolve paper capital without changing formal execution economics."""

    capital_contract = config.get("autopilot_capital_execution_contract")
    if not isinstance(capital_contract, dict):
        return float(
            requested_initial_cash
            if requested_initial_cash is not None
            else config.get("paper_initial_cash", _DEFAULT_PAPER_INITIAL_CASH)
        )
    expected_contract = {
        "contract_version": "autopilot-capital-execution-v1",
        "portfolio_construction": str(config.get("portfolio_construction") or ""),
        "capacity_notional": float(config.get("capacity_notional") or 0.0),
        "paper_initial_cash": float(config.get("paper_initial_cash") or 0.0),
        "topk": int(config.get("topk") or 0),
        "n_drop": int(config.get("n_drop") or 0),
        "lot_size": int(config.get("lot_size") or 0),
        "min_commission": float(config.get("min_commission") or 0.0),
        "cost_schedule_version": str(config.get("cost_schedule_version") or ""),
    }
    if (
        capital_contract != expected_contract
        or _canonical_sha256(capital_contract)
        != str(config.get("autopilot_capital_execution_contract_sha256") or "")
    ):
        raise ValueError("paper capital execution contract is inconsistent")
    configured_cash = float(capital_contract["paper_initial_cash"])
    capacity_notional = float(capital_contract["capacity_notional"])
    if capacity_notional <= 0.0 or configured_cash <= 0.0:
        raise ValueError("paper cash and formal capacity notional must be positive")
    resolved = float(
        requested_initial_cash
        if requested_initial_cash is not None
        else configured_cash
    )
    if resolved <= 0.0:
        raise ValueError("paper cash must be positive")
    # Capacity is a stress-test/evaluation scale.  A novice's paper principal
    # is an account input and must never be silently replaced by that scale.
    return resolved


@dataclass(frozen=True)
class ForwardGateThresholds:
    """Pre-registered forward evidence gate (design 4.5/6.11).

    The dataclass defaults preserve the legacy gate. Explicit product horizons
    always use ``forward_gate_thresholds_for_horizon``: natural time is counted
    in certified exchange sessions, swing reviews in distinct ISO weeks, long
    reviews in distinct calendar months, and full holding cycles in valid
    buy-to-flat round trips rather than sell-batch proxies.
    """

    min_forward_calendar_days: int = 20
    min_forward_trading_days: int = 0
    min_decision_batches: int = 10
    min_completed_cycles: int = 0
    min_closed_round_trips: int = 0
    min_review_events: int = 0
    min_financial_report_reviews: int = 0
    min_data_completeness: float = 0.95
    min_reconciliation_rate: float = 1.0
    max_cost_deviation: float = 0.005

    def __post_init__(self) -> None:
        if (
            self.min_forward_calendar_days < 0
            or self.min_forward_trading_days < 0
            or self.min_decision_batches < 0
            or self.min_completed_cycles < 0
            or self.min_closed_round_trips < 0
            or self.min_review_events < 0
            or self.min_financial_report_reviews < 0
        ):
            raise ValueError("forward gate count thresholds must be non-negative")
        if not 0 < self.min_data_completeness <= 1 or not 0 < self.min_reconciliation_rate <= 1:
            raise ValueError("forward gate rates must be in (0, 1]")
        if not 0 <= self.max_cost_deviation < 1:
            raise ValueError("forward gate cost deviation must be in [0, 1)")


def _now() -> datetime:
    return datetime.now(UTC)


def _insufficient(reasons: list[str], **extra: Any) -> dict[str, Any]:
    return {"status": "insufficient_evidence", "passed": False, "reasons": reasons, **extra}


def _count_closed_round_trips(fills: list[Any]) -> tuple[int, int]:
    """Count valid A-share long-only buy-to-flat sequences per instrument.

    A paper stage starts with no positions and follows the production T+1
    contract: a sell may consume only lots acquired on an earlier Shanghai
    trade date.  An invalid fill clears that instrument's local proof state so
    a later transaction cannot turn the invalid quantity into a phantom closed
    round trip.  The caller separately treats any invalid fill as a hard gate
    failure.
    """

    lots: dict[str, list[tuple[date, int]]] = {}
    closed_round_trips = 0
    invalid_fills = 0
    for fill in fills:
        instrument = str(fill.instrument)
        side = str(fill.side).lower()
        quantity = int(fill.quantity)
        executed_at = getattr(fill, "executed_at", None)
        if (
            quantity <= 0
            or not isinstance(executed_at, datetime)
            or executed_at.tzinfo is None
        ):
            invalid_fills += 1
            lots.pop(instrument, None)
            continue
        trade_date = executed_at.astimezone(_SHANGHAI).date()
        if side == "buy":
            lots.setdefault(instrument, []).append((trade_date, quantity))
            continue
        if side != "sell":
            invalid_fills += 1
            lots.pop(instrument, None)
            continue
        instrument_lots = lots.get(instrument, [])
        eligible = sum(
            lot_quantity
            for acquired_on, lot_quantity in instrument_lots
            if acquired_on < trade_date
        )
        if quantity > eligible:
            invalid_fills += 1
            lots.pop(instrument, None)
            continue
        remaining = quantity
        retained: list[tuple[date, int]] = []
        for acquired_on, lot_quantity in instrument_lots:
            if acquired_on < trade_date and remaining > 0:
                consumed = min(lot_quantity, remaining)
                lot_quantity -= consumed
                remaining -= consumed
            if lot_quantity > 0:
                retained.append((acquired_on, lot_quantity))
        if retained:
            lots[instrument] = retained
        else:
            lots.pop(instrument, None)
            closed_round_trips += 1
    return closed_round_trips, invalid_fills


def _review_period_key(horizon_profile: str, completed_at: datetime) -> str:
    """Return the one-per-period bucket used by the frozen horizon gate."""

    local = completed_at.astimezone(_SHANGHAI)
    if horizon_profile == SWING_1_6M:
        iso_year, iso_week, _ = local.isocalendar()
        return f"{iso_year:04d}-W{iso_week:02d}"
    if horizon_profile == LONG_1_3Y:
        return f"{local.year:04d}-{local.month:02d}"
    return local.date().isoformat()


def forward_gate_thresholds_for_horizon(profile: str) -> ForwardGateThresholds:
    """Return the frozen minimum forward proof for one product horizon."""

    if profile == SHORT_1_5D:
        return ForwardGateThresholds(
            min_forward_calendar_days=0,
            min_forward_trading_days=90,
            min_decision_batches=60,
            min_completed_cycles=0,
            min_closed_round_trips=30,
        )
    if profile == SWING_1_6M:
        return ForwardGateThresholds(
            min_forward_calendar_days=0,
            min_forward_trading_days=252,
            min_decision_batches=0,
            min_completed_cycles=0,
            min_closed_round_trips=6,
            min_review_events=24,
        )
    if profile == LONG_1_3Y:
        return ForwardGateThresholds(
            min_forward_calendar_days=0,
            # Three years of live operation is a useful maturity badge, but
            # it is too slow to be the first recommendation gate.  The
            # historical sealed OOS contract remains 756 sessions; the live
            # gate requires roughly one exchange year plus monthly and PIT
            # financial-report reviews.
            min_forward_trading_days=252,
            min_decision_batches=0,
            min_completed_cycles=0,
            min_review_events=12,
            min_financial_report_reviews=4,
        )
    if profile == LEGACY_AMBIGUOUS:
        return ForwardGateThresholds()
    raise ValueError(f"unsupported forward-gate horizon profile: {profile}")


def _require_horizon_forward_minima(
    profile: str, thresholds: ForwardGateThresholds
) -> None:
    minimum = forward_gate_thresholds_for_horizon(profile)
    fields = {
        SHORT_1_5D: (
            "min_forward_trading_days",
            "min_decision_batches",
            "min_closed_round_trips",
        ),
        SWING_1_6M: (
            "min_forward_trading_days",
            "min_review_events",
            "min_closed_round_trips",
        ),
        LONG_1_3Y: (
            "min_forward_trading_days",
            "min_review_events",
            "min_financial_report_reviews",
        ),
        LEGACY_AMBIGUOUS: (),
    }[profile]
    weaker = [
        field
        for field in fields
        if int(getattr(thresholds, field)) < int(getattr(minimum, field))
    ]
    if weaker:
        raise ValueError(
            f"forward gate is weaker than the frozen {profile} minimum: "
            + ", ".join(weaker)
        )


def build_forward_gate_criteria(
    *,
    horizon_profile: str,
    horizon_contract_sha256: str,
    thresholds: ForwardGateThresholds,
) -> dict[str, Any]:
    """Build the canonical immutable forward-gate criteria document."""

    if len(horizon_contract_sha256) != 64:
        raise ValueError("forward gate requires a sealed horizon contract")
    return {
        "contract_version": FORWARD_GATE_CRITERIA_VERSION,
        "horizon_profile": str(horizon_profile),
        "horizon_contract_sha256": str(horizon_contract_sha256),
        "thresholds": asdict(thresholds),
    }


def build_horizon_review_evidence(
    *,
    event_id: str,
    review_type: str,
    horizon_profile: str,
    completed_at: datetime,
    strategy_version_id: str,
    signal_date: date,
    dataset_identity_sha256: str,
    trigger_source: str,
    trigger_effective_date: date,
    report_period: str | None = None,
    announcement_date: date | None = None,
    previous_signal_date: date | None = None,
    source_datasets: list[str] | None = None,
    source_event_count: int | None = None,
    source_event_sha256: str | None = None,
    report_periods: list[str] | None = None,
    reviewed_instruments: list[str] | None = None,
    review_scope_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one deduplicated review marker for a governed order plan."""

    if not event_id.strip():
        raise ValueError("horizon review event_id is required")
    if not strategy_version_id.strip():
        raise ValueError("horizon review strategy version is required")
    normalized_dataset_identity = str(dataset_identity_sha256 or "").lower()
    if len(normalized_dataset_identity) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized_dataset_identity
    ):
        raise ValueError("horizon review dataset identity must be a SHA-256")
    if review_type not in {"scheduled_review", "financial_report_review"}:
        raise ValueError("unsupported horizon review type")
    if horizon_profile not in {SWING_1_6M, LONG_1_3Y}:
        raise ValueError("horizon reviews are only counted for swing or long strategies")
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise ValueError("horizon review completed_at must be timezone-aware")
    if completed_at.astimezone(_SHANGHAI).date() != signal_date:
        raise ValueError("horizon review completion must match the signal date")
    if trigger_effective_date != signal_date:
        raise ValueError("horizon review trigger must be effective on the signal date")
    expected_scheduled_source = {
        SWING_1_6M: "rebalance_calendar:week",
        LONG_1_3Y: "rebalance_calendar:month",
    }[horizon_profile]
    if review_type == "scheduled_review":
        if trigger_source != expected_scheduled_source:
            raise ValueError("scheduled review trigger does not match horizon cadence")
        if any(
            value is not None
            for value in (
                report_period,
                announcement_date,
                previous_signal_date,
                source_event_count,
                source_event_sha256,
                review_scope_sha256,
            )
        ) or source_datasets or report_periods or reviewed_instruments:
            raise ValueError("scheduled reviews must not claim financial announcement evidence")
    else:
        if horizon_profile != LONG_1_3Y or trigger_source != "pit_financial_announcement":
            raise ValueError("financial reviews are reserved for PIT long-horizon events")
        normalized_sources = sorted(set(str(item) for item in source_datasets or []))
        normalized_source_sha256 = str(source_event_sha256 or "").lower()
        normalized_report_periods = sorted(
            set(str(item or "").strip() for item in report_periods or [])
        )
        normalized_reviewed_instruments = sorted(
            set(str(item or "").strip().upper() for item in reviewed_instruments or [])
        )
        normalized_scope_sha256 = str(review_scope_sha256 or "").lower()
        if (
            not str(report_period or "").strip()
            or not normalized_report_periods
            or str(report_period).strip() != normalized_report_periods[-1]
            or any(
                len(period) != 6
                or not period[:4].isdigit()
                or period[4] != "Q"
                or period[5] not in "1234"
                for period in normalized_report_periods
            )
            or announcement_date is None
            or previous_signal_date is None
            or not previous_signal_date <= announcement_date < signal_date
            or not normalized_sources
            or not normalized_reviewed_instruments
            or any(
                len(instrument) < 3
                or instrument[:2] not in {"SH", "SZ", "BJ"}
                or not instrument[2:].isdigit()
                for instrument in normalized_reviewed_instruments
            )
            or isinstance(source_event_count, bool)
            or not isinstance(source_event_count, int)
            or source_event_count < 1
            or len(normalized_source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in normalized_source_sha256
            )
            or len(normalized_scope_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in normalized_scope_sha256
            )
        ):
            raise ValueError(
                "financial review requires report periods, PIT dates, reviewed "
                "instruments, and sealed source/scope identity"
            )
    payload = {
        "contract_version": HORIZON_REVIEW_CONTRACT_VERSION,
        "event_id": event_id.strip(),
        "review_type": review_type,
        "horizon_profile": horizon_profile,
        "status": "completed",
        "completed_at": completed_at.astimezone(UTC).replace(microsecond=0).isoformat(),
        "strategy_version_id": strategy_version_id.strip(),
        "signal_date": signal_date.isoformat(),
        "dataset_identity_sha256": normalized_dataset_identity,
        "trigger_source": trigger_source,
        "trigger_effective_date": trigger_effective_date.isoformat(),
        "report_period": str(report_period).strip() if report_period is not None else None,
        "report_periods": (
            normalized_report_periods
            if review_type == "financial_report_review"
            else []
        ),
        "announcement_date": (
            announcement_date.isoformat() if announcement_date is not None else None
        ),
        "previous_signal_date": (
            previous_signal_date.isoformat() if previous_signal_date is not None else None
        ),
        "source_datasets": sorted(set(str(item) for item in source_datasets or [])),
        "source_event_count": source_event_count,
        "source_event_sha256": (
            str(source_event_sha256).lower() if source_event_sha256 is not None else None
        ),
        "reviewed_instruments": (
            normalized_reviewed_instruments
            if review_type == "financial_report_review"
            else []
        ),
        "review_scope_sha256": (
            normalized_scope_sha256
            if review_type == "financial_report_review"
            else None
        ),
    }
    return {**payload, "evidence_sha256": _canonical_sha256(payload)}


def validate_horizon_review_evidence(
    value: dict[str, Any],
    *,
    strategy_version_id: str,
    horizon_profile: str,
    signal_date: date,
    dataset_identity_sha256: str,
    stage_opened_at: datetime | None = None,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Validate a review marker against its actual immutable paper batch."""

    if not isinstance(value, dict):
        raise ValueError("horizon review evidence must be an object")
    review = dict(value)
    evidence_sha256 = str(review.pop("evidence_sha256", "")).lower()
    if (
        len(evidence_sha256) != 64
        or any(character not in "0123456789abcdef" for character in evidence_sha256)
        or _canonical_sha256(review) != evidence_sha256
    ):
        raise ValueError("horizon review evidence seal is invalid")
    try:
        completed_at = datetime.fromisoformat(str(review["completed_at"]))
        recorded_signal_date = date.fromisoformat(str(review["signal_date"]))
        trigger_effective_date = date.fromisoformat(
            str(review["trigger_effective_date"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("horizon review evidence dates are invalid") from exc
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise ValueError("horizon review completion must be timezone-aware")
    if (
        review.get("contract_version")
        not in {
            HORIZON_REVIEW_CONTRACT_VERSION,
            LEGACY_HORIZON_REVIEW_CONTRACT_VERSION,
        }
        or review.get("status") != "completed"
        or review.get("horizon_profile") != horizon_profile
        or review.get("strategy_version_id") != strategy_version_id
        or review.get("dataset_identity_sha256") != dataset_identity_sha256
        or recorded_signal_date != signal_date
        or trigger_effective_date != signal_date
        or completed_at.astimezone(_SHANGHAI).date() != signal_date
        or not str(review.get("event_id") or "").strip()
    ):
        raise ValueError("horizon review evidence does not match its paper batch")
    review_type = str(review.get("review_type") or "")
    contract_version = str(review.get("contract_version") or "")
    trigger_source = str(review.get("trigger_source") or "")
    expected_scheduled_source = {
        SWING_1_6M: "rebalance_calendar:week",
        LONG_1_3Y: "rebalance_calendar:month",
    }.get(horizon_profile)
    if review_type == "scheduled_review":
        financial_values = tuple(
            review.get(field)
            for field in (
                "report_period",
                "announcement_date",
                "previous_signal_date",
                "source_event_count",
                "source_event_sha256",
                "review_scope_sha256",
            )
        )
        if (
            trigger_source != expected_scheduled_source
            or any(value not in (None, "") for value in financial_values)
            or review.get("source_datasets") not in (None, [], ())
            or review.get("report_periods") not in (None, [], ())
            or review.get("reviewed_instruments") not in (None, [], ())
        ):
            raise ValueError("scheduled horizon review evidence is invalid")
    elif review_type == "financial_report_review":
        try:
            announcement_date = date.fromisoformat(str(review["announcement_date"]))
            previous_signal_date = date.fromisoformat(
                str(review["previous_signal_date"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("financial review evidence dates are invalid") from exc
        sources = review.get("source_datasets")
        event_count = review.get("source_event_count")
        event_sha256 = str(review.get("source_event_sha256") or "").lower()
        report_period = str(review.get("report_period") or "").strip()
        if (
            horizon_profile != LONG_1_3Y
            or trigger_source != "pit_financial_announcement"
            or not report_period
            or not previous_signal_date <= announcement_date < signal_date
            or not isinstance(sources, list)
            or not sources
            or sources != sorted(set(str(item) for item in sources))
            or isinstance(event_count, bool)
            or not isinstance(event_count, int)
            or event_count < 1
            or len(event_sha256) != 64
            or any(character not in "0123456789abcdef" for character in event_sha256)
        ):
            raise ValueError("financial horizon review evidence is invalid")
        if contract_version == HORIZON_REVIEW_CONTRACT_VERSION:
            report_periods = review.get("report_periods")
            reviewed_instruments = review.get("reviewed_instruments")
            scope_sha256 = str(review.get("review_scope_sha256") or "").lower()
            if (
                not isinstance(report_periods, list)
                or report_periods
                != sorted(set(str(item or "").strip() for item in report_periods))
                or not report_periods
                or report_period != report_periods[-1]
                or any(
                    len(period) != 6
                    or not period[:4].isdigit()
                    or period[4] != "Q"
                    or period[5] not in "1234"
                    for period in report_periods
                )
                or not isinstance(reviewed_instruments, list)
                or reviewed_instruments
                != sorted(
                    set(
                        str(instrument or "").strip().upper()
                        for instrument in reviewed_instruments
                    )
                )
                or not reviewed_instruments
                or any(
                    len(instrument) < 3
                    or instrument[:2] not in {"SH", "SZ", "BJ"}
                    or not instrument[2:].isdigit()
                    for instrument in reviewed_instruments
                )
                or len(scope_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in scope_sha256
                )
            ):
                raise ValueError("financial horizon review scope binding is invalid")
        elif any(
            review.get(field) not in (None, [], ())
            for field in (
                "report_periods",
                "reviewed_instruments",
                "review_scope_sha256",
            )
        ):
            raise ValueError("legacy financial review cannot claim scoped evidence")
        if stage_opened_at is not None and not (
            stage_opened_at.astimezone(_SHANGHAI).date() < announcement_date
        ):
            raise ValueError("financial review announcement predates the paper stage")
    else:
        raise ValueError("unsupported horizon review evidence type")
    if stage_opened_at is not None and completed_at < stage_opened_at:
        raise ValueError("horizon review completion predates the paper stage")
    if observed_at is not None and completed_at > observed_at:
        raise ValueError("horizon review completion is in the future")
    return {**review, "evidence_sha256": evidence_sha256}


def _forward_gate_criteria(version: Any, gate: ForwardGateThresholds) -> dict[str, Any]:
    return build_forward_gate_criteria(
        horizon_profile=str(version.horizon_profile),
        horizon_contract_sha256=str(version.horizon_contract_sha256),
        thresholds=gate,
    )


def _require_forward_gate_criteria(version: Any, gate: Any) -> dict[str, Any]:
    thresholds = ForwardGateThresholds(
        **{
            key: getattr(gate, key)
            for key in (
                "min_forward_calendar_days",
                "min_forward_trading_days",
                "min_decision_batches",
                "min_completed_cycles",
                "min_closed_round_trips",
                "min_review_events",
                "min_financial_report_reviews",
                "min_data_completeness",
                "min_reconciliation_rate",
                "max_cost_deviation",
            )
        }
    )
    _require_horizon_forward_minima(str(version.horizon_profile), thresholds)
    expected = _forward_gate_criteria(version, thresholds)
    if (
        dict(gate.criteria_json or {}) != expected
        or str(gate.criteria_sha256 or "") != _canonical_sha256(expected)
    ):
        raise ValueError("forward evidence gate criteria seal is invalid")
    return expected


def _require_forward_only_qualification(
    connection: Any,
    *,
    version: Any,
    gate: Any | None = None,
    backtest: Any | None = None,
) -> dict[str, Any] | None:
    if str(getattr(version, "evidence_mode", "legacy_ambiguous")) != EVIDENCE_MODE_REPLAY:
        return None
    qualification = require_qualification(
        connection,
        version=version,
        backtest=backtest,
    )
    if gate is not None and (
        dict(gate.criteria_json or {})
        != dict(qualification["forward_criteria_json"] or {})
        or str(gate.criteria_sha256 or "")
        != str(qualification["forward_criteria_sha256"] or "")
        or rehabilitation_canonical_sha256(dict(gate.criteria_json or {}))
        != str(gate.criteria_sha256 or "")
    ):
        raise ValueError(
            "forward-only rehabilitation gate differs from its immutable qualification"
        )
    return qualification


class PromotionStore:
    """Paper stage lifecycle and forward evidence gate evaluation."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = open_database(database_url)

    # ------------------------------------------------------------------
    # Gate registration (pre-paper only)
    # ------------------------------------------------------------------

    def register_forward_gate(
        self,
        version_id: str,
        *,
        actor: str,
        thresholds: ForwardGateThresholds | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Pre-register the immutable forward gate before a paper stage exists.

        ``StrategyStore.approve`` commits the formal approval before it invokes
        this store, so an approved version already carries ``promotion_stage =
        paper`` here.  The irreversible boundary is therefore the creation of
        the first paper stage, not that marker.  Identical retries are allowed;
        thresholds can never be inserted or changed after any stage exists.
        """

        if len(actor.strip()) < 2:
            raise ValueError("a responsible actor is required")
        if thresholds is not None and overrides:
            raise ValueError("pass thresholds or overrides, not both")
        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .with_for_update()
            ).first()
            if version is None:
                raise KeyError(version_id)
            gate = (
                thresholds
                if thresholds is not None
                else (
                    ForwardGateThresholds(**overrides)
                    if overrides
                    else forward_gate_thresholds_for_horizon(
                        str(version.horizon_profile)
                    )
                )
            )
            _require_horizon_forward_minima(str(version.horizon_profile), gate)
            if version.promotion_stage == STAGE_RECOMMENDATION_ENABLED:
                raise ValueError(
                    "the forward gate cannot change after recommendations are enabled"
                )
            now = _now()
            criteria = _forward_gate_criteria(version, gate)
            values = {
                **asdict(gate),
                "criteria_json": criteria,
                "criteria_sha256": _canonical_sha256(criteria),
                "registered_by": actor.strip(),
                "updated_at": now,
            }
            existing = connection.execute(
                select(strategy_forward_gates).where(
                    strategy_forward_gates.c.strategy_version_id == version_id
                )
            ).first()
            qualification = _require_forward_only_qualification(
                connection,
                version=version,
                gate=existing,
            )
            if qualification is not None and existing is None:
                raise ValueError(
                    "forward-only rehabilitation gate is created atomically at admission"
                )
            if existing is not None:
                immutable_values = {
                    key: values[key]
                    for key in (
                        "min_forward_calendar_days",
                        "min_forward_trading_days",
                        "min_decision_batches",
                        "min_completed_cycles",
                        "min_closed_round_trips",
                        "min_review_events",
                        "min_financial_report_reviews",
                        "min_data_completeness",
                        "min_reconciliation_rate",
                        "max_cost_deviation",
                        "criteria_json",
                        "criteria_sha256",
                    )
                }
                recorded_values = {
                    key: getattr(existing, key) for key in immutable_values
                }
                if recorded_values == immutable_values:
                    return {
                        "strategy_version_id": version_id,
                        **asdict(gate),
                        "criteria_json": criteria,
                        "criteria_sha256": _canonical_sha256(criteria),
                    }
                raise ValueError(
                    "the forward gate is immutable once registered; use a new "
                    "StrategyVersion for different criteria"
                )
            existing_stage = connection.execute(
                select(strategy_promotion_stages.c.id)
                .where(strategy_promotion_stages.c.strategy_version_id == version_id)
                .limit(1)
            ).first()
            if existing_stage is not None:
                raise ValueError(
                    "a paper stage exists without a pre-registered forward gate"
                )

            connection.execute(
                insert(strategy_forward_gates).values(
                    strategy_version_id=version_id, registered_at=now, **values
                )
            )
        return {
            "strategy_version_id": version_id,
            **asdict(gate),
            "criteria_json": criteria,
            "criteria_sha256": _canonical_sha256(criteria),
        }

    # ------------------------------------------------------------------
    # Paper stage opening (candidate -> paper, automatic after the hard gate)
    # ------------------------------------------------------------------

    def open_paper_stage(self, version_id: str, *, actor: str) -> dict[str, Any]:
        """Open the isolated paper stage after the formal hard gate passes.

        Idempotent: an already-active stage is returned unchanged. When the
        approval backtest carries no consumable dataset descriptors
        (``datasets.json`` in the artifact root), the stage stays in
        ``awaiting_simulation`` until :meth:`attach_paper_simulation` binds an
        isolated account — evidence cannot accumulate without it.
        """

        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .with_for_update()
            ).first()
            if version is None:
                raise KeyError(version_id)
            if str(version.status) != "approved":
                raise ValueError("paper stage requires an approved strategy version")
            # Keep the advisory lock and the stage insert in the same
            # transaction.  This closes both the direct ``open_paper_stage``
            # compatibility path and the race between two automatic approvals.
            require_horizon_challenger_capacity(
                connection,
                horizon_profile=str(version.horizon_profile),
                version_id=version_id,
            )
            gate = connection.execute(
                select(strategy_forward_gates).where(
                    strategy_forward_gates.c.strategy_version_id == version_id
                )
            ).first()
            if gate is None:
                raise ValueError(
                    "paper stage requires a pre-registered immutable forward gate"
                )
            _require_forward_gate_criteria(version, gate)
            _require_forward_only_qualification(
                connection,
                version=version,
                gate=gate,
            )
            existing = connection.execute(
                select(strategy_promotion_stages)
                .where(
                    strategy_promotion_stages.c.strategy_version_id == version_id,
                    strategy_promotion_stages.c.status.in_([_STAGE_ACTIVE, _STAGE_AWAITING]),
                )
                .order_by(strategy_promotion_stages.c.stage_index.desc())
                .limit(1)
            ).first()
            if existing is not None and str(existing.status) == _STAGE_ACTIVE:
                return self._stage_dict(existing)
            if existing is not None:
                stage_id = str(existing.id)
            else:
                next_index = (
                    connection.execute(
                        select(func.max(strategy_promotion_stages.c.stage_index)).where(
                            strategy_promotion_stages.c.strategy_version_id == version_id
                        )
                    ).scalar()
                    or 0
                ) + 1
                stage_id = uuid.uuid4().hex
                connection.execute(
                    insert(strategy_promotion_stages).values(
                        id=stage_id,
                        strategy_version_id=version_id,
                        stage_index=next_index,
                        simulation_portfolio_id=None,
                        status=_STAGE_AWAITING,
                        opened_at=_now(),
                        created_by=actor.strip(),
                    )
                )
        try:
            datasets = self._load_backtest_datasets(version_id)
            if datasets is None:
                raise ValueError("approval backtest carries no dataset descriptors")
            self.attach_paper_simulation(
                version_id,
                actor=actor,
                daily_dataset=datasets["daily"],
                execution_dataset=datasets["execution"],
            )
        except ValueError as exc:
            self._event(
                version_id,
                event_type="strategy.paper_stage_awaiting_simulation",
                actor=actor,
                payload={"stage_id": stage_id, "error": str(exc)},
            )
        return self.current_stage(version_id)

    def prepare_paper_stage(self, version_id: str, *, actor: str) -> dict[str, Any]:
        """Idempotently register the default gate, then open the paper stage.

        Registration and stage creation deliberately use separate transactions:
        a crash can leave a registered gate with no stage, which is safe and
        recoverable by retry.  The inverse (a stage with no gate) is rejected by
        :meth:`open_paper_stage`.
        """

        # Preserve an operator's pre-registered strategy-specific gate.  The
        # automatic approval transition supplies the governed default only
        # when no gate exists; it must never silently replace a stricter
        # preregistration just before opening the stage.
        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .with_for_update()
            ).first()
            if version is None:
                raise KeyError(version_id)
            require_horizon_challenger_capacity(
                connection,
                horizon_profile=str(version.horizon_profile),
                version_id=version_id,
            )
            existing_gate = connection.execute(
                select(strategy_forward_gates).where(
                    strategy_forward_gates.c.strategy_version_id == version_id
                )
            ).first()
            existing_stage = connection.execute(
                select(strategy_promotion_stages.c.id)
                .where(strategy_promotion_stages.c.strategy_version_id == version_id)
                .limit(1)
            ).first()
            if existing_gate is None:
                if str(version.evidence_mode) == EVIDENCE_MODE_REPLAY:
                    raise ValueError(
                        "forward-only rehabilitation is missing its atomic qualified gate"
                    )
                if existing_stage is not None:
                    raise ValueError(
                        "a paper stage exists without a pre-registered forward gate"
                    )
                now = _now()
                default_gate = forward_gate_thresholds_for_horizon(
                    str(version.horizon_profile)
                )
                criteria = _forward_gate_criteria(version, default_gate)
                connection.execute(
                    insert(strategy_forward_gates).values(
                        strategy_version_id=version_id,
                        **asdict(default_gate),
                        criteria_json=criteria,
                        criteria_sha256=_canonical_sha256(criteria),
                        registered_by=actor.strip(),
                        registered_at=now,
                        updated_at=now,
                    )
                )
            else:
                _require_forward_only_qualification(
                    connection,
                    version=version,
                    gate=existing_gate,
                )
        return self.open_paper_stage(version_id, actor=actor)

    def require_paper_signal(
        self,
        version_id: str,
        *,
        portfolio_id: str,
        signal_date: date,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Bind a paper signal to the active stage's genuinely forward period."""

        current = now or _now()
        with self.engine.connect() as connection:
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).first()
            stage = connection.execute(
                select(strategy_promotion_stages)
                .where(
                    strategy_promotion_stages.c.strategy_version_id == version_id,
                    strategy_promotion_stages.c.simulation_portfolio_id == portfolio_id,
                    strategy_promotion_stages.c.status == _STAGE_ACTIVE,
                )
                .order_by(strategy_promotion_stages.c.stage_index.desc())
                .limit(1)
            ).first()
            gate = connection.execute(
                select(strategy_forward_gates).where(
                    strategy_forward_gates.c.strategy_version_id == version_id
                )
            ).first()
        if version is None or stage is None or gate is None:
            raise ValueError(
                "paper signal requires the active isolated promotion stage and its forward gate"
            )
        _require_forward_gate_criteria(version, gate)
        opened_date = stage.opened_at.astimezone(_SHANGHAI).date()
        current_date = current.astimezone(_SHANGHAI).date()
        if signal_date <= opened_date:
            raise ValueError(
                "paper signal must be from a trading day after the promotion stage opened"
            )
        if signal_date > current_date:
            raise ValueError("paper signal cannot use a future date")
        return {
            **self._stage_dict(stage),
            "forward_signal_after": opened_date.isoformat(),
        }

    def require_recommendation_signal(
        self,
        version_id: str,
        *,
        signal_date: date,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Reject recommendation backfills from before atomic promotion."""

        current = now or _now()
        with self.engine.connect() as connection:
            stage = connection.execute(
                select(strategy_promotion_stages)
                .where(
                    strategy_promotion_stages.c.strategy_version_id == version_id,
                    strategy_promotion_stages.c.promoted_at.is_not(None),
                )
                .order_by(strategy_promotion_stages.c.stage_index.desc())
                .limit(1)
            ).first()
        if stage is None or stage.promoted_at is None:
            raise ValueError("recommendation signal requires a promoted paper stage")
        promoted_date = stage.promoted_at.astimezone(_SHANGHAI).date()
        current_date = current.astimezone(_SHANGHAI).date()
        if signal_date <= promoted_date:
            raise ValueError(
                "recommendation signal must be from a trading day after promotion"
            )
        if signal_date > current_date:
            raise ValueError("recommendation signal cannot use a future date")
        return {
            **self._stage_dict(stage),
            "forward_signal_after": promoted_date.isoformat(),
        }

    def attach_paper_simulation(
        self,
        version_id: str,
        *,
        actor: str,
        daily_dataset: dict[str, Any],
        execution_dataset: dict[str, Any],
        initial_cash: float | None = None,
    ) -> dict[str, Any]:
        """Bind the isolated paper simulation account to the awaiting stage."""

        with self.engine.begin() as connection:
            stage = connection.execute(
                select(strategy_promotion_stages)
                .where(
                    strategy_promotion_stages.c.strategy_version_id == version_id,
                    strategy_promotion_stages.c.status == _STAGE_AWAITING,
                )
                .order_by(strategy_promotion_stages.c.stage_index.desc())
                .limit(1)
            ).first()
            if stage is None:
                raise ValueError("no paper stage is waiting for a simulation account")
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).one()
        config = dict(version.config_json or {})
        profile_binding: dict[str, Any] | None = None
        if str(version.horizon_profile) != "legacy_ambiguous":
            active_profile = InvestorSimulationProfileStore(
                self.database_url
            ).get_active("primary")
            if active_profile is None:
                raise ValueError(
                    "paper account awaits explicit investor capital and market permissions"
                )
            profile_binding = bind_investor_profile(active_profile)
            profile_cash = float(active_profile["initial_capital"])
            if initial_cash is not None and abs(float(initial_cash) - profile_cash) > 1e-6:
                raise ValueError(
                    "paper account capital must equal the active investor-profile version"
                )
            initial_cash = profile_cash
        # Capacity remains in the research contract; the paper principal is
        # the explicit investor-owned account input for all new horizons.
        cash = resolve_paper_initial_cash(
            config, requested_initial_cash=initial_cash
        )
        simulation = SimulationStore(self.database_url)
        with self.engine.connect() as connection:
            existing_portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.promotion_stage_id == stage.id
                )
            ).first()
        if existing_portfolio is not None:
            portfolio = simulation.get(str(existing_portfolio.id))
        else:
            try:
                portfolio = simulation.create(
                    name=(
                        f"paper:{version_id[:8]}:v{int(version.version)}:stage{int(stage.stage_index)}"
                    )[:150],
                    source_type="strategy_version",
                    source_id=version_id,
                    promotion_stage_id=str(stage.id),
                    daily_dataset=daily_dataset,
                    execution_dataset=execution_dataset,
                    initial_cash=cash,
                    execution_policy={
                        "execution_algorithm": str(
                            config.get("execution_method") or "twap"
                        )
                    },
                    cost_schedule_version=str(
                        config.get("cost_schedule_version") or config.get("version") or ""
                    ),
                    actor=actor,
                    # Paper evidence must advance with immutable descendants of the
                    # formal snapshot.  The anchor identities remain fixed on the
                    # account while each batch records the exact rolled snapshots.
                    daily_roll_policy="latest_compatible",
                    execution_roll_policy="latest_compatible",
                    investor_profile_binding=profile_binding,
                )
            except ValueError:
                # A crash/concurrent retry may have committed the account but
                # not yet attached it to the stage. Recover only that exact
                # stage-owned row; unrelated uniqueness errors still surface.
                with self.engine.connect() as connection:
                    raced = connection.execute(
                        select(simulation_portfolios.c.id).where(
                            simulation_portfolios.c.promotion_stage_id == stage.id
                        )
                    ).first()
                if raced is None:
                    raise
                portfolio = simulation.get(str(raced.id))
        daily_provenance = dict(daily_dataset.get("provenance") or {})
        execution_provenance = dict(execution_dataset.get("provenance") or {})
        expected = (
            (portfolio.get("source_type"), "strategy_version"),
            (portfolio.get("source_id"), version_id),
            (portfolio.get("promotion_stage_id"), str(stage.id)),
            (portfolio.get("daily_dataset"), str(daily_dataset.get("name") or "")),
            (
                portfolio.get("daily_dataset_identity_sha256"),
                str(daily_provenance.get("dataset_identity_sha256") or ""),
            ),
            (
                portfolio.get("execution_dataset"),
                str(execution_dataset.get("name") or ""),
            ),
            (
                portfolio.get("execution_dataset_identity_sha256"),
                str(execution_provenance.get("dataset_identity_sha256") or ""),
            ),
        )
        if any(
            str(observed or "") != str(required or "")
            for observed, required in expected
        ):
            raise ValueError(
                "paper stage already owns a simulation account with different evidence"
            )
        if abs(float(portfolio.get("initial_cash") or 0.0) - cash) > 1e-6:
            raise ValueError(
                "paper stage account notional differs from the frozen formal contract"
            )
        if profile_binding is not None:
            raw_portfolio_binding = dict(
                portfolio.get("execution_policy") or {}
            ).get("investor_profile_binding")
            if (
                not isinstance(raw_portfolio_binding, dict)
                or validate_investor_profile_binding(raw_portfolio_binding)
                != profile_binding
            ):
                raise ValueError(
                    "paper stage account investor-profile binding differs from "
                    "the active immutable profile version"
                )
        with self.engine.begin() as connection:
            current_portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio["id"])
                .with_for_update()
            ).first()
            if current_portfolio is None:
                raise ValueError("paper stage simulation account disappeared")
            SimulationStore._require_current_source_contract(
                connection, current_portfolio
            )

            # Explicit horizon lanes retain up to two forward-shadow accounts
            # concurrently; the capacity gate is authoritative.  Legacy
            # Autopilot accounts keep the historical single-primary behavior.
            # Superseded legacy ledgers are frozen, never deleted.
            if _is_autopilot_config(config):
                prior_rows = connection.execute(
                    select(
                        strategy_promotion_stages,
                        strategy_versions.c.strategy_id.label("source_strategy_id"),
                        strategy_versions.c.config_json.label("source_config_json"),
                        strategy_versions.c.horizon_profile.label(
                            "source_horizon_profile"
                        ),
                    )
                    .join(
                        strategy_versions,
                        strategy_versions.c.id
                        == strategy_promotion_stages.c.strategy_version_id,
                    )
                    .where(
                        strategy_promotion_stages.c.status == _STAGE_ACTIVE,
                        strategy_promotion_stages.c.strategy_version_id != version_id,
                        strategy_versions.c.promotion_stage == STAGE_PAPER,
                    )
                    .with_for_update()
                ).all()
                now = _now()
                for prior in prior_rows:
                    if not _is_autopilot_config(prior.source_config_json):
                        continue
                    if not _autopilot_paper_accounts_compete(
                        str(version.horizon_profile or LEGACY_AMBIGUOUS),
                        str(prior.source_horizon_profile or LEGACY_AMBIGUOUS),
                    ):
                        continue
                    prior_portfolio_id = str(
                        prior.simulation_portfolio_id or ""
                    ).strip()
                    if prior_portfolio_id:
                        connection.execute(
                            update(simulation_portfolios)
                            .where(
                                simulation_portfolios.c.id == prior_portfolio_id,
                                simulation_portfolios.c.status == "active",
                            )
                            .values(status="paused", updated_at=now)
                        )
                    connection.execute(
                        update(strategy_promotion_stages)
                        .where(
                            strategy_promotion_stages.c.id == prior.id,
                            strategy_promotion_stages.c.status == _STAGE_ACTIVE,
                        )
                        .values(status=_STAGE_FROZEN)
                    )
                    connection.execute(
                        insert(strategy_events).values(
                            strategy_id=str(prior.source_strategy_id),
                            strategy_version_id=str(prior.strategy_version_id),
                            event_type="strategy.paper_stage_superseded",
                            actor=actor.strip(),
                            payload_json={
                                "stage_id": str(prior.id),
                                "simulation_portfolio_id": prior_portfolio_id or None,
                                "superseded_by_strategy_version_id": version_id,
                                "history_retained": True,
                            },
                            created_at=now,
                        )
                    )

            connection.execute(
                update(simulation_portfolios)
                .where(simulation_portfolios.c.id == portfolio["id"])
                .values(status="active", updated_at=_now())
            )
            attached = connection.execute(
                update(strategy_promotion_stages)
                .where(
                    strategy_promotion_stages.c.id == stage.id,
                    strategy_promotion_stages.c.status == _STAGE_AWAITING,
                    strategy_promotion_stages.c.simulation_portfolio_id.is_(None),
                )
                .values(
                    simulation_portfolio_id=portfolio["id"],
                    status=_STAGE_ACTIVE,
                    source_contract_hash=portfolio["execution_contract_hash"],
                    initial_cash=cash,
                )
            )
            if not attached.rowcount:
                current = connection.execute(
                    select(strategy_promotion_stages).where(
                        strategy_promotion_stages.c.id == stage.id
                    )
                ).first()
                if (
                    current is None
                    or str(current.status) != _STAGE_ACTIVE
                    or str(current.simulation_portfolio_id) != str(portfolio["id"])
                ):
                    raise ValueError("paper stage changed while attaching its account")
        self._event(
            version_id,
            event_type="strategy.paper_stage_opened",
            actor=actor,
            payload={
                "stage_id": str(stage.id),
                "simulation_portfolio_id": portfolio["id"],
                "initial_cash": cash,
            },
        )
        return self.current_stage(version_id)

    # ------------------------------------------------------------------
    # Forward evidence gate
    # ------------------------------------------------------------------

    def current_stage(self, version_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(strategy_promotion_stages)
                .where(
                    strategy_promotion_stages.c.strategy_version_id == version_id,
                    strategy_promotion_stages.c.status.in_([_STAGE_ACTIVE, _STAGE_AWAITING]),
                )
                .order_by(strategy_promotion_stages.c.stage_index.desc())
                .limit(1)
            ).first()
        return self._stage_dict(row) if row is not None else None

    def list_stages(self, version_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(strategy_promotion_stages)
                .where(strategy_promotion_stages.c.strategy_version_id == version_id)
                .order_by(strategy_promotion_stages.c.stage_index)
            ).all()
        return [self._stage_dict(row) for row in rows]

    def evaluate_forward_gate(self, version_id: str) -> dict[str, Any]:
        """Evaluate the pre-registered gate against the active stage's evidence."""

        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).first()
            if version is None:
                raise KeyError(version_id)
            gate = connection.execute(
                select(strategy_forward_gates).where(
                    strategy_forward_gates.c.strategy_version_id == version_id
                )
            ).first()
            if gate is None:
                return _insufficient(["forward evidence gate is not pre-registered"])
            try:
                criteria = _require_forward_gate_criteria(version, gate)
                _require_forward_only_qualification(
                    connection,
                    version=version,
                    gate=gate,
                )
            except ValueError as exc:
                return _insufficient([str(exc)])
            stage = connection.execute(
                select(strategy_promotion_stages)
                .where(
                    strategy_promotion_stages.c.strategy_version_id == version_id,
                    strategy_promotion_stages.c.status.in_([_STAGE_ACTIVE, _STAGE_AWAITING]),
                )
                .order_by(strategy_promotion_stages.c.stage_index.desc())
                .limit(1)
            ).first()
            if stage is None:
                return _insufficient(["no active paper stage exists"])
            if stage.simulation_portfolio_id is None:
                return _insufficient(
                    ["paper stage has no isolated simulation account; evidence is zero"]
                )
            portfolio = connection.execute(
                select(simulation_portfolios).where(
                    simulation_portfolios.c.id == stage.simulation_portfolio_id
                )
            ).first()
            try:
                SimulationStore._require_current_source_contract(connection, portfolio)
            except ValueError as exc:
                # Design 9.5: substantive contract drift freezes the stage
                # read-only and starts a fresh stage from zero evidence.
                connection.execute(
                    update(strategy_promotion_stages)
                    .where(strategy_promotion_stages.c.id == stage.id)
                    .values(
                        status=_STAGE_FROZEN,
                        frozen_at=_now(),
                        freeze_reason=str(exc),
                    )
                )
                connection.execute(
                    insert(strategy_promotion_stages).values(
                        id=uuid.uuid4().hex,
                        strategy_version_id=version_id,
                        stage_index=int(stage.stage_index) + 1,
                        simulation_portfolio_id=None,
                        status=_STAGE_AWAITING,
                        opened_at=_now(),
                        created_by="promotion-chain",
                    )
                )
                connection.execute(
                    insert(strategy_events).values(
                        strategy_id=str(version.strategy_id),
                        strategy_version_id=version_id,
                        event_type="strategy.paper_stage_frozen_contract_drift",
                        actor="promotion-chain",
                        payload_json={"stage_id": str(stage.id), "error": str(exc)},
                        created_at=_now(),
                    )
                )
                return _insufficient(
                    [
                        "paper stage frozen on source contract drift; "
                        "a new stage starts from zero evidence",
                        str(exc),
                    ],
                    stage_reset=True,
                )
            evidence = self._collect_evidence(connection, stage, portfolio, version)

        checks = {
            # A generic/manual batch on the paper account is not forward
            # evidence.  Treating it merely as absent would allow a mixed
            # ledger to hide an attempted replay, so any such row blocks the
            # stage until it is investigated/reset.
            "governed_batch_integrity": (
                1.0
                if evidence["ungoverned_batches"] == 0
                and evidence["duplicate_decision_batches"] == 0
                and evidence["invalid_lifecycle_batches"] == 0
                and evidence["invalid_round_trip_fills"] == 0
                else 0.0,
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
        if int(gate.min_forward_calendar_days) > 0:
            checks["forward_calendar_days"] = (
                evidence["forward_calendar_days"],
                int(gate.min_forward_calendar_days),
                "min",
            )
        profile = str(version.horizon_profile)
        if profile == LEGACY_AMBIGUOUS:
            checks.update(
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
        elif profile == SHORT_1_5D:
            checks.update(
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
        elif profile == SWING_1_6M:
            checks.update(
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
        elif profile == LONG_1_3Y:
            checks.update(
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
            return _insufficient([f"unsupported horizon profile: {profile}"])
        results = {
            name: {
                "observed": observed,
                "threshold": threshold,
                "passed": (
                    observed >= threshold if mode == "min" else observed <= threshold
                ),
            }
            for name, (observed, threshold, mode) in checks.items()
        }
        failures = [name for name, result in results.items() if not result["passed"]]
        if failures:
            return _insufficient(
                [f"{name} below/above the pre-registered threshold" for name in failures],
                checks=results,
                evidence=evidence,
                stage_id=str(stage.id),
                criteria_json=criteria,
                criteria_sha256=str(gate.criteria_sha256),
            )
        return {
            "status": "ok",
            "passed": True,
            "reasons": [],
            "checks": results,
            "evidence": evidence,
            "stage_id": str(stage.id),
            "contract_version": PROMOTION_CONTRACT_VERSION,
            "criteria_json": criteria,
            "criteria_sha256": str(gate.criteria_sha256),
        }

    def promote(self, version_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        """Atomically move a paper strategy through its frozen forward gate."""

        if not actor.strip() or len(reason.strip()) < 10:
            raise ValueError("actor and a meaningful approval reason are required")
        evaluation = self.evaluate_forward_gate(version_id)
        if not evaluation["passed"]:
            raise ValueError(
                "forward evidence gate is not satisfied (insufficient_evidence; "
                "thresholds are never lowered): " + "; ".join(evaluation["reasons"])
            )
        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .with_for_update()
            ).first()
            if version is None:
                raise KeyError(version_id)
            if str(version.status) != "approved" or version.promotion_stage != STAGE_PAPER:
                raise ValueError("only a paper-stage approved version can be promoted")
            promotion_gate = connection.execute(
                select(strategy_forward_gates).where(
                    strategy_forward_gates.c.strategy_version_id == version_id
                )
            ).first()
            if promotion_gate is None:
                raise ValueError("promotion requires an immutable forward gate")
            _require_forward_only_qualification(
                connection,
                version=version,
                gate=promotion_gate,
            )
            version_config = dict(version.config_json or {})
            if str(
                version.source_research_artifact_id
                or version_config.get("source_research_artifact_id")
                or ""
            ).strip():
                # A research artifact may be invalidated while the strategy is
                # accumulating months or years of paper evidence. Revalidate
                # the exact historical admission in this same promotion
                # transaction so stale/withdrawn research can never acquire
                # recommendation authority.
                from quant_platform.strategy_store import StrategyStore

                version_value = row_dict(version)
                version_value["config"] = version_config
                fresh_admission = StrategyStore(
                    self.database_url
                )._require_fin_strategy_formal_admission(
                    connection,
                    version_value,
                    allow_approved_paper=True,
                )
                approval_backtest = connection.execute(
                    select(backtest_runs)
                    .where(
                        backtest_runs.c.strategy_version_id == version_id,
                        backtest_runs.c.status == "succeeded",
                    )
                    .order_by(backtest_runs.c.created_at.desc())
                    .limit(1)
                ).first()
                recorded_admission = (
                    dict(approval_backtest.periods_json or {}).get(
                        "fin_strategy_formal_admission"
                    )
                    if approval_backtest is not None
                    else None
                )
                if recorded_admission != fresh_admission:
                    raise ValueError(
                        "fin_strategy historical admission changed before promotion"
                    )
            gate = connection.execute(
                select(strategy_forward_gates)
                .where(strategy_forward_gates.c.strategy_version_id == version_id)
                .with_for_update()
            ).first()
            if gate is None:
                raise ValueError("forward evidence gate disappeared during promotion")
            criteria = _require_forward_gate_criteria(version, gate)
            if (
                criteria != evaluation.get("criteria_json")
                or str(gate.criteria_sha256) != evaluation.get("criteria_sha256")
            ):
                raise ValueError("forward evidence gate changed during promotion")
            stage = connection.execute(
                select(strategy_promotion_stages)
                .where(strategy_promotion_stages.c.id == evaluation["stage_id"])
                .with_for_update()
            ).first()
            if (
                stage is None
                or str(stage.strategy_version_id) != version_id
                or str(stage.status) != _STAGE_ACTIVE
                or stage.promoted_at is not None
                or stage.simulation_portfolio_id is None
            ):
                raise ValueError("paper stage changed after forward-gate evaluation")
            portfolio = connection.execute(
                select(simulation_portfolios)
                .where(simulation_portfolios.c.id == stage.simulation_portfolio_id)
                .with_for_update()
            ).first()
            if portfolio is None:
                raise ValueError("paper stage simulation account disappeared")
            SimulationStore._require_current_source_contract(connection, portfolio)
            fresh_evidence = self._collect_evidence(
                connection, stage, portfolio, version
            )
            if fresh_evidence != evaluation["evidence"]:
                raise ValueError(
                    "paper evidence changed during promotion; evaluate the forward gate again"
                )
            now = _now()
            health_gate = load_production_health_gate(
                connection,
                version_id,
                now=now,
            )
            if health_gate.get("allow_new_risk") is not True:
                raise ValueError(
                    "sealed collector health does not allow recommendation activation: "
                    + str(health_gate.get("reason") or "missing live health evidence")
                )
            initial_health_snapshot_id = str(health_gate["snapshot_id"])
            health_status = str(health_gate["health_status"])
            incumbents = connection.execute(
                select(strategy_versions)
                .where(
                    strategy_versions.c.horizon_profile == version.horizon_profile,
                    strategy_versions.c.status == "approved",
                    strategy_versions.c.promotion_stage == STAGE_RECOMMENDATION_ENABLED,
                    strategy_versions.c.id != version_id,
                )
                .with_for_update()
            ).all()
            activation_token = uuid.uuid4().hex
            incumbent_bindings = [
                {
                    "strategy_version_id": str(incumbent.id),
                    "strategy_id": str(incumbent.strategy_id),
                    "status": str(incumbent.status),
                    "promotion_stage": str(incumbent.promotion_stage),
                }
                for incumbent in incumbents
            ]
            for incumbent in incumbents:
                connection.execute(
                    update(strategy_versions)
                    .where(strategy_versions.c.id == incumbent.id)
                    .values(status="retired")
                )
                connection.execute(
                    insert(strategy_events).values(
                        strategy_id=str(incumbent.strategy_id),
                        strategy_version_id=str(incumbent.id),
                        event_type="strategy.recommendation_replaced",
                        actor=actor.strip(),
                        payload_json={
                            "replacement_strategy_version_id": version_id,
                            "horizon_profile": str(version.horizon_profile),
                            "reason": reason.strip(),
                        },
                        created_at=now,
                    )
                )
            connection.execute(
                update(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .values(promotion_stage=STAGE_RECOMMENDATION_ENABLED)
            )
            connection.execute(
                update(strategy_promotion_stages)
                .where(strategy_promotion_stages.c.id == evaluation["stage_id"])
                .values(promoted_at=now)
            )
            connection.execute(
                insert(strategy_events).values(
                    strategy_id=str(version.strategy_id),
                    strategy_version_id=version_id,
                    event_type="strategy.recommendation_enabled",
                    actor=actor.strip(),
                    payload_json={
                        "reason": reason.strip(),
                        "evidence": evaluation["evidence"],
                        "stage_id": evaluation["stage_id"],
                        "initial_health_snapshot_id": initial_health_snapshot_id,
                        "initial_health_status": health_status,
                        "activation_token": activation_token,
                        "replaced_incumbents": incumbent_bindings,
                    },
                    created_at=now,
                )
            )
            # Strategy authority and the three-horizon account are persisted by
            # different stores.  This immutable event is the deliberately thin
            # saga boundary between them: projections keep serving the replaced
            # incumbent until the account records a complete cutover, and a
            # failed account activation can restore both sides atomically.
            connection.execute(
                insert(strategy_events).values(
                    strategy_id=str(version.strategy_id),
                    strategy_version_id=version_id,
                    event_type=ACTIVATION_PENDING_EVENT,
                    actor=actor.strip(),
                    payload_json={
                        "activation_token": activation_token,
                        "horizon_profile": str(version.horizon_profile),
                        "replaced_incumbents": incumbent_bindings,
                        "promotion_stage_id": evaluation["stage_id"],
                    },
                    created_at=now,
                )
            )
        return {
            "strategy_version_id": version_id,
            "promotion_stage": STAGE_RECOMMENDATION_ENABLED,
            "evidence": evaluation["evidence"],
            "initial_health_snapshot_id": initial_health_snapshot_id,
            "initial_health_status": health_status,
            "activation_token": activation_token,
            "replaced_incumbents": incumbent_bindings,
        }

    @staticmethod
    def _terminal_activation_tokens(connection: Any, version_id: str) -> set[str]:
        rows = connection.execute(
            select(strategy_events.c.payload_json).where(
                strategy_events.c.strategy_version_id == version_id,
                strategy_events.c.event_type.in_(
                    (ACTIVATION_COMPLETED_EVENT, ACTIVATION_ROLLED_BACK_EVENT)
                ),
            )
        ).all()
        return {
            str((row.payload_json or {}).get("activation_token") or "")
            for row in rows
            if (row.payload_json or {}).get("activation_token")
        }

    def pending_activation_cutovers(
        self, *, horizon_profile: str | None = None
    ) -> list[dict[str, Any]]:
        """Return unfinished account cutovers from the immutable strategy trace.

        This is a recovery projection, not another lifecycle state.  The
        authoritative lifecycle remains ``promotion_stage``; a pending event
        merely records that its account-side effect has not been acknowledged.
        """

        conditions = [
            strategy_events.c.event_type == ACTIVATION_PENDING_EVENT,
            strategy_versions.c.status == "approved",
            strategy_versions.c.promotion_stage == STAGE_RECOMMENDATION_ENABLED,
        ]
        if horizon_profile is not None:
            conditions.append(strategy_versions.c.horizon_profile == horizon_profile)
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    strategy_events,
                    strategy_versions.c.horizon_profile,
                )
                .join(
                    strategy_versions,
                    strategy_versions.c.id == strategy_events.c.strategy_version_id,
                )
                .where(*conditions)
                .order_by(strategy_events.c.id.desc())
            ).all()
            terminal_by_version: dict[str, set[str]] = {}
            result: list[dict[str, Any]] = []
            seen_versions: set[str] = set()
            for row in rows:
                version_id = str(row.strategy_version_id)
                if version_id in seen_versions:
                    continue
                payload = dict(row.payload_json or {})
                token = str(payload.get("activation_token") or "")
                if not token:
                    continue
                terminal = terminal_by_version.setdefault(
                    version_id,
                    self._terminal_activation_tokens(connection, version_id),
                )
                if token in terminal:
                    continue
                seen_versions.add(version_id)
                result.append(
                    {
                        "event_id": int(row.id),
                        "strategy_id": str(row.strategy_id),
                        "strategy_version_id": version_id,
                        "horizon_profile": str(row.horizon_profile),
                        "activation_token": token,
                        "replaced_incumbents": list(
                            payload.get("replaced_incumbents") or []
                        ),
                        "created_at": row.created_at.isoformat(),
                    }
                )
        return result

    def serving_incumbent_for_pending_cutover(
        self, horizon_profile: str
    ) -> str | None:
        """Keep the last verified advice visible until account cutover commits."""

        pending = self.pending_activation_cutovers(horizon_profile=horizon_profile)
        if not pending:
            return None
        incumbents = pending[0].get("replaced_incumbents") or []
        if len(incumbents) != 1:
            return None
        version_id = str(incumbents[0].get("strategy_version_id") or "")
        return version_id or None

    def complete_activation_cutover(
        self,
        version_id: str,
        *,
        activation_token: str,
        account_evidence: dict[str, Any],
        actor: str = "system:auto-promotion",
    ) -> dict[str, Any]:
        """Acknowledge a usable unified account after checking durable evidence."""

        token = activation_token.strip()
        allocation_id = str(account_evidence.get("allocation_id") or "")
        plan_id = str(account_evidence.get("netting_plan_id") or "")
        snapshot_evidence = dict(account_evidence.get("member_snapshot_evidence") or {})
        if not token or not allocation_id or not plan_id or len(snapshot_evidence) != 3:
            raise ValueError("activation cutover requires allocation, plan, and three snapshots")
        now = _now()
        with self.engine.begin() as connection:
            pending = connection.execute(
                select(strategy_events)
                .where(
                    strategy_events.c.strategy_version_id == version_id,
                    strategy_events.c.event_type == ACTIVATION_PENDING_EVENT,
                )
                .order_by(strategy_events.c.id.desc())
                .with_for_update()
            ).first()
            if pending is None or str(
                (pending.payload_json or {}).get("activation_token") or ""
            ) != token:
                raise ValueError("activation cutover token is not the current pending event")
            if token in self._terminal_activation_tokens(connection, version_id):
                return {
                    "strategy_version_id": version_id,
                    "activation_token": token,
                    "status": "already_terminal",
                }
            version = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .with_for_update()
            ).first()
            if (
                version is None
                or str(version.status) != "approved"
                or str(version.promotion_stage) != STAGE_RECOMMENDATION_ENABLED
            ):
                raise ValueError("pending strategy no longer has recommendation authority")
            allocation = connection.execute(
                select(strategy_allocations).where(
                    strategy_allocations.c.id == allocation_id,
                    strategy_allocations.c.status == "active",
                )
            ).first()
            if allocation is None:
                raise ValueError("activation allocation is not active")
            netting_plan = connection.execute(
                select(account_netting_plans.c.id).where(
                    account_netting_plans.c.id == plan_id,
                    account_netting_plans.c.account_id == allocation_id,
                )
            ).first()
            if netting_plan is None:
                raise ValueError("activation netting plan is not bound to the allocation")
            members = connection.execute(
                select(strategy_allocation_members).where(
                    strategy_allocation_members.c.allocation_id == allocation_id
                )
            ).all()
            member_ids = {str(member.strategy_version_id) for member in members}
            if len(member_ids) != 3 or version_id not in member_ids:
                raise ValueError("activation allocation is not the complete three-horizon set")
            for member in members:
                expected = snapshot_evidence.get(str(member.strategy_version_id))
                if not isinstance(expected, dict):
                    raise ValueError("activation evidence is missing a member snapshot")
                snapshot = connection.execute(
                    select(recommendation_snapshots)
                    .where(
                        recommendation_snapshots.c.portfolio_id
                        == member.recommendation_portfolio_id,
                        recommendation_snapshots.c.id == expected.get("snapshot_id"),
                        recommendation_snapshots.c.status == "succeeded",
                    )
                    .limit(1)
                ).first()
                if snapshot is None:
                    raise ValueError("activation member snapshot is not durable and succeeded")
            connection.execute(
                insert(strategy_events).values(
                    strategy_id=str(version.strategy_id),
                    strategy_version_id=version_id,
                    event_type=ACTIVATION_COMPLETED_EVENT,
                    actor=actor.strip(),
                    payload_json={
                        "activation_token": token,
                        "allocation_id": allocation_id,
                        "netting_plan_id": plan_id,
                        "member_snapshot_evidence": snapshot_evidence,
                        "account_evidence_sha256": _canonical_sha256(account_evidence),
                    },
                    created_at=now,
                )
            )
        return {
            "strategy_version_id": version_id,
            "activation_token": token,
            "status": "completed",
            "allocation_id": allocation_id,
            "netting_plan_id": plan_id,
        }

    def rollback_activation_cutover(
        self,
        version_id: str,
        *,
        activation_token: str,
        reason: str,
        actor: str = "system:auto-promotion",
    ) -> dict[str, Any]:
        """Atomically restore the prior strategy/account after cutover failure."""

        token = activation_token.strip()
        if not token or len(reason.strip()) < 10:
            raise ValueError("activation token and meaningful rollback reason are required")
        now = _now()
        with self.engine.begin() as connection:
            pending = connection.execute(
                select(strategy_events)
                .where(
                    strategy_events.c.strategy_version_id == version_id,
                    strategy_events.c.event_type == ACTIVATION_PENDING_EVENT,
                )
                .order_by(strategy_events.c.id.desc())
                .with_for_update()
            ).first()
            if pending is None or str(
                (pending.payload_json or {}).get("activation_token") or ""
            ) != token:
                raise ValueError("activation cutover token is not the current pending event")
            if token in self._terminal_activation_tokens(connection, version_id):
                return {
                    "strategy_version_id": version_id,
                    "activation_token": token,
                    "status": "already_terminal",
                }
            payload = dict(pending.payload_json or {})
            incumbents = list(payload.get("replaced_incumbents") or [])
            version = connection.execute(
                select(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .with_for_update()
            ).first()
            if version is None:
                raise KeyError(version_id)
            locked_incumbents: list[Any] = []
            for incumbent in incumbents:
                incumbent_id = str(incumbent.get("strategy_version_id") or "")
                incumbent_row = connection.execute(
                    select(strategy_versions)
                    .where(strategy_versions.c.id == incumbent_id)
                    .with_for_update()
                ).first()
                if (
                    incumbent_row is None
                    or str(incumbent_row.horizon_profile)
                    != str(version.horizon_profile)
                    or str(incumbent_row.promotion_stage)
                    != STAGE_RECOMMENDATION_ENABLED
                ):
                    raise ValueError(
                        "recorded incumbent is unavailable for an atomic rollback"
                    )
                locked_incumbents.append(incumbent_row)

            active_allocation = connection.execute(
                select(strategy_allocations)
                .join(
                    strategy_allocation_members,
                    strategy_allocation_members.c.allocation_id == strategy_allocations.c.id,
                )
                .where(
                    strategy_allocations.c.status == "active",
                    strategy_allocation_members.c.strategy_version_id == version_id,
                )
                .with_for_update()
            ).first()
            restored_allocation_id: str | None = None
            if active_allocation is not None:
                new_allocation_id = str(active_allocation.id)
                account_has_orders = connection.execute(
                    select(simulation_batches.c.id)
                    .join(
                        simulation_portfolios,
                        simulation_portfolios.c.id == simulation_batches.c.portfolio_id,
                    )
                    .where(
                        simulation_portfolios.c.source_type == "allocation",
                        simulation_portfolios.c.source_id == new_allocation_id,
                        simulation_batches.c.account_netting_plan_id.is_not(None),
                        simulation_batches.c.created_at >= pending.created_at,
                    )
                    .limit(1)
                ).first()
                if account_has_orders is not None:
                    raise ValueError(
                        "activation already created an account order plan; automatic rollback "
                        "is blocked and new risk must remain restricted"
                    )
                new_portfolios = connection.scalars(
                    select(strategy_allocation_members.c.recommendation_portfolio_id).where(
                        strategy_allocation_members.c.allocation_id == new_allocation_id,
                        strategy_allocation_members.c.recommendation_portfolio_id.is_not(None),
                    )
                ).all()
                connection.execute(
                    update(recommendation_portfolios)
                    .where(recommendation_portfolios.c.id.in_(new_portfolios))
                    .values(status="paused", risk_exposure_override=0.0, updated_at=now)
                )
                connection.execute(
                    update(simulation_portfolios)
                    .where(
                        simulation_portfolios.c.source_type == "allocation",
                        simulation_portfolios.c.source_id == new_allocation_id,
                    )
                    .values(status="paused", updated_at=now)
                )
                connection.execute(
                    update(strategy_allocations)
                    .where(strategy_allocations.c.id == new_allocation_id)
                    .values(status="paused", updated_at=now)
                )
                replaced_events = connection.execute(
                    select(strategy_allocation_events)
                    .where(
                        strategy_allocation_events.c.event_type == "allocation.replaced",
                        strategy_allocation_events.c.created_at >= pending.created_at,
                    )
                    .order_by(strategy_allocation_events.c.created_at.desc())
                ).all()
                prior = next(
                    (
                        event
                        for event in replaced_events
                        if str(
                            (event.details_json or {}).get("replacement_allocation_id")
                            or ""
                        )
                        == new_allocation_id
                    ),
                    None,
                )
                if prior is not None:
                    restored_allocation_id = str(prior.allocation_id)
                    old_portfolios = connection.scalars(
                        select(
                            strategy_allocation_members.c.recommendation_portfolio_id
                        ).where(
                            strategy_allocation_members.c.allocation_id
                            == restored_allocation_id,
                            strategy_allocation_members.c.recommendation_portfolio_id.is_not(
                                None
                            ),
                        )
                    ).all()
                    connection.execute(
                        update(recommendation_portfolios)
                        .where(recommendation_portfolios.c.id.in_(old_portfolios))
                        .values(status="active", risk_exposure_override=1.0, updated_at=now)
                    )
                    connection.execute(
                        update(simulation_portfolios)
                        .where(
                            simulation_portfolios.c.source_type == "allocation",
                            simulation_portfolios.c.source_id == restored_allocation_id,
                        )
                        .values(status="active", updated_at=now)
                    )
                    connection.execute(
                        update(strategy_allocations)
                        .where(strategy_allocations.c.id == restored_allocation_id)
                        .values(status="active", updated_at=now)
                    )

            # Order matters because the database permits only one approved
            # recommendation-enabled strategy per horizon.
            connection.execute(
                update(strategy_versions)
                .where(strategy_versions.c.id == version_id)
                .values(status="approved", promotion_stage=STAGE_PAPER)
            )
            connection.execute(
                update(strategy_promotion_stages)
                .where(strategy_promotion_stages.c.id == payload.get("promotion_stage_id"))
                .values(promoted_at=None)
            )
            for incumbent, incumbent_row in zip(
                incumbents, locked_incumbents, strict=True
            ):
                connection.execute(
                    update(strategy_versions)
                    .where(
                        strategy_versions.c.id == incumbent_row.id,
                        strategy_versions.c.horizon_profile == version.horizon_profile,
                    )
                    .values(
                        status=str(incumbent.get("status") or "approved"),
                        promotion_stage=str(
                            incumbent.get("promotion_stage")
                            or STAGE_RECOMMENDATION_ENABLED
                        ),
                    )
                )
            connection.execute(
                insert(strategy_events).values(
                    strategy_id=str(version.strategy_id),
                    strategy_version_id=version_id,
                    event_type=ACTIVATION_ROLLED_BACK_EVENT,
                    actor=actor.strip(),
                    payload_json={
                        "activation_token": token,
                        "reason": reason.strip(),
                        "restored_incumbents": incumbents,
                        "restored_allocation_id": restored_allocation_id,
                    },
                    created_at=now,
                )
            )
        return {
            "strategy_version_id": version_id,
            "activation_token": token,
            "status": "rolled_back",
            "restored_incumbents": incumbents,
            "restored_allocation_id": restored_allocation_id,
        }

    def auto_promote_if_ready(self, version_id: str) -> dict[str, Any]:
        """Promote only when immutable historical and forward evidence is complete.

        This is deliberately a thin, auditable system action around ``promote``;
        it does not reinterpret metrics, lower thresholds, or let research code
        write an active strategy.  The existing transaction remains the single
        authority for the version switch.
        """

        # Approval and paper-account creation use separate transactions so a
        # transient failure cannot roll a validated historical result back.
        # Reconcile that safe gap on every scheduler pass before reading the
        # forward gate. Both gate registration and stage/account creation are
        # idempotent; thresholds remain immutable once the first stage exists.
        self.prepare_paper_stage(version_id, actor="system:auto-promotion")
        evaluation = self.evaluate_forward_gate(version_id)
        if not evaluation["passed"]:
            return {
                "strategy_version_id": version_id,
                "promotion_stage": STAGE_PAPER,
                "promoted": False,
                "evaluation": evaluation,
            }
        result = self.promote(
            version_id,
            actor="system:auto-promotion",
            reason="All sealed historical, independent, and forward gates passed.",
        )
        return {**result, "promoted": True, "evaluation": evaluation}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_backtest_datasets(self, version_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            backtest = connection.execute(
                select(backtest_runs)
                .where(
                    backtest_runs.c.strategy_version_id == version_id,
                    backtest_runs.c.status == "succeeded",
                )
                .order_by(backtest_runs.c.created_at.desc())
                .limit(1)
            ).first()
        if backtest is None or not backtest.artifact_path:
            return None
        descriptor = Path(str(backtest.artifact_path)) / "datasets.json"
        if not descriptor.is_file():
            return None
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        daily = payload.get("daily")
        execution = payload.get("execution")
        if not isinstance(daily, dict) or not isinstance(execution, dict):
            return None
        return {"daily": daily, "execution": execution}

    def _collect_evidence(
        self, connection: Any, stage: Any, portfolio: Any, version: Any
    ) -> dict[str, Any]:
        portfolio_id = str(stage.simulation_portfolio_id)
        opened_at = stage.opened_at
        opened_date = opened_at.astimezone(_SHANGHAI).date()
        observed_at = _now()
        observed_date = observed_at.astimezone(_SHANGHAI).date()
        batches = connection.execute(
            select(
                simulation_batches.c.id,
                simulation_batches.c.status,
                simulation_batches.c.signal_date,
                simulation_batches.c.trade_date,
                simulation_batches.c.signal_at,
                simulation_batches.c.execution_not_before,
                simulation_batches.c.execution_contract_hash,
                simulation_batches.c.daily_dataset,
                simulation_batches.c.daily_dataset_identity_sha256,
                simulation_batches.c.daily_dataset_lineage_id,
                simulation_batches.c.execution_dataset,
                simulation_batches.c.execution_dataset_identity_sha256,
                simulation_batches.c.execution_dataset_lineage_id,
                simulation_batches.c.simulation_semantics_sha256,
                simulation_batches.c.source_snapshot_id,
                simulation_batches.c.target_payload_json,
                simulation_batches.c.summary_json,
                simulation_batches.c.created_at,
                simulation_batches.c.started_at,
                simulation_batches.c.finished_at,
            ).where(
                simulation_batches.c.portfolio_id == portfolio_id,
                simulation_batches.c.created_at >= opened_at,
                simulation_batches.c.created_at <= observed_at,
                simulation_batches.c.signal_date > opened_date,
                simulation_batches.c.signal_date <= observed_date,
                simulation_batches.c.trade_date <= observed_date,
            )
        ).all()
        formal_backtest_ids = {
            str(item)
            for item in connection.scalars(
                select(backtest_runs.c.id).where(
                    backtest_runs.c.strategy_version_id == str(version.id),
                    backtest_runs.c.status == "succeeded",
                    backtest_runs.c.is_legacy.is_(False),
                )
            )
        }
        governed: list[Any] = []
        ungoverned: list[Any] = []
        for batch in batches:
            payload = dict(batch.target_payload_json or {})
            plan = payload.get("governed_order_plan")
            source_snapshot = (
                plan.get("source_snapshot") if isinstance(plan, dict) else None
            )
            manifest_sha256 = (
                str(plan.get("manifest_sha256") or "")
                if isinstance(plan, dict)
                else ""
            )
            snapshot_id = (
                str(source_snapshot.get("id") or "")
                if isinstance(source_snapshot, dict)
                else ""
            )
            source_lineage_id = (
                str(source_snapshot.get("dataset_lineage_id") or "")
                if isinstance(source_snapshot, dict)
                else ""
            )
            formal_backtest_id = (
                str(plan.get("formal_backtest_id") or "")
                if isinstance(plan, dict)
                else ""
            )
            bindings = {
                "daily_dataset": str(batch.daily_dataset),
                "daily_dataset_identity_sha256": str(
                    batch.daily_dataset_identity_sha256
                ),
                "daily_dataset_lineage_id": str(batch.daily_dataset_lineage_id),
                "execution_dataset": str(batch.execution_dataset),
                "execution_dataset_identity_sha256": str(
                    batch.execution_dataset_identity_sha256
                ),
                "execution_dataset_lineage_id": str(
                    batch.execution_dataset_lineage_id
                ),
            }
            expected_semantics_sha256 = (
                SimulationStore._batch_simulation_semantics_sha256(
                    portfolio, bindings
                )
            )
            is_governed = (
                isinstance(plan, dict)
                and plan.get("format_version") == QLIB_ORDER_PLAN_FORMAT_VERSION
                and plan.get("promotion_stage_id") == str(stage.id)
                and plan.get("promotion_stage_opened_at") == opened_at.isoformat()
                and plan.get("execution_contract_hash")
                == str(portfolio.execution_contract_hash)
                and str(batch.execution_contract_hash)
                == str(portfolio.execution_contract_hash)
                and len(manifest_sha256) == 64
                and all(value in "0123456789abcdef" for value in manifest_sha256.lower())
                and len(snapshot_id) == 64
                and snapshot_id == str(batch.source_snapshot_id or "")
                and snapshot_id == str(batch.daily_dataset_identity_sha256)
                and isinstance(source_snapshot, dict)
                and snapshot_id
                == str(source_snapshot.get("dataset_identity_sha256") or "")
                and source_lineage_id == str(batch.daily_dataset_lineage_id)
                and source_lineage_id == str(portfolio.daily_dataset_lineage_id)
                and str(batch.execution_dataset_lineage_id)
                == str(portfolio.execution_dataset_lineage_id)
                and str(batch.simulation_semantics_sha256)
                == expected_semantics_sha256
                and formal_backtest_id in formal_backtest_ids
            )
            (governed if is_governed else ungoverned).append(batch)
        succeeded_candidates = [
            batch for batch in governed if str(batch.status) == "succeeded"
        ]
        version_config = getattr(version, "config_json", None)
        signal_frequency = str(
            getattr(version, "signal_frequency", None)
            or dict(version_config or {}).get("signal_frequency")
            or "day"
        ).lower()

        def _has_valid_lifecycle(batch: Any) -> bool:
            if batch.finished_at is None or batch.started_at is None:
                return False
            common_lifecycle_is_valid = (
                batch.finished_at >= opened_at
                and batch.finished_at <= observed_at
                and batch.started_at >= opened_at
                and batch.created_at <= batch.started_at
                and batch.started_at <= batch.finished_at
                and batch.trade_date > batch.signal_date
            )
            if not common_lifecycle_is_valid:
                return False
            plan = dict(batch.target_payload_json or {}).get(
                "governed_order_plan", {}
            )
            if not isinstance(plan, dict):
                return False
            # Mirror SimulationStore._validate_order_plan_timing exactly. A
            # daily signal is session-based, so adding fabricated intraday
            # timestamps is forbidden even when its execution dataset is
            # minute-frequency. Minute signals require the immutable timing
            # boundary to be carried by both the batch and order plan.
            if signal_frequency == "day":
                return (
                    batch.signal_at is None
                    and batch.execution_not_before is None
                    and plan.get("signal_at") is None
                    and plan.get("execution_not_before") is None
                )
            if (
                batch.signal_at is None
                or batch.execution_not_before is None
                or batch.signal_at.tzinfo is None
                or batch.execution_not_before.tzinfo is None
            ):
                return False
            return (
                batch.signal_at <= batch.created_at
                and batch.execution_not_before > batch.signal_at
                and batch.execution_not_before <= batch.started_at
                and batch.signal_at.astimezone(_SHANGHAI).date()
                == batch.signal_date
                and batch.execution_not_before.astimezone(_SHANGHAI).date()
                == batch.trade_date
                and plan.get("signal_at") == batch.signal_at.isoformat()
                and plan.get("execution_not_before")
                == batch.execution_not_before.isoformat()
            )

        succeeded = [
            batch for batch in succeeded_candidates if _has_valid_lifecycle(batch)
        ]
        invalid_lifecycle_batches = len(succeeded_candidates) - len(succeeded)
        succeeded_trade_dates = sorted({batch.trade_date for batch in succeeded})
        nav_stats = connection.execute(
            select(
                func.count(func.distinct(simulation_nav.c.trade_date)),
            ).where(
                simulation_nav.c.portfolio_id == portfolio_id,
                simulation_nav.c.created_at >= opened_at,
                simulation_nav.c.created_at <= observed_at,
                simulation_nav.c.trade_date > opened_date,
                simulation_nav.c.trade_date <= observed_date,
                simulation_nav.c.market_date == simulation_nav.c.trade_date,
                simulation_nav.c.performance_certified.is_(True),
                simulation_nav.c.has_stale_prices.is_(False),
            )
        ).one()
        legacy_nav_span = connection.execute(
            select(
                func.min(simulation_nav.c.trade_date),
                func.max(simulation_nav.c.trade_date),
            ).where(
                simulation_nav.c.portfolio_id == portfolio_id,
                simulation_nav.c.created_at >= opened_at,
                simulation_nav.c.created_at <= observed_at,
                simulation_nav.c.trade_date > opened_date,
                simulation_nav.c.trade_date <= observed_date,
                simulation_nav.c.trade_date.in_(succeeded_trade_dates or [opened_date]),
                simulation_nav.c.market_date == simulation_nav.c.trade_date,
                simulation_nav.c.performance_certified.is_(True),
                simulation_nav.c.has_stale_prices.is_(False),
            )
        ).one()
        calendar_days = 0
        if legacy_nav_span[0] is not None and legacy_nav_span[1] is not None:
            calendar_days = (legacy_nav_span[1] - legacy_nav_span[0]).days + 1
        trading_days = int(nav_stats[0] or 0)
        seen_review_event_ids: set[str] = set()
        review_periods: set[str] = set()
        financial_report_periods: set[str] = set()
        for batch in succeeded:
            plan = dict(batch.target_payload_json or {}).get("governed_order_plan")
            review = plan.get("horizon_review") if isinstance(plan, dict) else None
            if not isinstance(review, dict):
                continue
            try:
                review_payload = validate_horizon_review_evidence(
                    review,
                    strategy_version_id=str(version.id),
                    horizon_profile=str(version.horizon_profile),
                    signal_date=batch.signal_date,
                    dataset_identity_sha256=str(batch.source_snapshot_id or ""),
                    stage_opened_at=opened_at,
                    observed_at=observed_at,
                )
            except (KeyError, TypeError, ValueError):
                continue
            review_id = str(review_payload["event_id"])
            if review_id in seen_review_event_ids:
                continue
            review_type = str(review_payload["review_type"])
            completed_at = datetime.fromisoformat(str(review_payload["completed_at"]))
            if review_type == "financial_report_review":
                financial_report_periods.update(
                    str(period).strip()
                    for period in (
                        review_payload.get("report_periods")
                        or [review_payload["report_period"]]
                    )
                )
            seen_review_event_ids.add(review_id)
            review_periods.add(
                _review_period_key(str(version.horizon_profile), completed_at)
            )
        reconciled = 0
        for batch in succeeded:
            conservation = (batch.summary_json or {}).get("conservation") or {}
            difference = conservation.get("cash_difference")
            if difference is not None and abs(float(difference)) <= _RECONCILIATION_TOLERANCE:
                reconciled += 1
        all_fill_rows = connection.execute(
            select(
                simulation_fills.c.id,
                simulation_fills.c.order_id,
                simulation_fills.c.batch_id,
                simulation_fills.c.instrument,
                simulation_fills.c.side,
                simulation_fills.c.position_side,
                simulation_fills.c.quantity,
                simulation_fills.c.executed_at,
                simulation_batches.c.trade_date.label("batch_trade_date"),
                simulation_batches.c.execution_not_before.label(
                    "batch_execution_not_before"
                ),
                simulation_batches.c.finished_at.label("batch_finished_at"),
                simulation_orders.c.not_before.label("order_not_before"),
                simulation_orders.c.not_after.label("order_not_after"),
                simulation_orders.c.expires_at.label("order_expires_at"),
                simulation_orders.c.batch_id.label("order_batch_id"),
                simulation_orders.c.portfolio_id.label("order_portfolio_id"),
                simulation_orders.c.instrument.label("order_instrument"),
                simulation_orders.c.side.label("order_side"),
                simulation_orders.c.position_side.label("order_position_side"),
                simulation_orders.c.requested_quantity.label(
                    "order_requested_quantity"
                ),
                simulation_orders.c.filled_quantity.label("order_filled_quantity"),
            )
            .join(
                simulation_batches,
                simulation_batches.c.id == simulation_fills.c.batch_id,
            )
            .join(
                simulation_orders,
                simulation_orders.c.id == simulation_fills.c.order_id,
            )
            .where(
                simulation_fills.c.batch_id.in_(
                    [str(batch.id) for batch in succeeded] or [""]
                )
            )
            .order_by(
                simulation_batches.c.trade_date,
                simulation_fills.c.executed_at,
                simulation_fills.c.id,
            )
        ).all()
        eligible_fill_rows: list[Any] = []
        invalid_fill_contract_rows = 0
        filled_quantity_by_order: dict[str, int] = {}
        for fill in all_fill_rows:
            order_id = str(fill.order_id)
            filled_quantity_by_order[order_id] = (
                filled_quantity_by_order.get(order_id, 0) + int(fill.quantity)
            )
        for fill in all_fill_rows:
            executed_at = fill.executed_at
            order_window_ends = [
                value
                for value in (fill.order_not_after, fill.order_expires_at)
                if value is not None
            ]
            order_window_end = min(order_window_ends) if order_window_ends else None
            order_window_is_valid = (
                fill.order_not_before is None
                or (
                    isinstance(executed_at, datetime)
                    and fill.order_not_before <= executed_at
                )
            ) and (
                order_window_end is None
                or (
                    isinstance(executed_at, datetime)
                    and executed_at <= order_window_end
                )
            )
            minute_boundary_is_valid = signal_frequency == "day" or (
                fill.batch_execution_not_before is not None
                and isinstance(executed_at, datetime)
                and fill.batch_execution_not_before <= executed_at
            )
            order_identity_is_valid = (
                str(fill.order_batch_id) == str(fill.batch_id)
                and str(fill.order_portfolio_id) == portfolio_id
                and str(fill.order_instrument) == str(fill.instrument)
                and str(fill.order_side) == str(fill.side)
                and str(fill.order_position_side) == str(fill.position_side)
                and int(fill.quantity) > 0
                and 0 < int(fill.order_filled_quantity)
                <= int(fill.order_requested_quantity)
                and filled_quantity_by_order[str(fill.order_id)]
                == int(fill.order_filled_quantity)
            )
            valid_timestamp = (
                isinstance(executed_at, datetime)
                and executed_at.tzinfo is not None
                and opened_at <= executed_at <= observed_at
                # ``executed_at`` is the immutable market-bar time. The batch
                # timestamps are wall-clock processing times written after a
                # historical or post-close match, so execution may correctly
                # precede ``started_at`` but can never follow ``finished_at``.
                and executed_at <= fill.batch_finished_at
                and executed_at.astimezone(_SHANGHAI).date()
                == fill.batch_trade_date
                and order_window_is_valid
                and minute_boundary_is_valid
                and order_identity_is_valid
            )
            if str(fill.position_side) != "long" or not valid_timestamp:
                invalid_fill_contract_rows += 1
                continue
            eligible_fill_rows.append(fill)
        eligible_fill_ids = [str(fill.id) for fill in eligible_fill_rows]
        fill_stats = connection.execute(
            select(
                func.coalesce(func.sum(simulation_fills.c.fee), 0.0),
                func.coalesce(func.sum(simulation_fills.c.gross_value), 0.0),
            ).where(simulation_fills.c.id.in_(eligible_fill_ids or [""]))
        ).one()
        sell_batches = connection.execute(
            select(simulation_fills.c.batch_id)
            .where(
                simulation_fills.c.id.in_(eligible_fill_ids or [""]),
                simulation_fills.c.side == "sell",
            )
            .distinct()
        ).all()
        closed_round_trips, invalid_sequence_fills = _count_closed_round_trips(
            eligible_fill_rows
        )
        invalid_round_trip_fills = (
            invalid_fill_contract_rows + invalid_sequence_fills
        )
        decision_dates = {batch.signal_date for batch in succeeded}
        duplicate_decision_batches = max(0, len(succeeded) - len(decision_dates))
        total_fees = float(fill_stats[0])
        total_value = float(fill_stats[1])
        realized_rate = total_fees / total_value if total_value > 0 else 0.0
        scheduled_rate = self._scheduled_one_side_rate(str(portfolio.cost_schedule_version))
        return {
            "stage_opened_at": opened_at.isoformat(),
            "forward_signal_after": opened_date.isoformat(),
            "forward_calendar_days": calendar_days,
            "forward_trading_days": trading_days,
            "decision_batches": len(decision_dates),
            "duplicate_decision_batches": duplicate_decision_batches,
            "total_batches": len(governed),
            "ungoverned_batches": len(ungoverned),
            "invalid_lifecycle_batches": invalid_lifecycle_batches,
            "completed_cycles": len(sell_batches),
            "closed_round_trips": closed_round_trips,
            "invalid_round_trip_fills": invalid_round_trip_fills,
            "invalid_fill_contract_rows": invalid_fill_contract_rows,
            "review_events": len(review_periods),
            "review_event_unit": (
                "iso_week"
                if str(version.horizon_profile) == SWING_1_6M
                else "calendar_month"
                if str(version.horizon_profile) == LONG_1_3Y
                else "trading_date"
            ),
            "review_event_periods": sorted(review_periods),
            "financial_report_reviews": len(financial_report_periods),
            "financial_report_periods": sorted(financial_report_periods),
            "data_completeness": (
                (len(succeeded) / len(governed)) if governed else 0.0
            ),
            "reconciliation_rate": (reconciled / len(succeeded)) if succeeded else 0.0,
            "realized_cost_rate": realized_rate,
            "scheduled_cost_rate": scheduled_rate,
            "cost_deviation": abs(realized_rate - scheduled_rate),
        }

    @staticmethod
    def _scheduled_one_side_rate(cost_schedule_version: str) -> float:
        if cost_schedule_version == COST_SCHEDULE_VERSION:
            config = CostModelConfig()
        else:
            config = next(
                (
                    item
                    for item in CN_COST_SCHEDULE_VERSIONS
                    if item.version == cost_schedule_version
                ),
                None,
            )
        if config is None:
            raise ValueError(f"unknown cost schedule version: {cost_schedule_version}")
        participation = config.max_volume_participation
        buy = config.estimate(
            side="buy",
            gross_value=_REFERENCE_ORDER_VALUE,
            participation=participation,
        )
        sell = config.estimate(
            side="sell",
            gross_value=_REFERENCE_ORDER_VALUE,
            participation=participation,
        )
        return (buy + sell) / (2.0 * _REFERENCE_ORDER_VALUE)

    def _event(self, version_id: str, *, event_type: str, actor: str, payload: dict) -> None:
        with self.engine.begin() as connection:
            version = connection.execute(
                select(strategy_versions).where(strategy_versions.c.id == version_id)
            ).first()
            connection.execute(
                insert(strategy_events).values(
                    strategy_id=str(version.strategy_id) if version else "",
                    strategy_version_id=version_id,
                    event_type=event_type,
                    actor=actor.strip(),
                    payload_json=payload,
                    created_at=_now(),
                )
            )

    def record_paper_stage_failure(self, version_id: str, *, actor: str, error: str) -> None:
        """Best-effort failure trace when auto paper opening hits a system error."""

        try:
            self._event(
                version_id,
                event_type="strategy.paper_stage_open_failed",
                actor=actor,
                payload={"error": error},
            )
        except Exception:  # the approval itself is already committed; tracing is best-effort
            pass

    @staticmethod
    def _stage_dict(row: Any) -> dict[str, Any]:
        return {
            "id": str(row.id),
            "strategy_version_id": str(row.strategy_version_id),
            "stage_index": int(row.stage_index),
            "simulation_portfolio_id": (
                str(row.simulation_portfolio_id) if row.simulation_portfolio_id else None
            ),
            "status": str(row.status),
            "source_contract_hash": (
                str(row.source_contract_hash) if row.source_contract_hash else None
            ),
            "initial_cash": float(row.initial_cash) if row.initial_cash is not None else None,
            "opened_at": row.opened_at.isoformat(),
            "frozen_at": row.frozen_at.isoformat() if row.frozen_at else None,
            "freeze_reason": str(row.freeze_reason) if row.freeze_reason else None,
            "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
        }
