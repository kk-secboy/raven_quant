"""Point-in-time availability policies and recoverability levels (design draft 3.3).

Every dataset that can change historical research results declares:

- an *availability policy* (when a market participant could first know a row):
  ``strictly_after_announcement_date`` for financial reports,
  ``same_trade_date_after_close`` for trade-date-derived market fields, and
  ``effective_date_with_lag(days=N)`` for index/industry metadata whose real
  publication lag has no machine-readable source;
- a *recoverability level* (``native_history`` / ``reconstructed`` /
  ``current_only`` / ``unavailable``) marking whether historical versions can
  be restored. Only the first two levels may feed formal evidence features.

`filter_available` is the read-side guard: it applies the registered policy to
a frame for a given decision timestamp and fails closed on unregistered
datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from .reference_data import AUDITED_REFERENCE_DATASETS

AVAILABILITY_POLICY_VERSION = 1

STRICTLY_AFTER_ANNOUNCEMENT_DATE = "strictly_after_announcement_date"
SAME_TRADE_DATE_AFTER_CLOSE = "same_trade_date_after_close"
EFFECTIVE_DATE_WITH_LAG = "effective_date_with_lag"

NATIVE_HISTORY = "native_history"
RECONSTRUCTED = "reconstructed"
CURRENT_ONLY = "current_only"
UNAVAILABLE = "unavailable"
EVIDENCE_RECOVERABILITY_LEVELS = frozenset({NATIVE_HISTORY, RECONSTRUCTED})

# Versioned configuration for the conservative publication lag applied to
# index/industry metadata on the consumer side. The true publication lag of
# index weights and industry membership changes has no machine-readable source,
# so this is an approximation: five calendar days, erring on the late side.
# Changing this value changes evidence semantics and must bump
# AVAILABILITY_LAG_CONFIG_VERSION with a note in the change record.
AVAILABILITY_LAG_CONFIG_VERSION = 1
METADATA_AVAILABILITY_LAG_DAYS = 5

_METADATA_LAG_NOTE = (
    "approximation: the true publication lag has no source data; "
    "a conservative calendar-day lag is applied instead"
)


@dataclass(frozen=True, slots=True)
class AvailabilityPolicy:
    kind: str
    date_columns: tuple[str, ...]
    lag_days: int = 0
    interval: bool = False
    note: str = ""

    @property
    def contract_label(self) -> str:
        if self.kind == EFFECTIVE_DATE_WITH_LAG:
            return f"{EFFECTIVE_DATE_WITH_LAG}(days={self.lag_days})"
        return self.kind


AVAILABILITY_POLICIES: dict[str, AvailabilityPolicy] = {
    # Financial reports are usable strictly after their announcement date.
    "income": AvailabilityPolicy(STRICTLY_AFTER_ANNOUNCEMENT_DATE, ("ann_date",)),
    "balancesheet": AvailabilityPolicy(STRICTLY_AFTER_ANNOUNCEMENT_DATE, ("ann_date",)),
    "cashflow": AvailabilityPolicy(STRICTLY_AFTER_ANNOUNCEMENT_DATE, ("ann_date",)),
    "fina_indicator": AvailabilityPolicy(STRICTLY_AFTER_ANNOUNCEMENT_DATE, ("ann_date",)),
    "forecast": AvailabilityPolicy(STRICTLY_AFTER_ANNOUNCEMENT_DATE, ("ann_date",)),
    "express": AvailabilityPolicy(STRICTLY_AFTER_ANNOUNCEMENT_DATE, ("ann_date",)),
    # Trade-date-derived market fields are known after that session closes.
    "daily": AvailabilityPolicy(SAME_TRADE_DATE_AFTER_CLOSE, ("trade_date",)),
    "adj_factor": AvailabilityPolicy(SAME_TRADE_DATE_AFTER_CLOSE, ("trade_date",)),
    "daily_basic": AvailabilityPolicy(SAME_TRADE_DATE_AFTER_CLOSE, ("trade_date",)),
    "index_dailybasic": AvailabilityPolicy(SAME_TRADE_DATE_AFTER_CLOSE, ("trade_date",)),
    "stk_limit": AvailabilityPolicy(SAME_TRADE_DATE_AFTER_CLOSE, ("trade_date",)),
    "index_daily": AvailabilityPolicy(SAME_TRADE_DATE_AFTER_CLOSE, ("trade_date",)),
    # Index weights and industry membership: effective dates are known, but the
    # publication lag is not, so a conservative lag is applied (see note above).
    "index_weight": AvailabilityPolicy(
        EFFECTIVE_DATE_WITH_LAG,
        ("trade_date",),
        lag_days=METADATA_AVAILABILITY_LAG_DAYS,
        note=_METADATA_LAG_NOTE,
    ),
    "index_member_all": AvailabilityPolicy(
        EFFECTIVE_DATE_WITH_LAG,
        ("in_date", "out_date"),
        lag_days=METADATA_AVAILABILITY_LAG_DAYS,
        interval=True,
        note=_METADATA_LAG_NOTE,
    ),
}

# Recoverability level per dataset (design draft 3.3). Datasets not listed are
# treated as "unavailable" by recoverability_level: fail-safe labeling.
RECOVERABILITY_LEVELS: dict[str, str] = {
    # Trade-date series re-pullable for any historical session: the provider
    # keeps the original rows keyed by business date and does not rewrite them.
    "daily": NATIVE_HISTORY,
    "adj_factor": NATIVE_HISTORY,
    "daily_basic": NATIVE_HISTORY,
    "index_dailybasic": NATIVE_HISTORY,
    "index_daily": NATIVE_HISTORY,
    "index_global": NATIVE_HISTORY,
    "trade_cal": NATIVE_HISTORY,
    "stk_limit": NATIVE_HISTORY,
    "suspend_d": NATIVE_HISTORY,
    "limit_list_d": NATIVE_HISTORY,
    "stk_premarket": NATIVE_HISTORY,
    "stk_auction_o": NATIVE_HISTORY,
    "stk_auction_c": NATIVE_HISTORY,
    "moneyflow": NATIVE_HISTORY,
    "margin_detail": NATIVE_HISTORY,
    "hsgt_top10": NATIVE_HISTORY,
    "top_list": NATIVE_HISTORY,
    "top_inst": NATIVE_HISTORY,
    "fund_daily": NATIVE_HISTORY,
    "fund_adj": NATIVE_HISTORY,
    "fx_daily": NATIVE_HISTORY,
    "fut_daily": NATIVE_HISTORY,
    "shibor": NATIVE_HISTORY,
    "shibor_lpr": NATIVE_HISTORY,
    "us_tycr": NATIVE_HISTORY,
    # Monthly constituent weights stay re-pullable for historical months.
    "index_weight": NATIVE_HISTORY,
    # Financial reports keep announcement-versioned rows (ann_date plus
    # f_ann_date/update_flag revision markers), so what was knowable at each
    # point in time is natively retained upstream.
    "income": NATIVE_HISTORY,
    "balancesheet": NATIVE_HISTORY,
    "cashflow": NATIVE_HISTORY,
    "fina_indicator": NATIVE_HISTORY,
    "forecast": NATIVE_HISTORY,
    "express": NATIVE_HISTORY,
    # Announcement/event series keyed by announcement or period date.
    "anns_d": NATIVE_HISTORY,
    "disclosure_date": NATIVE_HISTORY,
    "namechange": NATIVE_HISTORY,
    "dividend": NATIVE_HISTORY,
    "repurchase": NATIVE_HISTORY,
    "share_float": NATIVE_HISTORY,
    "pledge_stat": NATIVE_HISTORY,
    "pledge_detail": NATIVE_HISTORY,
    "stk_holdertrade": NATIVE_HISTORY,
    # Historical membership is reconstructed from the in_date/out_date
    # intervals carried by the current full dump; the intervals themselves are
    # the reconstruction source, not a native version history.
    "index_member_all": RECONSTRUCTED,
    # As-of masters: the provider exposes only the latest revision. Pulls are
    # versioned going forward, but history before the first pull is
    # unrecoverable (the 2026-07-15 audit in reference_data.py covers the
    # remaining 29 datasets of this kind).
    "stock_basic": CURRENT_ONLY,
    "index_classify": CURRENT_ONLY,
    **{dataset: CURRENT_ONLY for dataset in AUDITED_REFERENCE_DATASETS},
}


def availability_policy(dataset: str) -> AvailabilityPolicy | None:
    return AVAILABILITY_POLICIES.get(dataset)


def availability_contract_label(dataset: str) -> str | None:
    policy = AVAILABILITY_POLICIES.get(dataset)
    return policy.contract_label if policy else None


def recoverability_level(dataset: str) -> str:
    return RECOVERABILITY_LEVELS.get(dataset, UNAVAILABLE)


class AvailabilityPolicyError(RuntimeError):
    """Raised when a dataset without a declared availability policy is read."""


def _naive_cutoff(decision_at: Any) -> pd.Timestamp:
    cutoff = pd.Timestamp(decision_at)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    return cutoff.normalize()


def _normalized_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def filter_available(
    dataset: str,
    frame: pd.DataFrame,
    decision_at: Any,
    *,
    date_column: str | None = None,
) -> pd.DataFrame:
    """Return the rows of ``frame`` knowable at ``decision_at`` per the registry.

    Fail-closed: datasets without a declared availability policy and frames
    missing the policy date columns raise AvailabilityPolicyError. Row dates
    that cannot be parsed are dropped rather than trusted. ``decision_at`` is
    normalized to its calendar date; policies compare whole dates.
    """

    policy = AVAILABILITY_POLICIES.get(dataset)
    if policy is None:
        raise AvailabilityPolicyError(
            f"dataset {dataset!r} has no declared availability policy; "
            "refusing to read it for point-in-time evidence"
        )
    cutoff = _naive_cutoff(decision_at)
    if policy.kind == EFFECTIVE_DATE_WITH_LAG:
        cutoff = cutoff - timedelta(days=policy.lag_days)

    if policy.interval:
        start_column, end_column = policy.date_columns
        missing = [column for column in policy.date_columns if column not in frame.columns]
        if missing:
            raise AvailabilityPolicyError(
                f"dataset {dataset!r} frame lacks availability columns: {missing}"
            )
        in_dates = _normalized_dates(frame[start_column])
        out_dates = _normalized_dates(frame[end_column])
        active = (in_dates <= cutoff) & (out_dates.isna() | (out_dates >= cutoff))
        return frame.loc[active].copy()

    column = date_column or next(
        (name for name in (*policy.date_columns, "datetime") if name in frame.columns),
        None,
    )
    if column is None:
        raise AvailabilityPolicyError(
            f"dataset {dataset!r} frame lacks availability date column "
            f"{policy.date_columns}"
        )
    dates = _normalized_dates(frame[column])
    if policy.kind == STRICTLY_AFTER_ANNOUNCEMENT_DATE:
        available = dates < cutoff
    elif policy.kind in {SAME_TRADE_DATE_AFTER_CLOSE, EFFECTIVE_DATE_WITH_LAG}:
        available = dates <= cutoff
    else:  # pragma: no cover - registry invariant
        raise AvailabilityPolicyError(
            f"dataset {dataset!r} has an unsupported availability policy: {policy.kind}"
        )
    return frame.loc[available].copy()
