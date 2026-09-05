from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ELIGIBILITY_CONTRACT_VERSION = "cn-stock-etf-point-in-time-eligibility-v3"
STANDARD_AUDIT_OPINIONS = frozenset(
    {
        "standard_unqualified",
        "unqualified",
        "标准无保留意见",
        "无保留意见",
    }
)

INSTRUMENT_RISK_STATES = frozenset(
    {"normal", "watch", "restricted", "reduce", "exit"}
)
_RISK_STATE_SEVERITY = {
    "normal": 0,
    "watch": 1,
    "restricted": 2,
    "reduce": 3,
    "exit": 4,
}
_SEVERE_AUDIT_OPINIONS = frozenset(
    {
        "adverse",
        "adverse_opinion",
        "disclaimer",
        "disclaimer_of_opinion",
        "unable_to_express",
        "no_opinion",
        "否定意见",
        "无法表示意见",
        "拒绝表示意见",
    }
)


@dataclass(frozen=True)
class EligibilityPolicy:
    min_listing_trading_days: int = 60
    min_average_daily_amount: float = 500_000_000.0
    liquidity_lookback_days: int = 20
    require_regulatory_events: bool = False


def build_point_in_time_eligibility(
    *,
    market: pd.DataFrame,
    listings: pd.DataFrame,
    st_intervals: pd.DataFrame,
    suspensions: pd.DataFrame,
    financials: pd.DataFrame,
    audits: pd.DataFrame,
    regulatory_events: pd.DataFrame | None,
    policy: EligibilityPolicy | None = None,
    trading_calendar: pd.Index | pd.Series | list[Any] | tuple[Any, ...] | None = None,
) -> pd.DataFrame:
    """Build one fail-closed eligibility row per observed instrument/session.

    ``market.asset_type`` is optional for compatibility and defaults to
    ``stock``.  ETF rows follow the same listing, liquidity, suspension and
    regulatory gates, while stock-only ST, shareholder-equity and audit gates
    are deliberately not applied to funds that do not publish company
    financial statements.
    """

    rules = policy or EligibilityPolicy()
    if rules.min_listing_trading_days < 1 or rules.liquidity_lookback_days < 2:
        raise ValueError("eligibility listing and liquidity windows are invalid")
    base = _required_frame(
        market,
        {"datetime", "instrument", "amount", "paused"},
        "market",
    )
    base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce").dt.normalize()
    base["instrument"] = base["instrument"].astype(str).str.upper()
    if "asset_type" not in base.columns:
        base["asset_type"] = "stock"
    base["asset_type"] = base["asset_type"].astype(str).str.strip().str.lower()
    unknown_asset_types = sorted(set(base["asset_type"]) - {"stock", "etf"})
    if unknown_asset_types:
        raise ValueError(
            "eligibility market has unsupported asset types: "
            + ", ".join(unknown_asset_types)
        )
    base["amount"] = pd.to_numeric(base["amount"], errors="coerce")
    base["paused"] = pd.to_numeric(base["paused"], errors="coerce").fillna(1).gt(0)
    base = base.dropna(subset=["datetime", "instrument"])
    if base.duplicated(["datetime", "instrument"]).any():
        raise ValueError("eligibility market observations are duplicated")
    base.sort_values(["instrument", "datetime"], inplace=True)
    base["average_daily_amount_20d"] = (
        base.groupby("instrument", sort=False)["amount"]
        .rolling(rules.liquidity_lookback_days, min_periods=rules.liquidity_lookback_days)
        .mean()
        .reset_index(level=0, drop=True)
    )

    listing = _required_frame(
        listings,
        {"instrument", "list_date", "delist_date"},
        "listing",
    )
    listing["instrument"] = listing["instrument"].astype(str).str.upper()
    listing["list_date"] = pd.to_datetime(listing["list_date"], errors="coerce").dt.normalize()
    listing["delist_date"] = pd.to_datetime(
        listing["delist_date"], errors="coerce"
    ).dt.normalize()
    if listing["instrument"].duplicated().any() or listing["list_date"].isna().any():
        raise ValueError("listing metadata is duplicated or missing list dates")
    base = base.merge(listing, on="instrument", how="left", validate="many_to_one")
    observed_calendar = pd.DatetimeIndex(base["datetime"].unique()).sort_values()
    if trading_calendar is None:
        calendar = observed_calendar
    else:
        normalized_calendar = pd.to_datetime(
            pd.Series(list(trading_calendar)), errors="coerce"
        ).dt.normalize()
        calendar = pd.DatetimeIndex(normalized_calendar.dropna().unique()).sort_values()
        if calendar.empty:
            raise ValueError("eligibility trading calendar has no valid dates")
        if not observed_calendar.isin(calendar).all():
            raise ValueError("eligibility market dates fall outside the trading calendar")
    # Count calendar positions with vectorized binary searches.  Expanding a
    # calendar mask per market row is O(rows * trading_days), which becomes
    # tens of billions of comparisons for a full A-share history.
    current_positions = calendar.searchsorted(
        pd.DatetimeIndex(base["datetime"]), side="right"
    )
    listed = pd.to_datetime(base["list_date"], errors="coerce")
    listed_positions = calendar.searchsorted(
        pd.DatetimeIndex(listed.fillna(pd.Timestamp.max.normalize())), side="left"
    )
    base["listing_trading_days"] = np.where(
        listed.notna(),
        np.maximum(current_positions - listed_positions, 0),
        0,
    )
    base["normal_listing_status"] = base["list_date"].notna() & (
        base["delist_date"].isna() | (base["datetime"] < base["delist_date"])
    )
    base["delisted"] = base["delist_date"].notna() & (
        base["datetime"] >= base["delist_date"]
    )

    st = _normalize_intervals(st_intervals, value_column="is_st", label="ST")
    base["is_st"] = _interval_flags(base, st, value_column="is_st")
    base.loc[base["asset_type"].eq("etf"), "is_st"] = False
    suspension = _required_frame(
        suspensions,
        {"datetime", "instrument", "suspended"},
        "suspension",
    )
    suspension["datetime"] = pd.to_datetime(
        suspension["datetime"], errors="coerce"
    ).dt.normalize()
    suspension["instrument"] = suspension["instrument"].astype(str).str.upper()
    suspension["suspended"] = suspension["suspended"].fillna(True).astype(bool)
    suspension = suspension.drop_duplicates(["datetime", "instrument"], keep="last")
    base = base.merge(
        suspension,
        on=["datetime", "instrument"],
        how="left",
        validate="one_to_one",
    )
    explicit_suspension = base["suspended"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    base["suspended"] = base["paused"] | explicit_suspension

    base = _asof_disclosure(
        base,
        financials,
        value_columns=["equity"],
        label="financial",
    )
    base = _asof_disclosure(
        base,
        audits,
        value_columns=["audit_opinion"],
        label="audit",
    )
    base["positive_equity"] = pd.to_numeric(base["equity"], errors="coerce").gt(0)
    base["standard_audit_opinion"] = base["audit_opinion"].isin(STANDARD_AUDIT_OPINIONS)
    base["financial_gate_required"] = base["asset_type"].eq("stock")

    regulatory_available = regulatory_events is not None
    base["regulatory_data_available"] = regulatory_available
    base["major_violation"] = False
    if regulatory_events is not None:
        events = _required_frame(
            regulatory_events,
            {"instrument", "event_date", "known_date", "major"},
            "regulatory event",
        )
        events["instrument"] = events["instrument"].astype(str).str.upper()
        events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.normalize()
        events["known_date"] = pd.to_datetime(events["known_date"], errors="coerce").dt.normalize()
        if events[["event_date", "known_date"]].isna().any().any() or (
            events["known_date"] < events["event_date"]
        ).any():
            raise ValueError("regulatory events have invalid occurrence/knowledge dates")
        normalized_major = events["major"].map(_strict_boolean)
        if normalized_major.isna().any():
            raise ValueError("regulatory event major flags must be boolean")
        major = events.loc[normalized_major.astype(bool)]
        first_known_date = major.groupby("instrument", sort=False)["known_date"].min()
        instrument_known_date = base["instrument"].map(first_known_date)
        base["major_violation"] = instrument_known_date.notna() & base["datetime"].ge(
            instrument_known_date
        )

    checks = {
        "new_listing": base["listing_trading_days"] < rules.min_listing_trading_days,
        "st": base["is_st"],
        "suspended": base["suspended"],
        "abnormal_listing": ~base["normal_listing_status"],
        "negative_or_missing_equity": (
            base["financial_gate_required"] & ~base["positive_equity"]
        ),
        "nonstandard_or_missing_audit": (
            base["financial_gate_required"] & ~base["standard_audit_opinion"]
        ),
        "insufficient_liquidity": base["average_daily_amount_20d"].fillna(0).lt(
            rules.min_average_daily_amount
        ),
        "major_violation": base["major_violation"],
        "regulatory_data_missing": (
            pd.Series(True, index=base.index)
            if rules.require_regulatory_events and not regulatory_available
            else pd.Series(False, index=base.index)
        ),
    }
    check_names = list(checks)
    reason_mask = np.zeros(len(base), dtype=np.uint16)
    for bit, name in enumerate(check_names):
        reason_mask |= checks[name].to_numpy(dtype=np.uint16) * np.uint16(1 << bit)
    reason_lookup = np.asarray(
        [
            json.dumps(
                [name for bit, name in enumerate(check_names) if mask & (1 << bit)]
            )
            for mask in range(1 << len(check_names))
        ],
        dtype=object,
    )
    base["eligible"] = reason_mask == 0
    base["reasons"] = reason_lookup[reason_mask]
    base["contract_version"] = ELIGIBILITY_CONTRACT_VERSION
    columns = [
        "datetime",
        "instrument",
        "asset_type",
        "eligible",
        "reasons",
        "listing_trading_days",
        "is_st",
        "suspended",
        "delisted",
        "normal_listing_status",
        "average_daily_amount_20d",
        "equity",
        "financial_announcement_date",
        "audit_opinion",
        "audit_announcement_date",
        "financial_gate_required",
        "regulatory_data_available",
        "major_violation",
        "contract_version",
    ]
    return base[columns].sort_values(["datetime", "instrument"]).reset_index(drop=True)


class PreparedPointInTimeRiskStates:
    """Reuse validated eligibility history without scanning it on every trading day.

    Each query selects at most one existing row per requested instrument and
    delegates the risk rules to the unchanged projection below. Binary searches
    are independent of query order, including restarts and validation windows.
    The instance owns its prepared frame; no cache is shared between datasets.
    """

    def __init__(self, values: pd.DataFrame) -> None:
        required = {
            "datetime", "instrument", "eligible", "reasons", "is_st", "suspended",
            "delisted", "normal_listing_status", "equity", "audit_opinion",
            "financial_gate_required", "regulatory_data_available", "major_violation",
            "contract_version",
        }
        source = _required_frame(values, required, "eligibility risk projection")
        source["datetime"] = pd.to_datetime(source["datetime"], errors="coerce").dt.normalize()
        source["instrument"] = source["instrument"].astype(str).str.upper()
        if source[["datetime", "instrument"]].isna().any().any():
            raise ValueError("eligibility risk projection has invalid dates or instruments")
        if source.duplicated(["datetime", "instrument"]).any():
            raise ValueError("eligibility risk projection observations are duplicated")
        if source["contract_version"].ne(ELIGIBILITY_CONTRACT_VERSION).any():
            raise ValueError("eligibility risk projection contract is obsolete")
        self._source = source.sort_values(["instrument", "datetime"], ignore_index=True)
        dates = pd.DatetimeIndex(self._source["datetime"])
        counts = self._source.groupby("instrument", sort=False).size()
        self._history: dict[str, tuple[pd.DatetimeIndex, int]] = {}
        start = 0
        for instrument, count in counts.items():
            stop = start + int(count)
            self._history[str(instrument)] = (dates[start:stop], start)
            start = stop

    def project(
        self, *, as_of: Any, instruments: Iterable[str] | None = None
    ) -> pd.DataFrame:
        timestamp = pd.Timestamp(as_of).normalize()
        if pd.isna(timestamp):
            raise ValueError("eligibility risk projection as_of is invalid")
        requested = (
            sorted(self._history)
            if instruments is None
            else sorted({str(instrument).upper() for instrument in instruments})
        )
        positions = []
        for instrument in requested:
            history = self._history.get(instrument)
            if history is None:
                continue
            dates, start = history
            offset = int(dates.searchsorted(timestamp, side="right")) - 1
            if offset >= 0:
                positions.append(start + offset)
        # Keep original datatypes/timezones even for an empty selection. The
        # legacy projection also validates comparison compatibility in that
        # case and retains its exact missing/stale-evidence behavior.
        selected = self._source.iloc[positions]
        return project_point_in_time_risk_states(
            selected, as_of=timestamp,
            instruments=None if instruments is None else requested,
        )


def project_point_in_time_risk_states(
    values: pd.DataFrame,
    *,
    as_of: Any,
    instruments: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Project the latest known eligibility evidence into trading risk states.

    The projection is deliberately point-in-time: only observations on or before
    ``as_of`` are considered.  Explicit adverse facts can force a reduction or
    exit, while absent evidence is fail-closed for *new* risk without pretending
    that a bad event occurred.  A requested instrument with no available row is
    therefore ``restricted`` and non-tradable, not ``exit``.

    The returned frame contains one deterministic row per requested instrument:

    - ``normal``: eligible and tradable;
    - ``watch``: temporarily non-tradable (currently suspension only);
    - ``restricted``: no new risk because eligibility evidence is incomplete or
      the instrument fails a non-terminal entry gate;
    - ``reduce``: explicit non-standard (but not severe) audit evidence;
    - ``exit``: explicit ST, delisting, non-positive equity, severe audit opinion,
      or major regulatory violation.

    ``tradable`` is independent from severity.  For example, a suspended ST stock
    remains an ``exit`` risk but has ``tradable=False`` until execution is possible.
    """

    required = {
        "datetime",
        "instrument",
        "eligible",
        "reasons",
        "is_st",
        "suspended",
        "delisted",
        "normal_listing_status",
        "equity",
        "audit_opinion",
        "financial_gate_required",
        "regulatory_data_available",
        "major_violation",
        "contract_version",
    }
    source = _required_frame(values, required, "eligibility risk projection")
    source["datetime"] = pd.to_datetime(source["datetime"], errors="coerce").dt.normalize()
    source["instrument"] = source["instrument"].astype(str).str.upper()
    if source[["datetime", "instrument"]].isna().any().any():
        raise ValueError("eligibility risk projection has invalid dates or instruments")
    if source.duplicated(["datetime", "instrument"]).any():
        raise ValueError("eligibility risk projection observations are duplicated")
    if source["contract_version"].ne(ELIGIBILITY_CONTRACT_VERSION).any():
        raise ValueError("eligibility risk projection contract is obsolete")

    timestamp = pd.Timestamp(as_of).normalize()
    if pd.isna(timestamp):
        raise ValueError("eligibility risk projection as_of is invalid")
    available = source[source["datetime"].le(timestamp)].sort_values(
        ["instrument", "datetime"]
    )
    latest = available.drop_duplicates("instrument", keep="last").set_index("instrument")

    if instruments is None:
        requested = sorted(latest.index.astype(str).unique())
    else:
        requested = sorted({str(instrument).upper() for instrument in instruments})

    rows: list[dict[str, Any]] = []
    for instrument in requested:
        if instrument not in latest.index:
            rows.append(
                {
                    "datetime": timestamp,
                    "instrument": instrument,
                    "evidence_datetime": pd.NaT,
                    "risk_state": "restricted",
                    "tradable": False,
                    "allow_new_risk": False,
                    "risk_reasons": json.dumps(["eligibility_evidence_missing"]),
                    "contract_version": ELIGIBILITY_CONTRACT_VERSION,
                }
            )
            continue

        row = latest.loc[instrument]
        source_reasons, reasons_valid = _decode_eligibility_reasons(row["reasons"])
        risk_state = "normal"
        risk_reasons: list[str] = []
        suspended = _safe_boolean(row["suspended"])
        evidence_datetime = pd.Timestamp(row["datetime"]).normalize()
        stale_evidence = evidence_datetime < timestamp
        financial_gate_required = _safe_boolean(row["financial_gate_required"])

        def apply(
            state: str,
            reason: str,
            recorded_reasons: list[str] = risk_reasons,
        ) -> None:
            nonlocal risk_state
            if _RISK_STATE_SEVERITY[state] > _RISK_STATE_SEVERITY[risk_state]:
                risk_state = state
            recorded_reasons.append(reason)

        if suspended:
            apply("watch", "suspended")
        if stale_evidence:
            # Eligibility is a daily point-in-time matrix.  Carrying an older
            # `normal` row forward could miss a new ST flag, suspension,
            # delisting or filing, so stale evidence is non-tradable.
            apply("restricted", "eligibility_evidence_stale")
        if _safe_boolean(row["is_st"]):
            apply("exit", "st")
        if _safe_boolean(row["delisted"]):
            apply("exit", "delisted")
        elif "abnormal_listing" in source_reasons or not _safe_boolean(
            row["normal_listing_status"]
        ):
            apply("restricted", "listing_status_unavailable_or_inactive")
        if _safe_boolean(row["major_violation"]):
            apply("exit", "major_violation")

        if financial_gate_required:
            equity = pd.to_numeric(pd.Series([row["equity"]]), errors="coerce").iloc[0]
            if pd.isna(equity):
                apply("restricted", "equity_evidence_missing")
            elif float(equity) <= 0:
                apply("exit", "non_positive_equity")

            audit_opinion = _normalized_audit_opinion(row["audit_opinion"])
            if audit_opinion is None:
                apply("restricted", "audit_evidence_missing")
            elif not _is_standard_audit_opinion(audit_opinion):
                if audit_opinion in _SEVERE_AUDIT_OPINIONS:
                    apply("exit", "severe_nonstandard_audit")
                else:
                    apply("reduce", "nonstandard_audit")

        restricted_source_reasons = {
            "new_listing",
            "insufficient_liquidity",
            "regulatory_data_missing",
        }
        for reason in sorted(source_reasons & restricted_source_reasons):
            apply("restricted", reason)
        if "negative_or_missing_equity" in source_reasons and not financial_gate_required:
            apply("restricted", "unexpected_financial_gate_rejection")
        if "nonstandard_or_missing_audit" in source_reasons and not financial_gate_required:
            apply("restricted", "unexpected_audit_gate_rejection")
        if not reasons_valid:
            apply("restricted", "eligibility_reasons_invalid")

        known_reasons = {
            "new_listing",
            "st",
            "suspended",
            "abnormal_listing",
            "negative_or_missing_equity",
            "nonstandard_or_missing_audit",
            "insufficient_liquidity",
            "major_violation",
            "regulatory_data_missing",
        }
        unknown_reasons = sorted(source_reasons - known_reasons)
        for reason in unknown_reasons:
            apply("restricted", f"unrecognized_eligibility_reason:{reason}")
        if not _safe_boolean(row["eligible"]) and risk_state == "normal":
            apply("restricted", "eligibility_rejection_unexplained")

        rows.append(
            {
                "datetime": timestamp,
                "instrument": instrument,
                "evidence_datetime": evidence_datetime,
                "risk_state": risk_state,
                "tradable": not suspended and not stale_evidence,
                "allow_new_risk": (
                    risk_state == "normal" and not suspended and not stale_evidence
                ),
                "risk_reasons": json.dumps(sorted(set(risk_reasons))),
                "contract_version": ELIGIBILITY_CONTRACT_VERSION,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "datetime",
            "instrument",
            "evidence_datetime",
            "risk_state",
            "tradable",
            "allow_new_risk",
            "risk_reasons",
            "contract_version",
        ],
    )


def eligibility_statistics(values: pd.DataFrame) -> dict[str, Any]:
    required = {"datetime", "instrument", "eligible", "reasons", "contract_version"}
    if not required.issubset(values.columns) or values.empty:
        raise ValueError("eligibility matrix is missing required evidence")
    if set(values["contract_version"]) != {ELIGIBILITY_CONTRACT_VERSION}:
        raise ValueError("eligibility matrix contract is obsolete")
    reason_counts: dict[str, int] = {}
    for raw in values["reasons"]:
        for reason in json.loads(str(raw)):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "contract_version": ELIGIBILITY_CONTRACT_VERSION,
        "rows": len(values),
        "dates": int(pd.to_datetime(values["datetime"]).nunique()),
        "instruments": int(values["instrument"].nunique()),
        "eligible_rows": int(values["eligible"].sum()),
        "eligible_rate": float(values["eligible"].mean()),
        "rejection_counts": reason_counts,
        "regulatory_data_available": bool(values["regulatory_data_available"].all()),
    }


def _required_frame(values: pd.DataFrame, required: set[str], label: str) -> pd.DataFrame:
    if not required.issubset(values.columns):
        missing = ", ".join(sorted(required - set(values.columns)))
        raise ValueError(f"{label} data is missing: {missing}")
    return values.copy()


def _strict_boolean(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in {0, 1}:
        return bool(value)
    return None


def _normalize_intervals(values: pd.DataFrame, *, value_column: str, label: str) -> pd.DataFrame:
    result = _required_frame(
        values,
        {"instrument", "start_date", "end_date", value_column},
        label,
    )
    result["instrument"] = result["instrument"].astype(str).str.upper()
    result["start_date"] = pd.to_datetime(result["start_date"], errors="coerce").dt.normalize()
    result["end_date"] = pd.to_datetime(result["end_date"], errors="coerce").dt.normalize()
    if result["start_date"].isna().any():
        raise ValueError(f"{label} intervals have no start date")
    result[value_column] = result[value_column].astype(bool)
    return result


def _interval_flags(
    base: pd.DataFrame, intervals: pd.DataFrame, *, value_column: str
) -> pd.Series:
    flags = pd.Series(False, index=base.index)
    base_groups = base.groupby("instrument", sort=False).groups
    for instrument, source in intervals.groupby("instrument", sort=False):
        index = base_groups.get(instrument)
        if index is None:
            continue
        target = base.loc[index, ["datetime"]].sort_values("datetime")
        dates = pd.DatetimeIndex(target["datetime"])
        for row in source.itertuples(index=False):
            left = dates.searchsorted(row.start_date, side="left")
            right = (
                dates.searchsorted(row.end_date, side="right")
                if pd.notna(row.end_date)
                else len(dates)
            )
            flags.loc[target.index[left:right]] = bool(getattr(row, value_column))
    return flags


def _asof_disclosure(
    base: pd.DataFrame,
    values: pd.DataFrame,
    *,
    value_columns: list[str],
    label: str,
) -> pd.DataFrame:
    required = {"instrument", "announcement_date", *value_columns}
    disclosure = _required_frame(values, required, label)
    disclosure["instrument"] = disclosure["instrument"].astype(str).str.upper()
    disclosure["announcement_date"] = pd.to_datetime(
        disclosure["announcement_date"], errors="coerce"
    ).dt.normalize()
    disclosure = disclosure.dropna(subset=["announcement_date", "instrument"])
    disclosure.sort_values(["instrument", "announcement_date"], inplace=True)
    result = base.copy()
    for column in value_columns:
        result[column] = np.nan if column != "audit_opinion" else None
    result[f"{label}_announcement_date"] = pd.NaT
    disclosure_groups = {
        instrument: source[["announcement_date", *value_columns]]
        for instrument, source in disclosure.groupby("instrument", sort=False)
    }
    for instrument, index in result.groupby("instrument").groups.items():
        source = disclosure_groups.get(instrument)
        if source is None:
            continue
        target = result.loc[index, ["datetime"]].sort_values("datetime")
        joined = pd.merge_asof(
            target,
            source,
            left_on="datetime",
            right_on="announcement_date",
            direction="backward",
            allow_exact_matches=False,
        )
        joined.index = target.index
        for column in value_columns:
            result.loc[joined.index, column] = joined[column]
        result.loc[joined.index, f"{label}_announcement_date"] = joined[
            "announcement_date"
        ]
    return result


def _decode_eligibility_reasons(value: Any) -> tuple[set[str], bool]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return set(), False
    if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
        return set(), False
    return set(decoded), True


def _safe_boolean(value: Any) -> bool:
    normalized = _strict_boolean(value)
    return bool(normalized) if normalized is not None else False


def _normalized_audit_opinion(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized.lower() if normalized.isascii() else normalized


def _is_standard_audit_opinion(value: str) -> bool:
    return value in {
        item.lower() if item.isascii() else item for item in STANDARD_AUDIT_OPINIONS
    }
