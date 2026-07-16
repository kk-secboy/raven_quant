from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ELIGIBILITY_CONTRACT_VERSION = "ashare-point-in-time-eligibility-v1"
STANDARD_AUDIT_OPINIONS = frozenset(
    {
        "standard_unqualified",
        "unqualified",
        "标准无保留意见",
        "无保留意见",
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
) -> pd.DataFrame:
    """Build one fail-closed eligibility row per observed stock and trading date."""

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
    calendar = pd.DatetimeIndex(base["datetime"].unique()).sort_values()
    base["listing_trading_days"] = base["list_date"].map(
        lambda listed: int((calendar >= listed).sum()) if pd.notna(listed) else 0
    )
    # Correct the count to each row's point in time, not the full calendar end.
    base["listing_trading_days"] = [
        int(((calendar >= listed) & (calendar <= current)).sum())
        if pd.notna(listed)
        else 0
        for listed, current in zip(base["list_date"], base["datetime"], strict=True)
    ]
    base["normal_listing_status"] = base["list_date"].notna() & (
        base["delist_date"].isna() | (base["datetime"] < base["delist_date"])
    )
    base["delisted"] = base["delist_date"].notna() & (
        base["datetime"] >= base["delist_date"]
    )

    st = _normalize_intervals(st_intervals, value_column="is_st", label="ST")
    base["is_st"] = _interval_flags(base, st, value_column="is_st")
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
        major = events[normalized_major]
        for row in major.itertuples(index=False):
            mask = (base["instrument"] == row.instrument) & (
                base["datetime"] >= row.known_date
            )
            base.loc[mask, "major_violation"] = True

    checks = {
        "new_listing": base["listing_trading_days"] < rules.min_listing_trading_days,
        "st": base["is_st"],
        "suspended": base["suspended"],
        "abnormal_listing": ~base["normal_listing_status"],
        "negative_or_missing_equity": ~base["positive_equity"],
        "nonstandard_or_missing_audit": ~base["standard_audit_opinion"],
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
    base["eligible"] = ~pd.DataFrame(checks).any(axis=1)
    base["reasons"] = [
        json.dumps([name for name, values in checks.items() if bool(values.loc[index])])
        for index in base.index
    ]
    base["contract_version"] = ELIGIBILITY_CONTRACT_VERSION
    columns = [
        "datetime",
        "instrument",
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
        "regulatory_data_available",
        "major_violation",
        "contract_version",
    ]
    return base[columns].sort_values(["datetime", "instrument"]).reset_index(drop=True)


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
    for row in intervals.itertuples(index=False):
        mask = (base["instrument"] == row.instrument) & (base["datetime"] >= row.start_date)
        if pd.notna(row.end_date):
            mask &= base["datetime"] <= row.end_date
        flags.loc[mask] = bool(getattr(row, value_column))
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
    for instrument, index in result.groupby("instrument").groups.items():
        source = disclosure[disclosure["instrument"] == instrument]
        if source.empty:
            continue
        target = result.loc[index, ["datetime"]].sort_values("datetime")
        joined = pd.merge_asof(
            target,
            source[["announcement_date", *value_columns]],
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
