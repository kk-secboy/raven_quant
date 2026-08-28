from __future__ import annotations

from copy import deepcopy
from datetime import date, time
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .data_task_store import DATA_TASK_CATALOG

DATA_AUTOMATION_CONFIG_KEY = "data_automation"
DATA_AUTOMATION_CONTRACT_VERSION = "data-automation-v1"

MANAGED_SCHEDULE_NAMES = {
    "market_daily": "governed daily raw data and Qlib publication",
    "research_assets": "governed daily research-report metadata publication",
    "information_daily": "governed daily bounded information NLP",
    "information_weekly": "governed weekly structured information factors",
    "ashare_5m": "governed daily A-share five-minute publication",
    "auxiliary_daily": "governed daily auxiliary research data publication",
}

# Every catalog capability has one authoritative refresh owner.  Derived and
# Qlib tasks intentionally share the schedule that publishes their upstream
# immutable snapshot; they are durable successor jobs, not separate downloads.
TASK_SCHEDULE_GROUPS: dict[str, tuple[str, ...]] = {
    "market_daily": (
        "cn_ashare_daily_full",
        "cn_data_verify",
        "cn_snapshot_build",
        "cn_qlib_build",
        "cn_qlib_baseline",
        "cn_extended_daily",
        "cn_funds",
        "cn_macro",
        "cn_institutional",
        "cn_governance_risk",
        "cn_capital_flow",
        "cn_fund_index_enhanced",
        "cn_derivatives_enhanced",
        "global_rates_enhanced",
        "cn_futures",
        "cn_options_bonds",
        "hk_market",
        "us_market",
        "global_markets",
        "strategy_specialty",
    ),
    "research_assets": ("research_corpus",),
    "information_daily": (
        "cn_cninfo_announcements",
        "cn_announcement_nlp",
        "cn_corpus_nlp",
        "cn_event_market_response",
    ),
    "information_weekly": ("cn_structured_information_factors",),
    "ashare_5m": ("cn_ashare_5m", "cn_ashare_5m_qlib"),
    "auxiliary_daily": (
        "cn_margin_eligibility",
        "pair_execution_1m",
        "liquid_intraday_1m",
        "liquid_intraday_qlib",
        "strategy_specialty_minutes",
    ),
}

DEFAULT_STRATEGY_MINUTE_SYMBOLS = (
    "801010.SI",
    "801030.SI",
    "801040.SI",
    "801050.SI",
    "801080.SI",
    "801110.SI",
    "801120.SI",
    "801130.SI",
    "801140.SI",
    "801150.SI",
    "801160.SI",
    "801170.SI",
    "801180.SI",
    "801200.SI",
    "801210.SI",
    "801230.SI",
    "801710.SI",
    "801720.SI",
    "801730.SI",
    "801740.SI",
    "801750.SI",
    "801760.SI",
    "801770.SI",
    "801780.SI",
    "801790.SI",
    "801880.SI",
    "801890.SI",
    "801950.SI",
    "801960.SI",
    "801970.SI",
    "801980.SI",
    "00005.HK",
    "00388.HK",
    "00669.HK",
    "00700.HK",
    "00857.HK",
    "00883.HK",
    "00939.HK",
    "00941.HK",
    "00981.HK",
    "01024.HK",
    "01211.HK",
    "01299.HK",
    "01398.HK",
    "01810.HK",
    "02318.HK",
    "03690.HK",
    "03988.HK",
    "09618.HK",
    "09888.HK",
    "09988.HK",
)

DEFAULT_DATA_AUTOMATION_CONFIG: dict[str, Any] = {
    "contract_version": DATA_AUTOMATION_CONTRACT_VERSION,
    "enabled": True,
    "timezone": "Asia/Shanghai",
    "market_daily_time": "18:00",
    "research_assets_time": "19:30",
    "research_assets_history_start": "2023-08-25",
    "market_lookback_days": 7,
    "information_daily_time": "02:00",
    "information_lookback_days": 7,
    "information_weekly_time": "12:30",
    "information_weekday": 4,
    "ashare_5m_time": "23:30",
    "ashare_5m_history_start": "2024-01-01",
    "auxiliary_daily_time": "04:00",
    "auxiliary_history_start": "2024-01-01",
    "max_stocks": 100,
    "max_options": 100,
    "download_workers": 4,
    "requests_per_minute": 99,
    "strategy_minute_symbols": list(DEFAULT_STRATEGY_MINUTE_SYMBOLS),
}


def _clock(value: Any, key: str) -> str:
    candidate = str(value or "").strip()
    try:
        parsed = time.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{key} must be an HH:MM time") from exc
    if parsed.second or parsed.microsecond:
        raise ValueError(f"{key} must use minute precision")
    return parsed.isoformat(timespec="minutes")


def normalize_data_automation_config(value: Any = None) -> dict[str, Any]:
    raw = deepcopy(DEFAULT_DATA_AUTOMATION_CONFIG)
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError("data automation configuration must be an object")
        unknown = sorted(set(value) - set(raw))
        if unknown:
            raise ValueError(f"unsupported data automation settings: {unknown}")
        raw.update(value)
    if raw.get("contract_version") != DATA_AUTOMATION_CONTRACT_VERSION:
        raise ValueError("data automation contract version is invalid")
    if not isinstance(raw.get("enabled"), bool):
        raise ValueError("data automation enabled must be boolean")
    timezone = str(raw.get("timezone") or "").strip()
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("data automation timezone is unavailable") from exc
    normalized = {
        **raw,
        "timezone": timezone,
        "market_daily_time": _clock(raw["market_daily_time"], "market_daily_time"),
        "research_assets_time": _clock(
            raw["research_assets_time"], "research_assets_time"
        ),
        "information_daily_time": _clock(
            raw["information_daily_time"], "information_daily_time"
        ),
        "information_weekly_time": _clock(
            raw["information_weekly_time"], "information_weekly_time"
        ),
        "ashare_5m_time": _clock(raw["ashare_5m_time"], "ashare_5m_time"),
        "auxiliary_daily_time": _clock(
            raw["auxiliary_daily_time"], "auxiliary_daily_time"
        ),
    }
    for key in ("market_daily_time", "research_assets_time", "ashare_5m_time"):
        if time.fromisoformat(normalized[key]) < time(15, 10):
            raise ValueError(f"{key} must run after the A-share close")
    for key, minimum, maximum in (
        ("market_lookback_days", 1, 90),
        ("information_lookback_days", 1, 30),
        ("information_weekday", 0, 4),
        ("max_stocks", 1, 500),
        ("max_options", 1, 500),
        ("download_workers", 1, 16),
        ("requests_per_minute", 1, 99),
    ):
        candidate = normalized.get(key)
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise ValueError(f"{key} must be an integer")
        if not minimum <= candidate <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
    for key in (
        "research_assets_history_start",
        "ashare_5m_history_start",
        "auxiliary_history_start",
    ):
        candidate = str(normalized.get(key) or "")
        if len(candidate) != 10:
            raise ValueError(f"{key} must be an ISO date")
        date.fromisoformat(candidate)
        normalized[key] = candidate
    symbols = normalized.get("strategy_minute_symbols")
    if not isinstance(symbols, list) or not symbols or len(symbols) > 200:
        raise ValueError("strategy_minute_symbols must contain 1-200 symbols")
    normalized["strategy_minute_symbols"] = sorted(
        {str(item).strip().upper() for item in symbols if str(item).strip()}
    )
    if not normalized["strategy_minute_symbols"]:
        raise ValueError("strategy_minute_symbols cannot be empty")
    return normalized


def automation_coverage(
    schedules: list[dict[str, Any]],
    *,
    enabled: bool,
) -> dict[str, Any]:
    by_name = {str(item.get("name")): item for item in schedules}
    catalog = {item.task_key: item for item in DATA_TASK_CATALOG}
    task_rows: list[dict[str, Any]] = []
    for group, task_keys in TASK_SCHEDULE_GROUPS.items():
        schedule = by_name.get(MANAGED_SCHEDULE_NAMES[group])
        active = bool(
            enabled
            and schedule
            and schedule.get("status") == "active"
            and schedule.get("desired_status") == "active"
        )
        for task_key in task_keys:
            definition = catalog[task_key]
            task_rows.append(
                {
                    "task_key": task_key,
                    "title": definition.title,
                    "frequency": definition.frequency,
                    "schedule_group": group,
                    "schedule_id": schedule.get("id") if schedule else None,
                    "covered": active,
                    "reason": (
                        "active"
                        if active
                        else "automation_disabled"
                        if not enabled
                        else "schedule_missing"
                        if schedule is None
                        else "schedule_paused"
                    ),
                }
            )
    expected = {item.task_key for item in DATA_TASK_CATALOG}
    assigned = {item["task_key"] for item in task_rows}
    if assigned != expected:
        missing = sorted(expected - assigned)
        duplicate_or_unknown = sorted(assigned - expected)
        raise RuntimeError(
            "data automation task map does not match the catalog: "
            f"missing={missing}, unknown={duplicate_or_unknown}"
        )
    covered = sum(1 for item in task_rows if item["covered"])
    return {
        "contract_version": DATA_AUTOMATION_CONTRACT_VERSION,
        "covered": covered,
        "total": len(task_rows),
        "ready": covered == len(task_rows),
        "tasks": task_rows,
    }
