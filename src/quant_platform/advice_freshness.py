"""Fail-closed freshness contracts for novice-facing formal advice."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant_data.cninfo_announcements import load_trade_calendar_open_days

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_DAILY_CLOSE = time(15, 0)


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


def latest_closed_trading_day(open_days: list[date], *, now: datetime) -> date:
    """Return the latest persisted exchange session whose daily bar is closed."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("advice projection time must be timezone-aware")
    local = now.astimezone(_SHANGHAI)
    cutoff = local.date()
    if local.timetz().replace(tzinfo=None) < _DAILY_CLOSE:
        cutoff -= timedelta(days=1)
    eligible = [value for value in open_days if value <= cutoff]
    if not eligible:
        raise ValueError("trade calendar has no closed trading day")
    return max(eligible)


def advice_session_requirement(
    data_root: Path | None,
    *,
    now: datetime,
) -> dict[str, Any]:
    """Resolve the formal advice session from the persisted SSE calendar."""

    if data_root is None:
        return {
            "status": "blocked",
            "required_signal_date": None,
            "reason": "正式荐股未绑定受治理数据目录，无法确认最新闭市交易日",
        }
    try:
        open_days = load_trade_calendar_open_days(data_root)
        required = latest_closed_trading_day(open_days, now=now)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "status": "blocked",
            "required_signal_date": None,
            "reason": f"交易日历不可用，正式荐股已停止：{str(exc)[:300]}",
        }
    return {
        "status": "current",
        "required_signal_date": required.isoformat(),
        "reason": None,
    }


def assess_signal_freshness(
    *,
    required_signal_date: date | None,
    observed_signal_date: Any,
    requirement_reason: str | None = None,
) -> dict[str, Any]:
    """Require a formal recommendation snapshot for the exact closed session."""

    observed = _date_value(observed_signal_date)
    base = {
        "required_signal_date": (
            required_signal_date.isoformat() if required_signal_date else None
        ),
        "observed_signal_date": observed.isoformat() if observed else None,
    }
    if required_signal_date is None:
        return {
            **base,
            "status": "blocked",
            "passed": False,
            "reason": requirement_reason or "无法确认最新闭市交易日",
        }
    if observed is None:
        return {
            **base,
            "status": "missing",
            "passed": False,
            "reason": (
                f"尚未生成 {required_signal_date.isoformat()} 的正式推荐快照"
            ),
        }
    if observed != required_signal_date:
        relation = "stale" if observed < required_signal_date else "future_mismatch"
        return {
            **base,
            "status": relation,
            "passed": False,
            "reason": (
                f"正式推荐快照日期为 {observed.isoformat()}，"
                f"最新闭市交易日为 {required_signal_date.isoformat()}"
            ),
        }
    return {
        **base,
        "status": "current",
        "passed": True,
        "reason": None,
    }


def assess_netting_plan_freshness(
    *,
    required_signal_date: date | None,
    inputs_as_of: Any,
    plan: dict[str, Any],
    requirement_reason: str | None = None,
) -> dict[str, Any]:
    """Require current input dates for both the plan and every member snapshot."""

    plan_freshness = assess_signal_freshness(
        required_signal_date=required_signal_date,
        observed_signal_date=inputs_as_of,
        requirement_reason=requirement_reason,
    )
    if not plan_freshness["passed"]:
        return {
            **plan_freshness,
            "scope": "account_netting_plan",
            "member_signal_dates": {},
        }

    raw_evidence = dict(plan.get("input_evidence") or {}).get("member_snapshots")
    if not isinstance(raw_evidence, dict) or not raw_evidence:
        return {
            **plan_freshness,
            "scope": "account_netting_plan",
            "status": "member_evidence_missing",
            "passed": False,
            "reason": "统一账户净额计划缺少逐策略推荐快照证据",
            "member_signal_dates": {},
        }
    member_dates = {
        str(member): _date_value(dict(evidence).get("as_of_date"))
        for member, evidence in raw_evidence.items()
        if isinstance(evidence, dict)
    }
    if len(member_dates) != len(raw_evidence) or any(
        value is None for value in member_dates.values()
    ):
        return {
            **plan_freshness,
            "scope": "account_netting_plan",
            "status": "member_evidence_invalid",
            "passed": False,
            "reason": "统一账户净额计划包含无效的成员推荐日期",
            "member_signal_dates": {
                member: value.isoformat() if value else None
                for member, value in member_dates.items()
            },
        }
    assert required_signal_date is not None
    mismatched = {
        member: value
        for member, value in member_dates.items()
        if value != required_signal_date
    }
    if mismatched:
        return {
            **plan_freshness,
            "scope": "account_netting_plan",
            "status": "member_snapshots_stale",
            "passed": False,
            "reason": (
                "统一账户净额计划仍引用过期成员推荐："
                + "、".join(
                    f"{member}={value.isoformat()}"
                    for member, value in sorted(mismatched.items())
                )
            ),
            "member_signal_dates": {
                member: value.isoformat() for member, value in member_dates.items()
            },
        }
    return {
        **plan_freshness,
        "scope": "account_netting_plan",
        "member_signal_dates": {
            member: value.isoformat() for member, value in member_dates.items()
        },
    }
