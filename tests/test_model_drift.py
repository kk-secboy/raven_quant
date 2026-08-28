from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_platform.model_drift import build_persistent_drift_evidence

pytestmark = pytest.mark.no_database


def _days(count: int) -> list[date]:
    start = date(2026, 1, 1)
    return [start + timedelta(days=index) for index in range(count)]


def _rows(days: list[date], *, excess: float = -0.001) -> list[dict[str, object]]:
    return [
        {
            "trade_date": day,
            "twr_daily_return": 0.001 + excess,
            "benchmark_return": 0.001,
            "performance_certified": True,
            "has_stale_prices": False,
            "status": "ok",
        }
        for day in days
    ]


def test_persistent_drift_requires_three_complete_non_overlapping_windows() -> None:
    days = _days(60)
    evidence = build_persistent_drift_evidence(
        nav_rows=_rows(days),
        trading_days=days,
        as_of=days[-1],
        dataset_lineage_id="a" * 64,
    )

    assert evidence is not None
    assert evidence["window_semantics"] == "adjacent_non_overlapping"
    assert len(evidence["windows"]) == 3
    assert evidence["observed"] == pytest.approx(-0.001)
    assert len(evidence["nav_evidence_sha256"]) == 64


def test_persistent_drift_fails_closed_on_gap_stale_or_unconfirmed_window() -> None:
    days = _days(60)
    assert (
        build_persistent_drift_evidence(
            nav_rows=_rows(days[:-1]),
            trading_days=days,
            as_of=days[-1],
            dataset_lineage_id="b" * 64,
        )
        is None
    )
    stale = _rows(days)
    stale[-1]["has_stale_prices"] = True
    assert (
        build_persistent_drift_evidence(
            nav_rows=stale,
            trading_days=days,
            as_of=days[-1],
            dataset_lineage_id="b" * 64,
        )
        is None
    )
    recovered = _rows(days)
    for row in recovered[-20:]:
        row["twr_daily_return"] = 0.003
    assert (
        build_persistent_drift_evidence(
            nav_rows=recovered,
            trading_days=days,
            as_of=days[-1],
            dataset_lineage_id="b" * 64,
        )
        is None
    )


def test_persistent_drift_rejects_overlapping_or_duplicate_evidence() -> None:
    days = _days(60)
    duplicate = _rows(days)
    duplicate.append(dict(duplicate[-1]))
    with pytest.raises(ValueError, match="duplicate NAV"):
        build_persistent_drift_evidence(
            nav_rows=duplicate,
            trading_days=days,
            as_of=days[-1],
            dataset_lineage_id="c" * 64,
        )
