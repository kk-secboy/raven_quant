from __future__ import annotations

from datetime import date

import pytest

from quant_platform.simulation_store import _annotate_position_holding_ages

pytestmark = pytest.mark.no_database


def _position(*, quantity: int = 300) -> dict:
    return {
        "portfolio_id": "paper-1",
        "instrument": "SH600000",
        "position_side": "long",
        "quantity": quantity,
        "available_quantity": quantity,
        "frozen_quantity": 0,
        "average_cost": 10.5,
    }


def test_holding_age_uses_earliest_lot_and_qlib_sessions_idempotently() -> None:
    calendar = {
        date(2026, 7, 10),
        date(2026, 7, 13),
        date(2026, 7, 14),
    }
    lots = [
        {
            "instrument": "SH600000",
            "lot_key": "lot-1",
            "quantity": 100,
            "acquired_at": date(2026, 7, 10),
        },
        {
            "instrument": "SH600000",
            "lot_key": "lot-2",
            "quantity": 200,
            "acquired_at": date(2026, 7, 13),
        },
    ]

    first = _annotate_position_holding_ages(
        [_position()],
        lots,
        calendar_days=calendar,
        as_of_date=date(2026, 7, 13),
        require_complete_age=True,
    )
    retried = _annotate_position_holding_ages(
        [_position()],
        lots,
        calendar_days=calendar,
        as_of_date=date(2026, 7, 13),
        require_complete_age=True,
    )

    assert first == retried
    assert first[0]["holding_age_sessions"] == 1
    assert first[0]["holding_age_evidence"] == {
        "status": "proven",
        "calendar": "qlib_day",
        "as_of_date": "2026-07-13",
        "earliest_acquired_at": "2026-07-10",
        "lot_count": 2,
        "position_quantity": 300,
    }


@pytest.mark.parametrize(
    ("lots", "reason"),
    [
        ([], "no_position_lots"),
        (
            [
                {
                    "instrument": "SH600000",
                    "lot_key": "legacy",
                    "quantity": 300,
                    "acquired_at": None,
                }
            ],
            "acquired_at_unknown",
        ),
        (
            [
                {
                    "instrument": "SH600000",
                    "lot_key": "partial",
                    "quantity": 100,
                    "acquired_at": date(2026, 7, 10),
                }
            ],
            "lot_quantity_mismatch",
        ),
    ],
)
def test_bounded_holding_policy_fails_closed_when_lot_age_is_unproven(
    lots: list[dict], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        _annotate_position_holding_ages(
            [_position()],
            lots,
            calendar_days={date(2026, 7, 10), date(2026, 7, 13)},
            as_of_date=date(2026, 7, 13),
            require_complete_age=True,
        )


def test_holding_age_never_uses_natural_day_distance() -> None:
    result = _annotate_position_holding_ages(
        [_position(quantity=100)],
        [
            {
                "instrument": "SH600000",
                "lot_key": "friday-fill",
                "quantity": 100,
                "acquired_at": date(2026, 7, 10),
            }
        ],
        calendar_days={date(2026, 7, 10), date(2026, 7, 13)},
        as_of_date=date(2026, 7, 13),
        require_complete_age=True,
    )

    assert result[0]["holding_age_sessions"] == 1
