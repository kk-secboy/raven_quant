from datetime import date

import pytest

from quant_platform.execution_algorithms import (
    build_execution_slices,
    normalize_execution_policy,
)


def test_twap_respects_ashare_sessions_lots_and_reconciliation() -> None:
    policy = normalize_execution_policy(
        {"execution_algorithm": "twap", "slice_minutes": 20, "max_slices": 24}
    )
    slices = build_execution_slices(
        quantity=10_000,
        side="buy",
        trade_date=date(2026, 7, 13),
        policy=policy,
    )
    assert [item["scheduled_for"][11:16] for item in slices] == [
        "10:00",
        "10:20",
        "10:40",
        "11:00",
        "11:20",
        "13:30",
        "13:50",
        "14:10",
        "14:30",
        "14:50",
    ]
    assert all(item["scheduled_for"].endswith("+08:00") for item in slices)
    assert sum(item["quantity"] for item in slices) == 10_000
    assert all(item["quantity"] % 100 == 0 for item in slices)
    assert sum(item["target_weight"] for item in slices) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="multiple of 100"):
        build_execution_slices(
            quantity=1_050,
            side="buy",
            trade_date=date(2026, 7, 13),
            policy=policy,
        )
    sell = build_execution_slices(
        quantity=1_050,
        side="sell",
        trade_date=date(2026, 7, 13),
        policy=policy,
    )
    assert sum(item["quantity"] for item in sell) == 1_050
    assert sell[-1]["quantity"] % 100 == 50


def test_vwap_requires_point_in_time_profile_inside_execution_sessions() -> None:
    with pytest.raises(ValueError, match="volume_profile evidence"):
        build_execution_slices(
            quantity=1_000,
            side="buy",
            trade_date=date(2026, 7, 13),
            policy={"execution_algorithm": "vwap"},
        )
    with pytest.raises(ValueError, match="inside execution sessions"):
        build_execution_slices(
            quantity=1_000,
            side="buy",
            trade_date=date(2026, 7, 13),
            policy={
                "execution_algorithm": "vwap",
                "volume_profile": [{"time": "12:00", "weight": 1.0}],
            },
        )

    slices = build_execution_slices(
        quantity=1_000,
        side="buy",
        trade_date=date(2026, 7, 13),
        policy={
            "execution_algorithm": "vwap",
            "volume_profile": [
                {"time": "10:20", "weight": 0.7},
                {"time": "14:30", "weight": 0.3},
            ],
        },
    )
    assert [item["quantity"] for item in slices] == [700, 300]
    assert [item["scheduled_for"][11:16] for item in slices] == ["10:20", "14:30"]
