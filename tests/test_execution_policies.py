from datetime import date, datetime

import pytest

from quant_platform.execution_algorithms import (
    build_execution_slices,
    execution_time_slots,
    normalize_execution_policy,
    plan_multi_day_transition,
    plan_participation_capped_slices,
    plan_wait_cancel_replace,
)

pytestmark = pytest.mark.no_database

TRADE_DATE = date(2026, 7, 13)


def _participation_policy() -> dict:
    return {
        "execution_algorithm": "participation_capped_slicing",
        "max_participation": 0.05,
        "slot_volumes": [
            {"time": "10:00", "volume": 100_000},
            {"time": "11:00", "volume": 60_000},
            {"time": "13:30", "volume": 80_000},
        ],
    }


def test_normalize_maps_canonical_policy_ids_and_validates_new_knobs() -> None:
    normalized = normalize_execution_policy({"execution_algorithm": "twap"})
    assert normalized["execution_policy_id"] == "twap_execution"
    assert normalize_execution_policy({"execution_algorithm": "next_bar_baseline"})[
        "execution_algorithm"
    ] == "next_bar"
    assert normalize_execution_policy(
        {"execution_algorithm": "wait_cancel_replace"}
    )["execution_policy_id"] == "wait_cancel_replace"
    with pytest.raises(ValueError, match="wait_checks"):
        normalize_execution_policy(
            {"execution_algorithm": "wait_cancel_replace", "wait_checks": 0}
        )
    with pytest.raises(ValueError, match="transition_days"):
        normalize_execution_policy(
            {"execution_algorithm": "multi_day_transition", "transition_days": 9}
        )
    with pytest.raises(ValueError, match="slot_volumes"):
        normalize_execution_policy(
            {
                "execution_algorithm": "participation_capped_slicing",
                "slot_volumes": [{"time": "12:00", "volume": 1_000}],
            }
        )
    with pytest.raises(ValueError, match="dedicated planner"):
        build_execution_slices(
            quantity=1_000,
            side="buy",
            trade_date=TRADE_DATE,
            policy={"execution_algorithm": "multi_day_transition"},
        )


def test_participation_capped_slicing_is_deterministic_and_respects_the_cap() -> None:
    policy = _participation_policy()
    first = plan_participation_capped_slices(
        quantity=20_000, side="buy", trade_date=TRADE_DATE, policy=policy
    )
    second = plan_participation_capped_slices(
        quantity=20_000, side="buy", trade_date=TRADE_DATE, policy=policy
    )
    assert first == second
    assert first["policy_id"] == "participation_capped_slicing"
    # 5% of 100k/60k/80k = 5000/3000/4000 -> 12000 placed, 8000 unallocated.
    assert [item["quantity"] for item in first["slices"]] == [5_000, 3_000, 4_000]
    assert first["allocated_quantity"] == 12_000
    assert first["unallocated_quantity"] == 8_000
    assert first["allocated_quantity"] + first["unallocated_quantity"] == 20_000
    for item in first["slices"]:
        assert item["participation"] <= 0.05 + 1e-12
        assert item["quantity"] % 100 == 0
    assert first["unallocated_disposition"] == "expire_or_requote_next_cycle"


def test_participation_capped_slicing_reports_shortfall_instead_of_faking_fills() -> None:
    plan = plan_participation_capped_slices(
        quantity=100_000,
        side="buy",
        trade_date=TRADE_DATE,
        policy=_participation_policy(),
    )
    # Capped liquidity absorbs only 12000; the rest must surface as unallocated.
    assert plan["allocated_quantity"] == 12_000
    assert plan["unallocated_quantity"] == 88_000
    with pytest.raises(ValueError, match="slot_volumes evidence"):
        plan_participation_capped_slices(
            quantity=1_000,
            side="buy",
            trade_date=TRADE_DATE,
            policy={"execution_algorithm": "participation_capped_slicing"},
        )
    slots = execution_time_slots(trade_date=TRADE_DATE, policy=_participation_policy())
    assert [slot.strftime("%H:%M") for slot in slots] == ["10:00", "11:00", "13:30"]


def test_participation_capped_sell_carries_odd_lot_within_the_cap() -> None:
    plan = plan_participation_capped_slices(
        quantity=4_150,
        side="sell",
        trade_date=TRADE_DATE,
        policy=_participation_policy(),
    )
    assert plan["allocated_quantity"] + plan["unallocated_quantity"] == 4_150
    assert plan["slices"][0]["quantity"] == 4_150  # 4000 lots + 150 odd lot, 4150 <= 5000
    assert plan["slices"][0]["participation"] <= 0.05 + 1e-12


def _wait_policy() -> dict:
    return {
        "execution_algorithm": "wait_cancel_replace",
        "execution_frequency": "5min",
        "wait_checks": 2,
        "max_replaces": 2,
        "replace_step_bps": 10,
    }


def test_wait_cancel_replace_is_deterministic() -> None:
    kwargs = dict(
        quantity=1_000,
        side="buy",
        trade_date=TRADE_DATE,
        policy=_wait_policy(),
        reference_price=10.0,
    )
    first = plan_wait_cancel_replace(**kwargs)
    second = plan_wait_cancel_replace(**kwargs)
    assert first == second
    assert first["policy_id"] == "wait_cancel_replace"


def test_wait_cancel_replace_cancel_then_requote_semantics() -> None:
    plan = plan_wait_cancel_replace(
        quantity=1_000,
        side="buy",
        trade_date=TRADE_DATE,
        policy=_wait_policy(),
        reference_price=10.0,
    )
    rounds = plan["rounds"]
    assert len(rounds) == 3  # 1 initial + max_replaces=2
    # Every round waits wait_checks bars, then cancels; only the last expires.
    assert [item["on_unfilled"] for item in rounds] == [
        "cancel_and_requote",
        "cancel_and_requote",
        "expire",
    ]
    for item in rounds:
        assert item["cancel_after"] > item["submitted_at"]
    # Buys re-quote one compounded step more aggressive each round, ticked up:
    # 10.00 -> 10.00*1.001=10.01 -> 10.01*1.001=10.02001 -> tick up 10.03.
    assert [item["limit_price"] for item in rounds] == [10.0, 10.01, 10.03]
    sell = plan_wait_cancel_replace(
        quantity=1_000,
        side="sell",
        trade_date=TRADE_DATE,
        policy=_wait_policy(),
        reference_price=10.0,
    )
    sell_prices = [item["limit_price"] for item in sell["rounds"]]
    assert sell_prices == sorted(sell_prices, reverse=True)
    assert sell_prices[0] == 10.0
    # The plan never grows the order and ends by expiring the unfilled remainder.
    assert all(item["quantity"] == 1_000 for item in rounds)
    assert plan["final_action"] == "expire_unfilled_remainder"
    assert "exactly once" in plan["semantics"]["cancel"]
    assert "never grows" in plan["semantics"]["replace"]


def test_wait_cancel_replace_zero_replaces_waits_then_expires() -> None:
    plan = plan_wait_cancel_replace(
        quantity=500,
        side="buy",
        trade_date=TRADE_DATE,
        policy={
            "execution_algorithm": "wait_cancel_replace",
            "wait_checks": 6,
            "max_replaces": 0,
        },
        reference_price=23.455,
    )
    assert len(plan["rounds"]) == 1
    assert plan["rounds"][0]["on_unfilled"] == "expire"
    assert plan["rounds"][0]["limit_price"] == 23.46  # buy limit ticks up


def test_wait_cancel_replace_respects_signal_time_and_window_end() -> None:
    plan = plan_wait_cancel_replace(
        quantity=1_000,
        side="buy",
        trade_date=TRADE_DATE,
        policy={
            "execution_algorithm": "wait_cancel_replace",
            "execution_frequency": "5min",
            "wait_checks": 4,
            "max_replaces": 16,
        },
        reference_price=10.0,
        signal_at=datetime.fromisoformat("2026-07-13T14:30:00+08:00"),
    )
    # Late signal: the window closes before the waits elapse, so the final
    # round is truncated at the last governed bar and expires there.
    assert plan["rounds"][-1]["cancel_after"] == "2026-07-13T14:50:00+08:00"
    assert plan["rounds"][-1]["on_unfilled"] == "expire"
    for earlier, later in zip(plan["rounds"], plan["rounds"][1:], strict=False):
        assert later["submitted_at"] >= earlier["cancel_after"]


def test_multi_day_transition_conserves_quantity_and_aligns_windows() -> None:
    days = [date(2026, 7, 13), date(2026, 7, 14), date(2026, 7, 15)]
    policy = {"execution_algorithm": "multi_day_transition", "transition_days": 3}
    first = plan_multi_day_transition(
        quantity=10_000, side="sell", trade_dates=days, policy=policy
    )
    second = plan_multi_day_transition(
        quantity=10_000, side="sell", trade_dates=days, policy=policy
    )
    assert first == second
    assert first["policy_id"] == "multi_day_transition"
    total = sum(entry["quantity"] for entry in first["days"])
    assert total == first["allocated_quantity"] == 10_000
    assert first["unallocated_quantity"] == 0
    assert first["allocated_quantity"] + first["unallocated_quantity"] == 10_000
    for entry in first["days"]:
        assert entry["not_before"].endswith("T10:00:00+08:00")
        assert entry["not_after"].endswith("T15:00:00+08:00")
        assert entry["not_after"][:10] == entry["trade_date"]


def test_multi_day_transition_participation_cap_reports_unallocated() -> None:
    days = [date(2026, 7, 13), date(2026, 7, 14)]
    plan = plan_multi_day_transition(
        quantity=10_000,
        side="buy",
        trade_dates=days,
        policy={
            "execution_algorithm": "multi_day_transition",
            "transition_days": 2,
            "max_participation": 0.04,
            "daily_volumes": [100_000, 100_000],
        },
    )
    # 4% of 100k = 4000/day -> 8000 placed, 2000 reported unallocated.
    assert [entry["quantity"] for entry in plan["days"]] == [4_000, 4_000]
    assert plan["allocated_quantity"] == 8_000
    assert plan["unallocated_quantity"] == 2_000
    for entry in plan["days"]:
        assert entry["quantity"] <= entry["participation_capacity"]
    assert plan["allocated_quantity"] + plan["unallocated_quantity"] == 10_000


def test_multi_day_transition_validates_dates_and_slices_intraday() -> None:
    days = [date(2026, 7, 13), date(2026, 7, 14)]
    with pytest.raises(ValueError, match="strictly increasing"):
        plan_multi_day_transition(
            quantity=1_000,
            side="buy",
            trade_dates=[days[1], days[0]],
            policy={"execution_algorithm": "multi_day_transition", "transition_days": 2},
        )
    with pytest.raises(ValueError, match="transition_days"):
        plan_multi_day_transition(
            quantity=1_000,
            side="buy",
            trade_dates=days,
            policy={"execution_algorithm": "multi_day_transition", "transition_days": 3},
        )
    plan = plan_multi_day_transition(
        quantity=10_000,
        side="buy",
        trade_dates=days,
        policy={
            "execution_algorithm": "multi_day_transition",
            "transition_days": 2,
            "intraday_algorithm": "twap",
            "slice_minutes": 20,
        },
    )
    for entry in plan["days"]:
        assert entry["slices"]
        assert sum(item["quantity"] for item in entry["slices"]) == entry["quantity"]
        for item in entry["slices"]:
            assert item["scheduled_for"][:10] == entry["trade_date"]
    assert sum(entry["quantity"] for entry in plan["days"]) == 10_000
