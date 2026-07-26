from __future__ import annotations

from datetime import date

import pytest

from quant_platform.cost_model import (
    CN_COST_SCHEDULE_BOOK,
    CN_COST_SCHEDULE_VERSIONS,
    CostModelConfig,
    CostScheduleBook,
)

pytestmark = pytest.mark.no_database


def test_stamp_duty_boundary_2023_08_28() -> None:
    before = CN_COST_SCHEDULE_BOOK.as_of(date(2023, 8, 27))
    after = CN_COST_SCHEDULE_BOOK.as_of(date(2023, 8, 28))
    assert before.stock_sell_stamp_duty_rate == pytest.approx(0.001)
    assert after.stock_sell_stamp_duty_rate == pytest.approx(0.0005)
    sell = after.estimate_breakdown(
        side="sell",
        gross_value=100_000,
        participation=0,
        asset_type="stock",
        trade_date=date(2023, 8, 28),
    )
    assert sell["stamp_duty"] == pytest.approx(50.0)


def test_transfer_fee_2022_04_29_applies_to_both_sides() -> None:
    before = CN_COST_SCHEDULE_BOOK.as_of(date(2022, 4, 28))
    after = CN_COST_SCHEDULE_BOOK.as_of(date(2022, 4, 29))
    assert before.transfer_fee_rate == pytest.approx(0.00002)
    assert after.transfer_fee_rate == pytest.approx(0.00001)
    buy = after.estimate_breakdown(
        side="buy",
        gross_value=100_000,
        participation=0,
        trade_date=date(2022, 4, 29),
    )
    sell = after.estimate_breakdown(
        side="sell",
        gross_value=100_000,
        participation=0,
        trade_date=date(2022, 4, 29),
    )
    assert buy["transfer_fee"] == pytest.approx(1.0)
    assert sell["transfer_fee"] == pytest.approx(1.0)


def test_transfer_fee_history_versions_are_recorded() -> None:
    baseline = CN_COST_SCHEDULE_BOOK.as_of(date(2010, 1, 4))
    unified = CN_COST_SCHEDULE_BOOK.as_of(date(2015, 8, 3))
    assert baseline.transfer_fee_rate == pytest.approx(0.00002)
    assert unified.transfer_fee_rate == pytest.approx(0.00002)
    assert baseline.version != unified.version
    assert all(version.source for version in CN_COST_SCHEDULE_VERSIONS)


def test_as_of_fails_closed_without_covering_version() -> None:
    with pytest.raises(ValueError, match="no effective cost schedule"):
        CN_COST_SCHEDULE_BOOK.as_of(date(1990, 1, 2))
    single = CostScheduleBook.from_mapping({"open_cost": 0.001})
    with pytest.raises(ValueError, match="no effective cost schedule"):
        single.as_of(date(1990, 1, 2))


def test_from_mapping_keeps_legacy_aliases() -> None:
    config = CostModelConfig.from_mapping(
        {"open_cost": 0.0004, "close_cost": 0.0006, "min_cost": 3.0}
    )
    assert config.buy_commission_rate == pytest.approx(0.0004)
    assert config.sell_commission_rate == pytest.approx(0.0006)
    assert config.min_commission == pytest.approx(3.0)
    book = CostScheduleBook.from_mapping(
        {"open_cost": 0.0004, "close_cost": 0.0006, "min_cost": 3.0}
    )
    assert len(book.versions) == 1
    assert book.as_of(date(2024, 1, 2)).buy_commission_rate == pytest.approx(0.0004)
    explicit = CostScheduleBook.from_mapping(book.to_dict())
    assert explicit.versions == book.versions


def test_from_mapping_defaults_to_recorded_schedule() -> None:
    assert CostScheduleBook.from_mapping(None).versions == CN_COST_SCHEDULE_VERSIONS
    assert CostScheduleBook.from_mapping({}).versions == CN_COST_SCHEDULE_VERSIONS


def test_doubled_doubles_every_recorded_version() -> None:
    doubled = CN_COST_SCHEDULE_BOOK.doubled()
    assert len(doubled.versions) == len(CN_COST_SCHEDULE_VERSIONS)
    for original, stressed in zip(CN_COST_SCHEDULE_VERSIONS, doubled.versions, strict=True):
        assert stressed.effective_from == original.effective_from
        assert stressed.effective_to == original.effective_to
        assert stressed.stock_sell_stamp_duty_rate == pytest.approx(
            original.stock_sell_stamp_duty_rate * 2
        )
        assert stressed.transfer_fee_rate == pytest.approx(original.transfer_fee_rate * 2)
        assert stressed.buy_commission_rate == pytest.approx(
            original.buy_commission_rate * 2
        )
        assert stressed.min_commission == pytest.approx(original.min_commission * 2)
        assert stressed.fixed_slippage_rate == pytest.approx(
            original.fixed_slippage_rate * 2
        )
    assert doubled.as_of(date(2023, 8, 28)).stock_sell_stamp_duty_rate == pytest.approx(
        0.001
    )


def test_scaled_stresses_only_the_named_components() -> None:
    original = CostModelConfig()
    stressed = original.scaled(fixed_slippage_rate=2.0)
    assert stressed.fixed_slippage_rate == pytest.approx(original.fixed_slippage_rate * 2)
    assert stressed.buy_commission_rate == pytest.approx(original.buy_commission_rate)
    assert stressed.min_commission == pytest.approx(original.min_commission)
    assert stressed.impact_at_max_participation == pytest.approx(
        original.impact_at_max_participation
    )
    assert stressed.version == original.version


def test_scaled_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown cost model fields"):
        CostModelConfig().scaled(unknown_rate=2.0)


def test_book_scaled_applies_to_every_recorded_version() -> None:
    stressed = CN_COST_SCHEDULE_BOOK.scaled(max_volume_participation=0.75)
    assert len(stressed.versions) == len(CN_COST_SCHEDULE_VERSIONS)
    for original, view in zip(CN_COST_SCHEDULE_VERSIONS, stressed.versions, strict=True):
        assert view.max_volume_participation == pytest.approx(
            original.max_volume_participation * 0.75
        )
        assert view.fixed_slippage_rate == pytest.approx(original.fixed_slippage_rate)


def test_flat_view_resolves_qlib_triple_at_explicit_date() -> None:
    flat = CN_COST_SCHEDULE_BOOK.flat_view(as_of=date(2024, 1, 2))
    assert flat["open_cost"] == pytest.approx(0.0005)
    assert flat["close_cost"] == pytest.approx(0.0005)
    assert flat["min_cost"] == pytest.approx(5.0)
    assert flat["cost_schedule_version"] == "cn-effective-cost-2023-08-28"
    assert flat["as_of"] == "2024-01-02"


def test_single_version_range_check_semantics_are_preserved() -> None:
    config = CN_COST_SCHEDULE_BOOK.as_of(date(2023, 8, 28))
    with pytest.raises(ValueError, match="no effective cost schedule"):
        config.estimate(
            side="buy",
            gross_value=10_000,
            participation=0,
            trade_date=date(2023, 8, 27),
        )


def test_book_rejects_overlapping_or_empty_versions() -> None:
    with pytest.raises(ValueError, match="at least one version"):
        CostScheduleBook(())
    overlapping = (
        CostModelConfig(effective_from="2020-01-01", effective_to="2021-06-30"),
        CostModelConfig(effective_from="2021-06-30", effective_to=None),
    )
    with pytest.raises(ValueError, match="overlap"):
        CostScheduleBook(overlapping)


def test_version_field_accepts_recorded_versions_and_rejects_unknown() -> None:
    CostModelConfig(version="cn-effective-cost-2023-08-28")
    with pytest.raises(ValueError, match="obsolete"):
        CostModelConfig(version="cn-effective-cost-v0")
