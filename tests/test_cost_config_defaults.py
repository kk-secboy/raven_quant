from datetime import date

import pytest

from quant_platform.cost_model import (
    CN_COST_SCHEDULE_VERSIONS,
    CURRENT_STOCK_SELL_STAMP_DUTY_RATE,
    CURRENT_TRANSFER_FEE_RATE,
    CostModelConfig,
)

pytestmark = pytest.mark.no_database

LATEST = CN_COST_SCHEDULE_VERSIONS[-1]


def test_current_rate_constants_match_the_latest_recorded_version() -> None:
    assert LATEST.version == "cn-effective-cost-2023-08-28"
    assert LATEST.stock_sell_stamp_duty_rate == CURRENT_STOCK_SELL_STAMP_DUTY_RATE == 0.0005
    assert LATEST.transfer_fee_rate == CURRENT_TRANSFER_FEE_RATE == 0.00001


def test_bare_cost_model_defaults_track_current_law() -> None:
    config = CostModelConfig()
    assert config.stock_sell_stamp_duty_rate == CURRENT_STOCK_SELL_STAMP_DUTY_RATE
    assert config.transfer_fee_rate == CURRENT_TRANSFER_FEE_RATE


def test_from_mapping_skips_none_values() -> None:
    config = CostModelConfig.from_mapping(
        {"stock_sell_stamp_duty_rate": None, "transfer_fee_rate": None, "lot_size": None}
    )
    assert config.stock_sell_stamp_duty_rate == CURRENT_STOCK_SELL_STAMP_DUTY_RATE
    assert config.transfer_fee_rate == CURRENT_TRANSFER_FEE_RATE
    assert config.lot_size == 100


def test_dated_versions_resolve_real_rates() -> None:
    from quant_platform.cost_model import CN_COST_SCHEDULE_BOOK

    assert (
        CN_COST_SCHEDULE_BOOK.as_of(date(2023, 8, 25)).stock_sell_stamp_duty_rate == 0.001
    )
    assert (
        CN_COST_SCHEDULE_BOOK.as_of(date(2023, 8, 28)).stock_sell_stamp_duty_rate == 0.0005
    )
    assert CN_COST_SCHEDULE_BOOK.as_of(date(2022, 4, 29)).transfer_fee_rate == 0.00001
    assert CN_COST_SCHEDULE_BOOK.as_of(date(2022, 4, 28)).transfer_fee_rate == 0.00002


def test_strategy_request_defaults_use_current_law_rates() -> None:
    from quant_platform.api import StrategyConfigRequest

    strategy = StrategyConfigRequest(recipe_id="custom", recipe_version="custom")
    assert strategy.stock_sell_stamp_duty_rate == CURRENT_STOCK_SELL_STAMP_DUTY_RATE
    assert strategy.transfer_fee_rate == CURRENT_TRANSFER_FEE_RATE


def test_strategy_request_accepts_dated_version_labels() -> None:
    from quant_platform.api import StrategyConfigRequest

    request = StrategyConfigRequest(
        recipe_id="custom",
        recipe_version="custom",
        cost_schedule_version="cn-effective-cost-2023-08-28",
    )
    assert request.cost_schedule_version == "cn-effective-cost-2023-08-28"

    with pytest.raises(ValueError, match="obsolete"):
        StrategyConfigRequest(
            recipe_id="custom",
            recipe_version="custom",
            cost_schedule_version="cn-effective-cost-1990-01-01",
        )


def test_governed_schedule_resolves_pre_2015_versions_exactly() -> None:
    from quant_platform.cost_model import CN_COST_SCHEDULE_BOOK

    # Pre-2015 fee structures (bilateral stamp duty, Shanghai par-value and
    # Shenzhen traded-value transfer fees) are recorded per effective range.
    assert (
        CN_COST_SCHEDULE_BOOK.as_of(date(2015, 7, 31)).version
        == "cn-effective-cost-2012-09-01"
    )
    opening = CN_COST_SCHEDULE_BOOK.as_of(date(2008, 1, 2))
    assert opening.version == "cn-effective-cost-2008-01-02"
    assert opening.stock_buy_stamp_duty_rate == 0.003
    assert opening.stock_sell_stamp_duty_rate == 0.003
    assert opening.sh_transfer_fee_par_rate == 0.0005
    assert opening.sz_transfer_fee_rate == 0.0000255
    with pytest.raises(ValueError, match="no effective cost schedule"):
        CN_COST_SCHEDULE_BOOK.as_of(date(2008, 1, 1))
    assert CN_COST_SCHEDULE_BOOK.as_of(date(2015, 8, 1)).transfer_fee_rate == 0.00002
    # No overlap and only the latest version is open-ended.
    versions = CN_COST_SCHEDULE_BOOK.versions
    assert [v.effective_to for v in versions[:-1]] == [
        "2008-04-23",
        "2008-09-18",
        "2012-05-31",
        "2012-08-31",
        "2015-07-31",
        "2022-04-28",
        "2023-08-27",
    ]
    assert versions[-1].effective_to is None
