from __future__ import annotations

from datetime import datetime

import pytest

from quant_data.execution_contract import (
    DAILY_QLIB_FIELD_CONTRACT_VERSION,
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
    build_strategy_execution_contract,
    require_daily_qlib_contract,
    require_minute_execution_contract,
    require_minute_signal_contract,
    require_next_bar_execution,
    require_strategy_execution_contract,
    strategy_execution_contract_hash,
)

pytestmark = pytest.mark.no_database


def test_daily_contract_requires_share_volume_and_verified_lineage() -> None:
    valid = {
        "frequency": "day",
        "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
        "source_volume_unit": "hand",
        "qlib_volume_unit": "share",
        "source_hand_size": 100,
        "index_volume_policy": "excluded_non_tradable_benchmark",
        "lineage_verified": True,
    }
    require_daily_qlib_contract(valid)

    with pytest.raises(ValueError, match="obsolete field contract"):
        require_daily_qlib_contract({**valid, "field_contract_version": "v1"})
    with pytest.raises(ValueError, match="lineage is not verified"):
        require_daily_qlib_contract({**valid, "lineage_verified": False})


def test_minute_contract_requires_version_frequency_and_verified_lineage() -> None:
    valid = {
        "frequency": "5min",
        "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
        "lineage_verified": True,
        "fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
        "source_datasets": ["ashare_5m"],
        "source_unit_contracts": {
            "ashare_5m": MINUTE_SOURCE_UNIT_CONTRACTS["ashare_5m"]
        },
    }
    require_minute_execution_contract(valid, frequency="5min")

    with pytest.raises(ValueError, match="frequency"):
        require_minute_execution_contract(valid, frequency="1min")
    with pytest.raises(ValueError, match="obsolete contract"):
        require_minute_execution_contract(
            {**valid, "execution_contract_version": "v1"}, frequency="5min"
        )

    with pytest.raises(ValueError, match="unit contract"):
        require_minute_execution_contract(
            {**valid, "source_unit_contracts": {"ashare_5m": {}}}, frequency="5min"
        )
    with pytest.raises(ValueError, match="signal-only"):
        require_minute_execution_contract(
            {**valid, "resampled": True}, frequency="5min"
        )


def test_simulation_contract_rejects_non_share_sources() -> None:
    provenance = {
        "frequency": "5min",
        "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
        "lineage_verified": True,
        "fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
        "source_datasets": ["futures_1m"],
        "source_unit_contracts": {
            "futures_1m": MINUTE_SOURCE_UNIT_CONTRACTS["futures_1m"]
        },
    }
    with pytest.raises(ValueError, match="stocks/ETFs"):
        require_minute_execution_contract(
            provenance, frequency="5min", simulation_eligible=True
        )


def test_resampled_signal_contract_requires_qlib_provenance() -> None:
    provenance = {
        "frequency": "30min",
        "source_frequency": "5min",
        "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
        "lineage_verified": True,
        "fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
        "source_datasets": ["ashare_5m"],
        "source_unit_contracts": {
            "ashare_5m": MINUTE_SOURCE_UNIT_CONTRACTS["ashare_5m"]
        },
        "resampled": True,
        "resample_contract_version": "qlib-minute-resample-v1",
        "resample_engine": "qlib.utils.resam.resam_calendar",
    }
    require_minute_signal_contract(provenance, frequency="30min")

    with pytest.raises(ValueError, match="missing or obsolete"):
        require_minute_signal_contract(
            {**provenance, "resample_engine": "pandas"}, frequency="30min"
        )


def test_strategy_contract_hashes_frequency_cost_risk_and_next_bar_policy() -> None:
    config = {
        "signal_frequency": "day",
        "signal_period": 1,
        "execution_frequency": "5min",
        "execution_lag_bars": 1,
        "execution_method": "vwap",
        "execution_days": 3,
        "execution_slice_minutes": 20,
        "max_execution_slices": 24,
        "max_volume_participation": 0.01,
        "min_average_daily_amount": 500_000_000,
        "max_position_weight": 0.02,
        "max_daily_turnover": 0.15,
        "target_volatility": 0.15,
        "cost_schedule_version": "cn-effective-cost-v1",
    }
    digest = strategy_execution_contract_hash(config)
    governed = {**config, "execution_contract_hash": digest}
    contract = require_strategy_execution_contract(governed)

    assert contract["execution"]["same_bar_execution"] is False
    assert contract["execution"]["data_mode"] == "native"
    assert len(digest) == 64
    assert strategy_execution_contract_hash(
        {**config, "max_daily_turnover": 0.10}
    ) != digest
    assert strategy_execution_contract_hash(
        {**config, "target_volatility": 0.12}
    ) != digest


def test_qlib_resampling_and_same_bar_execution_are_explicit() -> None:
    contract = build_strategy_execution_contract(
        {
            "signal_frequency": "30min",
            "signal_period": 2,
            "execution_frequency": "5min",
            "execution_lag_bars": 1,
            "execution_method": "twap",
        }
    )
    assert contract["signal"]["data_mode"] == "qlib_resample"
    assert contract["execution"]["data_mode"] == "native"

    with pytest.raises(ValueError, match="same-bar"):
        require_next_bar_execution(
            datetime(2026, 7, 16, 10, 0),
            datetime(2026, 7, 16, 10, 29),
            signal_frequency="30min",
        )
    require_next_bar_execution(
        datetime(2026, 7, 16, 10, 0),
        datetime(2026, 7, 16, 10, 30),
        signal_frequency="30min",
    )
    require_next_bar_execution(
        datetime(2026, 7, 16, 10, 5),
        datetime(2026, 7, 16, 10, 6),
        signal_frequency="5min",
        execution_frequency="1min",
    )
    with pytest.raises(ValueError, match="same-bar"):
        require_next_bar_execution(
            datetime(2026, 7, 16, 10, 5),
            datetime(2026, 7, 16, 10, 5),
            signal_frequency="5min",
            execution_frequency="1min",
        )

    with pytest.raises(ValueError, match="signal inputs"):
        build_strategy_execution_contract(
            {
                "signal_frequency": "15min",
                "signal_period": 1,
                "execution_frequency": "15min",
                "execution_lag_bars": 1,
                "execution_method": "next_bar",
            }
        )
