from __future__ import annotations

import pytest

from quant_data.execution_contract import (
    DAILY_QLIB_FIELD_CONTRACT_VERSION,
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
    require_daily_qlib_contract,
    require_minute_execution_contract,
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
