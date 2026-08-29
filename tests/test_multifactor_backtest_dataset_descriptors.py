from __future__ import annotations

from copy import deepcopy

import pytest

from quant_data.execution_contract import (
    DAILY_QLIB_FIELD_CONTRACT_VERSION,
    MINUTE_EXECUTION_CONTRACT_VERSION,
    MINUTE_SOURCE_UNIT_CONTRACTS,
    require_daily_qlib_contract,
    require_native_daily_execution_controls,
)
from quant_data.universe import governed_daily_etf_whitelist_contract
from quant_platform.strategy_recipes import (
    TRANSPARENT_RESEARCH_BASELINE_IDS,
    get_strategy_recipe,
)
from scripts.run_multifactor_backtest import _promotion_dataset_descriptors


def _ready_etf_evidence() -> dict:
    contract = governed_daily_etf_whitelist_contract()
    return {
        **contract,
        "status": "ready",
        "included_symbols": list(contract["symbols"]),
        "missing_symbols": [],
    }


def _daily_provenance() -> dict:
    return {
        "frequency": "day",
        "dataset_identity_sha256": "a" * 64,
        "dataset_lineage_id": "b" * 64,
        "source_lineage_id": "c" * 64,
        "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
        "source_volume_unit": "hand",
        "qlib_volume_unit": "share",
        "source_amount_unit": "thousand_cny",
        "qlib_amount_unit": "cny",
        "source_hand_size": 100,
        "index_volume_policy": "excluded_non_tradable_benchmark",
        "governed_etf_whitelist": _ready_etf_evidence(),
        "lineage_verified": True,
        "execution_controls": {
            "formal_execution_requires_native_controls": True,
            "native_complete_from": "2008-01-01",
        },
    }


@pytest.mark.no_database
@pytest.mark.parametrize("recipe_id", TRANSPARENT_RESEARCH_BASELINE_IDS)
def test_transparent_daily_baselines_freeze_daily_execution_descriptor(
    recipe_id: str,
) -> None:
    recipe = get_strategy_recipe(recipe_id)
    config = recipe["config_overrides"]
    assert config["execution_method"] == "open"
    assert config["execution_frequency"] == "day"

    descriptors = _promotion_dataset_descriptors(
        daily_dataset_name="daily-ready",
        daily_provenance=_daily_provenance(),
        execution_method=config["execution_method"],
        execution_frequency=config["execution_frequency"],
        formal_execution_start="2021-01-04",
    )

    assert descriptors["execution"] == descriptors["daily"]
    assert descriptors["execution"] is not descriptors["daily"]
    assert descriptors["execution"]["provenance"] is not descriptors["daily"][
        "provenance"
    ]
    assert descriptors["execution"]["provenance"]["frequency"] == "day"
    assert "execution_contract_version" not in descriptors["execution"]["provenance"]
    for descriptor in descriptors.values():
        require_daily_qlib_contract(descriptor["provenance"])
        require_native_daily_execution_controls(
            descriptor["provenance"], start="2021-01-04"
        )


@pytest.mark.no_database
def test_daily_open_descriptor_rejects_minute_fabrication() -> None:
    provenance = _daily_provenance()
    minute = {
        "frequency": "5min",
        "dataset_identity_sha256": "d" * 64,
        "dataset_lineage_id": "e" * 64,
        "source_lineage_id": provenance["source_lineage_id"],
        "execution_contract_version": MINUTE_EXECUTION_CONTRACT_VERSION,
        "fields": ["vwap", "volume", "paused", "up_limit", "down_limit"],
        "source_datasets": ["ashare_5m"],
        "source_unit_contracts": {
            "ashare_5m": MINUTE_SOURCE_UNIT_CONTRACTS["ashare_5m"]
        },
        "lineage_verified": True,
    }

    with pytest.raises(ValueError, match="must reuse the daily Qlib dataset"):
        _promotion_dataset_descriptors(
            daily_dataset_name="daily-ready",
            daily_provenance=provenance,
            execution_method="open",
            execution_frequency="day",
            formal_execution_start="2021-01-04",
            execution_dataset_name="minute-ready",
            execution_provenance=minute,
        )

    incomplete = deepcopy(provenance)
    incomplete.pop("dataset_lineage_id")
    with pytest.raises(ValueError, match="dataset_lineage_id must be a SHA-256"):
        _promotion_dataset_descriptors(
            daily_dataset_name="daily-ready",
            daily_provenance=incomplete,
            execution_method="open",
            execution_frequency="day",
            formal_execution_start="2021-01-04",
        )
