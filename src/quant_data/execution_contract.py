from __future__ import annotations

from typing import Any

DAILY_QLIB_FIELD_CONTRACT_VERSION = "daily-qlib-field-v2-share-volume"
TUSHARE_DAILY_VOLUME_UNIT = "hand"
QLIB_DAILY_VOLUME_UNIT = "share"
TUSHARE_HAND_SIZE = 100
INDEX_VOLUME_POLICY = "excluded_non_tradable_benchmark"

MINUTE_EXECUTION_CONTRACT_VERSION = "minute-qlib-execution-v4-source-units"

# Source-specific units are intentionally explicit.  In particular, an index's
# amount / volume is an average constituent trade price, not an index level, and
# futures/options need contract multipliers that are not present in minute bars.
MINUTE_SOURCE_UNIT_CONTRACTS: dict[str, dict[str, str]] = {
    "ashare_5m": {
        "asset_type": "stock",
        "source_volume_unit": "share",
        "qlib_volume_unit": "share",
        "source_amount_unit": "CNY",
        "vwap_method": "amount_div_volume",
    },
    "liquid_stocks_1m": {
        "asset_type": "stock",
        "source_volume_unit": "share",
        "qlib_volume_unit": "share",
        "source_amount_unit": "CNY",
        "vwap_method": "amount_div_volume",
    },
    "etf_1m": {
        "asset_type": "etf",
        "source_volume_unit": "share",
        "qlib_volume_unit": "share",
        "source_amount_unit": "CNY",
        "vwap_method": "amount_div_volume",
    },
    "indices_1m": {
        "asset_type": "index",
        "source_volume_unit": "share",
        "qlib_volume_unit": "share",
        "source_amount_unit": "CNY",
        "vwap_method": "close_no_constituent_price_vwap",
    },
    "futures_1m": {
        "asset_type": "future",
        "source_volume_unit": "hand",
        "qlib_volume_unit": "hand",
        "source_amount_unit": "CNY",
        "vwap_method": "close_no_contract_multiplier",
    },
    "options_1m": {
        "asset_type": "option",
        "source_volume_unit": "contract_count",
        "qlib_volume_unit": "contract_count",
        "source_amount_unit": "CNY",
        "vwap_method": "close_no_contract_multiplier",
    },
}
SIMULATION_MINUTE_SOURCE_DATASETS = frozenset(
    {"ashare_5m", "liquid_stocks_1m", "etf_1m"}
)


def require_daily_qlib_contract(provenance: dict[str, Any]) -> None:
    if provenance.get("frequency") != "day":
        raise ValueError("daily Qlib dataset provenance frequency is invalid")
    if provenance.get("field_contract_version") != DAILY_QLIB_FIELD_CONTRACT_VERSION:
        raise ValueError("daily Qlib dataset uses an obsolete field contract; rebuild it")
    if (
        provenance.get("source_volume_unit") != TUSHARE_DAILY_VOLUME_UNIT
        or provenance.get("qlib_volume_unit") != QLIB_DAILY_VOLUME_UNIT
        or int(provenance.get("source_hand_size") or 0) != TUSHARE_HAND_SIZE
        or provenance.get("index_volume_policy") != INDEX_VOLUME_POLICY
    ):
        raise ValueError("daily Qlib dataset volume units are missing or invalid")
    if provenance.get("lineage_verified") is not True:
        raise ValueError("daily Qlib dataset lineage is not verified")


def require_minute_execution_contract(
    provenance: dict[str, Any],
    *,
    frequency: str | None = None,
    simulation_eligible: bool = False,
) -> None:
    actual_frequency = str(provenance.get("frequency") or "")
    if frequency and actual_frequency != frequency:
        raise ValueError("minute execution frequency does not match dataset provenance")
    if provenance.get("execution_contract_version") != MINUTE_EXECUTION_CONTRACT_VERSION:
        raise ValueError("minute execution dataset uses an obsolete contract; rebuild it")
    fields = set(provenance.get("fields") or [])
    if not {"vwap", "volume", "paused", "up_limit", "down_limit"}.issubset(fields):
        raise ValueError("minute execution dataset lacks required capacity or price controls")
    if provenance.get("lineage_verified") is not True:
        raise ValueError("minute execution dataset lineage is not verified")
    source_datasets = provenance.get("source_datasets")
    unit_contracts = provenance.get("source_unit_contracts")
    if not isinstance(source_datasets, list) or not source_datasets:
        raise ValueError("minute execution dataset source datasets are missing")
    if not isinstance(unit_contracts, dict):
        raise ValueError("minute execution dataset source unit contracts are missing")
    normalized_sources = {str(value) for value in source_datasets}
    if normalized_sources != set(unit_contracts):
        raise ValueError("minute execution source datasets and unit contracts disagree")
    for dataset in normalized_sources:
        expected = MINUTE_SOURCE_UNIT_CONTRACTS.get(dataset)
        if expected is None or unit_contracts.get(dataset) != expected:
            raise ValueError(f"minute execution source unit contract is invalid: {dataset}")
    if simulation_eligible and not normalized_sources.issubset(
        SIMULATION_MINUTE_SOURCE_DATASETS
    ):
        raise ValueError("simulation execution data must contain only share-volume stocks/ETFs")
