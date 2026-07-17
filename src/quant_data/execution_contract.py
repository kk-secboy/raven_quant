from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

DAILY_QLIB_FIELD_CONTRACT_VERSION = "daily-qlib-field-v3-cny-amount"
TUSHARE_DAILY_VOLUME_UNIT = "hand"
QLIB_DAILY_VOLUME_UNIT = "share"
TUSHARE_DAILY_AMOUNT_UNIT = "thousand_cny"
QLIB_DAILY_AMOUNT_UNIT = "cny"
TUSHARE_HAND_SIZE = 100
INDEX_VOLUME_POLICY = "excluded_non_tradable_benchmark"

MINUTE_EXECUTION_CONTRACT_VERSION = "minute-qlib-execution-v4-source-units"
QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION = "qlib-minute-resample-v1"

STRATEGY_EXECUTION_CONTRACT_VERSION = "qlib-strategy-execution-v1"
NATIVE_STRATEGY_FREQUENCIES = frozenset({"day", "1min", "5min"})
QLIB_RESAMPLED_FREQUENCIES = frozenset({"15min", "30min", "60min"})
SUPPORTED_STRATEGY_FREQUENCIES = frozenset(
    {*NATIVE_STRATEGY_FREQUENCIES, *QLIB_RESAMPLED_FREQUENCIES}
)
_MINUTE_FREQUENCY_SECONDS = {
    "1min": 60,
    "5min": 5 * 60,
    "15min": 15 * 60,
    "30min": 30 * 60,
    "60min": 60 * 60,
}

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


def strategy_frequency_mode(frequency: str) -> str:
    """Return the single supported data path for a strategy frequency."""

    normalized = str(frequency).strip().lower()
    if normalized not in SUPPORTED_STRATEGY_FREQUENCIES:
        raise ValueError(f"unsupported strategy frequency: {frequency}")
    return "qlib_resample" if normalized in QLIB_RESAMPLED_FREQUENCIES else "native"


def build_strategy_execution_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize the immutable signal-to-execution contract stored with a version.

    The contract intentionally contains only values known when a strategy version is
    created. Dataset identities are bound later by the backtest/simulation manifest;
    those manifests retain this hash and their own immutable lineage identifiers.
    """

    signal_frequency = str(config.get("signal_frequency") or "day").lower()
    execution_frequency = str(config.get("execution_frequency") or "day").lower()
    signal_mode = strategy_frequency_mode(signal_frequency)
    execution_mode = strategy_frequency_mode(execution_frequency)
    if execution_frequency in QLIB_RESAMPLED_FREQUENCIES:
        raise ValueError(
            "15/30/60-minute datasets are Qlib signal inputs, not execution data"
        )
    signal_period = int(config.get("signal_period") or 0)
    if signal_period < 1:
        raise ValueError("signal_period must be at least one bar")
    lag_bars = int(config.get("execution_lag_bars") or 0)
    if lag_bars != 1:
        raise ValueError("strategy execution must use exactly one next-bar lag")
    method = str(config.get("execution_method") or "open").lower()
    if method not in {"open", "twap", "vwap", "next_bar"}:
        raise ValueError("unsupported strategy execution method")
    if method == "open" and execution_frequency != "day":
        raise ValueError("open execution requires day frequency")
    if method in {"twap", "vwap"} and execution_frequency == "day":
        raise ValueError("TWAP/VWAP execution requires a minute frequency")
    if method == "next_bar" and signal_frequency == "day":
        raise ValueError("next-bar execution requires a minute signal")
    if method == "next_bar" and execution_frequency == "day":
        raise ValueError("next-bar execution requires a minute execution frequency")
    if signal_frequency != "day" and execution_frequency == "day":
        raise ValueError("minute signals cannot be executed through a daily bar")
    if signal_frequency != "day":
        signal_seconds = _MINUTE_FREQUENCY_SECONDS[signal_frequency]
        execution_seconds = _MINUTE_FREQUENCY_SECONDS[execution_frequency]
        if execution_seconds > signal_seconds:
            raise ValueError(
                "minute signal execution frequency must be equal to or finer than the signal"
            )

    return {
        "version": STRATEGY_EXECUTION_CONTRACT_VERSION,
        "signal": {
            "frequency": signal_frequency,
            "period_bars": signal_period,
            "data_mode": signal_mode,
            "rebalance_frequency": str(
                config.get("rebalance_frequency") or "day"
            ).lower(),
        },
        "execution": {
            "frequency": execution_frequency,
            "data_mode": execution_mode,
            "method": method,
            "days": int(config.get("execution_days") or 1),
            "lag_bars": lag_bars,
            "same_bar_execution": False,
            "slice_minutes": int(config.get("execution_slice_minutes") or 20),
            "max_slices": int(config.get("max_execution_slices") or 24),
        },
        "capacity": {
            "max_volume_participation": float(
                config.get("max_volume_participation") or 0.0
            ),
            "min_average_daily_amount": float(
                config.get("min_average_daily_amount") or 0.0
            ),
        },
        "cost_model": {
            key: config.get(key)
            for key in (
                "cost_schedule_version",
                "buy_commission_rate",
                "sell_commission_rate",
                "stock_sell_stamp_duty_rate",
                "etf_sell_stamp_duty_rate",
                "transfer_fee_rate",
                "annual_borrow_rate",
                "fixed_slippage_rate",
                "impact_at_max_participation",
                "min_commission",
            )
        },
        "risk_limits": {
            key: config.get(key)
            for key in (
                "max_position_weight",
                "max_daily_turnover",
                "max_industry_weight",
                "max_industry_deviation",
                "max_tracking_error",
                "target_volatility",
                "max_drawdown",
            )
        },
        "market_rules": "cn-ashare-t1-limits-fees-v1",
    }


def strategy_execution_contract_hash(config: dict[str, Any]) -> str:
    contract = build_strategy_execution_contract(config)
    encoded = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_strategy_execution_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = build_strategy_execution_contract(config)
    expected = strategy_execution_contract_hash(config)
    if config.get("execution_contract_hash") != expected:
        raise ValueError("strategy execution contract hash is missing or inconsistent")
    return contract


def require_next_bar_execution(
    signal_timestamp: datetime,
    execution_timestamp: datetime,
    *,
    signal_frequency: str,
    execution_frequency: str | None = None,
) -> None:
    """Reject same-bar fills at the shared backtest/replay boundary."""

    frequency = str(signal_frequency).lower()
    strategy_frequency_mode(frequency)
    if signal_timestamp.tzinfo != execution_timestamp.tzinfo:
        raise ValueError("signal and execution timestamps must use the same timezone")
    if frequency == "day":
        valid = execution_timestamp.date() > signal_timestamp.date()
    else:
        execution = str(execution_frequency or frequency).lower()
        strategy_frequency_mode(execution)
        if execution == "day":
            raise ValueError("minute signals cannot use daily next-bar execution")
        valid = execution_timestamp >= signal_timestamp + timedelta(
            seconds=_MINUTE_FREQUENCY_SECONDS[execution]
        )
    if not valid:
        raise ValueError("same-bar execution is forbidden; use the next eligible bar")


def require_daily_qlib_contract(provenance: dict[str, Any]) -> None:
    if provenance.get("frequency") != "day":
        raise ValueError("daily Qlib dataset provenance frequency is invalid")
    if provenance.get("field_contract_version") != DAILY_QLIB_FIELD_CONTRACT_VERSION:
        raise ValueError("daily Qlib dataset uses an obsolete field contract; rebuild it")
    if (
        provenance.get("source_volume_unit") != TUSHARE_DAILY_VOLUME_UNIT
        or provenance.get("qlib_volume_unit") != QLIB_DAILY_VOLUME_UNIT
        or provenance.get("source_amount_unit") != TUSHARE_DAILY_AMOUNT_UNIT
        or provenance.get("qlib_amount_unit") != QLIB_DAILY_AMOUNT_UNIT
        or int(provenance.get("source_hand_size") or 0) != TUSHARE_HAND_SIZE
        or provenance.get("index_volume_policy") != INDEX_VOLUME_POLICY
    ):
        raise ValueError("daily Qlib dataset volume/amount units are missing or invalid")
    if provenance.get("lineage_verified") is not True:
        raise ValueError("daily Qlib dataset lineage is not verified")


def require_minute_execution_contract(
    provenance: dict[str, Any],
    *,
    frequency: str | None = None,
    simulation_eligible: bool = False,
) -> None:
    if provenance.get("resampled") is True:
        raise ValueError("Qlib-resampled minute datasets are signal-only")
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


def require_minute_signal_contract(
    provenance: dict[str, Any], *, frequency: str | None = None
) -> None:
    """Validate native or Qlib-resampled minute data used to form signals."""

    actual_frequency = str(provenance.get("frequency") or "")
    if frequency and actual_frequency != frequency:
        raise ValueError("minute signal frequency does not match dataset provenance")
    if actual_frequency not in _MINUTE_FREQUENCY_SECONDS:
        raise ValueError("minute signal dataset frequency is unsupported")
    if provenance.get("resampled") is not True:
        require_minute_execution_contract(provenance, frequency=frequency)
        return
    if actual_frequency not in QLIB_RESAMPLED_FREQUENCIES:
        raise ValueError("resampled minute signal frequency must be 15/30/60 minutes")
    source_frequency = str(provenance.get("source_frequency") or "")
    if source_frequency not in {"1min", "5min"}:
        raise ValueError("Qlib-resampled signal has no native 1/5-minute source")
    if (
        int(actual_frequency.removesuffix("min"))
        % int(source_frequency.removesuffix("min"))
    ):
        raise ValueError("Qlib-resampled signal frequency is incompatible with its source")
    if (
        provenance.get("resample_contract_version")
        != QLIB_MINUTE_RESAMPLE_CONTRACT_VERSION
        or provenance.get("resample_engine") != "qlib.utils.resam.resam_calendar"
    ):
        raise ValueError("Qlib minute resample provenance is missing or obsolete")
    native_view = {
        **provenance,
        "frequency": source_frequency,
        "resampled": False,
    }
    require_minute_execution_contract(native_view, frequency=source_frequency)
