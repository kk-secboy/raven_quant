from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from quant_platform.eligibility import ELIGIBILITY_CONTRACT_VERSION

pytestmark = pytest.mark.no_database

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runners() -> tuple[ModuleType, ModuleType]:
    return (
        _load_script("runner_risk_backtest", "scripts/run_multifactor_backtest.py"),
        _load_script("runner_risk_recommendation", "scripts/run_recommendation_refresh.py"),
    )


def _eligibility_rows() -> pd.DataFrame:
    common = {
        "datetime": pd.Timestamp("2025-06-03"),
        "is_st": False,
        "delisted": False,
        "normal_listing_status": True,
        "equity": np.nan,
        "audit_opinion": np.nan,
        "financial_gate_required": False,
        "regulatory_data_available": True,
        "major_violation": False,
        "contract_version": ELIGIBILITY_CONTRACT_VERSION,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "instrument": "A",
                "eligible": True,
                "reasons": json.dumps([]),
                "suspended": False,
            },
            {
                **common,
                "instrument": "B",
                "eligible": False,
                "reasons": json.dumps(["suspended"]),
                "suspended": True,
            },
        ]
    )


def test_backtest_metadata_projects_pit_risk_and_freezes_non_tradable_price(
    runners: tuple[ModuleType, ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    backtest, _ = runners
    when = pd.Timestamp("2025-06-03")
    index = pd.MultiIndex.from_product(
        [[when], ["A", "B"]], names=["datetime", "instrument"]
    )
    execution = pd.DataFrame(
        {
            "$open": [10.0, 20.0],
            "$close": [10.5, 20.5],
            "Ref(Mean($amount, 20), 1)": [1_000_000.0, 2_000_000.0],
        },
        index=index,
    )
    closes = pd.DataFrame({"$close": [10.5, 20.5]}, index=index)
    memberships = pd.DataFrame(
        {
            "instrument": ["A", "B"],
            "in_date": [when, when],
            "out_date": [pd.NaT, pd.NaT],
            "industry": ["one", "two"],
        }
    )
    monkeypatch.setattr(backtest, "filter_available", lambda _name, frame, _when: frame)
    provider = backtest._metadata_provider(
        memberships,
        None,
        pd.DataFrame(),
        _eligibility_rows(),
        execution,
        closes,
        None,
        strategy_config={"portfolio_construction": "topk_equal_weight"},
    )
    result = provider(when, pd.Index(["A", "B"], dtype=str))

    assert result["instrument_risk_states"].to_dict() == {"A": "normal", "B": "watch"}
    assert result["prices"]["A"] == 10.0
    assert pd.isna(result["prices"]["B"])
    assert pd.isna(result["average_daily_values"]["B"])
    assert result["current_prices"]["B"] == 20.5
    assert result["industries"].to_dict() == {"A": "one", "B": "two"}
    assert "style_exposures" not in result


def test_recommendation_execution_evidence_blocks_missing_close_and_suspension(
    runners: tuple[ModuleType, ModuleType]
) -> None:
    _, recommendation = runners
    point = pd.DataFrame(
        {
            "$open": [10.0, 20.0],
            "$close": [np.nan, 20.5],
            "Ref(Mean($amount, 20), 1)": [1_000_000.0, 2_000_000.0],
        },
        index=pd.Index(["A", "B"], dtype=str),
    )
    projection = pd.DataFrame(
        {"tradable": [True, False]}, index=pd.Index(["A", "B"], dtype=str)
    )

    prices, current, daily_values = recommendation._prepare_execution_evidence(
        point,
        instruments=pd.Index(["A", "B"], dtype=str),
        risk_projection=projection,
    )

    assert pd.isna(prices["A"])
    assert pd.isna(prices["B"])
    assert pd.isna(current["A"])
    assert current["B"] == 20.5
    assert daily_values["A"] == 1_000_000.0
    assert pd.isna(daily_values["B"])


def test_reference_price_fallback_is_only_for_frozen_existing_holdings(
    runners: tuple[ModuleType, ModuleType]
) -> None:
    _, recommendation = runners
    history = pd.DataFrame(
        {"HELD": [9.5, 10.0, np.nan], "NEW": [4.8, 5.0, np.nan]},
        index=pd.bdate_range("2025-05-30", periods=3),
    )
    prices, sources = recommendation._resolve_reference_prices(
        {"HELD": 0.4},
        current_prices=pd.Series({"HELD": np.nan}),
        close_history=history,
        previous_weights={"HELD": 0.4},
        frozen_instruments={"HELD"},
    )
    assert prices.to_dict() == {"HELD": 10.0}
    assert sources == {"HELD": "latest_positive_pit_close"}

    with pytest.raises(ValueError, match="new or tradable target NEW"):
        recommendation._resolve_reference_prices(
            {"NEW": 0.2},
            current_prices=pd.Series({"NEW": np.nan}),
            close_history=history,
            previous_weights={},
            frozen_instruments={"NEW"},
        )
