from __future__ import annotations

import pandas as pd
import pytest

from quant_data.universe import select_intraday_universe

pytestmark = pytest.mark.no_database


def test_selects_major_assets_and_dynamic_liquidity() -> None:
    frames = {
        "daily": pd.DataFrame(
            [
                {"ts_code": "600000.SH", "trade_date": "2024-01-02", "amount": 100},
                {"ts_code": "600001.SH", "trade_date": "2024-01-02", "amount": 300},
                {"ts_code": "600002.SH", "trade_date": "2024-01-02", "amount": 500},
            ]
        ),
        "stock_basic": pd.DataFrame(
            [
                {"ts_code": "600000.SH", "name": "正常股票"},
                {"ts_code": "600001.SH", "name": "另一个股票"},
                {"ts_code": "600002.SH", "name": "ST样本"},
            ]
        ),
        "fut_mapping": pd.DataFrame(
            [
                {
                    "ts_code": "IF.CFX",
                    "mapping_ts_code": "IF2401.CFX",
                    "trade_date": "2024-01-02",
                },
                {
                    "ts_code": "IM.CFX",
                    "mapping_ts_code": "IM2401.CFX",
                    "trade_date": "2024-01-02",
                },
            ]
        ),
        "opt_daily": pd.DataFrame(
            [
                {"ts_code": "10000001.SH", "trade_date": "2024-01-02", "amount": 20},
                {"ts_code": "10000002.SH", "trade_date": "2024-01-02", "amount": 40},
                {"ts_code": "MO2401.CFX", "trade_date": "2024-01-02", "amount": 100},
            ]
        ),
    }
    selection = select_intraday_universe(frames, max_stocks=2, max_options=1)

    assert selection.symbols_by_dataset["indices_1m"] == [
        "000016.SH",
        "000300.SH",
        "000852.SH",
        "000905.SH",
    ]
    assert selection.symbols_by_dataset["liquid_stocks_1m"] == [
        "600001.SH",
        "600000.SH",
    ]
    assert selection.symbols_by_dataset["futures_1m"] == [
        "IF2401.CFX",
        "IM2401.CFX",
    ]
    assert "reconstructed locally" in selection.evidence["rules"]["futures"]
    assert selection.symbols_by_dataset["options_1m"] == ["10000002.SH"]
    assert selection.evidence["source_dates"]["stocks"] == "2024-01-02"


def test_rejects_unknown_etf_category() -> None:
    with pytest.raises(ValueError, match="unknown ETF categories"):
        select_intraday_universe({}, etf_categories=("unknown",))
