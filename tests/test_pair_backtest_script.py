from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "run_pair_backtest.py"
    spec = importlib.util.spec_from_file_location("run_pair_backtest_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parquet_pair_readers_filter_and_normalize_real_provider_fields(tmp_path: Path) -> None:
    script = _script_module()
    minute_path = tmp_path / "minute" / "year=2024"
    short_path = tmp_path / "shortability"
    minute_path.mkdir(parents=True)
    short_path.mkdir()
    pd.DataFrame(
        {
            "ts_code": ["510300.SH", "159919.SZ", "000001.SZ", "510300.SH"],
            "trade_time": [
                "2024-01-03 10:00:00",
                "2024-01-03 10:00:00",
                "2024-01-03 10:00:00",
                "2023-12-29 10:00:00",
            ],
            "close": [3.5, 4.2, 10.0, 3.4],
            "vol": [1_000_000, 900_000, 800_000, 700_000],
        }
    ).to_parquet(minute_path / "bars.parquet", index=False)
    pd.DataFrame(
        {
            "instrument": ["SH510300", "SZ159919", "SZ000001"],
            "trade_date": [20240103, 20240103, 20240103],
            "is_shortable": [1, 0, 1],
        }
    ).to_parquet(short_path / "eligibility.parquet", index=False)

    legs = {"SH510300", "SZ159919"}
    minute_raw = script._read_parquet_dataset(
        tmp_path / "minute", legs=legs, start="2024-01-01", end="2024-01-31"
    )
    minute = script._normalize_minute(
        minute_raw, legs=legs, start="2024-01-01", end="2024-01-31"
    )
    short_raw = script._read_parquet_dataset(
        short_path, legs=legs, start="2024-01-01", end="2024-01-31"
    )
    shortability = script._normalize_shortability(short_raw, legs=legs)

    assert set(minute.index.get_level_values("instrument")) == legs
    assert len(minute) == 2
    assert minute.loc[(pd.Timestamp("2024-01-03 10:00:00"), "SH510300"), "amount"] == 3_500_000
    assert set(shortability.index.get_level_values("instrument")) == legs
    assert shortability.loc[(pd.Timestamp("2024-01-03"), "SZ159919"), "shortable"] == 0


def test_shortability_normalizer_rejects_duplicate_evidence() -> None:
    script = _script_module()
    frame = pd.DataFrame(
        {
            "ts_code": ["510300.SH", "510300.SH"],
            "trade_date": ["20240103", "20240103"],
            "shortable": [1, 0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        script._normalize_shortability(frame, legs={"SH510300"})
