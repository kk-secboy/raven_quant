import json
from pathlib import Path

import pandas as pd
import pytest

from quant_data.minute_qlib_builder import MINUTE_QLIB_FIELDS, MinuteQlibBuilder

pytestmark = pytest.mark.no_database


def _snapshot(tmp_path: Path, *, frequency: str = "1min") -> Path:
    snapshot = tmp_path / "minute-snapshot"
    target = snapshot / "parquet" / "etf_1m" / "partition_year=2024"
    target.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "trade_time": "2024-01-02 09:31:00",
                "open": 3.50,
                "high": 3.52,
                "low": 3.49,
                "close": 3.51,
                "vol": 1000,
                "amount": 3510,
            },
            {
                "ts_code": "510300.SH",
                "trade_time": "2024-01-02 09:32:00",
                "open": 3.51,
                "high": 3.54,
                "low": 3.50,
                "close": 3.53,
                "vol": 1200,
                "amount": 4236,
            },
        ]
    ).to_parquet(target / "bars.parquet")
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "frequency": frequency,
                "lineage_id": "a" * 64,
                "start_date": "2024-01-02",
                "end_date": "2024-01-02",
                "datasets": {"etf_1m": {}},
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def test_builds_minute_qlib_staging_from_execution_snapshot(tmp_path: Path) -> None:
    builder = MinuteQlibBuilder(_snapshot(tmp_path))

    by_symbol = builder.build_staging(tmp_path / "minute-staging")

    frame = pd.read_parquet(by_symbol / "SH510300.parquet")
    assert frame["symbol"].tolist() == ["SH510300", "SH510300"]
    assert frame["date"].dt.strftime("%H:%M").tolist() == ["09:31", "09:32"]
    assert frame["vwap"].tolist() == pytest.approx([3.51, 3.53])
    assert pd.isna(frame["change"].iloc[0])
    assert frame["change"].iloc[1] == pytest.approx(3.53 / 3.51 - 1)
    assert set(MINUTE_QLIB_FIELDS).issubset(frame.columns)


def test_rejects_non_minute_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1min"):
        MinuteQlibBuilder(_snapshot(tmp_path, frequency="day"))
