from __future__ import annotations

from collections import OrderedDict
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from quant_data.baostock_provider import (
    BaoStockProvider,
    baostock_code,
    tushare_code,
)
from quant_data.cli import app
from quant_data.legacy_market import (
    a_share_baostock_codes,
    baostock_history_specs,
    baostock_reference_specs,
    compare_baostock_overlap,
)
from quant_data.runner import DownloadRunner
from quant_platform.api import BaoStockOverlapRequest, LegacyMarketBackfillRequest
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def test_primary_worker_accepts_legacy_market_jobs() -> None:
    compose = (Path(__file__).parents[1] / "deploy" / "compose.yaml").read_text(
        encoding="utf-8"
    )
    worker_block = compose.split("\n  worker:\n", maxsplit=1)[1].split(
        "\n  scheduler:\n", maxsplit=1
    )[0]

    assert "baostock_overlap_validation" in worker_block
    assert "legacy_market_backfill" in worker_block


def test_security_code_conversion_is_explicit_and_reversible() -> None:
    assert tushare_code("sh.600000") == "600000.SH"
    assert tushare_code("sz.000001") == "000001.SZ"
    assert baostock_code("600000.SH") == "sh.600000"
    with pytest.raises(ValueError, match="unsupported"):
        tushare_code("hk.00700")


def test_a_share_universe_keeps_delisted_stocks_that_overlap_history() -> None:
    frame = pd.DataFrame(
        [
            {
                "code": "sh.600000",
                "ipo_date": "1999-11-10",
                "out_date": "",
                "security_type": "1",
            },
            {
                "code": "sz.000003",
                "ipo_date": "1991-04-03",
                "out_date": "2014-06-16",
                "security_type": "1",
            },
            {
                "code": "sz.300001",
                "ipo_date": "2009-10-30",
                "out_date": "",
                "security_type": "1",
            },
            {
                "code": "sh.900901",
                "ipo_date": "1992-02-21",
                "out_date": "",
                "security_type": "1",
            },
            {
                "code": "sh.000001",
                "ipo_date": "1991-07-15",
                "out_date": "",
                "security_type": "2",
            },
            {
                "code": "sh.605001",
                "ipo_date": "2020-08-18",
                "out_date": "",
                "security_type": "1",
            },
        ]
    )

    assert a_share_baostock_codes(
        frame,
        start=date(2008, 1, 1),
        end=date(2015, 12, 31),
    ) == ["sh.600000", "sz.000003", "sz.300001"]


def test_legacy_specs_preserve_source_and_contract_lineage() -> None:
    references = baostock_reference_specs(
        date(2008, 1, 1),
        date(2015, 12, 31),
        max_attempts=3,
    )
    assert [(spec.dataset, spec.api_name) for spec in references] == [
        ("trade_cal", "baostock_trade_cal"),
        ("baostock_stock_basic", "baostock_stock_basic"),
    ]

    specs = baostock_history_specs(
        ["sh.600000"],
        date(2008, 1, 1),
        date(2015, 12, 31),
        max_attempts=3,
    )
    assert {spec.dataset for spec in specs} == {"daily", "daily_basic", "adj_factor"}
    assert all(spec.scope["source"] == "baostock-0.9.3" for spec in specs)
    assert {spec.scope["contract"] for spec in specs} == {
        "daily",
        "daily_basic",
        "adj_factor",
    }


def _provider_with_history(
    raw_rows: list[dict[str, str]],
    adjusted_rows: list[dict[str, str]] | None = None,
) -> BaoStockProvider:
    provider = BaoStockProvider.__new__(BaoStockProvider)
    provider._closed = False
    provider._history_cache = OrderedDict()
    provider._lock = None

    def history(
        _code: str,
        _start: str,
        _end: str,
        adjustflag: str,
    ) -> tuple[list[str], list[dict[str, str]]]:
        rows = adjusted_rows if adjustflag == "1" else raw_rows
        assert rows is not None
        return list(rows[0]) if rows else [], rows

    provider._history_rows = history  # type: ignore[method-assign]
    return provider


def test_daily_mapping_converts_baostock_units_to_tushare_contract() -> None:
    raw = [
        {
            "date": "2015-12-31",
            "code": "sh.600000",
            "open": "18.52",
            "high": "18.52",
            "low": "18.26",
            "close": "18.27",
            "preclose": "18.57",
            "volume": "27936138",
            "amount": "513758896",
            "turn": "0.149764",
            "tradestatus": "1",
            "pctChg": "-1.6155",
            "peTTM": "6.9003",
            "psTTM": "2.399317",
            "pbMRQ": "1.260607",
            "isST": "0",
        }
    ]
    provider = _provider_with_history(raw)
    result = provider.fetch(
        "baostock_daily",
        {"code": "sh.600000", "start_date": "2015-12-31", "end_date": "2015-12-31"},
    )

    assert result.rows == [
        {
            "ts_code": "600000.SH",
            "trade_date": "20151231",
            "open": 18.52,
            "high": 18.52,
            "low": 18.26,
            "close": 18.27,
            "pre_close": 18.57,
            "change": pytest.approx(-0.3),
            "pct_chg": -1.6155,
            "vol": 279361.38,
            "amount": 513758.896,
        }
    ]


def test_adjustment_factor_uses_back_adjusted_to_raw_close_ratio() -> None:
    raw = [
        {
            "date": "2015-12-31",
            "close": "10",
            "tradestatus": "1",
        }
    ]
    adjusted = [
        {
            "date": "2015-12-31",
            "close": "25",
            "tradestatus": "1",
        }
    ]
    provider = _provider_with_history(raw, adjusted)
    result = provider.fetch(
        "baostock_adj_factor",
        {"code": "sh.600000", "start_date": "2015-12-31", "end_date": "2015-12-31"},
    )

    assert result.rows == [
        {
            "ts_code": "600000.SH",
            "trade_date": "20151231",
            "adj_factor": 2.5,
        }
    ]


def test_runner_can_claim_only_explicit_legacy_unit_keys() -> None:
    class Checkpoint:
        def __init__(self) -> None:
            self.claimed: list[
                tuple[set[str] | None, set[str] | None, set[str] | None]
            ] = []

        def reset_stale(self) -> int:
            return 0

        def claim(
            self,
            datasets: set[str] | None = None,
            *,
            unit_keys: set[str] | None = None,
            api_names: set[str] | None = None,
        ) -> None:
            self.claimed.append((datasets, unit_keys, api_names))
            return None

    checkpoint = Checkpoint()
    runner = DownloadRunner(
        checkpoint=checkpoint,  # type: ignore[arg-type]
        storage=object(),  # type: ignore[arg-type]
        provider=object(),  # type: ignore[arg-type]
        workers=1,
    )

    summary = runner.run(unit_keys={"legacy-a", "legacy-b"})

    assert summary.succeeded == 0
    assert checkpoint.claimed == [(None, {"legacy-a", "legacy-b"}, None)]

    checkpoint.claimed.clear()
    runner.run({"daily"}, api_names={"baostock_daily"})
    assert checkpoint.claimed == [({"daily"}, None, {"baostock_daily"})]


def _overlap_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = ["2016-01-04", "2016-01-05", "2016-01-06"]
    daily = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_date": item,
                "open": 10 + offset,
                "high": 11 + offset,
                "low": 9 + offset,
                "close": 10.5 + offset,
                "pre_close": 10 + offset,
                "pct_chg": 5,
                "vol": 1000 + offset,
                "amount": 2000 + offset,
            }
            for offset, item in enumerate(dates)
        ]
    )
    adj = pd.DataFrame(
        [
            {
                "ts_code": "600000.SH",
                "trade_date": item,
                "adj_factor": factor,
            }
            for item, factor in zip(dates, [2, 2, 2.2], strict=True)
        ]
    )
    return daily, daily.copy(), adj, adj.assign(adj_factor=[5, 5, 5.5])


def test_overlap_gate_accepts_equal_contracts_and_normalized_adjustment_path() -> None:
    tushare_daily, baostock_daily, tushare_adj, baostock_adj = _overlap_frames()

    report = compare_baostock_overlap(
        tushare_daily,
        baostock_daily,
        tushare_adj,
        baostock_adj,
        symbols=["600000.SH"],
        min_days_per_symbol=3,
    )

    assert report["ok"] is True
    assert report["errors"] == []


def test_overlap_gate_rejects_volume_unit_mismatch() -> None:
    tushare_daily, baostock_daily, tushare_adj, baostock_adj = _overlap_frames()
    baostock_daily["vol"] *= 100

    report = compare_baostock_overlap(
        tushare_daily,
        baostock_daily,
        tushare_adj,
        baostock_adj,
        symbols=["600000.SH"],
        min_days_per_symbol=3,
    )

    assert report["ok"] is False
    assert any("volume_relative_error_p99" in item for item in report["errors"])


def test_legacy_cli_refuses_import_without_overlap_report() -> None:
    result = CliRunner().invoke(
        app,
        ["bootstrap-legacy-market", "--start", "2008-01-01", "--end", "2015-12-31"],
    )

    assert result.exit_code != 0
    assert "validation-report" in result.output


def test_legacy_api_models_enforce_disjoint_source_periods() -> None:
    assert BaoStockOverlapRequest().start == date(2016, 1, 1)
    assert LegacyMarketBackfillRequest().end == date(2015, 12, 31)
    with pytest.raises(ValidationError, match="before 2016"):
        LegacyMarketBackfillRequest(start=date(2008, 1, 1), end=date(2016, 1, 1))


def test_worker_builds_legacy_commands_without_tushare_secrets() -> None:
    worker = LocalJobWorker.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root="unused")
    validation_path = Path("quality/overlap.json")
    result_path = Path("legacy/result.json")

    command, result, environment = worker._command(
        {
            "kind": "legacy_market_backfill",
            "payload": {
                "start": "2008-01-01",
                "end": "2015-12-31",
                "validation_report": str(validation_path),
                "result_path": str(result_path),
            },
        }
    )

    assert command[3:] == [
        "bootstrap-legacy-market",
        "--start",
        "2008-01-01",
        "--end",
        "2015-12-31",
        "--validation-report",
        str(validation_path),
        "--result",
        str(result_path),
    ]
    assert result == result_path
    assert environment == {}
