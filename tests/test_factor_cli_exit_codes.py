"""Exit-code contract for the fail-closed factor production commands.

Schedulers and the durable job worker key success off the process exit code,
so a fail-closed data error (short trade calendar, missing parquets) must
terminate the command with a non-zero exit instead of a traceback that an
invocation wrapper can flatten to 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from quant_data.cli import app

pytestmark = pytest.mark.no_database


def _write_parquet(directory: Path, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / "data.parquet", index=False)


def _seed_trade_cal(data_root: Path) -> None:
    open_days = list(pd.bdate_range("2024-01-01", periods=60).date)
    rows = [{"cal_date": day.strftime("%Y%m%d"), "is_open": 1} for day in open_days]
    _write_parquet(data_root / "units" / "trade_cal", rows)


def _seed_report_rc(data_root: Path, report_date: str) -> None:
    _write_parquet(
        data_root / "units" / "report_rc",
        [
            {
                "ts_code": "000001.SZ",
                "name": "测试股",
                "report_date": report_date,
                "report_title": "公司点评",
                "report_type": "一般报告",
                "classify": "一般报告",
                "org_name": "测试券商",
                "author_name": "分析师",
                "quarter": "2024Q1",
                "eps": 1.0,
                "np": 1000.0,
                "max_price": 20.0,
                "min_price": 15.0,
                "rating": "买入",
            }
        ],
    )


def test_report_rc_factors_fail_closed_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A report_date beyond the persisted trade_cal end cannot derive an
    # available_at; the run must fail closed with a non-zero exit code.
    _seed_trade_cal(tmp_path)
    _seed_report_rc(tmp_path, "2026-07-27")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    result = CliRunner().invoke(
        app, ["report-rc-factors", "--start", "2026-07-20", "--end", "2026-07-27"]
    )

    assert result.exit_code == 2, result.output
    assert "fail-closed" in result.output
    assert "2026-07-27" in result.output
    # A failed run produces no factor artifacts.
    assert not (tmp_path / "report_rc" / "factors").exists()


def test_report_rc_factors_success_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_trade_cal(tmp_path)
    _seed_report_rc(tmp_path, "2024-01-10")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    result = CliRunner().invoke(
        app, ["report-rc-factors", "--start", "2024-01-01", "--end", "2024-01-31"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dataset"] == "report_rc"
    assert payload["reports"] == 1


def test_news_flash_factors_fail_closed_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    result = CliRunner().invoke(app, ["news-flash-factors"])

    assert result.exit_code == 2, result.output
    assert "fail-closed" in result.output


def test_major_news_mentions_fail_closed_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    result = CliRunner().invoke(app, ["major-news-mentions"])

    assert result.exit_code == 2, result.output
    assert "fail-closed" in result.output
