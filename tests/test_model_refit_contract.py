from pathlib import Path

import pandas as pd
import pytest

from quant_platform.model_research_governance import MODEL_REFIT_POLICY
from scripts.run_model_refit import _calendar


def test_live_refit_window_is_rolling_purged_and_single_day(tmp_path: Path) -> None:
    provider = tmp_path / "qlib"
    (provider / "calendars").mkdir(parents=True)
    required = sum(
        int(MODEL_REFIT_POLICY[key])
        for key in (
            "train_trading_days",
            "validation_trading_days",
            "embargo_trading_days",
            "prediction_trading_days",
        )
    )
    days = [
        value.date().isoformat()
        for value in pd.bdate_range("2018-01-01", periods=required + 10)
    ]
    (provider / "calendars" / "day.txt").write_text("\n".join(days), encoding="utf-8")

    _, periods = _calendar(provider, days[-1], dict(MODEL_REFIT_POLICY))

    assert periods["test_start"] == periods["test_end"] == days[-1]
    assert days.index(periods["valid_start"]) - days.index(periods["train_end"]) == 1
    assert days.index(periods["test_start"]) - days.index(periods["valid_end"]) == (
        int(MODEL_REFIT_POLICY["embargo_trading_days"]) + 1
    )
    assert (
        days.index(periods["train_end"]) - days.index(periods["train_start"]) + 1
        == MODEL_REFIT_POLICY["train_trading_days"]
    )


def test_live_refit_rejects_non_trading_signal_date(tmp_path: Path) -> None:
    provider = tmp_path / "qlib"
    (provider / "calendars").mkdir(parents=True)
    required = sum(
        int(MODEL_REFIT_POLICY[key])
        for key in (
            "train_trading_days",
            "validation_trading_days",
            "embargo_trading_days",
            "prediction_trading_days",
        )
    )
    days = [
        value.date().isoformat()
        for value in pd.bdate_range("2018-01-01", periods=required)
    ]
    (provider / "calendars" / "day.txt").write_text("\n".join(days), encoding="utf-8")

    with pytest.raises(ValueError, match="signal date"):
        _calendar(provider, "2099-01-01", dict(MODEL_REFIT_POLICY))
