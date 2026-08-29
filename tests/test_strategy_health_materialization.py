from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from quant_platform.strategy_feature_drift_source import StrategyFeatureDriftSource
from scripts.materialize_factor_library import (
    _bounded_window_from_calendar,
    _persist_one,
    _query_features,
)

pytestmark = pytest.mark.no_database


def test_strategy_health_query_uses_last_64_calendar_sessions(tmp_path: Path) -> None:
    calendar = pd.bdate_range("2024-01-02", periods=700)
    calendar_path = tmp_path / "calendars" / "day.txt"
    calendar_path.parent.mkdir(parents=True)
    calendar_path.write_text(
        "\n".join(item.date().isoformat() for item in calendar) + "\n",
        encoding="utf-8",
    )
    start, end = _bounded_window_from_calendar(
        tmp_path,
        requested_start=calendar[0].date().isoformat(),
        requested_end=calendar[-1].date().isoformat(),
        session_limit=64,
    )
    calls: list[dict] = []

    class FakeData:
        @staticmethod
        def features(_instruments, _expressions, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame()

    _query_features(
        FakeData,
        "cn_all",
        ["$close"],
        materialized_start=start,
        materialized_end=end,
    )

    assert start == calendar[-64].date().isoformat()
    assert end == calendar[-1].date().isoformat()
    assert calls == [{"start_time": start, "end_time": end, "freq": "day"}]


def test_strategy_health_recent_only_writes_no_full_hdf(tmp_path: Path) -> None:
    dates = pd.bdate_range("2026-05-04", periods=80)
    index = pd.MultiIndex.from_product(
        [["SH600000", "SH600001"], dates],
        names=["instrument", "datetime"],
    )
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    checkpoint = {"completed": {}}
    values_root = tmp_path / "values"
    values_root.mkdir()

    _persist_one(
        values_root,
        "factor-a",
        values,
        checkpoint,
        storage_mode="recent_only",
        recent_session_limit=64,
    )

    entry = checkpoint["completed"]["factor-a"]
    assert entry["recent_session_limit"] == 64
    assert entry["recent_rows"] == 128
    assert not list(values_root.glob("*.h5"))
    assert "relative_path" not in entry


def test_long_model_label_uses_one_bounded_316_session_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = "a" * 64
    lineage = "b" * 64
    dataset = tmp_path / "qlib" / "daily-a"
    metadata = dataset / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "provenance.json").write_text(
        '{"dataset_identity_sha256":"' + identity + '",'
        '"dataset_lineage_id":"' + lineage + '"}',
        encoding="utf-8",
    )
    calendar = pd.bdate_range("2023-01-02", periods=700)
    calendar_path = dataset / "calendars" / "day.txt"
    calendar_path.parent.mkdir()
    calendar_path.write_text(
        "\n".join(item.date().isoformat() for item in calendar) + "\n",
        encoding="utf-8",
    )
    calls: list[dict] = []
    observed_index = pd.MultiIndex.from_product(
        [calendar[-316:], ["SH600000", "SH600001"]],
        names=["datetime", "instrument"],
    )

    class FakeData:
        @staticmethod
        def instruments(universe: str) -> str:
            return universe

        @staticmethod
        def features(_instruments, _expressions, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame(
                {"label": np.linspace(-0.1, 0.1, len(observed_index))},
                index=observed_index,
            )

    qlib = ModuleType("qlib")
    qlib.init = lambda **_kwargs: None
    qlib_data = ModuleType("qlib.data")
    qlib_data.D = FakeData
    monkeypatch.setitem(__import__("sys").modules, "qlib", qlib)
    monkeypatch.setitem(__import__("sys").modules, "qlib.data", qlib_data)
    source = StrategyFeatureDriftSource.__new__(StrategyFeatureDriftSource)
    source.data_root = tmp_path.resolve()

    values, digest, provenance_digest = source._current_label(
        materialization={"dataset_path": dataset},
        current_dataset_identity_sha256=identity,
        current_dataset_lineage_id=lineage,
        label_contract={
            "label_expression": "Ref($close,-252)/$close-1",
            "label_horizon_sessions": 252,
        },
        signal_date=calendar[-1].date(),
        session_limit=316,
    )

    assert len(values) == 632
    assert len(digest) == len(provenance_digest) == 64
    assert calls == [
        {
            "start_time": calendar[-316].date().isoformat(),
            "end_time": calendar[-1].date().isoformat(),
            "freq": "day",
        }
    ]
