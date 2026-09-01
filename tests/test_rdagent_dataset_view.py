from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from quant_platform.rdagent_dataset_view import (
    isolate_rdagent_periods,
    prepare_rdagent_dataset_view,
)

pytestmark = pytest.mark.no_database


def _write_dataset(root: Path, *, cn_all: str | None) -> None:
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    feature = root / "features" / "sh600000" / "close.day.bin"
    feature.parent.mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text(
        "2024-01-02\n2024-01-03\n2024-01-04\n",
        encoding="utf-8",
    )
    (root / "instruments" / "all.txt").write_text(
        "SH000300\t2024-01-02\t2024-01-04\n"
        "SH600000\t2024-01-02\t2024-01-04\n",
        encoding="utf-8",
    )
    if cn_all is not None:
        (root / "instruments" / "cn_all.txt").write_text(
            cn_all,
            encoding="utf-8",
        )
    feature.write_bytes(struct.pack("<4f", 0.0, 10.0, 11.0, 12.0))


def test_rdagent_internal_test_is_derived_only_from_pre_final_history() -> None:
    periods = isolate_rdagent_periods(
        {
            "train_start": "2018-01-01",
            "train_end": "2021-12-31",
            "valid_start": "2022-01-01",
            "valid_end": "2025-12-31",
            "test_start": "2026-01-08",
            "test_end": "2026-12-31",
        }
    )

    assert periods["train_start"] == "2018-01-01"
    assert periods["train_end"] == "2021-12-31"
    assert periods["valid_start"] == "2022-01-01"
    assert periods["test_end"] == "2025-12-31"
    assert periods["valid_end"] < periods["test_start"]
    assert periods["test_end"] < "2026-01-08"


def test_rdagent_period_isolation_rejects_short_validation_history() -> None:
    with pytest.raises(ValueError, match="at least 126 calendar days"):
        isolate_rdagent_periods(
            {
                "train_start": "2023-01-01",
                "train_end": "2023-12-31",
                "valid_start": "2024-01-01",
                "valid_end": "2024-03-31",
                "test_start": "2024-04-08",
                "test_end": "2025-04-30",
            }
        )


def test_dataset_view_requires_a_nonempty_cn_all_universe(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    _write_dataset(missing, cn_all=None)
    with pytest.raises(ValueError, match="governed cn_all"):
        prepare_rdagent_dataset_view(
            missing,
            tmp_path / "missing-view",
            cutoff="2024-01-03",
        )

    empty = tmp_path / "empty"
    _write_dataset(empty, cn_all="")
    with pytest.raises(ValueError, match="cn_all universe is empty"):
        prepare_rdagent_dataset_view(
            empty,
            tmp_path / "empty-view",
            cutoff="2024-01-03",
        )


def test_dataset_view_seals_the_governed_cn_all_universe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "view"
    _write_dataset(
        source,
        cn_all="SH600000\t2024-01-02\t2024-01-04\n",
    )

    prepare_rdagent_dataset_view(
        source,
        destination,
        cutoff="2024-01-03",
    )

    instruments = destination / "instruments" / "cn_all.txt"
    assert instruments.read_text(encoding="utf-8") == (
        "SH600000\t2024-01-02\t2024-01-03\n"
    )
    default_instruments = destination / "instruments" / "all.txt"
    assert default_instruments.read_bytes() == instruments.read_bytes()
    manifest = json.loads(
        (destination / "quantlab-rdagent-dataset-view.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == 3
    assert manifest["market"] == "cn_all"
    assert manifest["default_market_alias"] == "cn_all"
    assert manifest["calendar_rows"] == 2
    assert manifest["future_calendar_rows"] == 3
    assert manifest["future_calendar_boundary"] == "2024-01-04"
    assert manifest["future_calendar_contains_market_data"] is False
    assert (destination / "calendars" / "day.txt").read_text(
        encoding="utf-8"
    ) == "2024-01-02\n2024-01-03\n"
    assert (destination / "calendars" / "day_future.txt").read_text(
        encoding="utf-8"
    ) == "2024-01-02\n2024-01-03\n2024-01-04\n"
    feature = destination / "features" / "sh600000" / "close.day.bin"
    assert struct.unpack("<3f", feature.read_bytes()) == (0.0, 10.0, 11.0)

    assert (
        prepare_rdagent_dataset_view(
            source,
            destination,
            cutoff="2024-01-03",
        )
        == destination.resolve()
    )

    canonical = instruments.read_bytes()
    default_instruments.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid cn_all universe"):
        prepare_rdagent_dataset_view(
            source,
            destination,
            cutoff="2024-01-03",
        )
    default_instruments.write_bytes(canonical)

    future_calendar = destination / "calendars" / "day_future.txt"
    future_calendar.write_text(
        "2024-01-02\n2024-01-03\n2024-01-04\n2024-01-05\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid calendar"):
        prepare_rdagent_dataset_view(
            source,
            destination,
            cutoff="2024-01-03",
        )
    future_calendar.write_bytes(b"2024-01-02\n2024-01-03\n2024-01-04\n")

    instruments.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid cn_all universe"):
        prepare_rdagent_dataset_view(
            source,
            destination,
            cutoff="2024-01-03",
        )


def test_dataset_view_normalizes_qlib_timestamp_instrument_intervals(
    tmp_path: Path,
) -> None:
    source = tmp_path / "timestamp-source"
    destination = tmp_path / "timestamp-view"
    _write_dataset(
        source,
        cn_all=(
            "SH600000\t2024-01-02 00:00:00\t2024-01-04 00:00:00\n"
        ),
    )

    prepare_rdagent_dataset_view(
        source,
        destination,
        cutoff="2024-01-03",
    )

    assert (destination / "instruments" / "cn_all.txt").read_text(
        encoding="utf-8"
    ) == "SH600000\t2024-01-02\t2024-01-03\n"


def test_dataset_view_requires_a_later_interval_boundary(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(
        source,
        cn_all="SH600000\t2024-01-02\t2024-01-04\n",
    )

    with pytest.raises(ValueError, match="no next trading-session interval boundary"):
        prepare_rdagent_dataset_view(
            source,
            tmp_path / "view",
            cutoff="2024-01-04",
        )
