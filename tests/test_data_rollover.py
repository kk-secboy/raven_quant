import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from quant_platform.data_rollover import (
    next_qlib_trading_date,
    select_execution_snapshot,
    select_qlib_dataset,
)


def _units(*keys: str, changed: bool = False) -> list[dict[str, object]]:
    return [
        {
            "unit_key": key,
            "sha256": (("f" if changed and index == 0 else str(index + 1)) * 64),
            "row_count": index + 1,
        }
        for index, key in enumerate(keys)
    ]


def _snapshot(
    root: Path,
    name: str,
    *,
    lineage_id: str,
    end: str,
    units: list[dict[str, object]],
    execution: bool = False,
) -> None:
    path = root / "snapshots" / name
    path.mkdir(parents=True)
    datasets: dict[str, dict[str, object]] = {}
    names = ("etf_1m", "margin_eligibility") if execution else ("daily",)
    for dataset in names:
        data_path = path / "parquet" / dataset / "data.parquet"
        data_path.parent.mkdir(parents=True)
        data_path.write_bytes(f"{name}:{dataset}".encode())
        datasets[dataset] = {
            "rows": len(units),
            "source_sha256": hashlib.sha256(dataset.encode()).hexdigest(),
            "source_units": units,
            "files": [
                {
                    "path": data_path.relative_to(path).as_posix(),
                    "bytes": data_path.stat().st_size,
                    "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                }
            ],
        }
    manifest = {
        "name": name,
        "lineage_id": lineage_id,
        "lineage_generation": len(units) - 1,
        "start_date": "2024-01-01",
        "end_date": end,
        "datasets": datasets,
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _qlib(
    root: Path,
    name: str,
    *,
    snapshot: str,
    lineage_id: str,
    days: list[str],
    verified: bool = True,
) -> None:
    path = root / "qlib" / name
    (path / "calendars").mkdir(parents=True)
    (path / "instruments").mkdir()
    (path / "features").mkdir()
    (path / "metadata").mkdir()
    (path / "calendars" / "day.txt").write_text("\n".join(days), encoding="utf-8")
    (path / "instruments" / "cn_all.txt").write_text("SH600000\n", encoding="utf-8")
    provenance = {
        "snapshot_name": snapshot,
        "snapshot_manifest_sha256": "a" * 64,
        "dataset_identity_sha256": hashlib.sha256(name.encode()).hexdigest(),
        "dataset_lineage_id": lineage_id,
        "lineage_verified": verified,
    }
    (path / "metadata" / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


def test_selects_latest_verified_qlib_descendant(tmp_path: Path) -> None:
    source_lineage = "a" * 64
    dataset_lineage = "b" * 64
    _snapshot(
        tmp_path,
        "source-v1",
        lineage_id=source_lineage,
        end="2024-01-02",
        units=_units("day-1"),
    )
    _snapshot(
        tmp_path,
        "source-v2",
        lineage_id=source_lineage,
        end="2024-01-04",
        units=_units("day-1", "day-2"),
    )
    _snapshot(
        tmp_path,
        "source-rewritten",
        lineage_id=source_lineage,
        end="2024-01-05",
        units=_units("day-1", "day-2", changed=True),
    )
    _qlib(
        tmp_path,
        "qlib-v1",
        snapshot="source-v1",
        lineage_id=dataset_lineage,
        days=["2024-01-02"],
    )
    _qlib(
        tmp_path,
        "qlib-v2",
        snapshot="source-v2",
        lineage_id=dataset_lineage,
        days=["2024-01-02", "2024-01-03", "2024-01-04"],
    )
    _qlib(
        tmp_path,
        "qlib-rewritten",
        snapshot="source-rewritten",
        lineage_id=dataset_lineage,
        days=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    )

    selected = select_qlib_dataset(
        tmp_path,
        anchor_name="qlib-v1",
        roll_policy="latest_compatible",
        lineage_id=dataset_lineage,
        required_date=date(2024, 1, 3),
        require_later_date=True,
    )

    assert selected["name"] == "qlib-v2"
    assert next_qlib_trading_date(selected, date(2024, 1, 3)) == date(2024, 1, 4)


def test_rejects_unverified_qlib_anchor_for_latest_policy(tmp_path: Path) -> None:
    source_lineage = "a" * 64
    dataset_lineage = "b" * 64
    _snapshot(
        tmp_path,
        "source-v1",
        lineage_id=source_lineage,
        end="2024-01-02",
        units=_units("day-1"),
    )
    _qlib(
        tmp_path,
        "legacy",
        snapshot="source-v1",
        lineage_id=dataset_lineage,
        days=["2024-01-02"],
        verified=False,
    )

    with pytest.raises(ValueError, match="verified anchor lineage"):
        select_qlib_dataset(
            tmp_path,
            anchor_name="legacy",
            roll_policy="latest_compatible",
            lineage_id=dataset_lineage,
            required_date=date(2024, 1, 2),
        )


def test_selects_execution_descendant_and_resolves_exact_files(tmp_path: Path) -> None:
    lineage_id = "c" * 64
    _snapshot(
        tmp_path,
        "execution-v1",
        lineage_id=lineage_id,
        end="2024-01-02",
        units=_units("day-1"),
        execution=True,
    )
    _snapshot(
        tmp_path,
        "execution-v2",
        lineage_id=lineage_id,
        end="2024-01-03",
        units=_units("day-1", "day-2"),
        execution=True,
    )

    selected = select_execution_snapshot(
        tmp_path,
        anchor_name="execution-v1",
        roll_policy="latest_compatible",
        lineage_id=lineage_id,
        required_date=date(2024, 1, 3),
        minute_dataset="etf_1m",
        shortability_dataset="margin_eligibility",
    )

    assert selected["snapshot"]["name"] == "execution-v2"
    assert selected["minute"]["snapshot_name"] == "execution-v2"
    assert selected["shortability"]["snapshot_name"] == "execution-v2"
    assert all(Path(path).is_file() for path in selected["minute"]["files"])
