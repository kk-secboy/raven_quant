import time
from pathlib import Path

import pytest

from quant_data.config import Settings
from quant_platform import services
from quant_platform.services import CheckpointCatalogProjection, system_summary

pytestmark = pytest.mark.no_database


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _Checkpoint:
    def __init__(self) -> None:
        self.calls = 0
        self.rows = 100

    def counts(self) -> list[dict]:
        self.calls += 1
        return [
            {
                "dataset": "daily",
                "status": "succeeded",
                "units": 1,
                "rows": self.rows,
            }
        ]


def test_checkpoint_catalog_projection_reuses_counts_for_thirty_seconds() -> None:
    clock = _Clock()
    checkpoint = _Checkpoint()
    projection = CheckpointCatalogProjection(
        checkpoint,  # type: ignore[arg-type]
        ttl_seconds=30.0,
        clock=clock,
    )

    first = projection.refresh()
    checkpoint.rows = 200
    clock.value = 29.999
    second = projection.get()

    assert checkpoint.calls == 1
    assert second is first
    assert next(item for item in second if item["name"] == "daily")["rows"] == 100

    clock.value = 30.0
    stale = projection.get()
    assert stale is first
    deadline = time.monotonic() + 2
    while checkpoint.calls < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    refreshed = projection.get()

    assert checkpoint.calls == 2
    assert next(item for item in refreshed if item["name"] == "daily")["rows"] == 200


def test_checkpoint_catalog_projection_cold_read_is_non_blocking(tmp_path: Path) -> None:
    checkpoint = _Checkpoint()
    projection = CheckpointCatalogProjection(
        checkpoint,  # type: ignore[arg-type]
        cache_path=tmp_path / "checkpoint-catalog.json",
    )

    started = time.monotonic()
    assert projection.get() == []
    assert time.monotonic() - started < 0.5
    deadline = time.monotonic() + 2
    refreshed: list[dict] = []
    while time.monotonic() < deadline:
        refreshed = projection.get()
        if refreshed:
            break
        time.sleep(0.01)
    assert checkpoint.calls == 1
    assert refreshed


def test_checkpoint_catalog_projection_requires_positive_ttl() -> None:
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        CheckpointCatalogProjection(  # type: ignore[arg-type]
            _Checkpoint(), ttl_seconds=0
        )


def test_system_summary_accepts_precomputed_catalog_without_querying_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoQueryCheckpoint:
        def counts(self) -> list[dict]:
            raise AssertionError("precomputed display catalog must be reused")

    catalog = [
        {
            "name": "daily",
            "profile": "core",
            "planned": 2,
            "succeeded": 2,
            "failed": 0,
            "running": 0,
            "rows": 123,
            "coverage": 100.0,
            "state": "ready",
        }
    ]
    monkeypatch.setattr(services, "list_snapshots_for_display", lambda _root: [])
    monkeypatch.setattr(services, "list_qlib_datasets_for_display", lambda _root: [])
    settings = Settings(api_url="", token="", data_root=Path("."))

    summary = system_summary(
        settings,
        _NoQueryCheckpoint(),  # type: ignore[arg-type]
        [],
        [],
        catalog=catalog,
    )

    assert summary["rows"] == 123
    assert summary["planned_units"] == 2
    assert summary["succeeded_units"] == 2
