from pathlib import Path

from quant_data.checkpoint import CheckpointStore
from quant_data.models import FetchSpec, UnitResult


def spec() -> FetchSpec:
    return FetchSpec(
        dataset="daily",
        api_name="daily",
        scope={"trade_date": "20240102"},
        params={"trade_date": "20240102"},
        fields=("ts_code", "trade_date", "close"),
    )


def test_checkpoint_is_idempotent_and_resumable(tmp_path: Path, database_url: str) -> None:
    store = CheckpointStore(database_url)
    assert store.add([spec()]) == 1
    assert store.add([spec()]) == 0

    unit = store.claim()
    assert unit is not None
    assert unit.spec.params["trade_date"] == "20240102"
    store.fail(unit.unit_key, "temporary", retry_after_seconds=0)

    retry = store.claim()
    assert retry is not None
    store.succeed(
        retry.unit_key,
        UnitResult(output_path="units/daily/test.parquet", row_count=1, sha256="abc"),
    )
    counts = store.counts()
    assert counts == [{"dataset": "daily", "status": "succeeded", "units": 1, "rows": 1}]


def test_terminal_failure_exhausts_unit(tmp_path: Path, database_url: str) -> None:
    store = CheckpointStore(database_url)
    store.add([spec()])
    unit = store.claim()
    assert unit is not None
    store.fail(unit.unit_key, "permission denied", terminal=True)
    assert store.claim() is None


def test_checkpoint_batches_large_plans(database_url: str) -> None:
    store = CheckpointStore(database_url)
    specs = [
        FetchSpec(
            dataset="income",
            api_name="income",
            scope={"ts_code": f"{index:06d}.SZ"},
            params={"ts_code": f"{index:06d}.SZ", "start_date": "20240101"},
        )
        for index in range(6_000)
    ]

    assert store.add(specs) == 6_000
    assert store.add(specs) == 0
