from pathlib import Path

import quant_data.checkpoint as checkpoint_module
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


def test_successful_units_reads_only_the_requested_plan(database_url: str) -> None:
    store = CheckpointStore(database_url)
    first = spec()
    second = FetchSpec(
        dataset="daily",
        api_name="daily",
        scope={"trade_date": "20240103"},
        params={"trade_date": "20240103"},
    )
    unrelated = FetchSpec(
        dataset="daily",
        api_name="daily",
        scope={"trade_date": "20240104"},
        params={"trade_date": "20240104"},
    )
    store.add([first, second, unrelated])
    for item in (first, unrelated):
        store.succeed(
            item.unit_key,
            UnitResult(output_path=f"units/{item.unit_key}.parquet", row_count=1, sha256="abc"),
        )

    rows = store.successful_units([first.unit_key, second.unit_key])
    assert [row["unit_key"] for row in rows] == [first.unit_key]


def test_superseded_unit_keys_are_scoped_to_the_requested_plan(database_url: str) -> None:
    store = CheckpointStore(database_url)
    current = spec()
    obsolete = FetchSpec(
        dataset="daily",
        api_name="daily",
        scope={"trade_date": "20240103"},
        params={"trade_date": "20240103"},
    )
    unrelated = FetchSpec(
        dataset="daily",
        api_name="daily",
        scope={"trade_date": "20240104"},
        params={"trade_date": "20240104"},
    )
    store.add([current, obsolete, unrelated])
    store.supersede_units([obsolete.unit_key, unrelated.unit_key], "obsolete plan")

    assert store.superseded_unit_keys([current.unit_key, obsolete.unit_key]) == {
        obsolete.unit_key
    }


def test_superseded_units_are_retained_but_not_runnable_or_planned(
    database_url: str,
) -> None:
    store = CheckpointStore(database_url)
    item = spec()
    store.add([item])
    unit = store.claim()
    assert unit is not None
    store.fail(unit.unit_key, "provider offset cap")

    assert store.supersede_units([unit.unit_key], "continued by a smaller partition") == 1
    assert store.claim() is None
    rows = store.unit_rows([unit.unit_key])
    assert rows[0]["status"] == "superseded"
    assert rows[0]["last_error"] == "continued by a smaller partition"
    verification = store.verification_rows()[0]
    assert verification["planned"] == 0
    assert verification["superseded"] == 1


def test_bulk_status_updates_are_chunked(
    database_url: str,
    monkeypatch,
) -> None:
    store = CheckpointStore(database_url)
    items = [
        FetchSpec(
            dataset="daily",
            api_name="daily",
            scope={"trade_date": f"2024010{index}"},
            params={"trade_date": f"2024010{index}"},
        )
        for index in range(1, 5)
    ]
    store.add(items)
    for _item in items:
        unit = store.claim()
        assert unit is not None
        store.fail(unit.unit_key, "provider failure", terminal=True)

    monkeypatch.setattr(checkpoint_module, "_SELECT_BATCH_SIZE", 2)
    assert store.retry_failed_units(item.unit_key for item in items) == 4

    rows = store.unit_rows(item.unit_key for item in items)
    assert {row["status"] for row in rows} == {"pending"}
    assert store.supersede_units(
        (item.unit_key for item in items), "repartitioned"
    ) == 4
    rows = store.unit_rows(item.unit_key for item in items)
    assert {row["status"] for row in rows} == {"superseded"}


def test_unfinished_units_returns_only_pending_and_failed(database_url: str) -> None:
    store = CheckpointStore(database_url)
    pending = spec()
    failed = FetchSpec("daily", "daily", {"trade_date": "20240103"}, {"trade_date": "20240103"})
    succeeded = FetchSpec(
        "daily", "daily", {"trade_date": "20240104"}, {"trade_date": "20240104"}
    )
    store.add([pending, failed, succeeded])
    store.fail(failed.unit_key, "temporary")
    store.succeed(
        succeeded.unit_key,
        UnitResult(output_path="units/succeeded.parquet", row_count=1, sha256="abc"),
    )

    assert {row["unit_key"] for row in store.unfinished_units("daily")} == {
        pending.unit_key,
        failed.unit_key,
    }
