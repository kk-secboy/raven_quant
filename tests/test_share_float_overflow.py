from types import SimpleNamespace

import pandas as pd
import pytest

from quant_data.checkpoint import CheckpointStore
from quant_data.cli import _pagination_overflow_recovery, _share_float_overflow_recovery
from quant_data.models import FetchSpec
from quant_data.supplemental_data import share_float_overflow_repartition_specs


def test_overflow_recovery_preserves_prefix_and_supersedes_bad_cursor(
    database_url: str,
) -> None:
    checkpoint = CheckpointStore(database_url)
    common_scope = {
        "start_date": "20240101",
        "end_date": "20240131",
        "page_group": "share_float:20240101:20240131",
        "page_size": 1_000,
        "max_pages": 512,
    }
    prefix = FetchSpec(
        dataset="share_float",
        api_name="share_float",
        params={
            "start_date": "20240101",
            "end_date": "20240131",
            "limit": 1_000,
            "offset": 100_000,
        },
        scope={**common_scope, "offset": 100_000},
        allow_empty=True,
        max_attempts=5,
    )
    failed = FetchSpec(
        dataset="share_float",
        api_name="share_float",
        params={
            "start_date": "20240101",
            "end_date": "20240131",
            "limit": 1_000,
            "offset": 101_000,
        },
        scope={**common_scope, "offset": 101_000},
        allow_empty=True,
        max_attempts=5,
    )
    checkpoint.add([prefix, failed])
    checkpoint.fail(
        failed.unit_key,
        "cancelled deterministic share_float offset-cap request; retained for recovery",
        terminal=True,
    )

    context = SimpleNamespace(checkpoint=checkpoint)
    continuations, recovered = _share_float_overflow_recovery(
        context, [prefix, failed], set()
    )

    assert recovered == {failed.unit_key}
    assert len(continuations) == 31
    assert continuations[0].params["start_date"] == "20240101"
    assert continuations[-1].params["end_date"] == "20240131"
    assert all(item.params["offset"] == 0 for item in continuations)
    assert all(
        item.scope["supersedes_page_group"] == common_scope["page_group"]
        for item in continuations
    )
    failed_row = checkpoint.unit_rows([failed.unit_key])[0]
    assert failed_row["status"] == "superseded"
    assert "pagination offset cap" in failed_row["last_error"]


@pytest.mark.no_database
def test_restart_rehydrates_existing_daily_share_float_replacement() -> None:
    parent = FetchSpec(
        dataset="share_float",
        api_name="share_float",
        params={
            "start_date": "20240101",
            "end_date": "20240131",
            "limit": 1_000,
            "offset": 0,
        },
        scope={
            "start_date": "20240101",
            "end_date": "20240131",
            "page_group": "share_float:20240101:20240131",
            "page_size": 1_000,
            "max_pages": 512,
            "offset": 0,
        },
        allow_empty=True,
        max_attempts=5,
    )
    daily = share_float_overflow_repartition_specs(parent)
    durable = daily[::2]

    class Checkpoint:
        @staticmethod
        def unit_rows(unit_keys) -> list[dict]:
            keys = set(unit_keys)
            rows = []
            if parent.unit_key in keys:
                rows.append(
                    {
                        "unit_key": parent.unit_key,
                        "status": "superseded",
                        "last_error": "monthly pagination replaced by daily partitions",
                    }
                )
            return rows

        @staticmethod
        def dataset_units(dataset: str) -> list[dict]:
            assert dataset == "share_float"
            return [
                {
                    "unit_key": item.unit_key,
                    "status": "pending",
                    "dataset": item.dataset,
                    "api_name": item.api_name,
                    "scope_json": item.scope,
                    "params_json": item.params,
                    "fields_json": list(item.fields),
                    "allow_empty": item.allow_empty,
                    "max_attempts": item.max_attempts,
                }
                for item in durable
            ]

    context = SimpleNamespace(checkpoint=Checkpoint())
    replacements, recovered = _share_float_overflow_recovery(
        context, [parent], set()
    )

    assert recovered == {parent.unit_key}
    assert {item.unit_key for item in replacements} == {
        item.unit_key for item in durable
    }
    assert all(
        item.scope["supersedes_page_group"] == parent.scope["page_group"]
        for item in replacements
    )


@pytest.mark.no_database
def test_single_day_recovery_loads_historically_active_stock_symbols() -> None:
    failed = FetchSpec(
        dataset="share_float",
        api_name="share_float",
        params={
            "start_date": "20240219",
            "end_date": "20240219",
            "limit": 6_000,
            "offset": 102_000,
        },
        scope={
            "start_date": "20240219",
            "end_date": "20240219",
            "page_group": "share_float:20240201:20240229:daily:20240219",
            "page_size": 6_000,
            "max_pages": 512,
            "offset": 102_000,
        },
        allow_empty=True,
        max_attempts=5,
    )
    superseded: list[str] = []

    class Checkpoint:
        @staticmethod
        def unit_rows(unit_keys) -> list[dict]:
            return [
                {
                    "unit_key": failed.unit_key,
                    "status": "failed",
                    "last_error": "provider error code=50101: offset cap",
                }
            ] if failed.unit_key in set(unit_keys) else []

        @staticmethod
        def dataset_units(dataset: str) -> list[dict]:
            assert dataset == "share_float"
            return []

        @staticmethod
        def successful(dataset: str) -> list[dict]:
            assert dataset == "stock_basic"
            return [{"unit_key": "stock-master"}]

        @staticmethod
        def supersede_units(unit_keys, _reason: str) -> None:
            superseded.extend(unit_keys)

    master = pd.DataFrame.from_records(
        [
            {
                "ts_code": "600000.SH",
                "list_date": "19991110",
                "delist_date": None,
            },
            {
                "ts_code": "000001.SZ",
                "list_date": "19910403",
                "delist_date": None,
            },
            {
                "ts_code": "430047.BJ",
                "list_date": "20101001",
                "delist_date": "20231231",
            },
            {
                "ts_code": "600001.SH",
                "list_date": "20250101",
                "delist_date": None,
            },
        ]
    )
    storage = SimpleNamespace(read_units=lambda _rows: master)
    context = SimpleNamespace(checkpoint=Checkpoint(), storage=storage)

    replacements, recovered = _share_float_overflow_recovery(
        context,
        [failed],
        set(),
    )

    assert recovered == {failed.unit_key}
    assert superseded == [failed.unit_key]
    assert [item.params["ts_code"] for item in replacements] == [
        "000001.SZ",
        "600000.SH",
    ]
    assert all(
        item.scope["supersedes_page_group"] == failed.scope["page_group"]
        for item in replacements
    )


def test_etf_overflow_recovery_uses_listed_exchange_symbols(
    database_url: str,
) -> None:
    checkpoint = CheckpointStore(database_url)
    failed = FetchSpec(
        dataset="etf_sh_cons",
        api_name="etf_sh_cons",
        params={"trade_date": "20260316", "limit": 3_000, "offset": 102_000},
        scope={
            "trade_date": "20260316",
            "page_group": "etf_sh_cons:20260316",
            "page_size": 3_000,
            "offset": 102_000,
        },
        allow_empty=True,
        max_attempts=5,
    )
    checkpoint.add([failed])
    checkpoint.fail(
        failed.unit_key,
        "provider error code=50101 http=200: invalid offset",
        terminal=True,
    )
    master = pd.DataFrame.from_records(
        [
            {"ts_code": "510050.SH", "list_status": "L", "list_date": "20050223"},
            {"ts_code": "512000.SH", "list_status": "D", "list_date": "20160829"},
            {"ts_code": "513999.SH", "list_status": "P", "list_date": None},
            {"ts_code": "159001.SZ", "list_status": "L", "list_date": "20041220"},
        ]
    )
    storage = SimpleNamespace(read_units=lambda _rows: master)
    context = SimpleNamespace(checkpoint=checkpoint, storage=storage)

    continuations, recovered = _pagination_overflow_recovery(
        context, [failed], set()
    )

    assert recovered == {failed.unit_key}
    assert [item.params["ts_code"] for item in continuations] == [
        "510050.SH",
        "512000.SH",
    ]
    assert all(item.params["trade_date"] == "20260316" for item in continuations)
    failed_row = checkpoint.unit_rows([failed.unit_key])[0]
    assert failed_row["status"] == "superseded"
    assert "smaller ETF partitions" in failed_row["last_error"]
