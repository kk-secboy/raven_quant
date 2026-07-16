from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

from quant_data.checkpoint import CheckpointStore
from quant_data.cli import _full_page_partition_recovery, _pagination_overflow_recovery
from quant_data.execution_data import minute_specs
from quant_data.models import FetchSpec, UnitResult
from quant_data.partitioning import partition_metadata
from quant_data.provider import ProviderError
from quant_data.runner import DownloadRunner
from quant_data.storage import ParquetStore


def test_failed_minute_cap_is_superseded_by_disjoint_children(database_url: str) -> None:
    store = CheckpointStore(database_url)
    parent = minute_specs(
        {"ashare_5m": ["600000.SH"]},
        start=date(2024, 1, 2),
        end=date(2024, 1, 5),
        trading_dates=["20240102", "20240103", "20240104", "20240105"],
        max_attempts=3,
        freq="5min",
    )[0]
    store.add([parent])
    store.fail(parent.unit_key, "may be truncated at the 8000-row provider limit", terminal=True)

    children, recovered = _pagination_overflow_recovery(
        SimpleNamespace(checkpoint=store), [parent], set()
    )

    assert recovered == {parent.unit_key}
    assert len(children) == 2
    assert children[0].scope["partition_end"] == "2024-01-03"
    assert children[1].scope["partition_start"] == "2024-01-04"
    assert store.unit_rows([parent.unit_key])[0]["status"] == "superseded"


def test_full_last_page_splits_range_without_discarding_success(database_url: str) -> None:
    store = CheckpointStore(database_url)
    parent = FetchSpec(
        dataset="fund_share",
        api_name="fund_share",
        scope={
            "page_group": "fund_share:20240101:20240131",
            "page_size": 100,
            "max_pages": 1,
            "offset": 0,
            "expected_date_field": "trade_date",
            **partition_metadata("date", date(2024, 1, 1), date(2024, 1, 31)),
        },
        params={
            "start_date": "20240101",
            "end_date": "20240131",
            "limit": 100,
            "offset": 0,
        },
        allow_empty=True,
        max_attempts=3,
    )
    store.add([parent])
    store.succeed(
        parent.unit_key,
        UnitResult(output_path="units/fund-share.parquet", row_count=100, sha256="abc"),
    )

    children, recovered = _full_page_partition_recovery(
        SimpleNamespace(checkpoint=store),
        [parent],
        [{"unit_key": parent.unit_key, "row_count": 100}],
        set(),
    )

    assert recovered == {parent.unit_key}
    assert len(children) == 2
    assert children[0].params["end_date"] == "20240116"
    assert children[1].params["start_date"] == "20240117"
    assert children[0].scope["supersedes_page_group"] == parent.scope["page_group"]
    assert store.unit_rows([parent.unit_key])[0]["status"] == "succeeded"


def test_adaptive_offset_cap_splits_the_whole_date_range(database_url: str) -> None:
    store = CheckpointStore(database_url)
    failed = FetchSpec(
        dataset="fund_share",
        api_name="fund_share",
        scope={
            "page_group": "fund_share:20240101:20240131",
            "page_size": 2_000,
            "max_pages": 64,
            "offset": 100_000,
            **partition_metadata("date", date(2024, 1, 1), date(2024, 1, 31)),
        },
        params={
            "start_date": "20240101",
            "end_date": "20240131",
            "limit": 2_000,
            "offset": 100_000,
        },
        allow_empty=True,
        max_attempts=3,
    )
    store.add([failed])
    store.fail(failed.unit_key, "provider error code=50101: offset cap", terminal=True)

    children, recovered = _pagination_overflow_recovery(
        SimpleNamespace(checkpoint=store), [failed], set()
    )

    assert recovered == {failed.unit_key}
    assert [child.params["offset"] for child in children] == [0, 0]
    assert children[0].params["end_date"] == "20240116"
    assert children[1].params["start_date"] == "20240117"


def test_rate_limit_failure_is_handed_back_to_checkpoint_with_cooldown(
    database_url: str, tmp_path
) -> None:
    class RateLimitedProvider:
        def fetch(self, api_name, params, fields=()):
            raise ProviderError(
                "too many requests",
                retryable=True,
                rate_limited=True,
                retry_after_seconds=180,
            )

    store = CheckpointStore(database_url)
    spec = FetchSpec("daily", "daily", {"trade_date": "20240102"}, {"trade_date": "20240102"})
    store.add([spec])

    summary = DownloadRunner(
        checkpoint=store,
        storage=ParquetStore(tmp_path),
        provider=RateLimitedProvider(),
        workers=1,
    ).run({"daily"})
    row = store.unit_rows([spec.unit_key])[0]

    assert summary.failed == 1
    assert row["status"] == "failed"
    retry_at = datetime.fromisoformat(str(row["next_retry_at"]))
    updated_at = datetime.fromisoformat(str(row["updated_at"]))
    assert (retry_at - updated_at).total_seconds() >= 179
