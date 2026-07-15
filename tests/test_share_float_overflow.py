from types import SimpleNamespace

from quant_data.checkpoint import CheckpointStore
from quant_data.cli import _share_float_overflow_recovery
from quant_data.models import FetchSpec


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
