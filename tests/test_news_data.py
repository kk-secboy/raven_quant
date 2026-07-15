from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant_data.execution_data import NEWS_FIELDS, NEWS_SOURCES, news_specs, validate_and_normalize
from quant_data.models import ProviderResult
from quant_data.provider import ProviderError
from quant_data.storage import ParquetStore

pytestmark = pytest.mark.no_database


def _news_result(spec, rows):
    return ProviderResult(
        "news",
        list(NEWS_FIELDS),
        rows,
        b"{}",
    )


def test_news_planner_uses_source_aware_half_day_windows() -> None:
    specs = news_specs(date(2024, 1, 2), date(2024, 1, 2), max_attempts=5)

    assert len(specs) == len(NEWS_SOURCES) * 2
    assert {spec.params["src"] for spec in specs} == set(NEWS_SOURCES)
    assert {spec.scope["row_limit"] for spec in specs} == {1_500}
    assert {spec.fields for spec in specs} == {NEWS_FIELDS}
    assert {
        (spec.params["start_date"][-8:], spec.params["end_date"][-8:])
        for spec in specs
        if spec.params["src"] == "sina"
    } == {
        ("00:00:00", "11:59:59"),
        ("12:00:00", "23:59:59"),
    }


def test_news_validator_persists_source_and_deduplicates_response_rows() -> None:
    spec = news_specs(
        date(2024, 1, 2),
        date(2024, 1, 2),
        max_attempts=5,
        sources=["sina"],
    )[0]
    row = {
        "datetime": "2024-01-02 10:30:00",
        "content": "market update",
        "title": "headline",
        "channels": "finance",
    }

    normalized = validate_and_normalize(spec, _news_result(spec, [row, row]))

    assert normalized.columns == [*NEWS_FIELDS, "source"]
    assert normalized.rows == [{**row, "source": "sina"}]
    assert normalized.metadata["source"] == "sina"


def test_news_validator_rejects_a_capped_window() -> None:
    spec = news_specs(
        date(2024, 1, 2),
        date(2024, 1, 2),
        max_attempts=5,
        sources=["cls"],
    )[0]
    row = {
        "datetime": "2024-01-02 10:30:00",
        "content": "market update",
        "title": "headline",
        "channels": "finance",
    }

    with pytest.raises(ProviderError, match="1500-row limit"):
        validate_and_normalize(spec, _news_result(spec, [row] * 1_500))


def test_news_snapshot_replaces_legacy_duplicates_without_deleting_units(
    tmp_path: Path,
) -> None:
    storage = ParquetStore(tmp_path)
    duplicate = {
        "datetime": "2024-01-02 10:30:00",
        "content": "market update",
        "title": "headline",
        "channels": "finance",
    }
    legacy_only = {
        "datetime": "2024-01-02 11:00:00",
        "content": "legacy only",
        "title": "older headline",
        "channels": "finance",
    }
    results = [
        storage.write_unit(
            "news",
            "legacy",
            ProviderResult("news", list(NEWS_FIELDS), [duplicate, legacy_only], b"{}"),
        ),
        storage.write_unit(
            "news",
            "sina",
            ProviderResult(
                "news",
                [*NEWS_FIELDS, "source"],
                [{**duplicate, "source": "sina"}],
                b"{}",
            ),
        ),
        storage.write_unit(
            "news",
            "cls",
            ProviderResult(
                "news",
                [*NEWS_FIELDS, "source"],
                [{**duplicate, "source": "cls"}],
                b"{}",
            ),
        ),
    ]
    units = [
        {
            "unit_key": key,
            "output_path": result.output_path,
            "row_count": result.row_count,
            "sha256": result.sha256,
        }
        for key, result in zip(("legacy", "sina", "cls"), results, strict=True)
    ]

    snapshot = storage.build_snapshot(
        name="news-repair",
        successful_units={"news": units},
        manifest_extra={"profile": "test"},
    )
    frame = pd.concat(
        [pd.read_parquet(path) for path in (snapshot / "parquet" / "news").rglob("*.parquet")],
        ignore_index=True,
    )

    assert len(frame) == 3
    tagged = frame[frame["content"] == "market update"]
    assert set(tagged["source"]) == {"sina", "cls"}
    legacy = frame[frame["content"] == "legacy only"]
    assert len(legacy) == 1
    assert pd.isna(legacy.iloc[0]["source"])
