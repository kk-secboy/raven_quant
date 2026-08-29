from __future__ import annotations

import pytest

from quant_platform.three_horizon_account import (
    _select_current_three_horizon_dataset,
)

pytestmark = pytest.mark.no_database


def test_three_horizon_account_uses_latest_ready_snapshot_in_shared_lineage() -> None:
    lineage = "a" * 64

    selected = _select_current_three_horizon_dataset(
        formal_datasets={
            "short_1_5d": "qlib-2026-01-02",
            "swing_1_6m": "qlib-2026-03-02",
            "long_1_3y": "qlib-2026-06-02",
        },
        formal_lineages={
            "short_1_5d": lineage,
            "swing_1_6m": lineage,
            "long_1_3y": lineage,
        },
        qlib_datasets=[
            {
                "name": "qlib-2026-08-28",
                "lineage_id": lineage,
                "ready": True,
                "reproducible": True,
            },
            {
                "name": "qlib-2026-06-02",
                "lineage_id": lineage,
                "ready": True,
                "reproducible": True,
            },
        ],
    )

    assert selected == "qlib-2026-08-28"


def test_three_horizon_account_rejects_mixed_formal_lineages() -> None:
    with pytest.raises(ValueError, match="do not share one governed lineage"):
        _select_current_three_horizon_dataset(
            formal_datasets={"short": "one", "swing": "two", "long": "three"},
            formal_lineages={
                "short": "a" * 64,
                "swing": "a" * 64,
                "long": "b" * 64,
            },
            qlib_datasets=[],
        )


def test_three_horizon_account_rejects_partial_lineage_evidence() -> None:
    with pytest.raises(ValueError, match="lineage evidence is incomplete"):
        _select_current_three_horizon_dataset(
            formal_datasets={"short": "same", "swing": "same", "long": "same"},
            formal_lineages={"short": "a" * 64, "swing": "a" * 64},
            qlib_datasets=[],
        )


def test_legacy_three_horizon_evidence_must_name_the_same_dataset() -> None:
    assert (
        _select_current_three_horizon_dataset(
            formal_datasets={"short": "same", "swing": "same", "long": "same"},
            formal_lineages={},
            qlib_datasets=[],
        )
        == "same"
    )
    with pytest.raises(ValueError, match="legacy evidence requires one identical"):
        _select_current_three_horizon_dataset(
            formal_datasets={"short": "one", "swing": "two", "long": "three"},
            formal_lineages={},
            qlib_datasets=[],
        )
