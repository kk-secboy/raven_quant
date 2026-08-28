from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.legacy_research_retirement import (
    _classify_jobs,
    _is_protected_job_kind,
    _job_ids,
)

pytestmark = pytest.mark.no_database


def test_retirement_finds_only_explicit_legacy_job_links() -> None:
    assert _job_ids(
        {
            "research_job_id": "research-1",
            "nested": [{"evaluation_job_id": "evaluation-1"}],
            "job_ids": ["not-an-explicit-owner"],
            "unrelated": "ignored",
        }
    ) == {"research-1", "evaluation-1"}


@pytest.mark.parametrize(
    "kind",
    [
        "ashare_5m_download",
        "cninfo_announcements_download",
        "supplemental_cn_macro",
        "data_snapshot",
        "data_qlib",
        "minute_qlib",
        "qlib_baseline",
        "factor_library_materialize",
        "factor_library_cluster",
        "research_asset_acquire",
    ],
)
def test_download_and_materialization_jobs_are_always_protected(kind: str) -> None:
    assert _is_protected_job_kind(kind) is True


def test_retirement_blocks_shared_autopilot_and_only_cancels_old_research() -> None:
    active = {
        "old-research": {
            "id": "old-research",
            "kind": "model_evaluate",
            "status": "running",
        },
        "shared": {"id": "shared", "kind": "factor_evaluate", "status": "queued"},
        "download": {
            "id": "download",
            "kind": "ashare_5m_download",
            "status": "running",
        },
        "cluster": {
            "id": "cluster",
            "kind": "factor_library_cluster",
            "status": "running",
        },
    }

    exclusive, shared, protected = _classify_jobs(
        active,
        autopilot_job_ids={"shared"},
    )

    assert exclusive == ["old-research"]
    assert shared == ["shared"]
    assert protected == ["cluster", "download"]


def test_compose_runs_one_time_retirement_after_migrations_before_api() -> None:
    compose = (Path(__file__).parents[1] / "deploy" / "compose.yaml").read_text(
        encoding="utf-8"
    )
    command = (
        "quant-db upgrade && python scripts/retire_legacy_research.py --apply "
        "&& quant-web --host 0.0.0.0 --port 8765"
    )
    assert command in compose
