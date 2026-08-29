from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import insert

from quant_data.database import (
    open_database,
    research_program_events,
    research_programs,
)
from quant_platform.research_program_store import ResearchProgramStore


def _seed_legacy_program(database_url: str) -> str:
    program_id = "legacy-program-history"
    current = datetime(2020, 1, 1, tzinfo=UTC)
    with open_database(database_url).begin() as connection:
        connection.execute(
            insert(research_programs).values(
                id=program_id,
                name="retired program history",
                status="cancelled",
                recipe_id="legacy_recipe",
                objective="Preserve historical research policy evidence.",
                benchmark="SH000300",
                universe="cn_all",
                dataset_lineage_id="a" * 64,
                config_json={"legacy": True},
                min_new_trading_days=252,
                max_active_campaigns=1,
                next_check_at=current,
                created_by="legacy-migration",
                created_at=current,
                updated_at=current,
            )
        )
        connection.execute(
            insert(research_program_events).values(
                program_id=program_id,
                event_type="program.cancelled",
                actor="legacy-migration",
                payload_json={"reason": "single mainline cutover"},
                created_at=current,
            )
        )
    return program_id


def test_program_history_remains_queryable(database_url: str) -> None:
    program_id = _seed_legacy_program(database_url)
    store = ResearchProgramStore(database_url)

    program = store.get(program_id)

    assert program["status"] == "cancelled"
    assert program["config"] == {"legacy": True}
    assert program["events"][0]["event_type"] == "program.cancelled"
    assert [item["id"] for item in store.list()] == [program_id]
    assert store.irreversible_final_oos_windows(program_id) == []


def test_program_write_and_scheduler_paths_are_retired(database_url: str) -> None:
    store = ResearchProgramStore(database_url)
    message = "legacy research programs are read-only"

    with pytest.raises(RuntimeError, match=message):
        store.create(
            name="forbidden",
            recipe_id="legacy_recipe",
            objective="must not run",
            benchmark="SH000300",
            universe="cn_all",
            dataset_lineage_id="a" * 64,
            config={},
            min_new_trading_days=252,
            max_active_campaigns=1,
            actor="test",
        )
    with pytest.raises(RuntimeError, match=message):
        store.claim_due()
    with pytest.raises(RuntimeError, match=message):
        store.checked("missing", message="forbidden", delay_seconds=1)
    with pytest.raises(RuntimeError, match=message):
        store.set_status("missing", "cancelled", actor="test")
