from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from governance_fixtures import create_strategy_version

import quant_platform.strategy_health_authority as authority_module
from quant_data.database import open_database
from quant_platform.strategy_health_authority import (
    COLLECTOR_ACTOR,
    load_production_health_gate,
)
from quant_platform.strategy_store import StrategyStore


def test_manual_health_stop_survives_collector_and_system_observations(
    database_url: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    version_id = create_strategy_version(
        database_url,
        tmp_path,
        recipe_id="short_relative_strength",
    )
    store = StrategyStore(database_url)
    observed_at = datetime(2026, 8, 30, 8, 0, tzinfo=UTC)

    def record(offset: int, status: str, actor: str) -> None:
        store.record_health_snapshot(
            version_id,
            as_of=observed_at + timedelta(minutes=offset),
            health_status=status,
            criteria={"fixture": "manual-latch"},
            evidence={"fixture": actor},
            actor=actor,
        )

    record(0, "healthy", COLLECTOR_ACTOR)
    record(1, "restricted", "risk-operator")
    # Collector recovery and unrelated system observations must not clear the
    # operator's independent production latch.
    # The collector's own previous state is healthy, so a healthy observation
    # must persist directly despite the independent manual restriction.
    record(2, "healthy", COLLECTOR_ACTOR)
    record(3, "healthy", "system:auto-promotion")

    monkeypatch.setattr(
        authority_module,
        "current_paper_health_binding",
        lambda _connection, _version_id: {"horizon_profile": "short_1_5d"},
    )
    monkeypatch.setattr(
        authority_module,
        "validate_production_health_snapshot",
        lambda snapshot, **_kwargs: {
            "ready": True,
            "allow_new_risk": True,
            "health_status": str(snapshot.health_status),
            "snapshot_id": str(snapshot.id),
            "reasons": [],
        },
    )
    engine = open_database(database_url)
    with engine.connect() as connection:
        blocked = load_production_health_gate(connection, version_id)

    assert blocked["allow_new_risk"] is False
    assert blocked["health_status"] == "restricted"
    assert blocked["reasons"] == [
        "manual_strategy_health_restricted_blocks_new_risk"
    ]

    record(4, "healthy", "risk-operator")
    with engine.connect() as connection:
        waiting = load_production_health_gate(connection, version_id)
    assert waiting["allow_new_risk"] is False
    assert waiting["reasons"] == [
        "strategy_health_collector_precedes_manual_release"
    ]

    record(5, "healthy", COLLECTOR_ACTOR)
    with engine.connect() as connection:
        released = load_production_health_gate(connection, version_id)
    assert released["ready"] is True
    assert released["allow_new_risk"] is True
