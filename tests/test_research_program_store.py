from __future__ import annotations

from quant_platform.research_program_store import ResearchProgramStore


def _create(store: ResearchProgramStore) -> dict:
    return store.create(
        name="monthly index research",
        recipe_id="index_enhancement",
        objective="Research robust and low-turnover factors for index enhancement.",
        benchmark="SH000300",
        universe="cn_all",
        dataset_lineage_id="lineage-a",
        config={"window_days": {"train": 756, "validation": 252, "test": 504}},
        min_new_trading_days=20,
        max_active_campaigns=1,
        actor="research-admin",
    )


def test_program_is_durable_claimable_and_audited(database_url: str) -> None:
    store = ResearchProgramStore(database_url)
    created = _create(store)
    claimed = store.claim_due()
    assert claimed and claimed["id"] == created["id"]

    dataset = {
        "name": "cn-20260714",
        "end_date": "2026-07-14",
        "provenance": {"dataset_identity_sha256": "a" * 64},
    }
    triggered = store.triggered(
        created["id"], campaign_id="campaign-a", dataset=dataset
    )
    assert triggered["last_dataset_name"] == "cn-20260714"
    assert triggered["last_dataset_identity_sha256"] == "a" * 64
    assert triggered["events"][0]["event_type"] == "program.triggered"


def test_program_pause_resume_and_cancel_are_fail_closed(database_url: str) -> None:
    store = ResearchProgramStore(database_url)
    program = _create(store)
    assert store.set_status(program["id"], "paused", actor="admin")["status"] == "paused"
    assert store.claim_due() is None
    assert store.set_status(program["id"], "active", actor="admin")["status"] == "active"
    assert store.set_status(program["id"], "cancelled", actor="admin")["status"] == "cancelled"


def test_program_controller_failure_releases_lease_and_is_audited(
    database_url: str,
) -> None:
    store = ResearchProgramStore(database_url)
    program = _create(store)
    assert store.claim_due() is not None

    failed = store.failed_check(
        program["id"], error="Qlib calendar is missing", delay_seconds=300
    )

    assert failed["lease_until"] is None
    assert failed["last_message"] == "自动检查失败：Qlib calendar is missing"
    assert failed["events"][0]["event_type"] == "program.check_failed"
    assert failed["events"][0]["payload"]["error"] == "Qlib calendar is missing"


def test_program_tracks_cross_campaign_champion_and_decay(database_url: str) -> None:
    store = ResearchProgramStore(database_url)
    program = _create(store)

    first = store.record_campaign_outcome(
        program["id"],
        campaign={
            "id": "campaign-a",
            "status": "succeeded",
            "state": {
                "preferred_version_id": "version-a",
                "champion": {"decision": "baseline", "baseline_score": 1.0},
            },
        },
    )
    assert first["champion_campaign_id"] == "campaign-a"
    assert first["champion_strategy_version_id"] == "version-a"
    assert first["champion_score"] == 1.0
    assert first["decay_status"] == "healthy"

    retained = store.record_campaign_outcome(
        program["id"],
        campaign={
            "id": "campaign-b",
            "status": "succeeded",
            "state": {
                "preferred_version_id": "version-b",
                "champion": {"decision": "challenger", "challenger_score": 1.02},
            },
        },
    )
    assert retained["champion_campaign_id"] == "campaign-a"
    assert retained["decay_status"] == "healthy"
    assert retained["events"][0]["event_type"] == "program.champion_retained"

    decayed = store.record_campaign_outcome(
        program["id"],
        campaign={
            "id": "campaign-c",
            "status": "succeeded",
            "state": {
                "preferred_version_id": "version-c",
                "champion": {"decision": "baseline", "baseline_score": 0.70},
            },
        },
    )
    assert decayed["champion_campaign_id"] == "campaign-a"
    assert decayed["decay_status"] == "warning"
    assert decayed["events"][0]["event_type"] == "program.decay_detected"

    promoted = store.record_campaign_outcome(
        program["id"],
        campaign={
            "id": "campaign-d",
            "status": "succeeded",
            "state": {
                "preferred_version_id": "version-d",
                "champion": {"decision": "challenger", "challenger_score": 1.20},
            },
        },
    )
    assert promoted["champion_campaign_id"] == "campaign-d"
    assert promoted["champion_strategy_version_id"] == "version-d"
    assert promoted["champion_score"] == 1.20
    assert promoted["events"][0]["event_type"] == "program.champion_selected"
