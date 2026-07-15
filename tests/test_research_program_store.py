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


def test_program_records_final_test_without_using_it_for_selection(database_url: str) -> None:
    store = ResearchProgramStore(database_url)
    program = _create(store)

    recorded = store.record_campaign_outcome(
        program["id"],
        campaign={
            "id": "campaign-a",
            "status": "succeeded",
            "state": {"preferred_version_id": "version-a"},
        },
    )
    assert recorded["last_evaluated_campaign_id"] == "campaign-a"
    assert recorded["champion_campaign_id"] is None
    assert recorded["champion_strategy_version_id"] is None
    assert recorded["champion_score"] is None
    assert recorded["decay_status"] == "legacy"
    event = recorded["events"][0]
    assert event["event_type"] == "program.final_test_recorded"
    assert event["payload"]["used_for_selection"] is False
