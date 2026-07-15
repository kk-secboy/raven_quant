from __future__ import annotations

from quant_platform.research_campaign_store import ResearchCampaignStore


def _config() -> dict:
    return {
        "research": {"loop_n": 2, "duration": "1h", "periods": {}},
        "strategy_config": {"topk": 50},
        "backtest_periods": {"start": "2024-01-01", "end": "2025-12-31"},
        "experiment_periods": {},
        "parameter_grid": {"topk": [30, 50]},
        "experiment_trials": [],
        "max_factors": 5,
        "paper": {
            "initial_cash": 5_000_000,
            "timezone": "Asia/Shanghai",
            "run_time": "15:30",
            "slippage": 0.0005,
            "misfire_grace_seconds": 1800,
        },
    }


def test_campaign_store_claims_transitions_and_preserves_audit(
    database_url: str,
) -> None:
    store = ResearchCampaignStore(database_url)
    campaign = store.create(
        name="CSI 300 autonomous research",
        objective="Research stable low-turnover factors for CSI 300 enhancement.",
        dataset="cn-research",
        benchmark="SH000300",
        universe="cn_all",
        recipe_id="index_enhancement",
        config=_config(),
        actor="researcher",
    )
    claimed = store.claim_due()
    assert claimed and claimed["id"] == campaign["id"]
    assert claimed["status"] == "running"
    transitioned = store.transition(
        campaign["id"],
        stage="factor_selection",
        event_type="research.succeeded",
        state_patch={"selected": ["factor-a"]},
        links={"research_run_id": None},
    )
    assert transitioned["stage"] == "factor_selection"
    assert transitioned["state"]["selected"] == ["factor-a"]
    assert [item["event_type"] for item in transitioned["events"]][:2] == [
        "research.succeeded",
        "campaign.created",
    ]


def test_campaign_pause_and_resume_do_not_lose_stage(database_url: str) -> None:
    store = ResearchCampaignStore(database_url)
    campaign = store.create(
        name="Swing autonomous research",
        objective="Research robust A-share swing factors with strict capacity controls.",
        dataset="cn-research",
        benchmark="SH000300",
        universe="cn_all",
        recipe_id="swing_trend",
        config=_config(),
        actor="researcher",
    )
    paused = store.set_status(campaign["id"], "paused", actor="operator")
    assert paused["status"] == "paused"
    assert store.claim_due() is None
    resumed = store.set_status(campaign["id"], "running", actor="operator")
    assert resumed["stage"] == "research"
    assert store.claim_due()["id"] == campaign["id"]


def test_failed_campaign_can_be_retried_at_same_stage(database_url: str) -> None:
    store = ResearchCampaignStore(database_url)
    campaign = store.create(
        name="Retryable autonomous research",
        objective="Research a retryable factor campaign with complete audit evidence.",
        dataset="cn-research",
        benchmark="SH000300",
        universe="cn_all",
        recipe_id="index_enhancement",
        config=_config(),
        actor="researcher",
    )
    failed = store.fail(campaign["id"], "provider timeout")
    assert failed["status"] == "failed"
    retried = store.retry(campaign["id"], actor="operator")
    assert retried["status"] == "running"
    assert retried["stage"] == "research"
    assert retried["error"] is None
    assert retried["events"][0]["event_type"] == "campaign.retried"
