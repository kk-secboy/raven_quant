from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from quant_data.config import Settings
from quant_platform.continuous_research import ContinuousResearchController


def _dataset(root: Path) -> dict:
    path = root / "qlib" / "cn-research"
    calendar = path / "calendars" / "day.txt"
    calendar.parent.mkdir(parents=True)
    start = date(2020, 1, 1)
    days = [(start + timedelta(days=offset)).isoformat() for offset in range(1600)]
    calendar.write_text("\n".join(days), encoding="utf-8")
    return {
        "name": "cn-research",
        "path": str(path),
        "ready": True,
        "reproducible": True,
        "lineage_verified": True,
        "lineage_id": "lineage-a",
        "start_date": days[0],
        "end_date": days[-1],
        "provenance": {
            "dataset_identity_sha256": "a" * 64,
            "snapshot_manifest_sha256": "b" * 64,
        },
    }


def test_controller_creates_one_campaign_per_dataset_identity(
    database_url: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = _dataset(tmp_path)
    monkeypatch.setattr(
        "quant_platform.continuous_research.list_qlib_datasets",
        lambda _root: [dataset],
    )
    monkeypatch.setattr(
        "quant_platform.autonomous_research.list_qlib_datasets",
        lambda _root: [dataset],
    )
    settings = Settings(
        api_url="https://example.invalid/api/v1/query",
        token="test-token",
        data_root=tmp_path,
        database_url=database_url,
    )
    controller = ContinuousResearchController(settings)
    program = controller.programs.create(
        name="continuous index research",
        recipe_id="index_enhancement",
        objective="Research robust and low-turnover factors for index enhancement.",
        benchmark="SH000300",
        universe="cn_all",
        dataset_lineage_id="lineage-a",
        config={
            "window_days": {"train": 756, "validation": 252, "test": 504},
            "loop_n": 1,
            "duration": "30m",
            "max_factors": 3,
            "strategy_config": {"topk": 50, "n_drop": 5},
            "parameter_grid": {"topk": [30, 50]},
            "experiment_trials": [],
            "recommendation": {
                "hypothetical_initial_value": 5_000_000,
                "timezone": "Asia/Shanghai",
                "run_time": "15:30",
                "misfire_grace_seconds": 1800,
            },
        },
        min_new_trading_days=20,
        max_active_campaigns=1,
        actor="research-admin",
    )

    first = controller.tick(limit=1)
    assert first == {"checked": 1, "created": 1, "deferred": 0, "failed": 0}, (
        controller.programs.get(program["id"])["last_message"]
    )
    campaigns = controller.orchestrator.campaigns.list()
    assert len(campaigns) == 1
    assert campaigns[0]["research_program_id"] == program["id"]
    assert campaigns[0]["dataset_identity_sha256"] == "a" * 64
    assert campaigns[0]["config"]["research"]["periods"]["test_end"] == dataset["end_date"]

    controller.programs.check_now(program["id"], actor="research-admin")
    second = controller.tick(limit=1)
    assert second["created"] == 0
    assert len(controller.orchestrator.campaigns.list()) == 1
