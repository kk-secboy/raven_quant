from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform.data_automation import (
    DEFAULT_DATA_AUTOMATION_CONFIG,
    MANAGED_SCHEDULE_NAMES,
    TASK_SCHEDULE_GROUPS,
    automation_coverage,
    normalize_data_automation_config,
)
from quant_platform.data_task_store import DATA_TASK_CATALOG
from quant_platform.scheduler import SchedulerEngine

pytestmark = pytest.mark.no_database


def test_all_33_catalog_tasks_have_exactly_one_refresh_owner() -> None:
    assigned = [task for tasks in TASK_SCHEDULE_GROUPS.values() for task in tasks]
    expected = [item.task_key for item in DATA_TASK_CATALOG]
    assert len(expected) == 33
    assert len(assigned) == len(set(assigned)) == 33
    assert set(assigned) == set(expected)


def test_default_data_automation_config_is_enabled_and_governed() -> None:
    config = normalize_data_automation_config(DEFAULT_DATA_AUTOMATION_CONFIG)
    assert config["enabled"] is True
    assert config["market_daily_time"] == "18:00"
    assert config["ashare_5m_time"] == "23:30"
    assert config["auxiliary_daily_time"] == "04:00"
    assert config["download_workers"] == 4
    assert config["requests_per_minute"] == 99
    assert len(config["strategy_minute_symbols"]) >= 50


def test_invalid_data_automation_config_fails_closed() -> None:
    candidate = deepcopy(DEFAULT_DATA_AUTOMATION_CONFIG)
    candidate["market_lookback_days"] = 0
    with pytest.raises(ValueError, match="between 1 and 90"):
        normalize_data_automation_config(candidate)
    candidate = deepcopy(DEFAULT_DATA_AUTOMATION_CONFIG)
    candidate["strategy_minute_symbols"] = []
    with pytest.raises(ValueError, match="1-200"):
        normalize_data_automation_config(candidate)


def test_coverage_requires_all_five_managed_schedules_to_be_active() -> None:
    schedules = [
        {
            "id": group,
            "name": name,
            "status": "active",
            "desired_status": "active",
        }
        for group, name in MANAGED_SCHEDULE_NAMES.items()
    ]
    ready = automation_coverage(schedules, enabled=True)
    assert ready["ready"] is True
    assert ready["covered"] == ready["total"] == 33

    schedules[-1]["status"] = "paused"
    blocked = automation_coverage(schedules, enabled=True)
    assert blocked["ready"] is False
    assert blocked["covered"] == 28
    assert {item["task_key"] for item in blocked["tasks"] if not item["covered"]} == {
        "cn_margin_eligibility",
        "pair_execution_1m",
        "liquid_intraday_1m",
        "liquid_intraday_qlib",
        "strategy_specialty_minutes",
    }


def test_auxiliary_schedule_builds_the_missing_five_task_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class FakeJobs:
        def create(self, kind, payload, log_path, *, idempotency_key):
            captured.update(
                kind=kind,
                payload=payload,
                log_path=str(log_path),
                idempotency_key=idempotency_key,
            )
            return {"id": "job", "kind": kind, "payload": payload}

    engine = object.__new__(SchedulerEngine)
    engine.settings = SimpleNamespace(
        data_root=tmp_path,
        api_url="https://api.tushare.pro",
        token="token",
    )
    engine.runtime_secrets = SimpleNamespace(get=lambda _name: None)
    engine.jobs = FakeJobs()
    monkeypatch.setattr(
        "quant_platform.scheduler.list_qlib_datasets",
        lambda _root: [
            {
                "name": "cn-current",
                "ready": True,
                "reproducible": True,
                "frequency": "day",
                "end_date": "2026-08-24",
                "provenance": {"source_lineage_id": "a" * 64},
            }
        ],
    )
    monkeypatch.setattr(
        "quant_platform.scheduler.require_daily_qlib_contract", lambda _value: None
    )
    result = engine._enqueue_auxiliary_data(
        {
            "id": "run",
            "payload": {
                "history_start": "2024-01-01",
                "max_stocks": 100,
                "max_options": 100,
                "strategy_minute_symbols": ["801010.SI", "00700.HK"],
                "download_workers": 3,
                "requests_per_minute": 80,
            },
        },
        datetime(2026, 8, 25, 4, tzinfo=UTC),
    )

    assert result["kind"] == "margin_eligibility_download"
    assert captured["idempotency_key"] == "auxiliary-data:2026-08-24"
    assert captured["payload"]["download_workers"] == 3
    assert captured["payload"]["requests_per_minute"] == 80
    assert [item["kind"] for item in captured["payload"]["pipeline_steps"]] == [
        "core_intraday_download",
        "minute_qlib",
        "supplemental_strategy_specialty_minutes",
    ]
