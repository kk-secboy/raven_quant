from __future__ import annotations

from pathlib import Path

import pytest

from quant_data.config import Settings
from quant_platform.autopilot import AutopilotController
from quant_platform.rdagent_scenarios import (
    FROZEN_RDAGENT_SCENARIOS,
    SCENARIOS,
    rdagent_scenario_catalog,
)
from quant_platform.worker import (
    LocalJobWorker,
    _require_supported_rdagent_execution,
)

pytestmark = pytest.mark.no_database


def test_frozen_scenario_set_matches_the_weight_reduction_plan() -> None:
    assert FROZEN_RDAGENT_SCENARIOS == frozenset(
        {"fin_factor", "fin_model", "general_model", "data_science", "llm_finetune"}
    )
    # The registry itself is not pruned in phase 1: frozen scenarios stay
    # addressable so historical runs and assets remain readable.
    assert FROZEN_RDAGENT_SCENARIOS <= set(SCENARIOS)
    assert {"fin_quant", "fin_factor_report", "fin_strategy"}.isdisjoint(
        FROZEN_RDAGENT_SCENARIOS
    )


def test_scenario_catalog_marks_frozen_entries(tmp_path: Path) -> None:
    settings = Settings(api_url="", token="", data_root=tmp_path)
    catalog = {item["id"]: item for item in rdagent_scenario_catalog({}, settings)}
    assert {item["id"] for item in catalog.values() if item["frozen"]} == set(
        FROZEN_RDAGENT_SCENARIOS
    )


def test_worker_gate_rejects_every_frozen_scenario() -> None:
    for scenario_id in (
        "fin_factor",
        "fin_model",
        "general_model",
        "data_science",
        "llm_finetune",
    ):
        with pytest.raises(ValueError, match=f"scenario {scenario_id} is frozen"):
            _require_supported_rdagent_execution({"scenario": scenario_id})


def test_worker_gate_rejects_legacy_factor_jobs_without_a_scenario() -> None:
    # Historical rdagent_factor jobs predate the payload scenario field; they
    # resolve to the frozen fin_factor scenario and must fail closed.
    with pytest.raises(ValueError, match="scenario fin_factor is frozen"):
        _require_supported_rdagent_execution({})


def test_worker_gate_allows_retained_scenarios() -> None:
    # fin_strategy shares the generic rdagent_run kind with the frozen
    # general_model scenario, so the payload scenario decides.
    for scenario_id in ("fin_quant", "fin_factor_report", "fin_strategy"):
        _require_supported_rdagent_execution({"scenario": scenario_id})


def test_worker_command_enforces_the_freeze_before_building_a_command() -> None:
    worker = object.__new__(LocalJobWorker)
    with pytest.raises(ValueError, match="is frozen"):
        worker._command(
            {
                "id": "job-rdagent_run",
                "kind": "rdagent_run",
                "payload": {"scenario": "general_model"},
            }
        )


def test_frozen_worker_command_kinds_are_physically_removed() -> None:
    # Weight-reduction phase C3 deleted the frozen scenarios' worker command
    # kinds, so a historical queued job now fails closed as unsupported.
    worker = object.__new__(LocalJobWorker)
    for kind, scenario_id in (
        ("rdagent_factor", "fin_factor"),
        ("rdagent_model", "fin_model"),
        ("rdagent_data_science", "data_science"),
        ("rdagent_llm_finetune", "llm_finetune"),
    ):
        with pytest.raises(ValueError, match=f"unsupported job kind: {kind}"):
            worker._command(
                {
                    "id": f"job-{kind}",
                    "kind": kind,
                    "payload": {"scenario": scenario_id},
                }
            )


def test_autopilot_has_no_frozen_branch_creation_paths() -> None:
    # The RD-Agent fin_factor daily branch and the fin_model challenger lane
    # were the only automatic creation paths for the frozen quant scenarios.
    assert not hasattr(AutopilotController, "_factor_due")
    assert not hasattr(AutopilotController, "_enqueue_next_rdagent_model_challenger")


def test_autopilot_enqueue_fails_closed_for_frozen_scenarios() -> None:
    controller = AutopilotController.__new__(AutopilotController)
    for scenario_id in ("fin_factor", "fin_model"):
        with pytest.raises(ValueError, match=f"scenario {scenario_id} is frozen"):
            controller._enqueue({}, {}, scenario_id, "daily", config={})


def test_model_champions_do_not_wait_for_frozen_challenger_lanes() -> None:
    feature_set_id = "platform-seed-v1"
    platform_scope = f"platform:model_full:{feature_set_id}"
    cycle = {
        "id": "cycle-1",
        "state": {},
        "branches": [
            {
                "scenario": "fin_model",
                "scope_key": platform_scope,
                "status": "succeeded",
                "details": {
                    "branch_kind": "platform_model_model_full",
                    "tournament_id": "tournament-1",
                    "feature_set_id": feature_set_id,
                },
            }
        ],
    }
    tournament = {
        "id": "tournament-1",
        "dataset_identity_sha256": "a" * 64,
        "manifest": {},
        "trials": [],
    }
    blocked: list[str] = []

    class Store:
        @staticmethod
        def get_cycle(cycle_id):
            assert cycle_id == "cycle-1"
            return cycle

    class Tournaments:
        @staticmethod
        def block(tournament_id, *, reason):
            assert tournament_id == "tournament-1"
            blocked.append(reason)

    controller = AutopilotController.__new__(AutopilotController)
    controller.store = Store()
    controller.tournaments = Tournaments()
    controller._active_model_tournament = lambda _cycle: tournament

    # Before the freeze a missing monthly challenger scope made the gate wait
    # silently; now the platform full lane alone drives the comparison, so the
    # gate proceeds and fails closed on the absent evidence instead.
    assert controller._model_champions(cycle, tournament, [feature_set_id]) == []
    assert blocked == ["no full-round model produced immutable return evidence"]
