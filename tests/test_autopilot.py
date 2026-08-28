from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_platform import research_tournament as tournament_module
from quant_platform.autopilot import (
    AUTOPILOT_CONTRACT_VERSION,
    AutopilotController,
    _asset_available_on,
    _cycle_has_capital_commitment,
    _cycle_terminal_resolution,
    _derived_branch_status,
    _profile_family_multiple_testing,
    normalize_autopilot_config,
)
from quant_platform.data_automation import (
    MANAGED_SCHEDULE_NAMES,
    TASK_SCHEDULE_GROUPS,
    normalize_data_automation_config,
)
from quant_platform.job_store import research_asset_acquisition_idempotency_key
from quant_platform.research_tournament import (
    FULL_PROFILES,
    FULL_SEEDS,
    MODEL_FAMILIES,
    build_preregistered_manifest,
    canonical_sha256,
)

pytestmark = pytest.mark.no_database


def test_autopilot_defaults_to_safe_automatic_paper_only() -> None:
    config = normalize_autopilot_config()
    assert config["contract_version"] == AUTOPILOT_CONTRACT_VERSION
    assert config["enabled"] is True
    assert config["paper_min_calendar_days"] == 183
    assert config["report_daily_limit"] == 20
    assert config["quant_cooldown_days"] == 7
    assert config["factor_research_interval_days"] == 1


def test_autopilot_never_allows_a_shorter_real_money_observation_window() -> None:
    with pytest.raises(ValueError, match="paper_min_calendar_days"):
        normalize_autopilot_config({"paper_min_calendar_days": 182})


def test_research_assets_are_an_isolated_sixth_data_schedule() -> None:
    config = normalize_data_automation_config()
    assert config["research_assets_time"] == "19:30"
    assert config["research_assets_history_start"] == "2023-08-25"
    assert set(MANAGED_SCHEDULE_NAMES) == set(TASK_SCHEDULE_GROUPS)
    assert TASK_SCHEDULE_GROUPS["research_assets"] == ("research_corpus",)
    assert "research_corpus" not in TASK_SCHEDULE_GROUPS["market_daily"]


def test_report_backfill_job_identity_binds_the_report_date() -> None:
    values = {
        "research_day": "2026-08-25",
        "snapshot_name": "research-assets-20230825-20260825",
        "include_tushare": True,
        "include_arxiv": False,
    }
    first = research_asset_acquisition_idempotency_key(
        **values, report_date="2026-08-22"
    )
    second = research_asset_acquisition_idempotency_key(
        **values, report_date="2026-08-21"
    )
    assert first != second


def test_report_asset_availability_uses_shanghai_calendar_day() -> None:
    assert _asset_available_on({"available_at": "2025-07-31T16:00:00Z"}).isoformat() == (
        "2025-08-01"
    )
    assert _asset_available_on({"available_at": "not-a-time"}) is None


def test_autopilot_distinguishes_completed_jobs_from_blocked_research() -> None:
    assert _derived_branch_status("blocked", "succeeded") == "blocked"
    assert _derived_branch_status("blocked", "failed") == "blocked"
    assert _derived_branch_status("succeeded", "succeeded") == "succeeded"
    assert _derived_branch_status("running", "failed") == "running"
    assert _derived_branch_status("unknown", "failed") == "failed"


def test_autopilot_cycles_do_not_remain_active_after_terminal_failures() -> None:
    assert (
        _cycle_terminal_resolution(
            [("fin_factor", "failed"), ("fin_model", "blocked")]
        )
        is None
    )
    assert _cycle_terminal_resolution(
        [("fin_factor", "succeeded"), ("fin_model", "succeeded")]
    ) is None
    assert _cycle_terminal_resolution(
        [("fin_factor", "succeeded"), ("fin_quant", "failed")]
    ) == ("blocked", "joint_optimization_blocked")
    assert (
        _cycle_terminal_resolution(
            [("fin_factor", "succeeded"), ("fin_quant", "succeeded")]
        )
        is None
    )


def test_only_joint_winners_are_protected_from_daily_research_supersession() -> None:
    assert not _cycle_has_capital_commitment(
        {
            "state": {},
            "branches": [
                {"scenario": "fin_factor", "status": "running"},
                {"scenario": "fin_quant", "status": "running"},
            ],
        }
    )
    assert _cycle_has_capital_commitment(
        {
            "state": {},
            "branches": [{"scenario": "fin_quant", "status": "succeeded"}],
        }
    )
    assert _cycle_has_capital_commitment(
        {
            "state": {"capital_pipeline": {"phase": "formal_backtest"}},
            "branches": [],
        }
    )


def test_model_tournament_first_compares_feature_sets_with_one_fixed_model() -> None:
    manifest = build_preregistered_manifest()
    screens = [
        item for item in manifest["trials"] if item["trial_kind"] == "feature_set"
    ]
    assert {item["feature_set_id"] for item in screens} == {
        "qlib-alpha158",
        "qlib-alpha360",
        "platform-seed-v1",
    }
    assert {item["model_family"] for item in screens} == {"lightgbm"}
    assert all(item["spec"]["selection_seeds"] == [11] for item in screens)


def test_full_model_trials_are_conditionally_preregistered_before_screening() -> None:
    manifest = build_preregistered_manifest()
    full = [
        item
        for item in manifest["trials"]
        if item["trial_kind"] == "model"
        and item["spec"]["round"] == "model_full"
    ]
    assert len(full) == 3 * len(MODEL_FAMILIES)
    assert {item["model_family"] for item in full} == set(MODEL_FAMILIES)
    assert all(item["spec"]["profiles"] == list(FULL_PROFILES) for item in full)
    assert all(item["spec"]["seeds"] == list(FULL_SEEDS) for item in full)
    assert all(item["spec"]["eligible_if"] == "feature_screen_top_two" for item in full)


def test_active_sota_makes_four_feature_sets_by_four_model_families(
    monkeypatch,
) -> None:
    original = tournament_module.get_feature_set

    def get_feature_set(feature_set_id: str):
        if feature_set_id == "research-sota-v1":
            return {
                "id": feature_set_id,
                "definition_sha256": "f" * 64,
                "features": {"sota_factor": "$close"},
            }
        return original(feature_set_id)

    monkeypatch.setattr(tournament_module, "get_feature_set", get_feature_set)
    manifest = build_preregistered_manifest(
        active_sota_feature_set_id="research-sota-v1"
    )
    feature_trials = [
        item for item in manifest["trials"] if item["trial_kind"] == "feature_set"
    ]
    model_trials = [
        item for item in manifest["trials"] if item["trial_kind"] == "model"
    ]

    assert len(feature_trials) == 4
    assert len(model_trials) == 4 * len(MODEL_FAMILIES)
    assert {
        (item["feature_set_id"], item["model_family"]) for item in model_trials
    } == {
        (feature_set_id, family)
        for feature_set_id in {
            "qlib-alpha158",
            "qlib-alpha360",
            "platform-seed-v1",
            "research-sota-v1",
        }
        for family in MODEL_FAMILIES
    }


def test_quant_input_requires_current_identity_revalidation() -> None:
    finalist_multiple = {
        "contract_version": "prediction-finalist-multiple-testing-v2",
        "eligible_trial_names": ["model-full:model-1"],
    }
    finalist_multiple["evidence_sha256"] = canonical_sha256(finalist_multiple)
    selection = {
        "contract_version": "prediction-champion-selection-v1",
        "dataset_identity_sha256": "a" * 64,
        "selected_candidate_id": "model-1",
        "global_multiple_testing": finalist_multiple,
    }
    selection["evidence_sha256"] = canonical_sha256(selection)
    model_multiple = {
        "contract_version": "model-full-multiple-testing-v2",
        "eligible_trial_names": ["model-full:model-1"],
    }
    model_multiple["evidence_sha256"] = canonical_sha256(model_multiple)
    model_selection = {
        "contract_version": "model-family-champions-v2",
        "dataset_identity_sha256": "a" * 64,
        "champions": [],
        "multiple_testing": model_multiple,
    }
    model_selection["evidence_sha256"] = canonical_sha256(model_selection)
    cycle = {
        "dataset_identity_sha256": "a" * 64,
        "state": {
            "prediction_champion": {
                "kind": "model",
                "candidate_id": "model-1",
                "primary_model_candidate_id": "model-1",
                "manifest_sha256": "b" * 64,
                "admission_evidence_sha256": "c" * 64,
            },
            "prediction_champion_evidence": selection,
            "model_champion_evidence": model_selection,
        }
    }
    controller = AutopilotController.__new__(AutopilotController)
    controller._active_sota_feature_set_id = lambda _dataset: None

    first = controller._quant_input_sha256(
        cycle, {"name": "snapshot-a", "provenance": {"dataset_identity_sha256": "a" * 64}}
    )
    second = controller._quant_input_sha256(
        cycle,
        {
            "name": "same-immutable-snapshot-with-different-display-name",
            "provenance": {"dataset_identity_sha256": "a" * 64},
        },
    )
    stale = controller._quant_input_sha256(
        cycle, {"provenance": {"dataset_identity_sha256": "d" * 64}}
    )

    assert first is not None
    assert first == second
    assert stale is None


def test_prediction_champion_roll_forward_is_reference_only_until_revalidated() -> None:
    source_identity = "a" * 64
    target_identity = "d" * 64
    lineage_id = "f" * 64
    selection = {
        "dataset_identity_sha256": source_identity,
        "evidence_sha256": "b" * 64,
    }
    model_selection = {
        "dataset_identity_sha256": source_identity,
        "evidence_sha256": "c" * 64,
    }
    cycle = {
        "id": "cycle-current",
        "dataset_identity_sha256": target_identity,
        "dataset_lineage_id": lineage_id,
        "stage": "parallel_research",
        "state": {
            "prediction_champion": {
                "kind": "model",
                "candidate_id": "old-model",
            },
            "prediction_champion_evidence": selection,
            "model_champion_evidence": model_selection,
        },
    }

    class Store:
        @staticmethod
        def patch_cycle_state(cycle_id, *, state_patch, stage=None):
            assert cycle_id == "cycle-current"
            assert stage is None
            cycle["state"] = {**cycle["state"], **state_patch}
            return cycle

    controller = AutopilotController.__new__(AutopilotController)
    controller.store = Store()
    updated = controller._roll_forward_prediction_champion(
        cycle,
        {
            "lineage_id": lineage_id,
            "provenance": {"dataset_identity_sha256": target_identity},
        },
    )

    assert updated["state"]["prediction_champion"] is None
    assert updated["state"]["prediction_champion_evidence"] is None
    assert updated["state"]["prior_prediction_champion"]["candidate_id"] == "old-model"
    assert (
        updated["state"]["prediction_champion_status"]
        == "pending_current_identity_revalidation"
    )
    rollover = updated["state"]["prediction_champion_roll_forward"]
    assert rollover["reference_only"] is True
    assert rollover["eligible_for_fin_quant"] is False
    assert rollover["target_dataset_identity_sha256"] == target_identity


def test_quant_due_rejects_cross_identity_champion_before_database_access() -> None:
    controller = AutopilotController.__new__(AutopilotController)
    cycle = {
        "id": "cycle-current",
        "dataset_identity_sha256": "d" * 64,
        "dataset_lineage_id": "f" * 64,
        "state": {
            "prediction_champion": {
                "kind": "model",
                "candidate_id": "old-model",
            },
            "prediction_champion_evidence": {
                "dataset_identity_sha256": "a" * 64,
            },
            "model_champion_evidence": {
                "dataset_identity_sha256": "a" * 64,
            },
        },
    }

    assert controller._quant_due(
        cycle,
        {"provenance": {"dataset_identity_sha256": "d" * 64}},
        datetime(2026, 8, 26, tzinfo=UTC),
        {"quant_cooldown_days": 7},
        input_sha256="1" * 64,
    ) is False


def test_model_research_runs_once_on_the_first_available_snapshot_of_a_new_month() -> None:
    class Connection:
        @staticmethod
        def scalar(_statement):
            return "admitted-model"

    class ConnectionContext:
        def __enter__(self):
            return Connection()

        def __exit__(self, *_args):
            return False

    class Engine:
        @staticmethod
        def connect():
            return ConnectionContext()

    class Store:
        @staticmethod
        def latest_branch(_scenario):
            return {
                "cycle_id": "previous-cycle",
                "created_at": datetime(2026, 8, 1, tzinfo=UTC),
            }

        @staticmethod
        def get_cycle(_cycle_id):
            return {"state": {"dataset_end_date": "2026-08-31"}}

    controller = AutopilotController.__new__(AutopilotController)
    controller.engine = Engine()
    controller.store = Store()
    now = datetime(2026, 9, 1, 12, tzinfo=UTC)
    config = normalize_autopilot_config()

    assert controller._model_due({"end_date": "2026-08-31"}, now, config) is False
    assert controller._model_due({"end_date": "2026-09-02"}, now, config) is True


def test_full_model_score_uses_equal_three_window_mean_and_worst_window() -> None:
    def evidence(returns: dict[str, float]) -> dict:
        return {
            "cells": [
                {
                    "profile_id": profile_id,
                    "seed": seed,
                    "gate_status": "passed",
                    "metrics": {
                        "annualized_excess_return_with_cost": value,
                        "information_ratio": value * 5,
                        "rank_ic": value / 2,
                        "average_turnover": 0.10,
                        "max_drawdown": -0.12,
                    },
                }
                for profile_id, value in returns.items()
                for seed in FULL_SEEDS
            ]
        }

    recent_only_star = AutopilotController._model_tournament_score(
        evidence({"recent_3y": 0.30, "balanced_5y": 0.01, "robust_10y": 0.005})
    )
    stable = AutopilotController._model_tournament_score(
        evidence({"recent_3y": 0.12, "balanced_5y": 0.11, "robust_10y": 0.10})
    )

    assert stable > recent_only_star
    assert stable[1] == pytest.approx(0.10)


def test_model_tournament_closes_when_all_cross_family_pairs_are_too_correlated(
    tmp_path,
) -> None:
    trial_names = ["model-full:trial-ridge", "model-full:trial-gru"]
    returns_path = tmp_path / "model-returns.parquet"
    pd.DataFrame(
        {
            trial_names[0]: [0.0010 + ((index % 7) - 3) * 0.00001 for index in range(120)],
            trial_names[1]: [0.0008 + ((index % 5) - 2) * 0.00001 for index in range(120)],
        },
        index=pd.date_range("2025-01-01", periods=120),
    ).to_parquet(returns_path)
    returns = pd.read_parquet(returns_path)
    definitions = [
        {
            "name": trial_names[0],
            "candidate_id": "ridge-1",
            "trial_id": "trial-ridge",
            "kind": "model",
        },
        {
            "name": trial_names[1],
            "candidate_id": "gru-1",
            "trial_id": "trial-gru",
            "kind": "model",
        },
    ]
    model_multiple = _profile_family_multiple_testing(
        research_run_id="tournament:tournament-1:model_full",
        trial_series_by_profile={
            profile_id: [
                (definitions[0], returns[trial_names[0]]),
                (definitions[1], returns[trial_names[1]]),
            ]
            for profile_id in FULL_PROFILES
        },
        family_definitions=definitions,
        output=tmp_path / "model-multiple",
        contract_version="model-full-multiple-testing-v2",
    )
    model_selection = {
        "contract_version": "model-family-champions-v2",
        "dataset_identity_sha256": "d" * 64,
        "champions": [],
        "multiple_testing": model_multiple,
        "multiple_testing_evidence_sha256": model_multiple["evidence_sha256"],
    }
    model_selection["evidence_sha256"] = canonical_sha256(model_selection)
    cycle = {
        "id": "cycle-1",
        "state": {"model_champion_evidence": model_selection},
    }

    class Store:
        def __init__(self) -> None:
            self.stage = ""

        def patch_cycle_state(self, _cycle_id, *, state_patch, stage):
            cycle["state"].update(state_patch)
            self.stage = stage

    class Ensembles:
        @staticmethod
        def ensure_candidates(**_kwargs):
            return {
                "status": "no_admissible_combinations",
                "candidate_count": 0,
                "candidates": [],
                "pairwise_correlation_evidence": [
                    {
                        "left_candidate_id": "ridge-1",
                        "right_candidate_id": "gru-1",
                        "maximum_mean_absolute_daily_rank_correlation": 0.97,
                        "passed": False,
                    }
                ],
            }

    class Tournaments:
        def __init__(self) -> None:
            self.completed = None

        @staticmethod
        def get_for_cycle(_cycle_id):
            return {
                "id": "tournament-1",
                "status": "running",
                "trials": [
                    {
                        "id": "trial-ridge",
                        "trial_kind": "model",
                        "candidate_id": "ridge-1",
                        "status": "selected",
                    },
                    {
                        "id": "trial-gru",
                        "trial_kind": "model",
                        "candidate_id": "gru-1",
                        "status": "selected",
                    },
                ],
            }

        def complete_selection(self, tournament_id, *, selected_trial_ids, multiple_testing):
            self.completed = {
                "tournament_id": tournament_id,
                "selected_trial_ids": selected_trial_ids,
                "multiple_testing": multiple_testing,
            }

    controller = AutopilotController.__new__(AutopilotController)
    controller.store = Store()
    controller.model_ensembles = Ensembles()
    controller.tournaments = Tournaments()
    controller.settings = SimpleNamespace(data_root=tmp_path)
    champions = [
        {
            "trial_id": "trial-ridge",
            "candidate_id": "ridge-1",
            "model_family": "ridge",
            "feature_set_id": "qlib-alpha158",
                "score": [0.12, 0.03],
                "evidence_sha256": "a" * 64,
                "trial_name": trial_names[0],
        },
        {
            "trial_id": "trial-gru",
            "candidate_id": "gru-1",
            "model_family": "gru",
            "feature_set_id": "qlib-alpha360",
                "score": [0.10, 0.04],
                "evidence_sha256": "b" * 64,
                "trial_name": trial_names[1],
        },
    ]
    created, failed = controller._reconcile_model_ensembles(
        cycle,
        {
            "name": "snapshot-v1",
            "provenance": {"dataset_identity_sha256": "d" * 64},
        },
        {"id": "tournament-1", "status": "running"},
        champions,
    )
    assert (created, failed) == (0, 0)
    assert controller.tournaments.completed is not None
    assert controller.tournaments.completed["selected_trial_ids"] == [
        "trial-gru",
        "trial-ridge",
    ]
    assert cycle["state"]["prediction_champion"]["candidate_id"] == "ridge-1"
    assert cycle["state"]["model_ensemble_status"] == (
        "no_admissible_combinations"
    )
    assert controller.store.stage == "joint_optimization"
