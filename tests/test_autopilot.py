from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_platform.autopilot as autopilot_module
from quant_platform import research_tournament as tournament_module
from quant_platform.autopilot import (
    AUTOPILOT_CONTRACT_VERSION,
    AUTOPILOT_RESEARCH_HORIZONS,
    AutopilotController,
    _asset_available_on,
    _cycle_has_capital_commitment,
    _cycle_has_legacy_capital_state,
    _cycle_terminal_resolution,
    _derived_branch_status,
    _profile_family_multiple_testing,
    horizon_research_cadence_bucket,
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
    assert config["quant_loop_n"] == 2
    assert config["factor_research_interval_days"] == 1
    assert normalize_autopilot_config({"quant_loop_n": 1})["quant_loop_n"] == 2


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
    # A terminal current evaluator is authoritative when the parent run
    # projection is stale; otherwise the cycle looks active forever even
    # though no job can be claimed.
    assert _derived_branch_status("running", "failed") == "failed"
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
    assert _cycle_terminal_resolution(
        [("fin_factor", "succeeded"), ("fin_quant", "succeeded")]
    ) == (
        "succeeded",
        "research_complete",
    )


def test_historical_capital_commitment_classifier_remains_for_audit() -> None:
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


def test_only_persisted_old_capital_state_is_legacy_readonly() -> None:
    assert not _cycle_has_legacy_capital_state(
        {
            "state": {},
            "branches": [{"scenario": "fin_quant", "status": "succeeded"}],
        }
    )
    assert _cycle_has_legacy_capital_state(
        {"state": {"capital_pipeline": {"phase": "formal_backtest"}}}
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
        "horizon_profile": "short_1_5d",
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
    controller._active_sota_feature_set_id = (
        lambda _dataset, *, horizon_profile: None
    )

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


def test_model_research_uses_each_horizons_calendar_cadence() -> None:
    class Store:
        @staticmethod
        def latest_branch(_scenario, *, horizon_profile):
            return {
                "cycle_id": "previous-cycle",
                "created_at": datetime(2026, 8, 1, tzinfo=UTC),
            }

        @staticmethod
        def get_cycle(_cycle_id):
            return {
                "horizon_profile": Store.horizon_profile,
                "state": {"dataset_end_date": Store.source_end},
            }

    controller = AutopilotController.__new__(AutopilotController)
    controller.store = Store()
    now = datetime(2026, 9, 1, 12, tzinfo=UTC)
    config = normalize_autopilot_config()

    cases = (
        ("short_1_5d", "2026-08-24", "2026-08-28", "2026-08-31"),
        ("swing_1_6m", "2026-08-03", "2026-08-31", "2026-09-01"),
        ("long_1_3y", "2026-07-01", "2026-09-30", "2026-10-08"),
    )
    for horizon, source_end, same_bucket, next_bucket in cases:
        Store.horizon_profile = horizon
        Store.source_end = source_end
        assert controller._model_due(
            {"end_date": same_bucket},
            now,
            config,
            horizon_profile=horizon,
        ) is False
        assert controller._model_due(
            {"end_date": next_bucket},
            now,
            config,
            horizon_profile=horizon,
        ) is True


def test_factor_cadence_accepts_serialized_database_timestamp() -> None:
    class Store:
        @staticmethod
        def latest_branch(_scenario, *, horizon_profile):
            return {
                "cycle_id": "prior",
                "created_at": "2026-08-28T11:00:00+00:00",
            }

        @staticmethod
        def get_cycle(_cycle_id):
            return {
                "horizon_profile": "short_1_5d",
                "state": {"dataset_end_date": "2026-08-28"},
            }

    controller = AutopilotController.__new__(AutopilotController)
    controller.store = Store()

    assert controller._factor_due(
        {"end_date": "2026-08-31"},
        datetime(2026, 8, 29, 12, tzinfo=UTC),
        normalize_autopilot_config(),
        horizon_profile="short_1_5d",
    ) is True


def test_automatic_research_cadence_has_exactly_three_governed_lanes() -> None:
    assert AUTOPILOT_RESEARCH_HORIZONS == (
        "short_1_5d",
        "swing_1_6m",
        "long_1_3y",
    )
    assert horizon_research_cadence_bucket("short_1_5d", "2026-08-31").startswith(
        "week:"
    )
    assert horizon_research_cadence_bucket("swing_1_6m", "2026-08-31") == (
        "month:2026-08"
    )
    assert horizon_research_cadence_bucket("long_1_3y", "2026-08-31") == (
        "quarter:2026-Q3"
    )


def test_migrated_unbound_cycle_stays_read_only_legacy_evidence() -> None:
    policy = autopilot_module.primary_label_policy_contract()
    row = {
        "id": "legacy-cycle",
        "horizon_profile": "legacy_ambiguous",
        "primary_label_policy_sha256": policy["policy_sha256"],
        "state_json": {
            "horizon_profile": "legacy_ambiguous",
            "label_horizon_sessions": None,
            "primary_label_policy": policy,
            "historical_results_only": True,
            "capital_eligible": False,
            "final_oos_must_not_open": True,
            "migrated_from_unbound_autopilot_cycle": True,
        },
    }

    decoded = autopilot_module.AutopilotStore._decode_cycle(dict(row))
    assert decoded["horizon_profile"] == "legacy_ambiguous"
    assert decoded["state"]["label_horizon_sessions"] is None

    relabelled = dict(row)
    relabelled["state_json"] = {**row["state_json"], "label_horizon_sessions": 5}
    with pytest.raises(ValueError, match="reinterpreted"):
        autopilot_module.AutopilotStore._decode_cycle(relabelled)


def test_autopilot_tick_dispatches_the_same_mainline_for_all_horizons() -> None:
    calls: list[str] = []
    reconciliations: list[str] = []
    store_reconciliations: list[str] = []

    class Store:
        @staticmethod
        def reconcile() -> int:
            store_reconciliations.append("store")
            return 0

    controller = AutopilotController.__new__(AutopilotController)
    controller.store = Store()
    controller.research = SimpleNamespace(
        reconcile_autopilot_execution_state=lambda: reconciliations.append("research")
    )
    controller.config = lambda: (normalize_autopilot_config(), 7)
    controller._latest_dataset = lambda: {"name": "latest"}

    def tick_horizon(**kwargs):
        calls.append(str(kwargs["horizon_profile"]))
        return {"cycles": 1, "branches": 1, "failed": 0}

    controller._tick_horizon = tick_horizon

    result = controller.tick(datetime(2026, 8, 31, tzinfo=UTC))

    assert calls == ["short_1_5d", "swing_1_6m", "long_1_3y"]
    assert reconciliations == ["research", "research"]
    assert store_reconciliations == ["store", "store"]
    assert result == {"cycles": 3, "branches": 3, "failed": 0}


@pytest.mark.parametrize(
    ("horizon_profile", "old_end", "latest_end"),
    [
        ("short_1_5d", "2026-08-24", "2026-08-25"),
        ("swing_1_6m", "2026-08-03", "2026-08-31"),
        ("long_1_3y", "2026-07-01", "2026-09-30"),
    ],
)
def test_new_daily_snapshot_continues_same_cadence_immutable_cycle(
    monkeypatch, horizon_profile, old_end, latest_end
) -> None:
    old_dataset = {
        "name": f"daily-{old_end}",
        "end_date": old_end,
        "provenance": {
            "dataset_identity_sha256": "a" * 64,
            "frequency": "day",
            "field_contract_version": "daily-field-v8",
            "eligibility_contract_version": "eligibility-v1",
            "research_features": {"version": 8},
        },
    }
    latest_dataset = {
        "name": f"daily-{latest_end}",
        "end_date": latest_end,
        "provenance": {
            "dataset_identity_sha256": "b" * 64,
            "frequency": "day",
            "field_contract_version": "daily-field-v8",
            "eligibility_contract_version": "eligibility-v1",
            "research_features": {"version": 8},
        },
    }
    old_cycle = {
        "id": "cycle-old",
        "status": "active",
        "horizon_profile": horizon_profile,
        "dataset": old_dataset["name"],
        "dataset_identity_sha256": "a" * 64,
        "state": {"dataset_end_date": old_end},
        "branches": [{"scenario": "fin_factor", "status": "running"}],
    }
    calls: dict[str, object] = {}

    class Store:
        @staticmethod
        def list_cycles(*, limit):
            assert limit == 500
            return [old_cycle]

        @staticmethod
        def supersede_research_cycle(*_args, **_kwargs):
            raise AssertionError("same-cadence research must not be superseded")

        @staticmethod
        def ensure_cycle(*_args, **_kwargs):
            raise AssertionError("same-cadence research must not open a replacement")

        @staticmethod
        def get_cycle(cycle_id):
            assert cycle_id == old_cycle["id"]
            return old_cycle

    controller = AutopilotController.__new__(AutopilotController)
    controller.settings = SimpleNamespace(data_root="unused")
    controller.store = Store()
    controller._factor_due = lambda *_args, **_kwargs: False
    controller._model_due = lambda *_args, **_kwargs: False

    def roll_forward(cycle, bound_dataset):
        calls["cycle"] = cycle
        calls["dataset"] = bound_dataset
        return {**cycle, "status": "succeeded"}

    controller._roll_forward_prediction_champion = roll_forward
    monkeypatch.setattr(
        autopilot_module,
        "list_qlib_datasets",
        lambda _root: [old_dataset, latest_dataset],
    )

    result = controller._tick_horizon(
        current=datetime(2026, 8, 25, tzinfo=UTC),
        config=normalize_autopilot_config(),
        revision=1,
        dataset=latest_dataset,
        horizon_profile=horizon_profile,
    )

    assert calls == {"cycle": old_cycle, "dataset": old_dataset}
    assert result == {"cycles": 1, "branches": 0, "failed": 0}


def test_same_cadence_dataset_contract_migration_supersedes_obsolete_cycle(
    monkeypatch,
) -> None:
    old_dataset = {
        "name": "daily-obsolete-v5",
        "end_date": "2026-08-28",
        "provenance": {
            "dataset_identity_sha256": "a" * 64,
            "frequency": "day",
            "field_contract_version": "daily-field-v5",
            "eligibility_contract_version": "eligibility-v1",
            "research_features": {"version": 5},
        },
    }
    latest_dataset = {
        "name": "daily-pit-v8",
        "end_date": "2026-08-31",
        "provenance": {
            "dataset_identity_sha256": "b" * 64,
            "frequency": "day",
            "field_contract_version": "daily-field-v8",
            "eligibility_contract_version": "eligibility-v1",
            "research_features": {"version": 8},
        },
    }
    old_cycle = {
        "id": "cycle-obsolete",
        "status": "active",
        "horizon_profile": "swing_1_6m",
        "dataset": old_dataset["name"],
        "dataset_identity_sha256": "a" * 64,
        "state": {"dataset_end_date": old_dataset["end_date"]},
        "branches": [{"scenario": "fin_factor", "status": "running"}],
    }
    replacement_cycle = {
        "id": "cycle-current",
        "status": "active",
        "horizon_profile": "swing_1_6m",
        "dataset": latest_dataset["name"],
        "dataset_identity_sha256": "b" * 64,
        "state": {"dataset_end_date": latest_dataset["end_date"]},
        "branches": [],
    }
    superseded: list[tuple[str, str]] = []
    rolled_forward: list[str] = []

    class Store:
        @staticmethod
        def list_cycles(*, limit):
            assert limit == 500
            return [old_cycle]

        @staticmethod
        def supersede_research_cycle(cycle_id, *, replacement_dataset):
            superseded.append((cycle_id, str(replacement_dataset["name"])))
            return old_cycle

        @staticmethod
        def ensure_cycle(dataset, **_kwargs):
            assert dataset == latest_dataset
            return replacement_cycle

        @staticmethod
        def get_cycle(cycle_id):
            assert cycle_id == replacement_cycle["id"]
            return replacement_cycle

        @staticmethod
        def set_cycle_state(*_args, **_kwargs):
            raise AssertionError("contract migration must bypass the cadence no-op")

    controller = AutopilotController.__new__(AutopilotController)
    controller.settings = SimpleNamespace(data_root="unused")
    controller.store = Store()
    controller._factor_due = lambda *_args, **_kwargs: False
    controller._model_due = lambda *_args, **_kwargs: False

    def roll_forward(cycle, bound_dataset):
        assert cycle == replacement_cycle
        assert bound_dataset == latest_dataset
        rolled_forward.append(str(cycle["id"]))
        return {**cycle, "status": "succeeded"}

    controller._roll_forward_prediction_champion = roll_forward
    monkeypatch.setattr(
        autopilot_module,
        "list_qlib_datasets",
        lambda _root: [old_dataset, latest_dataset],
    )

    result = controller._tick_horizon(
        current=datetime(2026, 8, 31, tzinfo=UTC),
        config=normalize_autopilot_config(),
        revision=1,
        dataset=latest_dataset,
        horizon_profile="swing_1_6m",
    )

    assert superseded == [("cycle-obsolete", "daily-pit-v8")]
    assert rolled_forward == ["cycle-current"]
    assert result == {"cycles": 1, "branches": 0, "failed": 0}


def test_missing_factor_branch_is_not_retried() -> None:
    controller = AutopilotController.__new__(AutopilotController)

    assert controller._retry_failed_branch(None) is False


def test_factor_model_and_quant_runs_bind_each_horizons_primary_label(
    tmp_path, monkeypatch
) -> None:
    calendar = tmp_path / "calendars"
    calendar.mkdir()
    (calendar / "day.txt").write_text(
        "2024-01-02\n2024-01-03\n", encoding="utf-8"
    )
    dataset = {
        "name": "governed-daily",
        "path": str(tmp_path),
        "lineage_id": "b" * 64,
        "provenance": {"dataset_identity_sha256": "a" * 64},
    }
    labels = {
        "short_1_5d": 5,
        "swing_1_6m": 63,
        "long_1_3y": 252,
    }
    policy = autopilot_module.primary_label_policy_contract()
    run_configs: list[dict] = []
    job_payloads: list[dict] = []
    branches: list[dict] = []

    class Research:
        @staticmethod
        def create_run(**kwargs):
            run_configs.append(dict(kwargs["config"]))
            return {"id": f"run-{len(run_configs)}"}

        @staticmethod
        def attach_job(_run_id, _job_id):
            return None

        @staticmethod
        def mark_run(*_args, **_kwargs):
            return None

    class Jobs:
        @staticmethod
        def create(_kind, payload, *_args, **_kwargs):
            job_payloads.append(dict(payload))
            return {"id": f"job-{len(job_payloads)}"}

    class Store:
        @staticmethod
        def create_branch(*_args, **kwargs):
            branches.append(dict(kwargs["details"]))

    class Assets:
        @staticmethod
        def reserve_automatic(**_kwargs):
            raise AssertionError("empty asset manifest must not reserve assets")

    monkeypatch.setattr(
        autopilot_module,
        "get_rdagent_scenario",
        lambda scenario_id: SimpleNamespace(
            id=scenario_id,
            requires_feature_set=scenario_id in {"fin_model", "fin_quant"},
            research_kind=scenario_id,
            job_kind=f"job:{scenario_id}",
        ),
    )
    monkeypatch.setattr(autopilot_module, "probe_rdagent", lambda *_args: {})
    monkeypatch.setattr(
        autopilot_module, "require_ready_scenario", lambda *_args: None
    )
    monkeypatch.setattr(
        autopilot_module,
        "expected_rdagent_runtime_identity",
        lambda _runtime, scenario_id: {"scenario": scenario_id},
    )
    monkeypatch.setattr(
        autopilot_module,
        "resolve_rdagent_assets",
        lambda *_args, **_kwargs: {"manifest_sha256": ""},
    )
    monkeypatch.setattr(
        autopilot_module,
        "get_feature_set",
        lambda feature_set_id: {"id": feature_set_id},
    )

    def resolve_window(_dataset, _calendar, *, horizon_profile, feature_set):
        label = labels[horizon_profile]
        return (
            {
                "train_start": "2020-01-02",
                "train_end": "2021-12-31",
                "valid_start": "2022-01-04",
                "valid_end": "2022-12-30",
                "test_start": "2023-01-03",
                "test_end": "2024-01-03",
            },
            {
                "evaluation_profiles": [],
                "research_window_contract": {
                    "horizon_profile": horizon_profile,
                    "label_horizon_sessions": label,
                    "feature_set_id": (feature_set or {}).get("id"),
                },
                "research_window_contract_sha256": str(label) * 64,
            },
        )

    monkeypatch.setattr(
        autopilot_module, "resolve_research_window_contract", resolve_window
    )

    controller = AutopilotController.__new__(AutopilotController)
    controller.settings = SimpleNamespace(data_root=tmp_path)
    controller.research = Research()
    controller.jobs = Jobs()
    controller.store = Store()
    controller.assets = Assets()
    config = normalize_autopilot_config()

    for horizon, label in labels.items():
        cycle = {
            "id": f"cycle-{horizon}",
            "horizon_profile": horizon,
            "primary_label_policy_sha256": policy["policy_sha256"],
            "state": {"primary_label_policy": policy},
        }
        for scenario in ("fin_factor", "fin_model", "fin_quant"):
            controller._enqueue(
                cycle,
                dataset,
                scenario,
                f"{scenario}:{horizon}",
                config=config,
            )
            assert run_configs[-1]["horizon_profile"] == horizon
            assert run_configs[-1]["label_horizon_sessions"] == label
            assert job_payloads[-1]["horizon_profile"] == horizon
            assert job_payloads[-1]["label_horizon_sessions"] == label
            assert branches[-1]["horizon_profile"] == horizon
            assert branches[-1]["primary_label_policy_sha256"] == policy[
                "policy_sha256"
            ]


@pytest.mark.parametrize(
    ("horizon_profile", "label_horizon_sessions"),
    [("short_1_5d", 5), ("swing_1_6m", 63), ("long_1_3y", 252)],
)
def test_cycle_materializes_base_only_horizon_factor_champion(
    tmp_path,
    monkeypatch,
    horizon_profile,
    label_horizon_sessions,
) -> None:
    calendar = tmp_path / "calendars"
    calendar.mkdir()
    (calendar / "day.txt").write_text("2024-01-02\n", encoding="utf-8")
    feature_set = {
        "id": "qlib-alpha158",
        "definition_sha256": "c" * 64,
        "contract_version": "governed-feature-set-v1",
        "source": "qlib",
        "features": {"KMID": "$close/$open-1"},
    }
    dataset = {
        "name": "governed-daily",
        "path": str(tmp_path),
        "provenance": {"dataset_identity_sha256": "a" * 64},
    }
    cycle = {
        "id": f"cycle-{horizon_profile}",
        "horizon_profile": horizon_profile,
        "dataset_identity_sha256": "a" * 64,
        "state": {},
    }
    patch_calls: list[dict] = []

    class Store:
        @staticmethod
        def patch_cycle_state(cycle_id, *, state_patch):
            assert cycle_id == cycle["id"]
            patch_calls.append(dict(state_patch))
            return {**cycle, "state": {**cycle["state"], **state_patch}}

    controller = AutopilotController.__new__(AutopilotController)
    controller.store = Store()
    monkeypatch.setattr(
        autopilot_module, "get_feature_set", lambda _feature_id: feature_set
    )
    monkeypatch.setattr(
        autopilot_module,
        "resolve_research_window_contract",
        lambda *_args, **_kwargs: (
            {"valid_end": "2023-12-29"},
            {
                "research_window_contract": {"horizon_profile": horizon_profile},
                "research_window_contract_sha256": "d" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        autopilot_module,
        "resolve_research_label_binding",
        lambda _payload: {
            "horizon_profile": horizon_profile,
            "label_horizon_sessions": label_horizon_sessions,
            "feature_set_id": feature_set["id"],
            "feature_set_sha256": feature_set["definition_sha256"],
            "dataset_identity_sha256": "a" * 64,
            "binding_sha256": "e" * 64,
        },
    )

    materialized = controller._ensure_horizon_factor_bundle(
        cycle,
        dataset,
        feature_set_id=feature_set["id"],
    )
    bundle = materialized["state"]["horizon_factor_bundle"]

    assert bundle["incremental_factors"] == []
    assert bundle["incremental_challenge"]["mode"] == "baseline_seed"
    assert bundle["horizon_profile"] == horizon_profile
    assert bundle["label_horizon_sessions"] == label_horizon_sessions
    assert materialized["state"]["horizon_factor_bundle_sha256"] == bundle[
        "bundle_sha256"
    ]
    assert len(patch_calls) == 1

    replay = controller._ensure_horizon_factor_bundle(
        materialized,
        dataset,
        feature_set_id=feature_set["id"],
    )
    assert replay == materialized
    assert len(patch_calls) == 1


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
