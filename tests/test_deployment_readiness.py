from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from governance_fixtures import governed_etf_ready_evidence

import quant_platform.deployment_readiness as readiness_module
from quant_data.config import Settings
from quant_data.execution_contract import DAILY_QLIB_FIELD_CONTRACT_VERSION
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_data.qlib_builder import build_qlib_output_manifest
from quant_platform.alpha_spending_ledger import CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS
from quant_platform.auth_store import AuthStore
from quant_platform.data_automation import DEFAULT_STRATEGY_MINUTE_SYMBOLS
from quant_platform.data_task_store import DataTaskStore
from quant_platform.deployment_readiness import (
    RESEARCH_MINIMUM_TRADING_DAYS,
    DeploymentReadinessStore,
)
from quant_platform.health_store import OperationalHealthStore
from quant_platform.job_store import JobStore
from quant_platform.model_research_governance import (
    MODEL_LABEL_HORIZON_TRADING_DAYS,
)
from quant_platform.research_automation import (
    DEFAULT_RESEARCH_PERIOD_POLICY,
    MINIMUM_PROFILE_TRAINING_DAYS,
    RESEARCH_EVALUATION_PROFILES,
)
from quant_platform.runtime_secret_store import RuntimeSecretStore
from quant_platform.schedule_store import ScheduleStore
from quant_platform.scheduler import AUTOMATED_DATA_BUNDLES
from quant_platform.services import refresh_qlib_display_catalog
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.transparent_baseline_governance import (
    LOCKBOX_CONFIG_KEY,
    build_all_unavailable_cash_only_receipt,
    build_joint_lockbox,
    build_unopened_history_selection,
)
from quant_platform.transparent_baseline_runner import (
    FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    target_runner_for_recipe,
    target_runtime_bundle_for_recipe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.no_database
def test_governed_schedule_suite_cardinality_matches_all_five_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Row:
        def __init__(self, kind: str) -> None:
            self.id = f"schedule-{kind}"
            self.kind = kind

    for validator in (
        "_is_governed_suite_data_pipeline",
        "_is_governed_information_pipeline",
        "_is_governed_information_factor_refresh",
        "_is_governed_ashare_5m_sync",
        "_is_governed_auxiliary_data_pipeline",
    ):
        monkeypatch.setattr(
            readiness_module,
            validator,
            lambda *_args, **_kwargs: True,
        )
    rows = [
        Row(kind)
        for kind in sorted(readiness_module._GOVERNED_DATA_SCHEDULE_SUITE_KINDS)
    ]

    ready, governed = readiness_module._governed_schedule_suite_state(
        rows,
        data_root=tmp_path,
        reproducible_dataset_names=set(),
    )

    assert ready is True
    assert set(governed) == readiness_module._GOVERNED_DATA_SCHEDULE_SUITE_KINDS
    assert all(len(ids) == 1 for ids in governed.values())


@pytest.mark.no_database
def test_latest_closed_trading_day_excludes_an_unfinished_session() -> None:
    open_days = [date(2026, 8, 28), date(2026, 8, 31)]

    before_close = readiness_module._latest_closed_trading_day(
        open_days,
        now=datetime(2026, 8, 31, 6, 59, tzinfo=UTC),
    )
    after_close = readiness_module._latest_closed_trading_day(
        open_days,
        now=datetime(2026, 8, 31, 7, 1, tzinfo=UTC),
    )

    assert before_close == date(2026, 8, 28)
    assert after_close == date(2026, 8, 31)


@pytest.mark.no_database
def test_daily_business_check_requires_fresh_sealed_daily_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = {
        "frequency": "day",
        "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
        "source_volume_unit": "hand",
        "qlib_volume_unit": "share",
        "source_amount_unit": "thousand_cny",
        "qlib_amount_unit": "cny",
        "source_hand_size": 100,
        "index_volume_policy": "excluded_non_tradable_benchmark",
        "governed_etf_whitelist": governed_etf_ready_evidence(),
        "lineage_verified": True,
        "execution_controls": {
            "formal_execution_requires_native_controls": True,
            "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
            "native_complete_from": "2016-01-04",
        },
    }
    dataset = {
        "name": "daily-production",
        "frequency": "day",
        "ready": True,
        "reproducible": True,
        "lineage_verified": True,
        "output_files_verified": True,
        "output_verification": "verified",
        "end_date": "2026-08-31",
        "daily_contract": provenance,
    }
    monkeypatch.setattr(
        readiness_module,
        "load_trade_calendar_open_days",
        lambda _root: [date(2026, 8, 28), date(2026, 8, 31)],
    )
    monkeypatch.setattr(
        readiness_module,
        "list_qlib_datasets_for_display",
        lambda _root: [dataset],
    )

    fresh = readiness_module._daily_qlib_business_check(
        tmp_path,
        now=datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
    )
    dataset["end_date"] = "2026-08-28"
    stale = readiness_module._daily_qlib_business_check(
        tmp_path,
        now=datetime(2026, 8, 31, 8, 0, tzinfo=UTC),
    )

    assert fresh["status"] == "ok"
    assert fresh["expected_end_date"] == "2026-08-31"
    assert stale["status"] == "blocked"
    assert stale["datasets"][0]["reasons"] == ["stale"]


@pytest.mark.no_database
def test_horizon_readiness_requires_fresh_health_for_paper_and_recommendation() -> None:
    paper = {
        "strategy_version_id": "paper-short",
        "promotion_stage": "paper",
        "signal_frequency": "day",
        "execution_frequency": "day",
        "contract_ready": True,
        "health_status": "healthy",
        "health_evidence_ready": True,
        "health_evidence_reasons": [],
        "paper_stage_status": "active",
        "simulation_status": "active",
        "active_recommendation_portfolios": 0,
    }

    validating = readiness_module._assess_horizon_candidates(
        "short_1_5d",
        [paper],
    )
    unhealthy_recommendation = readiness_module._assess_horizon_candidates(
        "short_1_5d",
        [
            paper,
            {
                **paper,
                "strategy_version_id": "recommendation-short",
                "promotion_stage": "recommendation_enabled",
                "health_status": "suspended",
                "health_evidence_ready": True,
                "active_recommendation_portfolios": 1,
            },
        ],
    )

    assert validating["status"] == "ok"
    assert validating["stage"] == "paper"
    assert validating["health_status"] == "healthy"
    assert unhealthy_recommendation["status"] == "blocked"
    assert unhealthy_recommendation["candidates"][0]["blocking_reasons"] == [
        "strategy_health_suspended"
    ]

    paper["health_evidence_ready"] = False
    paper["health_evidence_reasons"] = ["strategy_health_evidence_stale"]
    stale = readiness_module._assess_horizon_candidates("short_1_5d", [paper])
    assert stale["status"] == "blocked"
    assert stale["candidates"][0]["blocking_reasons"] == [
        "strategy_health_evidence_stale"
    ]


def _cash_only_lockbox_rows() -> tuple[list[dict], dict]:
    recipe_ids = ("short_relative_strength",)
    members: list[dict[str, str]] = []
    for recipe_id in recipe_ids:
        recipe = get_strategy_recipe(recipe_id)
        member = {
            "recipe_id": recipe_id,
            "recipe_version": str(recipe["version"]),
            "recipe_sha256": readiness_module.canonical_sha256(recipe),
            "horizon_profile": str(recipe["horizon"]),
            "base_config_sha256": readiness_module.canonical_sha256(
                {"recipe_id": recipe_id}
            ),
            "baseline_definition_sha256": readiness_module.canonical_sha256(
                {"baseline": recipe_id}
            ),
            "research_window_contract_sha256": readiness_module.canonical_sha256(
                {"window": recipe_id}
            ),
            "historical_start": "2016-01-04",
            "historical_end": "2020-12-31",
            "test_start": "2021-01-04",
            "test_end": "2023-12-29",
        }
        runner = target_runner_for_recipe(recipe_id, str(recipe["version"]))
        if runner is not None:
            member[TRANSPARENT_BASELINE_RUNNER_FIELD] = runner
        runtime_bundle = target_runtime_bundle_for_recipe(
            recipe_id, str(recipe["version"])
        )
        if runtime_bundle is not None:
            member[TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD] = runtime_bundle
        members.append(member)
    version = str(get_strategy_recipe("short_relative_strength")["version"])
    selection = build_unopened_history_selection(
        calendar_days=["2016-01-04", "2026-08-28"],
        current_recipe_version=version,
        prior_batches=[],
    )
    unavailable_horizons = []
    for recipe_id in ("swing_trend", "long_quality_value"):
        recipe = get_strategy_recipe(recipe_id)
        evidence = {
            "capital_evaluation_eligible": False,
            "capital_evaluation_unavailable_reason": (
                "insufficient_native_execution_controlled_sessions_before_immutable_cutoff"
            ),
        }
        unavailable_horizons.append(
            {
                "recipe_id": recipe_id,
                "horizon_profile": str(recipe["horizon"]),
                "status": "unavailable",
                "reason": "the sealed OOS window is unavailable",
                "evidence": evidence,
                "evidence_sha256": readiness_module.canonical_sha256(evidence),
            }
        )
    lockbox = build_joint_lockbox(
        dataset="daily-v12",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
        members=members,
        unopened_history_selection=selection,
        unavailable_horizons=unavailable_horizons,
    )
    rows = [
        {
            "horizon_profile": member["horizon_profile"],
            "created_by": "system:transparent-baseline-bootstrap",
            "config_json": {
                "recipe_id": member["recipe_id"],
                "recipe_version": member["recipe_version"],
                "horizon_profile": member["horizon_profile"],
                LOCKBOX_CONFIG_KEY: lockbox,
            },
        }
        for member in members
    ]
    return rows, lockbox


def _all_unavailable_cash_only_audit_rows(
    *, recipe_version: str | None = None
) -> tuple[list[dict], dict]:
    version = recipe_version or str(
        get_strategy_recipe("short_relative_strength")["version"]
    )
    selection = build_unopened_history_selection(
        calendar_days=["2026-08-27", "2026-08-28"],
        current_recipe_version=version,
        prior_batches=[],
    )
    unavailable = []
    for recipe_id in (
        "short_relative_strength",
        "swing_trend",
        "long_quality_value",
    ):
        recipe = get_strategy_recipe(recipe_id)
        evidence = {
            "capital_evaluation_eligible": False,
            "capital_evaluation_unavailable_reason": "no honest unopened OOS",
        }
        unavailable.append(
            {
                "recipe_id": recipe_id,
                "horizon_profile": str(recipe["horizon"]),
                "status": "unavailable",
                "reason": "the sealed OOS window is unavailable",
                "evidence": evidence,
                "evidence_sha256": readiness_module.canonical_sha256(evidence),
            }
        )
    receipt = build_all_unavailable_cash_only_receipt(
        dataset="daily-v19",
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
        current_recipe_version=version,
        unopened_history_selection=selection,
        unavailable_horizons=unavailable,
    )
    row = {
        "id": 91,
        "user_id": None,
        "username": "system:transparent-baseline-bootstrap",
        "action": readiness_module.ALL_UNAVAILABLE_CASH_ONLY_ACTION,
        "method": "INTERNAL",
        "path": "transparent-baseline/all-unavailable-cash-only",
        "status_code": 201,
        "ip_hash": None,
        "user_agent": "transparent_baseline_bootstrap.py",
        "details_json": receipt,
        "created_at": datetime(2026, 9, 1, tzinfo=UTC),
    }
    return [row], receipt


@pytest.mark.no_database
def test_readiness_accepts_governed_all_unavailable_cash_only_receipt() -> None:
    rows, receipt = _all_unavailable_cash_only_audit_rows()

    lanes = readiness_module._validated_all_unavailable_cash_only_horizon_lanes(
        rows,
        lockbox_rows=[],
    )

    assert set(lanes) == {"short_1_5d", "swing_1_6m", "long_1_3y"}
    for lane in lanes.values():
        assert lane["status"] == "ok"
        assert lane["stage"] == "cash_only"
        assert lane["strategy_version_id"] is None
        assert lane["runner"] == "cash_only_no_orders"
        assert lane["sleeve_action"] == "remain_in_cash"
        assert lane["new_entries_allowed"] is False
        assert lane["recommendation_eligible"] is False
        assert lane["cash_only_evidence"]["receipt_sha256"] == (
            receipt["receipt_sha256"]
        )
        assert lane["cash_only_evidence"]["strategy_version_created"] is False
        assert lane["cash_only_evidence"]["oos_reserved"] is False
        assert lane["cash_only_evidence"]["orders_eligible"] is False

    production = readiness_module._project_three_horizon_production(
        {horizon: [] for horizon in lanes},
        lanes,
    )
    assert production["status"] == "ok"
    assert production["cash_only_horizons"] == [
        "short_1_5d",
        "swing_1_6m",
        "long_1_3y",
    ]


@pytest.mark.no_database
def test_all_unavailable_cash_only_readiness_fails_closed() -> None:
    rows, _receipt = _all_unavailable_cash_only_audit_rows()
    tampered = deepcopy(rows)
    tampered[0]["details_json"]["unavailable_horizons"][0]["reason"] = (
        "runtime crashed"
    )
    assert (
        readiness_module._validated_all_unavailable_cash_only_horizon_lanes(
            tampered,
            lockbox_rows=[],
        )
        == {}
    )

    stale, _ = _all_unavailable_cash_only_audit_rows(recipe_version="stale-v18")
    assert (
        readiness_module._validated_all_unavailable_cash_only_horizon_lanes(
            stale,
            lockbox_rows=[],
        )
        == {}
    )

    current_version = str(
        get_strategy_recipe("short_relative_strength")["version"]
    )
    current_strategy_row = {
        "created_by": "system:transparent-baseline-bootstrap",
        "config_json": {
            "recipe_id": "short_relative_strength",
            "recipe_version": current_version,
        },
    }
    assert (
        readiness_module._validated_all_unavailable_cash_only_horizon_lanes(
            rows,
            lockbox_rows=[current_strategy_row],
        )
        == {}
    )


@pytest.mark.no_database
def test_readiness_accepts_only_sealed_unavailable_horizon_as_cash_only() -> None:
    rows, lockbox = _cash_only_lockbox_rows()

    lanes = readiness_module._validated_cash_only_horizon_lanes(rows)

    assert set(lanes) == {"swing_1_6m", "long_1_3y"}
    long_lane = lanes["long_1_3y"]
    assert long_lane["status"] == "ok"
    assert long_lane["stage"] == "cash_only"
    assert long_lane["sleeve_action"] == "remain_in_cash"
    assert long_lane["new_entries_allowed"] is False
    assert long_lane["recommendation_eligible"] is False
    assert long_lane["lockbox_evidence"]["batch_sha256"] == lockbox["batch_sha256"]


@pytest.mark.no_database
def test_three_horizon_readiness_accepts_short_paper_and_sealed_cash_sleeves() -> None:
    rows, lockbox = _cash_only_lockbox_rows()
    assert lockbox["contract_version"] == readiness_module.LOCKBOX_CONTRACT_VERSION_V3
    cash_only_lanes = readiness_module._validated_cash_only_horizon_lanes(rows)
    short_paper = {
        "strategy_version_id": "v18-short-paper",
        "promotion_stage": "paper",
        "signal_frequency": "day",
        "execution_frequency": "day",
        "contract_ready": True,
        "health_status": "healthy",
        "health_evidence_ready": True,
        "health_evidence_reasons": [],
        "paper_stage_status": "active",
        "simulation_status": "active",
        "active_recommendation_portfolios": 0,
    }

    result = readiness_module._project_three_horizon_production(
        {
            "short_1_5d": [short_paper],
            "swing_1_6m": [],
            "long_1_3y": [],
        },
        cash_only_lanes,
    )

    assert result["status"] == "ok"
    assert result["blocked_horizons"] == []
    assert result["cash_only_horizons"] == ["swing_1_6m", "long_1_3y"]
    assert result["horizons"]["short_1_5d"]["strategy_version_id"] == (
        "v18-short-paper"
    )
    for horizon in result["cash_only_horizons"]:
        lane = result["horizons"][horizon]
        assert lane["stage"] == "cash_only"
        assert lane["strategy_version_id"] is None
        assert lane["new_entries_allowed"] is False
        assert lane["recommendation_eligible"] is False
        assert "simulation_portfolio_id" not in lane

    short_paper["simulation_status"] = "stopped"
    blocked = readiness_module._project_three_horizon_production(
        {
            "short_1_5d": [short_paper],
            "swing_1_6m": [],
            "long_1_3y": [],
        },
        cash_only_lanes,
    )
    assert blocked["status"] == "blocked"
    assert blocked["blocked_horizons"] == ["short_1_5d"]
    assert blocked["cash_only_horizons"] == ["swing_1_6m", "long_1_3y"]


@pytest.mark.no_database
def test_cash_only_readiness_fails_closed_without_current_valid_lockbox() -> None:
    rows, _ = _cash_only_lockbox_rows()
    tampered = deepcopy(rows)
    for row in tampered:
        row["config_json"][LOCKBOX_CONFIG_KEY]["unavailable_horizons"][0][
            "reason"
        ] = "runtime failed"

    # A malformed newest batch cannot fall back to an older valid lockbox.
    assert readiness_module._validated_cash_only_horizon_lanes([*tampered, *rows]) == {}
    newest_without_lockbox = {
        "horizon_profile": "short_1_5d",
        "created_by": "system:transparent-baseline-bootstrap",
        "config_json": {
            "recipe_id": "short_relative_strength",
            "recipe_version": get_strategy_recipe("short_relative_strength")[
                "version"
            ],
            "horizon_profile": "short_1_5d",
        },
    }
    assert readiness_module._validated_cash_only_horizon_lanes(
        [newest_without_lockbox, *rows]
    ) == {}
    untrusted = deepcopy(rows)
    for row in untrusted:
        row["created_by"] = "admin"
    assert readiness_module._validated_cash_only_horizon_lanes(untrusted) == {}
    assert readiness_module._validated_cash_only_horizon_lanes(
        [
            {
                "horizon_profile": "long_1_3y",
                "created_by": "system:transparent-baseline-bootstrap",
                "config_json": {
                    "recipe_id": "long_quality_value",
                    "recipe_version": get_strategy_recipe("long_quality_value")[
                        "version"
                    ],
                    "horizon_profile": "long_1_3y",
                    "last_error": "ordinary strategy failure",
                },
            }
        ]
    ) == {}
    missing = readiness_module._assess_horizon_candidates("long_1_3y", [])
    assert missing["status"] == "blocked"


def _terminal_short_cash_only_receipt() -> dict:
    return {
        "contract_version": readiness_module.TERMINAL_CASH_ONLY_CONTRACT_VERSION,
        "strategy_version_id": readiness_module.TERMINAL_CASH_ONLY_VERSION_ID,
        "backtest_id": readiness_module.TERMINAL_CASH_ONLY_BACKTEST_ID,
        "job_id": readiness_module.TERMINAL_CASH_ONLY_JOB_ID,
        "horizon_profile": "short_1_5d",
        "authority": "cash_only_projection_only",
        "cash_only_scope": "cash_only_projection_only",
        "formal_result_complete": False,
        "approval_eligible": False,
        "rerun_allowed": False,
        "robustness_gate": {
            "passed": 0,
            "total": 4,
            "min_pass_rate": 1.0,
            "passed_gate": False,
        },
        "receipt_sha256": "a" * 64,
    }


@pytest.mark.no_database
def test_terminally_rejected_short_baseline_projects_only_cash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def require_receipt(_connection, **kwargs):
        calls.append(kwargs)
        return _terminal_short_cash_only_receipt()

    monkeypatch.setattr(
        readiness_module,
        "require_terminal_cash_only_receipt",
        require_receipt,
    )

    lanes = readiness_module._validated_terminal_cash_only_horizon_lanes(
        object(), data_root=tmp_path
    )

    assert calls == [
        {"data_root": tmp_path, "verify_artifact_hashes": False}
    ]
    assert set(lanes) == {"short_1_5d"}
    lane = lanes["short_1_5d"]
    assert lane["status"] == "ok"
    assert lane["stage"] == "cash_only"
    assert lane["runner"] == "cash_only_no_orders"
    assert lane["strategy_version_id"] is None
    assert lane["new_entries_allowed"] is False
    assert lane["recommendation_eligible"] is False
    assert lane["terminal_failure_evidence"]["formal_result_complete"] is False
    assert lane["terminal_failure_evidence"]["robustness_gate"]["passed"] == 0


@pytest.mark.no_database
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("backtest_id",), "another-backtest"),
        (("strategy_version_id",), "another-version"),
        (("approval_eligible",), True),
        (("robustness_gate", "passed"), 1),
        (("receipt_sha256",), "x" * 64),
    ],
)
def test_terminal_short_cash_projection_fails_closed_on_tamper_or_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: object,
) -> None:
    receipt = _terminal_short_cash_only_receipt()
    target = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    monkeypatch.setattr(
        readiness_module,
        "require_terminal_cash_only_receipt",
        lambda *_args, **_kwargs: receipt,
    )

    assert (
        readiness_module._validated_terminal_cash_only_horizon_lanes(
            object(), data_root=tmp_path
        )
        == {}
    )


@pytest.mark.no_database
def test_terminal_short_cash_projection_fails_closed_when_receipt_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_receipt(*_args, **_kwargs):
        raise KeyError("tampered receipt")

    monkeypatch.setattr(
        readiness_module,
        "require_terminal_cash_only_receipt",
        reject_receipt,
    )

    assert (
        readiness_module._validated_terminal_cash_only_horizon_lanes(
            object(), data_root=tmp_path
        )
        == {}
    )


@pytest.mark.no_database
def test_real_short_production_lane_overrides_terminal_cash_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        readiness_module,
        "require_terminal_cash_only_receipt",
        lambda *_args, **_kwargs: _terminal_short_cash_only_receipt(),
    )
    cash_lanes = readiness_module._validated_terminal_cash_only_horizon_lanes(
        object(), data_root=tmp_path
    )
    short = _healthy_short_paper_candidate(strategy_version_id="passing-short")

    result = readiness_module._project_three_horizon_production(
        {
            "short_1_5d": [short],
            "swing_1_6m": [],
            "long_1_3y": [],
        },
        cash_lanes,
    )

    assert result["horizons"]["short_1_5d"]["stage"] == "paper"
    assert result["horizons"]["short_1_5d"]["strategy_version_id"] == (
        "passing-short"
    )
    assert "short_1_5d" not in result["cash_only_horizons"]


def _rehabilitation_cash_only_rows(
    *,
    short_version_id: str = "v18-short-paper",
) -> list[dict]:
    evidence_hashes = {
        "swing_1_6m": "c" * 64,
        "long_1_3y": "d" * 64,
    }
    qualification = {
        "contract_version": "forward-only-rehabilitation-v1",
        "evidence_mode": "consumed_historical_replay",
        "historical_replay_opened": True,
        "consumed_oos_replayed": True,
        "final_oos_opened": True,
        "capital_eligible": False,
        "sealed_final_oos": False,
        "unseen_oos": False,
        "authority": "historical_description_only",
        "source_strategy_version_id": "4414d202dbb641608975e5305bc18da4",
        "source_lockbox_contract_version": (
            "transparent-baseline-available-horizons-lockbox-v3"
        ),
        "source_lockbox_batch_sha256": "1" * 64,
        "source_lockbox_member_sha256": "2" * 64,
        "source_history_selection_sha256": "3" * 64,
        "source_unavailable_horizons_sha256": "4" * 64,
        "source_unavailable_evidence_sha256s": evidence_hashes,
        "source_cash_only_scope": "cash_only_projection_only",
        "strategy_version_id": short_version_id,
        "recipe_id": "short_relative_strength",
        "horizon_profile": "short_1_5d",
    }
    receipt_sha256 = readiness_module.canonical_sha256(qualification)
    return [
        {
            "receipt_sha256": receipt_sha256,
            "source_strategy_version_id": qualification[
                "source_strategy_version_id"
            ],
            "source_lockbox_contract_version": qualification[
                "source_lockbox_contract_version"
            ],
            "source_lockbox_batch_sha256": qualification[
                "source_lockbox_batch_sha256"
            ],
            "source_lockbox_member_sha256": qualification[
                "source_lockbox_member_sha256"
            ],
            "source_history_selection_sha256": qualification[
                "source_history_selection_sha256"
            ],
            "source_unavailable_horizons_sha256": qualification[
                "source_unavailable_horizons_sha256"
            ],
            "source_unavailable_evidence_sha256s_json": evidence_hashes,
            "source_cash_only_scope": qualification["source_cash_only_scope"],
            "strategy_version_id": short_version_id,
            "contract_version": qualification["contract_version"],
            "evidence_mode": qualification["evidence_mode"],
            "authority": qualification["authority"],
            "recipe_id": qualification["recipe_id"],
            "horizon_profile": qualification["horizon_profile"],
            "qualification_json": {
                **qualification,
                "receipt_sha256": receipt_sha256,
            },
            "target_status": "approved",
            "target_promotion_stage": "paper",
            "target_horizon_profile": "short_1_5d",
            "target_evidence_mode": "consumed_historical_replay",
            "target_config_json": {
                "recipe_id": "short_relative_strength",
                "recipe_version": (
                    FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
                ),
                "horizon_profile": "short_1_5d",
                "evidence_mode": "consumed_historical_replay",
            },
        }
    ]


def _healthy_short_paper_candidate(
    *,
    strategy_version_id: str = "v18-short-paper",
) -> dict:
    return {
        "strategy_version_id": strategy_version_id,
        "promotion_stage": "paper",
        "signal_frequency": "day",
        "execution_frequency": "day",
        "contract_ready": True,
        "health_status": "healthy",
        "health_evidence_ready": True,
        "health_evidence_reasons": [],
        "paper_stage_status": "active",
        "simulation_status": "active",
        "active_recommendation_portfolios": 0,
    }


@pytest.mark.no_database
def test_rehabilitation_receipt_projects_only_source_revalidated_cash_sleeves() -> None:
    rows = _rehabilitation_cash_only_rows()
    assert rows[0]["target_config_json"]["recipe_version"] == (
        FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
    )
    assert get_strategy_recipe("short_relative_strength")["version"] != (
        FORWARD_ONLY_REHABILITATION_TARGET_RECIPE_VERSION
    )
    assert LOCKBOX_CONFIG_KEY not in rows[0]["target_config_json"]
    short = _healthy_short_paper_candidate()
    short_lane = readiness_module._assess_horizon_candidates(
        "short_1_5d", [short]
    )

    cash_only_lanes = (
        readiness_module._validated_rehabilitation_cash_only_horizon_lanes(
            rows,
            short_lane=short_lane,
        )
    )
    result = readiness_module._project_three_horizon_production(
        {
            "short_1_5d": [short],
            "swing_1_6m": [],
            "long_1_3y": [],
        },
        cash_only_lanes,
    )

    assert result["status"] == "ok"
    assert result["cash_only_horizons"] == ["swing_1_6m", "long_1_3y"]
    for horizon in result["cash_only_horizons"]:
        lane = result["horizons"][horizon]
        assert "lockbox_evidence" not in lane
        assert lane["recommendation_eligible"] is False
        assert lane["rehabilitation_evidence"]["authority"] == (
            "historical_description_only"
        )
        assert lane["rehabilitation_evidence"]["sealed_final_oos"] is False
        assert lane["rehabilitation_evidence"]["unseen_oos"] is False


@pytest.mark.no_database
def test_rehabilitation_cash_projection_requires_exact_operable_target_receipt() -> None:
    rows = _rehabilitation_cash_only_rows()
    short = _healthy_short_paper_candidate()
    short_lane = readiness_module._assess_horizon_candidates(
        "short_1_5d", [short]
    )

    tampered = deepcopy(rows)
    tampered[0]["source_unavailable_horizons_sha256"] = "5" * 64
    assert (
        readiness_module._validated_rehabilitation_cash_only_horizon_lanes(
            tampered,
            short_lane=short_lane,
        )
        == {}
    )
    assert (
        readiness_module._validated_rehabilitation_cash_only_horizon_lanes(
            rows,
            short_lane={**short_lane, "strategy_version_id": "another-short"},
        )
        == {}
    )
    opened = deepcopy(rows)
    qualification = opened[0]["qualification_json"]
    qualification["sealed_final_oos"] = True
    core = {key: value for key, value in qualification.items() if key != "receipt_sha256"}
    opened[0]["receipt_sha256"] = readiness_module.canonical_sha256(core)
    qualification["receipt_sha256"] = opened[0]["receipt_sha256"]
    assert (
        readiness_module._validated_rehabilitation_cash_only_horizon_lanes(
            opened,
            short_lane=short_lane,
        )
        == {}
    )

    stopped = readiness_module._assess_horizon_candidates(
        "short_1_5d", [{**short, "simulation_status": "stopped"}]
    )
    assert (
        readiness_module._validated_rehabilitation_cash_only_horizon_lanes(
            rows,
            short_lane=stopped,
        )
        == {}
    )


@pytest.mark.no_database
@pytest.mark.parametrize("missing_marker", ["final_oos_opened", "capital_eligible"])
def test_rehabilitation_cash_projection_requires_complete_receipt_markers(
    missing_marker: str,
) -> None:
    rows = _rehabilitation_cash_only_rows()
    qualification = rows[0]["qualification_json"]
    del qualification[missing_marker]
    core = {key: value for key, value in qualification.items() if key != "receipt_sha256"}
    rows[0]["receipt_sha256"] = readiness_module.canonical_sha256(core)
    qualification["receipt_sha256"] = rows[0]["receipt_sha256"]

    short = _healthy_short_paper_candidate()
    short_lane = readiness_module._assess_horizon_candidates(
        "short_1_5d", [short]
    )
    cash_only_lanes = (
        readiness_module._validated_rehabilitation_cash_only_horizon_lanes(
            rows,
            short_lane=short_lane,
        )
    )
    result = readiness_module._project_three_horizon_production(
        {
            "short_1_5d": [short],
            "swing_1_6m": [],
            "long_1_3y": [],
        },
        cash_only_lanes,
    )

    assert cash_only_lanes == {}
    assert result["status"] == "blocked"
    assert result["cash_only_horizons"] == []
    assert result["blocked_horizons"] == ["swing_1_6m", "long_1_3y"]


def _settings(monkeypatch, database_url: str, data_root: Path, *, auth_mode: str) -> Settings:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("AUTH_MODE", auth_mode)
    monkeypatch.setenv("TUSHARE_API_URL", "https://api.tushare.pro")
    monkeypatch.setenv("TUSHARE_TOKEN", "verified-test-token")
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("BROKER_MODE", "disabled")
    monkeypatch.setenv("PLATFORM_SECRET_KEY", Fernet.generate_key().decode("ascii"))
    return Settings.from_env(PROJECT_ROOT / ".env.missing")


def _qlib_dataset(
    data_root: Path, *, trading_days: int = RESEARCH_MINIMUM_TRADING_DAYS
) -> None:
    target = data_root / "qlib" / "acceptance-snapshot"
    (target / "calendars").mkdir(parents=True)
    (target / "instruments").mkdir()
    (target / "features").mkdir()
    (target / "metadata").mkdir()
    start = date(2026, 7, 31) - timedelta(days=trading_days - 1)
    days = [(start + timedelta(days=index)).isoformat() for index in range(trading_days)]
    (target / "calendars" / "day.txt").write_text("\n".join(days) + "\n", encoding="utf-8")
    (target / "instruments" / "cn_all.txt").write_text(
        f"SH600000\t{days[0]}\t{days[-1]}\n", encoding="utf-8"
    )
    (target / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "frequency": "day",
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
                "dataset_lineage_id": "c" * 64,
                "source_lineage_id": "d" * 64,
                "lineage_verified": True,
                "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
                "source_volume_unit": "hand",
                "qlib_volume_unit": "share",
                "source_amount_unit": "thousand_cny",
                "qlib_amount_unit": "cny",
                "source_hand_size": 100,
                "index_volume_policy": "excluded_non_tradable_benchmark",
                "governed_etf_whitelist": governed_etf_ready_evidence(),
                "execution_controls": {
                    "formal_execution_requires_native_controls": True,
                    "scope_version": GOVERNED_DAILY_STOCK_SCOPE_VERSION,
                    "native_complete_from": days[0],
                },
                "output_manifest": build_qlib_output_manifest(target),
            }
        ),
        encoding="utf-8",
    )
    refresh_qlib_display_catalog(data_root)


def test_empty_deployment_is_fail_closed(tmp_path: Path, monkeypatch, database_url: str) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()

    result = DeploymentReadinessStore(settings, PROJECT_ROOT).assess()

    assert result["highest_ready_profile"] is None
    assert result["live_trading_supported"] is False
    research = result["profiles"][0]
    assert research["status"] == "blocked"
    blocked = {item["id"] for item in research["checks"] if item["status"] == "block"}
    assert {
        "authentication_enabled",
        "tushare_verified",
        "initialization_pipeline",
        "reproducible_qlib_dataset",
        "operational_health",
        "rdagent_runtime",
        "incremental_schedule",
    }.issubset(blocked)


def test_research_dataset_threshold_matches_multi_profile_contract() -> None:
    assert (
        DEFAULT_RESEARCH_PERIOD_POLICY["embargo_trading_days"]
        == CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS
    )
    assert RESEARCH_MINIMUM_TRADING_DAYS == (
        MINIMUM_PROFILE_TRAINING_DAYS
        + MODEL_LABEL_HORIZON_TRADING_DAYS
        + max(
            int(profile["validation_trading_days"])
            for profile in RESEARCH_EVALUATION_PROFILES
        )
        + CAPITAL_OOS_MIN_EMBARGO_TRADING_DAYS
        + DEFAULT_RESEARCH_PERIOD_POLICY["test_trading_days"]
    )


def _data_schedule_check(settings: Settings) -> dict:
    result = DeploymentReadinessStore(settings, PROJECT_ROOT).assess()
    return next(
        item
        for item in result["profiles"][0]["checks"]
        if item["id"] == "incremental_schedule"
    )


def _create_data_pipeline_schedule(
    database_url: str,
    *,
    name: str,
    payload: dict,
) -> None:
    ScheduleStore(database_url).create(
        name=name,
        kind="data_pipeline",
        timezone="Asia/Shanghai",
        run_time=time(18, 0),
        trading_days_only=True,
        payload=payload,
        misfire_grace_seconds=3600,
        actor="admin",
    )


def _create_incremental_schedule(
    database_url: str,
    *,
    name: str,
    payload: dict,
) -> None:
    ScheduleStore(database_url).create(
        name=name,
        kind="incremental_sync",
        timezone="Asia/Shanghai",
        run_time=time(18, 0),
        trading_days_only=True,
        payload=payload,
        misfire_grace_seconds=3600,
        actor="admin",
    )


def _create_governed_schedule_suite(
    database_url: str,
    *,
    minute_run_time: time = time(23, 30),
    short_final_oos: bool = False,
    include_npr: bool = False,
) -> None:
    evaluation = {
        "dataset": "acceptance-snapshot",
        "periods": {
            "train_start": "2019-01-01",
            "train_end": "2019-12-31",
            "valid_start": "2020-01-01",
            "valid_end": "2022-12-31",
            "test_start": "2026-06-01" if short_final_oos else "2023-01-09",
            "test_end": "2026-07-31",
        },
        "universe": "cn_all",
        "benchmark": "SH000300",
    }
    contracts = (
        (
            "governed daily raw data and Qlib publication",
            "data_pipeline",
            time(18, 0),
            True,
            {
                "profile": "full",
                "snapshot_start": "2008-01-01",
                "lookback_days": 7,
                "bundles": list(AUTOMATED_DATA_BUNDLES),
            },
        ),
        (
            "governed daily A-share five-minute publication",
            "ashare_5m_sync",
            minute_run_time,
            True,
            {"history_start": "2024-01-01", "lookback_days": 3},
        ),
        (
            "governed daily bounded information NLP",
            "information_pipeline",
            time(2, 0),
            False,
            {
                "lookback_days": 7,
                "regulatory_only": True,
                "download_limit": 0,
                "enable_nlp": True,
                "announcement_categories": ["regulatory_letter"],
                "announcement_nlp_limit": 500,
                "include_corpus_nlp": True,
                "corpus_datasets": [
                    "cctv_news",
                    "irm_qa_sh",
                    "irm_qa_sz",
                    "major_news",
                    *(["npr"] if include_npr else []),
                ],
                "corpus_nlp_limit": 500,
                "batch_size": 50,
                "major_news_per_day": 40,
                "irm_per_instrument_day": 2,
                "include_event_labels": True,
                "include_factor_evaluation": False,
                "horizons": [1, 3, 5, 20],
                "benchmark_code": "000300.SH",
            },
        ),
        (
            "governed weekly structured information factors",
            "information_factor_refresh",
            time(12, 30),
            False,
            {
                "sources": ["major_news_mentions", "news_flash", "report_rc"],
                "weekday": 4,
                "factor_evaluation": evaluation,
            },
        ),
        (
            "governed daily auxiliary research data publication",
            "auxiliary_data_pipeline",
            time(4, 0),
            False,
            {
                "history_start": "2024-01-01",
                "max_stocks": 100,
                "max_options": 100,
                "strategy_minute_symbols": list(DEFAULT_STRATEGY_MINUTE_SYMBOLS),
            },
        ),
    )
    for name, kind, run_time, trading_days_only, payload in contracts:
        ScheduleStore(database_url).create(
            name=name,
            kind=kind,
            timezone="Asia/Shanghai",
            run_time=run_time,
            trading_days_only=trading_days_only,
            payload=payload,
            misfire_grace_seconds=7200,
            actor="admin",
        )


def test_readiness_rejects_reproducible_dataset_below_research_history_contract(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root, trading_days=RESEARCH_MINIMUM_TRADING_DAYS - 1)

    result = DeploymentReadinessStore(settings, PROJECT_ROOT).assess()

    check = next(
        item
        for item in result["profiles"][0]["checks"]
        if item["id"] == "reproducible_qlib_dataset"
    )
    assert check["status"] == "block"
    assert check["details"]["minimum_trading_days"] == RESEARCH_MINIMUM_TRADING_DAYS
    assert check["details"]["maximum_available_trading_days"] == (
        RESEARCH_MINIMUM_TRADING_DAYS - 1
    )


def test_readiness_requires_runtime_secrets_to_be_decryptable(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="required")
    RuntimeSecretStore(database_url, settings.platform_secret_key).put(
        "tushare",
        {"api_url": "https://api.tushare.pro", "token": "database-token"},
        metadata={
            "api_url": "https://api.tushare.pro",
            "verified_at": datetime.now(UTC).isoformat(),
        },
        updated_by=None,
    )
    wrong_key = replace(settings, platform_secret_key=Fernet.generate_key().decode("ascii"))
    result = DeploymentReadinessStore(wrong_key, PROJECT_ROOT).assess()
    checks = {item["id"]: item for item in result["profiles"][0]["checks"]}
    assert checks["runtime_secret_storage"]["status"] == "block"
    assert checks["tushare_verified"]["status"] == "block"
    assert "无法解密" in checks["tushare_verified"]["evidence"]


def test_readiness_accepts_one_governed_full_data_pipeline(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _create_data_pipeline_schedule(
        database_url,
        name="governed full daily pipeline",
        payload={
            "profile": "full",
            "snapshot_start": "2008-01-01",
            "lookback_days": 30,
            "bundles": list(AUTOMATED_DATA_BUNDLES),
        },
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "pass"
    assert len(check["details"]["governed_data_pipeline_ids"]) == 1
    assert check["details"]["incremental_sync_ids"] == []


def test_readiness_accepts_exact_governed_five_schedule_suite(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    _create_governed_schedule_suite(database_url)

    check = _data_schedule_check(settings)

    assert check["status"] == "pass"
    assert check["details"]["mode"] == "governed_suite_v1"
    assert set(check["details"]["governed_suite_ids"]) == {
        "data_pipeline",
        "information_pipeline",
        "information_factor_refresh",
        "ashare_5m_sync",
        "auxiliary_data_pipeline",
    }
    assert all(
        len(ids) == 1 for ids in check["details"]["governed_suite_ids"].values()
    )
    assert check["details"]["rejected_suite_ids"] == []


def test_readiness_rejects_unreviewed_npr_in_governed_schedule_suite(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    _create_governed_schedule_suite(database_url, include_npr=True)

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert check["details"]["mode"] == "invalid"
    assert check["details"]["governed_suite_ids"]["information_pipeline"] == []
    assert len(check["details"]["rejected_suite_ids"]) == 1


def test_readiness_rejects_schedule_suite_time_drift(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    _create_governed_schedule_suite(
        database_url, minute_run_time=time(15, 0)
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert check["details"]["mode"] == "invalid"
    assert len(check["details"]["governed_suite_ids"]["ashare_5m_sync"]) == 0
    assert len(check["details"]["rejected_suite_ids"]) == 1


def test_readiness_rejects_suite_whose_fixed_evaluation_oos_is_too_short(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    _create_governed_schedule_suite(
        database_url, short_final_oos=True
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert check["details"]["mode"] == "invalid"
    assert check["details"]["governed_suite_ids"][
        "information_factor_refresh"
    ] == []


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "profile": "research-assets",
                "snapshot_start": "2008-01-01",
                "bundles": ["research_corpus"],
            },
            id="research-assets",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2008-01-01",
                "bundles": list(AUTOMATED_DATA_BUNDLES[:-1]),
            },
            id="missing-bundle",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2018-01-01",
                "bundles": list(AUTOMATED_DATA_BUNDLES),
            },
            id="non-2008-lineage",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2008-01-01",
                "lookback_days": "invalid",
                "bundles": list(AUTOMATED_DATA_BUNDLES),
            },
            id="invalid-lookback",
        ),
    ],
)
def test_readiness_rejects_ungoverned_active_data_pipelines(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
    payload: dict,
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _create_data_pipeline_schedule(
        database_url,
        name="ungoverned active pipeline",
        payload=payload,
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert check["details"]["governed_data_pipeline_ids"] == []
    assert len(check["details"]["rejected_data_pipeline_ids"]) == 1


def test_readiness_rejects_multiple_active_data_refresh_schedules(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _create_data_pipeline_schedule(
        database_url,
        name="governed full daily pipeline",
        payload={
            "profile": "full",
            "snapshot_start": "2008-01-01",
            "bundles": list(AUTOMATED_DATA_BUNDLES),
        },
    )
    _create_incremental_schedule(
        database_url,
        name="duplicate incremental sync",
        payload={
            "profile": "full",
            "snapshot_start": "2008-01-01",
            "lookback_days": 7,
            "build_qlib": True,
        },
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert len(check["details"]["active_schedule_ids"]) == 2


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "profile": "core",
                "snapshot_start": "2008-01-01",
                "lookback_days": 7,
                "build_qlib": True,
            },
            id="non-full-profile",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2018-01-01",
                "lookback_days": 7,
                "build_qlib": True,
            },
            id="non-2008-lineage",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2008-01-01",
                "lookback_days": 7,
                "build_qlib": False,
            },
            id="download-only",
        ),
        pytest.param(
            {
                "profile": "full",
                "snapshot_start": "2008-01-01",
                "lookback_days": "invalid",
                "build_qlib": True,
            },
            id="invalid-lookback",
        ),
        pytest.param({}, id="missing-contract"),
    ],
)
def test_readiness_rejects_ungoverned_active_incremental_sync(
    tmp_path: Path,
    monkeypatch,
    database_url: str,
    payload: dict,
) -> None:
    settings = _settings(monkeypatch, database_url, tmp_path / "data", auth_mode="disabled")
    DataTaskStore(database_url).sync_catalog()
    _create_incremental_schedule(
        database_url,
        name="ungoverned incremental sync",
        payload=payload,
    )

    check = _data_schedule_check(settings)

    assert check["status"] == "block"
    assert check["details"]["incremental_sync_ids"] == []
    assert len(check["details"]["rejected_incremental_sync_ids"]) == 1


def test_research_readiness_requires_complete_evidence_chain(
    tmp_path: Path, monkeypatch, database_url: str
) -> None:
    data_root = tmp_path / "data"
    settings = _settings(monkeypatch, database_url, data_root, auth_mode="required")
    AuthStore(database_url).bootstrap_admin(
        username="admin",
        display_name="Administrator",
        password="Secure-Admin-123!",
    )
    jobs = JobStore(database_url)
    for kind in ("bootstrap", "data_verify", "data_snapshot", "data_qlib", "qlib_baseline"):
        job = jobs.create(kind, {"fixture": True}, tmp_path / f"{kind}.log")
        jobs.finish(job["id"], exit_code=0, result={"accepted": True})
    DataTaskStore(database_url).sync_catalog()
    _qlib_dataset(data_root)
    ScheduleStore(database_url).create(
        name="daily data sync",
        kind="incremental_sync",
        timezone="Asia/Shanghai",
        run_time=time(18, 0),
        trading_days_only=True,
        payload={
            "profile": "full",
            "snapshot_start": "2008-01-01",
            "lookback_days": 7,
            "build_qlib": True,
        },
        misfire_grace_seconds=3600,
        actor="admin",
    )
    current = datetime.now(UTC)
    OperationalHealthStore(settings).record(
        {
            "status": "ok",
            "components": {
                "postgresql": {"status": "ok", "message": "ready"},
                "rdagent_runtime": {"status": "ok", "message": "ready"},
            },
            "summary": {
                "component_count": 2,
                "ok_count": 2,
                "problem_count": 0,
                "bootstrap_count": 0,
            },
            "recorded_at": current,
        }
    )

    result = DeploymentReadinessStore(settings, PROJECT_ROOT).assess(now=current)

    research, recommendation, allocation, pair = result["profiles"]
    assert result["highest_ready_profile"] == "research"
    assert research["status"] == "ready"
    assert research["passed"] == research["total"]
    assert (
        next(
            item
            for item in research["checks"]
            if item["id"] == "incremental_schedule"
        )["status"]
        == "pass"
    )
    assert pair["status"] == "blocked"
    assert (
        next(item for item in pair["checks"] if item["id"] == "pair_minute_data")["status"]
        == "block"
    )
    assert recommendation["status"] == "blocked"
    assert (
        next(item for item in recommendation["checks"] if item["id"] == "simulation_accounts")[
            "status"
        ]
        == "block"
    )
    assert (
        next(
            item
            for item in recommendation["checks"]
            if item["id"] == "simulation_60_day_replay"
        )["status"]
        == "block"
    )
    assert (
        next(
            item
            for item in recommendation["checks"]
            if item["id"] == "unsupported_schedules_retired"
        )["status"]
        == "pass"
    )
    assert (
        next(item for item in research["checks"] if item["id"] == "schema_current")["status"]
        == "pass"
    )
