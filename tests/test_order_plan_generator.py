from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from qlib_test_doubles import qlib_workflow_identity

from quant_platform.investor_profile import bind_investor_profile

pytestmark = pytest.mark.no_database


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "run_recommendation_refresh.py"
    spec = importlib.util.spec_from_file_location(
        "run_recommendation_refresh_order_plan", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile_binding() -> dict:
    return bind_investor_profile(
        {
            "id": "profile-1",
            "profile_key": "primary",
            "version": 1,
            "content_sha256": "a" * 64,
            "market_permissions": {
                "main_board": True,
                "star_market": True,
                "chi_next": True,
                "beijing_exchange": False,
                "etf": True,
            },
        }
    )


def test_paper_permission_overlay_blocks_new_bse_risk_but_keeps_exit() -> None:
    script = _script_module()
    projection = pd.DataFrame(
        {
            "risk_state": ["normal", "normal", "exit"],
            "allow_new_risk": [True, True, False],
            "risk_reasons": ["[]", "[]", '["hard_risk"]'],
        },
        index=["600000.SH", "830001.BJ", "920001.BJ"],
    )

    governed, evidence = script._apply_investor_profile_permissions(
        projection,
        _profile_binding(),
        on_date=date(2026, 8, 31),
    )

    assert governed.loc["600000.SH", "risk_state"] == "normal"
    assert governed.loc["830001.BJ", "risk_state"] == "restricted"
    assert governed.loc["920001.BJ", "risk_state"] == "exit"
    assert bool(governed.loc["830001.BJ", "allow_new_risk"]) is False
    assert evidence["830001.BJ"] == {
        "allowed": False,
        "permission_key": "beijing_exchange",
        "reason": "investor_permission_disabled:beijing_exchange",
    }


def test_production_order_plan_generator_records_and_hashes_qlib_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script_module()
    saved: list[Path] = []

    class Workflow:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def identity_dict():
            return qlib_workflow_identity()

        @staticmethod
        def log_params(_values):
            return None

        @staticmethod
        def log_metrics(_values):
            return None

        @staticmethod
        def save_artifacts(path):
            saved.append(Path(path))

    monkeypatch.setattr(script, "qlib_workflow_run", lambda **_kwargs: Workflow())
    result = script._write_qlib_order_plan(
        manifest={
            "order_plan_job_id": "job-1",
            "simulation_portfolio_id": "simulation-1",
            "strategy_version_id": "version-1",
            "formal_backtest_id": "backtest-1",
            "promotion_stage_id": "stage-1",
            "promotion_stage_opened_at": "2026-07-01T00:00:00+00:00",
            "dataset": "snapshot",
            "signal_date": "2026-07-10",
            "signal_at": None,
            "execution_not_before": None,
            "config": {"execution_contract_hash": "a" * 64},
        },
        result={
            "status": "ok",
            "as_of_date": "2026-07-10",
            "effective_date": "2026-07-13",
            "holdings": [
                {"instrument": "SH600001", "weight": 0.40},
                {"instrument": "SH600000", "weight": 0.50},
            ],
            "position_state": {
                "take_profit_stages": {"SH600001": 1},
                "holding_age_sessions": {"SH600000": 3, "SH600001": 2},
                "execution": {},
            },
        },
        dataset_provenance={
            "dataset_identity_sha256": "b" * 64,
            "dataset_lineage_id": "c" * 64,
        },
        order_plan_root=tmp_path / "order-plans",
        tracking_uri="sqlite:///tracking.db",
    )

    digest = result["order_plan_manifest_sha256"]
    artifact = tmp_path / "order-plans" / digest
    manifest_path = artifact / "manifest.json"
    target_path = artifact / "target_weights.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == digest
    target_payload = json.loads(target_path.read_text(encoding="utf-8"))
    assert target_payload["target_weights"] == {
        "SH600000": 0.50,
        "SH600001": 0.40,
    }
    assert target_payload["paper_policy_state"]["position_state"] == {
        "execution": {},
        "holding_age_sessions": {"SH600000": 3, "SH600001": 2},
        "take_profit_stages": {"SH600001": 1},
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["produced_by"] == "qlib-workflow-recorder"
    assert manifest["source_snapshot"]["id"] == "b" * 64
    assert manifest["qlib_workflow"] == qlib_workflow_identity()
    assert saved == [artifact]


def test_production_order_plan_generator_seals_explicit_empty_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script_module()

    class Workflow:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def identity_dict():
            return qlib_workflow_identity()

        @staticmethod
        def log_params(_values):
            return None

        @staticmethod
        def log_metrics(_values):
            return None

        @staticmethod
        def save_artifacts(_path):
            return None

    monkeypatch.setattr(script, "qlib_workflow_run", lambda **_kwargs: Workflow())
    result = script._write_qlib_order_plan(
        manifest={
            "order_plan_job_id": "job-empty",
            "simulation_portfolio_id": "simulation-1",
            "strategy_version_id": "version-1",
            "formal_backtest_id": "backtest-1",
            "promotion_stage_id": "stage-1",
            "promotion_stage_opened_at": "2026-08-01T00:00:00+00:00",
            "dataset": "snapshot",
            "signal_date": "2026-08-27",
            "signal_at": None,
            "execution_not_before": None,
            "config": {"execution_contract_hash": "a" * 64},
        },
        result={
            "status": "ok",
            "as_of_date": "2026-08-27",
            "effective_date": "2026-08-28",
            "holdings": [],
            "position_state": {
                "take_profit_stages": {},
                "holding_age_sessions": {},
                "execution": {},
            },
        },
        dataset_provenance={
            "dataset_identity_sha256": "b" * 64,
            "dataset_lineage_id": "c" * 64,
        },
        order_plan_root=tmp_path / "order-plans",
        tracking_uri="sqlite:///tracking.db",
    )

    artifact = tmp_path / "order-plans" / result["order_plan_manifest_sha256"]
    target = json.loads((artifact / "target_weights.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    expected_weights = json.dumps(
        {"target_weights": {}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert target["target_weights"] == {}
    assert manifest["target_weights_sha256"] == hashlib.sha256(
        expected_weights
    ).hexdigest()
