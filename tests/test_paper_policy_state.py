from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest

from quant_platform.paper_policy_state import (
    PAPER_POLICY_STATE_VERSION,
    bind_current_paper_holdings,
    previous_snapshot_from_paper_batch,
    seal_paper_policy_state,
    validate_paper_policy_state,
)
from quant_platform.portfolio_policy import is_rebalance_due
from scripts import run_recommendation_refresh


@pytest.mark.no_database
def test_sealed_paper_state_projects_the_previous_snapshot_for_weekly_cadence() -> None:
    state = seal_paper_policy_state(
        {
            "take_profit_stages": {"SH600000": 1},
            "execution": {},
            "holding_age_sessions": {"SH600000": 12},
        }
    )
    snapshot = previous_snapshot_from_paper_batch(
        {
            "id": "batch-1",
            "portfolio_id": "paper-short",
            "status": "succeeded",
            "signal_date": date(2026, 8, 27),
            "trade_date": date(2026, 8, 28),
            "target_payload": {
                "governed_order_plan": {"promotion_stage_id": "stage-1"},
                "paper_policy_state": state,
            },
        },
        expected_portfolio_id="paper-short",
        expected_promotion_stage_id="stage-1",
    )

    assert snapshot["position_state"]["holding_age_sessions"] == {
        "SH600000": 12
    }
    assert snapshot["paper_policy_state_contract_version"] == (
        PAPER_POLICY_STATE_VERSION
    )
    assert is_rebalance_due("2026-08-28", snapshot["as_of_date"], "week") is False
    assert is_rebalance_due("2026-08-31", snapshot["as_of_date"], "week") is True


@pytest.mark.no_database
def test_policy_state_rejects_nonfinite_or_tampered_content() -> None:
    with pytest.raises(ValueError, match="finite canonical JSON"):
        seal_paper_policy_state({"bad": math.nan})

    state = seal_paper_policy_state({"holding_age_sessions": {"SH600000": 1}})
    state["position_state"]["holding_age_sessions"]["SH600000"] = 2
    with pytest.raises(ValueError, match="seal is invalid"):
        validate_paper_policy_state(state)


@pytest.mark.no_database
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "queued", "succeeded batch"),
        ("portfolio_id", "other", "another account"),
        ("promotion_stage_id", "other-stage", "active stage"),
    ],
)
def test_previous_snapshot_fails_closed_on_wrong_batch_authority(
    field: str, value: str, message: str
) -> None:
    row = {
        "id": "batch-1",
        "portfolio_id": "paper-short",
        "status": "succeeded",
        "signal_date": "2026-08-27",
        "trade_date": "2026-08-28",
        "target_payload": {
            "governed_order_plan": {"promotion_stage_id": "stage-1"},
            "paper_policy_state": seal_paper_policy_state({}),
        },
    }
    if field == "promotion_stage_id":
        row["target_payload"]["governed_order_plan"][field] = value
    else:
        row[field] = value
    with pytest.raises(ValueError, match=message):
        previous_snapshot_from_paper_batch(
            row,
            expected_portfolio_id="paper-short",
            expected_promotion_stage_id="stage-1",
        )


@pytest.mark.no_database
def test_current_ledger_holdings_replace_stale_policy_ages_and_closed_names() -> None:
    snapshot = {
        "as_of_date": "2026-08-27",
        "effective_date": "2026-08-28",
        "position_state": {
            "take_profit_stages": {"SH600000": 1, "SZ000001": 2},
            "holding_age_sessions": {"SH600000": 4, "SZ000001": 8},
            "execution": {},
        },
    }
    bound = bind_current_paper_holdings(
        snapshot,
        [
            {
                "instrument": "sh600000",
                "weight": 0.2,
                "average_cost": 10.5,
                "holding_age_sessions": 5,
            }
        ],
    )

    assert bound is not None
    assert bound["position_state"]["holding_age_sessions"] == {"SH600000": 5}
    assert bound["position_state"]["take_profit_stages"] == {"SH600000": 1}
    assert bound["holdings"][0]["instrument"] == "SH600000"
    assert len(bound["position_state_sha256"]) == 64
    assert len(bound["current_holdings_sha256"]) == 64


@pytest.mark.no_database
def test_current_paper_holdings_fail_closed_without_proven_age() -> None:
    with pytest.raises(ValueError, match="proven non-negative session ages"):
        bind_current_paper_holdings(
            {"position_state": {}},
            [
                {
                    "instrument": "SH600000",
                    "weight": 0.2,
                    "average_cost": 10.5,
                    "holding_age_sessions": None,
                }
            ],
        )


@pytest.mark.no_database
def test_paper_policy_state_is_wired_through_artifact_batch_and_worker() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_recommendation_refresh.py").read_text(
        encoding="utf-8"
    )
    store = (root / "src" / "quant_platform" / "simulation_store.py").read_text(
        encoding="utf-8"
    )
    worker = (root / "src" / "quant_platform" / "worker.py").read_text(
        encoding="utf-8"
    )

    assert '"paper_policy_state": seal_paper_policy_state' in script
    assert '_canonical_bytes({"target_weights": target_weights})' in script
    assert "validate_paper_policy_state(raw_policy_state)" in store
    assert "def latest_paper_previous_snapshot(" in store
    assert "simulation_batches.c.status == \"succeeded\"" in store
    assert "simulation_batches.c.signal_date < before_signal_date" in store
    assert "self.simulations.latest_paper_previous_snapshot(" in worker
    assert "bind_current_paper_holdings(" in worker
    assert '"previous_snapshot": previous_snapshot' in worker


@pytest.mark.no_database
def test_order_plan_seals_policy_state_without_changing_weight_identity(
    tmp_path: Path, monkeypatch
) -> None:
    class Workflow:
        def identity_dict(self) -> dict[str, str]:
            return {"run_id": "paper-state-test", "provider": "test"}

        def log_params(self, _value) -> None:
            return None

        def log_metrics(self, _value) -> None:
            return None

        def save_artifacts(self, _value) -> None:
            return None

    @contextmanager
    def workflow_run(**_kwargs):
        yield Workflow()

    monkeypatch.setattr(
        run_recommendation_refresh,
        "qlib_workflow_run",
        workflow_run,
    )
    output = run_recommendation_refresh._write_qlib_order_plan(
        manifest={
            "order_plan_job_id": "job-1",
            "simulation_portfolio_id": "portfolio-1",
            "strategy_version_id": "version-1",
            "formal_backtest_id": "backtest-1",
            "promotion_stage_id": "stage-1",
            "promotion_stage_opened_at": "2026-08-20T00:00:00+00:00",
            "config": {"execution_contract_hash": "e" * 64},
            "dataset": "daily-1",
            "signal_date": "2026-08-27",
        },
        result={
            "as_of_date": "2026-08-27",
            "effective_date": "2026-08-28",
            "holdings": [{"instrument": "sh600000", "weight": 0.2}],
            "position_state": {
                "take_profit_stages": {"SH600000": 1},
                "holding_age_sessions": {"SH600000": 5},
                "execution": {},
            },
        },
        dataset_provenance={
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "b" * 64,
        },
        order_plan_root=tmp_path / "plans",
        tracking_uri="memory://paper-state-test",
    )

    artifact = Path(output["order_plan_artifact_path"])
    target_bytes = (artifact / "target_weights.json").read_bytes()
    target = json.loads(target_bytes)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    weight_bytes = json.dumps(
        {"target_weights": {"SH600000": 0.2}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert validate_paper_policy_state(target["paper_policy_state"])
    assert manifest["target_weights_file_sha256"] == hashlib.sha256(
        target_bytes
    ).hexdigest()
    assert manifest["target_weights_sha256"] == hashlib.sha256(weight_bytes).hexdigest()
    assert manifest["target_weights_sha256"] != manifest["target_weights_file_sha256"]
