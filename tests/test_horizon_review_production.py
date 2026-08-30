from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlalchemy import select, update
from test_promotion_chain import _gated_paper_version, _seed_evidence

from quant_data.database import (
    simulation_batches,
    simulation_portfolios,
    strategy_promotion_stages,
)
from quant_platform.horizon_review import (
    build_financial_review_scope,
    resolve_financial_review_trigger,
    validate_financial_review_scope,
)
from quant_platform.promotion import (
    PromotionStore,
    build_horizon_review_evidence,
    validate_horizon_review_evidence,
)
from quant_platform.research_horizon import LONG_1_3Y, SWING_1_6M
from scripts import run_recommendation_refresh


def _completed_at(value: date) -> datetime:
    return datetime.combine(
        value, time(15), tzinfo=ZoneInfo("Asia/Shanghai")
    ).astimezone(UTC)


def _snapshot_with_financials(root: Path) -> dict[str, str]:
    snapshot = root / "snapshots" / "snapshot-1"
    (snapshot / "parquet" / "fina_indicator").mkdir(parents=True)
    pd.DataFrame(
        [
            {"ts_code": "600000.SH", "ann_date": "20260826", "end_date": "20260331"},
            {"ts_code": "600001.SH", "ann_date": "20260828", "end_date": "20260630"},
            {"ts_code": "000001.SZ", "ann_date": "20260828", "end_date": "20260630"},
            # An older-period restatement that became PIT-effective in the
            # same interval must remain part of the governed review batch.
            {"ts_code": "600002.SH", "ann_date": "20260829", "end_date": "20251231"},
        ]
    ).to_parquet(snapshot / "parquet" / "fina_indicator" / "part.parquet")
    manifest = snapshot / "manifest.json"
    manifest.write_text('{"snapshot":"snapshot-1"}', encoding="utf-8")
    return {
        "snapshot_name": "snapshot-1",
        "snapshot_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "dataset_identity_sha256": "a" * 64,
    }


class _Workflow:
    def identity_dict(self) -> dict[str, str]:
        return {"run_id": "review-test", "provider": "test"}

    def log_params(self, _value) -> None:
        return None

    def log_metrics(self, _value) -> None:
        return None

    def save_artifacts(self, _value) -> None:
        return None


@contextmanager
def _workflow_run(**_kwargs):
    yield _Workflow()


def _stub_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_recommendation_refresh, "qlib_workflow_run", _workflow_run
    )


@pytest.mark.no_database
def test_financial_trigger_uses_strict_pit_interval_and_does_not_repeat_daily(
    tmp_path: Path,
) -> None:
    provenance = _snapshot_with_financials(tmp_path)

    trigger = resolve_financial_review_trigger(
        data_root=tmp_path,
        dataset_provenance=provenance,
        previous_signal_date=date(2026, 8, 27),
        signal_date=date(2026, 8, 31),
    )

    assert trigger is not None
    assert trigger["report_period"] == "2026Q2"
    assert trigger["report_periods"] == ["2025Q4", "2026Q2"]
    assert trigger["announcement_date"] == "2026-08-29"
    assert trigger["trigger_effective_date"] == "2026-08-31"
    assert trigger["source_event_count"] == 3
    assert trigger["affected_instrument_count"] == 3
    assert trigger["affected_instruments"] == [
        "SH600001",
        "SH600002",
        "SZ000001",
    ]
    assert resolve_financial_review_trigger(
        data_root=tmp_path,
        dataset_provenance=provenance,
        previous_signal_date=date(2026, 8, 31),
        signal_date=date(2026, 9, 1),
    ) is None


@pytest.mark.no_database
def test_financial_review_scope_only_opens_for_holdings_or_qualified_candidates(
    tmp_path: Path,
) -> None:
    provenance = _snapshot_with_financials(tmp_path)
    trigger = resolve_financial_review_trigger(
        data_root=tmp_path,
        dataset_provenance=provenance,
        previous_signal_date=date(2026, 8, 27),
        signal_date=date(2026, 8, 31),
    )
    assert trigger is not None

    governance_only = build_financial_review_scope(
        trigger,
        current_holdings=["SH600000"],
        qualified_candidates=["SZ000002"],
    )
    assert governance_only["review_mode"] == "governance_only"
    assert governance_only["reviewed_instruments"] == []
    assert governance_only["ignored_instruments"] == [
        "SH600001",
        "SH600002",
        "SZ000001",
    ]
    assert not run_recommendation_refresh._rebalance_due_for_financial_review(
        scheduled_rebalance_due=False,
        financial_review_scope=governance_only,
    )

    decision = build_financial_review_scope(
        trigger,
        current_holdings=["600001.SH"],
        qualified_candidates=["SZ000001", "SZ000002"],
    )
    validated = validate_financial_review_scope(decision, trigger=trigger)
    assert validated["review_mode"] == "decision_review"
    assert validated["affected_holdings"] == ["SH600001"]
    assert validated["affected_candidates"] == ["SZ000001"]
    assert validated["ignored_instruments"] == ["SH600002"]
    assert run_recommendation_refresh._rebalance_due_for_financial_review(
        scheduled_rebalance_due=False,
        financial_review_scope=validated,
    )
    assert run_recommendation_refresh._rebalance_instruments_for_financial_review(
        scheduled_rebalance_due=False,
        financial_review_scope=validated,
    ) == ["SH600001", "SZ000001"]
    assert run_recommendation_refresh._rebalance_due_for_financial_review(
        scheduled_rebalance_due=True,
        financial_review_scope=governance_only,
    )
    assert run_recommendation_refresh._rebalance_instruments_for_financial_review(
        scheduled_rebalance_due=True,
        financial_review_scope=validated,
    ) is None


@pytest.mark.no_database
def test_order_plan_writes_only_an_actual_swing_decision_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_workflow(monkeypatch)
    manifest = {
        "order_plan_job_id": "job-swing",
        "simulation_portfolio_id": "paper-swing",
        "strategy_version_id": "version-swing",
        "formal_backtest_id": "formal-swing",
        "promotion_stage_id": "stage-swing",
        "promotion_stage_opened_at": "2026-08-01T00:00:00+00:00",
        "dataset": "daily-1",
        "dataset_identity_sha256": "a" * 64,
        "signal_date": "2026-08-31",
        "config": {
            "execution_contract_hash": "e" * 64,
            "horizon_profile": SWING_1_6M,
        },
    }
    result = {
        "as_of_date": "2026-08-31",
        "effective_date": "2026-09-01",
        "holdings": [{"instrument": "SH600000", "weight": 0.2}],
        "position_state": {},
        "risk_summary": {"rebalance_due": True},
    }
    output = run_recommendation_refresh._write_qlib_order_plan(
        manifest=manifest,
        result=result,
        dataset_provenance={
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "b" * 64,
        },
        order_plan_root=tmp_path / "plans",
        tracking_uri="memory://review-test",
    )
    plan = json.loads(
        (Path(output["order_plan_artifact_path"]) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    review = validate_horizon_review_evidence(
        plan["horizon_review"],
        strategy_version_id="version-swing",
        horizon_profile=SWING_1_6M,
        signal_date=date(2026, 8, 31),
        dataset_identity_sha256="a" * 64,
    )
    assert review["trigger_source"] == "rebalance_calendar:week"

    result["risk_summary"]["rebalance_due"] = False
    manifest["order_plan_job_id"] = "job-swing-hold"
    manifest["signal_date"] = "2026-09-01"
    result["as_of_date"] = "2026-09-01"
    result["effective_date"] = "2026-09-02"
    held = run_recommendation_refresh._write_qlib_order_plan(
        manifest=manifest,
        result=result,
        dataset_provenance={
            "dataset_identity_sha256": "a" * 64,
            "dataset_lineage_id": "b" * 64,
        },
        order_plan_root=tmp_path / "plans",
        tracking_uri="memory://review-test",
    )
    held_plan = json.loads(
        (Path(held["order_plan_artifact_path"]) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "horizon_review" not in held_plan


@pytest.mark.no_database
def test_long_order_plan_persists_new_pit_financial_review_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_workflow(monkeypatch)
    provenance = _snapshot_with_financials(tmp_path)
    trigger = resolve_financial_review_trigger(
        data_root=tmp_path,
        dataset_provenance=provenance,
        previous_signal_date=date(2026, 8, 27),
        signal_date=date(2026, 8, 31),
    )
    assert trigger is not None
    scope = build_financial_review_scope(
        trigger,
        current_holdings=["SH600001"],
        qualified_candidates=[],
    )
    manifest = {
        "order_plan_job_id": "job-long-financial",
        "simulation_portfolio_id": "paper-long",
        "strategy_version_id": "version-long",
        "formal_backtest_id": "formal-long",
        "promotion_stage_id": "stage-long",
        "promotion_stage_opened_at": "2026-08-01T00:00:00+00:00",
        "dataset": "daily-1",
        "dataset_identity_sha256": "a" * 64,
        "signal_date": "2026-08-31",
        "financial_review_trigger": trigger,
        "config": {
            "execution_contract_hash": "e" * 64,
            "horizon_profile": LONG_1_3Y,
        },
    }
    result = {
        "as_of_date": "2026-08-31",
        "effective_date": "2026-09-01",
        "holdings": [{"instrument": "SH600000", "weight": 0.2}],
        "position_state": {},
        "risk_summary": {
            "rebalance_due": True,
            "financial_review_trigger": trigger,
            "financial_review_scope": scope,
        },
    }

    output = run_recommendation_refresh._write_qlib_order_plan(
        manifest=manifest,
        result=result,
        dataset_provenance={
            **provenance,
            "dataset_lineage_id": "b" * 64,
        },
        order_plan_root=tmp_path / "plans",
        tracking_uri="memory://review-test",
    )
    plan = json.loads(
        (Path(output["order_plan_artifact_path"]) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    review = validate_horizon_review_evidence(
        plan["horizon_review"],
        strategy_version_id="version-long",
        horizon_profile=LONG_1_3Y,
        signal_date=date(2026, 8, 31),
        dataset_identity_sha256="a" * 64,
    )
    assert review["review_type"] == "financial_report_review"
    assert review["report_period"] == "2026Q2"
    assert review["report_periods"] == ["2025Q4", "2026Q2"]
    assert review["trigger_effective_date"] == "2026-08-31"
    assert review["source_event_sha256"] == trigger["source_event_sha256"]
    assert review["reviewed_instruments"] == ["SH600001"]
    assert review["review_scope_sha256"] == scope["scope_sha256"]

    tampered = dict(review)
    tampered["review_scope_sha256"] = "f" * 63
    unsigned = {key: value for key, value in tampered.items() if key != "evidence_sha256"}
    tampered["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="scope binding"):
        validate_horizon_review_evidence(
            tampered,
            strategy_version_id="version-long",
            horizon_profile=LONG_1_3Y,
            signal_date=date(2026, 8, 31),
            dataset_identity_sha256="a" * 64,
        )


@pytest.mark.no_database
def test_unrelated_financial_batch_does_not_claim_an_off_cadence_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_workflow(monkeypatch)
    provenance = _snapshot_with_financials(tmp_path)
    trigger = resolve_financial_review_trigger(
        data_root=tmp_path,
        dataset_provenance=provenance,
        previous_signal_date=date(2026, 8, 27),
        signal_date=date(2026, 8, 31),
    )
    assert trigger is not None
    scope = build_financial_review_scope(
        trigger,
        current_holdings=["SH600000"],
        qualified_candidates=["SZ000002"],
    )
    manifest = {
        "order_plan_job_id": "job-long-governance-only",
        "simulation_portfolio_id": "paper-long",
        "strategy_version_id": "version-long",
        "formal_backtest_id": "formal-long",
        "promotion_stage_id": "stage-long",
        "promotion_stage_opened_at": "2026-08-01T00:00:00+00:00",
        "dataset": "daily-1",
        "dataset_identity_sha256": "a" * 64,
        "signal_date": "2026-08-31",
        "financial_review_trigger": trigger,
        "config": {
            "execution_contract_hash": "e" * 64,
            "horizon_profile": LONG_1_3Y,
        },
    }
    result = {
        "as_of_date": "2026-08-31",
        "effective_date": "2026-09-01",
        "holdings": [{"instrument": "SH600000", "weight": 0.2}],
        "position_state": {},
        "risk_summary": {
            "rebalance_due": False,
            "scheduled_rebalance_due": False,
            "financial_review_trigger": trigger,
            "financial_review_scope": scope,
        },
    }

    output = run_recommendation_refresh._write_qlib_order_plan(
        manifest=manifest,
        result=result,
        dataset_provenance={
            **provenance,
            "dataset_lineage_id": "b" * 64,
        },
        order_plan_root=tmp_path / "plans",
        tracking_uri="memory://review-test",
    )
    plan = json.loads(
        (Path(output["order_plan_artifact_path"]) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "horizon_review" not in plan

    # If the same governance-only batch lands on the normal monthly cadence,
    # it must remain a scheduled review rather than falsely claiming that an
    # unrelated filing caused the decision.
    manifest["order_plan_job_id"] = "job-long-scheduled-with-unrelated-filing"
    result["risk_summary"]["rebalance_due"] = True
    result["risk_summary"]["scheduled_rebalance_due"] = True
    scheduled = run_recommendation_refresh._write_qlib_order_plan(
        manifest=manifest,
        result=result,
        dataset_provenance={
            **provenance,
            "dataset_lineage_id": "b" * 64,
        },
        order_plan_root=tmp_path / "plans",
        tracking_uri="memory://review-test",
    )
    scheduled_plan = json.loads(
        (Path(scheduled["order_plan_artifact_path"]) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    scheduled_review = validate_horizon_review_evidence(
        scheduled_plan["horizon_review"],
        strategy_version_id="version-long",
        horizon_profile=LONG_1_3Y,
        signal_date=date(2026, 8, 31),
        dataset_identity_sha256="a" * 64,
    )
    assert scheduled_review["review_type"] == "scheduled_review"
    assert scheduled_review["trigger_source"] == "rebalance_calendar:month"


def _seed_review_batches(
    promotion: PromotionStore,
    *,
    portfolio_id: str,
    version_id: str,
    horizon: str,
    review_dates: list[date],
    financial_index: int | None = None,
) -> tuple[object, object]:
    _seed_evidence(
        promotion,
        portfolio_id,
        nav_days=len(review_dates),
        succeeded=len(review_dates),
    )
    opened_at = datetime(2026, 1, 1, tzinfo=UTC)
    with promotion.engine.begin() as connection:
        connection.execute(
            update(strategy_promotion_stages)
            .where(strategy_promotion_stages.c.simulation_portfolio_id == portfolio_id)
            .values(opened_at=opened_at)
        )
        rows = connection.execute(
            select(simulation_batches)
            .where(simulation_batches.c.portfolio_id == portfolio_id)
            .order_by(simulation_batches.c.id)
        ).all()
        assert len(rows) == len(review_dates)
        for index, (row, signal_date) in enumerate(zip(rows, review_dates, strict=True)):
            if index == financial_index:
                review = build_horizon_review_evidence(
                    event_id=f"financial-{index}",
                    review_type="financial_report_review",
                    horizon_profile=LONG_1_3Y,
                    completed_at=_completed_at(signal_date),
                    strategy_version_id=version_id,
                    signal_date=signal_date,
                    dataset_identity_sha256=str(row.source_snapshot_id),
                    trigger_source="pit_financial_announcement",
                    trigger_effective_date=signal_date,
                    report_period="2026Q2",
                    announcement_date=signal_date.replace(day=13),
                    previous_signal_date=review_dates[index - 1],
                    source_datasets=["fina_indicator"],
                    source_event_count=3,
                    source_event_sha256="d" * 64,
                    report_periods=["2025Q4", "2026Q2"],
                    reviewed_instruments=["SH600000"],
                    review_scope_sha256="e" * 64,
                )
            else:
                frequency = "week" if horizon == SWING_1_6M else "month"
                review = build_horizon_review_evidence(
                    event_id=f"scheduled-{index}",
                    review_type="scheduled_review",
                    horizon_profile=horizon,
                    completed_at=_completed_at(signal_date),
                    strategy_version_id=version_id,
                    signal_date=signal_date,
                    dataset_identity_sha256=str(row.source_snapshot_id),
                    trigger_source=f"rebalance_calendar:{frequency}",
                    trigger_effective_date=signal_date,
                )
            payload = dict(row.target_payload_json)
            plan = dict(payload["governed_order_plan"])
            plan.update(
                {
                    "promotion_stage_opened_at": opened_at.isoformat(),
                    "horizon_review": review,
                }
            )
            payload["governed_order_plan"] = plan
            connection.execute(
                update(simulation_batches)
                .where(simulation_batches.c.id == row.id)
                .values(
                    signal_date=signal_date,
                    trade_date=signal_date + timedelta(days=1),
                    target_payload_json=payload,
                    created_at=_completed_at(signal_date),
                )
            )
        stage = connection.execute(
            select(strategy_promotion_stages).where(
                strategy_promotion_stages.c.simulation_portfolio_id == portfolio_id
            )
        ).one()
        portfolio = connection.execute(
            select(simulation_portfolios).where(
                simulation_portfolios.c.id == portfolio_id
            )
        ).one()
    return stage, portfolio


def test_promotion_counts_one_swing_review_per_actual_iso_week(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    stage_row, portfolio = _seed_review_batches(
        promotion,
        portfolio_id=stage["simulation_portfolio_id"],
        version_id=version_id,
        horizon=SWING_1_6M,
        review_dates=[date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 12)],
    )
    with promotion.engine.connect() as connection:
        evidence = promotion._collect_evidence(
            connection,
            stage_row,
            portfolio,
            SimpleNamespace(id=version_id, horizon_profile=SWING_1_6M),
        )
    assert evidence["review_events"] == 2
    assert evidence["review_event_periods"] == ["2026-W02", "2026-W03"]


def test_promotion_counts_months_and_distinct_pit_report_periods(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    version_id, promotion, stage = _gated_paper_version(
        database_url, tmp_path, monkeypatch
    )
    stage_row, portfolio = _seed_review_batches(
        promotion,
        portfolio_id=stage["simulation_portfolio_id"],
        version_id=version_id,
        horizon=LONG_1_3Y,
        review_dates=[
            date(2026, 1, 5),
            date(2026, 1, 20),
            date(2026, 2, 2),
            date(2026, 2, 16),
        ],
        financial_index=3,
    )
    with promotion.engine.connect() as connection:
        evidence = promotion._collect_evidence(
            connection,
            stage_row,
            portfolio,
            SimpleNamespace(id=version_id, horizon_profile=LONG_1_3Y),
        )
    assert evidence["review_events"] == 2
    assert evidence["review_event_periods"] == ["2026-01", "2026-02"]
    assert evidence["financial_report_reviews"] == 2
    assert evidence["financial_report_periods"] == ["2025Q4", "2026Q2"]
