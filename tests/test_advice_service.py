from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import quant_platform.advice_service as advice_module
from quant_platform.advice_service import (
    AdviceService,
    _project_backtest_status,
    _project_stage,
)
from quant_platform.research_horizon import SHORT_1_5D

pytestmark = pytest.mark.no_database


@pytest.mark.parametrize(
    ("run_status", "job_status", "expected"),
    (
        (None, None, "not_started"),
        ("queued", None, "queued"),
        ("queued", "running", "running"),
        ("succeeded", "running", "running"),
        ("succeeded", "succeeded", "succeeded"),
        ("succeeded", "failed", "failed"),
        ("cancelled", "succeeded", "failed"),
    ),
)
def test_backtest_projection_combines_run_and_job_fail_closed(
    run_status: str | None,
    job_status: str | None,
    expected: str,
) -> None:
    assert _project_backtest_status(run_status, job_status) == expected


@pytest.mark.parametrize(
    ("version_status", "promotion_stage", "backtest_status", "passed", "expected"),
    (
        ("draft", None, "not_started", False, ("research", "研究中")),
        ("candidate", None, "queued", False, ("backtest", "回测排队")),
        ("candidate", None, "running", False, ("backtest", "回测运行中")),
        ("candidate", None, "failed", False, ("backtest", "回测失败")),
        (
            "approved",
            "paper",
            "succeeded",
            False,
            ("simulation_validation", "模拟验证中"),
        ),
        (
            "candidate",
            "recommendation_enabled",
            "succeeded",
            True,
            ("research", "研究中"),
        ),
        (
            "approved",
            "recommendation_enabled",
            "succeeded",
            True,
            ("verified", "已验证"),
        ),
        (
            "approved",
            "recommendation_enabled",
            "succeeded",
            False,
            ("restricted", "受限"),
        ),
        (
            "approved",
            "recommendation_enabled",
            "not_started",
            True,
            ("backtest", "回测证据不完整"),
        ),
    ),
)
def test_stage_projection_distinguishes_cold_start_without_granting_advice(
    version_status: str,
    promotion_stage: str | None,
    backtest_status: str,
    passed: bool,
    expected: tuple[str, str],
) -> None:
    assert (
        _project_stage(
            version_status=version_status,
            promotion_stage=promotion_stage,
            health_status="healthy",
            backtest_status=backtest_status,
            forward_gate_passed=passed,
        )
        == expected
    )


class _Result:
    def __init__(self, row: Any) -> None:
        self.row = row

    def first(self) -> Any:
        return self.row


class _Connection(AbstractContextManager["_Connection"]):
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.statements: list[Any] = []

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, statement: Any) -> _Result:
        self.statements.append(statement)
        return _Result(self.rows.pop(0))


class _Engine:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def connect(self) -> _Connection:
        return self.connection


class _Promotions:
    @staticmethod
    def serving_incumbent_for_pending_cutover(_horizon: str) -> None:
        return None


class _EvidencePromotions:
    @staticmethod
    def evaluate_forward_gate(_version_id: str) -> dict[str, Any]:
        raise AssertionError("the mutating operational evaluator must not be called")

    @staticmethod
    def _collect_evidence(
        _connection: Any,
        _stage: Any,
        _portfolio: Any,
        _version: Any,
    ) -> dict[str, Any]:
        return {
            "ungoverned_batches": 0,
            "data_completeness": 1.0,
            "reconciliation_rate": 1.0,
            "cost_deviation": 0.0,
            "forward_calendar_days": 90,
            "forward_trading_days": 89,
            "decision_batches": 60,
            "completed_cycles": 30,
            "closed_round_trips": 30,
            "review_events": 0,
            "financial_report_reviews": 0,
        }


class _Strategies:
    @staticmethod
    def get_version(version_id: str) -> dict[str, Any]:
        return {
            "id": version_id,
            "version": 1,
            "status": "draft",
            "promotion_stage": None,
            "config": {},
        }


def test_latest_version_query_includes_draft_candidate_and_approved() -> None:
    row = SimpleNamespace(id="draft-v1", strategy_name="short baseline")
    connection = _Connection([row])
    service = AdviceService.__new__(AdviceService)
    service.engine = _Engine(connection)
    service.promotions = _Promotions()
    service.strategies = _Strategies()

    version = service._latest_version(SHORT_1_5D)

    assert version is not None
    assert version["id"] == "draft-v1"
    assert version["strategy_name"] == "short baseline"
    params = connection.statements[0].compile().params
    status_values = [
        set(value)
        for value in params.values()
        if isinstance(value, (tuple, list))
    ]
    assert {"draft", "candidate", "approved"} in status_values


def test_latest_backtest_reads_worker_state_and_projects_failure() -> None:
    row = SimpleNamespace(
        _mapping={
            "run_id": "backtest-1",
            "run_status": "succeeded",
            "created_at": datetime(2026, 8, 29, tzinfo=UTC),
            "started_at": datetime(2026, 8, 29, tzinfo=UTC),
            "finished_at": datetime(2026, 8, 29, tzinfo=UTC),
            "job_id": "job-1",
            "job_status": "failed",
        }
    )
    connection = _Connection([row])
    service = AdviceService.__new__(AdviceService)
    service.engine = _Engine(connection)

    state = service._latest_backtest("version-1")

    assert state["status"] == "failed"
    assert state["run_status"] == "succeeded"
    assert state["job_status"] == "failed"
    sql = str(connection.statements[0])
    assert "backtest_runs" in sql
    assert "jobs" in sql
    assert "created_at DESC" in sql


def test_forward_evidence_projection_executes_selects_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version = SimpleNamespace(horizon_profile=SHORT_1_5D)
    gate = SimpleNamespace(
        min_forward_calendar_days=0,
        min_forward_trading_days=90,
        min_decision_batches=60,
        min_completed_cycles=0,
        min_closed_round_trips=30,
        min_review_events=0,
        min_financial_report_reviews=0,
        min_data_completeness=0.99,
        min_reconciliation_rate=0.99,
        max_cost_deviation=0.01,
        criteria_sha256="criteria-sha",
    )
    stage = SimpleNamespace(
        id="stage-1",
        stage_index=1,
        simulation_portfolio_id="paper-1",
    )
    portfolio = SimpleNamespace(id="paper-1")
    connection = _Connection([version, gate, stage, portfolio])
    service = AdviceService.__new__(AdviceService)
    service.engine = _Engine(connection)
    service.promotions = _EvidencePromotions()
    monkeypatch.setattr(
        advice_module,
        "_require_forward_gate_criteria",
        lambda _version, _gate: {"criteria": "sealed"},
    )
    monkeypatch.setattr(
        advice_module.SimulationStore,
        "_require_current_source_contract",
        staticmethod(lambda _connection, _portfolio: None),
    )

    evidence = service._forward_evidence(
        {"id": "version-1", "promotion_stage": "paper"}
    )

    assert evidence["passed"] is False
    assert evidence["checks"]["forward_trading_days"] == {
        "observed": 89,
        "threshold": 90,
        "passed": False,
    }
    assert all(str(statement).lstrip().startswith("SELECT") for statement in connection.statements)


def _card_service(
    *,
    version_status: str,
    promotion_stage: str | None,
    backtest_status: str,
    forward_passed: bool,
) -> AdviceService:
    service = AdviceService.__new__(AdviceService)
    service._latest_version = lambda _horizon: {
        "id": "version-1",
        "version": 1,
        "status": version_status,
        "promotion_stage": promotion_stage,
        "strategy_name": "transparent baseline",
        "config": {},
    }
    service._latest_health = lambda _version_id: {"health_status": "healthy"}
    service._latest_backtest = lambda _version_id: {
        "status": backtest_status,
        "run_id": "backtest-1",
        "run_status": backtest_status,
        "job_id": "job-1",
        "job_status": backtest_status,
    }
    service._forward_evidence = lambda _version: {
        "status": "passed" if forward_passed else "not_started",
        "passed": forward_passed,
        "reasons": [],
    }
    service.simulations = SimpleNamespace(
        paper_target_for_strategy_version=lambda _version_id: {
            "signal_date": "2026-08-28",
            "trade_date": "2026-08-31",
            "targets": [
                {
                    "instrument": "000001.SZ",
                    "action": "BUY",
                    "target_weight": 0.05,
                }
            ],
        }
    )
    service._verified_signals = lambda *args, **kwargs: (
        [
            {
                "instrument": "000001.SZ",
                "action": "BUY",
                "target_weight": 0.05,
            }
        ],
        "2026-08-28",
    )
    return service


def test_draft_is_visible_as_research_but_never_as_advice() -> None:
    service = _card_service(
        version_status="draft",
        promotion_stage=None,
        backtest_status="not_started",
        forward_passed=False,
    )

    card = service._horizon_card(SHORT_1_5D, now=datetime.now(UTC))

    assert card["strategy"]["status"] == "draft"
    assert card["stage"] == "research"
    assert card["stage_label"] == "研究中"
    assert card["is_investment_advice"] is False
    assert card["signals"] == []
    assert card["action"] == "NO_ACTION"


def test_failed_backtest_blocks_even_recommendation_enabled_version() -> None:
    service = _card_service(
        version_status="approved",
        promotion_stage="recommendation_enabled",
        backtest_status="failed",
        forward_passed=True,
    )

    card = service._horizon_card(SHORT_1_5D, now=datetime.now(UTC))

    assert card["stage"] == "backtest"
    assert card["stage_label"] == "回测失败"
    assert card["is_investment_advice"] is False
    assert card["signals"] == []
    assert card["action"] == "NO_ACTION"
    assert any("不能作为荐股依据" in reason for reason in card["veto_reasons"])


@pytest.mark.parametrize(
    ("health_status", "expected_stage", "expected_label"),
    (
        ("restricted", "restricted", "受限"),
        ("suspended", "suspended", "暂停"),
        ("retired", "retired", "已退役"),
    ),
)
def test_current_health_state_blocks_verified_advice(
    health_status: str,
    expected_stage: str,
    expected_label: str,
) -> None:
    service = _card_service(
        version_status="approved",
        promotion_stage="recommendation_enabled",
        backtest_status="succeeded",
        forward_passed=True,
    )
    service._latest_health = lambda _version_id: {"health_status": health_status}

    card = service._horizon_card(SHORT_1_5D, now=datetime.now(UTC))

    assert card["stage"] == expected_stage
    assert card["stage_label"] == expected_label
    assert card["is_investment_advice"] is False
    assert card["signals"] == []
    assert card["action"] == "NO_ACTION"


def test_paper_version_exposes_simulation_signal_without_formal_action() -> None:
    service = _card_service(
        version_status="approved",
        promotion_stage="paper",
        backtest_status="succeeded",
        forward_passed=False,
    )

    card = service._horizon_card(SHORT_1_5D, now=datetime.now(UTC))

    assert card["stage"] == "simulation_validation"
    assert card["stage_label"] == "模拟验证中"
    assert card["is_investment_advice"] is False
    assert card["signals"][0]["instrument"] == "000001.SZ"
    assert card["simulation_action"] == "BUY"
    assert card["action"] == "NO_ACTION"


def test_only_approved_forward_passed_version_is_verified_advice() -> None:
    service = _card_service(
        version_status="approved",
        promotion_stage="recommendation_enabled",
        backtest_status="succeeded",
        forward_passed=True,
    )

    card = service._horizon_card(SHORT_1_5D, now=datetime.now(UTC))

    assert card["stage"] == "verified"
    assert card["stage_label"] == "已验证"
    assert card["is_investment_advice"] is True
    assert card["signals"][0]["instrument"] == "000001.SZ"
    assert card["action"] == "BUY"


def test_review_date_never_falls_back_to_weekday_guess() -> None:
    service = AdviceService.__new__(AdviceService)
    service.data_root = None

    review = service._review_projection(
        effective_date=date(2026, 10, 9),
        review_sessions=1,
        dataset="daily-v1",
    )

    assert review["review_sessions"] == 1
    assert review["review_date"] is None
    assert review["review_date_estimate"] is None
    assert review["review_date_is_exchange_calendar"] is False
    assert review["review_date_source"] == "strategy_horizon_contract"


def test_review_date_uses_only_governed_qlib_exchange_sessions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = AdviceService.__new__(AdviceService)
    service.data_root = tmp_path
    monkeypatch.setattr(
        advice_module,
        "select_qlib_dataset",
        lambda *_args, **_kwargs: {"path": str(tmp_path / "qlib")},
    )
    monkeypatch.setattr(
        advice_module,
        "load_calendar_days",
        lambda _path: {
            date(2026, 10, 9),
            # 10-12 is deliberately absent (exchange holiday).
            date(2026, 10, 13),
            date(2026, 10, 14),
        },
    )

    review = service._review_projection(
        effective_date=date(2026, 10, 9),
        review_sessions=1,
        dataset="daily-v1",
    )

    assert review["review_date"] == "2026-10-13"
    assert review["review_date_estimate"] == "2026-10-13"
    assert review["review_date_is_exchange_calendar"] is True
    assert review["review_date_source"] == "governed_qlib_exchange_calendar"


def test_unified_account_facts_override_placeholder_quantity_and_age() -> None:
    cards = [
        {
            "signals": [
                {
                    "instrument": "SH600000",
                    "target_quantity": None,
                    "holding_age_sessions": None,
                }
            ]
        }
    ]
    unified = {
        "instrument_facts": {
            "SH600000": {
                "target_quantity": 800,
                "target_quantity_source": "unified_account_order_plan",
                "holding_age_sessions": 12,
                "holding_age_source": "simulation_position_lots_qlib_calendar",
                "holding_age_evidence": {"status": "proven"},
            }
        }
    }

    AdviceService._attach_unified_account_facts(cards, unified)

    signal = cards[0]["signals"][0]
    assert signal["target_quantity"] == 800
    assert signal["target_quantity_source"] == "unified_account_order_plan"
    assert signal["holding_age_sessions"] == 12
    assert signal["holding_age_source"] == "simulation_position_lots_qlib_calendar"


def test_verified_signal_projects_persisted_account_target_quantity() -> None:
    snapshot = SimpleNamespace(
        id="snapshot-1",
        effective_date=date(2026, 8, 31),
        as_of_date=date(2026, 8, 28),
        dataset="daily-v1",
        account_actions_json={
            "items": [
                {
                    "instrument": "SH600000",
                    "action": "BUY",
                    "target_quantity": 600,
                }
            ]
        },
    )
    holding = SimpleNamespace(
        instrument="SH600000",
        weight=0.06,
        previous_weight=0.0,
        action="increase",
        reason="rank and cost gate passed",
    )

    class ResultWithAll(_Result):
        def all(self) -> list[Any]:
            return list(self.row)

    class ConnectionWithAll(_Connection):
        def execute(self, statement: Any) -> ResultWithAll:
            self.statements.append(statement)
            return ResultWithAll(self.rows.pop(0))

    connection = ConnectionWithAll([snapshot, [holding]])
    service = AdviceService.__new__(AdviceService)
    service.engine = _Engine(connection)
    service.data_root = None

    signals, cutoff = service._verified_signals(
        "version-1",
        horizon=SHORT_1_5D,
        review_sessions=1,
    )

    assert cutoff == "2026-08-28"
    assert signals[0]["target_quantity"] == 600
    assert signals[0]["target_quantity_source"] == (
        "recommendation_account_action_plan"
    )
    assert signals[0]["review_date"] is None
