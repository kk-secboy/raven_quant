from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quant_platform.autopilot_capital_pipeline import (
    CAPITAL_PIPELINE_CONTRACT_VERSION,
    CAPITAL_PIPELINE_STATE_KEY,
    AutopilotCapitalBlocked,
    AutopilotCapitalPipeline,
    AutopilotCapitalWaiting,
)
from quant_platform.autopilot_completion import (
    build_long_only_strategy_config,
    canonical_sha256,
)
from quant_platform.parameter_experiments import (
    PORTFOLIO_CONSTRUCTION_CANDIDATES,
    build_portfolio_construction_trials,
)
from quant_platform.promotion import ForwardGateThresholds
from quant_platform.research_horizon import LONG_1_3Y, SHORT_1_5D, SWING_1_6M

pytestmark = pytest.mark.no_database

IDENTITY = "d" * 64


def _signal_config() -> dict[str, Any]:
    return {
        "signal_source": "model_prediction",
        "model_candidate_id": "model-winner",
        "model_evaluation_id": "evaluation-model-winner",
        "model_code_sha256": "1" * 64,
        "model_recipe_sha256": "2" * 64,
        "model_evidence_sha256": "3" * 64,
        "feature_set_id": "qlib-alpha158",
        "feature_set_definition_sha256": "4" * 64,
        "quant_bundle_candidate_id": "bundle-winner",
        "quant_bundle_evaluation_id": "evaluation-bundle-winner",
        "quant_bundle_sha256": "5" * 64,
    }


def _selection() -> dict[str, Any]:
    evidence = {
        "dataset": "snapshot-v1",
        "dataset_identity_sha256": IDENTITY,
        "selected_kind": "joint",
        "selected_candidate_id": "bundle-winner",
        "selected_model_candidate_id": "model-winner",
        "selected_strategy_config": _signal_config(),
        "selected_periods": {
            "historical_start": "2010-01-04",
            "historical_end": "2023-12-22",
            "start": "2024-01-02",
            "end": "2024-12-31",
        },
        "selected_portfolio_validation": {
            "start": "2022-01-03",
            "end": "2023-12-22",
        },
        "final_oos_opened": False,
    }
    return {
        "champion_selection_evidence": evidence,
        "champion_selection_evidence_sha256": canonical_sha256(evidence),
    }


class _Strategies:
    def __init__(self) -> None:
        self.versions: dict[str, dict[str, Any]] = {}
        self.backtests: dict[str, dict[str, Any]] = {}
        self.attach_calls = 0

    def get_version(self, version_id: str) -> dict[str, Any]:
        return dict(self.versions[version_id])

    def get_backtest(self, backtest_id: str) -> dict[str, Any]:
        return dict(self.backtests[backtest_id])

    def attach_job(self, backtest_id: str, job_id: str) -> None:
        self.attach_calls += 1
        self.backtests[backtest_id]["job_id"] = job_id


class _Completion:
    def __init__(self, strategies: _Strategies) -> None:
        self.strategies = strategies
        self.selection = _selection()
        self.selection_calls = 0
        self.strategy_create_calls = 0
        self.backtest_create_calls = 0
        self.approve_calls = 0
        self.paper_status = "active"
        self.paper_portfolio_id: str | None = "paper-account-1"
        self.forward_thresholds: Any = None

    def ensure_selection(self, *, state: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.selection_calls += 1
        existing = state.get("champion_selection_evidence")
        if existing is not None:
            assert state["champion_selection_evidence_sha256"] == canonical_sha256(
                existing
            )
            assert existing["selected_candidate_id"] == "bundle-winner"
            return dict(state)
        return {**state, **self.selection, "phase": "champion_frozen"}

    def ensure_strategy(
        self,
        *,
        state: dict[str, Any],
        portfolio_config: dict[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        assert actor == "autopilot"
        construction = str(portfolio_config["portfolio_construction"])
        version_id = (
            "portfolio-version"
            if construction == "topk_equal_weight"
            else "final-version"
        )
        existing_id = str(state.get("strategy_version_id") or "")
        if existing_id:
            assert existing_id == version_id
        if version_id not in self.strategies.versions:
            self.strategy_create_calls += 1
            config = build_long_only_strategy_config(
                signal_config=_signal_config(),
                portfolio_config=portfolio_config,
                selection_evidence_sha256=state[
                    "champion_selection_evidence_sha256"
                ],
            )
            self.strategies.versions[version_id] = {
                "id": version_id,
                "strategy_id": f"strategy-{version_id}",
                "status": "draft",
                "config": config,
                "model_signal": {
                    "primary_training_periods": {
                        "valid_start": "2022-01-03",
                        "valid_end": "2023-12-22",
                    }
                },
            }
        config = self.strategies.versions[version_id]["config"]
        return {
            **state,
            "strategy_id": f"strategy-{version_id}",
            "strategy_version_id": version_id,
            "strategy_config_sha256": canonical_sha256(config),
        }

    def ensure_formal_backtest(
        self, *, state: dict[str, Any], trading_dates: list[date], **_: Any
    ) -> dict[str, Any]:
        assert trading_dates
        backtest_id = str(state.get("formal_backtest_id") or "")
        if not backtest_id:
            self.backtest_create_calls += 1
            backtest_id = "formal-oos-1"
            self.strategies.backtests[backtest_id] = {
                "id": backtest_id,
                "strategy_version_id": state["strategy_version_id"],
                "dataset": "snapshot-v1",
                "periods": {
                    "historical_start": "2010-01-04",
                    "historical_end": "2023-12-22",
                    "start": "2024-01-02",
                    "end": "2024-12-31",
                },
                "status": "queued",
                "job_id": None,
            }
        status = self.strategies.backtests[backtest_id]["status"]
        return {
            **state,
            "formal_backtest_id": backtest_id,
            "formal_backtest_status": status,
            "final_oos_consumed": True,
            "phase": "formal_backtest_running",
        }

    def approve_if_ready(
        self,
        *,
        state: dict[str, Any],
        actor: str,
        forward_thresholds: Any = None,
    ) -> dict[str, Any]:
        assert actor == "autopilot"
        self.forward_thresholds = forward_thresholds
        self.approve_calls += 1
        status = self.strategies.backtests[state["formal_backtest_id"]]["status"]
        if status != "succeeded":
            return {
                **state,
                "phase": "formal_backtest_failed",
                "terminal": True,
                "runner_up_allowed": False,
            }
        self.strategies.versions[state["strategy_version_id"]]["status"] = (
            "approved"
        )
        return {
            **state,
            "phase": "paper",
            "paper_stage": {
                "status": self.paper_status,
                "simulation_portfolio_id": self.paper_portfolio_id,
            },
            "paper_stage_opened": True,
            "forward_gate": {
                "min_forward_calendar_days": (
                    forward_thresholds.min_forward_calendar_days
                ),
                "min_decision_batches": forward_thresholds.min_decision_batches,
            },
            "recommendation_enabled": False,
            "strategy_status": "approved",
            "formal_backtest_status": "succeeded",
        }


class _Experiments:
    def __init__(self, final_config: dict[str, Any]) -> None:
        self.ensure_calls = 0
        self.attach_calls = 0
        self.experiment = {
            "id": "portfolio-experiment-1",
            "status": "queued",
            "job_id": None,
            "error": None,
        }
        self.frozen = {
            "experiment_id": "portfolio-experiment-1",
            "trial_index": 1,
            "portfolio_construction": "industry_neutral_qp",
            "portfolio_config": final_config,
            "portfolio_config_sha256": canonical_sha256(final_config),
            "governed_trial_count": 2,
            "final_oos_opened": False,
        }

    def ensure_model_portfolio_competition(self, **values: Any) -> dict[str, Any]:
        self.ensure_calls += 1
        version = values["strategy_version"]
        assert version["config"]["portfolio_construction"] == "topk_equal_weight"
        assert values["candidate_valid_start"] == "2022-01-03"
        assert values["candidate_valid_end"] == "2023-12-22"
        return {
            "experiment": dict(self.experiment),
            "job_payload": {
                "parameter_experiment_id": self.experiment["id"],
                "dataset": values["dataset"]["name"],
            },
            "created": self.ensure_calls == 1,
            "needs_job": (
                self.experiment["status"] == "queued"
                and not self.experiment["job_id"]
            ),
        }

    def attach_job(self, experiment_id: str, job_id: str) -> None:
        assert experiment_id == self.experiment["id"]
        self.attach_calls += 1
        self.experiment["job_id"] = job_id

    def get(self, experiment_id: str) -> dict[str, Any]:
        assert experiment_id == self.experiment["id"]
        return dict(self.experiment)

    def frozen_portfolio_config(self, experiment_id: str) -> dict[str, Any]:
        assert experiment_id == self.experiment["id"]
        return dict(self.frozen)


class _Jobs:
    def __init__(self) -> None:
        self.by_key: dict[str, dict[str, Any]] = {}
        self.create_calls = 0

    def create(self, kind: str, payload: dict[str, Any], _log: Path, **values: Any) -> dict:
        key = str(values["idempotency_key"])
        if key not in self.by_key:
            self.create_calls += 1
            self.by_key[key] = {
                "id": f"job-{self.create_calls}",
                "kind": kind,
                "payload": dict(payload),
                "status": "queued",
            }
        return dict(self.by_key[key])


def _pipeline() -> tuple[
    AutopilotCapitalPipeline, _Completion, _Experiments, _Jobs, _Strategies
]:
    strategies = _Strategies()
    completion = _Completion(strategies)
    provisional = build_long_only_strategy_config(
        signal_config=_signal_config(),
        portfolio_config={
            "portfolio_construction": "topk_equal_weight",
            "execution_method": "vwap",
            "execution_frequency": "5min",
            "execution_slice_minutes": 20,
            "max_execution_slices": 24,
        },
        selection_evidence_sha256=_selection()[
            "champion_selection_evidence_sha256"
        ],
    )
    final_config = {**provisional, "portfolio_construction": "industry_neutral_qp"}
    experiments = _Experiments(final_config)
    jobs = _Jobs()
    pipeline = object.__new__(AutopilotCapitalPipeline)
    pipeline.settings = SimpleNamespace(data_root=Path("data"))
    pipeline.completion = completion
    pipeline.experiments = experiments
    pipeline.jobs = jobs
    pipeline.strategies = strategies
    # Unit tests here isolate state-machine progression; the real immutable
    # alpha reservation contract is covered by the ledger/receipt tests.
    pipeline._reserve_capital_oos = lambda **_: {  # type: ignore[method-assign]
        "batch_id": "a" * 64,
        "batch_key": "capital-batch",
        "preregistration_sha256": "b" * 64,
        "frozen_bundle_manifest_sha256": "c" * 64,
        "frozen_baseline_manifest_sha256": "d" * 64,
        "stable_mandate": {},
        "stable_mandate_sha256": "e" * 64,
        "trading_dates_sha256": "f" * 64,
        "vintage_link": {
            "capital_oos_alpha_batch_id": "a" * 64,
            "capital_oos_dataset_identity_sha256": IDENTITY,
            "sealed_candidate_set_patch": {
                "capital_oos_alpha_ledger": {
                    "contract_version": "capital-oos-vintage-link-v1",
                    "batch_id": "a" * 64,
                    "preregistration_sha256": "b" * 64,
                    "frozen_bundle_manifest_sha256": "c" * 64,
                    "frozen_baseline_manifest_sha256": "d" * 64,
                }
            },
        },
    }
    pipeline._require_current_quant_admission = lambda run_id: [  # type: ignore[method-assign]
        "bundle-winner"
    ] if run_id == "quant-run-1" else []
    pipeline._execution_dataset = lambda **_: {  # type: ignore[method-assign]
        "name": "ashare-5m-v1",
        "path": "datasets/ashare-5m-v1",
        "frequency": "5min",
        "start_date": "2010-01-04",
        "end_date": "2024-12-31",
        "ready": True,
        "reproducible": True,
        "lineage_verified": True,
        "provenance": {
            "dataset_identity_sha256": "e" * 64,
            "dataset_lineage_id": "f" * 64,
            "source_lineage_id": "a" * 64,
        },
    }
    return pipeline, completion, experiments, jobs, strategies


def _dataset() -> dict[str, Any]:
    return {
        "name": "snapshot-v1",
        "path": "datasets/snapshot-v1",
        "lineage_id": "lineage-v1",
        "provenance": {
            "dataset_identity_sha256": IDENTITY,
            "dataset_lineage_id": "b" * 64,
            "source_lineage_id": "a" * 64,
        },
    }


def _cycle(state: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "config_revision": 0,
        "state": {
            CAPITAL_PIPELINE_STATE_KEY: dict(state or {}),
        }
    }


def _quant(status: str = "succeeded") -> dict[str, Any]:
    return {
        "status": status,
        "research_run_id": "quant-run-1",
    }


def test_portfolio_competition_is_exactly_topk_vs_qp() -> None:
    baseline = build_long_only_strategy_config(
        signal_config=_signal_config(),
        portfolio_config={"portfolio_construction": "topk_equal_weight"},
        selection_evidence_sha256="8" * 64,
    )
    grid, trials = build_portfolio_construction_trials(baseline)
    assert grid == {
        "portfolio_construction": list(PORTFOLIO_CONSTRUCTION_CANDIDATES)
    }
    assert len(trials) == 2
    assert {
        trial["parameters"]["portfolio_construction"] for trial in trials
    } == {"topk_equal_weight", "industry_neutral_qp"}
    assert all(
        trial["config"]["recommendation_enabled"] is False
        and trial["config"]["financing_enabled"] is False
        and trial["config"]["shorting_enabled"] is False
        for trial in trials
    )


def test_quant_must_finish_before_capital_pipeline_moves() -> None:
    pipeline, completion, _, jobs, _ = _pipeline()
    progress = pipeline.advance(
        cycle=_cycle(), dataset=_dataset(), quant_branch=_quant("running")
    )
    assert progress.stage == "joint_optimization"
    assert progress.state == {}
    assert completion.selection_calls == 0
    assert jobs.create_calls == 0


def test_cycle_uses_its_exact_persisted_paper_threshold_revision() -> None:
    class _Result:
        @staticmethod
        def first() -> Any:
            return SimpleNamespace(
                value_json={
                    "paper_min_calendar_days": 240,
                    "paper_min_trading_days": 160,
                }
            )

    class _Connection:
        def __enter__(self) -> _Connection:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        @staticmethod
        def execute(_: Any) -> _Result:
            return _Result()

    class _Engine:
        @staticmethod
        def connect() -> _Connection:
            return _Connection()

    pipeline = object.__new__(AutopilotCapitalPipeline)
    pipeline.engine = _Engine()

    thresholds = pipeline._forward_thresholds_for_cycle({"config_revision": 7})

    assert thresholds.min_forward_calendar_days == 240
    assert thresholds.min_decision_batches == 160


@pytest.mark.parametrize(
    ("horizon_profile", "expected"),
    [
        (
            SHORT_1_5D,
            {
                "min_forward_trading_days": 126,
                "min_decision_batches": 60,
                "min_closed_round_trips": 30,
                "min_review_events": 0,
                "min_financial_report_reviews": 0,
            },
        ),
        (
            SWING_1_6M,
            {
                "min_forward_trading_days": 252,
                "min_decision_batches": 0,
                "min_closed_round_trips": 6,
                "min_review_events": 24,
                "min_financial_report_reviews": 0,
            },
        ),
        (
            LONG_1_3Y,
            {
                "min_forward_trading_days": 252,
                "min_decision_batches": 0,
                "min_closed_round_trips": 0,
                "min_review_events": 12,
                "min_financial_report_reviews": 4,
            },
        ),
    ],
)
def test_explicit_horizon_uses_complete_authoritative_forward_gate(
    horizon_profile: str, expected: dict[str, int]
) -> None:
    pipeline = object.__new__(AutopilotCapitalPipeline)

    thresholds = pipeline._forward_thresholds_for_cycle(
        {"config_revision": 0, "horizon_profile": horizon_profile}
    )

    assert thresholds.min_forward_calendar_days == 183
    for field, value in expected.items():
        assert getattr(thresholds, field) == value


def test_explicit_swing_web_floor_is_trading_time_not_daily_decisions() -> None:
    class _Result:
        @staticmethod
        def first() -> Any:
            return SimpleNamespace(
                value_json={
                    "paper_min_calendar_days": 240,
                    "paper_min_trading_days": 300,
                }
            )

    class _Connection:
        def __enter__(self) -> _Connection:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        @staticmethod
        def execute(_: Any) -> _Result:
            return _Result()

    class _Engine:
        @staticmethod
        def connect() -> _Connection:
            return _Connection()

    pipeline = object.__new__(AutopilotCapitalPipeline)
    pipeline.engine = _Engine()

    thresholds = pipeline._forward_thresholds_for_cycle(
        {"config_revision": 9, "horizon_profile": SWING_1_6M}
    )

    assert thresholds.min_forward_calendar_days == 240
    assert thresholds.min_forward_trading_days == 300
    assert thresholds.min_decision_batches == 0
    assert thresholds.min_review_events == 24
    assert thresholds.min_closed_round_trips == 6


def test_portfolio_job_creation_is_idempotent() -> None:
    pipeline, completion, experiments, jobs, _ = _pipeline()
    first = pipeline.advance(
        cycle=_cycle(), dataset=_dataset(), quant_branch=_quant()
    )
    second = pipeline.advance(
        cycle=_cycle(first.state), dataset=_dataset(), quant_branch=_quant()
    )
    assert first.stage == second.stage == "portfolio_selection"
    assert first.state["contract_version"] == CAPITAL_PIPELINE_CONTRACT_VERSION
    assert first.state["portfolio_experiment_id"] == "portfolio-experiment-1"
    assert completion.selection_calls == 2
    assert experiments.attach_calls == 1
    assert jobs.create_calls == 1
    assert first.created_jobs == 1
    assert second.created_jobs == 0


def test_one_formal_oos_then_active_paper_without_live_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_platform.autopilot_capital_pipeline.load_calendar_days",
        lambda _: [date(2010, 1, 4), date(2024, 1, 2)],
    )
    pipeline, completion, experiments, jobs, strategies = _pipeline()
    portfolio = pipeline.advance(
        cycle=_cycle(), dataset=_dataset(), quant_branch=_quant()
    )
    experiments.experiment["status"] = "succeeded"

    formal = pipeline.advance(
        cycle=_cycle(portfolio.state), dataset=_dataset(), quant_branch=_quant()
    )
    retry = pipeline.advance(
        cycle=_cycle(formal.state), dataset=_dataset(), quant_branch=_quant()
    )
    assert formal.stage == retry.stage == "formal_backtest"
    assert completion.backtest_create_calls == 1
    assert strategies.attach_calls == 1
    assert jobs.create_calls == 2  # one portfolio job and one formal-OOS job
    assert formal.created_jobs == 1
    assert retry.created_jobs == 0

    strategies.backtests["formal-oos-1"]["status"] = "succeeded"
    complete = pipeline.advance(
        cycle=_cycle(retry.state), dataset=_dataset(), quant_branch=_quant()
    )
    assert complete.complete is True
    assert complete.stage == "paper"
    assert complete.state["paper_stage"] == {
        "status": "active",
        "simulation_portfolio_id": "paper-account-1",
    }
    assert complete.state["recommendation_enabled"] is False
    assert complete.state["forward_gate"] == {
        "min_forward_calendar_days": 183,
        "min_decision_batches": 126,
    }
    assert completion.forward_thresholds.min_forward_calendar_days == 183
    assert completion.forward_thresholds.min_decision_batches == 126
    final = strategies.versions[complete.state["strategy_version_id"]]["config"]
    assert final["position_side"] == "long_only"
    assert final["shorting_enabled"] is False
    assert final["financing_enabled"] is False
    assert final["broker_connection_enabled"] is False
    assert final["real_trading_eligible"] is False
    assert final["recommendation_enabled"] is False


def test_unsettled_capital_batch_waits_without_blocking_the_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_platform.autopilot_capital_pipeline.load_calendar_days",
        lambda _: [date(2010, 1, 4), date(2024, 1, 2)],
    )
    pipeline, _, experiments, jobs, _ = _pipeline()
    portfolio = pipeline.advance(
        cycle=_cycle(), dataset=_dataset(), quant_branch=_quant()
    )
    experiments.experiment["status"] = "succeeded"

    def _wait(**_: Any) -> dict[str, Any]:
        raise AutopilotCapitalWaiting(
            "capital OOS family already has an unsettled reserved batch"
        )

    pipeline._reserve_capital_oos = _wait  # type: ignore[method-assign]
    waiting = pipeline.advance(
        cycle=_cycle(portfolio.state), dataset=_dataset(), quant_branch=_quant()
    )

    assert waiting.complete is False
    assert waiting.stage == "capital_oos_waiting"
    assert waiting.state["capital_oos_waiting"] is True
    assert "unsettled reserved batch" in waiting.state["capital_oos_wait_reason"]
    assert jobs.create_calls == 1  # only the existing portfolio competition job


def test_cycle_frozen_forward_thresholds_reach_paper_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_platform.autopilot_capital_pipeline.load_calendar_days",
        lambda _: [date(2010, 1, 4), date(2024, 1, 2)],
    )
    pipeline, completion, experiments, _, strategies = _pipeline()
    pipeline._forward_thresholds_for_cycle = lambda _cycle: (  # type: ignore[method-assign]
        pipeline.completion_forward_thresholds
    )
    pipeline.completion_forward_thresholds = ForwardGateThresholds(  # type: ignore[attr-defined]
        min_forward_calendar_days=240,
        min_decision_batches=160,
        min_completed_cycles=0,
        min_data_completeness=0.95,
        min_reconciliation_rate=1.0,
        max_cost_deviation=0.005,
    )
    portfolio = pipeline.advance(
        cycle=_cycle(), dataset=_dataset(), quant_branch=_quant()
    )
    experiments.experiment["status"] = "succeeded"
    formal = pipeline.advance(
        cycle=_cycle(portfolio.state), dataset=_dataset(), quant_branch=_quant()
    )
    strategies.backtests["formal-oos-1"]["status"] = "succeeded"

    complete = pipeline.advance(
        cycle=_cycle(formal.state), dataset=_dataset(), quant_branch=_quant()
    )

    assert complete.state["forward_gate"] == {
        "min_forward_calendar_days": 240,
        "min_decision_batches": 160,
    }
    assert completion.forward_thresholds.min_forward_calendar_days == 240
    assert completion.forward_thresholds.min_decision_batches == 160


def test_failed_portfolio_or_oos_never_opens_paper_or_tries_runner_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_platform.autopilot_capital_pipeline.load_calendar_days",
        lambda _: [date(2010, 1, 4), date(2024, 1, 2)],
    )
    pipeline, completion, experiments, _, strategies = _pipeline()
    initial = pipeline.advance(
        cycle=_cycle(), dataset=_dataset(), quant_branch=_quant()
    )
    experiments.experiment.update(status="failed", error="QP evidence failed")
    with pytest.raises(AutopilotCapitalBlocked, match="QP evidence failed"):
        pipeline.advance(
            cycle=_cycle(initial.state), dataset=_dataset(), quant_branch=_quant()
        )
    assert completion.backtest_create_calls == 0
    assert completion.approve_calls == 0

    experiments.experiment.update(status="succeeded", error=None)
    formal = pipeline.advance(
        cycle=_cycle(initial.state), dataset=_dataset(), quant_branch=_quant()
    )
    strategies.backtests["formal-oos-1"]["status"] = "failed"
    with pytest.raises(AutopilotCapitalBlocked, match="formal_backtest_failed"):
        pipeline.advance(
            cycle=_cycle(formal.state), dataset=_dataset(), quant_branch=_quant()
        )
    assert completion.approve_calls == 1
    assert completion.selection["champion_selection_evidence"][
        "selected_candidate_id"
    ] == "bundle-winner"


def test_paper_must_be_active_and_own_an_isolated_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_platform.autopilot_capital_pipeline.load_calendar_days",
        lambda _: [date(2010, 1, 4), date(2024, 1, 2)],
    )
    pipeline, completion, experiments, _, strategies = _pipeline()
    first = pipeline.advance(
        cycle=_cycle(), dataset=_dataset(), quant_branch=_quant()
    )
    experiments.experiment["status"] = "succeeded"
    formal = pipeline.advance(
        cycle=_cycle(first.state), dataset=_dataset(), quant_branch=_quant()
    )
    strategies.backtests["formal-oos-1"]["status"] = "succeeded"
    completion.paper_status = "prepared"
    completion.paper_portfolio_id = None
    with pytest.raises(AutopilotCapitalBlocked, match="paper account"):
        pipeline.advance(
            cycle=_cycle(formal.state), dataset=_dataset(), quant_branch=_quant()
        )
