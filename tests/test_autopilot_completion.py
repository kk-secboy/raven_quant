from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from quant_platform.autopilot_completion import (
    AutopilotCompletionService,
    aggregate_pre_final_grid,
    build_long_only_strategy_config,
    canonical_sha256,
)
from quant_platform.model_research_governance import (
    REQUIRED_MODEL_SEEDS,
    REQUIRED_RESEARCH_PROFILES,
)
from quant_platform.promotion import ForwardGateThresholds, resolve_paper_initial_cash

pytestmark = pytest.mark.no_database

IDENTITY = "d" * 64
RESEARCH_SCOPE = {
    "research_screening_only": True,
    "not_capital_confirmation": True,
    "cross_cycle_fwer_claimed": False,
    "final_oos_opened": False,
}


def _capital_reserved_state(state: dict[str, Any]) -> dict[str, Any]:
    """Minimal immutable preregistration link for Completion unit fakes."""

    return {
        **state,
        "capital_oos_dataset_identity_sha256": IDENTITY,
        "capital_oos_vintage_link": {
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


def _quant_receipt() -> dict[str, Any]:
    receipt = {
        "contract_version": "fin-quant-research-ledger-receipt-v1",
        "research_tournament_id": "quant-tournament",
        "parent_research_tournament_id": "model-tournament",
        "research_tournament_manifest_sha256": "9" * 64,
        "research_trial_ids": {"bundle-joint": "quant-trial"},
        "candidate_statuses": {"bundle-joint": "passed"},
        "run_multiple_testing_evidence_sha256": "8" * 64,
        "failed_and_rejected_trials_retained": True,
        "failed_candidate_raw_p_value": 1.0,
        **RESEARCH_SCOPE,
    }
    receipt["evidence_sha256"] = canonical_sha256(receipt)
    return receipt


def _signal_config(model_id: str, *, bundle_id: str | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "signal_source": "model_prediction",
        "model_candidate_id": model_id,
        "model_evaluation_id": f"evaluation-{model_id}",
        "model_code_sha256": "1" * 64,
        "model_recipe_sha256": "2" * 64,
        "model_evidence_sha256": "3" * 64,
        "feature_set_id": "qlib-alpha158",
        "feature_set_definition_sha256": "4" * 64,
    }
    if bundle_id:
        config.update(
            {
                "quant_bundle_candidate_id": bundle_id,
                "quant_bundle_evaluation_id": f"evaluation-{bundle_id}",
                "quant_bundle_sha256": "5" * 64,
            }
        )
    return config


def _grid(value: float) -> list[dict[str, Any]]:
    rows = []
    for profile in REQUIRED_RESEARCH_PROFILES:
        for seed in REQUIRED_MODEL_SEEDS:
            metrics = {
                "annualized_excess_return_with_cost": value,
                "information_ratio": value * 2,
                "rank_ic": value / 2,
                "average_turnover": 0.10,
                "max_drawdown": -0.08,
            }
            rows.append(
                {
                    "profile_id": profile,
                    "seed": seed,
                    "evidence_role": "independent_gate",
                    "gate_status": "passed",
                    "oos_vintage_id": None,
                    "metrics": metrics,
                    "metrics_sha256": canonical_sha256(metrics),
                    "train_start": date(2010, 1, 4),
                    "valid_start": date(2022, 1, 4),
                    "valid_end": date(2023, 12, 22),
                }
            )
    return rows


def _model(candidate_id: str, *, identity: str = IDENTITY) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "status": "research_admitted",
        "dataset": "snapshot-v1",
        "dataset_identity_sha256": identity,
        "manifest_sha256": "6" * 64,
        "pre_final_end": date(2023, 12, 22),
        "final_oos_start": date(2024, 1, 2),
        "final_oos_end": date(2024, 12, 31),
        "admission_evidence_json": dict(RESEARCH_SCOPE),
    }


class _Candidates:
    def __init__(self) -> None:
        self.models = {
            "model-strong": _model("model-strong"),
            "model-joint": _model("model-joint"),
            "model-other-dataset": _model("model-other-dataset", identity="e" * 64),
        }
        self.bundles = {
            "bundle-joint": {
                "id": "bundle-joint",
                "model_candidate_id": "model-joint",
                "status": "research_admitted",
                "dataset": "snapshot-v1",
                "dataset_identity_sha256": IDENTITY,
                "bundle_manifest_sha256": "7" * 64,
                "bundle_manifest_json": {
                    "baseline_prediction_champion": {
                        "kind": "model",
                        "candidate_id": "model-strong",
                    }
                },
                "admission_evidence_json": {
                    "research_trial_ledger_receipt": _quant_receipt(),
                    "research_trial_ledger_receipt_sha256": _quant_receipt()[
                        "evidence_sha256"
                    ],
                    **RESEARCH_SCOPE,
                },
            }
        }
        self.signals = [
            {
                "kind": "model",
                "id": "model-strong",
                "name": "strong model",
                "strategy_config": _signal_config("model-strong"),
            },
            {
                "kind": "joint",
                "id": "bundle-joint",
                "name": "complete joint",
                "strategy_config": _signal_config(
                    "model-joint", bundle_id="bundle-joint"
                ),
            },
            {
                "kind": "model",
                "id": "model-other-dataset",
                "name": "wrong identity",
                "strategy_config": _signal_config("model-other-dataset"),
            },
        ]

    def list_admitted_strategy_signals(self, *, limit: int) -> list[dict[str, Any]]:
        return self.signals[:limit]

    def get_model_candidate(self, candidate_id: str, *, verify: bool) -> dict[str, Any]:
        assert verify is True
        return dict(self.models[candidate_id])

    def get_quant_bundle_candidate(
        self, candidate_id: str, *, verify: bool
    ) -> dict[str, Any]:
        assert verify is True
        return dict(self.bundles[candidate_id])


class _Strategies:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.by_name: dict[str, dict[str, Any]] = {}
        self.versions: dict[str, dict[str, Any]] = {}
        self.backtests: dict[str, dict[str, Any]] = {}
        self.create_calls = 0
        self.backtest_create_calls = 0
        self.approve_calls = 0

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        return self.by_name.get(name)

    def create(self, **values: Any) -> dict[str, Any]:
        self.create_calls += 1
        strategy_id = f"strategy-{self.create_calls}"
        version_id = f"version-{self.create_calls}"
        version = {
            "id": version_id,
            "strategy_id": strategy_id,
            "status": "draft",
            "config": dict(values["config"]),
        }
        strategy = {"id": strategy_id, "name": values["name"], "versions": [version]}
        self.versions[version_id] = version
        self.by_name[values["name"]] = strategy
        return strategy

    def get_version(self, version_id: str) -> dict[str, Any]:
        return dict(self.versions[version_id])

    def create_backtest(self, **values: Any) -> dict[str, Any]:
        self.backtest_create_calls += 1
        backtest = {
            "id": f"backtest-{self.backtest_create_calls}",
            "strategy_version_id": values["version_id"],
            "dataset": values["dataset"],
            "execution_dataset": values.get("execution_dataset"),
            "periods": dict(values["periods"]),
            "status": "queued",
        }
        self.backtests[backtest["id"]] = backtest
        return dict(backtest)

    def list_backtests(self, *, version_id: str, limit: int) -> list[dict[str, Any]]:
        return [
            dict(value)
            for value in self.backtests.values()
            if value["strategy_version_id"] == version_id
        ][:limit]

    def get_backtest(self, backtest_id: str) -> dict[str, Any]:
        return dict(self.backtests[backtest_id])

    def approve(self, version_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        self.events.append("approve")
        self.approve_calls += 1
        self.versions[version_id]["status"] = "approved"
        return dict(self.versions[version_id])


class _Promotions:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.thresholds: Any = None
        self.prepare_calls = 0

    def register_forward_gate(self, version_id: str, *, actor: str, thresholds: Any) -> dict:
        self.events.append("register")
        self.thresholds = thresholds
        return {"strategy_version_id": version_id}

    def prepare_paper_stage(self, version_id: str, *, actor: str) -> dict[str, Any]:
        self.events.append("paper")
        self.prepare_calls += 1
        return {"strategy_version_id": version_id, "status": "active"}


class _Service(AutopilotCompletionService):
    def __init__(self, candidates: _Candidates, strategies: _Strategies, promotions: _Promotions):
        super().__init__(
            "unused",
            candidate_store=candidates,  # type: ignore[arg-type]
            strategy_store=strategies,  # type: ignore[arg-type]
            promotion_store=promotions,  # type: ignore[arg-type]
            tournament_store=_ResearchTournaments(),  # type: ignore[arg-type]
        )

    def _grid_rows(self, *, kind: str, candidate_id: str) -> list[dict[str, Any]]:
        values = {
            "model-strong": 0.10,
            "bundle-joint": 0.30,
            "model-other-dataset": 0.99,
        }
        return _grid(values[candidate_id])


class _EnsembleTournament:
    def get_ensemble(self, candidate_id: str, *, verify: bool) -> dict[str, Any]:
        assert candidate_id == "ensemble-1"
        assert verify is True
        evaluations = [
            {
                "id": f"ensemble-cell-{profile}-{seed}",
                "profile_id": profile,
                "seed": seed,
                "gate_status": "passed",
            }
            for profile in REQUIRED_RESEARCH_PROFILES
            for seed in REQUIRED_MODEL_SEEDS
        ]
        return {
            "id": candidate_id,
            "status": "research_admitted",
            "dataset": "snapshot-v1",
            "dataset_identity_sha256": IDENTITY,
            "manifest_sha256": "a" * 64,
            "admission_evidence_sha256": "b" * 64,
            "tournament_id": "model-tournament",
            "admission_evidence": dict(RESEARCH_SCOPE),
            "components": [
                {"model_candidate_id": "model-a", "model_family": "ridge"},
                {"model_candidate_id": "model-b", "model_family": "gru"},
            ],
            "evaluations": evaluations,
        }

    def get_tournament(self, tournament_id: str) -> dict[str, Any]:
        assert tournament_id == "model-tournament"
        return {
            "id": tournament_id,
            "status": "succeeded",
            "selected_trial_ids": ["ensemble-trial"],
            "trials": [
                {
                    "id": "ensemble-trial",
                    "candidate_id": "ensemble-1",
                    "trial_kind": "model_ensemble",
                    "status": "selected",
                }
            ],
        }


class _ResearchTournaments:
    def get_tournament(self, tournament_id: str) -> dict[str, Any]:
        assert tournament_id == "quant-tournament"
        receipt = _quant_receipt()
        return {
            "id": tournament_id,
            "stage": "quant",
            "status": "succeeded",
            "manifest_sha256": "9" * 64,
            "selected_trial_ids": ["quant-trial"],
            "multiple_testing": {
                "research_trial_ledger_receipt_sha256": receipt["evidence_sha256"],
                **RESEARCH_SCOPE,
            },
            "trials": [
                {
                    "id": "quant-trial",
                    "candidate_id": "bundle-joint",
                    "trial_kind": "quant_bundle",
                    "status": "passed",
                }
            ],
        }


class _EnsembleCandidates(_Candidates):
    def __init__(self) -> None:
        super().__init__()
        self.models.update({"model-a": _model("model-a"), "model-b": _model("model-b")})
        self.signals = [
            {
                "kind": "ensemble",
                "id": "ensemble-1",
                "name": "ridge + gru",
                "strategy_config": {
                    "signal_source": "model_prediction",
                    "model_ensemble_candidate_id": "ensemble-1",
                    "model_ensemble_evaluation_id": (
                        f"ensemble-cell-{REQUIRED_RESEARCH_PROFILES[0]}-"
                        f"{REQUIRED_MODEL_SEEDS[0]}"
                    ),
                    "model_ensemble_manifest_sha256": "a" * 64,
                    "model_ensemble_evidence_sha256": "b" * 64,
                    "model_ensemble_combiner": "equal_rank",
                    "model_ensemble_stacking": False,
                    "model_component_candidate_ids": ["model-a", "model-b"],
                    "model_component_families": ["ridge", "gru"],
                },
            }
        ]


class _EnsembleService(AutopilotCompletionService):
    def __init__(self) -> None:
        events: list[str] = []
        super().__init__(
            "unused",
            candidate_store=_EnsembleCandidates(),  # type: ignore[arg-type]
            strategy_store=_Strategies(events),  # type: ignore[arg-type]
            promotion_store=_Promotions(events),  # type: ignore[arg-type]
            tournament_store=_EnsembleTournament(),  # type: ignore[arg-type]
        )

    def _grid_rows(self, *, kind: str, candidate_id: str) -> list[dict[str, Any]]:
        assert kind == "ensemble"
        assert candidate_id == "ensemble-1"
        return _grid(0.18)


def _service() -> tuple[_Service, _Strategies, _Promotions]:
    events: list[str] = []
    strategies = _Strategies(events)
    promotions = _Promotions(events)
    return _Service(_Candidates(), strategies, promotions), strategies, promotions


def test_grid_averages_seeds_inside_each_profile() -> None:
    rows = _grid(0.10)
    rows[0]["metrics"]["annualized_excess_return_with_cost"] = 0.40
    rows[0]["metrics_sha256"] = canonical_sha256(rows[0]["metrics"])
    result = aggregate_pre_final_grid(rows)
    recent = result["profiles"][REQUIRED_RESEARCH_PROFILES[0]]
    assert recent["annualized_excess_return_with_cost"] == pytest.approx(0.20)
    assert result["seeds_are_robustness_repeats"] is True


def test_weaker_joint_does_not_displace_frozen_incumbent() -> None:
    class _WeakerJointService(_Service):
        def _grid_rows(self, *, kind: str, candidate_id: str) -> list[dict[str, Any]]:
            del kind
            return _grid(0.05 if candidate_id == "bundle-joint" else 0.30)

    events: list[str] = []
    service = _WeakerJointService(
        _Candidates(), _Strategies(events), _Promotions(events)
    )
    selected = service.select_champion(
        dataset="snapshot-v1", dataset_identity_sha256=IDENTITY
    )
    evidence = selected["champion_selection_evidence"]
    assert evidence["selected_kind"] == "model"
    assert evidence["selected_candidate_id"] == "model-strong"
    assert {item["candidate_id"] for item in evidence["eligible_candidates"]} == {
        "model-strong",
        "bundle-joint",
    }
    assert selected["champion_selection_evidence_sha256"] == canonical_sha256(evidence)
    state = service.ensure_selection(
        state={},
        dataset="snapshot-v1",
        dataset_identity_sha256=IDENTITY,
        allowed_candidate_ids={"bundle-joint"},
    )
    retried = service.ensure_selection(
        state=state,
        dataset="snapshot-v1",
        dataset_identity_sha256=IDENTITY,
        allowed_candidate_ids={"bundle-joint"},
    )
    assert retried["champion_selection_evidence"]["selected_candidate_id"] == (
        "model-strong"
    )


def test_joint_replaces_incumbent_only_after_three_window_improvement() -> None:
    class _QualifiedService(_Service):
        def _grid_rows(self, *, kind: str, candidate_id: str) -> list[dict[str, Any]]:
            del kind
            return _grid(0.20 if candidate_id == "bundle-joint" else 0.10)

    events: list[str] = []
    service = _QualifiedService(
        _Candidates(), _Strategies(events), _Promotions(events)
    )
    selected = service.select_champion(
        dataset="snapshot-v1",
        dataset_identity_sha256=IDENTITY,
        allowed_candidate_ids={"bundle-joint"},
    )
    evidence = selected["champion_selection_evidence"]
    assert evidence["selected_kind"] == "joint"
    assert evidence["selected_candidate_id"] == "bundle-joint"
    assert evidence["replacement_decisions"][0]["evidence"]["passed"] is True
    assert set(evidence["capital_pool_candidate_ids"]) == {
        "model-strong",
        "bundle-joint",
    }


def test_recent_gain_cannot_hide_balanced_window_degradation() -> None:
    class _UnstableService(_Service):
        def _grid_rows(self, *, kind: str, candidate_id: str) -> list[dict[str, Any]]:
            del kind
            rows = _grid(0.20 if candidate_id == "bundle-joint" else 0.10)
            if candidate_id == "bundle-joint":
                for row in rows:
                    if row["profile_id"] == "balanced_5y":
                        row["metrics"]["annualized_excess_return_with_cost"] = 0.09
                        row["metrics"]["rank_ic"] = 0.045
                        row["metrics_sha256"] = canonical_sha256(row["metrics"])
            return rows

    events: list[str] = []
    service = _UnstableService(_Candidates(), _Strategies(events), _Promotions(events))
    selected = service.select_champion(
        dataset="snapshot-v1",
        dataset_identity_sha256=IDENTITY,
        allowed_candidate_ids={"bundle-joint"},
    )
    evidence = selected["champion_selection_evidence"]
    assert evidence["selected_candidate_id"] == "model-strong"
    decision = evidence["replacement_decisions"][0]["evidence"]
    assert decision["recent_strict_improvement"]["passed"] is True
    assert decision["balanced_non_degradation"]["passed"] is False
    assert decision["passed"] is False


def test_completion_recognizes_a_strict_equal_rank_ensemble() -> None:
    selected = _EnsembleService().select_champion(
        dataset="snapshot-v1",
        dataset_identity_sha256=IDENTITY,
        allowed_candidate_ids={"ensemble-1"},
    )
    evidence = selected["champion_selection_evidence"]
    assert evidence["selected_kind"] == "ensemble"
    assert evidence["selected_candidate_id"] == "ensemble-1"
    assert evidence["selected_model_component_candidate_ids"] == [
        "model-a",
        "model-b",
    ]
    assert evidence["selected_strategy_config"]["model_ensemble_stacking"] is False


def test_completion_rejects_research_admission_without_scope_or_quant_receipt() -> None:
    candidates = _Candidates()
    events: list[str] = []
    service = _Service(candidates, _Strategies(events), _Promotions(events))
    candidates.models["model-strong"]["admission_evidence_json"] = {}
    with pytest.raises(ValueError, match="research-only scope"):
        service.select_champion(
            dataset="snapshot-v1", dataset_identity_sha256=IDENTITY
        )

    candidates = _Candidates()
    candidates.bundles["bundle-joint"]["admission_evidence_json"].pop(
        "research_trial_ledger_receipt"
    )
    service = _Service(candidates, _Strategies(events), _Promotions(events))
    with pytest.raises(ValueError, match="research-ledger receipt"):
        service.select_champion(
            dataset="snapshot-v1",
            dataset_identity_sha256=IDENTITY,
            allowed_candidate_ids={"bundle-joint"},
        )


def test_long_only_config_rejects_financing_and_impossible_qp() -> None:
    config = build_long_only_strategy_config(
        signal_config=_signal_config("model-strong"),
        portfolio_config={"portfolio_construction": "topk_equal_weight"},
        selection_evidence_sha256="8" * 64,
    )
    assert config["position_side"] == "long_only"
    assert config["financing_enabled"] is False
    assert config["shorting_enabled"] is False
    assert config["annual_borrow_rate"] == 0.0
    assert config["recommendation_enabled"] is False
    assert config["capacity_notional"] == 5_000_000
    assert config["paper_initial_cash"] == config["capacity_notional"]
    capital_contract = config["autopilot_capital_execution_contract"]
    assert capital_contract == {
        "contract_version": "autopilot-capital-execution-v1",
        "portfolio_construction": "topk_equal_weight",
        "capacity_notional": 5_000_000.0,
        "paper_initial_cash": 5_000_000.0,
        "topk": 100,
        "n_drop": 10,
        "lot_size": 100,
        "min_commission": 5.0,
        "cost_schedule_version": config["cost_schedule_version"],
    }
    assert config["autopilot_capital_execution_contract_sha256"] == canonical_sha256(
        capital_contract
    )
    assert config["model_drift_policy"] == {
        "contract_version": "model-drift-policy-v1",
        "metric": "cost_after_excess_return",
        "window_trading_days": 20,
        "consecutive_windows": 3,
        "threshold": 0.0,
        "comparison": "below",
    }
    assert config["model_drift_policy_sha256"] == canonical_sha256(
        config["model_drift_policy"]
    )
    assert resolve_paper_initial_cash(config) == 5_000_000
    # Personal paper principal is independent from the research capacity scale.
    assert resolve_paper_initial_cash(config, requested_initial_cash=100_000) == 100_000
    changed = dict(config)
    changed["topk"] = 50
    with pytest.raises(ValueError, match="capital execution contract is inconsistent"):
        resolve_paper_initial_cash(changed)
    assert len(config["execution_contract_hash"]) == 64

    with pytest.raises(ValueError, match="unsupported autopilot portfolio fields"):
        build_long_only_strategy_config(
            signal_config=_signal_config("model-strong"),
            portfolio_config={"financing_enabled": True},
            selection_evidence_sha256="8" * 64,
        )
    with pytest.raises(ValueError, match="cannot form a fully invested"):
        build_long_only_strategy_config(
            signal_config=_signal_config("model-strong"),
            portfolio_config={
                "portfolio_construction": "industry_neutral_qp",
                "topk": 10,
                "max_position_weight": 0.05,
            },
            selection_evidence_sha256="8" * 64,
        )


def test_completion_is_idempotent_and_registers_forward_gate_before_approval() -> None:
    service, strategies, promotions = _service()
    state = service.ensure_selection(
        state={}, dataset="snapshot-v1", dataset_identity_sha256=IDENTITY
    )
    state = service.ensure_strategy(
        state=_capital_reserved_state(state),
        portfolio_config={"portfolio_construction": "topk_equal_weight"},
        actor="autopilot",
    )
    duplicate = service.ensure_strategy(
        state=_capital_reserved_state(state),
        portfolio_config={"portfolio_construction": "topk_equal_weight"},
        actor="autopilot",
    )
    assert duplicate["strategy_version_id"] == state["strategy_version_id"]
    assert strategies.create_calls == 1

    state = service.ensure_formal_backtest(
        state=state,
        artifact_path=Path("backtests"),
        execution_dataset="ashare-5m-v1",
        trading_dates=[date(2010, 1, 4), date(2024, 1, 2)],
        dataset_lineage_id="lineage-v1",
    )
    duplicate = service.ensure_formal_backtest(
        state=state,
        artifact_path=Path("backtests"),
        execution_dataset="ashare-5m-v1",
        trading_dates=[date(2010, 1, 4), date(2024, 1, 2)],
        dataset_lineage_id="lineage-v1",
    )
    assert duplicate["formal_backtest_id"] == state["formal_backtest_id"]
    assert strategies.backtest_create_calls == 1
    strategies.backtests[state["formal_backtest_id"]]["status"] = "succeeded"

    approved = service.approve_if_ready(state=state, actor="autopilot")
    assert approved["phase"] == "paper"
    assert approved["recommendation_enabled"] is False
    assert strategies.events[:3] == ["register", "approve", "paper"]
    assert promotions.thresholds.min_forward_calendar_days == 183
    assert promotions.thresholds.min_decision_batches == 126
    assert strategies.approve_calls == 1

    retried = service.approve_if_ready(state=approved, actor="autopilot")
    assert retried["phase"] == "paper"
    assert strategies.approve_calls == 1


def test_completion_uses_stricter_frozen_forward_thresholds() -> None:
    service, strategies, promotions = _service()
    state = service.ensure_selection(
        state={}, dataset="snapshot-v1", dataset_identity_sha256=IDENTITY
    )
    state = service.ensure_strategy(
        state=_capital_reserved_state(state),
        portfolio_config={"portfolio_construction": "topk_equal_weight"},
        actor="autopilot",
    )
    state = service.ensure_formal_backtest(
        state=state,
        artifact_path=Path("backtests"),
        execution_dataset="ashare-5m-v1",
        trading_dates=[date(2010, 1, 4), date(2024, 1, 2)],
        dataset_lineage_id="lineage-v1",
    )
    strategies.backtests[state["formal_backtest_id"]]["status"] = "succeeded"
    thresholds = service._forward_thresholds(
        min_forward_calendar_days=240,
        min_decision_batches=160,
    )

    approved = service.approve_if_ready(
        state=state,
        actor="autopilot",
        forward_thresholds=thresholds,
    )

    assert promotions.thresholds.min_forward_calendar_days == 240
    assert promotions.thresholds.min_decision_batches == 160
    assert approved["forward_gate"] == {
        "min_forward_calendar_days": 240,
        "min_decision_batches": 160,
    }


def test_completion_rejects_weaker_forward_thresholds() -> None:
    service, _, _ = _service()
    with pytest.raises(ValueError, match="at least 183 calendar days"):
        service._require_autopilot_forward_thresholds(
            ForwardGateThresholds(
                min_forward_calendar_days=182,
                min_decision_batches=126,
            )
        )


def test_failed_formal_backtest_is_terminal_without_runner_up() -> None:
    service, strategies, _ = _service()
    state = service.ensure_selection(
        state={}, dataset="snapshot-v1", dataset_identity_sha256=IDENTITY
    )
    state = service.ensure_strategy(
        state=_capital_reserved_state(state),
        portfolio_config={"portfolio_construction": "topk_equal_weight"},
        actor="autopilot",
    )
    state = service.ensure_formal_backtest(
        state=state,
        artifact_path=Path("backtests"),
        execution_dataset="ashare-5m-v1",
        trading_dates=[date(2010, 1, 4), date(2024, 1, 2)],
        dataset_lineage_id="lineage-v1",
    )
    strategies.backtests[state["formal_backtest_id"]]["status"] = "failed"
    result = service.approve_if_ready(state=state, actor="autopilot")
    assert result["terminal"] is True
    assert result["runner_up_allowed"] is False
    assert result["champion_selection_evidence"]["selected_candidate_id"] == (
        "bundle-joint"
    )
    assert strategies.approve_calls == 0


def test_frozen_selection_hash_is_fail_closed() -> None:
    service, _, _ = _service()
    state = service.ensure_selection(
        state={}, dataset="snapshot-v1", dataset_identity_sha256=IDENTITY
    )
    state["champion_selection_evidence"]["selected_candidate_id"] = "model-strong"
    with pytest.raises(ValueError, match="selection evidence is inconsistent"):
        service.ensure_selection(
            state=state,
            dataset="snapshot-v1",
            dataset_identity_sha256=IDENTITY,
        )
