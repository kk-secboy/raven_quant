from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select

from quant_data.database import (
    model_ensemble_evaluations,
    model_evaluations,
    quant_bundle_evaluations,
)
from quant_data.execution_contract import (
    require_strategy_execution_contract,
    strategy_execution_contract_hash,
)
from quant_platform.cost_model import CostModelConfig
from quant_platform.model_research_governance import (
    PRIMARY_MODEL_PROFILE,
    PRIMARY_MODEL_SEED,
    REQUIRED_MODEL_SEEDS,
    REQUIRED_RESEARCH_PROFILES,
)
from quant_platform.model_strategy_contract import normalize_model_signal_config
from quant_platform.promotion import ForwardGateThresholds, PromotionStore
from quant_platform.rdagent_candidate_store import RDAGentCandidateStore
from quant_platform.research_tournament import ResearchTournamentStore
from quant_platform.strategy_recipes import RECIPE_VERSION, get_strategy_recipe
from quant_platform.strategy_store import StrategyStore

AUTOPILOT_COMPLETION_CONTRACT_VERSION = "autopilot-completion-v1"
CHAMPION_SELECTION_POLICY_VERSION = "pre-final-incumbent-challenge-equal-profile-v2"
AUTOPILOT_PORTFOLIO_CONTRACT_VERSION = "autopilot-long-only-portfolio-v1"

_SELECTION_METRICS = (
    "annualized_excess_return_with_cost",
    "information_ratio",
    "rank_ic",
    "average_turnover",
    "max_drawdown",
)
_PORTFOLIO_OVERRIDE_FIELDS = frozenset(
    {
        "portfolio_construction",
        "topk",
        "n_drop",
        "max_position_weight",
        "max_daily_turnover",
        "max_industry_weight",
        "max_industry_deviation",
        "max_size_deviation",
        "max_value_deviation",
        "max_growth_deviation",
        "max_volatility_deviation",
        "optimizer_alpha_weight",
        "optimizer_tracking_penalty",
        "optimizer_turnover_penalty",
        "target_volatility",
        "max_tracking_error",
        "execution_method",
        "execution_frequency",
        "execution_days",
        "execution_slice_minutes",
        "max_execution_slices",
        "rebalance_frequency",
    }
)
_SUPPORTED_PORTFOLIO_CONSTRUCTION = frozenset(
    {"topk_equal_weight", "benchmark_relative_qp", "industry_neutral_qp"}
)


class _DatasetIdentityMismatch(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 digest") from exc
    return normalized


def _iso(value: Any, label: str) -> str:
    try:
        return (value if isinstance(value, date) else date.fromisoformat(str(value))).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"independent grid metric {name} is missing") from exc
    if not math.isfinite(value):
        raise ValueError(f"independent grid metric {name} is not finite")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty independent grid")
    return sum(values) / len(values)


def aggregate_pre_final_grid(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate profiles and seeds as robustness repeats, never hypotheses.

    Each profile receives equal weight. Seeds are first averaged inside their
    profile, so adding another stochastic repeat cannot silently give one
    market window more influence in champion selection.
    """

    expected = {
        (profile, seed)
        for profile in REQUIRED_RESEARCH_PROFILES
        for seed in REQUIRED_MODEL_SEEDS
    }
    normalized: dict[tuple[str, int], dict[str, float]] = {}
    for raw in rows:
        profile = str(raw.get("profile_id") or "")
        try:
            seed = int(raw.get("seed"))
        except (TypeError, ValueError) as exc:
            raise ValueError("independent grid seed is invalid") from exc
        key = (profile, seed)
        if key in normalized:
            raise ValueError("independent grid contains a duplicate profile/seed cell")
        if str(raw.get("evidence_role") or "") != "independent_gate":
            raise ValueError("champion selection may only use independent gate evidence")
        if str(raw.get("gate_status") or "") != "passed":
            raise ValueError("champion selection grid contains a failed gate")
        if raw.get("oos_vintage_id") is not None:
            raise ValueError("champion selection must not use final-OOS evidence")
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping):
            metrics = raw.get("metrics_json")
        if not isinstance(metrics, Mapping):
            raise ValueError("independent grid metrics are missing")
        recorded_sha256 = str(raw.get("metrics_sha256") or "")
        if recorded_sha256 and canonical_sha256(dict(metrics)) != recorded_sha256:
            raise ValueError("independent grid metrics hash is inconsistent")
        normalized[key] = {name: _finite_metric(metrics, name) for name in _SELECTION_METRICS}
    if set(normalized) != expected:
        missing = sorted(expected.difference(normalized))
        extra = sorted(set(normalized).difference(expected))
        raise ValueError(
            "independent grid is incomplete "
            f"(missing={missing}, unexpected={extra})"
        )

    profiles: dict[str, dict[str, float]] = {}
    for profile in REQUIRED_RESEARCH_PROFILES:
        profile_cells = [normalized[(profile, seed)] for seed in REQUIRED_MODEL_SEEDS]
        profiles[profile] = {
            metric: _mean([cell[metric] for cell in profile_cells])
            for metric in _SELECTION_METRICS
        }
    aggregate = {
        metric: _mean([profiles[profile][metric] for profile in REQUIRED_RESEARCH_PROFILES])
        for metric in _SELECTION_METRICS
    }
    annualized = "annualized_excess_return_with_cost"
    score_vector = [
        aggregate[annualized],
        min(profiles[profile][annualized] for profile in REQUIRED_RESEARCH_PROFILES),
        aggregate["information_ratio"],
        aggregate["rank_ic"],
        -aggregate["average_turnover"],
        -abs(aggregate["max_drawdown"]),
    ]
    return {
        "aggregation": "equal_profile_mean_of_seed_means",
        "profiles_are_robustness_repeats": True,
        "seeds_are_robustness_repeats": True,
        "profile_count": len(REQUIRED_RESEARCH_PROFILES),
        "seed_count_per_profile": len(REQUIRED_MODEL_SEEDS),
        "profiles": profiles,
        "aggregate": aggregate,
        "score_vector": score_vector,
    }


def compare_joint_to_frozen_incumbent(
    *, challenger_grid: Mapping[str, Any], incumbent_grid: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen three-window replacement rule.

    Recent performance must improve on both cost-after return and RankIC,
    balanced may not degrade on either measure, and robust must already have
    passed every independent profile/seed gate (which is guaranteed by
    ``aggregate_pre_final_grid`` accepting only passed cells).
    """

    metrics = ("annualized_excess_return_with_cost", "rank_ic")
    challenger_profiles = challenger_grid.get("profiles")
    incumbent_profiles = incumbent_grid.get("profiles")
    if not isinstance(challenger_profiles, Mapping) or not isinstance(
        incumbent_profiles, Mapping
    ):
        raise ValueError("incumbent replacement requires complete profile aggregates")
    if set(challenger_profiles) != set(REQUIRED_RESEARCH_PROFILES) or set(
        incumbent_profiles
    ) != set(REQUIRED_RESEARCH_PROFILES):
        raise ValueError("incumbent replacement profile grid is incomplete")

    recent_profile, balanced_profile, robust_profile = REQUIRED_RESEARCH_PROFILES

    def metric_evidence(profile: str, *, strict: bool) -> dict[str, Any]:
        values: dict[str, Any] = {}
        passed = True
        for metric in metrics:
            challenger = _finite_metric(challenger_profiles[profile], metric)
            incumbent = _finite_metric(incumbent_profiles[profile], metric)
            improved = challenger > incumbent if strict else challenger >= incumbent
            values[metric] = {
                "challenger": challenger,
                "incumbent": incumbent,
                "difference": challenger - incumbent,
                "passed": improved,
            }
            passed = passed and improved
        return {"metrics": values, "passed": passed}

    recent = metric_evidence(recent_profile, strict=True)
    balanced = metric_evidence(balanced_profile, strict=False)
    # The robust aggregate can only exist after all nine independent cells
    # pass their governed metric gate.  Record its values as immutable audit
    # evidence instead of inventing a second threshold here.
    robust = {
        "profile_id": robust_profile,
        "gate_passed": True,
        "metrics": {
            metric: _finite_metric(challenger_profiles[robust_profile], metric)
            for metric in metrics
        },
    }
    result = {
        "contract_version": "incumbent-replacement-three-window-v1",
        "recent_profile_id": recent_profile,
        "balanced_profile_id": balanced_profile,
        "robust_profile_id": robust_profile,
        "recent_strict_improvement": recent,
        "balanced_non_degradation": balanced,
        "robust_governed_gate": robust,
        "passed": bool(recent["passed"] and balanced["passed"]),
    }
    result["evidence_sha256"] = canonical_sha256(result)
    return result


def _frozen_quant_baseline_reference(bundle: Mapping[str, Any]) -> dict[str, str]:
    admission = bundle.get("admission_evidence_json")
    independent = (
        admission.get("independent_bundle")
        if isinstance(admission, Mapping)
        else None
    )
    manifest = bundle.get("bundle_manifest_json")
    baseline = (
        independent.get("baseline_prediction_champion")
        if isinstance(independent, Mapping)
        else None
    )
    if not isinstance(baseline, Mapping) and isinstance(manifest, Mapping):
        baseline = manifest.get("baseline_prediction_champion")
    if not isinstance(baseline, Mapping):
        baseline = bundle.get("baseline_prediction_champion")
    if not isinstance(baseline, Mapping):
        raise ValueError("joint bundle has no frozen incumbent prediction reference")
    kind = str(baseline.get("kind") or "")
    candidate_id = str(baseline.get("candidate_id") or "")
    if kind not in {"model", "ensemble"} or not candidate_id:
        raise ValueError("joint bundle frozen incumbent identity is invalid")
    return {"kind": kind, "candidate_id": candidate_id}


def build_long_only_strategy_config(
    *,
    signal_config: Mapping[str, Any],
    portfolio_config: Mapping[str, Any],
    selection_evidence_sha256: str,
) -> dict[str, Any]:
    """Build a complete governed config without importing the API module."""

    selection_sha = _sha256(selection_evidence_sha256, "selection evidence")
    unknown = set(portfolio_config).difference(_PORTFOLIO_OVERRIDE_FIELDS)
    if unknown:
        raise ValueError(f"unsupported autopilot portfolio fields: {sorted(unknown)}")

    recipe = get_strategy_recipe("full_market_multifactor")
    cost = CostModelConfig().to_dict()
    cost["cost_schedule_version"] = cost.pop("version")
    config: dict[str, Any] = {
        "recipe_id": "full_market_multifactor",
        "recipe_version": RECIPE_VERSION,
        "factor_source_mode": "not_applicable_model_prediction",
        "challenger_weight": 0.0,
        "signal_source": "model_prediction",
        "signal_frequency": "day",
        "signal_period": 1,
        "execution_frequency": "day",
        "execution_lag_bars": 1,
        "execution_method": "open",
        "execution_days": 1,
        "execution_slice_minutes": 20,
        "max_execution_slices": 24,
        "position_side": "long_only",
        "shorting_enabled": False,
        "margin_enabled": False,
        "financing_enabled": False,
        "broker_connection_enabled": False,
        "real_trading_eligible": False,
        "recommendation_enabled": False,
        "annual_borrow_rate": 0.0,
        "topk": 100,
        "n_drop": 10,
        "max_position_weight": 0.05,
        "max_daily_turnover": 0.15,
        "max_daily_loss": 0.03,
        "stop_loss": 0.07,
        "take_profit_partial": 0.12,
        "take_profit_partial_fraction": 0.50,
        "take_profit": 0.20,
        "max_drawdown_reduce": 0.10,
        "max_drawdown_liquidate": 0.15,
        "drawdown_reduction_exposure": 0.50,
        "max_industry_weight": 0.15,
        "max_industry_deviation": 0.03,
        "max_size_deviation": 0.03,
        "max_value_deviation": 0.03,
        "max_growth_deviation": 0.03,
        "max_volatility_deviation": 0.03,
        "portfolio_construction": "industry_neutral_qp",
        "optimizer_alpha_weight": 0.05,
        "optimizer_tracking_penalty": 1.0,
        "optimizer_turnover_penalty": 0.10,
        "min_average_daily_amount": 500_000_000,
        "liquidity_lookback_days": 20,
        "require_regulatory_events": False,
        "max_tracking_error": 0.12,
        "min_cash_weight": 0.0,
        "target_volatility": 0.15,
        "max_drawdown": 0.25,
        "max_turnover": 0.60,
        "min_information_ratio": 0.0,
        "min_sharpe_ratio": 0.0,
        "min_sortino_ratio": 0.0,
        "min_robustness_pass_rate": 1.0,
        "annual_minimum_acceptable_return": 0.0,
        "annual_cash_yield_rate": 0.0,
        "cash_yield_source": "none_zero_yield",
        "rolling_window_days": 252,
        "rolling_step_days": 63,
        "min_rolling_windows": 3,
        "min_rolling_pass_rate": 0.60,
        "outer_train_days": 252,
        "outer_validation_days": 42,
        "outer_test_days": 42,
        "outer_purge_days": 5,
        "outer_embargo_days": 20,
        "minimum_outer_test_excess_return": 0.0,
        "minimum_outer_test_pass_rate": 0.60,
        "min_pre_final_history_days": 2520,
        "event_window_days": 20,
        "event_count": 5,
        "max_event_underperformance": 0.05,
        "min_event_stress_pass_rate": 0.60,
        "min_backtest_days": 252,
        "paper_initial_cash": 5_000_000,
        "capacity_notional": 5_000_000,
        "capacity_curve_notionals": [5_000_000, 20_000_000, 100_000_000],
        "min_capacity_excess_return": 0.0,
        "min_closed_trades": 20,
        "min_win_rate": 0.0,
        "min_profit_loss_ratio": 0.0,
        "min_capacity_fill_ratio": 0.95,
        "vwap_lookback_days": 20,
        "autopilot_completion_contract_version": AUTOPILOT_COMPLETION_CONTRACT_VERSION,
        "autopilot_selection_policy_version": CHAMPION_SELECTION_POLICY_VERSION,
        "autopilot_selection_evidence_sha256": selection_sha,
        "autopilot_portfolio_contract_version": AUTOPILOT_PORTFOLIO_CONTRACT_VERSION,
        **cost,
        **dict(recipe["config_overrides"]),
        **dict(signal_config),
        **dict(portfolio_config),
    }
    # These capital-boundary values are invariants, not user-selectable knobs.
    config.update(
        {
            "factor_source_mode": "not_applicable_model_prediction",
            "challenger_weight": 0.0,
            "signal_source": "model_prediction",
            "position_side": "long_only",
            "shorting_enabled": False,
            "margin_enabled": False,
            "financing_enabled": False,
            "broker_connection_enabled": False,
            "real_trading_eligible": False,
            "recommendation_enabled": False,
            "annual_borrow_rate": 0.0,
            # Formal OOS, capacity evidence and the isolated paper ledger use
            # one notional.  A smaller paper account changes lot rounding and
            # minimum-commission economics, so it is not the same strategy.
            "capacity_notional": 5_000_000,
            "paper_initial_cash": 5_000_000,
            "capacity_curve_notionals": [
                5_000_000,
                20_000_000,
                100_000_000,
            ],
            "model_drift_policy": {
                "contract_version": "model-drift-policy-v1",
                "metric": "cost_after_excess_return",
                "window_trading_days": 20,
                "consecutive_windows": 3,
                "threshold": 0.0,
                "comparison": "below",
            },
        }
    )
    construction = str(config.get("portfolio_construction") or "")
    if construction not in _SUPPORTED_PORTFOLIO_CONSTRUCTION:
        raise ValueError("autopilot portfolio construction is not governed")
    topk = int(config.get("topk") or 0)
    n_drop = int(config.get("n_drop") or 0)
    max_position_weight = float(config.get("max_position_weight") or 0.0)
    if topk < 5 or n_drop < 0 or n_drop > topk:
        raise ValueError("autopilot TopK/NDrop configuration is invalid")
    if not 0 < max_position_weight <= 0.20:
        raise ValueError("autopilot position limit is invalid")
    if float(config.get("max_industry_weight") or 0.0) < max_position_weight:
        raise ValueError("industry limit must not be below the single-position limit")
    if construction != "topk_equal_weight" and topk * max_position_weight < 1.0:
        raise ValueError("QP portfolio position limits cannot form a fully invested portfolio")
    if float(config.get("take_profit_partial") or 0.0) >= float(
        config.get("take_profit") or 0.0
    ):
        raise ValueError("partial take-profit must be below final take-profit")
    if float(config.get("max_drawdown_reduce") or 0.0) >= float(
        config.get("max_drawdown_liquidate") or 0.0
    ):
        raise ValueError("drawdown reduction must precede liquidation")
    cost_model = CostModelConfig.from_mapping(config)
    capacity_notional = float(config.get("capacity_notional") or 0.0)
    paper_initial_cash = float(config.get("paper_initial_cash") or 0.0)
    if capacity_notional != 5_000_000.0 or paper_initial_cash != capacity_notional:
        raise ValueError(
            "paper initial cash must equal the frozen formal capacity notional"
        )
    capital_execution_contract = {
        "contract_version": "autopilot-capital-execution-v1",
        "portfolio_construction": construction,
        "capacity_notional": capacity_notional,
        "paper_initial_cash": paper_initial_cash,
        "topk": topk,
        "n_drop": n_drop,
        "lot_size": int(cost_model.lot_size),
        "min_commission": float(cost_model.min_commission),
        "cost_schedule_version": str(config.get("cost_schedule_version") or ""),
    }
    config["autopilot_capital_execution_contract"] = capital_execution_contract
    config["autopilot_capital_execution_contract_sha256"] = canonical_sha256(
        capital_execution_contract
    )
    config["model_drift_policy_sha256"] = canonical_sha256(
        config["model_drift_policy"]
    )
    config["autopilot_portfolio_config_sha256"] = canonical_sha256(
        {
            "contract_version": AUTOPILOT_PORTFOLIO_CONTRACT_VERSION,
            "portfolio_config": dict(portfolio_config),
        }
    )
    config = normalize_model_signal_config(config)
    config["execution_contract_hash"] = strategy_execution_contract_hash(config)
    require_strategy_execution_contract(config)
    return config


def portfolio_overrides_from_frozen(
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover only the preregistered portfolio fields from a frozen winner.

    ``ParameterExperimentStore`` stores the complete trial StrategySpec so the
    TopK/QP comparison can prove that the signal, costs and execution contract
    were identical.  The final strategy builder, however, accepts only the
    portfolio knobs that were explicitly open during that comparison.  This
    narrow conversion prevents a result artifact from smuggling a model,
    financing or final-OOS field into the capital-boundary StrategySpec.
    """

    config = frozen.get("portfolio_config")
    if not isinstance(config, Mapping):
        raise ValueError("frozen portfolio winner has no complete StrategySpec")
    recorded_sha256 = str(frozen.get("portfolio_config_sha256") or "")
    if canonical_sha256(dict(config)) != recorded_sha256:
        raise ValueError("frozen portfolio winner hash is inconsistent")
    if frozen.get("final_oos_opened") is not False:
        raise ValueError("portfolio selection must not consume final OOS")
    construction = str(frozen.get("portfolio_construction") or "")
    if construction not in _SUPPORTED_PORTFOLIO_CONSTRUCTION:
        raise ValueError("frozen portfolio construction is not governed")
    if str(config.get("portfolio_construction") or "") != construction:
        raise ValueError("frozen portfolio construction changed after selection")
    overrides = {
        key: value for key, value in config.items() if key in _PORTFOLIO_OVERRIDE_FIELDS
    }
    if overrides.get("portfolio_construction") != construction:
        raise ValueError("frozen portfolio winner omitted its construction")
    return overrides


class AutopilotCompletionService:
    """Recoverable capital-boundary service for the single Autopilot pipeline."""

    def __init__(
        self,
        database_url: str,
        *,
        candidate_store: RDAGentCandidateStore | None = None,
        strategy_store: StrategyStore | None = None,
        promotion_store: PromotionStore | None = None,
        tournament_store: ResearchTournamentStore | None = None,
    ) -> None:
        self.database_url = database_url
        self.candidates = candidate_store or RDAGentCandidateStore(database_url)
        self.strategies = strategy_store or StrategyStore(database_url)
        self.promotions = promotion_store or PromotionStore(database_url)
        self._tournament_store = tournament_store

    @property
    def tournaments(self) -> ResearchTournamentStore:
        # Existing completion tests and non-ensemble callers use isolated
        # in-memory fakes.  Do not open a new database-backed store until an
        # ensemble signal actually needs it.
        if self._tournament_store is None:
            self._tournament_store = ResearchTournamentStore(self.database_url)
        return self._tournament_store

    def _grid_rows(self, *, kind: str, candidate_id: str) -> list[dict[str, Any]]:
        if kind == "joint":
            table = quant_bundle_evaluations
            id_column = table.c.quant_bundle_candidate_id
            extra = table.c.ablation == "joint"
        elif kind == "model":
            table = model_evaluations
            id_column = table.c.model_candidate_id
            extra = table.c.id.is_not(None)
        elif kind == "ensemble":
            table = model_ensemble_evaluations
            id_column = table.c.model_ensemble_candidate_id
            extra = table.c.id.is_not(None)
        else:
            raise ValueError("unknown admitted strategy signal kind")
        statement = select(table).where(id_column == candidate_id, extra)
        if kind != "ensemble":
            statement = statement.where(
                table.c.evidence_role == "independent_gate",
                table.c.oos_vintage_id.is_(None),
            )
        with self.candidates.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        if kind != "ensemble":
            return [dict(row) for row in rows]
        ensemble = self.tournaments.get_ensemble(candidate_id, verify=True)
        admission = dict(ensemble.get("admission_evidence") or {})
        profiles = dict((admission.get("independent_evidence") or {}).get("profiles") or {})
        normalized: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            profile_id = str(value.get("profile_id") or "")
            periods = dict((profiles.get(profile_id) or {}).get("periods") or {})
            value.update(
                {
                    "evidence_role": "independent_gate",
                    "oos_vintage_id": None,
                    "metrics": dict(value.get("metrics_json") or {}),
                    "metrics_sha256": canonical_sha256(
                        dict(value.get("metrics_json") or {})
                    ),
                    "train_start": periods.get("train_start"),
                }
            )
            normalized.append(value)
        return normalized

    def _verified_signal(
        self,
        signal: Mapping[str, Any],
        *,
        dataset: str,
        dataset_identity_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        kind = str(signal.get("kind") or "")
        config = signal.get("strategy_config")
        if not isinstance(config, Mapping):
            raise ValueError("admitted signal has no DB-derived StrategySpec config")
        if kind == "ensemble":
            ensemble_id = str(config.get("model_ensemble_candidate_id") or "")
            ensemble = self.tournaments.get_ensemble(ensemble_id, verify=True)
            if str(ensemble.get("status")) != "research_admitted":
                raise ValueError("listed ensemble is no longer research-admitted")
            self._require_research_screening_admission(
                ensemble.get("admission_evidence"), context="ensemble"
            )
            ensemble_tournament = self.tournaments.get_tournament(
                str(ensemble.get("tournament_id") or "")
            )
            ensemble_trial = next(
                (
                    item
                    for item in ensemble_tournament.get("trials") or []
                    if str(item.get("candidate_id") or "") == ensemble_id
                    and item.get("trial_kind") == "model_ensemble"
                ),
                None,
            )
            if (
                ensemble_tournament.get("status") != "succeeded"
                or ensemble_trial is None
                or ensemble_trial.get("status") != "selected"
                or str(ensemble_trial.get("id") or "")
                not in set(ensemble_tournament.get("selected_trial_ids") or [])
            ):
                raise ValueError("ensemble has no completed research-tournament receipt")
            if (
                str(ensemble.get("dataset")) != dataset
                or str(ensemble.get("dataset_identity_sha256"))
                != dataset_identity_sha256
            ):
                raise _DatasetIdentityMismatch(
                    "admitted ensemble does not match the requested dataset identity"
                )
            component_ids = [
                str(item.get("model_candidate_id") or "")
                for item in ensemble.get("components") or []
            ]
            component_families = [
                str(item.get("model_family") or "")
                for item in ensemble.get("components") or []
            ]
            primary_evaluation = next(
                (
                    item
                    for item in ensemble.get("evaluations") or []
                    if str(item.get("profile_id")) == REQUIRED_RESEARCH_PROFILES[0]
                    and int(item.get("seed") or 0) == REQUIRED_MODEL_SEEDS[0]
                    and str(item.get("gate_status")) == "passed"
                ),
                None,
            )
            if (
                not 2 <= len(component_ids) <= 3
                or len(set(component_ids)) != len(component_ids)
                or len(set(component_families)) != len(component_families)
                or component_ids
                != [str(item) for item in config.get("model_component_candidate_ids") or []]
                or component_families
                != [str(item) for item in config.get("model_component_families") or []]
                or config.get("model_ensemble_manifest_sha256")
                != ensemble.get("manifest_sha256")
                or config.get("model_ensemble_evidence_sha256")
                != ensemble.get("admission_evidence_sha256")
                or primary_evaluation is None
                or config.get("model_ensemble_evaluation_id")
                != primary_evaluation.get("id")
                or config.get("model_ensemble_combiner") != "equal_rank"
                or config.get("model_ensemble_stacking") is not False
            ):
                raise ValueError("listed ensemble StrategySpec identity changed")
            models = [
                self.candidates.get_model_candidate(candidate_id, verify=True)
                for candidate_id in component_ids
            ]
            if any(
                str(model.get("status")) != "research_admitted"
                or str(model.get("dataset")) != dataset
                or str(model.get("dataset_identity_sha256"))
                != dataset_identity_sha256
                or not self._is_research_screening_admission(
                    model.get("admission_evidence_json")
                )
                for model in models
            ):
                raise ValueError("ensemble component is no longer independently admitted")
            periods = {
                (
                    _iso(model.get("pre_final_end"), "pre-final end"),
                    _iso(model.get("final_oos_start"), "final OOS start"),
                    _iso(model.get("final_oos_end"), "final OOS end"),
                )
                for model in models
            }
            if len(periods) != 1:
                raise ValueError("ensemble components do not share one sealed OOS contract")
            primary = dict(models[0])
            primary["component_models"] = [dict(model) for model in models]
            return primary, dict(ensemble)
        model_id = str(config.get("model_candidate_id") or "")
        model = self.candidates.get_model_candidate(model_id, verify=True)
        if str(model.get("status")) != "research_admitted":
            raise ValueError("listed model signal is no longer research-admitted")
        self._require_research_screening_admission(
            model.get("admission_evidence_json"), context="model"
        )
        if (
            str(model.get("dataset")) != dataset
            or str(model.get("dataset_identity_sha256")) != dataset_identity_sha256
        ):
            raise _DatasetIdentityMismatch(
                "admitted signal does not match the requested dataset identity"
            )
        selected = model
        if kind == "joint":
            bundle_id = str(config.get("quant_bundle_candidate_id") or "")
            bundle = self.candidates.get_quant_bundle_candidate(bundle_id, verify=True)
            if str(bundle.get("status")) != "research_admitted":
                raise ValueError("listed joint bundle is no longer research-admitted")
            bundle_admission = self._require_research_screening_admission(
                bundle.get("admission_evidence_json"), context="fin_quant"
            )
            self._verify_quant_research_ledger_receipt(
                candidate_id=bundle_id,
                admission=bundle_admission,
            )
            if (
                str(bundle.get("dataset")) != dataset
                or str(bundle.get("dataset_identity_sha256")) != dataset_identity_sha256
            ):
                raise _DatasetIdentityMismatch(
                    "joint bundle does not match the requested dataset identity"
                )
            if str(bundle.get("model_candidate_id")) != model_id:
                raise ValueError("joint bundle changed its frozen model component")
            selected = bundle
        elif kind != "model":
            raise ValueError("admitted signal kind is not capital-governed")
        return dict(model), dict(selected)

    @staticmethod
    def _is_research_screening_admission(value: Any) -> bool:
        return isinstance(value, Mapping) and (
            value.get("research_screening_only") is True
            and value.get("not_capital_confirmation") is True
            and value.get("cross_cycle_fwer_claimed") is False
            and value.get("final_oos_opened") is False
        )

    @classmethod
    def _require_research_screening_admission(
        cls, value: Any, *, context: str
    ) -> dict[str, Any]:
        if not cls._is_research_screening_admission(value):
            raise ValueError(
                f"{context} admission does not declare its research-only scope"
            )
        return dict(value)

    def _verify_quant_research_ledger_receipt(
        self,
        *,
        candidate_id: str,
        admission: Mapping[str, Any],
    ) -> None:
        receipt = dict(admission.get("research_trial_ledger_receipt") or {})
        receipt_sha = canonical_sha256(
            {key: value for key, value in receipt.items() if key != "evidence_sha256"}
        )
        tournament_id = str(receipt.get("research_tournament_id") or "")
        trial_id = str(
            dict(receipt.get("research_trial_ids") or {}).get(candidate_id) or ""
        )
        if (
            receipt.get("contract_version")
            != "fin-quant-research-ledger-receipt-v1"
            or receipt.get("evidence_sha256") != receipt_sha
            or admission.get("research_trial_ledger_receipt_sha256") != receipt_sha
            or dict(receipt.get("candidate_statuses") or {}).get(candidate_id)
            != "passed"
            or receipt.get("research_screening_only") is not True
            or receipt.get("not_capital_confirmation") is not True
            or receipt.get("cross_cycle_fwer_claimed") is not False
            or receipt.get("final_oos_opened") is not False
            or not tournament_id
            or not trial_id
        ):
            raise ValueError("fin_quant admission has no valid research-ledger receipt")
        tournament = self.tournaments.get_tournament(tournament_id)
        trial = next(
            (
                item
                for item in tournament.get("trials") or []
                if str(item.get("id") or "") == trial_id
                and str(item.get("candidate_id") or "") == candidate_id
            ),
            None,
        )
        screening = dict(tournament.get("multiple_testing") or {})
        if (
            tournament.get("stage") != "quant"
            or tournament.get("status") != "succeeded"
            or tournament.get("manifest_sha256")
            != receipt.get("research_tournament_manifest_sha256")
            or screening.get("research_trial_ledger_receipt_sha256") != receipt_sha
            or screening.get("research_screening_only") is not True
            or screening.get("not_capital_confirmation") is not True
            or trial is None
            or trial.get("status") != "passed"
            or trial_id not in set(tournament.get("selected_trial_ids") or [])
        ):
            raise ValueError("fin_quant research-ledger settlement is missing or changed")

    def select_champion(
        self,
        *,
        dataset: str,
        dataset_identity_sha256: str,
        allowed_candidate_ids: set[str] | frozenset[str] | None = None,
    ) -> dict[str, Any]:
        identity = _sha256(dataset_identity_sha256, "dataset identity")
        dataset_name = str(dataset or "").strip()
        if not dataset_name:
            raise ValueError("dataset is required")
        eligible: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for signal in self.candidates.list_admitted_strategy_signals(limit=500):
            kind = str(signal.get("kind") or "")
            candidate_id = str(signal.get("id") or "")
            key = (kind, candidate_id)
            if (
                not candidate_id
                or key in seen
            ):
                continue
            seen.add(key)
            try:
                model, selected = self._verified_signal(
                    signal,
                    dataset=dataset_name,
                    dataset_identity_sha256=identity,
                )
                grid_rows = self._grid_rows(kind=kind, candidate_id=candidate_id)
                grid = aggregate_pre_final_grid(grid_rows)
                primary_rows = [
                    row
                    for row in grid_rows
                    if str(row.get("profile_id") or "") == PRIMARY_MODEL_PROFILE
                    and int(row.get("seed")) == PRIMARY_MODEL_SEED
                ]
                if len(primary_rows) != 1:
                    raise ValueError(
                        "admitted signal has no unique primary validation cell"
                    )
                primary = primary_rows[0]
                baseline_reference = (
                    _frozen_quant_baseline_reference(selected)
                    if kind == "joint"
                    else None
                )
            except _DatasetIdentityMismatch:
                # Another immutable snapshot is outside this tournament. A
                # corrupt artifact or incomplete admitted grid instead fails
                # the whole selection closed; it must never disappear from
                # the comparison merely because it is inconvenient.
                continue
            eligible.append(
                {
                    "kind": kind,
                    "candidate_id": candidate_id,
                    "model_candidate_id": str(model["id"]),
                    "model_component_candidate_ids": [
                        str(item["id"])
                        for item in model.get("component_models") or [model]
                    ],
                    "name": str(signal.get("name") or candidate_id),
                    "strategy_config": dict(signal["strategy_config"]),
                    "candidate_manifest_sha256": str(
                        selected.get("bundle_manifest_sha256")
                        or selected.get("manifest_sha256")
                        or ""
                    ),
                    "model_manifest_sha256": str(model.get("manifest_sha256") or ""),
                    "pre_final_end": _iso(model.get("pre_final_end"), "pre-final end"),
                    "final_oos_start": _iso(
                        model.get("final_oos_start"), "final OOS start"
                    ),
                    "final_oos_end": _iso(model.get("final_oos_end"), "final OOS end"),
                    "historical_start": min(
                        _iso(row.get("train_start"), "grid train start")
                        for row in grid_rows
                    ),
                    "portfolio_validation_start": _iso(
                        primary.get("valid_start"), "primary validation start"
                    ),
                    "portfolio_validation_end": _iso(
                        primary.get("valid_end"), "primary validation end"
                    ),
                    "grid": grid,
                    "baseline_reference": baseline_reference,
                }
            )
        if not eligible:
            raise ValueError("no independently admitted signal matches this dataset identity")

        scoped = list(eligible)
        if allowed_candidate_ids is not None:
            allowed = {str(value) for value in allowed_candidate_ids}
            explicitly_allowed = [
                item for item in eligible if item["candidate_id"] in allowed
            ]
            missing = allowed.difference(
                {str(item["candidate_id"]) for item in explicitly_allowed}
            )
            if missing:
                raise ValueError(
                    "the current Autopilot candidate set has no admitted evidence for "
                    f"{sorted(missing)}"
                )
            baseline_keys = {
                (
                    str(item["baseline_reference"]["kind"]),
                    str(item["baseline_reference"]["candidate_id"]),
                )
                for item in explicitly_allowed
                if item["kind"] == "joint"
            }
            scoped = explicitly_allowed + [
                item
                for item in eligible
                if (str(item["kind"]), str(item["candidate_id"])) in baseline_keys
                and item not in explicitly_allowed
            ]
        else:
            joint_references = {
                (
                    str(item["baseline_reference"]["kind"]),
                    str(item["baseline_reference"]["candidate_id"]),
                )
                for item in eligible
                if item["kind"] == "joint"
            }
            if joint_references:
                scoped = [
                    item
                    for item in eligible
                    if item["kind"] == "joint"
                    or (str(item["kind"]), str(item["candidate_id"]))
                    in joint_references
                ]

        scoped_by_key = {
            (str(item["kind"]), str(item["candidate_id"])): item for item in scoped
        }
        joint = [item for item in scoped if item["kind"] == "joint"]
        replacement_decisions: list[dict[str, Any]] = []
        capital_pool: dict[tuple[str, str], dict[str, Any]] = {}
        for challenger in joint:
            reference = dict(challenger["baseline_reference"] or {})
            baseline_key = (
                str(reference.get("kind") or ""),
                str(reference.get("candidate_id") or ""),
            )
            incumbent = scoped_by_key.get(baseline_key)
            if incumbent is None:
                raise ValueError(
                    "the frozen fin_quant incumbent is missing from the admitted "
                    "comparison pool"
                )
            capital_pool[baseline_key] = incumbent
            decision = compare_joint_to_frozen_incumbent(
                challenger_grid=challenger["grid"], incumbent_grid=incumbent["grid"]
            )
            replacement_decisions.append(
                {
                    "challenger_candidate_id": challenger["candidate_id"],
                    "incumbent_kind": incumbent["kind"],
                    "incumbent_candidate_id": incumbent["candidate_id"],
                    "evidence": decision,
                }
            )
            if decision["passed"] is True:
                capital_pool[
                    (str(challenger["kind"]), str(challenger["candidate_id"]))
                ] = challenger
        if not joint:
            capital_pool = {
                (str(item["kind"]), str(item["candidate_id"])): item
                for item in scoped
                if item["kind"] in {"model", "ensemble"}
            }
        pool = list(capital_pool.values())
        if not pool:
            raise ValueError("no frozen incumbent or qualified challenger is available")
        ranked = sorted(
            pool,
            key=lambda item: (
                *[-float(value) for value in item["grid"]["score_vector"]],
                str(item["candidate_id"]),
            ),
        )
        winner = ranked[0]
        evidence = {
            "contract_version": AUTOPILOT_COMPLETION_CONTRACT_VERSION,
            "selection_policy_version": CHAMPION_SELECTION_POLICY_VERSION,
            "dataset": dataset_name,
            "dataset_identity_sha256": identity,
            "final_oos_opened": False,
            "candidate_preference": (
                "frozen_incumbent_plus_three_window-qualified-joint-challengers"
            ),
            "score_order": [
                "mean_annualized_excess_return_with_cost",
                "worst_profile_annualized_excess_return_with_cost",
                "mean_information_ratio",
                "mean_rank_ic",
                "negative_mean_turnover",
                "negative_absolute_mean_drawdown",
            ],
            "eligible_candidates": sorted(
                scoped, key=lambda item: (str(item["kind"]), str(item["candidate_id"]))
            ),
            "replacement_decisions": sorted(
                replacement_decisions,
                key=lambda item: str(item["challenger_candidate_id"]),
            ),
            "capital_pool_candidate_ids": [
                str(item["candidate_id"])
                for item in sorted(
                    pool,
                    key=lambda item: (str(item["kind"]), str(item["candidate_id"])),
                )
            ],
            "selected_kind": winner["kind"],
            "selected_candidate_id": winner["candidate_id"],
            "selected_model_candidate_id": winner["model_candidate_id"],
            "selected_model_component_candidate_ids": winner[
                "model_component_candidate_ids"
            ],
            "selected_strategy_config": winner["strategy_config"],
            "selected_periods": {
                "historical_start": winner["historical_start"],
                "historical_end": winner["pre_final_end"],
                "start": winner["final_oos_start"],
                "end": winner["final_oos_end"],
            },
            "selected_portfolio_validation": {
                "start": winner["portfolio_validation_start"],
                "end": winner["portfolio_validation_end"],
            },
            "research_screening_only": True,
            "not_capital_confirmation": True,
            "cross_cycle_fwer_claimed": False,
        }
        return {
            "champion_selection_evidence": evidence,
            "champion_selection_evidence_sha256": canonical_sha256(evidence),
        }

    def ensure_selection(
        self,
        *,
        state: Mapping[str, Any] | None,
        dataset: str,
        dataset_identity_sha256: str,
        allowed_candidate_ids: set[str] | frozenset[str] | None = None,
    ) -> dict[str, Any]:
        result = dict(state or {})
        existing = result.get("champion_selection_evidence")
        existing_sha = result.get("champion_selection_evidence_sha256")
        if existing is not None or existing_sha is not None:
            if (
                not isinstance(existing, Mapping)
                or canonical_sha256(dict(existing)) != existing_sha
            ):
                raise ValueError("frozen champion selection evidence is inconsistent")
            if (
                existing.get("dataset") != dataset
                or existing.get("dataset_identity_sha256")
                != _sha256(dataset_identity_sha256, "dataset identity")
            ):
                raise ValueError("frozen champion belongs to another dataset identity")
            if (
                existing.get("contract_version")
                != AUTOPILOT_COMPLETION_CONTRACT_VERSION
                or existing.get("selection_policy_version")
                != CHAMPION_SELECTION_POLICY_VERSION
                or existing.get("research_screening_only") is not True
                or existing.get("not_capital_confirmation") is not True
                or existing.get("cross_cycle_fwer_claimed") is not False
                or existing.get("final_oos_opened") is not False
            ):
                raise ValueError("frozen champion used a legacy selection policy")
            if allowed_candidate_ids is not None:
                allowed = {str(value) for value in allowed_candidate_ids}
                selected_id = str(existing.get("selected_candidate_id") or "")
                selected_kind = str(existing.get("selected_kind") or "")
                retained_incumbent = any(
                    str(item.get("challenger_candidate_id") or "") in allowed
                    and str(item.get("incumbent_candidate_id") or "") == selected_id
                    and str(item.get("incumbent_kind") or "") == selected_kind
                    for item in existing.get("replacement_decisions") or []
                    if isinstance(item, Mapping)
                )
                if selected_id not in allowed and not retained_incumbent:
                    raise ValueError(
                        "frozen champion is outside this Autopilot candidate set"
                    )
            # Verify the winner still exists and is admitted. Never select a
            # runner-up after the final choice has been frozen.
            self._verified_signal(
                {
                    "kind": existing.get("selected_kind"),
                    "strategy_config": existing.get("selected_strategy_config"),
                },
                dataset=dataset,
                dataset_identity_sha256=str(existing["dataset_identity_sha256"]),
            )
            return result
        result.update(
            self.select_champion(
                dataset=dataset,
                dataset_identity_sha256=dataset_identity_sha256,
                allowed_candidate_ids=allowed_candidate_ids,
            )
        )
        result["phase"] = "champion_frozen"
        return result

    def _verify_version_binding(
        self,
        *,
        version: Mapping[str, Any],
        state: Mapping[str, Any],
        expected_config: Mapping[str, Any],
    ) -> str:
        config = version.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("autopilot strategy version has no immutable config")
        authoritative_additions = {
            "quant_bundle_factor_contract",
            "quant_bundle_factor_contract_sha256",
        }
        unexpected = set(config).difference(expected_config).difference(authoritative_additions)
        if (
            unexpected
            or any(config.get(key) != value for key, value in expected_config.items())
            or config.get("autopilot_selection_evidence_sha256")
            != state.get("champion_selection_evidence_sha256")
            or config.get("position_side") != "long_only"
            or bool(config.get("shorting_enabled"))
            or bool(config.get("financing_enabled"))
            or float(config.get("annual_borrow_rate") or 0.0) != 0.0
        ):
            raise ValueError("existing strategy version does not match the frozen champion")
        recorded_sha = canonical_sha256(dict(config))
        state_sha = state.get("strategy_config_sha256")
        if state_sha is not None and state_sha != recorded_sha:
            raise ValueError("frozen strategy config hash changed")
        return recorded_sha

    def ensure_strategy(
        self,
        *,
        state: Mapping[str, Any],
        portfolio_config: Mapping[str, Any],
        actor: str,
        benchmark: str = "SH000300",
        universe: str = "cn_all",
    ) -> dict[str, Any]:
        result = dict(state)
        selection = result.get("champion_selection_evidence")
        selection_sha = result.get("champion_selection_evidence_sha256")
        if not isinstance(selection, Mapping) or canonical_sha256(dict(selection)) != selection_sha:
            raise ValueError("a frozen champion selection is required")
        signal_config = selection.get("selected_strategy_config")
        if not isinstance(signal_config, Mapping):
            raise ValueError("frozen champion has no DB-derived StrategySpec config")
        config = build_long_only_strategy_config(
            signal_config=signal_config,
            portfolio_config=portfolio_config,
            selection_evidence_sha256=str(selection_sha),
        )
        existing_version_id = str(result.get("strategy_version_id") or "")
        if existing_version_id:
            version = self.strategies.get_version(existing_version_id)
            config_sha = self._verify_version_binding(
                version=version, state=result, expected_config=config
            )
            result["strategy_id"] = str(version["strategy_id"])
            result["strategy_config_sha256"] = config_sha
            return result

        name = (
            f"autopilot-{str(selection_sha)[:24]}-"
            f"{config['autopilot_portfolio_config_sha256'][:12]}"
        )
        existing = self.strategies.get_by_name(name)
        if existing is None:
            try:
                existing = self.strategies.create(
                    name=name,
                    description=(
                        "Autopilot frozen champion; governed long-only strategy for one "
                        "formal OOS test and isolated paper simulation."
                    ),
                    benchmark=benchmark,
                    universe=universe,
                    factors=[],
                    config=config,
                    actor=actor,
                    economic_hypothesis_group=(
                        f"autopilot:{selection['dataset_identity_sha256']}"
                    ),
                    hypothesis_group_cap=0.70,
                )
            except ValueError:
                # A concurrent retry may have committed the deterministic
                # family after our read. Only that exact family is recoverable.
                existing = self.strategies.get_by_name(name)
                if existing is None:
                    raise
        versions = list(existing.get("versions") or [])
        if len(versions) != 1:
            raise ValueError("autopilot strategy family must contain exactly one frozen version")
        version = versions[0]
        config_sha = self._verify_version_binding(
            version=version, state=result, expected_config=config
        )
        result.update(
            {
                "strategy_id": str(existing["id"]),
                "strategy_version_id": str(version["id"]),
                "strategy_config_sha256": config_sha,
                "phase": "strategy_frozen",
            }
        )
        return result

    def ensure_formal_backtest(
        self,
        *,
        state: Mapping[str, Any],
        artifact_path: Path,
        execution_dataset: str | None,
        trading_dates: Sequence[date | str],
        dataset_lineage_id: str,
    ) -> dict[str, Any]:
        result = dict(state)
        selection = result.get("champion_selection_evidence")
        version_id = str(result.get("strategy_version_id") or "")
        if not isinstance(selection, Mapping) or not version_id:
            raise ValueError("formal OOS requires a frozen champion and strategy version")
        periods = selection.get("selected_periods")
        if not isinstance(periods, Mapping):
            raise ValueError("frozen champion has no formal periods")
        expected_periods = {key: str(periods[key]) for key in ("start", "end")}
        expected_periods.update(
            {
                "historical_start": str(periods["historical_start"]),
                "historical_end": str(periods["historical_end"]),
            }
        )
        if not dataset_lineage_id.strip():
            raise ValueError("formal OOS requires a dataset lineage id")
        capital_link = result.get("capital_oos_vintage_link")
        capital_dataset_identity = str(
            result.get("capital_oos_dataset_identity_sha256") or ""
        )
        if not isinstance(capital_link, Mapping):
            raise ValueError("formal OOS requires an immutable capital alpha reservation")
        capital_batch_id = str(capital_link.get("capital_oos_alpha_batch_id") or "")
        sealed_patch = capital_link.get("sealed_candidate_set_patch")
        if (
            not capital_batch_id
            or not isinstance(sealed_patch, Mapping)
            or len(capital_dataset_identity) != 64
        ):
            raise ValueError("capital alpha reservation link is invalid")
        calendar = list(trading_dates)
        if not calendar:
            raise ValueError("formal OOS requires the exact Qlib trading calendar")
        version = self.strategies.get_version(version_id)
        method = str((version.get("config") or {}).get("execution_method") or "open")
        if method in {"twap", "vwap", "next_bar"} and not str(
            execution_dataset or ""
        ).strip():
            raise ValueError("minute execution strategy requires an execution dataset")

        existing_id = str(result.get("formal_backtest_id") or "")
        if existing_id:
            backtest = self.strategies.get_backtest(existing_id)
            self._verify_backtest_binding(
                backtest=backtest,
                version_id=version_id,
                dataset=str(selection["dataset"]),
                periods=expected_periods,
                execution_dataset=execution_dataset,
            )
            return self._backtest_state(result, backtest)
        existing_runs = self.strategies.list_backtests(version_id=version_id, limit=2)
        if len(existing_runs) > 1:
            raise ValueError("frozen strategy version has more than one formal backtest")
        if existing_runs:
            backtest = existing_runs[0]
            self._verify_backtest_binding(
                backtest=backtest,
                version_id=version_id,
                dataset=str(selection["dataset"]),
                periods=expected_periods,
                execution_dataset=execution_dataset,
            )
            return self._backtest_state(result, backtest)
        backtest = self.strategies.create_backtest(
            version_id=version_id,
            dataset=str(selection["dataset"]),
            periods=expected_periods,
            artifact_path=artifact_path,
            execution_dataset=execution_dataset,
            trading_dates=calendar,
            dataset_lineage_id=dataset_lineage_id,
            capital_oos_alpha_batch_id=capital_batch_id,
            capital_oos_sealed_candidate_set_patch=dict(sealed_patch),
            capital_oos_dataset_identity_sha256=capital_dataset_identity,
        )
        return self._backtest_state(result, backtest)

    @staticmethod
    def _verify_backtest_binding(
        *,
        backtest: Mapping[str, Any],
        version_id: str,
        dataset: str,
        periods: Mapping[str, str],
        execution_dataset: str | None,
    ) -> None:
        recorded = backtest.get("periods")
        if (
            str(backtest.get("strategy_version_id")) != version_id
            or str(backtest.get("dataset")) != dataset
            or str(backtest.get("execution_dataset") or "")
            != str(execution_dataset or "")
            or not isinstance(recorded, Mapping)
            or any(str(recorded.get(key)) != value for key, value in periods.items())
        ):
            raise ValueError("existing formal backtest does not match the frozen champion")

    @staticmethod
    def _backtest_state(state: Mapping[str, Any], backtest: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(state)
        status = str(backtest.get("status") or "")
        result.update(
            {
                "formal_backtest_id": str(backtest["id"]),
                "formal_backtest_status": status,
                "formal_execution_dataset": backtest.get("execution_dataset"),
                "final_oos_consumed": True,
                "phase": (
                    "formal_backtest_failed"
                    if status in {"failed", "cancelled"}
                    else "formal_backtest_succeeded"
                    if status == "succeeded"
                    else "formal_backtest_running"
                ),
                "terminal": status in {"failed", "cancelled"},
            }
        )
        return result

    def approve_if_ready(
        self,
        *,
        state: Mapping[str, Any],
        actor: str,
        forward_thresholds: ForwardGateThresholds | None = None,
    ) -> dict[str, Any]:
        result = dict(state)
        thresholds = self._require_autopilot_forward_thresholds(
            forward_thresholds or self._forward_thresholds()
        )
        forward_gate_state = {
            "min_forward_calendar_days": thresholds.min_forward_calendar_days,
            "min_decision_batches": thresholds.min_decision_batches,
        }
        version_id = str(result.get("strategy_version_id") or "")
        backtest_id = str(result.get("formal_backtest_id") or "")
        if not version_id or not backtest_id:
            raise ValueError("approval requires a frozen strategy and formal backtest")
        selection = result.get("champion_selection_evidence")
        if not isinstance(selection, Mapping):
            raise ValueError("approval requires frozen champion evidence")
        periods = selection.get("selected_periods")
        if not isinstance(periods, Mapping):
            raise ValueError("approval requires frozen formal periods")
        expected_periods = {
            key: str(periods[key])
            for key in ("start", "end", "historical_start", "historical_end")
        }
        backtest = self.strategies.get_backtest(backtest_id)
        self._verify_backtest_binding(
            backtest=backtest,
            version_id=version_id,
            dataset=str(selection.get("dataset") or ""),
            periods=expected_periods,
            execution_dataset=(
                str(result.get("formal_execution_dataset") or "") or None
            ),
        )
        version = self.strategies.get_version(version_id)
        if str(version.get("status")) == "approved":
            if str(backtest.get("status") or "") != "succeeded":
                raise ValueError(
                    "approved strategy recovery requires its succeeded formal OOS"
                )
            self.promotions.register_forward_gate(
                version_id,
                actor=actor,
                thresholds=thresholds,
            )
            stage = self.promotions.prepare_paper_stage(version_id, actor=actor)
            result.update(
                {
                    "phase": "paper",
                    "paper_stage": stage,
                    "paper_stage_opened": True,
                    "forward_gate": forward_gate_state,
                    "recommendation_enabled": False,
                    "terminal": False,
                }
            )
            return result
        status = str(backtest.get("status") or "")
        if status in {"failed", "cancelled"}:
            result.update(
                {
                    "formal_backtest_status": status,
                    "phase": "formal_backtest_failed",
                    "terminal": True,
                    "runner_up_allowed": False,
                }
            )
            return result
        if status != "succeeded":
            result.update(
                {
                    "formal_backtest_status": status,
                    "phase": "formal_backtest_running",
                    "terminal": False,
                }
            )
            return result
        self.promotions.register_forward_gate(
            version_id,
            actor=actor,
            thresholds=thresholds,
        )
        approved = self.strategies.approve(
            version_id,
            actor=actor,
            reason=(
                "Autopilot formal OOS and all immutable cost, PIT, statistics, "
                "risk and execution gates passed."
            ),
        )
        stage = self.promotions.prepare_paper_stage(version_id, actor=actor)
        result.update(
            {
                "formal_backtest_status": "succeeded",
                "strategy_status": str(approved.get("status") or "approved"),
                "phase": "paper",
                "paper_stage": stage,
                "paper_stage_opened": True,
                "forward_gate": forward_gate_state,
                "recommendation_enabled": False,
                "terminal": False,
            }
        )
        return result

    @staticmethod
    def _forward_thresholds(
        *,
        min_forward_calendar_days: int = 183,
        min_decision_batches: int = 126,
    ) -> ForwardGateThresholds:
        return ForwardGateThresholds(
            min_forward_calendar_days=min_forward_calendar_days,
            min_decision_batches=min_decision_batches,
            min_completed_cycles=0,
            min_data_completeness=0.95,
            min_reconciliation_rate=1.0,
            max_cost_deviation=0.005,
        )

    @staticmethod
    def _require_autopilot_forward_thresholds(
        thresholds: ForwardGateThresholds,
    ) -> ForwardGateThresholds:
        """Never let a caller weaken the six-month paper boundary."""

        if thresholds.min_forward_calendar_days < 183:
            raise ValueError(
                "autopilot paper gate requires at least 183 calendar days"
            )
        if thresholds.min_decision_batches < 126:
            raise ValueError(
                "autopilot paper gate requires at least 126 decision batches"
            )
        return thresholds

    def advance(
        self,
        *,
        state: Mapping[str, Any] | None,
        dataset: str,
        dataset_identity_sha256: str,
        portfolio_config: Mapping[str, Any],
        actor: str,
        artifact_path: Path,
        execution_dataset: str | None,
        trading_dates: Sequence[date | str],
        dataset_lineage_id: str,
        benchmark: str = "SH000300",
        universe: str = "cn_all",
        forward_thresholds: ForwardGateThresholds | None = None,
    ) -> dict[str, Any]:
        """Advance the frozen champion without ever choosing a runner-up."""

        result = self.ensure_selection(
            state=state,
            dataset=dataset,
            dataset_identity_sha256=dataset_identity_sha256,
        )
        result = self.ensure_strategy(
            state=result,
            portfolio_config=portfolio_config,
            actor=actor,
            benchmark=benchmark,
            universe=universe,
        )
        result = self.ensure_formal_backtest(
            state=result,
            artifact_path=artifact_path,
            execution_dataset=execution_dataset,
            trading_dates=trading_dates,
            dataset_lineage_id=dataset_lineage_id,
        )
        return self.approve_if_ready(
            state=result,
            actor=actor,
            forward_thresholds=forward_thresholds,
        )
