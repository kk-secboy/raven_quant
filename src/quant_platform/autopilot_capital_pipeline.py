from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import select

from quant_data.config import Settings
from quant_data.database import (
    open_database,
    platform_config_revisions,
    quant_bundle_candidates,
)
from quant_data.snapshot_lineage import canonical_sha256

from .alpha_spending_integration import capital_oos_family_manifest
from .alpha_spending_ledger import CapitalOOSAlphaLedgerStore
from .autopilot_completion import (
    AutopilotCompletionService,
    portfolio_overrides_from_frozen,
)
from .cost_model import CostModelConfig
from .job_store import JobStore
from .ops_calendar import load_calendar_days
from .parameter_experiment_store import ParameterExperimentStore
from .promotion import ForwardGateThresholds
from .services import list_qlib_datasets
from .strategy_store import StrategyStore

CAPITAL_PIPELINE_STATE_KEY = "capital_pipeline"
CAPITAL_PIPELINE_CONTRACT_VERSION = "autopilot-capital-pipeline-v1"


class AutopilotCapitalBlocked(ValueError):
    """A governed terminal result that must not fall through to a runner-up."""


class AutopilotCapitalWaiting(ValueError):
    """A non-terminal serialization wait for another capital OOS batch."""


@dataclass(frozen=True)
class CapitalPipelineProgress:
    state: dict[str, Any]
    stage: str
    created_jobs: int = 0
    complete: bool = False


def _selection_state(state: Mapping[str, Any]) -> dict[str, Any]:
    evidence = state.get("champion_selection_evidence")
    evidence_sha256 = state.get("champion_selection_evidence_sha256")
    if not isinstance(evidence, Mapping) or not evidence_sha256:
        raise AutopilotCapitalBlocked("capital pipeline has no frozen champion")
    return {
        "champion_selection_evidence": dict(evidence),
        "champion_selection_evidence_sha256": str(evidence_sha256),
    }


class AutopilotCapitalPipeline:
    """Advance the one-way pre-final portfolio -> OOS -> paper boundary.

    Research retries may produce many immutable candidates.  Once this service
    freezes one champion, every later operation is idempotently bound to that
    evidence hash.  A failed portfolio gate or formal OOS is terminal for the
    cycle: this class deliberately has no runner-up branch.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        completion: AutopilotCompletionService | None = None,
        experiments: ParameterExperimentStore | None = None,
        jobs: JobStore | None = None,
        strategies: StrategyStore | None = None,
        capital_ledger: CapitalOOSAlphaLedgerStore | None = None,
    ) -> None:
        self.settings = settings
        self.completion = completion or AutopilotCompletionService(
            settings.database_url
        )
        self.experiments = experiments or ParameterExperimentStore(
            settings.database_url
        )
        self.jobs = jobs or JobStore(settings.database_url)
        self.strategies = strategies or StrategyStore(settings.database_url)
        self.capital_ledger = capital_ledger or CapitalOOSAlphaLedgerStore(
            settings.database_url
        )
        self.engine = open_database(settings.database_url)

    def _forward_thresholds_for_cycle(
        self, cycle: Mapping[str, Any]
    ) -> ForwardGateThresholds:
        """Resolve the exact Web config revision frozen by this cycle.

        Revision zero is the built-in configuration used before an explicit
        Web save.  A persisted non-zero revision must exist; falling back to a
        newer revision would silently change an already-running experiment.
        """

        try:
            revision = int(cycle.get("config_revision") or 0)
        except (TypeError, ValueError) as exc:
            raise AutopilotCapitalBlocked(
                "autopilot cycle has an invalid configuration revision"
            ) from exc
        if revision < 0:
            raise AutopilotCapitalBlocked(
                "autopilot cycle has an invalid configuration revision"
            )
        config: Mapping[str, Any] = {}
        if revision:
            with self.engine.connect() as connection:
                row = connection.execute(
                    select(platform_config_revisions.c.value_json).where(
                        platform_config_revisions.c.key == "autopilot",
                        platform_config_revisions.c.revision == revision,
                    )
                ).first()
            if row is None or not isinstance(row.value_json, Mapping):
                raise AutopilotCapitalBlocked(
                    "frozen autopilot configuration revision is unavailable"
                )
            config = row.value_json
        calendar_days = config.get("paper_min_calendar_days", 183)
        trading_days = config.get("paper_min_trading_days", 126)
        if (
            isinstance(calendar_days, bool)
            or not isinstance(calendar_days, int)
            or not 183 <= calendar_days <= 3650
        ):
            raise AutopilotCapitalBlocked(
                "frozen paper_min_calendar_days is invalid"
            )
        if (
            isinstance(trading_days, bool)
            or not isinstance(trading_days, int)
            or not 126 <= trading_days <= 2520
        ):
            raise AutopilotCapitalBlocked(
                "frozen paper_min_trading_days is invalid"
            )
        return ForwardGateThresholds(
            min_forward_calendar_days=calendar_days,
            min_decision_batches=trading_days,
            min_completed_cycles=0,
            min_data_completeness=0.95,
            min_reconciliation_rate=1.0,
            max_cost_deviation=0.005,
        )

    def _require_current_quant_admission(self, research_run_id: str) -> list[str]:
        if not research_run_id:
            raise AutopilotCapitalBlocked(
                "successful fin_quant branch has no immutable research run"
            )
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    quant_bundle_candidates.c.id,
                    quant_bundle_candidates.c.status,
                ).where(
                    quant_bundle_candidates.c.research_run_id == research_run_id
                )
            ).all()
        admitted = sorted(
            str(row.id) for row in rows if str(row.status) == "research_admitted"
        )
        if not admitted:
            terminal = sorted(
                {str(row.status) for row in rows} or {"no_quant_bundle_candidate"}
            )
            raise AutopilotCapitalBlocked(
                "fin_quant completed without an independently admitted bundle "
                f"({', '.join(terminal)})"
            )
        return admitted

    @staticmethod
    def _primary_validation_period(version: Mapping[str, Any]) -> tuple[str, str]:
        signal = version.get("model_signal")
        if not isinstance(signal, Mapping):
            raise AutopilotCapitalBlocked(
                "frozen champion has no independently bound prediction signal"
            )
        periods = signal.get("primary_training_periods")
        if not isinstance(periods, Mapping):
            raise AutopilotCapitalBlocked(
                "frozen champion has no primary pre-final validation period"
            )
        valid_start = str(periods.get("valid_start") or "")
        valid_end = str(periods.get("valid_end") or "")
        if not valid_start or not valid_end or valid_start > valid_end:
            raise AutopilotCapitalBlocked(
                "frozen champion primary validation period is invalid"
            )
        return valid_start, valid_end

    def _ensure_portfolio_experiment(
        self,
        *,
        state: dict[str, Any],
        dataset: Mapping[str, Any],
    ) -> tuple[dict[str, Any], int]:
        selection = _selection_state(state)
        selection_evidence = state.get("champion_selection_evidence") or {}
        selection_periods = selection_evidence.get("selected_periods") or {}
        portfolio_validation = (
            selection_evidence.get("selected_portfolio_validation") or {}
        )
        valid_start = str(portfolio_validation.get("start") or "")
        valid_end = str(portfolio_validation.get("end") or "")
        if not valid_start or not valid_end or valid_start > valid_end:
            raise AutopilotCapitalBlocked(
                "frozen champion has no primary pre-final validation period"
            )
        execution_dataset = self._execution_dataset(
            state=state,
            daily_dataset=dataset,
            required_start=valid_start,
            required_end=str(selection_periods.get("end") or valid_end),
        )
        # The provisional version may have been created by a retry before the
        # minute binding was persisted. Rebuild/verify it with the exact
        # governed execution contract before creating the experiment.
        provisional_input = dict(selection)
        if state.get("portfolio_strategy_version_id"):
            provisional_input.update(
                {
                    "strategy_id": state.get("portfolio_strategy_id"),
                    "strategy_version_id": state.get(
                        "portfolio_strategy_version_id"
                    ),
                    "strategy_config_sha256": state.get(
                        "portfolio_strategy_config_sha256"
                    ),
                }
            )
        provisional = self.completion.ensure_strategy(
            state=provisional_input,
            portfolio_config={
                "portfolio_construction": "topk_equal_weight",
                # Daily close predictions are executed from the next tradable
                # session with a frozen minute VWAP profile.  ``next_bar`` is
                # reserved for intraday signals and would make a daily signal
                # contract internally inconsistent.
                "execution_method": "vwap",
                "execution_frequency": str(execution_dataset["frequency"]),
                "execution_slice_minutes": 20,
                "max_execution_slices": 24,
            },
            actor="autopilot",
        )
        version_id = str(provisional["strategy_version_id"])
        version = self.strategies.get_version(version_id)
        valid_start, valid_end = self._primary_validation_period(version)
        prepared = self.experiments.ensure_model_portfolio_competition(
            strategy_version=version,
            dataset={**dict(dataset), "execution_dataset": execution_dataset},
            candidate_valid_start=valid_start,
            candidate_valid_end=valid_end,
            artifact_root=self.settings.data_root
            / "artifacts"
            / "parameter-experiments",
            created_by="autopilot",
        )
        experiment = dict(prepared["experiment"])
        created_jobs = 0
        if prepared["needs_job"]:
            job = self.jobs.create(
                "parameter_experiment",
                dict(prepared["job_payload"]),
                self.settings.data_root
                / "platform"
                / "logs"
                / f"autopilot-portfolio-{experiment['id']}.log",
                dedupe_active_kind=False,
                idempotency_key=f"autopilot:portfolio:{experiment['id']}",
            )
            self.experiments.attach_job(str(experiment["id"]), str(job["id"]))
            experiment = self.experiments.get(str(experiment["id"]))
            created_jobs = int(prepared["created"] or str(job["status"]) == "queued")
        result = {
            **state,
            **selection,
            "portfolio_strategy_id": provisional.get("strategy_id"),
            "portfolio_strategy_version_id": version_id,
            "portfolio_strategy_config_sha256": provisional.get(
                "strategy_config_sha256"
            ),
            "portfolio_experiment_id": str(experiment["id"]),
            "portfolio_experiment_status": str(experiment.get("status") or ""),
            "execution_dataset": {
                "name": str(execution_dataset["name"]),
                "frequency": str(execution_dataset["frequency"]),
                "dataset_identity_sha256": str(
                    (execution_dataset.get("provenance") or {})[
                        "dataset_identity_sha256"
                    ]
                ),
                "dataset_lineage_id": str(
                    (execution_dataset.get("provenance") or {})[
                        "dataset_lineage_id"
                    ]
                ),
            },
            "phase": "portfolio_selection",
        }
        return result, created_jobs

    def _execution_dataset(
        self,
        *,
        state: Mapping[str, Any],
        daily_dataset: Mapping[str, Any],
        required_start: str,
        required_end: str,
    ) -> dict[str, Any]:
        daily_provenance = dict(daily_dataset.get("provenance") or {})
        source_lineage_id = str(daily_provenance.get("source_lineage_id") or "")
        if len(source_lineage_id) != 64:
            raise AutopilotCapitalBlocked(
                "daily Qlib dataset has no verified source lineage for minute execution"
            )
        candidates = [
            item
            for item in list_qlib_datasets(self.settings.data_root)
            if item.get("ready")
            and item.get("reproducible")
            and item.get("lineage_verified")
            and str(item.get("frequency") or "") in {"5min", "1min"}
            and str((item.get("provenance") or {}).get("source_lineage_id") or "")
            == source_lineage_id
            and str(item.get("start_date") or "")[:10] <= required_start
            and str(item.get("end_date") or "")[:10] >= required_end
        ]
        frozen = state.get("execution_dataset")
        if isinstance(frozen, Mapping):
            candidates = [
                item
                for item in candidates
                if str(item.get("name") or "") == str(frozen.get("name") or "")
                and str(
                    (item.get("provenance") or {}).get(
                        "dataset_identity_sha256"
                    )
                    or ""
                )
                == str(frozen.get("dataset_identity_sha256") or "")
                and str(
                    (item.get("provenance") or {}).get("dataset_lineage_id")
                    or ""
                )
                == str(frozen.get("dataset_lineage_id") or "")
            ]
            if not candidates:
                raise AutopilotCapitalBlocked(
                    "frozen minute execution dataset is unavailable or changed"
                )
        if not candidates:
            raise AutopilotCapitalBlocked(
                "no same-lineage 1/5-minute Qlib dataset covers both portfolio "
                "selection and final OOS"
            )
        candidates.sort(
            key=lambda item: (
                str(item.get("frequency") or "") == "5min",
                str(item.get("end_date") or ""),
                str(item.get("name") or ""),
            ),
            reverse=True,
        )
        return dict(candidates[0])

    def _ensure_final_strategy(
        self, state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        experiment_id = str(state.get("portfolio_experiment_id") or "")
        if not experiment_id:
            raise AutopilotCapitalBlocked("portfolio competition identity is missing")
        experiment = self.experiments.get(experiment_id)
        status = str(experiment.get("status") or "")
        if status in {"queued", "running"}:
            return state, {}
        if status in {"failed", "cancelled"}:
            raise AutopilotCapitalBlocked(
                "pre-final TopK/QP competition failed: "
                + str(experiment.get("error") or status)
            )
        try:
            frozen = self.experiments.frozen_portfolio_config(experiment_id)
            overrides = portfolio_overrides_from_frozen(frozen)
        except ValueError as exc:
            raise AutopilotCapitalBlocked(str(exc)) from exc
        recorded = state.get("frozen_portfolio")
        if recorded is not None and recorded != frozen:
            raise AutopilotCapitalBlocked(
                "pre-final portfolio winner changed after it was frozen"
            )
        final_input = _selection_state(state)
        if state.get("strategy_version_id"):
            final_input.update(
                {
                    "strategy_id": state.get("strategy_id"),
                    "strategy_version_id": state.get("strategy_version_id"),
                    "strategy_config_sha256": state.get("strategy_config_sha256"),
                }
            )
        final = self.completion.ensure_strategy(
            state=final_input,
            portfolio_config=overrides,
            actor="autopilot",
        )
        result = {
            **state,
            "frozen_portfolio": frozen,
            "strategy_id": final.get("strategy_id"),
            "strategy_version_id": final.get("strategy_version_id"),
            "strategy_config_sha256": final.get("strategy_config_sha256"),
            "phase": "strategy_frozen",
        }
        return result, frozen

    @staticmethod
    def _capital_cost_contract_sha256(config: Mapping[str, Any]) -> str:
        """Hash the frozen cost inputs, never the selected model/factor."""

        try:
            relevant = CostModelConfig.from_mapping(dict(config)).to_dict()
        except (TypeError, ValueError) as exc:
            raise AutopilotCapitalBlocked(
                "frozen strategy has no governed cost contract"
            ) from exc
        return canonical_sha256(
            {"contract_version": "autopilot-capital-cost-v1", "cost": relevant}
        )

    @staticmethod
    def _capital_governance_contract_sha256(config: Mapping[str, Any]) -> str:
        """Keep the alpha family stable across models and daily snapshots."""

        fields = (
            "autopilot_completion_contract_version",
            "autopilot_selection_policy_version",
            "autopilot_portfolio_contract_version",
            "position_side",
            "shorting_enabled",
            "margin_enabled",
            "financing_enabled",
            "broker_connection_enabled",
            "real_trading_eligible",
            "outer_embargo_days",
            "min_pre_final_history_days",
            "min_backtest_days",
            "min_rolling_windows",
            "min_rolling_pass_rate",
            "min_robustness_pass_rate",
        )
        return canonical_sha256(
            {
                "contract_version": "autopilot-capital-governance-v1",
                "controls": {key: config.get(key) for key in fields},
            }
        )

    def _reserve_capital_oos(
        self,
        *,
        state: Mapping[str, Any],
        dataset: Mapping[str, Any],
        trading_dates: list[date | str],
    ) -> dict[str, Any]:
        """Pre-register one fresh capital OOS before any execution can start."""

        selection = _selection_state(state)["champion_selection_evidence"]
        periods = selection.get("selected_periods")
        version_id = str(state.get("strategy_version_id") or "")
        if not isinstance(periods, Mapping) or not version_id:
            raise AutopilotCapitalBlocked(
                "capital OOS preregistration requires a frozen strategy and periods"
            )
        version = self.strategies.get_version(version_id)
        config = dict(version.get("config") or {})
        if config.get("autopilot_completion_contract_version") != "autopilot-completion-v1":
            raise AutopilotCapitalBlocked("strategy is not a governed Autopilot strategy")
        calendar = [
            item if isinstance(item, date) else date.fromisoformat(str(item))
            for item in trading_dates
        ]
        if calendar != sorted(calendar) or len(calendar) != len(set(calendar)):
            raise AutopilotCapitalBlocked("Qlib final OOS calendar is not canonical")
        research_end = date.fromisoformat(str(periods.get("historical_end") or ""))
        final_start = date.fromisoformat(str(periods.get("start") or ""))
        final_end = date.fromisoformat(str(periods.get("end") or ""))
        embargo = [
            item.isoformat()
            for item in calendar
            if research_end < item < final_start
        ]
        final_dates = [
            item.isoformat() for item in calendar if final_start <= item <= final_end
        ]
        execution_hash = str(config.get("execution_contract_hash") or "").lower()
        if len(execution_hash) != 64:
            raise AutopilotCapitalBlocked("frozen strategy execution contract is invalid")
        stable_mandate = capital_oos_family_manifest(
            str(version.get("universe") or ""),
            str(version.get("benchmark") or ""),
            int(config.get("signal_period") or 1),
            self._capital_cost_contract_sha256(config),
            execution_hash,
            self._capital_governance_contract_sha256(config),
        )
        bundle_manifest_sha256 = canonical_sha256(
            {
                "contract_version": "autopilot-capital-frozen-bundle-v1",
                "champion_selection_evidence_sha256": str(
                    state.get("champion_selection_evidence_sha256") or ""
                ),
                "selected_kind": selection.get("selected_kind"),
                "selected_candidate_id": selection.get("selected_candidate_id"),
                "selected_model_candidate_id": selection.get(
                    "selected_model_candidate_id"
                ),
                "selected_model_component_candidate_ids": selection.get(
                    "selected_model_component_candidate_ids"
                ),
                "strategy_config_sha256": str(
                    state.get("strategy_config_sha256") or ""
                ),
                "frozen_portfolio": state.get("frozen_portfolio"),
            }
        )
        baseline_manifest_sha256 = canonical_sha256(
            {
                "contract_version": "autopilot-capital-benchmark-baseline-v1",
                "benchmark": str(version.get("benchmark") or ""),
                "return_definition": "qlib_report.bench",
                "dataset": str(dataset.get("name") or ""),
                "final_oos_start": final_start.isoformat(),
                "final_oos_end": final_end.isoformat(),
            }
        )
        provenance = dict(dataset.get("provenance") or {})
        lineage = str(
            dataset.get("lineage_id") or provenance.get("dataset_lineage_id") or ""
        )
        identity = str(provenance.get("dataset_identity_sha256") or "")
        batch_key = canonical_sha256(
            {
                "contract_version": "autopilot-capital-final-oos-batch-v1",
                "strategy_version_id": version_id,
                "bundle_manifest_sha256": bundle_manifest_sha256,
                "baseline_manifest_sha256": baseline_manifest_sha256,
                "dataset_identity_sha256": identity,
                "final_oos_dates": final_dates,
            }
        )
        existing = state.get("capital_oos_batch")
        if isinstance(existing, Mapping):
            if (
                str(existing.get("batch_key") or "") != batch_key
                or str(existing.get("frozen_bundle_manifest_sha256") or "")
                != bundle_manifest_sha256
                or str(existing.get("frozen_baseline_manifest_sha256") or "")
                != baseline_manifest_sha256
            ):
                raise AutopilotCapitalBlocked(
                    "capital OOS preregistration changed after strategy freeze"
                )
        try:
            batch = self.capital_ledger.reserve_batch(
                dataset_lineage_id=lineage,
                dataset_identity_sha256=identity,
                stable_mandate=stable_mandate,
                batch_key=batch_key,
                frozen_bundle_manifest_sha256=bundle_manifest_sha256,
                frozen_baseline_manifest_sha256=baseline_manifest_sha256,
                research_data_end=research_end,
                final_oos_trading_dates=final_dates,
                embargo_trading_dates=embargo,
            )
        except ValueError as exc:
            # A different frozen bundle in the same investment mandate may
            # not open a second capital-facing OOS at the same time.  This is
            # expected scheduling back-pressure, not a failed research cycle.
            if str(exc) == "capital OOS family already has an unsettled reserved batch":
                raise AutopilotCapitalWaiting(str(exc)) from exc
            raise
        if str(batch.get("status") or "") == "settled" and batch.get("passed") is not True:
            raise AutopilotCapitalBlocked(
                "the preregistered capital final OOS already failed and cannot be rerun"
            )
        link = self.capital_ledger.vintage_link_contract(str(batch["id"]))
        if (
            str(link.get("capital_oos_alpha_batch_id") or "") != str(batch["id"])
            or str(link.get("capital_oos_dataset_identity_sha256") or "") != identity
        ):
            raise AutopilotCapitalBlocked(
                "capital OOS reservation link does not match the frozen dataset"
            )
        return {
            "batch_id": str(batch["id"]),
            "batch_key": batch_key,
            "preregistration_sha256": str(batch.get("preregistration_sha256") or ""),
            "frozen_bundle_manifest_sha256": bundle_manifest_sha256,
            "frozen_baseline_manifest_sha256": baseline_manifest_sha256,
            "stable_mandate": stable_mandate,
            "stable_mandate_sha256": canonical_sha256(stable_mandate),
            "trading_dates_sha256": str(batch.get("trading_dates_sha256") or ""),
            "vintage_link": link,
        }

    def _ensure_formal_backtest(
        self,
        *,
        state: dict[str, Any],
        dataset: Mapping[str, Any],
    ) -> tuple[dict[str, Any], int]:
        calendar = load_calendar_days(str(dataset["path"]))
        capital_oos = self._reserve_capital_oos(
            state=state,
            dataset=dataset,
            trading_dates=calendar,
        )
        formal_input = {
            **_selection_state(state),
            "strategy_id": state.get("strategy_id"),
            "strategy_version_id": state.get("strategy_version_id"),
            "strategy_config_sha256": state.get("strategy_config_sha256"),
            "formal_backtest_id": state.get("formal_backtest_id"),
            "formal_backtest_status": state.get("formal_backtest_status"),
            "formal_execution_dataset": (
                state.get("formal_execution_dataset")
                or (state.get("execution_dataset") or {}).get("name")
            ),
            "final_oos_consumed": state.get("final_oos_consumed"),
            "capital_oos_batch_id": capital_oos["batch_id"],
            "capital_oos_vintage_link": capital_oos["vintage_link"],
            "capital_oos_dataset_identity_sha256": str(
                (dataset.get("provenance") or {}).get("dataset_identity_sha256") or ""
            ),
        }
        try:
            formal = self.completion.ensure_formal_backtest(
                state=formal_input,
                artifact_path=self.settings.data_root / "artifacts" / "backtests",
                execution_dataset=str((state.get("execution_dataset") or {}).get("name") or ""),
                trading_dates=calendar,
                dataset_lineage_id=str(
                    dataset.get("lineage_id")
                    or (dataset.get("provenance") or {}).get("dataset_lineage_id")
                    or ""
                ),
            )
        except Exception as exc:
            # Once the untouched final window is reserved it is spent even if
            # command construction fails.  Do not turn a retry into a fresh
            # capital-facing test.
            self.capital_ledger.settle_batch(
                str(capital_oos["batch_id"]),
                failed=True,
                failure_reason=f"formal OOS could not be created: {exc}",
                supporting_evidence={
                    "stage": "formal_backtest_creation",
                    "strategy_version_id": str(state.get("strategy_version_id") or ""),
                },
            )
            raise
        backtest_id = str(formal["formal_backtest_id"])
        backtest = self.strategies.get_backtest(backtest_id)
        created_jobs = 0
        if not backtest.get("job_id") and str(backtest.get("status")) == "queued":
            payload = {
                "backtest_id": backtest_id,
                "strategy_version_id": str(state["strategy_version_id"]),
                "dataset": str(dataset["name"]),
                "dataset_path": str(dataset["path"]),
                "execution_dataset": self._execution_dataset(
                    state=state,
                    daily_dataset=dataset,
                    required_start=str(
                        (state["champion_selection_evidence"]["selected_periods"])[
                            "start"
                        ]
                    ),
                    required_end=str(
                        (state["champion_selection_evidence"]["selected_periods"])[
                            "end"
                        ]
                    ),
                ),
                "periods": dict(backtest["periods"]),
                "capital_oos_batch_id": capital_oos["batch_id"],
                "capital_oos_preregistration_sha256": capital_oos[
                    "preregistration_sha256"
                ],
                "capital_oos_frozen_bundle_manifest_sha256": capital_oos[
                    "frozen_bundle_manifest_sha256"
                ],
                "capital_oos_frozen_baseline_manifest_sha256": capital_oos[
                    "frozen_baseline_manifest_sha256"
                ],
            }
            job = self.jobs.create(
                "strategy_backtest",
                payload,
                self.settings.data_root
                / "platform"
                / "logs"
                / f"autopilot-formal-oos-{backtest_id}.log",
                dedupe_active_kind=False,
                idempotency_key=f"autopilot:formal-oos:{backtest_id}",
                max_attempts=1,
            )
            self.strategies.attach_job(backtest_id, str(job["id"]))
            backtest = self.strategies.get_backtest(backtest_id)
            created_jobs = 1
        result = {
            **state,
            **{
                key: value
                for key, value in formal.items()
                if key
                in {
                    "formal_backtest_id",
                    "formal_backtest_status",
                    "formal_execution_dataset",
                    "final_oos_consumed",
                    "phase",
                    "terminal",
                }
            },
            "formal_backtest_job_id": backtest.get("job_id"),
            "capital_oos_batch": capital_oos,
        }
        return result, created_jobs

    def advance(
        self,
        *,
        cycle: Mapping[str, Any],
        dataset: Mapping[str, Any],
        quant_branch: Mapping[str, Any],
    ) -> CapitalPipelineProgress:
        if str(quant_branch.get("status") or "") != "succeeded":
            return CapitalPipelineProgress(
                state=dict((cycle.get("state") or {}).get(CAPITAL_PIPELINE_STATE_KEY) or {}),
                stage="joint_optimization",
            )
        state = dict(
            (cycle.get("state") or {}).get(CAPITAL_PIPELINE_STATE_KEY) or {}
        )
        forward_thresholds = self._forward_thresholds_for_cycle(cycle)
        state.setdefault("contract_version", CAPITAL_PIPELINE_CONTRACT_VERSION)
        if state["contract_version"] != CAPITAL_PIPELINE_CONTRACT_VERSION:
            raise AutopilotCapitalBlocked("capital pipeline contract version changed")
        admitted = self._require_current_quant_admission(
            str(quant_branch.get("research_run_id") or "")
        )
        state["current_quant_bundle_candidate_ids"] = admitted
        state = self.completion.ensure_selection(
            state=state,
            dataset=str(dataset["name"]),
            dataset_identity_sha256=str(
                (dataset.get("provenance") or {})["dataset_identity_sha256"]
            ),
            allowed_candidate_ids=frozenset(admitted),
        )
        state, portfolio_jobs = self._ensure_portfolio_experiment(
            state=state,
            dataset=dataset,
        )
        experiment = self.experiments.get(str(state["portfolio_experiment_id"]))
        if str(experiment.get("status") or "") in {"queued", "running"}:
            return CapitalPipelineProgress(
                state=state,
                stage="portfolio_selection",
                created_jobs=portfolio_jobs,
            )
        state, frozen = self._ensure_final_strategy(state)
        if not frozen:
            return CapitalPipelineProgress(
                state=state,
                stage="portfolio_selection",
                created_jobs=portfolio_jobs,
            )
        try:
            state, backtest_jobs = self._ensure_formal_backtest(
                state=state,
                dataset=dataset,
            )
        except AutopilotCapitalWaiting as exc:
            return CapitalPipelineProgress(
                state={
                    **state,
                    "capital_oos_waiting": True,
                    "capital_oos_wait_reason": str(exc),
                },
                stage="capital_oos_waiting",
                created_jobs=portfolio_jobs,
            )
        backtest = self.strategies.get_backtest(str(state["formal_backtest_id"]))
        backtest_status = str(backtest.get("status") or "")
        if backtest_status in {"failed", "cancelled"}:
            failed = self.completion.approve_if_ready(
                state={**_selection_state(state), **state},
                actor="autopilot",
                forward_thresholds=forward_thresholds,
            )
            raise AutopilotCapitalBlocked(
                str(failed.get("phase") or "formal OOS failed")
            )
        if backtest_status != "succeeded":
            return CapitalPipelineProgress(
                state=state,
                stage="formal_backtest",
                created_jobs=portfolio_jobs + backtest_jobs,
            )
        approved = self.completion.approve_if_ready(
            state={**_selection_state(state), **state},
            actor="autopilot",
            forward_thresholds=forward_thresholds,
        )
        paper = approved.get("paper_stage")
        if (
            not isinstance(paper, Mapping)
            or str(paper.get("status") or "") != "active"
            or not paper.get("simulation_portfolio_id")
        ):
            raise AutopilotCapitalBlocked(
                "formal OOS passed but the isolated paper account could not be created"
            )
        result = {
            **state,
            **{
                key: value
                for key, value in approved.items()
                if key
                in {
                    "phase",
                    "paper_stage",
                    "paper_stage_opened",
                    "forward_gate",
                    "recommendation_enabled",
                    "strategy_status",
                    "formal_backtest_status",
                }
            },
        }
        return CapitalPipelineProgress(
            state=result,
            stage="paper",
            created_jobs=portfolio_jobs + backtest_jobs,
            complete=True,
        )
