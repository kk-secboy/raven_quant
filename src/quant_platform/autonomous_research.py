from __future__ import annotations

from datetime import time
from typing import Any

from quant_data.config import Settings

from .cost_model import CostModelConfig
from .job_store import JobStore
from .parameter_experiment_store import ParameterExperimentStore
from .qlib_factor_baseline import (
    FACTOR_SOURCE_QLIB_BASELINE,
    FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER,
    FACTOR_SOURCE_QLIB_CHALLENGER_REPLACEMENT,
    QLIB_BASELINE_RECIPE_IDS,
)
from .recommendation_store import RecommendationStore
from .research_automation import rank_factor_candidates
from .research_campaign_store import ResearchCampaignStore
from .research_store import ResearchStore
from .schedule_store import ScheduleStore
from .services import list_qlib_datasets
from .strategy_store import StrategyStore


class CampaignDeferred(RuntimeError):
    """The campaign is healthy but its current child operation is still running."""


class AutonomousResearchOrchestrator:
    """Advance governed research one durable stage at a time.

    RD-Agent and Qlib remain the compute engines. This class only owns orchestration,
    deterministic selection, lineage links, and the final manual-approval boundary.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.campaigns = ResearchCampaignStore(settings.database_url)
        self.jobs = JobStore(settings.database_url)
        self.research = ResearchStore(settings.database_url)
        self.strategies = StrategyStore(settings.database_url)
        self.experiments = ParameterExperimentStore(settings.database_url)
        self.recommendations = RecommendationStore(settings.database_url)
        self.schedules = ScheduleStore(settings.database_url)

    def create(
        self,
        *,
        name: str,
        objective: str,
        dataset: str,
        benchmark: str,
        universe: str,
        recipe_id: str,
        config: dict[str, Any],
        actor: str,
        research_program_id: str | None = None,
        dataset_identity_sha256: str | None = None,
    ) -> dict[str, Any]:
        evidence = self._dataset(dataset)
        pinned = dict(config)
        pinned["dataset_evidence"] = {
            "name": evidence["name"],
            "path": evidence["path"],
            "start_date": evidence.get("start_date"),
            "end_date": evidence.get("end_date"),
            "provenance": evidence.get("provenance"),
        }
        return self.campaigns.create(
            name=name,
            objective=objective,
            dataset=dataset,
            benchmark=benchmark,
            universe=universe,
            recipe_id=recipe_id,
            config=pinned,
            actor=actor,
            research_program_id=research_program_id,
            dataset_identity_sha256=dataset_identity_sha256,
        )

    def tick(self, *, limit: int = 10) -> dict[str, int]:
        processed = 0
        deferred = 0
        failed = 0
        while processed < limit:
            campaign = self.campaigns.claim_due()
            if campaign is None:
                break
            try:
                self._advance(campaign)
            except CampaignDeferred:
                self.campaigns.defer(campaign["id"], seconds=30)
                deferred += 1
            except Exception as exc:
                self.campaigns.fail(campaign["id"], str(exc))
                failed += 1
            processed += 1
        return {"processed": processed, "deferred": deferred, "failed": failed}

    def retry(self, campaign_id: str, *, actor: str) -> dict[str, Any]:
        campaign = self.campaigns.get(campaign_id, include_events=False)
        if campaign["status"] != "failed":
            raise ValueError("only failed research campaigns may be retried")
        if campaign["stage"] in {
            "final_backtest",
            "strategy_approval",
            "recommendation_portfolio",
            "recommendation_schedule",
            "complete",
        }:
            raise ValueError(
                "a failed frozen strategy or final test cannot be retried or reselected"
            )
        if campaign["stage"] == "research" and campaign.get("research_run_id"):
            run = self.research.get_run(str(campaign["research_run_id"]))
            self._retry_job(run.get("job_id"))
            self.research.requeue_run(run["id"], actor=actor)
        elif campaign["stage"] == "parameter_experiment":
            experiment = self.experiments.get(str(campaign["parameter_experiment_id"]))
            self._retry_job(experiment.get("job_id"))
            self.experiments.requeue(experiment["id"])
        return self.campaigns.retry(campaign_id, actor=actor)

    def _advance(self, campaign: dict[str, Any]) -> None:
        handlers = {
            "research": self._research,
            "factor_selection": self._factor_selection,
            "parameter_experiment": self._parameter_experiment,
            "final_backtest": self._final_backtest,
            "strategy_approval": self._strategy_approval,
            "recommendation_schedule": self._recommendation_schedule,
        }
        handler = handlers.get(campaign["stage"])
        if handler is None:
            raise ValueError(f"unsupported research campaign stage: {campaign['stage']}")
        handler(campaign)

    def _research(self, campaign: dict[str, Any]) -> None:
        run_id = campaign.get("research_run_id")
        if not run_id:
            self._start_research(campaign)
            return
        run = self.research.get_run(str(run_id))
        if run["status"] in {"queued", "running", "evaluating"}:
            raise CampaignDeferred(f"RD-Agent research is {run['status']}")
        if run["status"] != "succeeded":
            raise ValueError(f"RD-Agent research ended in {run['status']}: {run.get('error')}")
        self.campaigns.transition(
            campaign["id"],
            stage="factor_selection",
            status="running",
            event_type="research.succeeded",
            payload={"research_run_id": run["id"]},
        )

    def _start_research(self, campaign: dict[str, Any]) -> None:
        config = campaign["config"]
        research = config["research"]
        evidence = config["dataset_evidence"]
        strategy_config = config["strategy_config"]
        cost_model = CostModelConfig.from_mapping(strategy_config)
        reference_order_value = float(strategy_config.get("capacity_notional", 5_000_000)) / int(
            strategy_config.get("topk", 50)
        )
        try:
            run = self.research.create_run(
                kind="factor",
                objective=campaign["objective"],
                dataset=campaign["dataset"],
                requested_by=f"campaign:{campaign['id']}",
                budget={"loop_n": research["loop_n"], "duration": research["duration"]},
                config={
                    "periods": research["periods"],
                    "dataset_path": evidence["path"],
                    "dataset_identity_sha256": evidence["provenance"]["dataset_identity_sha256"],
                    "campaign_id": campaign["id"],
                    "universe": campaign["universe"],
                    "min_daily_instruments": max(
                        50, int(config["strategy_config"].get("topk", 50))
                    ),
                    "cost_model": cost_model.to_dict(),
                    "cost_reference_order_value": reference_order_value,
                },
                artifact_path=self.settings.data_root / "artifacts" / "rdagent",
            )
        except ValueError as exc:
            if "active factor research run" in str(exc):
                raise CampaignDeferred("another bounded RD-Agent research run is active") from exc
            raise
        log_path = (
            self.settings.data_root / "platform" / "logs" / f"campaign-rdagent-{campaign['id']}.log"
        )
        try:
            job = self.jobs.create(
                "rdagent_factor",
                {
                    "research_run_id": run["id"],
                    "research_campaign_id": campaign["id"],
                    "dataset": campaign["dataset"],
                    "dataset_path": evidence["path"],
                    "dataset_identity_sha256": evidence["provenance"]["dataset_identity_sha256"],
                    "objective": campaign["objective"],
                    "loop_n": research["loop_n"],
                    "duration": research["duration"],
                    "periods": research["periods"],
                    "universe": campaign["universe"],
                    "min_daily_instruments": max(
                        50, int(config["strategy_config"].get("topk", 50))
                    ),
                    "cost_model": cost_model.to_dict(),
                    "cost_reference_order_value": reference_order_value,
                },
                log_path,
                idempotency_key=f"research-campaign:{campaign['id']}:rdagent",
            )
        except Exception as exc:
            self.research.mark_run(run["id"], "failed", error=str(exc))
            raise
        self.research.attach_job(run["id"], job["id"])
        self.campaigns.transition(
            campaign["id"],
            status="running",
            event_type="research.enqueued",
            payload={"research_run_id": run["id"], "job_id": job["id"]},
            links={"research_run_id": run["id"]},
            delay_seconds=30,
        )

    def _factor_selection(self, campaign: dict[str, Any]) -> None:
        candidates = self.research.list_candidates(
            run_id=str(campaign["research_run_id"]), limit=500
        )
        selected = rank_factor_candidates(
            candidates,
            limit=int(campaign["config"]["max_factors"]),
            reference_candidates=self.research.list_candidates(status="promoted", limit=500),
        )
        if not selected:
            raise ValueError("RD-Agent produced no candidates that passed the Qlib factor gate")
        actor = f"research-campaign:{campaign['id']}"
        for candidate in selected:
            if candidate["status"] != "promoted":
                self.research.promote(
                    candidate["id"],
                    actor=actor,
                    reason=(
                        "Automatic promotion after independent Qlib gate and deterministic "
                        f"campaign ranking score {candidate['automation_score']:.6f}."
                    ),
                )
        factor_weight = 1.0 / len(selected)
        factors = [{"candidate_id": item["id"], "weight": factor_weight} for item in selected]
        strategy = self.strategies.get_by_name(campaign["name"])
        if strategy is not None and strategy["created_by"] != actor:
            raise ValueError("strategy name is already owned by another workflow")
        strategy_config = dict(campaign["config"]["strategy_config"])
        if campaign["recipe_id"] in QLIB_BASELINE_RECIPE_IDS:
            requested_mode = str(strategy_config.get("factor_source_mode") or "")
            if requested_mode not in {
                FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER,
                FACTOR_SOURCE_QLIB_CHALLENGER_REPLACEMENT,
            }:
                requested_mode = FACTOR_SOURCE_QLIB_BASELINE_PLUS_CHALLENGER
            requested_weight = float(strategy_config.get("challenger_weight") or 0.0)
            if requested_mode == FACTOR_SOURCE_QLIB_CHALLENGER_REPLACEMENT:
                requested_weight = 1.0
            elif not 0.0 < requested_weight < 1.0:
                requested_weight = 0.30
            if strategy is None:
                strategy = self.strategies.create(
                    name=campaign["name"],
                    description=(
                        "Autonomous governed research campaign. The immutable Qlib "
                        "baseline is version 1; RD-Agent challengers require later versions."
                    ),
                    benchmark=campaign["benchmark"],
                    universe=campaign["universe"],
                    factors=[],
                    config={
                        **strategy_config,
                        "factor_source_mode": FACTOR_SOURCE_QLIB_BASELINE,
                        "challenger_weight": 0.0,
                    },
                    actor=actor,
                )
            strategy = self.strategies.get(strategy["id"])
            candidate_ids = {item["candidate_id"] for item in factors}
            baseline_version = next(
                (
                    version
                    for version in strategy["versions"]
                    if version["created_by"] == actor
                    and version["config"].get("factor_source_mode") == requested_mode
                    and {
                        item["factor_candidate_id"] for item in version["factors"]
                    }
                    == candidate_ids
                ),
                None,
            )
            if baseline_version is None:
                baseline_version = self.strategies.create_version(
                    strategy["id"],
                    benchmark=campaign["benchmark"],
                    universe=campaign["universe"],
                    factors=factors,
                    config={
                        **strategy_config,
                        "factor_source_mode": requested_mode,
                        "challenger_weight": requested_weight,
                    },
                    actor=actor,
                )
        else:
            if strategy is None:
                strategy = self.strategies.create(
                    name=campaign["name"],
                    description=(
                        "Autonomous governed research campaign. Final recommendation "
                        "admission remains subject to explicit human approval."
                    ),
                    benchmark=campaign["benchmark"],
                    universe=campaign["universe"],
                    factors=factors,
                    config=strategy_config,
                    actor=actor,
                )
            baseline_version = strategy["versions"][0]
        config = campaign["config"]
        experiment = self.experiments.create(
            strategy_version_id=baseline_version["id"],
            dataset=campaign["dataset"],
            periods=config["experiment_periods"],
            parameter_grid=config["parameter_grid"],
            baseline_config=baseline_version["config"],
            trials=config["experiment_trials"],
            artifact_root=self.settings.data_root / "artifacts" / "parameter-experiments",
            created_by=actor,
        )
        job = self.jobs.create(
            "parameter_experiment",
            {
                "parameter_experiment_id": experiment["id"],
                "research_campaign_id": campaign["id"],
                "strategy_version_id": baseline_version["id"],
                "dataset": campaign["dataset"],
                "dataset_path": campaign["config"]["dataset_evidence"]["path"],
            },
            self.settings.data_root
            / "platform"
            / "logs"
            / f"campaign-experiment-{campaign['id']}.log",
            dedupe_active_kind=False,
            idempotency_key=f"research-campaign:{campaign['id']}:experiment",
        )
        self.experiments.attach_job(experiment["id"], job["id"])
        self.campaigns.transition(
            campaign["id"],
            stage="parameter_experiment",
            event_type="validation_experiment.enqueued",
            payload={
                "strategy_id": strategy["id"],
                "version_id": baseline_version["id"],
                "experiment_id": experiment["id"],
                "job_id": job["id"],
                "selected_factor_ids": [item["id"] for item in selected],
            },
            state_patch={
                "selected_factors": [
                    {
                        "candidate_id": item["id"],
                        "name": item["name"],
                        "score": item["automation_score"],
                    }
                    for item in selected
                ],
                "baseline_version_id": baseline_version["id"],
            },
            links={
                "strategy_id": strategy["id"],
                "strategy_version_id": baseline_version["id"],
                "parameter_experiment_id": experiment["id"],
            },
            delay_seconds=30,
        )

    def _parameter_experiment(self, campaign: dict[str, Any]) -> None:
        experiment = self.experiments.get(str(campaign["parameter_experiment_id"]))
        if experiment["status"] in {"queued", "running"}:
            raise CampaignDeferred(f"parameter experiment is {experiment['status']}")
        if experiment["status"] != "succeeded":
            raise ValueError(f"parameter experiment failed: {experiment.get('error')}")
        summary = experiment.get("summary") or {}
        best_parameters = summary.get("best_parameters")
        if not isinstance(best_parameters, dict) or not best_parameters:
            raise ValueError("parameter experiment produced no successful challenger")
        baseline_version_id = str(campaign["state"]["baseline_version_id"])
        baseline = self.strategies.get_version(baseline_version_id)
        factors = [
            {
                "candidate_id": item["factor_candidate_id"],
                "weight": item["weight"],
            }
            for item in baseline["factors"]
        ]
        actor = f"research-campaign:{campaign['id']}"
        strategy = self.strategies.get(str(campaign["strategy_id"]))
        frozen = next(
            (
                item
                for item in strategy["versions"]
                if item["id"] != baseline_version_id and item["created_by"] == actor
            ),
            None,
        )
        if frozen is None:
            frozen = self.strategies.create_version(
                str(campaign["strategy_id"]),
                benchmark=campaign["benchmark"],
                universe=campaign["universe"],
                factors=factors,
                config={**baseline["config"], **best_parameters},
                actor=actor,
            )
        backtest, job = self._enqueue_backtest(campaign, frozen["id"], label="final")
        self.campaigns.transition(
            campaign["id"],
            stage="final_backtest",
            event_type="strategy.frozen_final_test.enqueued",
            payload={
                "version_id": frozen["id"],
                "backtest_id": backtest["id"],
                "job_id": job["id"],
                "parameters": best_parameters,
            },
            state_patch={
                "frozen_version_id": frozen["id"],
                "final_backtest_id": backtest["id"],
                "frozen_parameters": best_parameters,
            },
            links={
                "strategy_version_id": frozen["id"],
                "backtest_id": backtest["id"],
            },
            delay_seconds=30,
        )

    def _final_backtest(self, campaign: dict[str, Any]) -> None:
        final = self.strategies.get_backtest(str(campaign["backtest_id"]))
        if final["status"] in {"queued", "running"}:
            raise CampaignDeferred(f"final test is {final['status']}")
        if final["status"] != "succeeded":
            raise ValueError(f"frozen strategy failed final test: {final.get('error')}")
        frozen_version_id = str(campaign["state"]["frozen_version_id"])
        self.campaigns.transition(
            campaign["id"],
            stage="strategy_approval",
            status="awaiting_approval",
            event_type="final_test.succeeded",
            payload={"backtest_id": final["id"], "strategy_version_id": frozen_version_id},
            state_patch={"preferred_version_id": frozen_version_id},
            links={"strategy_version_id": frozen_version_id},
            delay_seconds=30,
        )

    def _strategy_approval(self, campaign: dict[str, Any]) -> None:
        version = self.strategies.get_version(str(campaign["strategy_version_id"]))
        if version["status"] != "approved":
            raise CampaignDeferred(
                "frozen strategy is awaiting explicit approval before recommendations"
            )
        actor = f"research-campaign:{campaign['id']}"
        portfolio_name = f"{campaign['name']} 推荐组合"[:150]
        portfolio = next(
            (item for item in self.recommendations.list(500) if item["name"] == portfolio_name),
            None,
        )
        if portfolio is not None and portfolio["created_by"] != actor:
            raise ValueError("recommendation portfolio name is already owned by another workflow")
        if portfolio is None:
            portfolio = self.recommendations.create(
                name=portfolio_name,
                strategy_version_id=version["id"],
                dataset=campaign["dataset"],
                hypothetical_initial_value=float(
                    campaign["config"]["recommendation"]["hypothetical_initial_value"]
                ),
                actor=actor,
            )
        self.campaigns.transition(
            campaign["id"],
            stage="recommendation_schedule",
            status="running",
            event_type="recommendation_portfolio.created",
            payload={"recommendation_portfolio_id": portfolio["id"]},
            state_patch={"recommendation_portfolio_id": portfolio["id"]},
        )

    def _recommendation_schedule(self, campaign: dict[str, Any]) -> None:
        recommendation = campaign["config"]["recommendation"]
        actor = f"research-campaign:{campaign['id']}"
        schedule_name = f"{campaign['name']} 推荐刷新"[:150]
        schedule = self.schedules.get_by_name(schedule_name)
        if schedule is not None and schedule["created_by"] != actor:
            raise ValueError("recommendation schedule name is already owned by another workflow")
        if schedule is None:
            schedule = self.schedules.create(
                name=schedule_name,
                kind="recommendation_refresh",
                timezone=str(recommendation["timezone"]),
                run_time=time.fromisoformat(str(recommendation["run_time"])),
                trading_days_only=True,
                payload={
                    "recommendation_portfolio_id": campaign["state"]["recommendation_portfolio_id"],
                    "research_campaign_id": campaign["id"],
                },
                misfire_grace_seconds=int(recommendation["misfire_grace_seconds"]),
                actor=actor,
            )
        self.campaigns.transition(
            campaign["id"],
            stage="complete",
            status="succeeded",
            event_type="campaign.succeeded",
            payload={
                "recommendation_portfolio_id": campaign["state"]["recommendation_portfolio_id"],
                "recommendation_schedule_id": schedule["id"],
            },
        )

    def _enqueue_backtest(
        self, campaign: dict[str, Any], version_id: str, *, label: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        existing = self.strategies.list_backtests(version_id=version_id, limit=1)
        backtest = existing[0] if existing else None
        if backtest is None:
            backtest = self.strategies.create_backtest(
                version_id=version_id,
                dataset=campaign["dataset"],
                periods=campaign["config"]["backtest_periods"],
                artifact_path=self.settings.data_root / "artifacts" / "backtests",
            )
        if backtest.get("job_id"):
            return backtest, self.jobs.get(str(backtest["job_id"]))
        job = self.jobs.create(
            "strategy_backtest",
            {
                "backtest_id": backtest["id"],
                "research_campaign_id": campaign["id"],
                "strategy_version_id": version_id,
                "dataset": campaign["dataset"],
                "dataset_path": campaign["config"]["dataset_evidence"]["path"],
                "periods": backtest["periods"],
            },
            self.settings.data_root
            / "platform"
            / "logs"
            / f"campaign-{label}-backtest-{campaign['id']}.log",
            dedupe_active_kind=False,
            idempotency_key=f"research-campaign:{campaign['id']}:{label}-backtest",
        )
        self.strategies.attach_job(backtest["id"], job["id"])
        return backtest, job

    def _dataset(self, name: str) -> dict[str, Any]:
        datasets = {item["name"]: item for item in list_qlib_datasets(self.settings.data_root)}
        dataset = datasets.get(name)
        if not dataset or not dataset.get("ready") or not dataset.get("reproducible"):
            raise ValueError("autonomous research requires a reproducible Qlib dataset")
        return dataset

    def _retry_job(self, job_id: str | None) -> None:
        if not job_id:
            raise ValueError("failed campaign stage has no durable job to retry")
        job = self.jobs.get(str(job_id))
        if job["status"] not in {"failed", "cancelled"}:
            raise ValueError(f"campaign child job is {job['status']}, not retryable")
        self.jobs.retry(job["id"])
