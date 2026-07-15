from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from quant_data.config import Settings

from .autonomous_research import AutonomousResearchOrchestrator
from .parameter_experiments import split_research_period
from .research_automation import (
    derive_rolling_research_periods,
    normalize_research_schedule_payload,
    select_latest_program_dataset,
)
from .research_program_store import ResearchProgramStore
from .services import list_qlib_datasets


class ContinuousResearchController:
    """Create one governed campaign when an approved Qlib lineage advances."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.programs = ResearchProgramStore(settings.database_url)
        self.orchestrator = AutonomousResearchOrchestrator(settings)

    def tick(self, *, limit: int = 5, now: datetime | None = None) -> dict[str, int]:
        checked = 0
        created = 0
        deferred = 0
        failed = 0
        while checked < limit:
            program = self.programs.claim_due(now=now)
            if program is None:
                break
            try:
                outcome = self._check(program)
                if outcome == "created":
                    created += 1
                else:
                    deferred += 1
            except Exception as exc:
                self.programs.failed_check(
                    program["id"],
                    error=str(exc),
                    delay_seconds=300,
                )
                failed += 1
            checked += 1
        return {
            "checked": checked,
            "created": created,
            "deferred": deferred,
            "failed": failed,
        }

    def _check(self, program: dict[str, Any]) -> str:
        completed = self.orchestrator.campaigns.latest_completed_for_program(
            program["id"],
            exclude_campaign_id=program.get("last_evaluated_campaign_id"),
        )
        if completed is not None:
            program = self.programs.record_campaign_outcome(program["id"], campaign=completed)
        active = self.orchestrator.campaigns.active_count(research_program_id=program["id"])
        if active >= int(program["max_active_campaigns"]):
            self.programs.checked(
                program["id"],
                message=f"已有 {active} 个活动中的研究，等待其完成或审批",
                delay_seconds=60,
            )
            return "deferred"

        dataset = select_latest_program_dataset(
            list_qlib_datasets(self.settings.data_root),
            lineage_id=str(program["dataset_lineage_id"]),
        )
        if dataset is None:
            self.programs.checked(
                program["id"],
                message="等待同血缘、可复现且已验证的 Qlib 数据集",
                delay_seconds=300,
            )
            return "deferred"

        identity = str(dataset["provenance"]["dataset_identity_sha256"])
        existing = self.orchestrator.campaigns.for_program_dataset(
            research_program_id=program["id"],
            dataset_identity_sha256=identity,
        )
        if existing is not None:
            if program.get("last_dataset_identity_sha256") != identity:
                self.programs.triggered(program["id"], campaign_id=existing["id"], dataset=dataset)
            else:
                self.programs.checked(
                    program["id"],
                    message=f"{dataset['name']} 已创建过研究活动",
                    delay_seconds=300,
                )
            return "deferred"

        calendar = self._calendar(dataset)
        last_end = program.get("last_dataset_end_date")
        if last_end:
            new_days = sum(day > str(last_end) for day in calendar)
            if new_days < int(program["min_new_trading_days"]):
                self.programs.checked(
                    program["id"],
                    message=(
                        f"同血缘仅新增 {new_days} 个交易日；达到 "
                        f"{program['min_new_trading_days']} 日后再研究"
                    ),
                    delay_seconds=300,
                )
                return "deferred"

        template = program["config"]
        windows = template["window_days"]
        required_days = int(windows["train"]) + int(windows["validation"]) + int(windows["test"])
        if len(calendar) < required_days:
            self.programs.checked(
                program["id"],
                message=(
                    f"Qlib 数据集有 {len(calendar)} 个交易日，自动研究安全窗口需要 "
                    f"{required_days} 个"
                ),
                delay_seconds=3600,
            )
            return "deferred"
        periods = derive_rolling_research_periods(
            calendar,
            train_days=int(windows["train"]),
            validation_days=int(windows["validation"]),
            test_days=int(windows["test"]),
        )
        research = normalize_research_schedule_payload(
            {
                "objective": program["objective"],
                "dataset": dataset["name"],
                "loop_n": template["loop_n"],
                "duration": template["duration"],
                "requested_by": f"research-program:{program['id']}",
                "periods": periods,
            },
            max_loops=self.settings.rdagent_max_loops,
        )
        valid_start = date.fromisoformat(periods["valid_start"])
        valid_end = date.fromisoformat(periods["valid_end"])
        config = {
            "research": research,
            "strategy_config": template["strategy_config"],
            "backtest_periods": {
                "start": periods["test_start"],
                "end": periods["test_end"],
            },
            "experiment_periods": split_research_period(valid_start, valid_end),
            "parameter_grid": template["parameter_grid"],
            "experiment_trials": template["experiment_trials"],
            "max_factors": template["max_factors"],
            "recommendation": template["recommendation"],
            "manual_strategy_approval": True,
            "research_program_id": program["id"],
        }
        name = (f"{program['name']} · {dataset['end_date']} · {identity[:8]}")[:150]
        campaign = self.orchestrator.create(
            name=name,
            objective=program["objective"],
            dataset=dataset["name"],
            benchmark=program["benchmark"],
            universe=program["universe"],
            recipe_id=program["recipe_id"],
            config=config,
            actor=f"research-program:{program['id']}",
            research_program_id=program["id"],
            dataset_identity_sha256=identity,
        )
        self.programs.triggered(program["id"], campaign_id=campaign["id"], dataset=dataset)
        return "created"

    @staticmethod
    def _calendar(dataset: dict[str, Any]) -> list[str]:
        path = Path(str(dataset["path"])) / "calendars" / "day.txt"
        days = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [day for day in days if day]
