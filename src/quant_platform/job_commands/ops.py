"""Operational report command builders for LocalJobWorker."""

from __future__ import annotations

import sys
from pathlib import Path


def ops_report_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = worker.settings.data_root / "artifacts" / "ops-reports" / job["kind"] / job["id"]
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    command = [
        sys.executable,
        "-m",
        "quant_platform.ops_tasks",
        job["kind"],
        "--date",
        str(payload["local_date"]),
        "--result",
        str(result_path),
    ]
    if payload.get("dataset"):
        command.extend(["--dataset", str(payload["dataset"])])
    if job["kind"] == "intraday_execution_check":
        command.extend(["--as-of", str(payload["as_of"])])
    return command, result_path, {}


def strategy_health_collect_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    required = (
        "strategy_version_id",
        "promotion_stage_id",
        "simulation_batch_id",
        "formal_backtest_id",
        "daily_dataset_identity_sha256",
        "requested_at",
    )
    if any(not str(payload.get(key) or "") for key in required):
        raise ValueError("strategy health collection payload is incomplete")
    output = (
        worker.settings.data_root
        / "artifacts"
        / "strategy-health-jobs"
        / str(job["id"])
    )
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    script = worker.project_root / "scripts" / "collect_strategy_health.py"
    return (
        [
            sys.executable,
            str(script),
            "--strategy-version-id",
            str(payload["strategy_version_id"]),
            "--promotion-stage-id",
            str(payload["promotion_stage_id"]),
            "--simulation-batch-id",
            str(payload["simulation_batch_id"]),
            "--formal-backtest-id",
            str(payload["formal_backtest_id"]),
            "--daily-dataset-identity-sha256",
            str(payload["daily_dataset_identity_sha256"]),
            "--output",
            str(result_path),
        ],
        result_path,
        {},
    )


COMMANDS = {
    "weekly_report": ops_report_command,
    "monthly_decision_day": ops_report_command,
    "preopen_check": ops_report_command,
    "intraday_execution_check": ops_report_command,
    "strategy_health_collect": strategy_health_collect_command,
}
