"""Subprocess entrypoint for the operational run-calendar job kinds.

The worker (``quant_platform.worker._command``) runs these as
``python -m quant_platform.ops_tasks <kind> --date <YYYY-MM-DD> --result <path>``
so the lightweight reports follow the same durable-job discipline (logs,
retries, visible failures) as every other job kind. Any failure raises and
exits non-zero — the worker then records the job failure and the scheduler
projects it into an alert, keeping failures visible instead of silent.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from quant_data.config import Settings

from .ops_calendar import (
    OPS_TASK_KINDS,
    build_monthly_decision_day,
    build_preopen_check,
    build_weekly_report,
    ops_stores,
    select_ops_dataset,
)


def run_task(
    settings: Settings,
    kind: str,
    local_date: date,
    *,
    dataset_anchor: str | None = None,
) -> dict:
    stores = ops_stores(settings)
    if kind == "weekly_report":
        return build_weekly_report(settings, stores, local_date)
    if kind == "monthly_decision_day":
        return build_monthly_decision_day(settings, stores, local_date)
    if kind == "preopen_check":
        dataset = select_ops_dataset(settings.data_root, dataset_anchor)
        return build_preopen_check(settings, stores, local_date, dataset=dataset)
    raise ValueError(f"unsupported ops task kind: {kind}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant_platform.ops_tasks")
    parser.add_argument("kind", choices=OPS_TASK_KINDS)
    parser.add_argument("--date", required=True, help="local task date, YYYY-MM-DD")
    parser.add_argument("--result", required=True, help="result JSON output path")
    parser.add_argument("--dataset", default=None, help="Qlib dataset anchor name")
    args = parser.parse_args(argv)

    settings = Settings.from_env(Path(".env"))
    report = run_task(
        settings,
        args.kind,
        date.fromisoformat(args.date),
        dataset_anchor=args.dataset or None,
    )
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
