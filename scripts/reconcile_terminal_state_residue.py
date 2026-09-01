from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from quant_data.config import Settings
from quant_platform.terminal_state_reconciliation import TerminalStateReconciler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect historical parameter/research residue and apply only exact "
            "terminal projections. The default is read-only."
        )
    )
    parser.add_argument("--experiment-id", action="append", default=[])
    parser.add_argument("--research-run-id", action="append", default=[])
    parser.add_argument("--orphan-minimum-age-hours", type=float, default=1.0)
    parser.add_argument("--actor", default="system:terminal-state-reconciler")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.experiment_id and not args.research_run_id:
        parser.error("at least one explicit experiment or research-run ID is required")
    if args.orphan_minimum_age_hours < 1:
        parser.error("orphan minimum age must be at least one hour")

    settings = Settings.from_env(Path(".env"))
    reconciler = TerminalStateReconciler(settings.database_url)
    decisions = []
    for experiment_id in args.experiment_id:
        decisions.append(
            reconciler.reconcile_parameter_experiment(
                str(experiment_id), actor=args.actor, apply=args.apply
            ).as_dict()
        )
    minimum_age = timedelta(hours=args.orphan_minimum_age_hours)
    for run_id in args.research_run_id:
        decisions.append(
            reconciler.reconcile_research_run(
                str(run_id),
                actor=args.actor,
                apply=args.apply,
                orphan_minimum_age=minimum_age,
            ).as_dict()
        )
    print(
        json.dumps(
            {"mode": "apply" if args.apply else "dry_run", "decisions": decisions},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
