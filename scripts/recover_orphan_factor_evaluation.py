#!/usr/bin/env python3
"""Safely recover one completed orphan factor-evaluation result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.config import Settings
from quant_platform.factor_evaluation_recovery import (
    RecoverySafetyError,
    inspect_orphan_factor_evaluation,
)
from quant_platform.job_store import JobStore
from quant_platform.worker import LocalJobWorker


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a completed orphan factor_evaluate result. The default is read-only; "
            "pass --apply only after reviewing the dry-run report."
        )
    )
    parser.add_argument("job_id", help="exact 32-character factor_evaluate job id")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="import the frozen result and finalize the durable job",
    )
    parser.add_argument(
        "--expected-result-sha256",
        help="result hash printed by a prior dry-run; required with --apply",
    )
    arguments = parser.parse_args(argv)
    if arguments.apply and not arguments.expected_result_sha256:
        parser.error("--apply requires --expected-result-sha256 from a prior dry-run")
    return arguments


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path.cwd().resolve()
    mutation_started = False
    try:
        settings = Settings.from_env(project_root / ".env")
        store = JobStore(settings.database_url)
        inspection = inspect_orphan_factor_evaluation(
            store,
            data_root=settings.data_root,
            job_id=str(args.job_id),
        )
        report = inspection.public_report()
        if not args.apply:
            print(json.dumps({**report, "mode": "dry-run", "applied": False}, indent=2))
            return 0
        if str(args.expected_result_sha256) != inspection.result_sha256:
            raise RecoverySafetyError("expected result SHA-256 does not match the dry-run evidence")

        # Re-run every read-only check immediately before mutation. This also
        # proves the result bytes and durable job status did not change while
        # the operator reviewed the first inspection.
        fresh = inspect_orphan_factor_evaluation(
            store,
            data_root=settings.data_root,
            job_id=str(args.job_id),
        )
        if fresh.result_sha256 != inspection.result_sha256:
            raise RecoverySafetyError("factor evaluation result changed before apply")
        worker = LocalJobWorker(store, project_root, settings)
        mutation_started = True
        outcome = worker.finalize_completed_factor_evaluation(fresh.job, fresh.result)
        print(
            json.dumps(
                {**fresh.public_report(), "mode": "apply", "applied": True, "outcome": outcome},
                indent=2,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - report whether mutation may have begun
        print(
            json.dumps(
                {
                    "status": "failed" if mutation_started else "blocked",
                    "applied": None if mutation_started else False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
