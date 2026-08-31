#!/usr/bin/env python3
"""Register and queue the one governed production v17 execution continuation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from _project import PROJECT_ROOT

from quant_data.config import Settings
from quant_platform.formal_backtest_interruption_recovery import (
    V17_INTERRUPTION_RECOVERY_PROFILE,
)
from quant_platform.formal_backtest_recovery_runner import (
    REFERENCE_CONTAINER,
    SCHEDULER_CONTAINER,
    DockerCLI,
    resolve_host_database_url,
    run_v17_recovery,
)


def _strict_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("external interruption evidence must be a regular JSON file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"external interruption evidence repeats key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("external interruption evidence is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("external interruption evidence must be a JSON object")
    return value


def main() -> None:
    profile = V17_INTERRUPTION_RECOVERY_PROFILE
    parser = argparse.ArgumentParser(
        description=(
            "Seal the exact pre-result external SIGTERM evidence and queue the same "
            "v17 formal backtest job for its one authorized second execution"
        )
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--external-interruption-evidence", type=Path, required=True)
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--reference-container", default=REFERENCE_CONTAINER)
    parser.add_argument("--scheduler-container", default=SCHEDULER_CONTAINER)
    parser.add_argument(
        "--controller",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "v17_recovery_oneshot_controller.py",
    )
    parser.add_argument("--job-id", default=profile.job_id)
    parser.add_argument("--backtest-id", default=profile.backtest_id)
    args = parser.parse_args()
    if str(args.job_id).strip().lower() != profile.job_id:
        parser.error("--job-id must equal the one allowlisted production job")
    if str(args.backtest_id).strip().lower() != profile.backtest_id:
        parser.error("--backtest-id must equal the one allowlisted production backtest")

    try:
        external_interruption = _strict_json_object(
            args.external_interruption_evidence.resolve()
        )
    except ValueError as exc:
        parser.error(str(exc))
    # Settings performs the repository-standard dotenv load. Its development
    # database default is deliberately not valid for this host-only operation.
    Settings.from_env(args.env_file.resolve())
    try:
        database_url = resolve_host_database_url(os.environ)
    except ValueError as exc:
        parser.error(str(exc))
    result = run_v17_recovery(
        database_url=database_url,
        actor=args.actor,
        external_interruption=external_interruption,
        controller_path=args.controller,
        docker=DockerCLI(args.docker_executable),
        reference_container=args.reference_container,
        scheduler_container=args.scheduler_container,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
