#!/usr/bin/env python3
"""Register the exact production v8-to-v9 canonical-LF repair receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import PROJECT_ROOT

from quant_data.config import Settings
from quant_platform.transparent_baseline_lockbox import (
    CANONICAL_LF_PACKAGING_SOURCE_COMMIT,
    CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
)
from quant_platform.transparent_baseline_repair import (
    register_canonical_lf_packaging_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Append the exact three failed v8 no-performance attempts before "
            "opening governed canonical-LF v9 strategy versions"
        )
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--backtest-id", action="append", required=True)
    parser.add_argument("--source-runner-observed-sha256", required=True)
    parser.add_argument(
        "--target-runner-path",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "run_multifactor_backtest.py",
    )
    parser.add_argument(
        "--source-release-commit", default=CANONICAL_LF_PACKAGING_SOURCE_COMMIT
    )
    parser.add_argument(
        "--target-recipe-version",
        default=CANONICAL_LF_PACKAGING_TARGET_RECIPE_VERSION,
    )
    args = parser.parse_args()

    settings = Settings.from_env(args.env_file.resolve())
    result = register_canonical_lf_packaging_repair(
        settings.database_url,
        backtest_ids=args.backtest_id,
        actor=args.actor,
        source_runner_observed_sha256=args.source_runner_observed_sha256,
        target_runner_path=args.target_runner_path.resolve(),
        source_release_commit=args.source_release_commit,
        target_recipe_version=args.target_recipe_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
