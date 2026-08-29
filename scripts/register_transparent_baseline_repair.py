#!/usr/bin/env python3
"""Register the production v7-to-v8 pre-result baseline repair receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import PROJECT_ROOT

from quant_data.config import Settings
from quant_platform.transparent_baseline_lockbox import (
    OPTIMIZER_APPLICABILITY_SOURCE_COMMIT,
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
)
from quant_platform.transparent_baseline_repair import (
    register_optimizer_applicability_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Append the exact three failed v7 no-performance attempts before "
            "opening governed v8 strategy versions"
        )
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--backtest-id", action="append", required=True)
    parser.add_argument(
        "--source-release-commit", default=OPTIMIZER_APPLICABILITY_SOURCE_COMMIT
    )
    parser.add_argument(
        "--target-recipe-version",
        default=OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    )
    args = parser.parse_args()

    settings = Settings.from_env(args.env_file.resolve())
    result = register_optimizer_applicability_repair(
        settings.database_url,
        backtest_ids=args.backtest_id,
        actor=args.actor,
        source_release_commit=args.source_release_commit,
        target_recipe_version=args.target_recipe_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
