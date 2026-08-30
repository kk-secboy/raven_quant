#!/usr/bin/env python3
"""Register the exact production v15-short to v16 pre-result repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import PROJECT_ROOT

from quant_data.config import Settings
from quant_platform.transparent_baseline_lockbox import (
    DISCRETE_MAX_POSITION_SOURCE_BACKTEST_IDS,
    DISCRETE_MAX_POSITION_SOURCE_COMMIT,
    DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
)
from quant_platform.transparent_baseline_repair import (
    register_discrete_max_position_repair,
)


def main() -> None:
    only_backtest_id = next(iter(DISCRETE_MAX_POSITION_SOURCE_BACKTEST_IDS))
    parser = argparse.ArgumentParser(
        description=(
            "Append the exact failed v15 short no-performance attempt before "
            "opening the governed v16 replacement"
        )
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--backtest-id", default=only_backtest_id)
    parser.add_argument(
        "--target-runtime-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--source-release-commit",
        default=DISCRETE_MAX_POSITION_SOURCE_COMMIT,
    )
    parser.add_argument(
        "--target-recipe-version",
        default=DISCRETE_MAX_POSITION_TARGET_RECIPE_VERSION,
    )
    args = parser.parse_args()
    if str(args.backtest_id).strip().lower() != only_backtest_id:
        parser.error("--backtest-id must equal the one allowlisted production ID")

    settings = Settings.from_env(args.env_file.resolve())
    result = register_discrete_max_position_repair(
        settings.database_url,
        backtest_ids=[args.backtest_id],
        actor=args.actor,
        target_runtime_root=args.target_runtime_root.resolve(),
        source_release_commit=args.source_release_commit,
        target_recipe_version=args.target_recipe_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
