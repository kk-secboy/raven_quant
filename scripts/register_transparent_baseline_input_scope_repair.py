#!/usr/bin/env python3
"""Register the exact production v10-to-v11 runtime input-scope repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import PROJECT_ROOT

from quant_data.config import Settings
from quant_platform.transparent_baseline_lockbox import (
    RUNTIME_INPUT_SCOPE_SOURCE_COMMIT,
    RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION,
)
from quant_platform.transparent_baseline_repair import (
    register_runtime_input_scope_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Append the exact three failed v10 no-performance input-scope attempts "
            "before opening governed v11 strategy versions"
        )
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--backtest-id", action="append", required=True)
    parser.add_argument(
        "--source-runtime-root",
        type=Path,
        required=True,
        help=(
            "explicit read-only mount of the immutable v10 release whose executed "
            "runtime bytes are sealed"
        ),
    )
    parser.add_argument(
        "--target-runner-path",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "run_multifactor_backtest.py",
    )
    parser.add_argument(
        "--source-release-commit", default=RUNTIME_INPUT_SCOPE_SOURCE_COMMIT
    )
    parser.add_argument(
        "--target-recipe-version",
        default=RUNTIME_INPUT_SCOPE_TARGET_RECIPE_VERSION,
    )
    args = parser.parse_args()

    settings = Settings.from_env(args.env_file.resolve())
    result = register_runtime_input_scope_repair(
        settings.database_url,
        backtest_ids=args.backtest_id,
        actor=args.actor,
        source_runtime_root=args.source_runtime_root.resolve(),
        target_runner_path=args.target_runner_path.resolve(),
        source_release_commit=args.source_release_commit,
        target_recipe_version=args.target_recipe_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
