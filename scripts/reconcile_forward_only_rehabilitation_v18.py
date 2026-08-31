#!/usr/bin/env python3
"""Create or continue the exact v18 consumed-history replay and paper lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import PROJECT_ROOT

from quant_data.config import Settings
from quant_platform.transparent_baseline_bootstrap import (
    DEFAULT_ACTOR,
    reconcile_forward_only_rehabilitation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently create/reuse the exact v18 descriptive historical replay, "
            "queue its new backtest job, and enter the existing forward paper lane only "
            "after every replay hard gate passes"
        )
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    settings = Settings.from_env(args.env_file.resolve())
    result = reconcile_forward_only_rehabilitation(settings, actor=args.actor)
    output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.report:
        report = args.report.resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(output + "\n", encoding="utf-8")
    print(output)
    if result["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
