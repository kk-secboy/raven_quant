#!/usr/bin/env python3
"""Reconcile the three public QuantLab baseline strategies and formal tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import PROJECT_ROOT

from quant_data.config import Settings
from quant_platform.transparent_baseline_bootstrap import DEFAULT_ACTOR, reconcile


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently freeze, preregister and queue the short/swing/long "
            "transparent public baselines"
        )
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    settings = Settings.from_env(args.env_file.resolve())
    result = reconcile(settings, actor=args.actor)
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
