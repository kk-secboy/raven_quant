"""Collect one exact strategy-health lane in an isolated worker process."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from quant_data.config import Settings
from quant_platform.strategy_health_collector import StrategyHealthCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-version-id", required=True)
    parser.add_argument("--promotion-stage-id", required=True)
    parser.add_argument("--simulation-batch-id", required=True)
    parser.add_argument("--formal-backtest-id", required=True)
    parser.add_argument("--daily-dataset-identity-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    collector = StrategyHealthCollector(
        settings.database_url,
        data_root=settings.data_root,
    )
    result = collector.collect_one(
        strategy_version_id=str(args.strategy_version_id),
        promotion_stage_id=str(args.promotion_stage_id),
        simulation_batch_id=str(args.simulation_batch_id),
        formal_backtest_id=str(args.formal_backtest_id),
        daily_dataset_identity_sha256=str(
            args.daily_dataset_identity_sha256
        ),
        observed_at=datetime.now(UTC),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
