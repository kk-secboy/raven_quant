#!/usr/bin/env python3
"""Advance one governed pair-paper portfolio through next-session minute execution."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pair_backtest import (
    _canonical_sha256,
    _daily_market,
    _normalize_minute,
    _normalize_shortability,
    _read_parquet_dataset,
    _sha256_file,
)

from quant_platform.pair_trading import PairTradingConfig, run_pair_paper_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--minute-path", required=True)
    parser.add_argument("--shortability-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair = manifest["pair"]
    legs = [str(pair["leg_y"]), str(pair["leg_x"])]
    leg_set = set(legs)
    signal_date = date.fromisoformat(manifest["as_of_date"])
    execution_end = (signal_date + timedelta(days=15)).isoformat()
    minute = _normalize_minute(
        _read_parquet_dataset(
            Path(args.minute_path),
            legs=leg_set,
            start=manifest["as_of_date"],
            end=execution_end,
        ),
        legs=leg_set,
        start=manifest["as_of_date"],
        end=execution_end,
    )
    shortability = _normalize_shortability(
        _read_parquet_dataset(
            Path(args.shortability_path),
            legs=leg_set,
            start=manifest["as_of_date"],
            end=execution_end,
        ),
        legs=leg_set,
    )
    daily = _daily_market(
        args.provider_uri,
        legs=legs,
        periods={"start": manifest["dataset_start"], "end": execution_end},
        shortability=shortability,
    )
    result = run_pair_paper_step(
        daily,
        minute,
        leg_y=legs[0],
        leg_x=legs[1],
        as_of_date=manifest["as_of_date"],
        state=manifest["state"],
        config=PairTradingConfig(**manifest["config"]),
    )
    daily_provenance = manifest["daily_provenance"]
    result["provenance"] = {
        "daily_dataset_identity_sha256": daily_provenance["dataset_identity_sha256"],
        "daily_snapshot_manifest_sha256": daily_provenance["snapshot_manifest_sha256"],
        "minute_snapshot_manifest_sha256": manifest["minute_dataset"]["manifest_sha256"],
        "shortability_evidence_sha256": manifest["shortability_dataset"]["source_sha256"],
        "daily_dataset_lineage_id": daily_provenance.get("dataset_lineage_id"),
        "execution_snapshot_lineage_id": manifest["minute_dataset"].get(
            "snapshot_lineage_id"
        ),
        "strategy_config_sha256": _canonical_sha256(manifest["config"]),
        "execution_manifest_sha256": _sha256_file(manifest_path),
        "pair_engine_sha256": _sha256_file(
            Path(__file__).resolve().parents[1] / "src" / "quant_platform" / "pair_trading.py"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
