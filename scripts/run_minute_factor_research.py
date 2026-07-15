#!/usr/bin/env python3
"""Evaluate a fixed, auditable microstructure factor library on a minute Qlib dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.minute_research import (  # noqa: E402
    MINUTE_FACTOR_EXPRESSIONS,
    evaluate_minute_factor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizons", default="5,15,30")
    parser.add_argument("--cost-rate", type=float, default=0.0002)
    args = parser.parse_args()

    provider = Path(args.provider_uri)
    provenance_path = provider / "metadata" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("frequency") != "1min":
        raise ValueError("minute factor research requires a 1min Qlib dataset")
    horizons = sorted({int(item) for item in args.horizons.split(",") if item.strip()})
    if not horizons or horizons[0] < 1 or horizons[-1] > 240:
        raise ValueError("horizons must be between 1 and 240 minutes")

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(provider), region="cn")
    instruments = D.instruments("all")
    factor_names = list(MINUTE_FACTOR_EXPRESSIONS)
    expressions = [MINUTE_FACTOR_EXPRESSIONS[name] for name in factor_names]
    frame = D.features(
        instruments,
        expressions,
        start_time=args.start,
        end_time=args.end,
        freq="1min",
    )
    frame.columns = factor_names
    results = []
    for horizon in horizons:
        labels = D.features(
            instruments,
            [f"Ref($close,-{horizon})/$close-1"],
            start_time=args.start,
            end_time=args.end,
            freq="1min",
        )
        for name in factor_names:
            try:
                metrics = evaluate_minute_factor(
                    frame[[name]],
                    labels,
                    horizon_minutes=horizon,
                    cost_rate=args.cost_rate,
                )
                score = float(metrics["rank_ic"] or 0) + float(metrics["mean_net_return"] or 0)
                results.append(
                    {
                        "factor": name,
                        "expression": MINUTE_FACTOR_EXPRESSIONS[name],
                        "horizon_minutes": horizon,
                        "status": "ok",
                        "score": score,
                        "metrics": metrics,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "factor": name,
                        "expression": MINUTE_FACTOR_EXPRESSIONS[name],
                        "horizon_minutes": horizon,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    succeeded = sorted(
        (item for item in results if item["status"] == "ok"),
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    if not succeeded:
        raise RuntimeError("every minute factor evaluation failed")
    result = {
        "status": "ok",
        "dataset": provider.name,
        "frequency": "1min",
        "start": args.start,
        "end": args.end,
        "horizons": horizons,
        "cost_rate": args.cost_rate,
        "dataset_identity_sha256": provenance["dataset_identity_sha256"],
        "research_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
        "results": results,
        "ranking": succeeded,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "best": succeeded[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
