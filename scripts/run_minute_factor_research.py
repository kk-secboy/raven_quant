#!/usr/bin/env python3
"""Evaluate a fixed, auditable microstructure factor library on a minute Qlib dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.execution_contract import require_minute_signal_contract  # noqa: E402
from quant_platform.minute_research import (  # noqa: E402
    evaluate_minute_factor,
    minute_bar_minutes,
    minute_factor_expressions,
)
from quant_platform.qlib_workflow import qlib_workflow_run  # noqa: E402


def _record_research_result(
    result: dict[str, Any],
    *,
    output: Path,
    tracking_uri: str,
    workflow_factory: Callable[..., Any] = qlib_workflow_run,
) -> dict[str, Any]:
    ranking = list(result.get("ranking") or [])
    if not ranking:
        raise ValueError("minute research result has no successful ranking")
    run_id = (
        f"{result['dataset']}-{result['dataset_identity_sha256'][:12]}-"
        f"{result['start']}-{result['end']}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with workflow_factory(
        run_kind="minute-factor-research",
        run_id=run_id,
        tracking_uri=tracking_uri,
        dataset_identity_sha256=result["dataset_identity_sha256"],
    ) as workflow:
        workflow.log_params(
            {
                "dataset": result["dataset"],
                "frequency": result["frequency"],
                "start": result["start"],
                "end": result["end"],
                "horizons": ",".join(str(item) for item in result["horizons"]),
                "cost_rate": result["cost_rate"],
                "research_code_sha256": result["research_code_sha256"],
            }
        )
        workflow.log_metrics(
            {
                "successful_factor_horizons": len(ranking),
                "failed_factor_horizons": sum(
                    item.get("status") != "ok" for item in result.get("results") or []
                ),
                "best_score": ranking[0]["score"],
            }
        )
        workflow.set_tags(
            {
                "frequency": result["frequency"],
                "research_contract": "fixed-minute-factor-library",
            }
        )
        result["qlib_workflow"] = workflow.identity_dict()
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        workflow.save_artifacts(output.parent, artifact_path="minute-research")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizons", default="5,15,30")
    parser.add_argument("--cost-rate", type=float, default=0.0002)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()

    provider = Path(args.provider_uri)
    provenance_path = provider / "metadata" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    frequency = str(provenance.get("frequency") or "")
    require_minute_signal_contract(provenance, frequency=frequency)
    bar_minutes = minute_bar_minutes(frequency)
    horizons = sorted({int(item) for item in args.horizons.split(",") if item.strip()})
    if not horizons or horizons[0] < 1 or horizons[-1] > 240:
        raise ValueError("horizons must be between 1 and 240 minutes")
    if any(horizon % bar_minutes for horizon in horizons):
        raise ValueError("horizons must be integer multiples of the dataset frequency")

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(provider), region="cn")
    instruments = D.instruments("all")
    factor_expressions = minute_factor_expressions(frequency)
    factor_names = list(factor_expressions)
    expressions = [factor_expressions[name] for name in factor_names]
    frame = D.features(
        instruments,
        expressions,
        start_time=args.start,
        end_time=args.end,
        freq=frequency,
    )
    frame.columns = factor_names
    results = []
    for horizon in horizons:
        horizon_bars = horizon // bar_minutes
        labels = D.features(
            instruments,
            [f"Ref($close,-{horizon_bars})/$close-1"],
            start_time=args.start,
            end_time=args.end,
            freq=frequency,
        )
        for name in factor_names:
            try:
                metrics = evaluate_minute_factor(
                    frame[[name]],
                    labels,
                    horizon_minutes=horizon,
                    cost_rate=args.cost_rate,
                    bar_minutes=bar_minutes,
                )
                score = float(metrics["rank_ic"] or 0) + float(metrics["mean_net_return"] or 0)
                results.append(
                    {
                        "factor": name,
                        "expression": factor_expressions[name],
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
                        "expression": factor_expressions[name],
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
        "frequency": frequency,
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
    _record_research_result(
        result,
        output=Path(args.output),
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps({"status": "ok", "best": succeeded[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
