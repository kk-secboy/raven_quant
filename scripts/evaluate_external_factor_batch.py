#!/usr/bin/env python3
"""Independently evaluate external NLP factor candidates against a Qlib snapshot.

Mirrors scripts/evaluate_factor_batch.py, but for externally produced factor
artifacts (announcement/corpus NLP) whose values cannot be recomputed from
market data and whose shapes (sparse event sets, MARKET timeseries) do not fit
the cross-sectional coverage gates. Shape detection, evaluation paths and
gates live in quant_platform.external_factor_evaluation.

Manifest schema (JSON):
    research_run_id, dataset, dataset_identity_sha256, universe, benchmark,
    periods {train_start..test_end}, candidates [{id, values_path,
    code_sha256, values_sha256, shape?, experiment_family_id,
    experiment_count, label_horizon_days}],
    comparison_values [paths], cost_model, cost_reference_order_value

Pass --database-url to also persist the evaluations into factor_evaluations
(one-command evaluation + recording); without it only result.json is written,
so a worker-style import can happen later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.cost_model import CostScheduleBook
from quant_platform.external_factor_evaluation import (
    POLICY_BY_SHAPE,
    SHAPE_MARKET_TIMESERIES,
    ExternalEvaluationConfig,
    apply_family_bh_correction,
    build_external_evidence,
    detect_external_factor_shape,
    evaluate_external_walk_forward,
    evaluate_market_timeseries_factor,
    evaluate_sparse_event_factor,
    import_external_evaluations,
)
from quant_platform.qlib_workflow import qlib_workflow_run


def _load_values(path: str) -> pd.DataFrame:
    from quant_platform.external_factor_evaluation import to_qlib_instrument_format

    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        return to_qlib_instrument_format(pd.read_hdf(source))
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
        # NLP-produced artifacts store flat datetime/instrument columns; the
        # evaluator contract expects a datetime/instrument MultiIndex.
        if {"datetime", "instrument"} <= set(frame.columns):
            value_columns = [
                column for column in frame.columns if column not in {"datetime", "instrument"}
            ]
            if len(value_columns) == 1:
                return to_qlib_instrument_format(
                    frame.set_index(["datetime", "instrument"])[value_columns[0]]
                )
        return to_qlib_instrument_format(frame)
    raise ValueError(f"unsupported factor values format: {source.suffix}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forward_returns(instruments: Any, horizon: int, periods: dict[str, str]) -> pd.DataFrame:
    from qlib.data import D

    return D.features(
        instruments,
        [f"Ref($close, -{horizon + 1})/Ref($close, -1)-1"],
        start_time=periods["train_start"],
        end_time=periods["valid_end"],
        freq="day",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    import qlib
    from qlib.data import D

    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    periods = manifest["periods"]
    universe = str(manifest.get("universe") or "cn_all")
    benchmark = str(manifest.get("benchmark") or "SH000300")
    dataset_identity = str(manifest["dataset_identity_sha256"])
    qlib.init(provider_uri=args.provider_uri, region="cn")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_root = output.parent / "external-inputs"
    evidence_root.mkdir(parents=True, exist_ok=True)

    horizons = sorted({int(item.get("label_horizon_days") or 1) for item in candidates})
    labels_by_horizon: dict[int, pd.DataFrame] = {}
    benchmark_labels_by_horizon: dict[int, pd.DataFrame] = {}
    input_sha_by_horizon: dict[int, dict[str, str]] = {}
    for horizon in horizons:
        labels = _forward_returns(D.instruments(universe), horizon, periods)
        benchmark_labels = _forward_returns(D.instruments(benchmark), horizon, periods)
        labels_path = evidence_root / f"forward-returns-{horizon}d.h5"
        benchmark_path = evidence_root / f"benchmark-forward-returns-{horizon}d.h5"
        labels.to_hdf(labels_path, key="data", mode="w")
        benchmark_labels.to_hdf(benchmark_path, key="data", mode="w")
        labels_by_horizon[horizon] = labels
        benchmark_labels_by_horizon[horizon] = benchmark_labels
        input_sha_by_horizon[horizon] = {
            "forward_returns_sha256": _sha256_file(labels_path),
            "benchmark_forward_returns_sha256": _sha256_file(benchmark_path),
        }

    comparisons = [_load_values(path) for path in manifest.get("comparison_values", [])]
    cost_schedule = CostScheduleBook.from_mapping(manifest.get("cost_model"))
    reference_order_value = float(manifest.get("cost_reference_order_value", 100_000.0))
    config = ExternalEvaluationConfig(require_rolling_walk_forward=True)
    period_dates = {key: date.fromisoformat(value) for key, value in periods.items()}
    evaluations: list[dict[str, Any]] = []
    for item in candidates:
        try:
            horizon = int(item.get("label_horizon_days") or 1)
            values = _load_values(str(item["values_path"]))
            shape = detect_external_factor_shape(
                values,
                labels_by_horizon[horizon],
                valid_start=period_dates["valid_start"],
                valid_end=period_dates["valid_end"],
                config=config,
                shape_hint=item.get("shape"),
            )
            policy = POLICY_BY_SHAPE[shape]()
            if shape == SHAPE_MARKET_TIMESERIES:
                outcome = evaluate_market_timeseries_factor(
                    values,
                    benchmark_labels_by_horizon[horizon],
                    valid_start=period_dates["valid_start"],
                    valid_end=period_dates["valid_end"],
                    test_start=period_dates["test_start"],
                    test_end=period_dates["test_end"],
                    cost_schedule=cost_schedule,
                    reference_order_value=reference_order_value,
                    label_horizon_days=horizon,
                    config=config,
                )
            else:
                outcome = evaluate_sparse_event_factor(
                    values,
                    labels_by_horizon[horizon],
                    valid_start=period_dates["valid_start"],
                    valid_end=period_dates["valid_end"],
                    test_start=period_dates["test_start"],
                    test_end=period_dates["test_end"],
                    comparison_values=comparisons,
                    cost_schedule=cost_schedule,
                    reference_order_value=reference_order_value,
                    label_horizon_days=horizon,
                    config=config,
                )
            if outcome["status"] == "ok":
                rolling = evaluate_external_walk_forward(
                    values,
                    (
                        benchmark_labels_by_horizon[horizon]
                        if shape == SHAPE_MARKET_TIMESERIES
                        else labels_by_horizon[horizon]
                    ),
                    evaluation_shape=shape,
                    train_start=period_dates["train_start"],
                    valid_end=period_dates["valid_end"],
                    label_horizon_days=horizon,
                    config=config,
                )
                outcome["metrics"]["rolling_walk_forward"] = rolling
            evaluations.append(
                {
                    "candidate_id": item["id"],
                    "status": outcome["status"],
                    "shape": shape,
                    "metrics": outcome["metrics"],
                    "reasons": outcome["reasons"],
                    "experiment_family_id": item.get("experiment_family_id"),
                    "experiment_count": int(item.get("experiment_count") or 1),
                    "evidence": build_external_evidence(
                        evaluation_shape=shape,
                        config=config,
                        policy=policy,
                        candidate={
                            "code_sha256": item["code_sha256"],
                            "values_sha256": item["values_sha256"],
                        },
                        dataset_identity_sha256=dataset_identity,
                        periods=period_dates,
                        label_horizon_days=horizon,
                        input_data_sha256=input_sha_by_horizon[horizon][
                            "forward_returns_sha256"
                            if shape != SHAPE_MARKET_TIMESERIES
                            else "benchmark_forward_returns_sha256"
                        ],
                    ),
                }
            )
        except Exception as exc:
            evaluations.append({"candidate_id": item["id"], "status": "failed", "error": str(exc)})
    apply_family_bh_correction(evaluations)
    result: dict[str, Any] = {"status": "ok", "evaluations": evaluations}
    with qlib_workflow_run(
        run_kind="external-factor-evaluation",
        run_id=str(manifest.get("research_run_id") or output.parent.name),
        tracking_uri=args.tracking_uri,
        dataset_identity_sha256=dataset_identity,
    ) as workflow:
        workflow.log_params(
            {
                "research_run_id": manifest.get("research_run_id") or output.parent.name,
                "universe": universe,
                "benchmark": benchmark,
                "candidate_count": len(candidates),
                "evaluator_config_version": config.version,
            }
        )
        workflow.log_metrics(
            {
                "candidate_count": len(evaluations),
                "succeeded_count": sum(item.get("status") == "ok" for item in evaluations),
                "insufficient_count": sum(
                    item.get("status") == "insufficient_evidence" for item in evaluations
                ),
                "failed_count": sum(item.get("status") == "failed" for item in evaluations),
            }
        )
        result["qlib_workflow"] = workflow.identity_dict()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        workflow.save_artifacts(output.parent)
    if args.database_url:
        from quant_platform.research_store import ResearchStore

        imported = import_external_evaluations(
            ResearchStore(args.database_url),
            result,
            dataset=str(manifest["dataset"]),
            dataset_identity_sha256=dataset_identity,
            periods=period_dates,
            artifact_path=output,
        )
        result["imported_evaluation_ids"] = [item["id"] for item in imported]
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
