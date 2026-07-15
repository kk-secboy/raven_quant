#!/usr/bin/env python3
"""Independently evaluate RD-Agent factor values against a selected Qlib snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_platform.cost_model import CostModelConfig
from quant_platform.factor_evaluator import evaluate_factor_values


def _load_values(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(source)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"unsupported factor values format: {source.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import qlib
    from qlib.data import D

    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    factor_frames = {item["id"]: _load_values(item["values_path"]) for item in candidates}
    periods = manifest["periods"]
    qlib.init(provider_uri=args.provider_uri, region="cn")
    labels = D.features(
        D.instruments(str(manifest.get("universe") or "cn_all")),
        ["Ref($close, -2)/Ref($close, -1)-1"],
        start_time=periods["valid_start"],
        end_time=periods["valid_end"],
        freq="day",
    )
    comparisons = [_load_values(path) for path in manifest.get("comparison_values", [])]
    evaluations = []
    for item in candidates:
        try:
            metrics = evaluate_factor_values(
                factor_frames[item["id"]],
                labels,
                valid_start=pd.Timestamp(periods["valid_start"]).date(),
                valid_end=pd.Timestamp(periods["valid_end"]).date(),
                test_start=pd.Timestamp(periods["test_start"]).date(),
                test_end=pd.Timestamp(periods["test_end"]).date(),
                comparison_values=comparisons,
                cost_model=CostModelConfig.from_mapping(manifest.get("cost_model")),
                reference_order_value=float(manifest["cost_reference_order_value"]),
                min_daily_instruments=int(manifest.get("min_daily_instruments", 50)),
            )
            evaluations.append({"candidate_id": item["id"], "status": "ok", "metrics": metrics})
        except Exception as exc:
            evaluations.append({"candidate_id": item["id"], "status": "failed", "error": str(exc)})
    result = {"status": "ok", "evaluations": evaluations}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
