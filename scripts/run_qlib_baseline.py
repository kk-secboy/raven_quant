from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_platform.qlib_workflow import (
    qlib_workflow_run,
    require_qlib_workflow_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible Qlib Alpha158 baseline")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--provider-uri")
    parser.add_argument("--output")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--market", default="cn_all")
    parser.add_argument("--benchmark", default="SH000300")
    parser.add_argument("--account", type=float, default=5_000_000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=5)
    parser.add_argument("--open-cost", type=float, default=0.0005)
    parser.add_argument("--close-cost", type=float, default=0.0015)
    parser.add_argument("--min-cost", type=float, default=5.0)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def main() -> None:
    args = parse_args()
    import pandas as pd
    import qlib
    from qlib.constant import REG_CN

    if args.probe:
        import lightgbm

        print(
            json.dumps(
                {
                    "status": "ok",
                    "qlib_version": getattr(qlib, "__version__", "unknown"),
                    "lightgbm_version": getattr(lightgbm, "__version__", "unknown"),
                },
                ensure_ascii=False,
            )
        )
        return
    if not args.provider_uri or not args.output or not args.tracking_uri:
        raise SystemExit("--provider-uri, --output, and --tracking-uri are required")

    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data import D
    from qlib.data.dataset import DatasetH

    provider_uri = Path(args.provider_uri).resolve()
    provenance_path = provider_uri / "metadata" / "provenance.json"
    if not provenance_path.exists():
        raise RuntimeError("Qlib baseline requires dataset provenance metadata")
    dataset_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    if output.exists():
        completed = output / "result.json"
        if completed.exists():
            result = json.loads(completed.read_text(encoding="utf-8"))
            try:
                require_qlib_workflow_identity(result.get("qlib_workflow"))
            except ValueError:
                pass
            else:
                print(json.dumps(result, ensure_ascii=False))
                return
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    calendar = list(pd.to_datetime(D.calendar(freq="day")))
    if len(calendar) < 180:
        raise RuntimeError(
            f"Qlib dataset has only {len(calendar)} trading days; at least 180 required"
        )

    train_end = max(100, int(len(calendar) * 0.60))
    valid_end = max(train_end + 40, int(len(calendar) * 0.80))
    valid_end = min(valid_end, len(calendar) - 30)
    segments = {
        "train": (calendar[0], calendar[train_end - 1]),
        "valid": (calendar[train_end], calendar[valid_end - 1]),
        "test": (calendar[valid_end], calendar[-1]),
    }
    handler = Alpha158(
        start_time=calendar[0],
        end_time=calendar[-1],
        fit_start_time=segments["train"][0],
        fit_end_time=segments["train"][1],
        instruments=args.market,
    )
    dataset = DatasetH(handler=handler, segments=segments)
    model = LGBModel(
        loss="mse",
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        colsample_bytree=0.8,
        subsample=0.8,
        lambda_l1=1.0,
        lambda_l2=1.0,
        num_threads=8,
        num_boost_round=300,
        early_stopping_rounds=30,
    )
    workflow = qlib_workflow_run(
        run_kind="model-baseline",
        run_id=output.name,
        tracking_uri=args.tracking_uri,
        dataset_identity_sha256=dataset_provenance.get("dataset_identity_sha256"),
    )
    with workflow:
        workflow.log_params(
            {
                "model": "lightgbm",
                "features": "Alpha158",
                "market": args.market,
                "benchmark": args.benchmark,
                "account": args.account,
                "topk": args.topk,
                "n_drop": args.n_drop,
                "open_cost": args.open_cost,
                "close_cost": args.close_cost,
            }
        )
        model.fit(dataset)
        recorder_id = workflow.identity_dict()["recorder_id"]
        training_metrics = workflow.list_metrics()

    predictions = model.predict(dataset, segment="test").rename("score")
    labels = dataset.prepare("test", col_set="label")
    label = labels.iloc[:, 0].rename("label")
    aligned = pd.concat([predictions, label], axis=1).dropna()
    daily_ic = aligned.groupby(level="datetime").apply(
        lambda frame: frame["score"].corr(frame["label"]), include_groups=False
    )
    daily_rank_ic = aligned.groupby(level="datetime").apply(
        lambda frame: frame["score"].corr(frame["label"], method="spearman"),
        include_groups=False,
    )
    strategy = TopkDropoutStrategy(signal=predictions, topk=args.topk, n_drop=args.n_drop)
    backtest_end = calendar[-2]
    report, positions = backtest_daily(
        start_time=segments["test"][0],
        end_time=backtest_end,
        strategy=strategy,
        account=args.account,
        benchmark=args.benchmark,
        exchange_kwargs={
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": args.open_cost,
            "close_cost": args.close_cost,
            "min_cost": args.min_cost,
        },
    )
    excess_with_cost = report["return"] - report["bench"] - report["cost"]
    analysis = risk_analysis(excess_with_cost, freq="day")["risk"]
    metrics = {
        "ic": finite(daily_ic.mean()),
        "icir": finite(daily_ic.mean() / daily_ic.std()),
        "rank_ic": finite(daily_rank_ic.mean()),
        "rank_icir": finite(daily_rank_ic.mean() / daily_rank_ic.std()),
        "annualized_excess_return_with_cost": finite(analysis.get("annualized_return")),
        "information_ratio": finite(analysis.get("information_ratio")),
        "max_drawdown": finite(analysis.get("max_drawdown")),
        "average_turnover": finite(report.get("turnover", pd.Series(dtype=float)).mean()),
        "total_cost": finite(report["cost"].sum()),
    }
    predictions.to_frame().to_parquet(output / "predictions.parquet")
    aligned.to_parquet(output / "signals_and_labels.parquet")
    report.to_parquet(output / "portfolio_report.parquet")
    pd.to_pickle(positions, output / "positions.pkl")
    daily_ic.rename("ic").to_frame().to_parquet(output / "daily_ic.parquet")
    result = {
        "status": "succeeded",
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "provider_uri": str(provider_uri),
        "model": "lightgbm",
        "features": "Alpha158",
        "market": args.market,
        "benchmark": args.benchmark,
        "recorder_id": recorder_id,
        "qlib_workflow": workflow.identity_dict(),
        "segments": {
            name: [str(pd.Timestamp(bounds[0]).date()), str(pd.Timestamp(bounds[1]).date())]
            for name, bounds in segments.items()
        },
        "backtest_end": str(pd.Timestamp(backtest_end).date()),
        "metrics": metrics,
        "training_metrics": {key: finite(value) for key, value in training_metrics.items()},
        "provenance": {
            "dataset_identity_sha256": dataset_provenance.get("dataset_identity_sha256"),
            "snapshot_manifest_sha256": dataset_provenance.get("snapshot_manifest_sha256"),
            "qlib_builder_sha256": dataset_provenance.get("qlib_builder_sha256"),
            "qlib_version": workflow.identity_dict()["qlib_version"],
            "qlib_commit": workflow.identity_dict()["qlib_commit"],
            "baseline_config_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "market": args.market,
                        "benchmark": args.benchmark,
                        "account": args.account,
                        "topk": args.topk,
                        "n_drop": args.n_drop,
                        "open_cost": args.open_cost,
                        "close_cost": args.close_cost,
                        "min_cost": args.min_cost,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "artifacts": [
            "predictions.parquet",
            "signals_and_labels.parquet",
            "portfolio_report.parquet",
            "positions.pkl",
            "daily_ic.parquet",
        ],
    }
    with workflow:
        workflow.log_metrics(metrics)
        (output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        workflow.save_artifacts(output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
