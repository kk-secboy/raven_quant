from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_LABEL_HORIZON_TRADING_DAYS = 2
MODEL_FINAL_OOS_EMBARGO_TRADING_DAYS = 5


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/work/manifest.json")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("contract_version") != "model-sandbox-input-v1":
        raise ValueError("model sandbox input contract is invalid")
    model_type = str(manifest["model_type"])
    if model_type not in {"Tabular", "TimeSeries"}:
        raise ValueError("only Tabular and TimeSeries RD-Agent models are supported")
    features = manifest["feature_set"]["features"]
    if not isinstance(features, dict) or not 1 <= len(features) <= 512:
        raise ValueError("governed feature set is invalid")
    periods = manifest["periods"]
    if periods["valid_end"] >= periods["test_start"]:
        raise ValueError("model validation reaches the sealed final OOS")
    calendar_path = Path(str(manifest["provider_uri"])) / "calendars" / "day.txt"
    calendar = [
        value.strip()
        for value in calendar_path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    positions = {value: index for index, value in enumerate(calendar)}
    try:
        train_valid_gap = (
            positions[periods["valid_start"]] - positions[periods["train_end"]] - 1
        )
    except KeyError as exc:
        raise ValueError("model periods are outside the governed trading calendar") from exc
    if train_valid_gap < MODEL_LABEL_HORIZON_TRADING_DAYS:
        raise ValueError(
            "model train/validation boundary does not purge the forward label horizon"
        )
    fit_valid_end_position = (
        positions[periods["valid_end"]] - MODEL_LABEL_HORIZON_TRADING_DAYS
    )
    if fit_valid_end_position < positions[periods["valid_start"]]:
        raise ValueError("model validation window is shorter than its label purge")
    fit_valid_end = calendar[fit_valid_end_position]
    if periods["test_start"] in positions:
        valid_test_gap = (
            positions[periods["test_start"]] - positions[periods["valid_end"]] - 1
        )
        if valid_test_gap < MODEL_FINAL_OOS_EMBARGO_TRADING_DAYS:
            raise ValueError("model validation/final-OOS embargo is too short")
    prediction_segment = str(manifest.get("prediction_segment") or "valid")
    if prediction_segment not in {"valid", "test"}:
        raise ValueError("model prediction segment must be valid or test")
    inference_only = manifest.get("inference_only") is True
    if inference_only:
        if prediction_segment != "test" or manifest.get("final_oos_opened") is not False:
            raise ValueError("live inference must use the sealed test segment without final OOS")
        if periods["test_start"] != periods["test_end"]:
            raise ValueError("live inference must produce exactly one signal date")
    elif prediction_segment == "test" and manifest.get("final_oos_opened") is not True:
        raise ValueError("formal model prediction requires an opened final OOS ledger")
    prediction_start = periods[f"{prediction_segment}_start"]
    prediction_end = periods[f"{prediction_segment}_end"]

    import qlib
    import torch
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.model.pytorch_general_nn import GeneralPTNN
    from qlib.data.dataset import DatasetH, TSDatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    from qlib.workflow.record_temp import PortAnaRecord

    seed = int(manifest["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass

    sys.path.insert(0, "/work")
    qlib.init(provider_uri=manifest["provider_uri"], region="cn")
    names = list(features)
    expressions = [features[name] for name in names]
    qlib_loader: dict[str, Any] = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": {
                "feature": [expressions, names],
                "label": [["Ref($close, -2)/Ref($close, -1)-1"], ["LABEL0"]],
            }
        },
    }
    additional_factors = str(manifest.get("additional_factors_path") or "").strip()
    data_loader: dict[str, Any]
    if additional_factors:
        factor_path = Path(additional_factors)
        if not factor_path.is_file() or not factor_path.is_relative_to(Path("/work")):
            raise ValueError("additional factor values are outside the isolated workspace")
        data_loader = {
            "class": "NestedDataLoader",
            "kwargs": {
                "join": "left",
                "dataloader_l": [
                    qlib_loader,
                    {
                        "class": "StaticDataLoader",
                        "kwargs": {"config": {"feature": str(factor_path)}},
                    },
                ],
            },
        }
    else:
        data_loader = qlib_loader
    handler = DataHandlerLP(
        instruments=manifest.get("universe", "cn_all"),
        start_time=periods["train_start"],
        end_time=prediction_end,
        data_loader=data_loader,
        infer_processors=[
            {
                "class": "RobustZScoreNorm",
                "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True,
                    "fit_start_time": periods["train_start"],
                    "fit_end_time": periods["train_end"],
                },
            },
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
        ],
    )
    segments = {
        "train": (periods["train_start"], periods["train_end"]),
        # The last label-horizon sessions are prediction-only.  Excluding them
        # from early stopping keeps the pre-final truncated provider and the
        # formal/full provider on the same validation information set.
        "valid": (periods["valid_start"], fit_valid_end),
        "test": (prediction_start, prediction_end),
    }
    model_engine = str(manifest.get("model_engine") or "rdagent_pytorch")
    feature_count = len(features) + int(manifest.get("additional_factor_count") or 0)
    if model_engine == "lightgbm_baseline":
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
            num_threads=min(max(int(os.getenv("MODEL_SANDBOX_N_JOBS", "2")), 1), 8),
            num_boost_round=300,
            early_stopping_rounds=30,
            seed=seed,
            feature_fraction_seed=seed,
            bagging_seed=seed,
            data_random_seed=seed,
        )
    elif model_type == "TimeSeries":
        dataset = TSDatasetH(handler=handler, segments=segments, step_len=20)
        model_kwargs = {"num_features": feature_count, "num_timesteps": 20}
    else:
        dataset = DatasetH(handler=handler, segments=segments)
        model_kwargs = {"num_features": feature_count}
    hyperparameters = manifest.get("training_hyperparameters") or {}
    if model_engine == "rdagent_pytorch":
        model = GeneralPTNN(
            n_epochs=min(max(int(hyperparameters.get("n_epochs", 100)), 1), 500),
            lr=min(max(float(hyperparameters.get("lr", 2e-4)), 1e-7), 1.0),
            early_stop=min(max(int(hyperparameters.get("early_stop", 10)), 1), 100),
            batch_size=min(max(int(hyperparameters.get("batch_size", 256)), 8), 8192),
            weight_decay=min(max(float(hyperparameters.get("weight_decay", 1e-4)), 0.0), 10.0),
            metric="loss",
            loss="mse",
            n_jobs=min(max(int(os.getenv("MODEL_SANDBOX_N_JOBS", "2")), 1), 8),
            GPU=-1,
            seed=seed,
            pt_model_uri="model.model_cls",
            pt_model_kwargs=model_kwargs,
        )
    elif model_engine != "lightgbm_baseline":
        raise ValueError("model engine is not governed")
    output = Path("/work/output")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    if model_engine == "rdagent_pytorch":
        model.fit(dataset, save_path=str(checkpoint))
    else:
        model.fit(dataset)
        checkpoint.write_bytes(b"governed-lightgbm-baseline-no-pickle\n")
    if model_engine == "rdagent_pytorch":
        predictions = model.predict(dataset).rename("score").sort_index()
    else:
        predictions = model.predict(dataset, segment="test").rename("score").sort_index()
    if predictions.empty:
        raise ValueError("independent model validation produced no predictions")
    prediction_path = output / "predictions.parquet"
    predictions.to_frame().to_parquet(prediction_path)
    if inference_only:
        metrics: dict[str, float] = {}
    else:
        labels = dataset.prepare("test", col_set="label")
        label = labels.iloc[:, 0].rename("label")
        aligned = pd.concat([predictions, label], axis=1).dropna()
        if aligned.empty:
            raise ValueError("independent model validation has no aligned labels")
        daily_ic = aligned.groupby(level="datetime").apply(
            lambda frame: frame["score"].corr(frame["label"]), include_groups=False
        )
        daily_rank_ic = aligned.groupby(level="datetime").apply(
            lambda frame: frame["score"].corr(frame["label"], method="spearman"),
            include_groups=False,
        )
        exchange_kwargs = {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": float(manifest.get("open_cost", 0.0005)),
            "close_cost": float(manifest.get("close_cost", 0.0015)),
            "min_cost": float(manifest.get("min_cost", 5.0)),
        }
        with R.start(experiment_name="quantlab-independent-model"):
            recorder = R.get_recorder()
            recorder.log_params(**{"candidate_id": manifest["candidate_id"], "seed": seed})
            record = PortAnaRecord(
                recorder,
                config={
                    "strategy": {
                        "class": "TopkDropoutStrategy",
                        "module_path": "qlib.contrib.strategy",
                        "kwargs": {
                            "signal": predictions,
                            "topk": int(manifest.get("topk", 50)),
                            "n_drop": int(manifest.get("n_drop", 5)),
                        },
                    },
                    "backtest": {
                        "start_time": prediction_start,
                        "end_time": prediction_end,
                        "account": float(manifest.get("account", 100_000_000)),
                        "benchmark": manifest.get("benchmark", "SH000300"),
                        "exchange_kwargs": exchange_kwargs,
                    },
                },
                risk_analysis_freq="day",
            )
            record.generate()
            report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        excess = report["return"] - report["bench"] - report["cost"]
        risk = risk_analysis(excess, freq="day")["risk"]
        metrics = {
            "ic": finite(daily_ic.mean()),
            "icir": finite(daily_ic.mean() / daily_ic.std()),
            "rank_ic": finite(daily_rank_ic.mean()),
            "rank_icir": finite(daily_rank_ic.mean() / daily_rank_ic.std()),
            "information_ratio": finite(risk.get("information_ratio")),
            "annualized_excess_return_with_cost": finite(risk.get("annualized_return")),
            "max_drawdown": finite(risk.get("max_drawdown")),
            "total_cost": finite(report["cost"].sum()),
            "average_turnover": finite(
                report.get("turnover", pd.Series(dtype=float)).mean()
            ),
        }
        aligned.to_parquet(output / "signals_and_labels.parquet")
        report.to_parquet(output / "portfolio_report.parquet")
    result = {
        "status": "passed",
        "candidate_id": manifest["candidate_id"],
        "seed": seed,
        "model_type": model_type,
        "model_engine": model_engine,
        "metrics": metrics,
        "periods": periods,
        "prediction_segment": prediction_segment,
        "inference_only": inference_only,
        "latest_prediction_date": str(
            pd.Timestamp(predictions.index.get_level_values("datetime").max()).date()
        ),
        "predictions_path": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "feature_set_definition_sha256": manifest["feature_set"]["definition_sha256"],
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "final_oos_opened": bool(manifest.get("final_oos_opened")),
    }
    if not inference_only:
        portfolio_report = output / "portfolio_report.parquet"
        result.update(
            {
                "portfolio_report_path": str(portfolio_report),
                "portfolio_report_sha256": sha256_file(portfolio_report),
            }
        )
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
