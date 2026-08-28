from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_LABEL_HORIZON_TRADING_DAYS = 2
MODEL_FINAL_OOS_EMBARGO_TRADING_DAYS = 5
MODEL_RESOURCE_POLICY_VERSION = "model-resource-policy-v3-cpu-tournament"
MODEL_DATA_CONTRACT_VERSION = "model-data-contract-v1-train-window-normalized"
GOVERNED_MODEL_ENGINES = {
    "ridge_baseline",
    "lightgbm_baseline",
    "platform_gru",
    "platform_transformer",
    "rdagent_pytorch",
}
MODEL_CHECKPOINT_FORMATS = {
    "ridge_baseline": "ridge_numeric_json",
    "lightgbm_baseline": "lightgbm_text",
    "platform_gru": "pytorch_state_dict",
    "platform_transformer": "pytorch_state_dict",
    "rdagent_pytorch": "pytorch_state_dict",
}
MODEL_CHECKPOINT_SUFFIXES = {
    "ridge_numeric_json": ".json",
    "lightgbm_text": ".txt",
    "pytorch_state_dict": ".pt",
}


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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def checkpoint_filename(model_engine: str) -> str:
    checkpoint_format = MODEL_CHECKPOINT_FORMATS[model_engine]
    return f"checkpoint{MODEL_CHECKPOINT_SUFFIXES[checkpoint_format]}"


def _require_inference_checkpoint(
    manifest: dict[str, Any], *, model_engine: str
) -> Path:
    checkpoint_format = str(manifest.get("checkpoint_format") or "")
    expected_format = MODEL_CHECKPOINT_FORMATS[model_engine]
    checkpoint = Path(str(manifest.get("checkpoint_path") or ""))
    expected_sha256 = str(manifest.get("checkpoint_sha256") or "").lower()
    if (
        checkpoint_format != expected_format
        or not checkpoint.is_file()
        or not checkpoint.is_relative_to(Path("/work"))
        or checkpoint.suffix.lower() != MODEL_CHECKPOINT_SUFFIXES[checkpoint_format]
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or sha256_file(checkpoint) != expected_sha256
    ):
        raise ValueError("live inference checkpoint identity is invalid")
    return checkpoint


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
    model_engine = str(manifest.get("model_engine") or "rdagent_pytorch")
    if model_engine not in GOVERNED_MODEL_ENGINES:
        raise ValueError("model engine is not governed")
    resource_policy = manifest.get("resource_policy") or {}
    if resource_policy.get("contract_version") != MODEL_RESOURCE_POLICY_VERSION:
        raise ValueError("model sandbox resource policy is missing or invalid")
    if resource_policy.get("model_type") != model_type:
        raise ValueError("model sandbox resource policy changed the model type")
    if resource_policy.get("model_engine") != model_engine:
        raise ValueError("model sandbox resource policy changed the model engine")
    limits = resource_policy.get("limits") or {}
    if limits.get("date_segments_modified") is not False:
        raise ValueError("model resource policy may not modify governed date segments")
    if limits.get("universe_modified") is not False:
        raise ValueError("model resource policy may not modify the governed universe")
    if limits.get("cpu_only") is not True:
        raise ValueError("model sandbox requires a CPU-only policy")
    if limits.get("qlib_evaluation_concurrency_cap") != 3:
        raise ValueError("model sandbox Qlib concurrency policy is invalid")
    if float(limits.get("reserved_service_resource_fraction", -1.0)) != 0.25:
        raise ValueError("model sandbox service reservation policy is invalid")
    if model_engine == "platform_transformer" and limits.get("exclusive_concurrency") != 1:
        raise ValueError("Transformer must use the single-concurrency lane")
    if model_engine in {"ridge_baseline", "lightgbm_baseline"} and model_type != "Tabular":
        raise ValueError("Ridge and LightGBM are restricted to the tabular lane")
    if model_engine in {"platform_gru", "platform_transformer"} and model_type != "TimeSeries":
        raise ValueError("GRU and Transformer are restricted to the sequence lane")
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
    live_retrain = manifest.get("live_retrain") is True
    if inference_only and live_retrain:
        raise ValueError("live inference and live retraining are mutually exclusive")
    if inference_only:
        if prediction_segment != "test" or manifest.get("final_oos_opened") is not False:
            raise ValueError("live inference must use the sealed test segment without final OOS")
        if periods["test_start"] != periods["test_end"]:
            raise ValueError("live inference must produce exactly one signal date")
    elif live_retrain:
        if (
            prediction_segment != "test"
            or manifest.get("final_oos_opened") is not False
            or periods["test_start"] != periods["test_end"]
        ):
            raise ValueError("live retraining must produce one non-OOS signal date")
    elif prediction_segment == "test" and manifest.get("final_oos_opened") is not True:
        raise ValueError("formal model prediction requires an opened final OOS ledger")
    prediction_start = periods[f"{prediction_segment}_start"]
    prediction_end = periods[f"{prediction_segment}_end"]

    import lightgbm as lgb
    import qlib
    import torch
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.model.linear import LinearModel
    from qlib.contrib.model.pytorch_general_nn import GeneralPTNN
    from qlib.data.dataset import DatasetH, TSDatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    from qlib.workflow.record_temp import PortAnaRecord

    seed = int(manifest["seed"])
    seed_policy = resource_policy.get("seed_policy") or {}
    if (
        seed_policy.get("effective_seed") != seed
        or seed not in (seed_policy.get("allowed_seeds") or [])
    ):
        raise ValueError("model sandbox seed does not match the governed policy")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass

    sys.path.insert(0, "/work")
    template_path = Path("/work/platform_model_templates.py")
    expected_template_sha256 = str(
        (manifest.get("execution_environment") or {}).get("model_template_sha256") or ""
    )
    if (
        not template_path.is_file()
        or len(expected_template_sha256) != 64
        or sha256_file(template_path) != expected_template_sha256
    ):
        raise ValueError("governed model template identity is invalid")
    from platform_model_templates import (  # noqa: PLC0415
        TEMPLATE_CONTRACT_VERSION,
    )

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
    model_data_contract = {
        "contract_version": MODEL_DATA_CONTRACT_VERSION,
        "feature_normalization": {
            "class": "RobustZScoreNorm",
            "fit_start_time": periods["train_start"],
            "fit_end_time": periods["train_end"],
            "clip_outlier": True,
        },
        "missing_values": {"class": "Fillna", "value": 0.0},
        "label": "Ref($close,-2)/Ref($close,-1)-1",
        "label_normalization": "CSZScoreNorm",
        "train_validation_label_purge_sessions": MODEL_LABEL_HORIZON_TRADING_DAYS,
        "validation_final_oos_embargo_sessions": MODEL_FINAL_OOS_EMBARGO_TRADING_DAYS,
        "sequence_length": 20 if model_type == "TimeSeries" else None,
        "universe": manifest.get("universe", "cn_all"),
    }
    model_data_contract_sha256 = canonical_sha256(model_data_contract)
    expected_data_contract_sha256 = str(
        manifest.get("model_data_contract_sha256") or ""
    )
    if expected_data_contract_sha256 and (
        expected_data_contract_sha256 != model_data_contract_sha256
    ):
        raise ValueError("model data contract does not match the preregistered contract")
    feature_count = len(features) + int(manifest.get("additional_factor_count") or 0)
    hyperparameters = resource_policy.get("effective_training_hyperparameters") or {}
    pytorch_engine = model_engine in {
        "rdagent_pytorch",
        "platform_gru",
        "platform_transformer",
    }
    if model_engine == "ridge_baseline":
        dataset = DatasetH(handler=handler, segments=segments)
        model = LinearModel(
            estimator="ridge",
            alpha=1.0,
            fit_intercept=False,
            include_valid=False,
        )
        model_spec: dict[str, Any] = {
            "family": "ridge",
            "alpha": 1.0,
            "fit_intercept": False,
            "training_window_normalized": True,
        }
    elif model_engine == "lightgbm_baseline":
        dataset = DatasetH(handler=handler, segments=segments)
        model = LGBModel(
            loss="mse",
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            lambda_l1=1.0,
            lambda_l2=1.0,
            num_threads=min(max(int(os.getenv("MODEL_SANDBOX_N_JOBS", "2")), 1), 4),
            num_boost_round=300,
            early_stopping_rounds=30,
            seed=seed,
            feature_fraction_seed=seed,
            bagging_seed=seed,
            data_random_seed=seed,
        )
        model_spec = {
            "family": "lightgbm",
            "preset": "pinned-lightgbm-baseline-v1",
            "feature_sampling": 0.8,
            "row_sampling": 0.8,
            "row_sampling_frequency": 1,
        }
    elif model_engine in {"platform_gru", "platform_transformer"}:
        dataset = TSDatasetH(handler=handler, segments=segments, step_len=20)
        if model_engine == "platform_gru":
            model_uri = "platform_model_templates.GovernedGRU"
            model_kwargs = {
                "num_features": feature_count,
                "num_timesteps": 20,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.1,
            }
            model_spec = {
                "family": "gru",
                "layers": 1,
                "hidden_size": 32,
                "dropout": 0.1,
                "dropout_location": "final_hidden_state",
                "sequence_length": 20,
                "template_contract_version": TEMPLATE_CONTRACT_VERSION,
            }
        else:
            model_uri = "platform_model_templates.GovernedTransformer"
            model_kwargs = {
                "num_features": feature_count,
                "num_timesteps": 20,
                "d_model": 32,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.1,
            }
            model_spec = {
                "family": "transformer",
                "layers": 2,
                "d_model": 32,
                "nhead": 4,
                "dropout": 0.1,
                "sequence_length": 20,
                "template_contract_version": TEMPLATE_CONTRACT_VERSION,
            }
        model = GeneralPTNN(
            n_epochs=min(max(int(hyperparameters.get("n_epochs", 12)), 1), 12),
            lr=min(max(float(hyperparameters.get("lr", 2e-4)), 1e-7), 1.0),
            early_stop=min(max(int(hyperparameters.get("early_stop", 3)), 1), 3),
            batch_size=min(max(int(hyperparameters.get("batch_size", 256)), 32), 2048),
            weight_decay=min(
                max(float(hyperparameters.get("weight_decay", 1e-4)), 0.0), 10.0
            ),
            metric="loss",
            loss="mse",
            n_jobs=min(max(int(os.getenv("MODEL_SANDBOX_N_JOBS", "2")), 1), 4),
            GPU=-1,
            seed=seed,
            pt_model_uri=model_uri,
            pt_model_kwargs=model_kwargs,
        )
    elif model_engine == "rdagent_pytorch":
        if model_type == "TimeSeries":
            dataset = TSDatasetH(handler=handler, segments=segments, step_len=20)
            model_kwargs = {"num_features": feature_count, "num_timesteps": 20}
        else:
            dataset = DatasetH(handler=handler, segments=segments)
            model_kwargs = {"num_features": feature_count}
        model = GeneralPTNN(
            n_epochs=min(max(int(hyperparameters.get("n_epochs", 12)), 1), 12),
            lr=min(max(float(hyperparameters.get("lr", 2e-4)), 1e-7), 1.0),
            early_stop=min(max(int(hyperparameters.get("early_stop", 3)), 1), 3),
            batch_size=min(max(int(hyperparameters.get("batch_size", 256)), 32), 2048),
            weight_decay=min(max(float(hyperparameters.get("weight_decay", 1e-4)), 0.0), 10.0),
            metric="loss",
            loss="mse",
            n_jobs=min(max(int(os.getenv("MODEL_SANDBOX_N_JOBS", "2")), 1), 4),
            GPU=-1,
            seed=seed,
            pt_model_uri="model.model_cls",
            pt_model_kwargs=model_kwargs,
        )
        model_spec = {
            "family": "rdagent_pytorch",
            "candidate_code_sha256": manifest["code_sha256"],
            "sequence_length": 20 if model_type == "TimeSeries" else None,
        }
    else:  # pragma: no cover - guarded before Qlib initialization
        raise ValueError("model engine is not governed")
    output = Path("/work/output")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_format = MODEL_CHECKPOINT_FORMATS[model_engine]
    if inference_only:
        checkpoint = _require_inference_checkpoint(manifest, model_engine=model_engine)
        if checkpoint_format == "ridge_numeric_json":
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            coefficients = checkpoint_payload.get("coefficients")
            if (
                not isinstance(checkpoint_payload, dict)
                or checkpoint_payload.get("format") != "ridge-numeric-v1"
                or checkpoint_payload.get("model_engine") != model_engine
                or checkpoint_payload.get("model_spec_sha256")
                != canonical_sha256(model_spec)
                or not isinstance(coefficients, list)
                or checkpoint_payload.get("feature_count") != feature_count
                or len(coefficients) != feature_count
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in coefficients
                )
                or not isinstance(checkpoint_payload.get("intercept"), (int, float))
                or isinstance(checkpoint_payload.get("intercept"), bool)
                or not math.isfinite(float(checkpoint_payload["intercept"]))
            ):
                raise ValueError("Ridge checkpoint violates the governed numeric schema")
            model.coef_ = np.asarray(coefficients, dtype=np.float64)
            model.intercept_ = float(checkpoint_payload["intercept"])
        elif checkpoint_format == "lightgbm_text":
            model.model = lgb.Booster(model_file=str(checkpoint))
            if model.model.num_feature() != feature_count:
                raise ValueError("LightGBM checkpoint feature count changed")
        else:
            try:
                state_dict = torch.load(
                    checkpoint,
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            except TypeError as exc:  # fail closed on an older unsafe torch
                raise ValueError("PyTorch weights-only loading is unavailable") from exc
            if (
                not isinstance(state_dict, dict)
                or not state_dict
                or len(state_dict) > 4096
                or any(
                    not isinstance(key, str)
                    or not re.fullmatch(r"[A-Za-z0-9_.]+", key)
                    or not isinstance(value, torch.Tensor)
                    for key, value in state_dict.items()
                )
                or sum(value.numel() * value.element_size() for value in state_dict.values())
                > 2 * 1024 * 1024 * 1024
                or any(not torch.isfinite(value).all() for value in state_dict.values())
            ):
                raise ValueError("PyTorch checkpoint is not a finite state_dict")
            model.dnn_model.load_state_dict(state_dict, strict=True)
            model.fitted = True
    else:
        checkpoint = output / checkpoint_filename(model_engine)
        if pytorch_engine:
            # Qlib GeneralPTNN writes exactly the best state_dict, not a model
            # object.  Daily inference later reloads it with weights_only=True.
            model.fit(dataset, save_path=str(checkpoint))
        elif model_engine == "lightgbm_baseline":
            model.fit(dataset)
            model.model.save_model(str(checkpoint))
        else:
            model.fit(dataset)
            ridge_payload = {
                "coefficients": [float(value) for value in np.asarray(model.coef_).ravel()],
                "feature_count": feature_count,
                "format": "ridge-numeric-v1",
                "intercept": float(model.intercept_),
                "model_engine": model_engine,
                "model_spec_sha256": canonical_sha256(model_spec),
            }
            checkpoint.write_text(
                json.dumps(ridge_payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
    if pytorch_engine:
        predictions = model.predict(dataset).rename("score").sort_index()
    else:
        predictions = model.predict(dataset, segment="test").rename("score").sort_index()
    if predictions.empty:
        raise ValueError("independent model validation produced no predictions")
    prediction_path = output / "predictions.parquet"
    predictions.to_frame().to_parquet(prediction_path)
    if inference_only or live_retrain:
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
        "model_spec": model_spec,
        "model_spec_sha256": canonical_sha256(model_spec),
        "resource_policy": resource_policy,
        "model_data_contract": model_data_contract,
        "model_data_contract_sha256": model_data_contract_sha256,
        "metrics": metrics,
        "periods": periods,
        "prediction_segment": prediction_segment,
        "inference_only": inference_only,
        "live_retrain": live_retrain,
        "latest_prediction_date": str(
            pd.Timestamp(predictions.index.get_level_values("datetime").max()).date()
        ),
        "predictions_path": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_format": checkpoint_format,
        "checkpoint_reused": inference_only,
        "feature_set_definition_sha256": manifest["feature_set"]["definition_sha256"],
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "final_oos_opened": bool(manifest.get("final_oos_opened")),
    }
    if not inference_only and not live_retrain:
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
