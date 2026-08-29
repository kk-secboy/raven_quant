"""Deterministic calibration drift for governed model-prediction strategies."""

from __future__ import annotations

import hashlib
import math
import struct
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .research_horizon import canonical_sha256

MODEL_CALIBRATION_OBSERVATION_VERSION = "model-calibration-drift-observation-v1"
MODEL_CALIBRATION_FORMULA_VERSION = "formal-vs-matured-live-decile-rmse-v1"


def build_model_calibration_observation(
    *,
    strategy_version_id: str,
    formal_predictions: pd.Series | pd.DataFrame,
    formal_labels: pd.Series | pd.DataFrame,
    live_predictions: pd.Series | pd.DataFrame,
    live_labels: pd.Series | pd.DataFrame,
    label_horizon_sessions: int,
    label_contract_sha256: str,
    formal_predictions_sha256: str,
    live_prediction_hashes: dict[str, str],
    label_materialization_manifest_sha256: str,
    label_materialized_file_sha256: str,
    current_dataset_identity_sha256: str,
    current_dataset_lineage_id: str,
    as_of: date,
    minimum_formal_sessions: int = 20,
    minimum_formal_observations: int = 500,
    minimum_live_sessions: int = 5,
    minimum_live_observations: int = 250,
) -> dict[str, Any]:
    """Compare formal-OOS and matured live score/return decile calibration."""

    for value, field in (
        (label_contract_sha256, "label contract"),
        (formal_predictions_sha256, "formal predictions"),
        (label_materialization_manifest_sha256, "label materialization manifest"),
        (label_materialized_file_sha256, "label materialization values"),
        (current_dataset_identity_sha256, "current dataset identity"),
        (current_dataset_lineage_id, "current dataset lineage"),
    ):
        _digest(value, field=field)
    if not live_prediction_hashes:
        raise ValueError("model calibration has no live prediction artifacts")
    for digest in live_prediction_hashes.values():
        _digest(digest, field="live predictions")
    if label_horizon_sessions < 1:
        raise ValueError("model calibration label horizon is invalid")

    formal = _aligned(formal_predictions, formal_labels)
    live = _aligned(live_predictions, live_labels)
    formal_sessions = _session_count(formal)
    live_sessions = _session_count(live)
    if formal_sessions < minimum_formal_sessions or len(formal) < minimum_formal_observations:
        raise ValueError("formal model calibration evidence is insufficient")
    if live_sessions < minimum_live_sessions or len(live) < minimum_live_observations:
        raise ValueError("matured live model calibration evidence is insufficient")

    formal_deciles = _decile_summary(formal)
    live_deciles = _decile_summary(live)
    formal_curve = np.asarray(
        [formal_deciles[str(index)]["mean_realized_return"] for index in range(10)],
        dtype=float,
    )
    live_curve = np.asarray(
        [live_deciles[str(index)]["mean_realized_return"] for index in range(10)],
        dtype=float,
    )
    formal_returns = formal["realized_return"].to_numpy(dtype=float)
    q25, q75 = np.quantile(formal_returns, [0.25, 0.75])
    robust_scale = max(float((q75 - q25) / 1.349), float(np.std(formal_returns)), 1e-8)
    standardized_rmse = float(np.sqrt(np.mean(np.square(live_curve - formal_curve)))) / (
        robust_scale
    )
    normalized = 1.0 - math.exp(-max(0.0, standardized_rmse))
    observed = {
        "contract_version": MODEL_CALIBRATION_OBSERVATION_VERSION,
        "formula_version": MODEL_CALIBRATION_FORMULA_VERSION,
        "metric": "model_calibration_drift",
        "strategy_version_id": strategy_version_id,
        "label_horizon_sessions": int(label_horizon_sessions),
        "label_contract_sha256": label_contract_sha256,
        "formal_predictions_sha256": formal_predictions_sha256,
        "live_prediction_hashes": dict(sorted(live_prediction_hashes.items())),
        "label_materialization_manifest_sha256": (
            label_materialization_manifest_sha256
        ),
        "label_materialized_file_sha256": label_materialized_file_sha256,
        "current_dataset_identity_sha256": current_dataset_identity_sha256,
        "current_dataset_lineage_id": current_dataset_lineage_id,
        "formal_start": _first_date(formal).isoformat(),
        "formal_end": _last_date(formal).isoformat(),
        "formal_sessions": formal_sessions,
        "formal_observations": len(formal),
        "live_start": _first_date(live).isoformat(),
        "live_end": _last_date(live).isoformat(),
        "live_sessions": live_sessions,
        "live_observations": len(live),
        "as_of": as_of.isoformat(),
        "formal_deciles": formal_deciles,
        "live_deciles": live_deciles,
        "formal_realized_return_scale": robust_scale,
        "standardized_decile_rmse": standardized_rmse,
        "model_calibration_drift": normalized,
        "formal_values_sha256": _values_sha256(formal),
        "live_values_sha256": _values_sha256(live),
        "observed_at": f"{as_of.isoformat()}T15:00:00+08:00",
    }
    return {**observed, "observation_sha256": canonical_sha256(observed)}


def validate_model_calibration_observation(
    value: Any,
    *,
    strategy_version_id: str,
    current_dataset_identity_sha256: str,
    expected_as_of: date | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("model calibration observation must be an object")
    observed = dict(value)
    seal = str(observed.pop("observation_sha256", ""))
    if canonical_sha256(observed) != seal:
        raise ValueError("model calibration observation seal is invalid")
    if (
        observed.get("contract_version") != MODEL_CALIBRATION_OBSERVATION_VERSION
        or observed.get("formula_version") != MODEL_CALIBRATION_FORMULA_VERSION
        or observed.get("strategy_version_id") != strategy_version_id
        or observed.get("current_dataset_identity_sha256")
        != current_dataset_identity_sha256
    ):
        raise ValueError("model calibration observation identity is invalid")
    for field in (
        "label_contract_sha256",
        "formal_predictions_sha256",
        "label_materialization_manifest_sha256",
        "label_materialized_file_sha256",
        "current_dataset_identity_sha256",
        "current_dataset_lineage_id",
        "formal_values_sha256",
        "live_values_sha256",
    ):
        _digest(observed.get(field), field=f"model calibration {field}")
    hashes = observed.get("live_prediction_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("model calibration live hashes are missing")
    for digest in hashes.values():
        _digest(digest, field="model calibration live predictions")
    as_of = date.fromisoformat(str(observed.get("as_of") or ""))
    if expected_as_of is not None and as_of != expected_as_of:
        raise ValueError("model calibration observation is stale")
    drift = float(observed.get("model_calibration_drift"))
    if not math.isfinite(drift) or not 0 <= drift < 1:
        raise ValueError("model calibration drift is invalid")
    for prefix in ("formal", "live"):
        if int(observed.get(f"{prefix}_sessions") or 0) < 1 or int(
            observed.get(f"{prefix}_observations") or 0
        ) < 1:
            raise ValueError("model calibration sample evidence is invalid")
        deciles = observed.get(f"{prefix}_deciles")
        if not isinstance(deciles, dict) or set(deciles) != {
            str(index) for index in range(10)
        }:
            raise ValueError("model calibration decile evidence is invalid")
    return {**observed, "observation_sha256": seal}


def _aligned(
    predictions: pd.Series | pd.DataFrame,
    labels: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    score = _series(predictions, field="score")
    realized = _series(labels, field="realized_return")
    frame = pd.concat([score, realized], axis=1, join="inner").dropna()
    if frame.empty or frame.index.has_duplicates:
        raise ValueError("model calibration score/return alignment is empty or duplicated")
    return frame.sort_index()


def _series(value: pd.Series | pd.DataFrame, *, field: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("model calibration artifacts require one value column")
        result = value.iloc[:, 0]
    elif isinstance(value, pd.Series):
        result = value
    else:
        raise ValueError("model calibration artifact is invalid")
    if not isinstance(result.index, pd.MultiIndex) or set(result.index.names) != {
        "datetime",
        "instrument",
    }:
        raise ValueError("model calibration requires datetime/instrument indexing")
    if result.index.names != ["datetime", "instrument"]:
        result = result.reorder_levels(["datetime", "instrument"])
    dates = pd.to_datetime(result.index.get_level_values("datetime"), errors="coerce")
    numeric = pd.to_numeric(result, errors="coerce")
    numeric.index = pd.MultiIndex.from_arrays(
        [dates.tz_localize(None), result.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    numeric.name = field
    return numeric.astype(float)


def _decile_summary(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rank = frame["score"].groupby(level="datetime").rank(method="first", pct=True)
    decile = np.minimum(np.ceil(rank.to_numpy(dtype=float) * 10).astype(int), 10) - 1
    grouped = frame.assign(decile=decile).groupby("decile", sort=True)
    result: dict[str, dict[str, Any]] = {}
    for index in range(10):
        try:
            sample = grouped.get_group(index)
        except KeyError as exc:
            raise ValueError("model calibration has an empty score decile") from exc
        result[str(index)] = {
            "observations": len(sample),
            "mean_score": float(sample["score"].mean()),
            "mean_realized_return": float(sample["realized_return"].mean()),
        }
    return result


def _session_count(frame: pd.DataFrame) -> int:
    return len(pd.DatetimeIndex(frame.index.get_level_values("datetime")).normalize().unique())


def _first_date(frame: pd.DataFrame) -> date:
    return pd.Timestamp(frame.index.get_level_values("datetime").min()).date()


def _last_date(frame: pd.DataFrame) -> date:
    return pd.Timestamp(frame.index.get_level_values("datetime").max()).date()


def _values_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for (raw_date, instrument), row in frame.sort_index().iterrows():
        digest.update(pd.Timestamp(raw_date).isoformat().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(instrument).encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack("<d", float(row["score"])))
        digest.update(struct.pack("<d", float(row["realized_return"])))
    return digest.hexdigest()


def _digest(value: Any, *, field: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest
