"""Frozen, bounded reference evidence for live strategy-health observations.

The formal runner writes this document before the backtest artifact manifest is
sealed.  Production health collection consumes only the summary; it never
reconstructs the formal distribution from a descendant dataset.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .research_horizon import canonical_sha256

STRATEGY_HEALTH_REFERENCE_VERSION = "strategy-health-reference-v1"
STRATEGY_HEALTH_REFERENCE_NAME = "strategy_health_reference.json"
REFERENCE_BIN_COUNT = 10
REFERENCE_MINIMUM_SESSIONS = 20
REFERENCE_MINIMUM_OBSERVATIONS = 500


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def factor_values_sha256(values: pd.Series | pd.DataFrame) -> str:
    """Hash normalized factor values without persisting a private sidecar."""

    return _digest_series(_factor_series(values))


def factor_reference_summary(
    values: pd.Series | pd.DataFrame,
    *,
    factor_id: str,
    reference_start: date,
    reference_end: date,
    source_sha256: str,
    bins: int = REFERENCE_BIN_COUNT,
    minimum_sessions: int = REFERENCE_MINIMUM_SESSIONS,
    minimum_observations: int = REFERENCE_MINIMUM_OBSERVATIONS,
) -> dict[str, Any]:
    """Reduce one formal factor to immutable reference-bin sufficient statistics."""

    series = _factor_series(values)
    dates = pd.DatetimeIndex(series.index.get_level_values("datetime")).normalize()
    mask = (dates.date >= reference_start) & (dates.date <= reference_end)
    selected = series.loc[mask]
    sessions = len(set(dates[mask].date))
    if sessions < minimum_sessions or len(selected) < minimum_observations:
        raise ValueError(f"formal factor reference is insufficient: {factor_id}")
    raw = selected.to_numpy(dtype=float)
    edges = np.unique(np.quantile(raw, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 4:
        raise ValueError(f"formal factor reference has insufficient variation: {factor_id}")
    edges[0] = -np.inf
    edges[-1] = np.inf
    counts, _ = np.histogram(raw, bins=edges)
    _require_sha256(source_sha256, field=f"formal factor {factor_id} source")
    payload = {
        "factor_id": factor_id,
        "reference_start": reference_start.isoformat(),
        "reference_end": reference_end.isoformat(),
        "reference_sessions": sessions,
        "reference_observations": len(selected),
        "bin_edges": [
            None if not math.isfinite(float(value)) else float(value) for value in edges
        ],
        "reference_counts": counts.astype(int).tolist(),
        "reference_values_sha256": _digest_series(selected),
        "source_sha256": source_sha256.lower(),
    }
    return {**payload, "summary_sha256": canonical_sha256(payload)}


def model_calibration_reference_summary(
    predictions: pd.Series | pd.DataFrame,
    labels: pd.Series | pd.DataFrame,
    *,
    label_horizon_sessions: int,
    label_contract_sha256: str,
    predictions_sha256: str,
    labels_source_sha256: str,
    minimum_sessions: int = REFERENCE_MINIMUM_SESSIONS,
    minimum_observations: int = REFERENCE_MINIMUM_OBSERVATIONS,
) -> dict[str, Any]:
    """Freeze formal score/realized-return deciles used by live calibration drift."""

    if label_horizon_sessions < 1:
        raise ValueError("model calibration label horizon is invalid")
    for value, field in (
        (label_contract_sha256, "model label contract"),
        (predictions_sha256, "formal model predictions"),
        (labels_source_sha256, "formal model labels"),
    ):
        _require_sha256(value, field=field)
    frame = _aligned(predictions, labels)
    sessions = _session_count(frame)
    if sessions < minimum_sessions or len(frame) < minimum_observations:
        raise ValueError("formal model calibration evidence is insufficient")
    deciles = _decile_summary(frame)
    realized = frame["realized_return"].to_numpy(dtype=float)
    q25, q75 = np.quantile(realized, [0.25, 0.75])
    scale = max(float((q75 - q25) / 1.349), float(np.std(realized)), 1e-8)
    payload = {
        "label_horizon_sessions": int(label_horizon_sessions),
        "label_contract_sha256": label_contract_sha256.lower(),
        "formal_predictions_sha256": predictions_sha256.lower(),
        "formal_labels_source_sha256": labels_source_sha256.lower(),
        "formal_start": _first_date(frame).isoformat(),
        "formal_end": _last_date(frame).isoformat(),
        "formal_sessions": sessions,
        "formal_observations": len(frame),
        "formal_deciles": deciles,
        "formal_realized_return_scale": scale,
        "formal_values_sha256": _digest_frame(frame),
    }
    return {**payload, "summary_sha256": canonical_sha256(payload)}


def build_strategy_health_reference(
    *,
    strategy_version_id: str,
    formal_backtest_id: str,
    formal_dataset_identity_sha256: str,
    formal_dataset_lineage_id: str,
    strategy_rules_sha256: str,
    signal_source: str,
    reference_start: date,
    reference_end: date,
    factor_summaries: Mapping[str, Mapping[str, Any]],
    model_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if signal_source not in {"factor_score", "model_prediction"}:
        raise ValueError("strategy health reference signal source is invalid")
    for value, field in (
        (formal_dataset_identity_sha256, "formal dataset identity"),
        (formal_dataset_lineage_id, "formal dataset lineage"),
        (strategy_rules_sha256, "strategy rules"),
    ):
        _require_sha256(value, field=field)
    factors = {str(key): dict(value) for key, value in sorted(factor_summaries.items())}
    if not strategy_version_id or not formal_backtest_id or not factors:
        raise ValueError("strategy health reference identity or factors are missing")
    for factor_id, summary in factors.items():
        _validate_factor_summary(summary, factor_id=factor_id)
    calibration = dict(model_calibration) if model_calibration is not None else None
    if signal_source == "model_prediction":
        _validate_model_calibration_summary(calibration)
    elif calibration is not None:
        raise ValueError("factor strategy cannot carry model calibration reference")
    payload = {
        "contract_version": STRATEGY_HEALTH_REFERENCE_VERSION,
        "strategy_version_id": strategy_version_id,
        "formal_backtest_id": formal_backtest_id,
        "formal_dataset_identity_sha256": formal_dataset_identity_sha256.lower(),
        "formal_dataset_lineage_id": formal_dataset_lineage_id.lower(),
        "strategy_rules_sha256": strategy_rules_sha256.lower(),
        "signal_source": signal_source,
        "reference_start": reference_start.isoformat(),
        "reference_end": reference_end.isoformat(),
        "factors": factors,
        "model_calibration": calibration,
    }
    return {**payload, "reference_sha256": canonical_sha256(payload)}


def validate_strategy_health_reference(
    value: Any,
    *,
    strategy_version_id: str,
    formal_backtest_id: str,
    formal_dataset_identity_sha256: str,
    formal_dataset_lineage_id: str,
    strategy_rules_sha256: str,
    expected_factor_ids: Iterable[str],
    signal_source: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("strategy health reference is missing")
    reference = dict(value)
    seal = str(reference.pop("reference_sha256", ""))
    if canonical_sha256(reference) != seal:
        raise ValueError("strategy health reference seal is invalid")
    expected = set(expected_factor_ids)
    factors = reference.get("factors")
    if (
        reference.get("contract_version") != STRATEGY_HEALTH_REFERENCE_VERSION
        or reference.get("strategy_version_id") != strategy_version_id
        or reference.get("formal_backtest_id") != formal_backtest_id
        or reference.get("formal_dataset_identity_sha256")
        != formal_dataset_identity_sha256
        or reference.get("formal_dataset_lineage_id") != formal_dataset_lineage_id
        or reference.get("strategy_rules_sha256") != strategy_rules_sha256
        or reference.get("signal_source") != signal_source
        or not isinstance(factors, Mapping)
        or set(factors) != expected
    ):
        raise ValueError("strategy health reference identity is invalid")
    for factor_id, summary in factors.items():
        _validate_factor_summary(summary, factor_id=str(factor_id))
    calibration = reference.get("model_calibration")
    if signal_source == "model_prediction":
        _validate_model_calibration_summary(calibration)
    elif calibration is not None:
        raise ValueError("factor strategy has unexpected model calibration reference")
    return {**reference, "reference_sha256": seal}


def factor_drift_from_reference(
    current_values: pd.Series | pd.DataFrame,
    *,
    reference: Mapping[str, Any],
    as_of: date,
    current_window_sessions: int,
    current_file_sha256: str,
    minimum_current_observations: int = 100,
) -> dict[str, Any]:
    factor_id = str(reference.get("factor_id") or "")
    _validate_factor_summary(reference, factor_id=factor_id)
    _require_sha256(current_file_sha256, field=f"current factor {factor_id}")
    current = _factor_series(current_values)
    all_dates = pd.DatetimeIndex(current.index.get_level_values("datetime")).normalize()
    candidate_dates = sorted({item.date() for item in all_dates if item.date() <= as_of})
    selected_dates = candidate_dates[-current_window_sessions:]
    if not selected_dates or selected_dates[-1] != as_of:
        raise ValueError(f"current factor window does not reach as_of: {factor_id}")
    selected = current.loc[np.isin(all_dates.date, selected_dates)]
    if len(selected) < minimum_current_observations:
        raise ValueError(f"current factor observations are insufficient: {factor_id}")
    edges = np.asarray(
        [
            -np.inf if index == 0 and value is None else
            np.inf if index == len(reference["bin_edges"]) - 1 and value is None else
            float(value)
            for index, value in enumerate(reference["bin_edges"])
        ],
        dtype=float,
    )
    reference_counts = np.asarray(reference["reference_counts"], dtype=float)
    current_counts, _ = np.histogram(selected.to_numpy(dtype=float), bins=edges)
    epsilon = 0.5
    reference_share = (reference_counts + epsilon) / (
        float(reference_counts.sum()) + epsilon * len(reference_counts)
    )
    current_share = (current_counts + epsilon) / (
        float(current_counts.sum()) + epsilon * len(current_counts)
    )
    raw_psi = float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )
    return {
        "reference_start": reference["reference_start"],
        "reference_end": reference["reference_end"],
        "current_start": selected_dates[0].isoformat(),
        "current_end": selected_dates[-1].isoformat(),
        "reference_observations": int(reference["reference_observations"]),
        "current_observations": len(selected),
        "bin_edges": list(reference["bin_edges"]),
        "reference_counts": [int(item) for item in reference["reference_counts"]],
        "current_counts": current_counts.astype(int).tolist(),
        "reference_values_sha256": reference["reference_values_sha256"],
        "current_values_sha256": _digest_series(selected),
        "reference_file_sha256": reference["source_sha256"],
        "materialized_file_sha256": current_file_sha256.lower(),
        "reference_summary_sha256": reference["summary_sha256"],
        "raw_psi": raw_psi,
        "normalized_psi": 1.0 - math.exp(-max(0.0, raw_psi)),
    }


def live_model_calibration_from_reference(
    predictions: pd.Series | pd.DataFrame,
    labels: pd.Series | pd.DataFrame,
    *,
    reference: Mapping[str, Any],
    minimum_live_sessions: int = 5,
    minimum_live_observations: int = 250,
) -> dict[str, Any]:
    _validate_model_calibration_summary(reference)
    live = _aligned(predictions, labels)
    sessions = _session_count(live)
    if sessions < minimum_live_sessions or len(live) < minimum_live_observations:
        raise ValueError("matured live model calibration evidence is insufficient")
    deciles = _decile_summary(live)
    formal_curve = np.asarray(
        [reference["formal_deciles"][str(index)]["mean_realized_return"] for index in range(10)],
        dtype=float,
    )
    live_curve = np.asarray(
        [deciles[str(index)]["mean_realized_return"] for index in range(10)], dtype=float
    )
    standardized = float(np.sqrt(np.mean(np.square(live_curve - formal_curve)))) / float(
        reference["formal_realized_return_scale"]
    )
    return {
        "live_start": _first_date(live).isoformat(),
        "live_end": _last_date(live).isoformat(),
        "live_sessions": sessions,
        "live_observations": len(live),
        "live_deciles": deciles,
        "live_values_sha256": _digest_frame(live),
        "standardized_decile_rmse": standardized,
        "model_calibration_drift": 1.0 - math.exp(-max(0.0, standardized)),
    }


def _validate_factor_summary(value: Any, *, factor_id: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"formal factor summary is missing: {factor_id}")
    summary = dict(value)
    seal = str(summary.pop("summary_sha256", ""))
    edges = summary.get("bin_edges")
    counts = summary.get("reference_counts")
    if (
        summary.get("factor_id") != factor_id
        or canonical_sha256(summary) != seal
        or not isinstance(edges, list)
        or not isinstance(counts, list)
        or len(edges) != len(counts) + 1
        or len(edges) < 4
        or edges[0] is not None
        or edges[-1] is not None
        or int(summary.get("reference_sessions") or 0) < REFERENCE_MINIMUM_SESSIONS
        or int(summary.get("reference_observations") or 0)
        < REFERENCE_MINIMUM_OBSERVATIONS
    ):
        raise ValueError(f"formal factor summary is invalid: {factor_id}")
    _require_sha256(summary.get("reference_values_sha256"), field="formal factor values")
    _require_sha256(summary.get("source_sha256"), field="formal factor source")


def _validate_model_calibration_summary(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("formal model calibration summary is missing")
    summary = dict(value)
    seal = str(summary.pop("summary_sha256", ""))
    if (
        canonical_sha256(summary) != seal
        or int(summary.get("label_horizon_sessions") or 0) < 1
        or int(summary.get("formal_sessions") or 0) < REFERENCE_MINIMUM_SESSIONS
        or int(summary.get("formal_observations") or 0) < REFERENCE_MINIMUM_OBSERVATIONS
        or float(summary.get("formal_realized_return_scale") or 0.0) <= 0
        or set(summary.get("formal_deciles") or {}) != {str(index) for index in range(10)}
    ):
        raise ValueError("formal model calibration summary is invalid")
    for field in (
        "label_contract_sha256",
        "formal_predictions_sha256",
        "formal_labels_source_sha256",
        "formal_values_sha256",
    ):
        _require_sha256(summary.get(field), field=f"formal model calibration {field}")


def _factor_series(values: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("strategy health factor must have one column")
        series = values.iloc[:, 0]
    elif isinstance(values, pd.Series):
        series = values
    else:
        raise ValueError("strategy health factor is invalid")
    if not isinstance(series.index, pd.MultiIndex) or set(series.index.names) != {
        "datetime",
        "instrument",
    }:
        raise ValueError("strategy health factor requires datetime/instrument indexing")
    if series.index.names != ["datetime", "instrument"]:
        series = series.reorder_levels(["datetime", "instrument"])
    dates = pd.to_datetime(series.index.get_level_values("datetime"), errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric.index = pd.MultiIndex.from_arrays(
        [dates.tz_localize(None), series.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if numeric.empty or numeric.index.has_duplicates:
        raise ValueError("strategy health factor is empty or duplicated")
    return numeric.astype(float)


def _aligned(
    predictions: pd.Series | pd.DataFrame, labels: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    score = _factor_series(predictions).rename("score")
    realized = _factor_series(labels).rename("realized_return")
    frame = pd.concat([score, realized], axis=1, join="inner").dropna()
    if frame.empty or frame.index.has_duplicates:
        raise ValueError("model calibration score/return alignment is empty or duplicated")
    return frame.sort_index()


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


def _digest_series(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for (raw_date, instrument), raw_value in values.sort_index().items():
        digest.update(pd.Timestamp(raw_date).isoformat().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(instrument).encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack("<d", float(raw_value)))
    return digest.hexdigest()


def _digest_frame(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for (raw_date, instrument), row in frame.sort_index().iterrows():
        digest.update(pd.Timestamp(raw_date).isoformat().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(instrument).encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack("<d", float(row["score"])))
        digest.update(struct.pack("<d", float(row["realized_return"])))
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} SHA-256 is invalid")
    return text
