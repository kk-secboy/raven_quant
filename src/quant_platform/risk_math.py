from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .upstream_versions import QLIB_COMMIT, upstream_runtime_identity

COVARIANCE_MODEL_VERSION = f"qlib-shrink-cov-0.10-const-var-v1@{QLIB_COMMIT}"


def _load_qlib_risk_model() -> type[Any]:
    try:
        from qlib.model.riskmodel.shrink import ShrinkCovEstimator
    except ImportError as exc:  # pragma: no cover - configured runtime assertion
        raise RuntimeError("Qlib risk model runtime is unavailable") from exc
    return ShrinkCovEstimator


def validate_covariance(covariance: np.ndarray) -> np.ndarray:
    """Validate upstream covariance output without maintaining a second estimator."""

    values = np.asarray(covariance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError("covariance matrix must be non-empty and square")
    if not np.isfinite(values).all():
        raise ValueError("covariance matrix must contain finite values")
    symmetric = (values + values.T) / 2.0
    if (np.diag(symmetric) <= 0).any():
        raise ValueError("covariance matrix must contain positive variances")
    tolerance = max(float(np.trace(symmetric)) / len(symmetric), 1e-12) * 1e-9
    if float(np.linalg.eigvalsh(symmetric).min()) < -tolerance:
        raise ValueError("Qlib risk model returned a non-positive-semidefinite covariance")
    return symmetric


def estimate_covariance(
    returns: pd.DataFrame,
    *,
    estimator_factory: Callable[..., Any] | None = None,
    runtime_identity: Callable[[str], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Estimate covariance through the pinned Qlib risk-model implementation."""

    frame = returns.copy().apply(pd.to_numeric, errors="coerce")
    if frame.empty or frame.shape[1] < 2 or not frame.columns.is_unique:
        raise ValueError("Qlib covariance estimation requires at least two unique return series")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Qlib covariance estimation requires complete finite returns")
    identity_fn = runtime_identity or upstream_runtime_identity
    identity = identity_fn("qlib")
    if identity.get("commit") != QLIB_COMMIT:
        raise RuntimeError("Qlib risk model is not running from the validated commit")
    factory = estimator_factory or _load_qlib_risk_model()
    estimator = factory(
        alpha=0.10,
        target="const_var",
        nan_option="ignore",
        assume_centered=False,
        scale_return=False,
    )
    measured = estimator.predict(frame, is_price=False)
    values = validate_covariance(np.asarray(measured, dtype=float))
    return pd.DataFrame(values, index=frame.columns, columns=frame.columns)
