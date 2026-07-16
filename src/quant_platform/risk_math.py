from __future__ import annotations

import numpy as np

COVARIANCE_MODEL_VERSION = "daily-sample-diagonal-shrinkage-v1"


def regularize_covariance(
    covariance: np.ndarray, *, shrinkage: float = 0.10, eigenvalue_floor: float = 1e-10
) -> np.ndarray:
    values = np.asarray(covariance, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError("covariance matrix must be non-empty and square")
    if not np.isfinite(values).all():
        raise ValueError("covariance matrix must contain finite values")
    if not 0 <= shrinkage < 1 or eigenvalue_floor <= 0:
        raise ValueError("covariance regularization parameters are invalid")
    symmetric = (values + values.T) / 2.0
    diagonal = np.diag(np.diag(symmetric))
    shrunk = (1.0 - shrinkage) * symmetric + shrinkage * diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    scale = max(float(np.trace(shrunk)) / len(shrunk), 1.0)
    floored = np.maximum(eigenvalues, eigenvalue_floor * scale)
    regularized = eigenvectors @ np.diag(floored) @ eigenvectors.T
    return (regularized + regularized.T) / 2.0
