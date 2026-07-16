from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_platform.qlib_workflow import QLIB_WORKFLOW_ADAPTER_VERSION
from quant_platform.upstream_versions import QLIB_COMMIT


def qlib_runtime_identity(_kind: str) -> dict[str, str]:
    return {"name": "qlib", "version": "test-qlib", "commit": QLIB_COMMIT}


def qlib_workflow_identity() -> dict[str, str]:
    return {
        "adapter_version": QLIB_WORKFLOW_ADAPTER_VERSION,
        "experiment_id": "test-experiment",
        "experiment_name": "quantlab-test",
        "recorder_id": "test-recorder",
        "recorder_name": "test-run",
        "tracking_backend": "postgresql",
        "artifact_backend": "file",
        "qlib_version": "0.0.dev0+gd5379c5",
        "qlib_commit": QLIB_COMMIT,
    }


class QlibRiskEstimator:
    def __init__(self, *, alpha: float, target: str, **_kwargs: Any) -> None:
        assert alpha == 0.10
        assert target == "const_var"
        self.alpha = alpha

    def predict(self, frame: pd.DataFrame, *, is_price: bool) -> np.ndarray:
        assert is_price is False
        sample = frame.cov().to_numpy(dtype=float)
        target = np.eye(len(sample)) * float(np.trace(sample) / len(sample))
        return (1.0 - self.alpha) * sample + self.alpha * target


class QlibPortfolioOptimizer:
    def __init__(self, *, method: str) -> None:
        self.method = method

    def __call__(self, *, S: pd.DataFrame) -> pd.Series:
        covariance = S.to_numpy(dtype=float)
        if self.method == "inv":
            values = 1.0 / np.sqrt(np.diag(covariance))
            return pd.Series(values / values.sum(), index=S.index)
        count = len(covariance)

        def objective(weights: np.ndarray) -> float:
            marginal = covariance @ weights
            variance = float(weights @ marginal)
            if variance <= 0 or np.any(np.abs(marginal) < 1e-14):
                return 1e12
            implied = variance / (marginal * count)
            return float(np.square(weights - implied).sum())

        result = minimize(
            objective,
            np.full(count, 1.0 / count),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * count,
            constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        )
        if not result.success:
            raise RuntimeError(result.message)
        return pd.Series(result.x, index=S.index)


def risk_analysis(
    returns: pd.Series,
    *,
    N: int,
    freq: str | None,
    mode: str,
) -> pd.DataFrame:
    assert N == 252
    assert freq is None
    if mode == "product":
        curve = (1.0 + returns).cumprod()
        mean = float(curve.iloc[-1] ** (1.0 / len(returns)) - 1.0)
        std = float(np.log1p(returns).std(ddof=1))
        annualized = float(curve.iloc[-1] ** (N / len(returns)) - 1.0)
        drawdown = float((curve / curve.cummax() - 1.0).min())
    else:
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        annualized = mean * N
        cumulative = returns.cumsum()
        drawdown = float((cumulative - cumulative.cummax()).min())
    information_ratio = mean / std * np.sqrt(N) if std else np.nan
    return pd.Series(
        {
            "mean": mean,
            "std": std,
            "annualized_return": annualized,
            "information_ratio": information_ratio,
            "max_drawdown": drawdown,
        }
    ).to_frame("risk")
