from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_data.availability import filter_available

from .style_exposures import NON_STYLE_COLUMNS
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


# ---------------------------------------------------------------------------
# Barra CNE5/6-style structured risk model (simplified, medium/low frequency)
# ---------------------------------------------------------------------------
#
# The structured model complements the Qlib shrinkage estimator above:
# ``estimate_covariance`` stays the sample-covariance fallback, while the
# structured model decomposes risk as ``cov = B F B' + D`` from point-in-time
# style/industry exposures. Factor returns are estimated by daily
# cross-sectional weighted least squares of stock returns on exposures
# (cap-weighted, the standard Barra choice; equal weights when no weight
# panel is supplied), the factor covariance is an exponentially weighted
# covariance of the factor-return series (half-life 90 trading days), and the
# specific risk is the EWMA variance of each stock's regression residuals
# (same half-life), with thin-history stocks shrunk to the cross-sectional
# median. Industry memberships enter through the platform read-side
# availability guard (effective date plus the conservative publication lag),
# so no future membership can leak into an exposure.

STRUCTURED_RISK_MODEL_VERSION = "barra-lite-cne6-v2-pit-complete"

MARKET_FACTOR = "market"
DEFAULT_FACTOR_HALF_LIFE_DAYS = 90.0
DEFAULT_SPECIFIC_HALF_LIFE_DAYS = 90.0
DEFAULT_MIN_CROSS_SECTION = 30
DEFAULT_MIN_SPECIFIC_OBSERVATIONS = 20
MIN_SPECIFIC_VARIANCE = 1e-8


@dataclass(frozen=True)
class PortfolioRiskReport:
    """Portfolio risk decomposition into market/style/industry/specific parts."""

    volatility: float
    total_variance: float
    market_variance: float
    style_variance: float
    industry_variance: float
    specific_variance: float
    factor_exposures: pd.Series
    factor_contributions: pd.Series
    annualized: bool = False


@dataclass(frozen=True)
class StructuredRiskModel:
    """Structured risk model artifact: cov = B F B' + D.

    ``exposures`` (B) is indexed by instrument with columns ``market``, the
    style factors and one dummy per non-reference industry;
    ``factor_covariance`` (F) is the PSD-clipped EWMA factor covariance;
    ``specific_variance`` (diagonal D) holds per-instrument specific variance.
    The three pieces are exposed separately so a future portfolio optimizer
    can consume the factor structure directly instead of the dense matrix.
    """

    as_of: pd.Timestamp
    style_factors: tuple[str, ...]
    industry_factors: tuple[str, ...]
    exposures: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    reference_industry: str | None = None
    version: str = STRUCTURED_RISK_MODEL_VERSION

    @property
    def factor_names(self) -> list[str]:
        return list(self.factor_covariance.columns)

    def covariance(self, instruments: list[str] | pd.Index | None = None) -> pd.DataFrame:
        """Dense asset covariance B F B' + D, drop-in for estimate_covariance."""

        exposures = self.exposures
        specific = self.specific_variance
        if instruments is not None:
            wanted = pd.Index([str(item) for item in instruments])
            missing = wanted.difference(exposures.index)
            if len(missing):
                raise ValueError(
                    f"structured risk model has no exposures for: {list(missing)[:5]}"
                )
            exposures = exposures.reindex(wanted)
            specific = specific.reindex(wanted)
        factors = self.factor_covariance.to_numpy(dtype=float)
        base = exposures.to_numpy(dtype=float)
        dense = base @ factors @ base.T
        dense = (dense + dense.T) / 2.0
        dense[np.diag_indices_from(dense)] += specific.to_numpy(dtype=float)
        return pd.DataFrame(dense, index=exposures.index, columns=exposures.index)

    def portfolio_risk(
        self,
        weights: pd.Series,
        *,
        annualize: bool = False,
        periods_per_year: int = 252,
    ) -> PortfolioRiskReport:
        """Decompose portfolio variance into market/style/industry/specific."""

        if annualize and periods_per_year <= 0:
            raise ValueError("annualized portfolio risk requires positive periods per year")
        if not isinstance(weights, pd.Series) or weights.empty or not weights.index.is_unique:
            raise ValueError("portfolio weights must be a unique non-empty Series")
        normalized = pd.to_numeric(weights, errors="coerce").astype(float)
        normalized.index = normalized.index.astype(str)
        unknown = normalized.index.difference(self.exposures.index)
        if len(unknown):
            raise ValueError(
                f"portfolio holds instruments outside the risk model: {list(unknown)[:5]}"
            )
        if normalized.isna().any() or not np.isfinite(normalized.to_numpy()).all():
            raise ValueError("portfolio weights must be finite")

        exposures = self.exposures.reindex(normalized.index)
        specific = self.specific_variance.reindex(normalized.index)
        if specific.isna().any():
            raise ValueError("structured risk model has incomplete specific variance")
        w = normalized.to_numpy(dtype=float)
        factor_exposure = exposures.T @ normalized
        factors = self.factor_covariance
        marginal = factors.to_numpy(dtype=float) @ factor_exposure.to_numpy(dtype=float)
        contributions = factor_exposure * pd.Series(marginal, index=factor_exposure.index)
        specific_variance = float(np.dot(w**2, specific.to_numpy(dtype=float)))
        factor_variance = float(contributions.sum())
        total = factor_variance + specific_variance

        scale = float(periods_per_year) if annualize else 1.0
        return PortfolioRiskReport(
            volatility=float(np.sqrt(max(total, 0.0) * scale)),
            total_variance=total * scale,
            market_variance=float(contributions.get(MARKET_FACTOR, 0.0)) * scale,
            style_variance=float(contributions.reindex(list(self.style_factors)).fillna(0.0).sum())
            * scale,
            industry_variance=float(
                contributions.reindex(list(self.industry_factors)).fillna(0.0).sum()
            )
            * scale,
            specific_variance=specific_variance * scale,
            factor_exposures=factor_exposure,
            factor_contributions=contributions * scale,
            annualized=annualize,
        )

    def save(self, directory: Path) -> Path:
        """Persist the model artifact (parquet pieces + JSON manifest)."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.exposures.to_parquet(directory / "exposures.parquet")
        self.factor_covariance.to_parquet(directory / "factor_covariance.parquet")
        self.specific_variance.to_frame("specific_variance").to_parquet(
            directory / "specific_variance.parquet"
        )
        manifest = {
            "version": self.version,
            "as_of": pd.Timestamp(self.as_of).isoformat(),
            "style_factors": list(self.style_factors),
            "industry_factors": list(self.industry_factors),
            "reference_industry": self.reference_industry,
        }
        (directory / "model.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> StructuredRiskModel:
        directory = Path(directory)
        manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if manifest.get("version") != STRUCTURED_RISK_MODEL_VERSION:
            raise ValueError(
                f"structured risk model version mismatch: {manifest.get('version')}"
            )
        specific = pd.read_parquet(directory / "specific_variance.parquet")[
            "specific_variance"
        ].rename(None)
        return cls(
            as_of=pd.Timestamp(manifest["as_of"]),
            style_factors=tuple(manifest["style_factors"]),
            industry_factors=tuple(manifest["industry_factors"]),
            exposures=pd.read_parquet(directory / "exposures.parquet"),
            factor_covariance=pd.read_parquet(directory / "factor_covariance.parquet"),
            specific_variance=specific,
            reference_industry=manifest.get("reference_industry"),
            version=manifest["version"],
        )


def estimate_structured_risk_model(
    returns: pd.DataFrame,
    style_exposures: pd.DataFrame,
    industry_memberships: pd.DataFrame,
    regression_weights: pd.DataFrame | None = None,
    *,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    min_specific_observations: int = DEFAULT_MIN_SPECIFIC_OBSERVATIONS,
    factor_half_life_days: float = DEFAULT_FACTOR_HALF_LIFE_DAYS,
    specific_half_life_days: float = DEFAULT_SPECIFIC_HALF_LIFE_DAYS,
) -> StructuredRiskModel:
    """Estimate the structured model from point-in-time panels.

    ``returns`` is a datetime-indexed frame of daily simple returns (NaN
    cells are dropped per date, so suspensions are tolerated).
    ``style_exposures`` is the long standardized panel written to qlib
    metadata (instrument, datetime, style columns; raw ``log_market_cap`` is
    ignored). ``industry_memberships`` carries instrument/industry/in_date/
    out_date intervals consumed through the platform availability guard.
    ``regression_weights`` optionally supplies per-date cap weights
    (instrument, datetime, weight); without it every stock weighs equally.
    """

    if (
        min_cross_section < 2
        or min_specific_observations < 2
        or not np.isfinite(factor_half_life_days)
        or factor_half_life_days <= 0
        or not np.isfinite(specific_half_life_days)
        or specific_half_life_days <= 0
    ):
        raise ValueError("structured risk model estimation parameters are invalid")
    panel = _normalize_returns(returns)
    styles = _normalize_style_panel(style_exposures)
    memberships = _normalize_membership_frame(industry_memberships)
    weights_panel = _normalize_weight_panel(regression_weights)

    dates = panel.index
    instruments = panel.columns
    industry_labels = _industry_labels(memberships, dates[-1])
    if industry_labels is None:
        raise ValueError("structured risk estimation requires point-in-time industry metadata")
    reference, industries = industry_labels
    industry_columns = [f"industry:{name}" for name in industries]
    factor_names = [MARKET_FACTOR, *styles.columns.tolist(), *industry_columns]

    factor_returns: dict[pd.Timestamp, pd.Series] = {}
    residuals: dict[str, list[tuple[pd.Timestamp, float]]] = {
        str(item): [] for item in instruments
    }
    for timestamp in dates:
        daily = panel.loc[timestamp].dropna()
        if len(daily) < min_cross_section:
            continue
        design = _daily_design(
            daily.index,
            timestamp,
            styles,
            memberships,
            industries,
            weights_panel,
        )
        if design is None:
            continue
        common = daily.index.intersection(design.index)
        if len(common) < max(min_cross_section, len(factor_names) + 1):
            continue
        x = design.loc[common, factor_names].to_numpy(dtype=float)
        y = daily.loc[common].to_numpy(dtype=float)
        if np.linalg.matrix_rank(x) < x.shape[1]:
            continue
        w = (
            weights_panel.loc[timestamp].reindex(common)
            if weights_panel is not None
            else pd.Series(1.0, index=common)
        )
        if w.isna().any():
            continue
        root = np.sqrt(w.to_numpy(dtype=float))
        solution, *_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)
        factor_returns[timestamp] = pd.Series(solution, index=factor_names)
        errors = y - x @ solution
        for instrument, error in zip(common, errors, strict=True):
            residuals[str(instrument)].append((timestamp, float(error)))

    if len(factor_returns) < 2:
        raise ValueError("structured risk estimation produced too few factor returns")
    series = pd.DataFrame(factor_returns).T.sort_index()
    covariance = _ewma_covariance(series, factor_half_life_days)
    covariance = _nearest_psd(covariance)

    specific = _specific_variances(
        residuals,
        instruments,
        half_life_days=specific_half_life_days,
        min_observations=min_specific_observations,
    )

    as_of = series.index[-1]
    exposures = _final_exposures(
        instruments, as_of, styles, memberships, industries, factor_names
    )
    return StructuredRiskModel(
        as_of=as_of,
        style_factors=tuple(styles.columns.tolist()),
        industry_factors=tuple(industry_columns),
        exposures=exposures,
        factor_covariance=covariance,
        specific_variance=specific,
        reference_industry=reference,
    )


def _normalize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty or returns.shape[1] < 2:
        raise ValueError("structured risk estimation requires at least two return series")
    frame = returns.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[frame.index.notna()]
    if not frame.index.is_unique:
        raise ValueError("structured risk returns must have a unique datetime index")
    frame.columns = frame.columns.astype(str)
    if not frame.columns.is_unique:
        raise ValueError("structured risk returns must have unique instruments")
    frame = frame.sort_index().apply(pd.to_numeric, errors="coerce")
    values = frame.to_numpy(dtype=float)
    finite = np.isfinite(values) | frame.isna().to_numpy()
    if not finite.all():
        raise ValueError("structured risk returns must be numeric")
    return frame


def _normalize_style_panel(style_exposures: pd.DataFrame) -> pd.DataFrame:
    required = {"instrument", "datetime"}
    if not required.issubset(style_exposures.columns):
        raise ValueError("style exposure panel requires instrument and datetime columns")
    style_columns = [
        column for column in style_exposures.columns if column not in NON_STYLE_COLUMNS
    ]
    if not style_columns:
        raise ValueError("style exposure panel has no standardized style columns")
    frame = style_exposures.loc[:, ["instrument", "datetime", *style_columns]].copy()
    frame["instrument"] = frame["instrument"].astype(str)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame[style_columns] = frame[style_columns].apply(pd.to_numeric, errors="coerce")
    style_values = frame[style_columns].to_numpy(dtype=float)
    if not (np.isfinite(style_values) | pd.isna(style_values)).all():
        raise ValueError("style exposure panel must not contain infinite values")
    frame = frame.dropna(subset=["instrument", "datetime"])
    frame = frame.sort_values("datetime").drop_duplicates(
        ["instrument", "datetime"], keep="last"
    )
    if frame.empty:
        raise ValueError("style exposure panel is empty")
    return frame.set_index(["datetime", "instrument"]).sort_index()


def _normalize_membership_frame(memberships: pd.DataFrame) -> pd.DataFrame:
    required = {"instrument", "industry", "in_date", "out_date"}
    if not required.issubset(memberships.columns):
        raise ValueError("industry memberships are incomplete")
    frame = memberships.loc[:, ["instrument", "industry", "in_date", "out_date"]].copy()
    frame = frame.dropna(subset=["instrument", "industry", "in_date"])
    frame["instrument"] = frame["instrument"].astype(str)
    frame["industry"] = frame["industry"].astype(str)
    frame["in_date"] = pd.to_datetime(
        frame["in_date"], errors="coerce", format="mixed"
    )
    frame["out_date"] = pd.to_datetime(
        frame["out_date"], errors="coerce", format="mixed"
    )
    if frame["in_date"].isna().any():
        raise ValueError("industry memberships contain an invalid in_date")
    if (
        frame["out_date"].notna()
        & (frame["out_date"] < frame["in_date"])
    ).any():
        raise ValueError("industry membership out_date cannot precede in_date")
    frame = frame[
        frame["instrument"].str.strip().ne("") & frame["industry"].str.strip().ne("")
    ]
    if frame.empty:
        raise ValueError("industry memberships are empty")
    return frame


def _normalize_weight_panel(weights: pd.DataFrame | None) -> pd.DataFrame | None:
    if weights is None:
        return None
    required = {"instrument", "datetime", "weight"}
    if not required.issubset(weights.columns):
        raise ValueError("regression weights require instrument, datetime and weight")
    frame = weights.loc[:, ["instrument", "datetime", "weight"]].copy()
    frame["instrument"] = frame["instrument"].astype(str)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna()
    if not np.isfinite(frame["weight"].to_numpy(dtype=float)).all():
        raise ValueError("regression weights must be finite")
    frame = frame[frame["weight"] > 0]
    frame = frame.drop_duplicates(["datetime", "instrument"], keep="last")
    if frame.empty:
        return None
    return frame.set_index(["datetime", "instrument"])["weight"]


def _industries_at(memberships: pd.DataFrame, when: pd.Timestamp) -> pd.Series:
    active = filter_available("index_member_all", memberships, when)
    if active.empty:
        return pd.Series(dtype=str)
    latest = active.sort_values("in_date").drop_duplicates("instrument", keep="last")
    return pd.Series(
        latest["industry"].to_numpy(dtype=str),
        index=pd.Index(latest["instrument"].astype(str), name="instrument"),
    )


def _industry_labels(
    memberships: pd.DataFrame, as_of: pd.Timestamp
) -> tuple[str, list[str]] | None:
    current = _industries_at(memberships, as_of)
    pool = (
        current
        if not current.empty
        else memberships["industry"].dropna().astype(str)
    )
    if len(pool) == 0:
        return None
    counts = pool.value_counts()
    # Deterministic reference: largest membership, then alphabetically first.
    reference = sorted(counts.index.tolist(), key=lambda name: (-int(counts[name]), name))[0]
    industries = sorted(name for name in counts.index if name != reference)
    if not industries:
        industries = []
    return reference, industries


def _daily_design(
    instruments: pd.Index,
    timestamp: pd.Timestamp,
    styles: pd.DataFrame,
    memberships: pd.DataFrame,
    industries: list[str],
    weights_panel: pd.Series | None,
) -> pd.DataFrame | None:
    design = pd.DataFrame(index=pd.Index(instruments.astype(str), name="instrument"))
    try:
        daily_styles = styles.loc[timestamp]
    except KeyError:
        return None
    daily_styles = daily_styles.reindex(design.index)
    if daily_styles.isna().any().any():
        keep = daily_styles.dropna().index
        design = design.loc[keep]
        daily_styles = daily_styles.loc[keep]
    for column in daily_styles.columns:
        design[column] = daily_styles[column]
    assigned = _industries_at(memberships, timestamp).reindex(design.index)
    design = design[assigned.notna()]
    if design.empty:
        return None
    assigned = assigned.loc[design.index]
    for name in industries:
        design[f"industry:{name}"] = (assigned == name).astype(float)
    if weights_panel is not None:
        try:
            daily_weights = weights_panel.loc[timestamp]
        except KeyError:
            return None
        design = design[daily_weights.reindex(design.index).notna()]
    if design.empty:
        return None
    design.insert(0, MARKET_FACTOR, 1.0)
    return design


def _ewma_covariance(series: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    if half_life_days <= 0:
        raise ValueError("EWMA half-life must be positive")
    values = series.to_numpy(dtype=float)
    ages = np.arange(len(series) - 1, -1, -1, dtype=float)
    decay = 0.5 ** (1.0 / half_life_days)
    weights = decay**ages
    weights /= weights.sum()
    mean = weights @ values
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    covariance = (covariance + covariance.T) / 2.0
    return pd.DataFrame(covariance, index=series.columns, columns=series.columns)


def _nearest_psd(covariance: pd.DataFrame) -> pd.DataFrame:
    values = covariance.to_numpy(dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    floor = max(float(np.trace(values)) / len(values), 1e-12) * 1e-8
    clipped = np.clip(eigenvalues, floor, None)
    rebuilt = (eigenvectors * clipped) @ eigenvectors.T
    rebuilt = (rebuilt + rebuilt.T) / 2.0
    return pd.DataFrame(rebuilt, index=covariance.index, columns=covariance.columns)


def _specific_variances(
    residuals: dict[str, list[tuple[pd.Timestamp, float]]],
    instruments: pd.Index,
    *,
    half_life_days: float,
    min_observations: int,
) -> pd.Series:
    decay = 0.5 ** (1.0 / half_life_days)
    estimates: dict[str, float] = {}
    for instrument in instruments.astype(str):
        history = residuals.get(instrument) or []
        if len(history) < min_observations:
            continue
        errors = np.array([error for _, error in history], dtype=float)
        ages = np.arange(len(errors) - 1, -1, -1, dtype=float)
        weights = decay**ages
        weights /= weights.sum()
        estimates[instrument] = float(np.dot(weights, errors**2))
    if not estimates:
        raise ValueError("structured risk estimation produced no specific variance")
    fallback = float(np.median(list(estimates.values())))
    result = pd.Series(
        {str(item): estimates.get(str(item), fallback) for item in instruments},
        dtype=float,
    )
    return result.clip(lower=MIN_SPECIFIC_VARIANCE)


def _final_exposures(
    instruments: pd.Index,
    as_of: pd.Timestamp,
    styles: pd.DataFrame,
    memberships: pd.DataFrame,
    industries: list[str],
    factor_names: list[str],
) -> pd.DataFrame:
    universe = pd.Index(instruments.astype(str), name="instrument")
    available = styles.loc[slice(None, as_of)]
    latest = (
        available.reset_index()
        .sort_values("datetime")
        .drop_duplicates("instrument", keep="last")
        .set_index("instrument")
    )
    latest_dates = pd.to_datetime(
        latest.reindex(universe)["datetime"], errors="coerce"
    )
    if latest_dates.isna().any() or not latest_dates.eq(pd.Timestamp(as_of)).all():
        raise ValueError("final risk exposures have stale style data")
    style_frame = latest.reindex(universe)[styles.columns].astype(float)
    if style_frame.isna().any().any() or not np.isfinite(
        style_frame.to_numpy(dtype=float)
    ).all():
        raise ValueError("final risk exposures have incomplete style data")
    assigned = _industries_at(memberships, as_of).reindex(universe)
    if assigned.isna().any():
        raise ValueError("final risk exposures have incomplete industry data")
    exposures = pd.DataFrame(1.0, index=universe, columns=[MARKET_FACTOR])
    for column in style_frame.columns:
        exposures[column] = style_frame[column]
    for name in industries:
        exposures[f"industry:{name}"] = (assigned == name).astype(float)
    # Completeness was verified above; zero dummies identify only the
    # explicitly selected reference industry.
    return exposures.reindex(columns=factor_names, fill_value=0.0)
