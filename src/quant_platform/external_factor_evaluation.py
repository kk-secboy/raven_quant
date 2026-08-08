"""Evaluate externally produced NLP factors against a selected Qlib snapshot.

External NLP factor artifacts (announcement_tone, irm_qa_sentiment_daily,
news_sentiment_daily) land in ``factor_candidates`` via the announcement/corpus
registries with values that derive from text fields, not market data, so the
standard container-isolated RD-Agent recompute path does not apply.
Their shapes also differ from dense cross-sectional factors, so the
cross-sectional coverage gates of ``FactorGatePolicy`` cannot apply either
(design draft 4.3: independent samples are effective decisions, not covered
instruments; 6.9: per-strategy admission gates, no universal threshold table).

Two shapes are recognized, everything else fails closed:

- ``sparse_event`` (announcement_tone, irm_qa_sentiment_daily): values exist
  only on the few event days/instruments. Evaluation reuses
  :func:`quant_platform.factor_evaluator.evaluate_factor_values` on the
  factor's non-empty event set (event-day IC/RankIC, event long-short, HAC),
  and the gate counts *independent event days* instead of coverage.
- ``market_timeseries`` (news_sentiment_daily, MARKET pseudo-instrument): no
  cross-section at all. The signal is evaluated as a timeseries against
  benchmark forward returns (timeseries IC/RankIC, quantile long-short, HAC)
  and gated on *independent signal days*.

Gate outcomes are recorded through
:meth:`quant_platform.research_store.ResearchStore.record_external_evaluation`;
multiple-testing correction reuses ``benjamini_hochberg`` exactly like
``scripts/evaluate_factor_batch.py``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .cost_model import CN_COST_SCHEDULE_BOOK, CostModelConfig, CostScheduleBook
from .factor_evaluator import (
    evaluate_factor_values,
    normalize_series,
    purged_factor_evaluation_days,
)
from .formal_validation import build_outer_walk_forward_folds
from .research_store import (
    EXTERNAL_EVALUATOR_VERSION,
    ExternalEventGatePolicy,
    ExternalGatePolicy,
    MarketTimeseriesGatePolicy,
)
from .statistical_validation import benjamini_hochberg, newey_west_mean_test

if TYPE_CHECKING:
    from .research_store import ResearchStore

SHAPE_SPARSE_EVENT = "sparse_event"
SHAPE_MARKET_TIMESERIES = "market_timeseries"
KNOWN_SHAPES = (SHAPE_SPARSE_EVENT, SHAPE_MARKET_TIMESERIES)

# Pseudo-instrument for market-level series; mirrors corpus_nlp.MARKET_INSTRUMENT
# (kept as a literal to avoid pulling the corpus NLP import chain in here).
MARKET_INSTRUMENT = "MARKET"

POLICY_BY_SHAPE: dict[str, type[ExternalGatePolicy]] = {
    SHAPE_SPARSE_EVENT: ExternalEventGatePolicy,
    SHAPE_MARKET_TIMESERIES: MarketTimeseriesGatePolicy,
}

PLACEBO_DIAGNOSTICS_VERSION = "placebo-leakage-sentinel-v1"


@dataclass(frozen=True, slots=True)
class ExternalEvaluationConfig:
    """Versioned knobs for external factor shape detection and evaluation.

    候选参数：以下取值均为保守默认值，需预注册评审后冻结（设计稿 4.3/6.9），
    不作为已评审依据。
    """

    version: str = EXTERNAL_EVALUATOR_VERSION
    # Shape detection boundaries. A factor covering almost the whole universe
    # on almost every day is a dense cross-sectional factor (rejected: use the
    # standard RD-Agent evaluation path). A factor is event-sparse when it is
    # present on at most half of the universe days OR when its mean per-day
    # instrument coverage is small — real announcement/event factors appear on
    # more than half of trading days (companies disclose on most days) while
    # still covering only a handful of instruments per day, so day coverage
    # alone cannot discriminate them from dense factors; anything in between
    # is ambiguous and fails closed.
    dense_day_coverage_rate: float = 0.95
    dense_mean_coverage_ratio: float = 0.80
    max_event_day_coverage_rate: float = 0.50
    sparse_mean_coverage_ratio: float = 0.25
    # Quantile bucket count for the market-timeseries long/short diagnostic.
    quantile_count: int = 5
    # Negative-control (shuffled-label placebo) and leakage-sentinel knobs
    # (design draft 6.8). Placebo rounds permute the factor/label link with a
    # fixed seed so the diagnostic is reproducible; the real statistic must sit
    # above placebo_percentile_threshold of the placebo distribution or the
    # evaluation is flagged. The leakage sentinel replays the statistic with
    # the factor shifted +/-1 trading day per instrument: a signal whose
    # aligned IC collapses (below leakage_collapse_ratio of the aligned value)
    # under a one-day shift has suspiciously exact timing (look-ahead or
    # same-bar contamination). Both are recorded as diagnostics, never as
    # hard rejections.
    placebo_rounds: int = 20
    placebo_seed: int = 20260721
    placebo_percentile_threshold: float = 0.95
    leakage_collapse_ratio: float = 0.5
    # Production batch evaluation additionally requires expanding-window,
    # pre-final walk-forward evidence.  The pure one-window evaluators keep
    # this disabled by default for backwards-compatible research use; the
    # durable production entrypoint enables it explicitly and binds the full
    # configuration into the evaluation evidence.
    require_rolling_walk_forward: bool = False
    rolling_train_days: int = 252
    rolling_validation_days: int = 63
    rolling_test_days: int = 63
    rolling_purge_days: int = 5
    rolling_embargo_days: int = 5
    rolling_min_folds: int = 3
    rolling_min_pass_rate: float = 0.60


# ValueError messages raised by evaluate_factor_values for caller/config bugs;
# anything else it raises reflects data insufficiency and is mapped to the
# "insufficient_evidence" outcome instead of crashing the evaluation round.
_CONFIG_ERROR_MARKERS = (
    "must not overlap",
    "min_daily_instruments",
    "coverage thresholds",
    "max_constant_day_rate",
    "label_horizon_days must be positive",
)


def _window(values: pd.Series | pd.DataFrame, start: date, end: date) -> pd.Series | pd.DataFrame:
    """Same datetime-level filter as factor_evaluator._validation_window."""

    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels != 2:
        return values
    datetime_level = values.index.names.index("datetime") if "datetime" in values.index.names else 0
    dates = pd.to_datetime(values.index.get_level_values(datetime_level)).tz_localize(None)
    return values[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def _coverage_stats(factor: pd.Series, label: pd.Series) -> tuple[float, float]:
    """(day coverage rate, mean per-day instrument ratio) vs the label universe."""

    factor_counts = factor.groupby(level="datetime").size()
    universe_counts = label.groupby(level="datetime").size()
    if universe_counts.empty:
        return 0.0, 0.0
    aligned = factor_counts.reindex(universe_counts.index, fill_value=0)
    day_coverage_rate = float((aligned > 0).mean())
    mean_coverage_ratio = float(aligned.div(universe_counts).mean())
    return day_coverage_rate, mean_coverage_ratio


_TUSHARE_INSTRUMENT_RE = re.compile(r"^(\d{6})\.(SH|SZ|BJ)$", re.IGNORECASE)


def to_qlib_instrument_format(values: Any) -> Any:
    """Normalize Tushare-style instruments (``600000.SH``) to qlib (``SH600000``).

    NLP pipelines persist Tushare ``ts_code`` values while qlib label
    universes use the exchange-prefixed form; without the mapping, factor and
    label never join. Already-normalized instruments pass through unchanged.
    """

    def convert(value: Any) -> Any:
        match = _TUSHARE_INSTRUMENT_RE.match(str(value))
        if match:
            return f"{match.group(2).upper()}{match.group(1)}"
        return value

    if isinstance(values, pd.DataFrame) and "instrument" in values.columns:
        frame = values.copy()
        frame["instrument"] = frame["instrument"].map(convert)
        return frame
    if isinstance(values.index, pd.MultiIndex):
        names = list(values.index.names)
        level = names.index("instrument") if "instrument" in names else 1
        arrays = [
            values.index.get_level_values(position).map(convert)
            if position == level
            else values.index.get_level_values(position)
            for position in range(values.index.nlevels)
        ]
        frame_or_series = values.copy()
        frame_or_series.index = pd.MultiIndex.from_arrays(arrays, names=names)
        return frame_or_series
    return values


def detect_external_factor_shape(
    factor_values: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
    *,
    valid_start: date,
    valid_end: date,
    config: ExternalEvaluationConfig | None = None,
    shape_hint: str | None = None,
) -> str:
    """Classify an external factor as sparse_event/market_timeseries; fail closed.

    An explicit hint (candidate/manifest metadata) is honored only when it
    matches the observed shape; contradictions and unknown shapes raise
    ValueError instead of guessing an evaluation path.
    """

    config = config or ExternalEvaluationConfig()
    factor = normalize_series(_window(factor_values, valid_start, valid_end), "factor")
    if factor.empty:
        raise ValueError("factor values are empty in the validation window")
    instruments = set(factor.index.get_level_values("instrument").unique())
    if instruments == {MARKET_INSTRUMENT}:
        observed = SHAPE_MARKET_TIMESERIES
    elif MARKET_INSTRUMENT in instruments:
        raise ValueError(
            "factor mixes the MARKET pseudo-instrument with real instruments; shape unknown"
        )
    else:
        label = normalize_series(_window(forward_returns, valid_start, valid_end), "label")
        day_coverage_rate, mean_coverage_ratio = _coverage_stats(factor, label)
        if (
            day_coverage_rate >= config.dense_day_coverage_rate
            and mean_coverage_ratio >= config.dense_mean_coverage_ratio
        ):
            raise ValueError(
                "factor is cross-sectionally dense "
                f"(day coverage {day_coverage_rate:.3f}, mean ratio {mean_coverage_ratio:.3f}); "
                "dense factors must use the standard RD-Agent evaluation path"
            )
        if (
            day_coverage_rate > config.max_event_day_coverage_rate
            and mean_coverage_ratio > config.sparse_mean_coverage_ratio
        ):
            raise ValueError(
                "factor day coverage "
                f"{day_coverage_rate:.3f} with mean instrument ratio {mean_coverage_ratio:.3f} "
                "is neither event-sparse nor dense; shape unknown"
            )
        observed = SHAPE_SPARSE_EVENT
    if shape_hint is not None:
        hint = str(shape_hint).strip()
        if hint not in KNOWN_SHAPES:
            raise ValueError(f"unsupported external factor shape hint: {shape_hint!r}")
        if hint != observed:
            raise ValueError(f"shape hint {hint!r} contradicts the observed shape {observed!r}")
    return observed


def _insufficient(shape: str, reasons: list[str]) -> dict[str, Any]:
    return {"status": "insufficient_evidence", "shape": shape, "metrics": None, "reasons": reasons}


# ---------------------------------------------------------------------------
# Negative-control (placebo) and leakage-sentinel diagnostics (design 6.8)
# ---------------------------------------------------------------------------


def _joined_frame(factor: pd.Series, label: pd.Series) -> tuple[pd.DataFrame, bool]:
    """Inner-joined factor/label frame; timeseries mode for single-instrument pairs."""

    factor_instruments = set(factor.index.get_level_values("instrument").unique())
    label_instruments = set(label.index.get_level_values("instrument").unique())
    timeseries = len(factor_instruments) == 1 and len(label_instruments) == 1
    if timeseries:
        frame = pd.concat(
            [factor.droplevel("instrument"), label.droplevel("instrument")],
            axis=1,
            join="inner",
        ).dropna()
    else:
        frame = pd.concat([factor, label], axis=1, join="inner").dropna()
    return frame.sort_index(), timeseries


def _ic_statistic(frame: pd.DataFrame, timeseries: bool) -> float:
    """Mean daily Spearman IC (cross-sectional) or full-series Spearman (timeseries)."""

    if frame.empty:
        return float("nan")
    if timeseries:
        if len(frame) < 5 or frame["factor"].nunique() < 2 or frame["label"].nunique() < 2:
            return float("nan")
        return float(frame["factor"].rank().corr(frame["label"].rank()))

    def daily(group: pd.DataFrame) -> float:
        if len(group) < 5 or group["factor"].nunique() < 2 or group["label"].nunique() < 2:
            return float("nan")
        return float(group["factor"].rank().corr(group["label"].rank()))

    daily_ic = frame.groupby(level="datetime", sort=True).apply(daily).dropna()
    return float(daily_ic.mean()) if len(daily_ic) else float("nan")


def _pearson_statistic(frame: pd.DataFrame, timeseries: bool) -> float:
    """Mean daily Pearson IC (cross-sectional) or full-series correlation."""

    if frame.empty:
        return float("nan")
    if timeseries:
        if len(frame) < 5 or frame["factor"].nunique() < 2 or frame["label"].nunique() < 2:
            return float("nan")
        return float(frame["factor"].corr(frame["label"]))

    def daily(group: pd.DataFrame) -> float:
        if len(group) < 5 or group["factor"].nunique() < 2 or group["label"].nunique() < 2:
            return float("nan")
        return float(group["factor"].corr(group["label"]))

    daily_ic = frame.groupby(level="datetime", sort=True).apply(daily).dropna()
    return float(daily_ic.mean()) if len(daily_ic) else float("nan")


def evaluate_external_walk_forward(
    factor_values: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
    *,
    evaluation_shape: str,
    train_start: date,
    valid_end: date,
    label_horizon_days: int = 1,
    config: ExternalEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen external factor on expanding pre-final holdout folds.

    Each fold learns only the factor direction on its validation window and
    applies that frozen direction after the configured embargo.  Test-window
    returns never influence the direction or another fold's selection.  The
    final reserved OOS window is not touched here.
    """

    config = config or ExternalEvaluationConfig()
    if evaluation_shape not in KNOWN_SHAPES:
        raise ValueError(f"unknown external factor shape: {evaluation_shape!r}")
    if label_horizon_days < 1:
        raise ValueError("label_horizon_days must be positive")
    factor = normalize_series(_window(factor_values, train_start, valid_end), "factor")
    label = normalize_series(_window(forward_returns, train_start, valid_end), "label")
    if factor.empty or label.empty:
        return {
            "status": "insufficient_evidence",
            "passed": False,
            "uses_final_test_data": False,
            "reasons": ["factor or forward-return history is empty before the final OOS"],
            "folds": [],
        }
    factor_instruments = set(factor.index.get_level_values("instrument").unique())
    if evaluation_shape == SHAPE_MARKET_TIMESERIES:
        if factor_instruments != {MARKET_INSTRUMENT}:
            raise ValueError("market timeseries walk-forward requires MARKET factor values")
        if len(set(label.index.get_level_values("instrument").unique())) != 1:
            raise ValueError("market timeseries walk-forward requires one benchmark instrument")
    elif MARKET_INSTRUMENT in factor_instruments:
        raise ValueError("sparse event walk-forward cannot consume MARKET factor values")

    factor_dates = pd.DatetimeIndex(
        factor.index.get_level_values("datetime").unique()
    ).sort_values()
    label_dates = pd.DatetimeIndex(label.index.get_level_values("datetime").unique()).sort_values()
    first_observed = max(pd.Timestamp(train_start), factor_dates[0])
    calendar = label_dates[
        (label_dates >= first_observed) & (label_dates <= pd.Timestamp(valid_end))
    ]
    purge_days = max(int(config.rolling_purge_days), int(label_horizon_days))
    embargo_days = max(int(config.rolling_embargo_days), int(label_horizon_days))
    try:
        folds = build_outer_walk_forward_folds(
            calendar,
            train_days=int(config.rolling_train_days),
            validation_days=int(config.rolling_validation_days),
            test_days=int(config.rolling_test_days),
            purge_days=purge_days,
            embargo_days=embargo_days,
        )
    except ValueError as exc:
        return {
            "status": "insufficient_evidence",
            "passed": False,
            "uses_final_test_data": False,
            "reasons": [str(exc)],
            "folds": [],
            "observed_start": first_observed.date().isoformat(),
            "observed_end": pd.Timestamp(valid_end).date().isoformat(),
        }

    evidence: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    timeseries = evaluation_shape == SHAPE_MARKET_TIMESERIES
    for fold in folds:
        valid_factor = normalize_series(
            _window(
                factor,
                date.fromisoformat(fold.validation_start),
                date.fromisoformat(fold.validation_end),
            ),
            "factor",
        )
        valid_label = normalize_series(
            _window(
                label,
                date.fromisoformat(fold.validation_start),
                date.fromisoformat(fold.validation_end),
            ),
            "label",
        )
        test_factor = normalize_series(
            _window(factor, date.fromisoformat(fold.test_start), date.fromisoformat(fold.test_end)),
            "factor",
        )
        test_label = normalize_series(
            _window(label, date.fromisoformat(fold.test_start), date.fromisoformat(fold.test_end)),
            "label",
        )
        validation_frame, _ = _joined_frame(valid_factor, valid_label)
        test_frame, _ = _joined_frame(test_factor, test_label)
        validation_days = int(validation_frame.index.get_level_values("datetime").nunique())
        test_days = int(test_frame.index.get_level_values("datetime").nunique())
        raw_validation_ic = _pearson_statistic(validation_frame, timeseries)
        raw_test_ic = _pearson_statistic(test_frame, timeseries)
        raw_test_rank_ic = _ic_statistic(test_frame, timeseries)
        if (
            validation_days < 5
            or test_days < 5
            or not all(
                np.isfinite(value) for value in (raw_validation_ic, raw_test_ic, raw_test_rank_ic)
            )
        ):
            skipped.append(
                {
                    "fold": fold.fold,
                    "validation_days": validation_days,
                    "test_days": test_days,
                    "reason": "too few independent observations or undefined IC",
                }
            )
            continue
        direction = -1.0 if raw_validation_ic < 0 else 1.0
        test_ic = float(raw_test_ic * direction)
        test_rank_ic = float(raw_test_rank_ic * direction)
        evidence.append(
            {
                "fold": fold.__dict__,
                "validation_days": validation_days,
                "test_days": test_days,
                "raw_validation_ic": float(raw_validation_ic),
                "direction": "inverted" if direction < 0 else "original",
                "raw_test_ic": float(raw_test_ic),
                "raw_test_rank_ic": float(raw_test_rank_ic),
                "test_ic": test_ic,
                "test_rank_ic": test_rank_ic,
                "passed": bool(test_ic > 0.0 and test_rank_ic > 0.0),
            }
        )
    minimum_folds = int(config.rolling_min_folds)
    if len(evidence) < minimum_folds:
        return {
            "status": "insufficient_evidence",
            "passed": False,
            "uses_final_test_data": False,
            "reasons": [
                f"evaluable walk-forward folds={len(evidence)} below required {minimum_folds}"
            ],
            "fold_count": len(evidence),
            "candidate_fold_count": len(folds),
            "skipped_folds": skipped,
            "folds": evidence,
            "purge_days": purge_days,
            "embargo_days": embargo_days,
        }
    pass_rate = float(np.mean([item["passed"] for item in evidence]))
    mean_ic = float(np.mean([item["test_ic"] for item in evidence]))
    mean_rank_ic = float(np.mean([item["test_rank_ic"] for item in evidence]))
    passed = bool(
        pass_rate >= float(config.rolling_min_pass_rate) and mean_ic > 0.0 and mean_rank_ic > 0.0
    )
    return {
        "status": "completed",
        "passed": passed,
        "fold_count": len(evidence),
        "candidate_fold_count": len(folds),
        "pass_rate": pass_rate,
        "minimum_pass_rate": float(config.rolling_min_pass_rate),
        "mean_test_ic": mean_ic,
        "mean_test_rank_ic": mean_rank_ic,
        "purge_days": purge_days,
        "embargo_days": embargo_days,
        "uses_final_test_data": False,
        "skipped_folds": skipped,
        "folds": evidence,
        "reasons": [] if passed else ["walk-forward holdout stability gate failed"],
    }


def _placebo_statistic(frame: pd.DataFrame, timeseries: bool, rng: np.random.Generator) -> float:
    """One shuffled-label placebo round: same marginals, broken factor/label link."""

    if timeseries:
        if len(frame) < 2:
            return float("nan")
        shifted = frame.copy()
        offset = int(rng.integers(1, len(frame)))
        shifted["label"] = np.roll(frame["label"].to_numpy(dtype=float), offset)
        return _ic_statistic(shifted, timeseries)
    labels = frame["label"]
    permuted = labels.groupby(level="datetime", group_keys=False).apply(
        lambda series: pd.Series(rng.permutation(series.to_numpy(dtype=float)), index=series.index)
    )
    shuffled = frame.copy()
    shuffled["label"] = permuted
    return _ic_statistic(shuffled, timeseries)


def _shifted_factor_stat(frame: pd.DataFrame, timeseries: bool, periods: int) -> float:
    """IC with the factor shifted ``periods`` trading days within each instrument.

    ``periods=-1`` pairs today's label with tomorrow's factor (look-ahead
    simulation); ``periods=+1`` replays the signal one day late.
    """

    if timeseries:
        shifted = frame.copy()
        shifted["factor"] = frame["factor"].shift(periods)
        return _ic_statistic(shifted.dropna(), timeseries)
    shifted_factor = frame.groupby(level="instrument")["factor"].shift(periods)
    shifted = frame.copy()
    shifted["factor"] = shifted_factor
    return _ic_statistic(shifted.dropna(), timeseries)


def evaluate_placebo_leakage_diagnostics(
    factor: pd.Series,
    label: pd.Series,
    *,
    config: ExternalEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Shuffled-label placebo distribution and label-shift leakage sentinel.

    Both inputs must already be normalized to the evaluation window (the
    callers pass their own normalized, windowed series). The same-bar fill
    side of the leakage contract is enforced at the execution layer
    (open-price dealing plus limit thresholds in qlib_exchange); here the
    +/-1 trading-day shift replay covers same-bar/look-ahead contamination
    of the factor values themselves. Diagnostics are recorded, never hard
    rejections.
    """

    config = config or ExternalEvaluationConfig()
    base: dict[str, Any] = {
        "version": PLACEBO_DIAGNOSTICS_VERSION,
        "rounds": int(config.placebo_rounds),
        "seed": int(config.placebo_seed),
        "placebo_percentile_threshold": float(config.placebo_percentile_threshold),
        "leakage_collapse_ratio": float(config.leakage_collapse_ratio),
    }
    if config.placebo_rounds <= 0:
        return {**base, "status": "disabled", "placebo_warning": False, "leakage_warning": False}
    frame, timeseries = _joined_frame(factor, label)
    real = _ic_statistic(frame, timeseries)
    if not np.isfinite(real):
        return {
            **base,
            "status": "undefined",
            "placebo_warning": False,
            "leakage_warning": False,
        }
    rng = np.random.default_rng(int(config.placebo_seed))
    placebo = [
        value
        for value in (
            _placebo_statistic(frame, timeseries, rng) for _ in range(int(config.placebo_rounds))
        )
        if np.isfinite(value)
    ]
    if not placebo:
        return {
            **base,
            "status": "undefined",
            "placebo_warning": False,
            "leakage_warning": False,
        }
    placebo_abs = np.abs(np.asarray(placebo, dtype=float))
    real_abs = abs(real)
    percentile = float((placebo_abs <= real_abs).mean())
    placebo_warning = percentile < float(config.placebo_percentile_threshold)

    forward = _shifted_factor_stat(frame, timeseries, -1)
    lagged = _shifted_factor_stat(frame, timeseries, 1)
    shifted_magnitudes = [abs(value) for value in (forward, lagged) if np.isfinite(value)]
    if real_abs > 1e-12 and shifted_magnitudes:
        collapse_ratio = float(max(shifted_magnitudes) / real_abs)
    else:
        collapse_ratio = None
    leakage_warning = bool(
        not placebo_warning
        and collapse_ratio is not None
        and collapse_ratio < float(config.leakage_collapse_ratio)
    )
    return {
        **base,
        "status": "ok",
        "mode": "market_timeseries" if timeseries else "cross_sectional",
        "ic_aligned": real,
        "real_abs_ic": real_abs,
        "placebo_abs_ic_mean": float(placebo_abs.mean()),
        "placebo_abs_ic_max": float(placebo_abs.max()),
        "placebo_percentile": percentile,
        "placebo_warning": bool(placebo_warning),
        "ic_factor_shift_forward1": forward if np.isfinite(forward) else None,
        "ic_factor_shift_lag1": lagged if np.isfinite(lagged) else None,
        "shift_collapse_ratio": collapse_ratio,
        "leakage_warning": leakage_warning,
    }


def evaluate_sparse_event_factor(
    factor_values: pd.Series | pd.DataFrame,
    forward_returns: pd.Series | pd.DataFrame,
    *,
    valid_start: date,
    valid_end: date,
    test_start: date,
    test_end: date,
    comparison_values: Iterable[pd.Series | pd.DataFrame] = (),
    cost_model: CostModelConfig | None = None,
    cost_schedule: CostScheduleBook | None = None,
    reference_order_value: float = 100_000.0,
    label_horizon_days: int = 1,
    config: ExternalEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a sparse event factor on its non-empty event set.

    Delegates the core metrics to ``evaluate_factor_values`` (same direction
    split, event-day IC/RankIC, event long-short, HAC, costs), so identical
    inputs yield identical metrics. The cross-sectional coverage gates are
    reported as diagnostics but not applied here; admission is gated on
    independent event days by ``ExternalEventGatePolicy``.
    """

    config = config or ExternalEvaluationConfig()
    factor = normalize_series(_window(factor_values, valid_start, valid_end), "factor")
    label = normalize_series(_window(forward_returns, valid_start, valid_end), "label")
    joined = pd.concat([factor, label], axis=1, join="inner").dropna()
    event_days = joined.index.get_level_values("datetime").unique()
    if len(event_days) < 10:
        return _insufficient(
            SHAPE_SPARSE_EVENT,
            [
                f"independent event days={len(event_days)} below the 10-day minimum "
                "required for any statistical inference"
            ],
        )
    try:
        metrics = evaluate_factor_values(
            factor_values,
            forward_returns,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            comparison_values=comparison_values,
            cost_model=cost_model,
            cost_schedule=cost_schedule,
            reference_order_value=reference_order_value,
            # The event set itself is the evaluation domain; the cross-sectional
            # coverage floor does not apply to event-driven factors. Coverage
            # metrics stay in the report as diagnostics only.
            min_daily_instruments=5,
            label_horizon_days=label_horizon_days,
        )
    except ValueError as exc:
        if any(marker in str(exc) for marker in _CONFIG_ERROR_MARKERS):
            raise
        return _insufficient(SHAPE_SPARSE_EVENT, [str(exc)])
    metrics.update(
        {
            "evaluation_shape": SHAPE_SPARSE_EVENT,
            "event_days": int(len(event_days)),
            "event_observations": int(len(joined)),
            "external_evaluator_version": config.version,
        }
    )
    metrics["placebo_diagnostics"] = evaluate_placebo_leakage_diagnostics(
        factor, label, config=config
    )
    return {"status": "ok", "shape": SHAPE_SPARSE_EVENT, "metrics": metrics, "reasons": []}


def evaluate_market_timeseries_factor(
    factor_values: pd.Series | pd.DataFrame,
    benchmark_forward_returns: pd.Series | pd.DataFrame,
    *,
    valid_start: date,
    valid_end: date,
    test_start: date,
    test_end: date,
    cost_model: CostModelConfig | None = None,
    cost_schedule: CostScheduleBook | None = None,
    reference_order_value: float = 100_000.0,
    label_horizon_days: int = 1,
    config: ExternalEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a MARKET-level signal as a timeseries against benchmark returns.

    Mirrors the ``evaluate_factor_values`` protocol (direction on the first 20%
    of days, selection on the rest, HAC inference, screening cost rate) but on
    a single-instrument timeseries: timeseries IC/RankIC plus a quantile
    long/short diagnostic on the benchmark forward returns.
    """

    config = config or ExternalEvaluationConfig()
    if valid_end >= test_start or test_start > test_end:
        raise ValueError("validation and reserved final-test windows must not overlap")
    if label_horizon_days < 1:
        raise ValueError("label_horizon_days must be positive")
    factor = normalize_series(_window(factor_values, valid_start, valid_end), "factor")
    instruments = set(factor.index.get_level_values("instrument").unique())
    if instruments != {MARKET_INSTRUMENT}:
        raise ValueError(
            "market timeseries evaluation requires MARKET pseudo-instrument factor values"
        )
    label = normalize_series(_window(benchmark_forward_returns, valid_start, valid_end), "label")
    benchmark_instruments = set(label.index.get_level_values("instrument").unique())
    if len(benchmark_instruments) != 1:
        raise ValueError("benchmark forward returns must carry exactly one instrument")
    benchmark_instrument = next(iter(benchmark_instruments))
    joined = pd.concat(
        [
            factor.droplevel("instrument"),
            label.droplevel("instrument"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    days = pd.DatetimeIndex(joined.index.unique()).sort_values()
    if len(days) < 10:
        return _insufficient(
            SHAPE_MARKET_TIMESERIES,
            [
                f"independent signal days={len(days)} below the 10-day minimum "
                "required for any statistical inference"
            ],
        )
    try:
        windows = purged_factor_evaluation_days(days, label_horizon_days=label_horizon_days)
    except ValueError as exc:
        return _insufficient(
            SHAPE_MARKET_TIMESERIES,
            [str(exc)],
        )
    direction_days = windows["direction"]
    selection_days = windows["selection"]
    assert isinstance(direction_days, pd.DatetimeIndex)
    assert isinstance(selection_days, pd.DatetimeIndex)
    direction_start = direction_days[0]
    direction_end = direction_days[-1]
    selection_start = selection_days[0]
    selection_end = selection_days[-1]
    direction_frame = joined[joined.index.isin(direction_days)]
    selection = joined[joined.index.isin(selection_days)]
    if (
        len(selection) < 10
        or direction_frame["factor"].nunique() < 2
        or direction_frame["label"].nunique() < 2
    ):
        return _insufficient(
            SHAPE_MARKET_TIMESERIES,
            ["selection/direction windows have too few observations or constant values"],
        )
    raw_valid_ic = float(direction_frame["factor"].corr(direction_frame["label"]))
    if not np.isfinite(raw_valid_ic):
        return _insufficient(
            SHAPE_MARKET_TIMESERIES, ["direction-window timeseries IC is undefined"]
        )
    direction = -1.0 if raw_valid_ic < 0 else 1.0
    directed_signal = selection["factor"] * direction
    raw_selection_ic = float(selection["factor"].corr(selection["label"]))
    raw_selection_rank_ic = float(selection["factor"].rank().corr(selection["label"].rank()))
    ic = float(directed_signal.corr(selection["label"]))
    rank_ic = float(directed_signal.rank().corr(selection["label"].rank()))
    if not (np.isfinite(ic) and np.isfinite(rank_ic)):
        return _insufficient(
            SHAPE_MARKET_TIMESERIES, ["selection-window timeseries IC is undefined"]
        )
    hac_input = pd.Series(
        directed_signal.to_numpy() * selection["label"].to_numpy(), index=selection.index
    )
    hac = newey_west_mean_test(hac_input, max_lag=label_horizon_days)
    if cost_model is not None and cost_schedule is not None:
        raise ValueError("factor evaluation accepts either cost_model or cost_schedule, not both")
    costs = cost_model
    schedule = cost_schedule or (None if costs is not None else CN_COST_SCHEDULE_BOOK)
    if schedule is not None:
        screening_cost_rate = schedule.factor_screening_rate(
            reference_order_value=reference_order_value,
            start=valid_start,
            end=valid_end,
        )
        cost_evidence = schedule.to_dict()
        cost_rate_resolution = "maximum_effective_rate_in_validation_period"
    else:
        assert costs is not None
        screening_cost_rate = costs.factor_screening_rate(
            reference_order_value=reference_order_value
        )
        cost_evidence = costs.to_dict()
        cost_rate_resolution = "explicit_flat_model"
    quantile_count = max(2, int(config.quantile_count))
    return_selection = selection.iloc[::label_horizon_days]
    return_signal = directed_signal.reindex(return_selection.index)
    ranks = return_signal.rank(method="average", pct=True)
    top_bucket = ranks >= 1.0 - 1.0 / quantile_count
    bottom_bucket = ranks <= 1.0 / quantile_count
    position = pd.Series(0.0, index=return_selection.index)
    position[top_bucket] = 1.0
    position[bottom_bucket] = -1.0
    daily_long_short = position * return_selection["label"]
    turnover_daily = position.diff().abs()
    turnover_daily.iloc[0] = abs(float(position.iloc[0]))
    daily_net = daily_long_short - turnover_daily * screening_cost_rate
    annualization = 252.0 / label_horizon_days
    top_return = float(return_selection["label"][top_bucket].mean())
    bottom_return = float(return_selection["label"][bottom_bucket].mean())
    metrics: dict[str, Any] = {
        "ic": ic,
        "icir": None,
        "rank_ic": rank_ic,
        "rank_icir": None,
        "hac_p_value": hac["p_value"],
        "hac_test": hac,
        "statistical_contract_version": hac["contract_version"],
        "turnover": float(turnover_daily.mean()),
        "max_correlation": None,
        "cost_adjusted_return": float(daily_net.mean() * annualization),
        "gross_annualized_return": float(daily_long_short.mean() * annualization),
        "return_annualization_horizon_days": label_horizon_days,
        "quantile_count": quantile_count,
        "quantile_top_return": top_return,
        "quantile_bottom_return": bottom_return,
        "quantile_spread": top_return - bottom_return,
        "raw_valid_ic": raw_valid_ic,
        "raw_selection_ic": raw_selection_ic,
        "raw_selection_rank_ic": raw_selection_rank_ic,
        "direction": "inverted" if direction < 0 else "original",
        "observations": int(len(selection)),
        "selection_days": int(len(selection)),
        "signal_days": int(len(days)),
        "direction_start": direction_start.date().isoformat(),
        "direction_end": direction_end.date().isoformat(),
        "selection_start": selection_start.date().isoformat(),
        "selection_end": selection_end.date().isoformat(),
        "direction_purge_days": windows["direction_purge_days"],
        "final_test_purge_days": windows["final_test_purge_days"],
        "benchmark_instrument": benchmark_instrument,
        "evaluation_shape": SHAPE_MARKET_TIMESERIES,
        "external_evaluator_version": config.version,
        "cost_rate": screening_cost_rate,
        "cost_model": cost_evidence,
        "cost_rate_resolution": cost_rate_resolution,
        "cost_reference_order_value": reference_order_value,
    }
    metrics["placebo_diagnostics"] = evaluate_placebo_leakage_diagnostics(
        factor, label, config=config
    )
    return {"status": "ok", "shape": SHAPE_MARKET_TIMESERIES, "metrics": metrics, "reasons": []}


def apply_family_bh_correction(evaluations: list[dict[str, Any]]) -> None:
    """In-place BH-FDR per experiment family; same padding rule as the batch script."""

    families: dict[str, list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        family = str(evaluation.get("experiment_family_id") or "")
        if family:
            families.setdefault(family, []).append(evaluation)
    for family in families.values():
        declared = max(int(item.get("experiment_count") or len(family)) for item in family)
        p_values = [
            # Undefined HAC (zero long-run variance) pads to 1.0: undefined is
            # treated as insufficient evidence, never as a significant result.
            float(
                1.0
                if (p_value := (item.get("metrics") or {}).get("hac_p_value")) is None
                else p_value
            )
            for item in family
        ]
        p_values.extend([1.0] * max(0, declared - len(p_values)))
        q_values = benjamini_hochberg(p_values)
        for item, q_value in zip(family, q_values, strict=False):
            if item.get("status") == "ok" and item.get("metrics") is not None:
                item["metrics"]["bh_q_value"] = q_value
                item["metrics"]["experiment_count"] = declared


def build_external_evidence(
    *,
    evaluation_shape: str,
    config: ExternalEvaluationConfig,
    policy: ExternalGatePolicy,
    candidate: dict[str, Any],
    dataset_identity_sha256: str,
    periods: dict[str, date],
    label_horizon_days: int,
    input_data_sha256: str,
) -> dict[str, Any]:
    """Evidence chain bound into factor_evaluations.recompute_evidence_json."""

    return {
        "executor_version": EXTERNAL_EVALUATOR_VERSION,
        "evaluation_shape": evaluation_shape,
        "policy": asdict(policy),
        "config": asdict(config),
        "candidate_code_sha256": candidate["code_sha256"],
        "candidate_values_sha256": candidate["values_sha256"],
        "authoritative_values_sha256": candidate["values_sha256"],
        "dataset_identity_sha256": dataset_identity_sha256,
        "periods": {key: value.isoformat() for key, value in periods.items()},
        "label_horizon_days": int(label_horizon_days),
        "input_data_sha256": input_data_sha256,
    }


def import_external_evaluations(
    store: ResearchStore,
    result: dict[str, Any],
    *,
    dataset: str,
    dataset_identity_sha256: str,
    periods: dict[str, date],
    artifact_path: str | Path | None,
    actor: str = "external-factor-evaluator",
) -> list[dict[str, Any]]:
    """Persist successful/insufficient external evaluations into factor_evaluations.

    Items with status "failed" (operational errors) are ledgered through
    :meth:`ResearchStore.record_failed_evaluation` as ``evaluation_failed``
    (design draft 4.2/6.6: failed trials are recorded, never silently
    dropped); the candidate may be re-evaluated later. Items referencing an
    unknown candidate id raise, because a trial that cannot be attributed to a
    ledgered candidate is a producer bug, not a skippable row.
    """

    imported = []
    for item in result.get("evaluations", []):
        status = item.get("status")
        if status not in {"ok", "insufficient_evidence"}:
            store.record_failed_evaluation(
                str(item["candidate_id"]),
                dataset=dataset,
                dataset_identity_sha256=dataset_identity_sha256,
                **periods,
                error=str(item.get("error") or "evaluation failed"),
                actor=actor,
            )
            continue
        shape = str(item.get("shape") or "")
        policy_type = POLICY_BY_SHAPE.get(shape)
        if policy_type is None:
            raise ValueError(f"unknown external evaluation shape {shape!r}; refusing to import")
        imported.append(
            store.record_external_evaluation(
                str(item["candidate_id"]),
                policy=policy_type(),
                dataset=dataset,
                dataset_identity_sha256=dataset_identity_sha256,
                **periods,
                evaluation_shape=shape,
                metrics=item.get("metrics"),
                insufficient_reasons=item.get("reasons"),
                external_evidence=item["evidence"],
                artifact_path=str(artifact_path) if status == "ok" else None,
                actor=actor,
            )
        )
    return imported
