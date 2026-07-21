"""Evaluate externally produced NLP factors against a selected Qlib snapshot.

External NLP factor artifacts (announcement_tone, irm_qa_sentiment_daily,
news_sentiment_daily) land in ``factor_candidates`` via the announcement/corpus
registries with values that derive from text fields, not market data, so the
standard RD-Agent recompute path (``factor-recompute-v1``) does not apply.
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

from .cost_model import CostModelConfig
from .factor_evaluator import evaluate_factor_values, normalize_series
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


def _window(
    values: pd.Series | pd.DataFrame, start: date, end: date
) -> pd.Series | pd.DataFrame:
    """Same datetime-level filter as factor_evaluator._validation_window."""

    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels != 2:
        return values
    datetime_level = values.index.names.index("datetime") if "datetime" in values.index.names else 0
    dates = pd.to_datetime(values.index.get_level_values(datetime_level)).tz_localize(None)
    return values[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def _coverage_stats(
    factor: pd.Series, label: pd.Series
) -> tuple[float, float]:
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
        mapped = values.index.get_level_values(level).map(convert)
        frame_or_series = values.copy()
        frame_or_series.index = frame_or_series.index.set_levels(mapped, level=level)
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
            raise ValueError(
                f"shape hint {hint!r} contradicts the observed shape {observed!r}"
            )
    return observed


def _insufficient(shape: str, reasons: list[str]) -> dict[str, Any]:
    return {"status": "insufficient_evidence", "shape": shape, "metrics": None, "reasons": reasons}


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
    label = normalize_series(
        _window(benchmark_forward_returns, valid_start, valid_end), "label"
    )
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
    direction_count = max(1, int(np.ceil(len(days) * 0.20)))
    if direction_count >= len(days):
        return _insufficient(
            SHAPE_MARKET_TIMESERIES,
            ["signal window cannot be split into direction and selection periods"],
        )
    direction_end = days[direction_count - 1]
    selection_start = days[direction_count]
    direction_frame = joined[joined.index <= direction_end]
    selection = joined[joined.index >= selection_start]
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
    costs = cost_model or CostModelConfig()
    screening_cost_rate = costs.factor_screening_rate(reference_order_value=reference_order_value)
    quantile_count = max(2, int(config.quantile_count))
    ranks = directed_signal.rank(method="average", pct=True)
    top_bucket = ranks >= 1.0 - 1.0 / quantile_count
    bottom_bucket = ranks <= 1.0 / quantile_count
    position = pd.Series(0.0, index=selection.index)
    position[top_bucket] = 1.0
    position[bottom_bucket] = -1.0
    daily_long_short = position * selection["label"]
    turnover_daily = position.diff().abs()
    turnover_daily.iloc[0] = abs(float(position.iloc[0]))
    daily_net = daily_long_short - turnover_daily * screening_cost_rate
    top_return = float(selection["label"][top_bucket].mean())
    bottom_return = float(selection["label"][bottom_bucket].mean())
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
        "cost_adjusted_return": float(daily_net.mean() * 252),
        "gross_annualized_return": float(daily_long_short.mean() * 252),
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
        "direction_start": days[0].date().isoformat(),
        "direction_end": direction_end.date().isoformat(),
        "selection_start": selection_start.date().isoformat(),
        "selection_end": days[-1].date().isoformat(),
        "benchmark_instrument": benchmark_instrument,
        "evaluation_shape": SHAPE_MARKET_TIMESERIES,
        "external_evaluator_version": config.version,
        "cost_rate": screening_cost_rate,
        "cost_model": costs.to_dict(),
        "cost_reference_order_value": reference_order_value,
    }
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
            float((item.get("metrics") or {}).get("hac_p_value", 1.0)) for item in family
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

    Items with status "failed" (operational errors) are skipped, mirroring the
    RD-Agent import in worker._import_factor_evaluations; the candidate then
    stays in its previous state for a later retry.
    """

    imported = []
    for item in result.get("evaluations", []):
        status = item.get("status")
        if status not in {"ok", "insufficient_evidence"}:
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
