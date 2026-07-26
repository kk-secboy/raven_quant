from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quant_platform import external_factor_evaluation as ext

pytestmark = pytest.mark.no_database

DAYS = pd.bdate_range("2024-01-02", periods=80)
INSTRUMENTS = [f"6000{i:02d}.SH" for i in range(20)]

PERIODS = {
    "valid_start": DAYS[0].date(),
    "valid_end": DAYS[-1].date(),
    "test_start": DAYS[-1].date() + timedelta(days=10),
    "test_end": DAYS[-1].date() + timedelta(days=370),
}


def _series(rows: list[tuple], name: str) -> pd.Series:
    frame = pd.DataFrame(rows, columns=["datetime", "instrument", name])
    return frame.set_index(["datetime", "instrument"])[name].sort_index()


def _persistent_signal_inputs(
    *, effect: float = 0.03, noise: float = 0.01, seed: int = 11
) -> tuple[pd.Series, pd.Series]:
    """Cross-sectional factor with a persistent per-instrument signal."""

    rng = np.random.default_rng(seed)
    signal = {instrument: float(rng.standard_normal()) for instrument in INSTRUMENTS}
    factor_rows = [
        (day, instrument, signal[instrument] + 0.05 * float(rng.standard_normal()))
        for day in DAYS
        for instrument in INSTRUMENTS
    ]
    label_rows = [
        (day, instrument, effect * signal[instrument] + noise * float(rng.standard_normal()))
        for day in DAYS
        for instrument in INSTRUMENTS
    ]
    return _series(factor_rows, "factor"), _series(label_rows, "label")


def _random_factor_inputs(*, seed: int = 13) -> tuple[pd.Series, pd.Series]:
    """Factor and label are fully independent (the placebo null)."""

    rng = np.random.default_rng(seed)
    factor_rows = [
        (day, instrument, float(rng.standard_normal()))
        for day in DAYS
        for instrument in INSTRUMENTS
    ]
    label_rows = [
        (day, instrument, 0.01 * float(rng.standard_normal()))
        for day in DAYS
        for instrument in INSTRUMENTS
    ]
    return _series(factor_rows, "factor"), _series(label_rows, "label")


def test_real_factor_beats_the_placebo_distribution() -> None:
    factor, label = _persistent_signal_inputs()

    diagnostics = ext.evaluate_placebo_leakage_diagnostics(factor, label)

    assert diagnostics["status"] == "ok"
    assert diagnostics["mode"] == "cross_sectional"
    assert diagnostics["placebo_warning"] is False
    assert diagnostics["leakage_warning"] is False
    assert diagnostics["placebo_percentile"] >= 0.95
    assert diagnostics["real_abs_ic"] > diagnostics["placebo_abs_ic_max"]


def test_placebo_is_reproducible_under_the_fixed_seed() -> None:
    factor, label = _persistent_signal_inputs()

    first = ext.evaluate_placebo_leakage_diagnostics(factor, label)
    second = ext.evaluate_placebo_leakage_diagnostics(factor, label)

    assert first == second


def test_random_factor_is_flagged_by_the_placebo_control() -> None:
    factor, label = _random_factor_inputs()

    diagnostics = ext.evaluate_placebo_leakage_diagnostics(factor, label)

    assert diagnostics["status"] == "ok"
    assert diagnostics["placebo_warning"] is True
    assert diagnostics["placebo_percentile"] < 0.95


def test_same_bar_contaminated_factor_trips_the_leakage_sentinel() -> None:
    """Factor == same-day label: aligned IC is perfect but a +/-1 day shift
    destroys it, which is the look-ahead/same-bar signature."""

    rng = np.random.default_rng(17)
    label_rows = [
        (day, instrument, 0.02 * float(rng.standard_normal()))
        for day in DAYS
        for instrument in INSTRUMENTS
    ]
    label = _series(label_rows, "label")
    factor = label.rename("factor")

    diagnostics = ext.evaluate_placebo_leakage_diagnostics(factor, label)

    assert diagnostics["placebo_warning"] is False
    assert diagnostics["ic_aligned"] == pytest.approx(1.0)
    assert abs(diagnostics["ic_factor_shift_forward1"]) < 0.2
    assert abs(diagnostics["ic_factor_shift_lag1"]) < 0.2
    assert diagnostics["shift_collapse_ratio"] < 0.5
    assert diagnostics["leakage_warning"] is True


def test_persistent_signal_does_not_trip_the_leakage_sentinel() -> None:
    factor, label = _persistent_signal_inputs()

    diagnostics = ext.evaluate_placebo_leakage_diagnostics(factor, label)

    assert diagnostics["shift_collapse_ratio"] > 0.5
    assert diagnostics["leakage_warning"] is False


def test_market_timeseries_placebo_uses_date_shifts() -> None:
    rng = np.random.default_rng(23)
    signal = np.zeros(len(DAYS))
    for index in range(1, len(DAYS)):
        signal[index] = 0.9 * signal[index - 1] + float(rng.standard_normal())
    returns = 0.02 * signal + 0.01 * rng.standard_normal(len(DAYS))
    factor = _series(
        [
            (day, ext.MARKET_INSTRUMENT, float(value))
            for day, value in zip(DAYS, signal, strict=True)
        ],
        "factor",
    )
    label = _series(
        [(day, "SH000300", float(value)) for day, value in zip(DAYS, returns, strict=True)],
        "label",
    )

    diagnostics = ext.evaluate_placebo_leakage_diagnostics(factor, label)

    assert diagnostics["mode"] == "market_timeseries"
    assert diagnostics["placebo_warning"] is False

    random_factor = _series(
        [
            (day, ext.MARKET_INSTRUMENT, float(value))
            for day, value in zip(DAYS, rng.standard_normal(len(DAYS)), strict=True)
        ],
        "factor",
    )
    random_diagnostics = ext.evaluate_placebo_leakage_diagnostics(random_factor, label)
    assert random_diagnostics["placebo_warning"] is True


def test_placebo_can_be_disabled() -> None:
    factor, label = _persistent_signal_inputs()
    config = ext.ExternalEvaluationConfig(placebo_rounds=0)

    diagnostics = ext.evaluate_placebo_leakage_diagnostics(factor, label, config=config)

    assert diagnostics["status"] == "disabled"
    assert diagnostics["placebo_warning"] is False


def test_sparse_event_evaluation_attaches_diagnostics() -> None:
    rng = np.random.default_rng(29)
    signal = {instrument: float(rng.standard_normal()) for instrument in INSTRUMENTS}
    chosen_days = list(DAYS[::2])
    event_instruments = INSTRUMENTS[:12]
    factor_rows = [
        (day, instrument, signal[instrument] + 0.05 * float(rng.standard_normal()))
        for day in chosen_days
        for instrument in event_instruments
    ]
    label_rows = []
    for day in DAYS:
        for instrument in INSTRUMENTS:
            label_rows.append(
                (day, instrument, 0.03 * signal[instrument] + 0.01 * float(rng.standard_normal()))
            )
    factor = _series(factor_rows, "factor")
    label = _series(label_rows, "label")

    result = ext.evaluate_sparse_event_factor(factor, label, **PERIODS)

    assert result["status"] == "ok"
    diagnostics = result["metrics"]["placebo_diagnostics"]
    assert diagnostics["version"] == ext.PLACEBO_DIAGNOSTICS_VERSION
    assert diagnostics["status"] == "ok"
    assert diagnostics["placebo_warning"] is False


def test_market_timeseries_evaluation_attaches_diagnostics() -> None:
    rng = np.random.default_rng(31)
    signal = np.zeros(len(DAYS))
    for index in range(1, len(DAYS)):
        signal[index] = 0.9 * signal[index - 1] + float(rng.standard_normal())
    returns = 0.02 * signal + 0.01 * rng.standard_normal(len(DAYS))
    factor = _series(
        [
            (day, ext.MARKET_INSTRUMENT, float(value))
            for day, value in zip(DAYS, signal, strict=True)
        ],
        "factor",
    )
    label = _series(
        [(day, "SH000300", float(value)) for day, value in zip(DAYS, returns, strict=True)],
        "label",
    )

    result = ext.evaluate_market_timeseries_factor(factor, label, **PERIODS)

    assert result["status"] == "ok"
    diagnostics = result["metrics"]["placebo_diagnostics"]
    assert diagnostics["version"] == ext.PLACEBO_DIAGNOSTICS_VERSION
    assert diagnostics["mode"] == "market_timeseries"
    assert diagnostics["placebo_warning"] is False
