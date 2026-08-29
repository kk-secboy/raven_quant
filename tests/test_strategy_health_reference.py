from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.strategy_health_reference import (
    build_strategy_health_reference,
    factor_drift_from_reference,
    factor_reference_summary,
    live_model_calibration_from_reference,
    model_calibration_reference_summary,
    validate_strategy_health_reference,
)

pytestmark = pytest.mark.no_database


def test_runner_seals_health_reference_before_artifact_manifest() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "run_multifactor_backtest.py"
    ).read_text(encoding="utf-8")

    assert source.index("strategy_health_reference = _write_strategy_health_reference") < (
        source.index("artifact_manifest = write_backtest_artifact_manifest")
    )


def _series(
    dates: pd.DatetimeIndex,
    *,
    instruments: int = 40,
    shift: float = 0.0,
) -> pd.Series:
    index = pd.MultiIndex.from_product(
        [dates, [f"SH{600000 + item:06d}" for item in range(instruments)]],
        names=["datetime", "instrument"],
    )
    values = np.linspace(-2.5, 2.5, len(index)) + shift
    return pd.Series(values, index=index, name="value")


def test_reference_summary_supports_streamed_current_factor() -> None:
    formal_dates = pd.bdate_range("2024-01-02", periods=30)
    formal = _series(formal_dates)
    summary = factor_reference_summary(
        formal,
        factor_id="momentum_20d",
        reference_start=formal_dates[0].date(),
        reference_end=formal_dates[-1].date(),
        source_sha256="a" * 64,
    )
    reference = build_strategy_health_reference(
        strategy_version_id="version-a",
        formal_backtest_id="backtest-a",
        formal_dataset_identity_sha256="b" * 64,
        formal_dataset_lineage_id="c" * 64,
        strategy_rules_sha256="d" * 64,
        signal_source="factor_score",
        reference_start=formal_dates[0].date(),
        reference_end=formal_dates[-1].date(),
        factor_summaries={"momentum_20d": summary},
    )
    validated = validate_strategy_health_reference(
        reference,
        strategy_version_id="version-a",
        formal_backtest_id="backtest-a",
        formal_dataset_identity_sha256="b" * 64,
        formal_dataset_lineage_id="c" * 64,
        strategy_rules_sha256="d" * 64,
        expected_factor_ids={"momentum_20d"},
        signal_source="factor_score",
    )
    current_dates = pd.bdate_range("2026-07-01", periods=25)
    drift = factor_drift_from_reference(
        _series(current_dates, shift=0.75),
        reference=validated["factors"]["momentum_20d"],
        as_of=current_dates[-1].date(),
        current_window_sessions=20,
        current_file_sha256="e" * 64,
    )

    assert drift["current_start"] == current_dates[-20].date().isoformat()
    assert drift["current_end"] == current_dates[-1].date().isoformat()
    assert drift["current_observations"] == 800
    assert 0.0 < drift["normalized_psi"] < 1.0


def test_reference_identity_and_seal_fail_closed() -> None:
    formal_dates = pd.bdate_range("2024-01-02", periods=30)
    summary = factor_reference_summary(
        _series(formal_dates),
        factor_id="factor-a",
        reference_start=formal_dates[0].date(),
        reference_end=formal_dates[-1].date(),
        source_sha256="a" * 64,
    )
    reference = build_strategy_health_reference(
        strategy_version_id="version-a",
        formal_backtest_id="backtest-a",
        formal_dataset_identity_sha256="b" * 64,
        formal_dataset_lineage_id="c" * 64,
        strategy_rules_sha256="d" * 64,
        signal_source="factor_score",
        reference_start=formal_dates[0].date(),
        reference_end=formal_dates[-1].date(),
        factor_summaries={"factor-a": summary},
    )
    reference["formal_backtest_id"] = "newer-backtest"

    with pytest.raises(ValueError, match="seal"):
        validate_strategy_health_reference(
            reference,
            strategy_version_id="version-a",
            formal_backtest_id="newer-backtest",
            formal_dataset_identity_sha256="b" * 64,
            formal_dataset_lineage_id="c" * 64,
            strategy_rules_sha256="d" * 64,
            expected_factor_ids={"factor-a"},
            signal_source="factor_score",
        )


def test_model_reference_uses_bounded_mature_live_window() -> None:
    formal_dates = pd.bdate_range("2022-01-03", periods=30)
    formal_scores = _series(formal_dates)
    formal_labels = formal_scores * 0.01 + 0.002
    formal = model_calibration_reference_summary(
        formal_scores,
        formal_labels,
        label_horizon_sessions=5,
        label_contract_sha256="1" * 64,
        predictions_sha256="2" * 64,
        labels_source_sha256="3" * 64,
    )
    live_dates = pd.bdate_range("2026-08-03", periods=10)
    live_scores = _series(live_dates)
    live_labels = live_scores * 0.008 - 0.001
    live = live_model_calibration_from_reference(
        live_scores,
        live_labels,
        reference=formal,
    )

    assert live["live_sessions"] == 10
    assert live["live_observations"] == 400
    assert date.fromisoformat(live["live_start"]) == live_dates[0].date()
    assert 0.0 <= live["model_calibration_drift"] < 1.0
