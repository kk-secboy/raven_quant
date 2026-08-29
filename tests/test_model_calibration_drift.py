from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_platform.model_calibration_drift import (
    build_model_calibration_observation,
    validate_model_calibration_observation,
)

pytestmark = pytest.mark.no_database


def _series(start: str, sessions: int, *, reverse: bool = False) -> tuple[pd.Series, pd.Series]:
    dates = pd.bdate_range(start, periods=sessions)
    instruments = [f"SH{600000 + index:06d}" for index in range(60)]
    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    score = np.tile(np.linspace(-1.0, 1.0, len(instruments)), sessions)
    realized = (-score if reverse else score) * 0.02
    return (
        pd.Series(score, index=index, name="score"),
        pd.Series(realized, index=index, name="realized_return"),
    )


def test_model_calibration_uses_matured_live_decile_evidence() -> None:
    formal_score, formal_return = _series("2025-01-02", 30)
    live_score, live_return = _series("2026-07-01", 6, reverse=True)
    observed = build_model_calibration_observation(
        strategy_version_id="model-version",
        formal_predictions=formal_score,
        formal_labels=formal_return,
        live_predictions=live_score,
        live_labels=live_return,
        label_horizon_sessions=1,
        label_contract_sha256="a" * 64,
        formal_predictions_sha256="b" * 64,
        live_prediction_hashes={"live-a": "c" * 64},
        label_materialization_manifest_sha256="d" * 64,
        label_materialized_file_sha256="e" * 64,
        current_dataset_identity_sha256="f" * 64,
        current_dataset_lineage_id="1" * 64,
        as_of=date(2026, 7, 10),
    )

    validated = validate_model_calibration_observation(
        observed,
        strategy_version_id="model-version",
        current_dataset_identity_sha256="f" * 64,
        expected_as_of=date(2026, 7, 10),
    )

    assert validated["live_sessions"] == 6
    assert validated["model_calibration_drift"] > 0.5
    observed["live_deciles"]["9"]["mean_realized_return"] = 99.0
    with pytest.raises(ValueError, match="seal"):
        validate_model_calibration_observation(
            observed,
            strategy_version_id="model-version",
            current_dataset_identity_sha256="f" * 64,
        )


def test_model_calibration_blocks_until_live_labels_mature() -> None:
    formal_score, formal_return = _series("2025-01-02", 30)
    live_score, live_return = _series("2026-07-01", 4)

    with pytest.raises(ValueError, match="matured live"):
        build_model_calibration_observation(
            strategy_version_id="model-version",
            formal_predictions=formal_score,
            formal_labels=formal_return,
            live_predictions=live_score,
            live_labels=live_return,
            label_horizon_sessions=1,
            label_contract_sha256="a" * 64,
            formal_predictions_sha256="b" * 64,
            live_prediction_hashes={"live-a": "c" * 64},
            label_materialization_manifest_sha256="d" * 64,
            label_materialized_file_sha256="e" * 64,
            current_dataset_identity_sha256="f" * 64,
            current_dataset_lineage_id="1" * 64,
            as_of=date(2026, 7, 10),
        )
