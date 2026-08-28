from pathlib import Path

import pandas as pd
import pytest

from quant_platform.model_recompute import governed_model_resource_policy
from quant_platform.model_research_governance import MODEL_REFIT_POLICY
from quant_platform.scheduler import model_refresh_decision
from scripts.run_model_refit import (
    _calendar,
    _inference_periods,
    _validate_retrain_evidence,
)

pytestmark = pytest.mark.no_database


def test_live_refit_window_is_rolling_purged_and_single_day(tmp_path: Path) -> None:
    provider = tmp_path / "qlib"
    (provider / "calendars").mkdir(parents=True)
    required = sum(
        int(MODEL_REFIT_POLICY[key])
        for key in (
            "train_trading_days",
            "train_validation_purge_trading_days",
            "validation_trading_days",
            "embargo_trading_days",
            "prediction_trading_days",
        )
    )
    days = [
        value.date().isoformat()
        for value in pd.bdate_range("2018-01-01", periods=required + 10)
    ]
    (provider / "calendars" / "day.txt").write_text("\n".join(days), encoding="utf-8")

    _, periods = _calendar(provider, days[-1], dict(MODEL_REFIT_POLICY))

    assert periods["test_start"] == periods["test_end"] == days[-1]
    assert days.index(periods["valid_start"]) - days.index(periods["train_end"]) == (
        int(MODEL_REFIT_POLICY["train_validation_purge_trading_days"]) + 1
    )
    assert days.index(periods["test_start"]) - days.index(periods["valid_end"]) == (
        int(MODEL_REFIT_POLICY["embargo_trading_days"]) + 1
    )
    assert (
        days.index(periods["train_end"]) - days.index(periods["train_start"]) + 1
        == MODEL_REFIT_POLICY["train_trading_days"]
    )


def test_live_refit_rejects_non_trading_signal_date(tmp_path: Path) -> None:
    provider = tmp_path / "qlib"
    (provider / "calendars").mkdir(parents=True)
    required = sum(
        int(MODEL_REFIT_POLICY[key])
        for key in (
            "train_trading_days",
            "train_validation_purge_trading_days",
            "validation_trading_days",
            "embargo_trading_days",
            "prediction_trading_days",
        )
    )
    days = [
        value.date().isoformat()
        for value in pd.bdate_range("2018-01-01", periods=required)
    ]
    (provider / "calendars" / "day.txt").write_text("\n".join(days), encoding="utf-8")

    with pytest.raises(ValueError, match="signal date"):
        _calendar(provider, "2099-01-01", dict(MODEL_REFIT_POLICY))


def test_live_refit_resource_stage_preserves_the_frozen_model_engine() -> None:
    policy = governed_model_resource_policy(
        model_type="TimeSeries",
        model_engine="platform_gru",
        requested_hyperparameters={"n_epochs": 500},
        stage="production_refit",
        requested_timeout_seconds=99_999,
        seed=11,
    )

    assert policy["stage"] == "production_refit"
    assert policy["model_engine"] == "platform_gru"
    assert policy["effective_training_hyperparameters"]["n_epochs"] == 12


def test_daily_inference_reuses_frozen_training_window(tmp_path: Path) -> None:
    provider = tmp_path / "qlib"
    (provider / "calendars").mkdir(parents=True)
    days = [
        value.date().isoformat()
        for value in pd.bdate_range("2014-01-01", periods=2900)
    ]
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(days), encoding="utf-8"
    )
    frozen = {
        "train_start": days[0],
        "train_end": days[2015],
        "valid_start": days[2018],
        "valid_end": days[2773],
    }

    periods = _inference_periods(
        provider,
        days[-1],
        frozen,
        dict(MODEL_REFIT_POLICY),
    )

    assert {key: periods[key] for key in frozen} == frozen
    assert periods["test_start"] == periods["test_end"] == days[-1]


def test_monthly_retrain_requires_actual_first_trading_day(tmp_path: Path) -> None:
    provider = tmp_path / "qlib"
    (provider / "calendars").mkdir(parents=True)
    days = ["2026-08-03", "2026-08-04", "2026-08-05"]
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(days), encoding="utf-8"
    )
    evidence = {
        "calendar_dataset_identity_sha256": "a" * 64,
        "is_first_trading_day": True,
        "signal_date": days[0],
        "signal_month": "2026-08",
        "trigger": "monthly_first_trading_day",
    }
    _validate_retrain_evidence(
        provider=provider,
        signal_date=days[0],
        reason="monthly_first_trading_day",
        evidence=evidence,
        dataset_identity_sha256="a" * 64,
        dataset_lineage_id="b" * 64,
        source_model_data_contract_sha256="c" * 64,
    )
    evidence["signal_date"] = days[1]
    with pytest.raises(ValueError, match="first governed trading day"):
        _validate_retrain_evidence(
            provider=provider,
            signal_date=days[1],
            reason="monthly_first_trading_day",
            evidence=evidence,
            dataset_identity_sha256="a" * 64,
            dataset_lineage_id="b" * 64,
            source_model_data_contract_sha256="c" * 64,
        )


def test_early_retrain_rejects_unproven_drift(tmp_path: Path) -> None:
    provider = tmp_path / "qlib"
    evidence = {
        "as_of": "2026-08-25",
        "comparison": "above",
        "consecutive_windows": 2,
        "contract_version": "model-drift-trigger-v1",
        "dataset_lineage_id": "b" * 64,
        "metric": "population_stability_index",
        "observed": 0.3,
        "threshold": 0.2,
        "window_trading_days": 20,
    }
    with pytest.raises(ValueError, match="does not cross"):
        _validate_retrain_evidence(
            provider=provider,
            signal_date="2026-08-25",
            reason="persistent_drift",
            evidence=evidence,
            dataset_identity_sha256="a" * 64,
            dataset_lineage_id="b" * 64,
            source_model_data_contract_sha256="c" * 64,
        )


def test_scheduler_fits_monthly_and_only_predicts_other_trading_days() -> None:
    calendar = {
        pd.Timestamp("2026-08-03").date(),
        pd.Timestamp("2026-08-04").date(),
        pd.Timestamp("2026-08-05").date(),
    }
    monthly = model_refresh_decision(
        signal_date=pd.Timestamp("2026-08-03").date(),
        calendar_days=calendar,
        dataset_name="cn-daily",
        dataset_identity_sha256="a" * 64,
    )
    daily = model_refresh_decision(
        signal_date=pd.Timestamp("2026-08-04").date(),
        calendar_days=calendar,
        dataset_name="cn-daily",
        dataset_identity_sha256="a" * 64,
    )

    assert monthly["operation"] == "retrain"
    assert monthly["retrain_reason"] == "monthly_first_trading_day"
    assert len(monthly["retrain_evidence_sha256"]) == 64
    assert daily == {
        "operation": "inference",
        "retrain_reason": "",
        "retrain_evidence": None,
        "retrain_evidence_sha256": "",
    }
