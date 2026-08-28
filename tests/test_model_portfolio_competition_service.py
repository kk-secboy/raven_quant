from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from quant_platform.parameter_experiment_store import (
    ParameterExperimentStore,
    _portfolio_competition_spec_sha256,
)

pytestmark = pytest.mark.no_database


def _version() -> dict[str, Any]:
    return {
        "id": "version-1",
        "config": {
            "signal_source": "model_prediction",
            "signal_frequency": "day",
            "signal_period": 1,
            "rebalance_frequency": "day",
            "execution_frequency": "day",
            "execution_method": "open",
            "execution_days": 1,
            "execution_lag_bars": 1,
            "max_volume_participation": 0.10,
            "max_position_weight": 0.05,
            "max_daily_turnover": 0.20,
            "portfolio_construction": "topk_equal_weight",
        },
    }


def test_portfolio_competition_service_is_idempotent_and_builds_job_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = object.__new__(ParameterExperimentStore)
    created_kwargs: dict[str, Any] = {}
    existing: dict[str, Any] | None = None

    def latest_for_version(
        strategy_version_id: str, *, created_by: str | None = None
    ) -> dict[str, Any] | None:
        assert strategy_version_id == "version-1"
        assert created_by is None
        return existing

    def create(**kwargs: Any) -> dict[str, Any]:
        nonlocal existing
        created_kwargs.update(kwargs)
        spec_sha256 = _portfolio_competition_spec_sha256(
            strategy_version_id=kwargs["strategy_version_id"],
            dataset=kwargs["dataset"],
            dataset_identity_sha256=kwargs["dataset_identity_sha256"],
            periods=kwargs["periods"],
            parameter_grid=kwargs["parameter_grid"],
            baseline_config=kwargs["baseline_config"],
        )
        existing = {
            "id": "experiment-1",
            "status": "queued",
            "job_id": None,
            "periods": {
                **kwargs["periods"],
                "governance": {"competition_spec_sha256": spec_sha256},
            },
        }
        return existing

    monkeypatch.setattr(store, "latest_for_version", latest_for_version)
    monkeypatch.setattr(store, "create", create)
    call = {
        "strategy_version": _version(),
        "dataset": {
            "name": "snapshot-1",
            "path": str(tmp_path / "qlib"),
            "provenance": {"dataset_identity_sha256": "a" * 64},
        },
        "candidate_valid_start": "2023-01-03",
        "candidate_valid_end": "2025-12-31",
        "artifact_root": tmp_path / "experiments",
        "created_by": "autopilot",
    }

    first = store.ensure_model_portfolio_competition(**call)
    second = store.ensure_model_portfolio_competition(**call)

    assert first["created"] is True
    assert second["created"] is False
    assert first["experiment"]["id"] == second["experiment"]["id"]
    assert first["job_payload"] == {
        "parameter_experiment_id": "experiment-1",
        "strategy_version_id": "version-1",
        "dataset": "snapshot-1",
        "dataset_identity_sha256": "a" * 64,
        "dataset_path": str(tmp_path / "qlib"),
        "execution_dataset": None,
    }
    assert created_kwargs["parameter_grid"] == {
        "portfolio_construction": ["topk_equal_weight", "industry_neutral_qp"]
    }
    assert created_kwargs["periods"]["in_sample"]["start"] > "2023-01-03"


def test_portfolio_competition_service_blocks_a_conflicting_existing_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = object.__new__(ParameterExperimentStore)
    monkeypatch.setattr(
        store,
        "latest_for_version",
        lambda *_args, **_kwargs: {
            "id": "other",
            "status": "succeeded",
            "periods": {"governance": {"competition_spec_sha256": "0" * 64}},
        },
    )

    with pytest.raises(ValueError, match="already exists"):
        store.ensure_model_portfolio_competition(
            strategy_version=_version(),
            dataset={
                "name": "snapshot-1",
                "path": str(tmp_path / "qlib"),
                "provenance": {"dataset_identity_sha256": "a" * 64},
            },
            candidate_valid_start="2023-01-03",
            candidate_valid_end="2025-12-31",
            artifact_root=tmp_path / "experiments",
            created_by="autopilot",
        )
