from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert

from quant_data.database import open_database, strategies, strategy_versions
from quant_platform.api import create_app
from quant_platform.parameter_experiment_store import ParameterExperimentStore


def _strategy_version(database_url: str) -> str:
    now = datetime.now(UTC)
    engine = open_database(database_url)
    with engine.begin() as connection:
        connection.execute(
            insert(strategies).values(
                id="strategy-experiment",
                name="parameter experiment fixture",
                description="Immutable strategy used to test durable parameter evidence.",
                status="draft",
                created_by="researcher",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(strategy_versions).values(
                id="version-experiment",
                strategy_id="strategy-experiment",
                version=1,
                status="draft",
                strategy_type="multifactor",
                benchmark="SH000300",
                universe="cn_all",
                config_json={"topk": 50, "max_daily_turnover": 0.2},
                created_by="researcher",
                created_at=now,
            )
        )
    return "version-experiment"


def test_parameter_experiment_store_persists_trials_progress_and_result(
    tmp_path: Path, database_url: str
) -> None:
    version_id = _strategy_version(database_url)
    store = ParameterExperimentStore(database_url)
    experiment = store.create(
        strategy_version_id=version_id,
        dataset="qlib-snapshot",
        periods={
            "in_sample": {"start": "2024-01-01", "end": "2025-03-15"},
            "out_of_sample": {"start": "2025-03-16", "end": "2026-01-01"},
        },
        parameter_grid={"topk": [30, 50]},
        baseline_config={"topk": 50, "max_daily_turnover": 0.2},
        trials=[
            {"parameters": {"topk": 30}, "config": {"topk": 30}},
            {"parameters": {"topk": 50}, "config": {"topk": 50}},
        ],
        artifact_root=tmp_path / "experiments",
        created_by="researcher",
    )
    assert experiment["status"] == "queued"
    assert experiment["trial_count"] == 2

    artifact_path = Path(experiment["artifact_path"])
    artifact_path.mkdir(parents=True)
    (artifact_path / "progress.json").write_text(
        json.dumps({"completed_count": 1, "trial_count": 2}), encoding="utf-8"
    )
    assert store.get(experiment["id"])["progress"]["completed_count"] == 1

    store.mark(experiment["id"], "running")
    store.apply_result(
        experiment["id"],
        {
            "trials": [
                {
                    "trial_index": 0,
                    "status": "succeeded",
                    "score": 0.8,
                    "metrics": {"in_sample": {}, "out_of_sample": {}},
                    "warnings": [],
                },
                {
                    "trial_index": 1,
                    "status": "failed",
                    "score": None,
                    "metrics": None,
                    "warnings": [],
                    "error": "fixture failure",
                },
            ],
            "summary": {
                "trial_count": 2,
                "succeeded_count": 1,
                "failed_count": 1,
                "leaderboard": [],
                "warnings": [],
            },
        },
    )
    completed = store.get(experiment["id"])
    assert completed["status"] == "succeeded"
    assert completed["trials"][0]["score"] == 0.8
    assert completed["trials"][1]["error"] == "fixture failure"


def test_parameter_experiment_result_must_cover_every_trial(
    tmp_path: Path, database_url: str
) -> None:
    version_id = _strategy_version(database_url)
    store = ParameterExperimentStore(database_url)
    experiment = store.create(
        strategy_version_id=version_id,
        dataset="qlib-snapshot",
        periods={"in_sample": {}, "out_of_sample": {}},
        parameter_grid={"topk": [50]},
        baseline_config={"topk": 50},
        trials=[{"parameters": {"topk": 50}, "config": {"topk": 50}}],
        artifact_root=tmp_path / "experiments",
        created_by="researcher",
    )
    with pytest.raises(ValueError, match="every trial"):
        store.apply_result(experiment["id"], {"trials": [], "summary": {}})


def test_api_creates_a_bounded_parameter_experiment_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database_url: str
) -> None:
    version_id = _strategy_version(database_url)
    data_root = tmp_path / "data"
    dataset = data_root / "qlib" / "experiment-snapshot"
    (dataset / "calendars").mkdir(parents=True)
    (dataset / "instruments").mkdir()
    (dataset / "features").mkdir()
    (dataset / "metadata").mkdir()
    (dataset / "calendars" / "day.txt").write_text(
        "2024-01-01\n2026-01-01\n", encoding="utf-8"
    )
    (dataset / "instruments" / "cn_all.txt").write_text(
        "SH600000\t2024-01-01\t2026-01-01\n", encoding="utf-8"
    )
    (dataset / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "dataset_identity_sha256": "a" * 64,
                "snapshot_manifest_sha256": "b" * 64,
                "qlib_builder_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            f"/api/strategy-versions/{version_id}/parameter-experiments",
            json={
                "dataset": "experiment-snapshot",
                "start": "2024-01-01",
                "end": "2026-01-01",
                "parameter_grid": {"topk": [30, 50]},
                "max_trials": 2,
            },
        )
        assert response.status_code == 202, response.text
        created = response.json()
        detail = client.get(f"/api/parameter-experiments/{created['id']}")
    assert created["status"] == "queued"
    assert created["trial_count"] == 2
    assert created["periods"]["out_of_sample"]["start"] > created["periods"]["in_sample"]["end"]
    assert detail.status_code == 200
