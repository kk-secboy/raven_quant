from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from qlib_test_doubles import qlib_workflow_identity

pytestmark = pytest.mark.no_database


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "run_recommendation_refresh.py"
    spec = importlib.util.spec_from_file_location(
        "run_recommendation_refresh_order_plan", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_order_plan_generator_records_and_hashes_qlib_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script_module()
    saved: list[Path] = []

    class Workflow:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def identity_dict():
            return qlib_workflow_identity()

        @staticmethod
        def log_params(_values):
            return None

        @staticmethod
        def log_metrics(_values):
            return None

        @staticmethod
        def save_artifacts(path):
            saved.append(Path(path))

    monkeypatch.setattr(script, "qlib_workflow_run", lambda **_kwargs: Workflow())
    result = script._write_qlib_order_plan(
        manifest={
            "order_plan_job_id": "job-1",
            "simulation_portfolio_id": "simulation-1",
            "strategy_version_id": "version-1",
            "formal_backtest_id": "backtest-1",
            "dataset": "snapshot",
            "signal_date": "2026-07-10",
            "signal_at": None,
            "execution_not_before": None,
            "config": {"execution_contract_hash": "a" * 64},
        },
        result={
            "status": "ok",
            "as_of_date": "2026-07-10",
            "effective_date": "2026-07-13",
            "holdings": [
                {"instrument": "SH600001", "weight": 0.40},
                {"instrument": "SH600000", "weight": 0.50},
            ],
        },
        dataset_provenance={
            "dataset_identity_sha256": "b" * 64,
            "dataset_lineage_id": "c" * 64,
        },
        order_plan_root=tmp_path / "order-plans",
        tracking_uri="sqlite:///tracking.db",
    )

    digest = result["order_plan_manifest_sha256"]
    artifact = tmp_path / "order-plans" / digest
    manifest_path = artifact / "manifest.json"
    target_path = artifact / "target_weights.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == digest
    assert json.loads(target_path.read_text(encoding="utf-8")) == {
        "target_weights": {"SH600000": 0.50, "SH600001": 0.40}
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["produced_by"] == "qlib-workflow-recorder"
    assert manifest["source_snapshot"]["id"] == "b" * 64
    assert manifest["qlib_workflow"] == qlib_workflow_identity()
    assert saved == [artifact]
