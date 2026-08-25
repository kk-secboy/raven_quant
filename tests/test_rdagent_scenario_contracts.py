from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from quant_data.config import Settings
from quant_platform import api
from quant_platform.rdagent_runtime import (
    require_matching_rdagent_runtime_identity,
    validate_duration_limit,
)
from quant_platform.rdagent_scenarios import (
    RDAGENT_DATA_SCIENCE_JOB_KIND,
    RDAGENT_GENERIC_JOB_KIND,
    RDAGENT_LEGACY_FACTOR_JOB_KIND,
    RDAGENT_LLM_FINETUNE_JOB_KIND,
    SCENARIOS,
    rdagent_scenario_catalog,
)

pytestmark = pytest.mark.no_database


def test_registry_contains_exactly_the_seven_public_scenarios() -> None:
    assert set(SCENARIOS) == {
        "fin_factor",
        "fin_model",
        "fin_quant",
        "fin_factor_report",
        "general_model",
        "data_science",
        "llm_finetune",
    }


def test_registry_keeps_capital_and_lab_boundaries_explicit() -> None:
    assert {item.id for item in SCENARIOS.values() if item.capital_eligible} == {
        "fin_factor",
        "fin_model",
        "fin_quant",
        "fin_factor_report",
    }
    assert all(
        not item.capital_eligible for item in SCENARIOS.values() if item.category == "lab"
    )


def test_scenario_catalog_has_the_public_api_contract(tmp_path: Path) -> None:
    settings = Settings(api_url="", token="", data_root=tmp_path)
    runtime = {
        "status": "ok",
        "llm_credentials_configured": True,
        "docker_available": True,
        "source_tree_sha256": "a" * 64,
        "runtime_image_digest": "sha256:" + "b" * 64,
    }
    required = {
        "id",
        "label",
        "description",
        "category",
        "ready",
        "blockers",
        "requires_dataset",
        "requires_assets",
        "capital_eligible",
        "gpu_required",
        "limits",
    }
    assert all(required <= set(item) for item in rdagent_scenario_catalog(runtime, settings))


def test_dangerous_lab_scenarios_have_dedicated_physical_queues() -> None:
    assert SCENARIOS["fin_factor"].job_kind == RDAGENT_LEGACY_FACTOR_JOB_KIND
    assert SCENARIOS["fin_model"].job_kind == RDAGENT_GENERIC_JOB_KIND
    assert SCENARIOS["data_science"].job_kind == RDAGENT_DATA_SCIENCE_JOB_KIND
    assert SCENARIOS["llm_finetune"].job_kind == RDAGENT_LLM_FINETUNE_JOB_KIND


def test_public_run_request_defaults_to_factor_and_forbids_raw_execution_fields() -> None:
    fields = api.RDAgentRunRequest.model_fields
    assert fields["scenario"].default == "fin_factor"
    assert {"asset_ids", "feature_set_id"} <= set(fields)
    assert not ({"command", "environment", "path", "cwd"} & set(fields))


def test_research_asset_acquisition_api_accepts_only_governed_inputs() -> None:
    automatic_fields = api.ResearchAssetAutomaticRequest.model_fields
    manual_fields = api.ResearchAssetManualHttpsRequest.model_fields
    assert set(automatic_fields) == {
        "as_of",
        "include_tushare",
        "include_arxiv",
        "actor",
    }
    assert {"url", "title", "document_kind", "published_at", "actor"} == set(
        manual_fields
    )
    assert not ({"command", "environment", "path", "cwd"} & set(manual_fields))
    with pytest.raises(ValueError, match="public HTTPS URL"):
        api.ResearchAssetManualHttpsRequest.model_validate(
            {
                "url": "http://127.0.0.1/private.pdf",
                "title": "untrusted local file",
            }
        )


def test_ui_uses_feature_registry_and_server_asset_endpoints() -> None:
    source = (
        Path(__file__).parents[1] / "web" / "app" / "rdagent-panel.tsx"
    ).read_text(encoding="utf-8")
    assert "/api/rdagent/feature-sets" in source
    assert "/api/rdagent/assets/acquisitions/automatic" in source
    assert "/api/rdagent/assets/acquisitions/manual-https" in source
    assert '<select value={featureSetId}' in source
    assert '<input value={featureSetId}' not in source


def test_general_model_handoff_accepts_only_governed_ids_and_recipe() -> None:
    fields = api.GeneralModelValidationRequest.model_fields
    assert {
        "artifact_id",
        "dataset",
        "feature_set_id",
        "model_type",
        "architecture",
        "model_hyperparameters",
        "training_hyperparameters",
        "period_policy",
    } <= set(fields)
    assert not ({"command", "environment", "path", "cwd", "url"} & set(fields))
    with pytest.raises(ValueError):
        api.GeneralModelValidationRequest.model_validate(
            {
                "artifact_id": "a" * 32,
                "dataset": "daily-v1",
                "feature_set_id": "governed-baseline",
                "path": "C:/untrusted/model.py",
            }
        )


@pytest.mark.no_database
def test_model_capital_scenarios_reject_noncanonical_embargo() -> None:
    for scenario in ("fin_model", "fin_quant"):
        with pytest.raises(ValueError, match="5-trading-day model embargo"):
            api.RDAgentRunRequest.model_validate(
                {
                    "objective": "Research a governed model without changing the final OOS.",
                    "scenario": scenario,
                    "dataset": "daily-v1",
                    "feature_set_id": "governed-baseline",
                    "period_policy": {
                        "test_trading_days": 252,
                        "embargo_trading_days": 10,
                    },
                }
            )

    with pytest.raises(ValueError, match="5-trading-day model embargo"):
        api.GeneralModelValidationRequest.model_validate(
            {
                "artifact_id": "a" * 32,
                "dataset": "daily-v1",
                "feature_set_id": "governed-baseline",
                "period_policy": {
                    "test_trading_days": 252,
                    "embargo_trading_days": 10,
                },
            }
        )


def test_runner_is_allowlisted_and_fin_quant_replays_accepted_state() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "rdagent_bridge.py").read_text(
        encoding="utf-8"
    )
    runner = (Path(__file__).parents[1] / "scripts" / "run_rdagent_scenario.py").read_text(
        encoding="utf-8"
    )
    assert "_materialize_quant_bundles" in source
    assert "accepted_factors" in source and "accepted_model" in source
    assert "shell=True" not in runner
    assert "get_rdagent_scenario" in runner
    assert "one governed report per allowed loop" in runner


def test_official_health_check_is_side_effect_limited_and_diagnostic_only() -> None:
    runtime = (
        Path(__file__).parents[1]
        / "src"
        / "quant_platform"
        / "rdagent_runtime.py"
    ).read_text(encoding="utf-8")
    assert '"health_check", "--no-check-docker", "--no-check-ports"' in runtime
    assert '"platform_readiness_unchanged": True' in runtime


def test_duration_limit_is_enforced_before_queueing() -> None:
    assert validate_duration_limit("30m", "2h") == "30m"
    with pytest.raises(ValueError, match="configured limit"):
        validate_duration_limit("3h", "2h")


def test_public_job_projection_removes_paths_urls_logs_and_raw_errors() -> None:
    projected = api._public_job(
        {
            "id": "job-1",
            "kind": "rdagent_run",
            "status": "failed",
            "payload": {
                "scenario": "fin_factor",
                "url": "https://example.invalid/report.pdf",
                "dataset_path": "/data/private/dataset",
            },
            "progress": {
                "execution_phase": "terminal_failure",
                "trace_path": "/data/private/trace",
                "result": {"secret": "must-not-leak"},
            },
            "error": "failed at /data/private/trace",
        }
    )
    assert projected["payload"] == {"scenario": "fin_factor"}
    assert projected["progress"] == {"execution_phase": "terminal_failure"}
    assert projected["error"] == "job execution failed"


def test_rdagent_runtime_identity_is_frozen_across_queue_and_worker() -> None:
    identity = {
        "version": "0.0.dev0",
        "commit": "4f9ecb005881cddc08df0124a2e894c018007679",
        "commit_evidence": ["repository"],
        "source_tree_sha256": "a" * 64,
        "repository_dirty": None,
        "runtime_image_digest": "sha256:" + "b" * 64,
        "production_reproducible": True,
    }
    assert require_matching_rdagent_runtime_identity(identity, dict(identity))[
        "source_tree_sha256"
    ] == "a" * 64
    with pytest.raises(ValueError, match="changed after enqueue"):
        require_matching_rdagent_runtime_identity(
            identity, {**identity, "source_tree_sha256": "c" * 64}
        )


def test_worker_closes_model_quant_and_generic_audit_chains() -> None:
    from quant_platform.worker import Worker

    source = inspect.getsource(Worker)
    for marker in (
        "_queue_model_evaluation",
        "_queue_quant_bundle_evaluation",
        "_import_quant_bundle_evaluation_artifact",
        "_archive_rdagent_run_evidence",
        "_archive_rdagent_lab_artifacts",
    ):
        assert marker in source


def test_data_science_and_finetune_use_explicit_isolated_runtime_contracts() -> None:
    runtime_path = (
        Path(__file__).parents[1] / "src" / "quant_platform" / "rdagent_runtime.py"
    )
    runtime = runtime_path.read_text(encoding="utf-8")
    for marker in (
        '"DS_SCEN": "rdagent.scenarios.data_science.scen.DataScienceScen"',
        '"DS_CODER_COSTEER_ENV_TYPE": "docker"',
        '"DS_DOCKER_NETWORK": "none"',
        '"DS_DOCKER_ENABLE_GPU": "false"',
        '"DS_DOCKER_MEM_LIMIT": "16g"',
        '"FT_CODER_COSTEER_ENV_TYPE": "docker"',
        '"FT_DOCKER_ENABLE_CACHE": "false"',
    ):
        assert marker in runtime


def test_quant_exporter_accumulates_alternating_accepted_factor_model_rounds(
    tmp_path: Path,
) -> None:
    bridge_path = Path(__file__).parents[1] / "scripts" / "rdagent_bridge.py"
    spec = importlib.util.spec_from_file_location("rdagent_bridge_contract", bridge_path)
    assert spec and spec.loader
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    factor = {
        "name": "factor_a",
        "description": "a",
        "code": "def calculate(data):\n    return data\n",
    }
    factor["code_sha256"] = bridge._sha256(factor["code"])
    model = {
        "name": "model_a",
        "description": "m",
        "model_type": "Tabular",
        "architecture": {},
        "model_hyperparameters": {},
        "training_hyperparameters": {},
        "code": "class Net:\n    pass\n",
    }
    model["code_sha256"] = bridge._sha256(model["code"])
    factor_b = {**factor, "name": "factor_b"}
    rounds = {
        0: {
            "feedback": {"decision": True},
            "runner_snapshot": {
                "kind": "factor",
                "artifacts": [factor],
                "based_artifacts": [],
                "base_features": {"BASE": "$close"},
            },
        },
        1: {
            "feedback": {"decision": True},
            "runner_snapshot": {
                "kind": "model",
                "artifacts": [model],
                "based_artifacts": [],
                "base_features": {"BASE": "$close"},
            },
        },
        2: {
            "feedback": {"decision": True},
            "runner_snapshot": {
                "kind": "factor",
                "artifacts": [factor_b],
                "based_artifacts": [],
                "base_features": {"BASE": "$close"},
            },
        },
    }
    code_root = tmp_path / "code"
    values_root = tmp_path / "values"
    code_root.mkdir()
    values_root.mkdir()
    bundles = bridge._materialize_quant_bundles(
        rounds,
        code_root=code_root,
        values_root=values_root,
        feature_set_id="governed-baseline",
        feature_set_sha256="a" * 64,
    )
    assert [item["source_iteration"] for item in bundles] == [1, 2]
    assert [len(item["factors"]) for item in bundles] == [1, 2]
    assert bundles[1]["model"]["code_sha256"] == model["code_sha256"]
