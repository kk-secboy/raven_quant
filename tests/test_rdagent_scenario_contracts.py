from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from quant_data.config import Settings
from quant_data.database import jobs
from quant_platform import api
from quant_platform import worker as worker_module
from quant_platform.job_store import (
    EVALUATION_STATUS_COUNTS_KEY,
    JobStore,
    _with_evaluation_status_counts,
)
from quant_platform.rdagent_runtime import (
    require_matching_rdagent_runtime_identity,
    validate_duration_limit,
)
from quant_platform.rdagent_scenarios import (
    RDAGENT_DATA_SCIENCE_JOB_KIND,
    RDAGENT_GENERIC_JOB_KIND,
    RDAGENT_LEGACY_FACTOR_JOB_KIND,
    RDAGENT_LLM_FINETUNE_JOB_KIND,
    RDAGENT_MODEL_JOB_KIND,
    RDAGENT_QUANT_JOB_KIND,
    RDAGENT_REPORT_JOB_KIND,
    SCENARIOS,
    rdagent_scenario_catalog,
)
from quant_platform.worker import LocalJobWorker


def test_public_rdagent_status_keeps_safe_credential_readiness_boolean() -> None:
    from quant_platform.api import _public_rdagent_status

    public = _public_rdagent_status(
        {
            "status": "ok",
            "enabled": True,
            "ready": True,
            "llm_credentials_configured": True,
            "docker_available": True,
            "scenarios": [],
        }
    )

    assert public["llm_credentials_configured"] is True
    assert "credential" not in str(public.get("blockers") or []).lower()


def test_public_rdagent_trace_text_redacts_credential_shaped_content() -> None:
    from quant_platform.api import _public_rdagent_run

    public = _public_rdagent_run(
        {
            "kind": "factor",
            "trace_view": {
                "status": "recorded",
                "loops": [
                    {
                        "loop_id": 1,
                        "hypothesis": {"text": "api_key=sk-not-for-the-browser"},
                    }
                ],
            },
        }
    )

    assert public["trace_view"]["loops"][0]["hypothesis"]["text"] == "[redacted]"

pytestmark = pytest.mark.no_database


def test_registry_contains_exactly_the_governed_public_scenarios() -> None:
    assert set(SCENARIOS) == {
        "fin_factor",
        "fin_model",
        "fin_quant",
        "fin_strategy",
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
    assert all(not item.capital_eligible for item in SCENARIOS.values() if item.category == "lab")
    assert SCENARIOS["fin_strategy"].capital_eligible is False
    assert SCENARIOS["fin_strategy"].requires_feature_set is True
    assert SCENARIOS["fin_factor"].requires_feature_set is True


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


def test_scenarios_have_bounded_physical_queues() -> None:
    assert SCENARIOS["fin_factor"].job_kind == RDAGENT_LEGACY_FACTOR_JOB_KIND
    assert SCENARIOS["fin_model"].job_kind == RDAGENT_MODEL_JOB_KIND
    assert SCENARIOS["fin_quant"].job_kind == RDAGENT_QUANT_JOB_KIND
    assert SCENARIOS["fin_strategy"].job_kind == RDAGENT_GENERIC_JOB_KIND
    assert SCENARIOS["fin_factor_report"].job_kind == RDAGENT_REPORT_JOB_KIND
    assert SCENARIOS["general_model"].job_kind == RDAGENT_GENERIC_JOB_KIND
    assert SCENARIOS["data_science"].job_kind == RDAGENT_DATA_SCIENCE_JOB_KIND
    assert SCENARIOS["llm_finetune"].job_kind == RDAGENT_LLM_FINETUNE_JOB_KIND


def test_public_run_request_defaults_to_factor_and_forbids_raw_execution_fields() -> None:
    fields = api.RDAgentRunRequest.model_fields
    assert fields["scenario"].default == "fin_factor"
    assert {
        "asset_ids",
        "feature_set_id",
        "horizon",
        "incumbent_strategy_version_id",
    } <= set(fields)
    assert not ({"command", "environment", "path", "cwd"} & set(fields))


def test_active_quant_research_requires_one_horizon_and_freezes_period_policy() -> None:
    expected = {
        "short": (252, 20),
        "swing": (504, 127),
        "long": (756, 253),
    }
    for scenario in ("fin_factor", "fin_model", "fin_quant", "fin_strategy"):
        base = {
            "objective": "Research one governed quant component without producing advice.",
            "scenario": scenario,
            "dataset": "daily-v1",
            "feature_set_id": "governed-baseline",
        }
        with pytest.raises(ValueError, match="explicit short, swing, or long horizon"):
            api.RDAgentRunRequest.model_validate(base)
        for horizon, (oos_days, embargo_days) in expected.items():
            request = api.RDAgentRunRequest.model_validate(
                {
                    **base,
                    "horizon": horizon,
                    **(
                        {"incumbent_strategy_version_id": "a" * 32}
                        if scenario == "fin_strategy"
                        else {}
                    ),
                }
            )
            assert request.period_policy.test_trading_days == oos_days
            assert request.period_policy.embargo_trading_days == embargo_days

    with pytest.raises(ValueError, match="weaker than its frozen horizon contract"):
        api.RDAgentRunRequest.model_validate(
            {
                "objective": "Research a governed long-horizon model candidate.",
                "scenario": "fin_model",
                "dataset": "daily-v1",
                "feature_set_id": "governed-baseline",
                "horizon": "long",
                "period_policy": {
                    "test_trading_days": 756,
                    "embargo_trading_days": 252,
                },
            }
        )


def test_only_strategy_research_accepts_an_incumbent_binding() -> None:
    request = api.RDAgentRunRequest.model_validate(
        {
            "objective": "Research a governed factor using the frozen feature library.",
            "scenario": "fin_factor",
            "dataset": "daily-v1",
            "feature_set_id": "governed-baseline",
            "horizon": "short",
        }
    )
    assert request.horizon == "short"
    with pytest.raises(ValueError, match="accepted only by fin_strategy"):
        api.RDAgentRunRequest.model_validate(
            {
                "objective": "Research a governed factor using the frozen feature library.",
                "scenario": "fin_factor",
                "dataset": "daily-v1",
                "feature_set_id": "governed-baseline",
                "horizon": "short",
                "incumbent_strategy_version_id": "a" * 32,
            }
        )


def test_factor_worker_keeps_research_horizon_without_strategy_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        worker_module,
        "probe_rdagent",
        lambda *_args, **_kwargs: {"runtime_identity": {"commit": "pinned"}},
    )
    monkeypatch.setattr(
        worker_module,
        "require_matching_rdagent_runtime_identity",
        lambda *_args, **_kwargs: None,
    )

    def capture_command(_settings: object, **kwargs: object):
        captured.update(kwargs)
        return ["rdagent"], {}

    monkeypatch.setattr(worker_module, "rdagent_command", capture_command)

    worker = object.__new__(LocalJobWorker)
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.project_root = tmp_path / "project"
    worker.runtime_secrets = SimpleNamespace(get=lambda _name: None)
    worker.rdagent_candidates = SimpleNamespace(import_manifest=lambda *_args, **_kwargs: None)
    worker.factor_library = SimpleNamespace(
        list_library_versions=lambda: [
            {
                "id": "library-1",
                "status": "active",
                "definition_sha256": "a" * 64,
            }
        ]
    )
    payload = {
        "scenario": "fin_factor",
        "research_run_id": "factor-short-1",
        "dataset_path": str(tmp_path / "dataset"),
        "loop_n": 1,
        "duration": "1h",
        "periods": {},
        "objective": "Research a governed short-horizon factor.",
        "feature_set": {"id": "governed-baseline"},
        "horizon_profile": "short_1_5d",
    }

    command, _result_path, environment = worker._command(
        {"kind": "rdagent_factor", "payload": payload}
    )

    assert command == ["rdagent"]
    assert payload["horizon_profile"] == "short_1_5d"
    assert captured["strategy_horizon_profile"] is None
    assert environment["QUANTLAB_FACTOR_LIBRARY_VERSION_ID"] == "library-1"


def test_research_asset_acquisition_api_accepts_only_governed_inputs() -> None:
    automatic_fields = api.ResearchAssetAutomaticRequest.model_fields
    manual_fields = api.ResearchAssetManualHttpsRequest.model_fields
    assert set(automatic_fields) == {
        "as_of",
        "include_tushare",
        "include_arxiv",
        "actor",
    }
    assert {"url", "title", "document_kind", "published_at", "actor"} == set(manual_fields)
    assert not ({"command", "environment", "path", "cwd"} & set(manual_fields))
    with pytest.raises(ValueError, match="public HTTPS URL"):
        api.ResearchAssetManualHttpsRequest.model_validate(
            {
                "url": "http://127.0.0.1/private.pdf",
                "title": "untrusted local file",
            }
        )


def test_ui_uses_feature_registry_and_server_asset_endpoints() -> None:
    source = (Path(__file__).parents[1] / "web" / "app" / "rdagent-panel.tsx").read_text(
        encoding="utf-8"
    )
    assert "/api/rdagent/feature-sets" in source
    assert "/api/rdagent/assets/acquisitions/automatic" in source
    assert "/api/rdagent/assets/acquisitions/manual-https" in source
    assert "<select value={featureSetId}" in source
    assert "<input value={featureSetId}" not in source


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
        with pytest.raises(ValueError, match="weaker than its frozen horizon contract"):
            api.RDAgentRunRequest.model_validate(
                {
                    "objective": "Research a governed model without changing the final OOS.",
                    "scenario": scenario,
                    "dataset": "daily-v1",
                    "feature_set_id": "governed-baseline",
                    "horizon": "swing",
                    "period_policy": {
                        "test_trading_days": 504,
                        "embargo_trading_days": 20,
                    },
                }
            )

    with pytest.raises(ValueError, match="20-trading-day final-OOS embargo"):
        api.GeneralModelValidationRequest.model_validate(
            {
                "artifact_id": "a" * 32,
                "dataset": "daily-v1",
                "feature_set_id": "governed-baseline",
                "period_policy": {
                    "test_trading_days": 252,
                    "embargo_trading_days": 21,
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


def test_factor_prompt_uses_library_digest_without_embedding_all_member_hashes() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "quant_platform" / "rdagent_scenario.py"
    ).read_text(encoding="utf-8")
    assert 'if key != "member_definition_sha256"' in source
    assert '"active_library_definition_sha256"' in source
    assert '"active_library_version_id"' in source


def test_official_health_check_is_side_effect_limited_and_diagnostic_only() -> None:
    runtime = (
        Path(__file__).parents[1] / "src" / "quant_platform" / "rdagent_runtime.py"
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


def test_public_job_projection_exposes_model_gate_outcome_not_private_evidence() -> None:
    projected = api._public_job(
        {
            "id": "job-model-1",
            "kind": "model_evaluate",
            "status": "succeeded",
            "payload": {"dataset": "cn-v1"},
            "progress": {
                "resource_blocked_count": 1,
                EVALUATION_STATUS_COUNTS_KEY: {"resource_blocked": 1},
                "evaluations": [
                    {
                        "status": "resource_blocked",
                        "error": "private path /data/model.py",
                    }
                ],
            },
        }
    )
    assert projected["status"] == "succeeded"
    assert projected["outcome_status"] == "blocked"
    assert "计算资源预算" in projected["outcome_message"]
    assert projected["progress"] == {}
    assert "/data/model.py" not in str(projected)


def test_public_job_projection_preserves_passed_model_outcome_from_bounded_count() -> None:
    projected = api._public_job(
        {
            "id": "job-model-2",
            "kind": "model_evaluate",
            "status": "succeeded",
            "payload": {"dataset": "cn-v1"},
            "progress": {EVALUATION_STATUS_COUNTS_KEY: {"passed": 2}},
        }
    )
    assert projected["outcome_status"] == "passed"
    assert projected["outcome_message"] == "独立模型门禁通过 2 个候选。"
    assert projected["progress"] == {}


def test_job_progress_projection_uses_bounded_evaluation_counts() -> None:
    normalized = _with_evaluation_status_counts(
        {
            "evaluations": [
                {"status": "passed", "evidence": "large-private-evidence"},
                {"status": "passed"},
                {"status": "resource_blocked"},
            ]
        }
    )
    assert normalized is not None
    assert normalized[EVALUATION_STATUS_COUNTS_KEY] == {
        "passed": 2,
        "resource_blocked": 1,
    }

    columns = JobStore._projected_columns(
        payload_keys=(),
        progress_keys=("resource_blocked_count",),
        progress_evaluation_statuses=("passed", "resource_blocked"),
        progress_evaluation_kind="model_evaluate",
    )
    assert all(column is not jobs.c.progress_json for column in columns)
    statement = select(*columns)
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "jsonb_path_query_array" in sql
    assert any("evaluations" in str(value) for value in compiled.params.values())
    assert "model_evaluate" in compiled.params.values()


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
    assert (
        require_matching_rdagent_runtime_identity(identity, dict(identity))["source_tree_sha256"]
        == "a" * 64
    )
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
        "_archive_fin_strategy_artifacts",
        "_archive_rdagent_lab_artifacts",
    ):
        assert marker in source


def test_official_trace_projection_is_read_only_and_bounded() -> None:
    bridge_path = Path(__file__).parents[1] / "scripts" / "rdagent_bridge.py"
    spec = importlib.util.spec_from_file_location("rdagent_bridge_trace_view", bridge_path)
    assert spec and spec.loader
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)

    loops = bridge._trace_loop_projection(
        {
            2: {
                "hypothesis": {
                    "hypothesis": "Short-term relative strength survives costs.",
                    "reason": "Test the economic claim on the frozen window.",
                    "action": "factor",
                },
                "tasks": [
                    {
                        "name": "strength_5d",
                        "description": "Five-session relative strength.",
                        "code_path": "/secret/workspace/factor.py",
                    }
                ],
                "codes": {"strength_5d": "raise RuntimeError('must not leak')"},
                "feedback": {
                    "decision": True,
                    "reason": "net improvement",
                    "hypothesis_evaluation": "accepted after cost check",
                },
                "implementation_feedback": [
                    {"decision": True, "feedback": "implementation is reproducible"}
                ],
            }
        }
    )

    assert loops == [
        {
            "loop_id": 2,
            "hypothesis": {
                "text": "Short-term relative strength survives costs.",
                "reason": "Test the economic claim on the frozen window.",
                "action": "factor",
            },
            "tasks": [
                {
                    "kind": "factor",
                    "name": "strength_5d",
                    "description": "Five-session relative strength.",
                }
            ],
            "feedback": {
                "recorded": True,
                "decision": True,
                "reason": "net improvement",
                "hypothesis_evaluation": "accepted after cost check",
            },
            "implementation_feedback": [
                {"decision": True, "feedback": "implementation is reproducible"}
            ],
        }
    ]
    assert "secret" not in str(loops)
    assert "RuntimeError" not in str(loops)


def test_api_trace_view_reads_only_the_verified_sanitized_artifact(tmp_path: Path) -> None:
    from quant_platform.api import _rdagent_trace_view

    artifact_path = tmp_path / "sanitized-result.json"
    artifact_path.write_text(
        '{"trace_contract_version":"rdagent-trace-web-v1",'
        '"trace_summary":{"message_count":3},'
        '"trace_loops":[{"loop_id":1,"hypothesis":{},"tasks":[],'
        '"feedback":{},"implementation_feedback":[]}]}',
        encoding="utf-8",
    )

    class VerifiedStore:
        def get_run_artifact(self, artifact_id: str, *, verify: bool = False) -> dict:
            assert artifact_id == "artifact-1"
            assert verify is True
            return {
                "research_run_id": "run-1",
                "storage_path": str(artifact_path),
            }

    view = _rdagent_trace_view(
        VerifiedStore(),  # type: ignore[arg-type]
        research_run_id="run-1",
        scenario_id="fin_factor",
        run_artifacts=[
            {
                "id": "artifact-1",
                "artifact_type": "fin_factor_sanitized_result",
                "status": "recorded",
            }
        ],
    )

    assert view["status"] == "recorded"
    assert view["contract_version"] == "rdagent-trace-web-v1"
    assert view["loops"][0]["loop_id"] == 1


def test_data_science_and_finetune_use_explicit_isolated_runtime_contracts() -> None:
    runtime_path = Path(__file__).parents[1] / "src" / "quant_platform" / "rdagent_runtime.py"
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
    coverage = bridge._fin_quant_arm_coverage(rounds)
    assert coverage["complete"] is True
    assert coverage["attempted_complete"] is True
    assert coverage["single_arm"] is False
    assert coverage["attempted_arms"] == ["factor", "model"]
    assert bundles[0]["arm_coverage"] == {"factor": True, "model": True}
    assert bundles[0]["delivery_status"] == "research_only"
    assert bundles[0]["capital_authority"] is False
    assert bundles[0]["joint_ablation_completed"] is False
    assert bundles[0]["required_independent_ablations"] == [
        "factor_only",
        "model_only",
        "joint",
        "joint_vs_incumbent",
    ]
    outcome = bridge._fin_quant_research_outcome(coverage, bundles)
    assert outcome["status"] == "joint_proposal_ready"
    assert outcome["capital_authority"] is False
    assert worker_module._validate_fin_quant_research_result(
        {
            "fin_quant_coverage": coverage,
            "fin_quant_outcome": outcome,
            "quant_bundles": bundles,
        }
    ) == outcome


def test_quant_exporter_rejects_based_counterpart_as_arm_coverage(
    tmp_path: Path,
) -> None:
    bridge_path = Path(__file__).parents[1] / "scripts" / "rdagent_bridge.py"
    spec = importlib.util.spec_from_file_location("rdagent_bridge_single_arm", bridge_path)
    assert spec and spec.loader
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    factor_code = "def calculate(data):\n    return data\n"
    model_code = "class Net:\n    pass\n"
    rounds = {
        0: {
            "hypothesis": {"action": "factor"},
            "feedback": {"decision": True},
            "runner_snapshot": {
                "kind": "factor",
                "artifacts": [
                    {
                        "name": "factor_a",
                        "code": factor_code,
                        "code_sha256": bridge._sha256(factor_code),
                    }
                ],
                "based_artifacts": [
                    {
                        "kind": "model",
                        "name": "based_model",
                        "model_type": "Tabular",
                        "code": model_code,
                        "code_sha256": bridge._sha256(model_code),
                    }
                ],
                "base_features": {"BASE": "$close"},
            },
        }
    }
    code_root = tmp_path / "code"
    values_root = tmp_path / "values"
    code_root.mkdir()
    values_root.mkdir()

    assert bridge._materialize_quant_bundles(
        rounds,
        code_root=code_root,
        values_root=values_root,
        feature_set_id="governed-baseline",
        feature_set_sha256="a" * 64,
    ) == []
    coverage = bridge._fin_quant_arm_coverage(rounds)
    assert coverage["complete"] is False
    assert coverage["single_arm"] is True
    assert coverage["attempted_arms"] == ["factor"]
    assert coverage["accepted_arms"] == ["factor"]
    assert coverage["based_artifacts_count_as_arm_coverage"] is False
    outcome = bridge._fin_quant_research_outcome(coverage, [])
    assert outcome["status"] == "governed_negative"
    assert outcome["reason_code"] == "arm_attempt_coverage_incomplete"
    assert outcome["capital_authority"] is False
    assert worker_module._validate_fin_quant_research_result(
        {
            "fin_quant_coverage": coverage,
            "fin_quant_outcome": outcome,
            "quant_bundles": [],
        }
    ) == outcome
    with pytest.raises(ValueError, match="arm-coverage evidence is inconsistent"):
        worker_module._validate_fin_quant_research_result(
            {
                "fin_quant_coverage": coverage,
                "fin_quant_outcome": {**outcome, "capital_authority": True},
                "quant_bundles": [],
            }
        )


def test_quant_exporter_does_not_count_an_empty_current_factor_arm(
    tmp_path: Path,
) -> None:
    bridge_path = Path(__file__).parents[1] / "scripts" / "rdagent_bridge.py"
    spec = importlib.util.spec_from_file_location("rdagent_bridge_empty_arm", bridge_path)
    assert spec and spec.loader
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    factor_code = "def calculate(data):\n    return data\n"
    model_code = "class Net:\n    pass\n"
    rounds = {
        0: {
            "feedback": {"decision": True},
            "runner_snapshot": {
                "kind": "factor",
                "artifacts": [],
                "based_artifacts": [
                    {
                        "kind": "factor",
                        "name": "based_factor",
                        "code": factor_code,
                        "code_sha256": bridge._sha256(factor_code),
                    }
                ],
            },
        },
        1: {
            "feedback": {"decision": True},
            "runner_snapshot": {
                "kind": "model",
                "artifacts": [
                    {
                        "name": "current_model",
                        "model_type": "Tabular",
                        "code": model_code,
                        "code_sha256": bridge._sha256(model_code),
                    }
                ],
                "based_artifacts": [],
            },
        },
    }
    code_root = tmp_path / "code"
    values_root = tmp_path / "values"
    code_root.mkdir()
    values_root.mkdir()

    assert bridge._materialize_quant_bundles(
        rounds,
        code_root=code_root,
        values_root=values_root,
        feature_set_id="governed-baseline",
        feature_set_sha256="a" * 64,
    ) == []
    coverage = bridge._fin_quant_arm_coverage(rounds)
    assert coverage["accepted_arms"] == ["model"]
    assert coverage["single_arm"] is True
    assert coverage["attempted_arms"] == ["factor", "model"]
    outcome = bridge._fin_quant_research_outcome(coverage, [])
    assert outcome["status"] == "governed_negative"
    assert outcome["reason_code"] == "arm_acceptance_incomplete"
