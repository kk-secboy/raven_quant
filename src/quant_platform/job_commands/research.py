"""RD-Agent research command builders for LocalJobWorker."""

from __future__ import annotations

import json
from pathlib import Path

from ..rdagent_runtime import (
    probe_rdagent,
    rdagent_command,
    require_matching_rdagent_runtime_identity,
)
from ..rdagent_scenarios import get_rdagent_scenario
from ._shared import _require_supported_rdagent_execution


def rdagent_job_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    _require_supported_rdagent_execution(payload)
    scenario = get_rdagent_scenario(str(payload.get("scenario") or "fin_factor"))
    raw_strategy_signal_binding = payload.get(
        "strategy_research_signal_binding"
    )
    if raw_strategy_signal_binding is not None and not isinstance(
        raw_strategy_signal_binding, dict
    ):
        raise ValueError("fin_strategy signal binding payload is invalid")
    llm = worker.runtime_secrets.get("llm")
    runtime_env = None
    if llm:
        runtime_env = {
            worker.settings.rdagent_llm_key_env: llm["api_key"],
            "OPENAI_API_BASE": llm.get("api_base", ""),
            "CHAT_MODEL": llm.get("chat_model", "gpt-4.1-mini"),
        }
    local_runtime = probe_rdagent(
        worker.settings,
        worker.project_root,
        runtime_env=runtime_env,
        force_local=True,
    )
    require_matching_rdagent_runtime_identity(
        payload.get("expected_rdagent_runtime"),
        local_runtime.get("runtime_identity", local_runtime),
    )
    for asset_id in payload.get("asset_ids") or []:
        worker.rdagent_candidates.import_manifest(
            worker.settings.data_root
            / "artifacts"
            / "research-assets"
            / str(asset_id)
            / "manifest.json",
            actor="worker",
        )
    output = worker.settings.data_root / "artifacts" / "rdagent" / payload["research_run_id"]
    trace = output / "trace"
    result_path = output / "result.json"
    output.mkdir(parents=True, exist_ok=True)
    command, env = rdagent_command(
        worker.settings,
        project_root=worker.project_root,
        trace_path=trace,
        result_path=result_path,
        dataset_path=(
            Path(str(payload["dataset_path"]))
            if scenario.requires_dataset and payload.get("dataset_path")
            else None
        ),
        loop_n=int(payload["loop_n"]),
        duration=str(payload["duration"]),
        periods=payload.get("periods"),
        objective=str(payload["objective"]),
        scenario=scenario.id,
        asset_ids=list(payload.get("asset_ids") or []),
        asset_manifest_sha256=dict(payload.get("asset_manifest_sha256") or {}),
        feature_set=payload.get("feature_set"),
        strategy_horizon_profile=(
            (
                payload.get("strategy_horizon_profile")
                or payload.get("horizon_profile")
            )
            if scenario.id == "fin_strategy"
            else None
        ),
        incumbent_strategy_version_id=(
            str(payload["incumbent_strategy"]["id"])
            if isinstance(payload.get("incumbent_strategy"), dict)
            else None
        ),
        strategy_research_signal_binding=(
            dict(raw_strategy_signal_binding)
            if isinstance(raw_strategy_signal_binding, dict)
            else None
        ),
    )
    if scenario.id == "fin_quant":
        baseline = worker._freeze_fin_quant_baseline(payload)
        reference = {
            "contract_version": baseline["contract_version"],
            "kind": baseline["kind"],
            "candidate_id": baseline["candidate_id"],
            "candidate_manifest_sha256": baseline[
                "candidate_manifest_sha256"
            ],
            "admission_evidence_sha256": baseline[
                "admission_evidence_sha256"
            ],
            "selection_evidence_sha256": baseline[
                "selection_evidence_sha256"
            ],
            "feature_set_id": baseline.get("feature_set_id"),
            "feature_set_definition_sha256": baseline.get(
                "feature_set_definition_sha256"
            ),
            "combiner": baseline.get("combiner"),
            "stacking": baseline.get("stacking"),
            "component_count": len(baseline.get("components") or []),
            "quant_retraining_supported": baseline[
                "quant_retraining_supported"
            ],
            "unsupported_reason_code": baseline.get(
                "unsupported_reason_code"
            ),
        }
        env["QUANTLAB_PREDICTION_CHAMPION_JSON"] = json.dumps(
            reference, ensure_ascii=False, sort_keys=True
        )
    if scenario.factor_output or scenario.id == "fin_quant":
        active_library = next(
            (
                item
                for item in worker.factor_library.list_library_versions()
                if item["status"] == "active"
            ),
            None,
        )
        if active_library is None:
            raise ValueError("RD-Agent factor research has no active factor library")
        env["QUANTLAB_FACTOR_LIBRARY_VERSION_ID"] = str(active_library["id"])
        env["QUANTLAB_FACTOR_LIBRARY_DEFINITION_SHA256"] = str(
            active_library["definition_sha256"]
        )
    if llm:
        env[worker.settings.rdagent_llm_key_env] = llm["api_key"]
        env["OPENAI_API_BASE"] = llm.get("api_base", "")
        env["CHAT_MODEL"] = llm.get("chat_model", "gpt-4.1-mini")
    return command, result_path, env


COMMANDS = {
    "rdagent_run": rdagent_job_command,
    "rdagent_factor": rdagent_job_command,
    "rdagent_model": rdagent_job_command,
    "rdagent_quant": rdagent_job_command,
    "rdagent_factor_report": rdagent_job_command,
    "rdagent_data_science": rdagent_job_command,
    "rdagent_llm_finetune": rdagent_job_command,
}
