"""Shared helpers for the job_commands domain builders.

These functions were lifted verbatim out of ``worker.py`` so the per-domain
command builders can use them without importing the worker module itself.
``worker`` re-exports the ones tests and other modules still import from it.
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path
from typing import Any

from quant_data.config import Settings
from quant_data.path_utils import to_wsl_path as _to_wsl_path

from ..feature_set_registry import get_feature_set
from ..model_recompute import GOVERNED_MODEL_ENGINES
from ..model_research_governance import canonical_sha256 as model_canonical_sha256
from ..rdagent_scenarios import FROZEN_RDAGENT_SCENARIOS, get_rdagent_scenario
from ..simulation_store import (
    build_settlement_calendar_binding,
    validate_settlement_calendar_binding,
)


def _frozen_model_label_contract(model_signal: dict[str, Any] | None) -> tuple[dict, str] | None:
    """Resolve the one sealed label contract shared by a model signal."""

    if model_signal is None:
        return None
    candidates = []
    if model_signal.get("candidate") is not None:
        candidates.append(model_signal["candidate"])
    for component in model_signal.get("components") or []:
        candidate = component.get("candidate") if isinstance(component, dict) else None
        if candidate is not None:
            candidates.append(candidate)
    contracts: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        admission = dict(candidate.admission_evidence_json or {})
        for profile in (admission.get("profiles") or {}).values():
            for cell in ((profile or {}).get("seeds") or {}).values():
                contract = (cell or {}).get("model_label_contract")
                digest = str((cell or {}).get("model_label_contract_sha256") or "")
                if not isinstance(contract, dict) or model_canonical_sha256(contract) != digest:
                    raise ValueError("model signal label contract evidence is invalid")
                contracts[digest] = dict(contract)
    if len(contracts) != 1:
        raise ValueError("model signal does not have one frozen label contract")
    digest, contract = next(iter(contracts.items()))
    return contract, digest

def _qlib_workflow_environment(settings: Settings, *, is_wsl: bool) -> dict[str, str]:
    artifact_root = settings.data_root / "artifacts" / "mlflow"
    return {
        "_MLFLOW_SERVER_ARTIFACT_ROOT": (
            _to_wsl_path(artifact_root) if is_wsl else str(artifact_root)
        )
    }

def _model_evaluation_attempt_result_path(
    evaluation_root: Path,
    job: dict[str, Any],
) -> Path:
    """Allocate a producer-only result path for one model evaluation attempt.

    A durable job can be resubmitted with the same job id, and administrative
    retries may reset its numerical attempt counter.  The random execution
    token therefore forms part of the directory identity.  Nothing below this
    attempts directory is ever registered as immutable evidence.
    """

    try:
        attempt = max(1, int(job.get("attempts") or 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("model evaluation attempt counter is invalid") from exc
    attempt_root = (
        evaluation_root
        / "attempts"
        / f"attempt-{attempt:04d}-{uuid.uuid4().hex}"
    )
    attempt_root.mkdir(parents=True, exist_ok=False)
    return attempt_root / "result.json"

def _frozen_model_engine(model_signal: dict) -> str:
    recipe = dict(model_signal.get("recipe") or {})
    recipe_hyperparameters = dict(recipe.get("model_hyperparameters") or {})
    signal_hyperparameters = dict(model_signal.get("model_hyperparameters") or {})
    engine = str(
        recipe.get("model_engine")
        or recipe_hyperparameters.get("model_engine")
        or signal_hyperparameters.get("model_engine")
        or "rdagent_pytorch"
    )
    if engine not in GOVERNED_MODEL_ENGINES:
        raise ValueError("frozen strategy requests an ungoverned model engine")
    return engine

def _frozen_evaluation_feature_set(payload: dict) -> dict:
    feature_set_id = str(payload.get("feature_set_id") or "")
    embedded = payload.get("feature_set")
    feature_set = dict(embedded) if isinstance(embedded, dict) else get_feature_set(feature_set_id)
    if (
        not feature_set_id
        or str(feature_set.get("id") or "") != feature_set_id
        or str(feature_set.get("definition_sha256") or "")
        != str(payload.get("feature_set_definition_sha256") or "")
        or not isinstance(feature_set.get("features"), dict)
        or not feature_set["features"]
    ):
        raise ValueError("model evaluation feature set changed after job creation")
    return feature_set

def _require_supported_simulation_execution(
    job_kind: str, *, execution_adapter: str | None = None
) -> None:
    """Fail closed for every retired short-selling execution path.

    Historical pair jobs remain queryable and generic job retry is intentionally
    broad.  The worker is therefore the final authority boundary: neither an
    old queued job nor a retried cancelled job may start pair replay code after
    the long-only Autopilot release.
    """

    if (
        str(job_kind) == "simulation_replay"
        and str(execution_adapter or "") != "long_only"
    ):
        raise ValueError(
            "pair simulation execution is retired; historical ledgers are read-only"
        )

def _require_supported_rdagent_execution(payload: dict) -> None:
    """Fail closed for every frozen RD-Agent scenario.

    Frozen scenarios keep their registry entries and historical runs, but the
    worker is the final authority boundary: neither an old queued job nor a
    retried cancelled job may start fin_factor, fin_model, general_model,
    data_science, or llm_finetune execution after the freeze.  fin_strategy
    shares the generic rdagent_run kind with general_model, so the payload
    scenario, not the job kind, decides that case.
    """

    scenario = get_rdagent_scenario(str(payload.get("scenario") or "fin_factor"))
    if scenario.id in FROZEN_RDAGENT_SCENARIOS:
        raise ValueError(
            f"RD-Agent scenario {scenario.id} is frozen; "
            "historical runs and artifacts are read-only"
        )

def _bind_daily_simulation_settlement_calendar(
    manifest: dict[str, Any], execution_dataset: dict[str, Any]
) -> dict[str, Any]:
    """Re-verify the batch calendar against the exact worker-side dataset."""

    result = dict(manifest)
    if str(result.get("execution_frequency") or "") != "day":
        return result
    settlement_trade_date = date.fromisoformat(str(result["trade_date"]))
    provenance = dict(execution_dataset.get("provenance") or {})
    persisted = validate_settlement_calendar_binding(
        result.get("settlement_calendar_binding"),
        trade_date=settlement_trade_date,
        dataset_identity_sha256=str(
            provenance.get("dataset_identity_sha256") or ""
        ),
        dataset_lineage_id=str(provenance.get("dataset_lineage_id") or ""),
    )
    observed = build_settlement_calendar_binding(
        execution_dataset,
        trade_date=settlement_trade_date,
    )
    if observed != persisted:
        raise ValueError(
            "daily simulation settlement calendar changed after batch binding"
        )
    result["settlement_calendar_binding"] = observed
    return result
