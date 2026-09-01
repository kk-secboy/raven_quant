"""Evaluation, research-audit and backtest command builders for LocalJobWorker."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from quant_data.path_utils import to_wsl_path as _to_wsl_path

from ..cost_model import CostModelConfig
from ..feature_set_registry import get_feature_set, register_feature_set
from ..forward_only_rehabilitation import (
    EVIDENCE_MODE_REPLAY,
    EVIDENCE_MODE_SEALED,
    REPLAY_MARKERS,
    incomplete_family_eligibility_for_version,
    require_replay_config,
)
from ..parameter_experiments import merge_admitted_trial_ledgers
from ..research_execution_cadence import build_research_execution_cadence_contract
from ..research_label_binding import (
    resolve_research_label_binding,
    validate_research_label_binding,
)
from ..research_tournament import RESEARCH_SCREENING_MARKERS
from ..strategy_research_evaluation import STRATEGY_RESEARCH_EVALUATION_MODES
from ..transparent_baseline_runner import (
    TRANSPARENT_BASELINE_JOB_RUNNER_FIELD,
    TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD,
    TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD,
    require_transparent_baseline_runner,
)
from ._shared import (
    _frozen_evaluation_feature_set,
    _frozen_model_engine,
    _frozen_model_label_contract,
    _model_evaluation_attempt_result_path,
    _qlib_workflow_environment,
)


def multiface_audit_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = worker.settings.data_root / "artifacts" / "execution-data" / job["id"]
    result_path = output / "result.json"
    command = [
        sys.executable,
        "-m",
        "quant_platform.db_cli",
        "audit-multiface",
        "--dataset",
        str(payload["dataset"]),
        "--result",
        str(result_path),
    ]
    if payload.get("snapshot_name"):
        command.extend(["--snapshot", str(payload["snapshot_name"])])
    return command, result_path, {}


def minute_research_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = worker.settings.data_root / "artifacts" / "minute-research" / job["id"]
    result_path = output / "result.json"
    script = worker.project_root / "scripts" / "run_minute_factor_research.py"
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            _to_wsl_path(Path(payload["dataset_path"]))
            if is_wsl
            else str(Path(payload["dataset_path"])),
            "--output",
            _to_wsl_path(result_path) if is_wsl else str(result_path),
            "--start",
            payload["start"],
            "--end",
            payload["end"],
            "--horizons",
            ",".join(str(item) for item in payload["horizons"]),
            "--cost-rate",
            str(payload["cost_rate"]),
            "--tracking-uri",
            worker.settings.mlflow_tracking_uri,
        ]
    )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def qlib_baseline_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = worker.settings.data_root / "artifacts" / "qlib" / job["id"]
    script = worker.project_root / "scripts" / "run_qlib_baseline.py"
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            _to_wsl_path(Path(payload["dataset_path"]))
            if is_wsl
            else str(Path(payload["dataset_path"])),
            "--output",
            _to_wsl_path(output) if is_wsl else str(output),
            "--tracking-uri",
            worker.settings.mlflow_tracking_uri,
            "--market",
            payload["market"],
            "--benchmark",
            payload["benchmark"],
            "--account",
            str(payload["account"]),
            "--topk",
            str(payload["topk"]),
            "--n-drop",
            str(payload["n_drop"]),
            "--open-cost",
            str(payload["open_cost"]),
            "--close-cost",
            str(payload["close_cost"]),
            "--min-cost",
            str(payload["min_cost"]),
        ]
    )
    return (
        command,
        output / "result.json",
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def model_ensemble_evaluate_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = (
        worker.settings.data_root
        / "artifacts"
        / "model-ensemble-evaluations"
        / str(payload["tournament_id"])
        / job["id"]
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str | Path) -> str:
        path = Path(value)
        return _to_wsl_path(path) if is_wsl else str(path)

    candidates: list[dict] = []
    for raw_candidate in payload.get("candidates") or []:
        candidate = dict(raw_candidate)
        components: list[dict] = []
        for raw_component in candidate.get("components") or []:
            component = dict(raw_component)
            grid = dict(component.get("prediction_grid") or {})
            profiles: dict[str, dict] = {}
            for profile_id, raw_profile in (grid.get("profiles") or {}).items():
                profile = dict(raw_profile)
                seeds = {
                    str(seed): {
                        **dict(cell),
                        "predictions_path": runtime_path(
                            str(cell["predictions_path"])
                        ),
                    }
                    for seed, cell in (profile.get("seeds") or {}).items()
                }
                profiles[str(profile_id)] = {**profile, "seeds": seeds}
            component["prediction_grid"] = {**grid, "profiles": profiles}
            components.append(component)
        candidate["components"] = components
        candidates.append(candidate)
    manifest = {
        "contract_version": "model-ensemble-evaluation-input-v1",
        "tournament_id": payload["tournament_id"],
        "dataset": payload["dataset"],
        "dataset_identity_sha256": payload["dataset_identity_sha256"],
        "evaluation_profiles": payload.get("evaluation_profiles") or [],
        "ensemble_label_contract": payload["ensemble_label_contract"],
        "research_execution_cadence": payload[
            "research_execution_cadence"
        ],
        "candidates": candidates,
        "universe": payload.get("universe", "cn_all"),
        "benchmark": payload.get("benchmark", "SH000300"),
        "account": int(payload.get("account", 100_000_000)),
        "topk": int(payload.get("topk", 50)),
        "n_drop": int(payload.get("n_drop", 5)),
        "open_cost": float(payload.get("open_cost", 0.0005)),
        "close_cost": float(payload.get("close_cost", 0.0015)),
        "min_cost": float(payload.get("min_cost", 5.0)),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    script = worker.project_root / "scripts" / "evaluate_model_ensemble.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            runtime_path(str(payload["dataset_path"])),
            "--manifest",
            runtime_path(manifest_path),
            "--output",
            runtime_path(result_path),
        ]
    )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def model_evaluate_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    if not str(os.getenv("MODEL_SANDBOX_IMAGE") or "").strip():
        raise ValueError("MODEL_SANDBOX_IMAGE is required for model isolation")
    evaluation_name = (
        "model-evaluations"
        if job["kind"] == "model_evaluate"
        else "quant-bundle-evaluations"
    )
    output = (
        worker.settings.data_root / "artifacts" / evaluation_name / payload["research_run_id"]
        / job["id"]
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    result_path = _model_evaluation_attempt_result_path(output, job)
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str | Path) -> str:
        path = Path(value)
        return _to_wsl_path(path) if is_wsl else str(path)

    candidates = []
    for candidate in payload["candidates"]:
        item = dict(candidate)
        if job["kind"] == "model_evaluate":
            item["code_path"] = runtime_path(str(item["code_path"]))
        else:
            item["factors"] = [
                {
                    **factor,
                    "code_path": runtime_path(str(factor["code_path"])),
                    **(
                        {
                            "submitted_values_path": runtime_path(
                                str(factor["submitted_values_path"])
                            )
                        }
                        if factor.get("submitted_values_path")
                        else {}
                    ),
                }
                for factor in item["factors"]
            ]
            item["model"] = {
                **item["model"],
                "code_path": runtime_path(str(item["model"]["code_path"])),
            }
            frozen_baseline = dict(
                item.get("baseline_prediction_champion") or {}
            )
            runtime_baseline = json.loads(
                json.dumps(frozen_baseline, ensure_ascii=False)
            )
            if runtime_baseline.get("kind") == "model":
                runtime_baseline["model"]["code_path"] = runtime_path(
                    str(runtime_baseline["model"]["code_path"])
                )
                for profile in runtime_baseline.get(
                    "profiles", {}
                ).values():
                    for cell in profile.get("seeds", {}).values():
                        cell["predictions_path"] = runtime_path(
                            str(cell["predictions_path"])
                        )
                        cell["portfolio_report_path"] = runtime_path(
                            str(cell["portfolio_report_path"])
                        )
                        if cell.get("checkpoint_path"):
                            cell["checkpoint_path"] = runtime_path(
                                str(cell["checkpoint_path"])
                            )
            elif runtime_baseline.get("kind") == "ensemble":
                for component in runtime_baseline.get("components", []):
                    component["model"]["code_path"] = runtime_path(
                        str(component["model"]["code_path"])
                    )
                    for profile in component.get("profiles", {}).values():
                        for cell in profile.get("seeds", {}).values():
                            for path_key in (
                                "predictions_path",
                                "checkpoint_path",
                                "portfolio_report_path",
                            ):
                                cell[path_key] = runtime_path(
                                    str(cell[path_key])
                                )
                for profile in runtime_baseline.get("profiles", {}).values():
                    for cell in profile.get("seeds", {}).values():
                        cell["predictions_path"] = runtime_path(
                            str(cell["predictions_path"])
                        )
                        cell["portfolio_report_path"] = runtime_path(
                            str(cell["portfolio_report_path"])
                        )
                        for member in cell.get(
                            "member_prediction_artifacts", []
                        ):
                            member["predictions_path"] = runtime_path(
                                str(member["predictions_path"])
                            )
            item["baseline_prediction_runtime"] = runtime_baseline
        candidates.append(item)
    feature_set = _frozen_evaluation_feature_set(payload)
    evaluation_label_binding = validate_research_label_binding(
        payload.get("research_label_binding") or {}
    )
    if (
        payload.get("research_label_binding_sha256")
        != evaluation_label_binding["binding_sha256"]
    ):
        raise ValueError("active model evaluation label binding is invalid")
    research_execution_cadence = (
        build_research_execution_cadence_contract(
            str(evaluation_label_binding["horizon_profile"])
        )
    )
    quant_label_binding = (
        evaluation_label_binding
        if job["kind"] == "quant_bundle_evaluate"
        else None
    )
    if quant_label_binding is not None and any(
        candidate.get("research_label_binding") != quant_label_binding
        or candidate.get("research_label_binding_sha256")
        != quant_label_binding["binding_sha256"]
        for candidate in candidates
    ):
        raise ValueError(
            "quant evaluation candidate labels differ from the research window"
        )
    manifest = {
        "research_run_id": payload["research_run_id"],
        "candidates": candidates,
        "feature_set_id": payload["feature_set_id"],
        # Dynamic SOTA feature sets are registered in the long-lived
        # worker process but are not necessarily present in the clean
        # evaluator subprocess.  Freeze the complete definition into
        # the immutable job manifest so the subprocess validates the
        # same feature set instead of falling back to its static
        # registry.
        "feature_set": feature_set,
        "research_label_binding": evaluation_label_binding,
        "research_label_binding_sha256": evaluation_label_binding[
            "binding_sha256"
        ],
        "research_execution_cadence": research_execution_cadence,
        **(
            {
                "evaluation_stage": str(
                    payload.get("evaluation_stage") or "model_full"
                ),
                "research_window_contract": payload[
                    "research_window_contract"
                ],
                "research_window_contract_sha256": payload[
                    "research_window_contract_sha256"
                ],
                "label_horizon_sessions": payload[
                    "label_horizon_sessions"
                ],
                **(
                    {
                        "research_tournament_id": payload[
                            "research_tournament_id"
                        ],
                        "candidate_bindings": payload[
                            "candidate_bindings"
                        ],
                    }
                    if str(payload.get("evaluation_stage") or "")
                    == "feature_screen"
                    else {}
                ),
            }
            if job["kind"] == "model_evaluate"
            else {}
        ),
        **(
            {
                "horizon_profile": quant_label_binding[
                    "horizon_profile"
                ],
                "periods": quant_label_binding["periods"],
                "research_window_contract": quant_label_binding[
                    "research_window_contract"
                ],
                "research_window_contract_sha256": quant_label_binding[
                    "research_window_contract_sha256"
                ],
                "label_horizon_sessions": quant_label_binding[
                    "label_horizon_sessions"
                ],
                "research_label_binding": quant_label_binding,
                "research_label_binding_sha256": quant_label_binding[
                    "binding_sha256"
                ],
                "baseline_prediction_champion": payload[
                    "baseline_prediction_champion"
                ],
                "research_tournament_id": payload[
                    "research_tournament_id"
                ],
                "parent_research_tournament_id": payload[
                    "parent_research_tournament_id"
                ],
                "research_tournament_manifest_sha256": payload[
                    "research_tournament_manifest_sha256"
                ],
                "research_trial_ids": payload["research_trial_ids"],
                **RESEARCH_SCREENING_MARKERS,
            }
            if job["kind"] == "quant_bundle_evaluate"
            and quant_label_binding is not None
            else (
                {
                    "baseline_prediction_champion": payload[
                        "baseline_prediction_champion"
                    ],
                    "research_tournament_id": payload[
                        "research_tournament_id"
                    ],
                    "parent_research_tournament_id": payload[
                        "parent_research_tournament_id"
                    ],
                    "research_tournament_manifest_sha256": payload[
                        "research_tournament_manifest_sha256"
                    ],
                    "research_trial_ids": payload["research_trial_ids"],
                    **RESEARCH_SCREENING_MARKERS,
                }
                if job["kind"] == "quant_bundle_evaluate"
                else {}
            )
        ),
        "dataset_identity_sha256": payload["dataset_identity_sha256"],
        "evaluation_profiles": payload.get("evaluation_profiles") or [],
        "universe": payload.get("universe", "cn_all"),
        "benchmark": payload.get("benchmark", "SH000300"),
        "account": int(payload.get("account", 100_000_000)),
        "topk": int(payload.get("topk", 50)),
        "n_drop": int(payload.get("n_drop", 5)),
        "open_cost": float(payload.get("open_cost", 0.0005)),
        "close_cost": float(payload.get("close_cost", 0.0015)),
        "min_cost": float(payload.get("min_cost", 5.0)),
        "model_timeout_seconds": int(payload.get("model_timeout_seconds", 7200)),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    script_name = (
        "evaluate_model_batch.py"
        if job["kind"] == "model_evaluate"
        else "evaluate_quant_bundle.py"
    )
    script = worker.project_root / "scripts" / script_name
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            runtime_path(str(payload["dataset_path"])),
            "--manifest",
            runtime_path(manifest_path),
            "--output",
            runtime_path(result_path),
        ]
    )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def factor_evaluate_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = (
        worker.settings.data_root
        / "artifacts"
        / "factor-evaluations"
        / payload["research_run_id"]
        / job["id"]
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str) -> str:
        return _to_wsl_path(Path(value)) if is_wsl else str(value)

    label_binding = resolve_research_label_binding(payload)
    if label_binding is not None and any(
        int(item.get("label_horizon_days") or 0)
        != int(label_binding["label_horizon_sessions"])
        for item in payload.get("candidates") or []
    ):
        raise ValueError(
            "factor evaluation candidate labels differ from the research window"
        )

    promoted = worker.research.list_candidates(status="promoted", limit=500)
    library_comparisons: list[dict[str, str]] = []
    materialization_root = (
        worker.settings.data_root
        / "artifacts"
        / "factor-library-materializations"
        / str(payload["dataset_identity_sha256"])
    )
    manifests = sorted(
        materialization_root.glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if manifests:
        materialized = json.loads(manifests[0].read_text(encoding="utf-8"))
        if (
            materialized.get("dataset_identity_sha256")
            == payload["dataset_identity_sha256"]
        ):
            for name, evidence in (materialized.get("completed") or {}).items():
                path = manifests[0].parent / str(evidence["relative_path"])
                if path.is_file():
                    library_comparisons.append(
                        {
                            "candidate_id": f"library:{name}",
                            "path": runtime_path(str(path)),
                            "source": "unified_factor_library",
                        }
                    )
    manifest = {
        "research_run_id": payload["research_run_id"],
        "candidates": [
            {
                **item,
                "code_path": runtime_path(item["code_path"]),
                "submitted_values_path": runtime_path(item["values_path"]),
            }
            for item in payload["candidates"]
        ],
        "dataset_identity_sha256": payload["dataset_identity_sha256"],
        "periods": payload["periods"],
        "evaluation_profiles": payload.get("evaluation_profiles") or [],
        **(
            {
                "horizon_profile": label_binding["horizon_profile"],
                "research_window_contract": label_binding[
                    "research_window_contract"
                ],
                "research_window_contract_sha256": label_binding[
                    "research_window_contract_sha256"
                ],
                "label_horizon_sessions": label_binding[
                    "label_horizon_sessions"
                ],
                "research_label_binding": label_binding,
                "research_label_binding_sha256": label_binding[
                    "binding_sha256"
                ],
            }
            if label_binding is not None
            else {}
        ),
        "universe": payload.get("universe", "cn_all"),
        "min_daily_instruments": int(payload.get("min_daily_instruments", 50)),
        "comparison_values": library_comparisons + [
            {
                "candidate_id": str(item["id"]),
                "path": runtime_path(str(item["values_path"])),
                "source": "promoted_library",
            }
            for item in promoted
            if item.get("values_path") and Path(item["values_path"]).exists()
        ],
        "cost_model": CostModelConfig.from_mapping(payload.get("cost_model")).to_dict(),
        "cost_reference_order_value": float(
            payload.get("cost_reference_order_value", 100_000.0)
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    script = worker.project_root / "scripts" / "evaluate_factor_batch.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            _to_wsl_path(Path(payload["dataset_path"]))
            if is_wsl
            else str(Path(payload["dataset_path"])),
            "--manifest",
            _to_wsl_path(manifest_path) if is_wsl else str(manifest_path),
            "--output",
            _to_wsl_path(result_path) if is_wsl else str(result_path),
            "--tracking-uri",
            worker.settings.mlflow_tracking_uri,
        ]
    )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def factor_library_materialize_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    embedded_feature_set = payload.get("feature_set_definition")
    feature_set = (
        register_feature_set(dict(embedded_feature_set))
        if isinstance(embedded_feature_set, dict)
        else get_feature_set(str(payload["feature_set_id"]))
    )
    if (
        feature_set["definition_sha256"]
        != payload["feature_set_definition_sha256"]
    ):
        raise ValueError("factor library materialization feature set changed")
    output = (
        worker.settings.data_root
        / "artifacts"
        / "factor-library-materializations"
        / str(payload["dataset_identity_sha256"])
        / str(payload["feature_set_definition_sha256"])[:16]
    )
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "manifest.json"
    feature_set_path: Path | None = None
    if isinstance(embedded_feature_set, dict):
        feature_set_path = output / "feature-set-input.json"
        feature_set_path.write_text(
            json.dumps(
                feature_set,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str | Path) -> str:
        path = Path(value)
        return _to_wsl_path(path) if is_wsl else str(path)

    script = worker.project_root / "scripts" / "materialize_factor_library.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            runtime_path(payload["dataset_path"]),
            "--output",
            runtime_path(output),
            "--feature-set-id",
            feature_set["id"],
            "--universe",
            str(payload["universe"]),
            "--start",
            str(payload["start"]),
            "--end",
            str(payload["end"]),
        ]
    )
    if feature_set_path is not None:
        command.extend(
            ["--feature-set-definition", runtime_path(feature_set_path)]
        )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def factor_library_cluster_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    materialization = Path(str(payload["materialization_path"])).resolve(
        strict=True
    )
    output = materialization / "clusters"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str | Path) -> str:
        path = Path(value)
        return _to_wsl_path(path) if is_wsl else str(path)

    script = worker.project_root / "scripts" / "cluster_factor_library.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--materialization",
            runtime_path(materialization),
            "--output",
            runtime_path(output),
        ]
    )
    return command, result_path, {}


def factor_sota_evaluate_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    evaluation_scope_id = str(
        payload.get("evaluation_scope_id")
        or payload.get("research_campaign_id")
        or job["id"]
    )
    output = (
        worker.settings.data_root
        / "artifacts"
        / "factor-sota-evaluations"
        / evaluation_scope_id
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str | Path) -> str:
        path = Path(value)
        return _to_wsl_path(path) if is_wsl else str(path)

    manifest = {
        **payload,
        "frozen_model_runtime_code_path": runtime_path(
            payload["frozen_model"]["code_path"]
        ),
        "baseline_members": [
            {**item, "values_path": runtime_path(item["values_path"])}
            for item in payload.get("baseline_members") or []
        ],
        "candidates": [
            {**item, "values_path": runtime_path(item["values_path"])}
            for item in payload.get("candidates") or []
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    script = worker.project_root / "scripts" / "evaluate_factor_sota_increment.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            runtime_path(payload["dataset_path"]),
            "--manifest",
            runtime_path(manifest_path),
            "--output",
            runtime_path(result_path),
        ]
    )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def external_factor_evaluate_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = (
        worker.settings.data_root / "artifacts" / "external-factor-evaluations" / job["id"]
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str | Path) -> str:
        path = Path(value)
        return _to_wsl_path(path) if is_wsl else str(path)

    candidates = (
        worker._resolve_information_factor_candidates(payload)
        if job["kind"] == "information_factor_evaluate"
        else payload["candidates"]
    )
    manifest = {
        "research_run_id": job["id"],
        "dataset": payload["dataset"],
        "dataset_identity_sha256": payload["dataset_identity_sha256"],
        "periods": payload["periods"],
        "evaluation_profiles": payload.get("evaluation_profiles") or [],
        "universe": payload.get("universe", "cn_all"),
        "benchmark": payload.get("benchmark", "SH000300"),
        "candidates": [
            {
                **item,
                "values_path": runtime_path(item["values_path"]),
            }
            for item in candidates
        ],
        "comparison_values": [
            {
                "candidate_id": str(item["id"]),
                "path": runtime_path(item["values_path"]),
                "source": "promoted_library",
            }
            for item in worker.research.list_candidates(
                status="promoted", limit=500
            )
            if item.get("values_path")
            and Path(str(item["values_path"])).exists()
        ],
        "cost_model": CostModelConfig.from_mapping(payload.get("cost_model")).to_dict(),
        "cost_reference_order_value": float(
            payload.get("cost_reference_order_value", 100_000.0)
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if job["kind"] == "information_factor_evaluate" and not candidates:
        result_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "evaluations": [],
                    "skipped": (
                        "all registered artifacts already have an evaluation outcome"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return [sys.executable, "-c", "pass"], result_path, {}
    script = worker.project_root / "scripts" / "evaluate_external_factor_batch.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            runtime_path(payload["dataset_path"]),
            "--manifest",
            runtime_path(manifest_path),
            "--output",
            runtime_path(result_path),
            "--tracking-uri",
            worker.settings.mlflow_tracking_uri,
        ]
    )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def model_refit_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    version = worker.strategies.get_version(str(payload["strategy_version_id"]))
    if version.get("status") != "approved" or version.get("is_legacy"):
        raise ValueError("model refit requires an approved non-legacy StrategySpec")
    model_signal = version.get("model_signal")
    if not isinstance(model_signal, dict):
        raise ValueError("model refit requires a governed model-prediction strategy")
    source_artifact = worker.model_artifacts.get(
        str(payload["source_model_artifact_id"])
    )
    if source_artifact.get("strategy_version_id") != version["id"]:
        raise ValueError("model refit source is not the active StrategySpec artifact")
    if source_artifact.get("status") == "retired":
        try:
            replay_artifact = worker.model_artifacts.get_by_key(
                version["id"],
                f"live-refit-{str(payload['signal_date']).replace('-', '')}",
            )
        except KeyError as exc:
            raise ValueError(
                "retired model source has no idempotent active refresh"
            ) from exc
        if (
            replay_artifact.get("status") != "active"
            or (replay_artifact.get("training_evidence") or {}).get(
                "source_model_artifact_id"
            )
            != source_artifact["id"]
        ):
            raise ValueError(
                "retired model source does not own the active replay artifact"
            )
    elif source_artifact.get("status") != "active":
        raise ValueError("model refit source is not the active StrategySpec artifact")
    operation = str(payload.get("operation") or "")
    if operation not in {"inference", "retrain"}:
        raise ValueError("model live refresh operation is invalid")
    source_training_evidence = source_artifact.get("training_evidence")
    source_periods = (
        source_training_evidence.get("periods")
        if isinstance(source_training_evidence, dict)
        else None
    )
    if not isinstance(source_periods, dict):
        raise ValueError("source ModelArtifact has no frozen training periods")
    output = (
        worker.settings.data_root
        / "artifacts"
        / "model-refits"
        / str(payload["strategy_version_id"])
        / str(payload["signal_date"])
    )
    manifest_path = output.parent / f"{payload['signal_date']}-manifest.json"
    result_path = output / "result.json"
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str | Path) -> str:
        path = Path(value)
        return _to_wsl_path(path) if is_wsl else str(path)

    candidate = worker.rdagent_candidates.get_model_candidate(
        str(model_signal["model_candidate_id"]), verify=True
    )
    base = dict(candidate.get("base_features_manifest_json") or {})
    frozen_model_engine = _frozen_model_engine(model_signal)
    feature_set = {
        "id": str(base.get("feature_set_id") or ""),
        "contract_version": str(base.get("contract_version") or ""),
        "features": dict(base.get("feature_expressions") or {}),
        "definition_sha256": str(base.get("definition_sha256") or ""),
    }
    manifest = {
        "contract_version": "model-live-refresh-v2",
        "operation": operation,
        "strategy_version_id": version["id"],
        "source_model_artifact_id": source_artifact["id"],
        "execution_environment_sha256": str(
            source_artifact["execution_environment_sha256"]
        ),
        "dataset": str(payload["dataset"]),
        "dataset_identity_sha256": str(payload["dataset_identity_sha256"]),
        "dataset_lineage_id": str(payload["dataset_lineage_id"]),
        "signal_date": str(payload["signal_date"]),
        "frozen_training_periods": {
            key: str(source_periods[key])
            for key in (
                "train_start",
                "train_end",
                "valid_start",
                "valid_end",
            )
        },
        "source_checkpoint_path": (
            runtime_path(str(source_artifact["checkpoint_path"]))
            if operation == "inference"
            else None
        ),
        "source_checkpoint_sha256": (
            str(source_artifact["checkpoint_sha256"])
            if operation == "inference"
            else None
        ),
        "source_checkpoint_format": (
            str(source_artifact["checkpoint_format"])
            if operation == "inference"
            else None
        ),
        "source_model_data_contract_sha256": str(
            source_artifact["model_data_contract_sha256"]
        ),
        "retrain_reason": (
            str(payload.get("retrain_reason") or "")
            if operation == "retrain"
            else ""
        ),
        "retrain_evidence": (
            payload.get("retrain_evidence")
            if operation == "retrain"
            else None
        ),
        "retrain_evidence_sha256": (
            str(payload.get("retrain_evidence_sha256") or "")
            if operation == "retrain"
            else ""
        ),
        "universe": version["universe"],
        "feature_set": feature_set,
        "feature_set_definition_sha256": str(
            model_signal["feature_set_definition_sha256"]
        ),
        "model": {
            "candidate_id": str(model_signal["model_candidate_id"]),
            "code_path": runtime_path(str(model_signal["code_path"])),
            "code_sha256": str(model_signal["model_code_sha256"]),
            "recipe_sha256": str(model_signal["model_recipe_sha256"]),
            "model_type": str(model_signal.get("model_type") or "Tabular"),
            "model_engine": frozen_model_engine,
            "training_hyperparameters": dict(
                model_signal.get("training_hyperparameters") or {}
            ),
            "seed": int(model_signal["primary_seed"]),
        },
        "refit_policy": dict(model_signal["refit_policy"]),
        "refit_policy_sha256": str(model_signal["refit_policy_sha256"]),
        "bundle_factors": [
            {
                "candidate_id": str(item["candidate_id"]),
                "code_sha256": str(item["code_sha256"]),
                "code_path": runtime_path(str(item["code_path"])),
            }
            for item in model_signal.get("bundle_factors") or []
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    script = worker.project_root / "scripts" / "run_model_refit.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            runtime_path(payload["dataset_path"]),
            "--manifest",
            runtime_path(manifest_path),
            "--output",
            runtime_path(output),
        ]
    )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def parameter_experiment_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    experiment = worker.parameter_experiments.get(payload["parameter_experiment_id"])
    output = Path(experiment["artifact_path"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    version = worker.strategies.get_version(payload["strategy_version_id"])
    if version.get("strategy_type") != "multifactor":
        raise ValueError("parameter experiments require a multifactor strategy")
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")
    backtest_script = worker.project_root / "scripts" / "run_multifactor_backtest.py"
    runner_sha256 = require_transparent_baseline_runner(
        config=version["config"],
        job_payload=payload,
        runner_path=backtest_script,
    )

    def runtime_path(value: str) -> str:
        return _to_wsl_path(Path(value)) if is_wsl else str(Path(value))

    with worker.strategies.engine.connect() as connection:
        model_signal = worker.strategies._model_signal_evidence(
            connection, version["config"]
        )
    governance = experiment["periods"].get("governance") or {}
    strategy_evaluation_mode = payload.get("strategy_evaluation_mode")
    if strategy_evaluation_mode is not None:
        if (
            strategy_evaluation_mode not in STRATEGY_RESEARCH_EVALUATION_MODES
            or governance.get("mode") != strategy_evaluation_mode
            or governance.get("final_oos_opened") is not False
            or payload.get("strategy_competition_plan_sha256")
            != governance.get("plan_sha256")
            or payload.get("strategy_competition_stage")
            != governance.get("stage")
            or payload.get("dataset_identity_sha256")
            != governance.get("dataset_identity_sha256")
        ):
            raise ValueError(
                "fin_strategy parameter experiment governance changed"
            )
        if model_signal is not None:
            raise ValueError(
                "fin_strategy rule comparison currently requires its frozen factor grid"
            )
    if model_signal is not None:
        if (
            governance.get("mode") != "model_portfolio_pre_final"
            or governance.get("final_oos_opened") is not False
            or payload.get("dataset_identity_sha256")
            != governance.get("dataset_identity_sha256")
            or governance.get("dataset_identity_sha256")
            != str(model_signal["candidate"].dataset_identity_sha256)
            or governance.get("model_signal_identity_sha256")
            != model_signal["identity"]["identity_sha256"]
            or governance.get("formal_admission_binding_sha256")
            != model_signal["formal_admission_binding"]["binding_sha256"]
        ):
            raise ValueError(
                "model portfolio experiment governance no longer matches admission"
            )
        admitted_multiple_testing = merge_admitted_trial_ledgers(
            model_signal["formal_admission_binding"]
        )
    else:
        admitted_multiple_testing = None

    manifest = {
        "experiment_id": experiment["id"],
        "strategy_version_id": version["id"],
        "dataset": experiment["dataset"],
        "benchmark": version["benchmark"],
        "universe": version["universe"],
        "execution_dataset": ((payload.get("execution_dataset") or {}).get("name")),
        "periods": experiment["periods"],
        "parameter_grid": experiment["parameter_grid"],
        "evaluation_mode": (
            "pre_final_portfolio_trial"
            if model_signal is not None
            else strategy_evaluation_mode
        ),
        "pre_final_cutoff": (
            str(model_signal["candidate"].pre_final_end)
            if model_signal is not None
            else governance.get("pre_final_cutoff")
        ),
        "historical_validation_periods": (
            {
                "start": model_signal["evaluation"].train_start.isoformat(),
                "end": model_signal["evaluation"].train_end.isoformat(),
            }
            if model_signal is not None
            else governance.get("historical_validation_periods")
        ),
        "strategy_trial_count": (
            int(admitted_multiple_testing["trial_count"])
            + len(experiment["trials"])
            if admitted_multiple_testing is not None
            else len(experiment["trials"])
        ),
        "shared_multiple_testing": admitted_multiple_testing,
        "model_signal": (
            model_signal["identity"] if model_signal is not None else None
        ),
        "model_formal_admission": (
            model_signal["formal_admission_binding"]
            if model_signal is not None
            else None
        ),
        "model_candidate": (
            {
                "candidate_manifest": dict(
                    model_signal["candidate"].manifest_json or {}
                ),
                "feature_set": {
                    "id": str(
                        (
                            model_signal[
                                "candidate"
                            ].base_features_manifest_json
                            or {}
                        ).get("feature_set_id")
                        or ""
                    ),
                    "contract_version": str(
                        (
                            model_signal[
                                "candidate"
                            ].base_features_manifest_json
                            or {}
                        ).get("contract_version")
                        or ""
                    ),
                    "features": dict(
                        (
                            model_signal[
                                "candidate"
                            ].base_features_manifest_json
                            or {}
                        ).get("feature_expressions")
                        or {}
                    ),
                    "definition_sha256": str(
                        model_signal[
                            "candidate"
                        ].feature_set_definition_sha256
                    ),
                },
                "code_path": runtime_path(model_signal["code_path"]),
                "training_periods": {
                    "train_start": model_signal[
                        "evaluation"
                    ].train_start.isoformat(),
                    "train_end": model_signal["evaluation"].train_end.isoformat(),
                    "valid_start": model_signal[
                        "evaluation"
                    ].valid_start.isoformat(),
                    "valid_end": model_signal["evaluation"].valid_end.isoformat(),
                    "seed": int(model_signal["evaluation"].seed),
                },
                "primary_profile_id": str(model_signal["evaluation"].profile_id),
                "refit_policy": version["config"].get("model_refit_policy"),
                "refit_policy_sha256": version["config"].get(
                    "model_refit_policy_sha256"
                ),
            }
            if model_signal is not None
            else None
        ),
        "model_bundle_factors": (
            [
                {
                    **{
                        key: item[key]
                        for key in (
                            "candidate_id",
                            "feature_name",
                            "code_sha256",
                            "direction",
                            "weight",
                            "factor_execution_mode",
                        )
                    },
                    "code_path": runtime_path(str(item["code_path"])),
                }
                for item in model_signal["bundle_factors"]
            ]
            if model_signal is not None
            else []
        ),
        "factors": [
            {
                "candidate_id": item["factor_candidate_id"],
                "values_path": runtime_path(item["values_path"]),
                "code_path": (
                    runtime_path(item["code_path"]) if item.get("code_path") else None
                ),
                "code_sha256": item["code_sha256"],
                "factor_execution_mode": (
                    "frozen_code_recompute"
                    if item.get("source_iteration") is not None
                    else "frozen_values"
                ),
                "weight": item["weight"],
                "direction": item["direction"],
            }
            for item in version["factors"]
        ],
        "trials": [
            {
                "trial_index": item["trial_index"],
                "parameters": item["parameters"],
                "config": item["config"],
            }
            for item in experiment["trials"]
        ],
    }
    if runner_sha256 is not None:
        manifest[TRANSPARENT_BASELINE_JOB_RUNNER_FIELD] = runner_sha256
    for identity_field in (
        TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD,
        TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD,
    ):
        if payload.get(identity_field) is not None:
            manifest[identity_field] = payload[identity_field]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    script = worker.project_root / "scripts" / "run_parameter_experiment.py"
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            _to_wsl_path(Path(payload["dataset_path"]))
            if is_wsl
            else str(Path(payload["dataset_path"])),
            "--manifest",
            _to_wsl_path(manifest_path) if is_wsl else str(manifest_path),
            "--output",
            _to_wsl_path(output) if is_wsl else str(output),
            "--tracking-uri",
            worker.settings.mlflow_tracking_uri,
        ]
    )
    execution_dataset = payload.get("execution_dataset")
    if execution_dataset:
        command.extend(
            [
                "--execution-provider-uri",
                runtime_path(str(execution_dataset["path"])),
                "--execution-frequency",
                str(execution_dataset["frequency"]),
            ]
        )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


def strategy_backtest_command(
    worker, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    payload = job["payload"]
    output = (
        worker.settings.data_root
        / "artifacts"
        / "backtests"
        / payload["backtest_id"]
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    version = worker.strategies.get_version(payload["strategy_version_id"])
    backtest = worker.strategies.get_backtest(str(payload["backtest_id"]))
    evidence_mode = str(backtest.get("evidence_mode") or "legacy_ambiguous")
    if (
        str(backtest.get("strategy_version_id") or "") != str(version["id"])
        or evidence_mode != str(version.get("evidence_mode") or "legacy_ambiguous")
    ):
        raise ValueError("strategy backtest evidence authority changed after creation")
    replay = evidence_mode == EVIDENCE_MODE_REPLAY
    if replay:
        require_replay_config(version["config"])
        if payload.get("capital_oos_batch_id") or payload.get(
            "fin_strategy_research_run_id"
        ):
            raise ValueError(
                "consumed historical replay cannot carry capital or fin_strategy settlement"
            )
    elif evidence_mode != EVIDENCE_MODE_SEALED:
        raise ValueError(
            "multifactor strategy backtest has ambiguous evidence authority"
        )
    is_wsl = os.name == "nt" and worker.settings.qlib_python.startswith("/")

    def runtime_path(value: str) -> str:
        return _to_wsl_path(Path(value)) if is_wsl else str(value)

    execution_dataset = payload.get("execution_dataset")
    hypothesis_evidence = worker.strategies.hypothesis_group_evidence(version["id"])
    incomplete_family_eligibility = None
    if replay and int(hypothesis_evidence["shared_experiment_count"]) > 1:
        with worker.strategies.engine.connect() as connection:
            incomplete_family_eligibility = (
                incomplete_family_eligibility_for_version(
                    connection,
                    strategy_version_id=str(version["id"]),
                    hypothesis_group_evidence=hypothesis_evidence,
                )
            )
    final_periods = {
        "start": payload["periods"]["start"],
        "end": payload["periods"]["end"],
    }
    historical_validation_periods = {
        "start": payload["periods"]["historical_start"],
        "end": payload["periods"]["historical_end"],
    }
    with worker.strategies.engine.connect() as connection:
        model_signal = worker.strategies._model_signal_evidence(connection, version["config"])
    model_label_binding = _frozen_model_label_contract(model_signal)
    manifest = {
        "backtest_id": payload["backtest_id"],
        "strategy_version_id": version["id"],
        "evaluation_mode": evidence_mode if replay else "formal_final_oos",
        "evidence_mode": evidence_mode,
        **(REPLAY_MARKERS if replay else {}),
        "final_oos_opened": True,
        "strategy_rules_sha256": version["strategy_rules_sha256"],
        "dataset": payload["dataset"],
        "execution_dataset": (
            execution_dataset.get("name") if isinstance(execution_dataset, dict) else None
        ),
        "execution_frequency": (
            execution_dataset.get("frequency")
            if isinstance(execution_dataset, dict)
            else None
        ),
        "execution_contract_version": (
            (execution_dataset.get("provenance") or {}).get("execution_contract_version")
            if isinstance(execution_dataset, dict)
            else None
        ),
        "benchmark": version["benchmark"],
        "universe": version["universe"],
        "factor_source_mode": version["config"].get("factor_source_mode"),
        "challenger_weight": version["config"].get("challenger_weight"),
        "baseline": (
            {
                "definition": version["config"].get("baseline_definition"),
                "definition_sha256": version["config"].get("baseline_definition_sha256"),
            }
            if version["config"].get("baseline_definition")
            else None
        ),
        "strategy_trial_count": hypothesis_evidence["shared_experiment_count"],
        "economic_hypothesis_group": hypothesis_evidence["economic_hypothesis_group"],
        "hypothesis_group_evidence": hypothesis_evidence,
        "incomplete_factor_family_eligibility": (
            incomplete_family_eligibility
        ),
        "periods": final_periods,
        "historical_validation_periods": historical_validation_periods,
        "config": version["config"],
        "model_signal": (model_signal["identity"] if model_signal is not None else None),
        "model_formal_admission": (
            model_signal["formal_admission_binding"] if model_signal is not None else None
        ),
        "model_candidate": (
            {
                "candidate_manifest": dict(model_signal["candidate"].manifest_json or {}),
                "feature_set": {
                    "id": str(
                        (model_signal["candidate"].base_features_manifest_json or {}).get(
                            "feature_set_id"
                        )
                        or ""
                    ),
                    "contract_version": str(
                        (model_signal["candidate"].base_features_manifest_json or {}).get(
                            "contract_version"
                        )
                        or ""
                    ),
                    "features": dict(
                        (model_signal["candidate"].base_features_manifest_json or {}).get(
                            "feature_expressions"
                        )
                        or {}
                    ),
                    "definition_sha256": str(
                        model_signal["candidate"].feature_set_definition_sha256
                    ),
                },
                "label_contract": (
                    model_label_binding[0]
                    if model_label_binding is not None
                    else None
                ),
                "label_contract_sha256": (
                    model_label_binding[1]
                    if model_label_binding is not None
                    else None
                ),
                "code_path": runtime_path(model_signal["code_path"]),
                "training_periods": {
                    "train_start": model_signal["evaluation"].train_start.isoformat(),
                    "train_end": model_signal["evaluation"].train_end.isoformat(),
                    "valid_start": model_signal["evaluation"].valid_start.isoformat(),
                    "valid_end": model_signal["evaluation"].valid_end.isoformat(),
                    "seed": int(model_signal["evaluation"].seed),
                },
                "primary_profile_id": str(
                    model_signal["evaluation"].profile_id
                ),
                "refit_policy": version["config"].get("model_refit_policy"),
                "refit_policy_sha256": version["config"].get(
                    "model_refit_policy_sha256"
                ),
            }
            if model_signal is not None
            else None
        ),
        "model_bundle_factors": (
            [
                {
                    **{
                        key: item[key]
                        for key in (
                            "candidate_id",
                            "feature_name",
                            "code_sha256",
                            "direction",
                            "weight",
                            "factor_execution_mode",
                        )
                    },
                    "code_path": runtime_path(str(item["code_path"])),
                }
                for item in model_signal["bundle_factors"]
            ]
            if model_signal is not None
            else []
        ),
        "min_daily_instruments": int(payload.get("min_daily_instruments", 50)),
        "factors": [
            {
                "candidate_id": item["factor_candidate_id"],
                "values_path": runtime_path(item["values_path"]),
                "code_path": (
                    runtime_path(item["code_path"]) if item.get("code_path") else None
                ),
                "code_sha256": item["code_sha256"],
                "factor_execution_mode": (
                    "frozen_code_recompute"
                    if item.get("source_iteration") is not None
                    else "frozen_values"
                ),
                "weight": item["weight"],
                "direction": item["direction"],
            }
            for item in version["factors"]
        ],
    }
    script = worker.project_root / "scripts" / "run_multifactor_backtest.py"
    runner_sha256 = require_transparent_baseline_runner(
        config=version["config"],
        job_payload=payload,
        runner_path=script,
    )
    if runner_sha256 is not None:
        manifest["transparent_baseline_runner_sha256"] = runner_sha256
    runtime_bundle_sha256 = dict(
        version["config"].get("transparent_baseline_bootstrap") or {}
    ).get(TRANSPARENT_BASELINE_RUNTIME_BUNDLE_FIELD)
    if runtime_bundle_sha256 is not None:
        manifest[TRANSPARENT_BASELINE_JOB_RUNTIME_BUNDLE_FIELD] = (
            runtime_bundle_sha256
        )
    worker_runtime_image_digest = dict(
        version["config"].get("transparent_baseline_bootstrap") or {}
    ).get(TRANSPARENT_BASELINE_WORKER_RUNTIME_IMAGE_FIELD)
    if worker_runtime_image_digest is not None:
        manifest[TRANSPARENT_BASELINE_JOB_WORKER_RUNTIME_IMAGE_FIELD] = (
            worker_runtime_image_digest
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    command = (
        [
            "wsl",
            "-d",
            worker.settings.qlib_wsl_distro,
            "--exec",
            worker.settings.qlib_python,
            _to_wsl_path(script),
        ]
        if is_wsl
        else [worker.settings.qlib_python, str(script)]
    )
    command.extend(
        [
            "--provider-uri",
            _to_wsl_path(Path(payload["dataset_path"]))
            if is_wsl
            else str(Path(payload["dataset_path"])),
            "--manifest",
            _to_wsl_path(manifest_path) if is_wsl else str(manifest_path),
            "--output",
            _to_wsl_path(output) if is_wsl else str(output),
            "--tracking-uri",
            worker.settings.mlflow_tracking_uri,
        ]
    )
    if isinstance(execution_dataset, dict):
        command.extend(
            [
                "--execution-provider-uri",
                runtime_path(execution_dataset["path"]),
                "--execution-frequency",
                str(execution_dataset["frequency"]),
            ]
        )
    return (
        command,
        result_path,
        _qlib_workflow_environment(worker.settings, is_wsl=is_wsl),
    )


COMMANDS = {
    "multiface_audit": multiface_audit_command,
    "minute_research": minute_research_command,
    "qlib_baseline": qlib_baseline_command,
    "model_ensemble_evaluate": model_ensemble_evaluate_command,
    "model_evaluate": model_evaluate_command,
    "quant_bundle_evaluate": model_evaluate_command,
    "factor_evaluate": factor_evaluate_command,
    "factor_library_materialize": factor_library_materialize_command,
    "factor_library_cluster": factor_library_cluster_command,
    "factor_sota_evaluate": factor_sota_evaluate_command,
    "external_factor_evaluate": external_factor_evaluate_command,
    "information_factor_evaluate": external_factor_evaluate_command,
    "model_refit": model_refit_command,
    "parameter_experiment": parameter_experiment_command,
    "strategy_backtest": strategy_backtest_command,
}
