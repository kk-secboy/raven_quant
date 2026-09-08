from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .model_compute_policy import (
    MODEL_RESOURCE_POLICY_VERSION,
    fixed_model_cell_grid_policy,
    governed_cell_resource_allocation,
    model_thread_environment,
)
from .model_prepared_execution import (
    PreparedDataBuildError,
    prepared_runtime_identity,
    run_with_prepared_data,
)
from .model_research_governance import (
    canonical_sha256,
    file_sha256,
    is_sha256,
    resolve_model_label_contract,
)
from .research_execution_cadence import (
    validate_research_execution_cadence_contract,
)

MODEL_RECOMPUTE_EXECUTOR_VERSION = "model-recompute-docker-v12-progress-observed"
MODEL_MEMORY_AUDIT_CONTRACT_VERSION = "model-memory-audit-v1-cgroup-peak"
MODEL_DATA_CONTRACT_VERSION = "model-data-contract-v1-train-window-normalized"
HORIZON_MODEL_DATA_CONTRACT_VERSION = "model-data-contract-v2-horizon-label"
MODEL_TEMPLATE_FILENAME = "platform_model_templates.py"
MODEL_RESOURCE_STAGES = frozenset(
    {"screening", "full_validation", "production_refit", "inference"}
)
DEEP_SCREENING_EPOCH_CAP = 4
DEEP_FULL_EPOCH_CAP = 12
DEEP_EARLY_STOP_CAP = 3
QLIB_EVALUATION_CONCURRENCY_CAP = 3
TRANSFORMER_CONCURRENCY_CAP = 1
RESERVED_SERVICE_RESOURCE_FRACTION = 0.25
MODEL_SANDBOX_MEMORY_GB = 40
# Serial expression loading avoids multiplying the full-universe handler's
# in-flight frames across Qlib processes; dates, features and training stay fixed.
MODEL_QLIB_KERNELS = 1
MODEL_SANDBOX_MLFLOW_ALLOW_FILE_STORE = "true"
TOURNAMENT_SCREEN_SEED = 11
TOURNAMENT_FULL_SEEDS = (11, 29, 47)
GOVERNED_MODEL_ENGINES = frozenset(
    {
        "ridge_baseline",
        "lightgbm_baseline",
        "platform_gru",
        "platform_transformer",
        "rdagent_pytorch",
    }
)
MODEL_CHECKPOINT_FORMATS = {
    "ridge_baseline": "ridge_numeric_json",
    "lightgbm_baseline": "lightgbm_text",
    "platform_gru": "pytorch_state_dict",
    "platform_transformer": "pytorch_state_dict",
    "rdagent_pytorch": "pytorch_state_dict",
}
MODEL_CHECKPOINT_SUFFIXES = {
    "ridge_numeric_json": ".json",
    "lightgbm_text": ".txt",
    "pytorch_state_dict": ".pt",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_IMPORT_ROOTS = {"math", "numpy", "torch", "typing"}
_FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
    "__import__",
}
_FORBIDDEN_ATTRIBUTES = {
    "__builtins__",
    "__class__",
    "__dict__",
    "__getattribute__",
    "__globals__",
    "__import__",
    "__subclasses__",
    "popen",
    "run",
    "system",
}


class ModelResourceLimitError(RuntimeError):
    """The candidate exceeded a governed compute budget, not an investment gate."""


def _read_model_memory_audit(path: Path) -> list[dict[str, Any]]:
    """Read the append-only sandbox memory audit without trusting partial lines."""

    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            # A cgroup kill can interrupt the final append. Earlier flushed
            # stages remain useful evidence and the truncated tail is ignored.
            continue
        if (
            not isinstance(record, dict)
            or record.get("contract_version") != MODEL_MEMORY_AUDIT_CONTRACT_VERSION
            or not isinstance(record.get("stage"), str)
        ):
            continue
        records.append(record)
    return records


def _model_memory_peak_bytes(
    records: list[dict[str, Any]], *, prepared_binding: dict[str, Any] | None = None,
) -> int | None:
    peaks = [
        value
        for record in records
        for value in (
            record.get("cgroup_peak_bytes"),
            record.get("process_peak_rss_bytes"),
        )
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    ]
    producer = (prepared_binding or {}).get("producer") or {}
    if producer.get("summary_validated") is True:
        memory = (producer.get("summary") or {}).get("memory") or {}
        peak = memory.get("observed_memory_peak_bytes")
        if isinstance(peak, int) and not isinstance(peak, bool) and peak > 0:
            peaks.append(peak)
    return max(peaks, default=None)


def _memory_limit_failure_details(
    records: list[dict[str, Any]],
    *,
    governed_limit_bytes: int,
) -> str:
    """Describe an OOM without presenting a pre-kill sample as the true peak."""

    last_flushed_peak = _model_memory_peak_bytes(records)
    suffix = (
        f", last_flushed_peak_bytes={last_flushed_peak}"
        if last_flushed_peak is not None
        else ""
    )
    return (
        "cgroup_limit_hit=true, "
        f"governed_limit_bytes={governed_limit_bytes}{suffix}"
    )


def governed_checkpoint_format(model_engine: str) -> str:
    try:
        return MODEL_CHECKPOINT_FORMATS[model_engine]
    except KeyError as exc:
        raise ValueError("model engine has no governed checkpoint format") from exc


def governed_checkpoint_filename(model_engine: str) -> str:
    checkpoint_format = governed_checkpoint_format(model_engine)
    return f"checkpoint{MODEL_CHECKPOINT_SUFFIXES[checkpoint_format]}"


def verify_governed_checkpoint(
    path: Path,
    *,
    model_engine: str,
    checkpoint_format: str,
    expected_sha256: str,
) -> None:
    """Reject executable/general pickle checkpoints before sandbox loading.

    PyTorch payload structure is verified again with ``weights_only=True`` in
    the isolated runner.  Here we bind the approved format, extension and hash;
    Ridge JSON and LightGBM text also receive a cheap non-executable signature
    check on the host.
    """

    expected_format = governed_checkpoint_format(model_engine)
    if checkpoint_format != expected_format:
        raise ValueError("model checkpoint format does not match its governed engine")
    if not path.is_file() or not is_sha256(expected_sha256):
        raise ValueError("model checkpoint identity is incomplete")
    if path.suffix.lower() != MODEL_CHECKPOINT_SUFFIXES[checkpoint_format]:
        raise ValueError("model checkpoint extension is not governed")
    if file_sha256(path) != expected_sha256.lower():
        raise ValueError("model checkpoint failed immutable verification")
    if checkpoint_format == "ridge_numeric_json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Ridge checkpoint is not valid numeric JSON") from exc
        coefficients = payload.get("coefficients") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or set(payload) != {
                "coefficients",
                "feature_count",
                "format",
                "intercept",
                "model_engine",
                "model_spec_sha256",
            }
            or payload.get("format") != "ridge-numeric-v1"
            or payload.get("model_engine") != "ridge_baseline"
            or not isinstance(coefficients, list)
            or not 1 <= len(coefficients) <= 512
            or payload.get("feature_count") != len(coefficients)
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in coefficients
            )
            or not isinstance(payload.get("intercept"), (int, float))
            or isinstance(payload.get("intercept"), bool)
            or not math.isfinite(float(payload["intercept"]))
            or not is_sha256(payload.get("model_spec_sha256"))
        ):
            raise ValueError("Ridge checkpoint violates the governed numeric schema")
    elif checkpoint_format == "lightgbm_text":
        try:
            header = path.read_bytes()[:4096].decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError("LightGBM checkpoint is not native text") from exc
        if not header.startswith("tree\n") or "\nversion=" not in header:
            raise ValueError("LightGBM checkpoint is not a native text model")


def governed_model_resource_policy(
    *,
    model_type: str,
    model_engine: str = "rdagent_pytorch",
    requested_hyperparameters: dict[str, Any] | None,
    stage: str,
    requested_timeout_seconds: int,
    seed: int | None = None,
    data_contract_version: str = MODEL_DATA_CONTRACT_VERSION,
    evaluation_profile_id: str | None = None,
) -> dict[str, Any]:
    """Freeze a reproducible compute budget without changing date segments.

    Elapsed time is observational, never a model rejection or a termination
    deadline. Callers keep the registered universe, train/validation dates,
    epoch limits and PIT cutoffs unchanged. Actual memory/resource failures
    remain resource-blocked rather than scored as a bad model.
    """

    if stage not in MODEL_RESOURCE_STAGES:
        raise ValueError("unknown governed model resource stage")
    if model_type not in {"Tabular", "TimeSeries"}:
        raise ValueError("model resource policy supports Tabular or TimeSeries models")
    if model_engine not in GOVERNED_MODEL_ENGINES:
        raise ValueError("model resource policy does not recognize the model engine")
    if model_engine in {"ridge_baseline", "lightgbm_baseline"} and model_type != "Tabular":
        raise ValueError("Ridge and LightGBM belong to the governed tabular lane")
    if model_engine in {"platform_gru", "platform_transformer"} and model_type != "TimeSeries":
        raise ValueError("GRU and Transformer belong to the governed sequence lane")
    requested = dict(requested_hyperparameters or {})

    def bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
        value = requested.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"model hyperparameter {name} must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"model hyperparameter {name} must be an integer") from exc
        return min(max(parsed, minimum), maximum)

    def bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
        value = requested.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"model hyperparameter {name} must be numeric")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"model hyperparameter {name} must be numeric") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"model hyperparameter {name} must be finite")
        if not minimum <= parsed <= maximum:
            parsed = min(max(parsed, minimum), maximum)
        return parsed

    if isinstance(requested_timeout_seconds, bool):
        raise ValueError("model timeout must be an integer")
    # Historical manifests carry this field. Preserve it as request evidence;
    # it must not silently reintroduce a fixed wall-clock cutoff.
    requested_timeout_seconds = int(requested_timeout_seconds)
    deep_engine = model_engine in {
        "platform_gru",
        "platform_transformer",
        "rdagent_pytorch",
    }
    if deep_engine:
        epoch_cap = (
            DEEP_SCREENING_EPOCH_CAP
            if stage == "screening"
            else DEEP_FULL_EPOCH_CAP
        )
        effective: dict[str, Any] = {
            "n_epochs": bounded_int("n_epochs", epoch_cap, 1, epoch_cap),
            "early_stop": bounded_int(
                "early_stop", DEEP_EARLY_STOP_CAP, 1, DEEP_EARLY_STOP_CAP
            ),
            "batch_size": bounded_int("batch_size", 256, 32, 2048),
            "lr": bounded_float("lr", 2e-4, 1e-7, 1.0),
            "weight_decay": bounded_float("weight_decay", 1e-4, 0.0, 10.0),
        }
    elif model_engine == "ridge_baseline":
        effective = {
            "alpha": 1.0,
            "fit_intercept": False,
            "normalization": "RobustZScoreNorm(train_only,clip_outlier=True)+Fillna(0)",
        }
    else:
        effective = {
            "preset": "pinned-lightgbm-baseline-v1",
            "feature_sampling": 0.8,
            "row_sampling": 0.8,
            "random_sampling": True,
        }

    if stage == "screening":
        allowed_seeds = (TOURNAMENT_SCREEN_SEED,)
    elif model_engine == "ridge_baseline":
        # Ridge is deterministic.  Keep the formal three-cell grid as an
        # integrity repeat (all cells must agree); it is still counted as one
        # preregistered model hypothesis by the tournament ledger.
        allowed_seeds = TOURNAMENT_FULL_SEEDS
    else:
        allowed_seeds = TOURNAMENT_FULL_SEEDS
    if seed is not None:
        if isinstance(seed, bool) or int(seed) not in allowed_seeds:
            raise ValueError(
                f"model seed is outside the governed policy for {model_engine}/{stage}"
            )
        effective_seed = int(seed)
    else:
        effective_seed = None
    if model_engine == "ridge_baseline":
        seed_policy = "deterministic-integrity-repeats"
    elif model_engine == "lightgbm_baseline":
        seed_policy = "fixed-three-seeds-because-sampling-is-enabled"
    elif model_engine in {"platform_gru", "platform_transformer"}:
        seed_policy = "fixed-deep-model-seeds"
    else:
        seed_policy = "fixed-rdagent-model-seeds"
    allocation = governed_cell_resource_allocation({
        "model_engine": model_engine,
        "resource_stage": stage,
        "evaluation_profile_id": evaluation_profile_id,
    })
    return {
        "contract_version": MODEL_RESOURCE_POLICY_VERSION,
        "stage": stage,
        "model_type": model_type,
        "model_engine": model_engine,
        "requested_training_hyperparameters": requested,
        "effective_training_hyperparameters": effective,
        "seed_policy": {
            "name": seed_policy,
            "effective_seed": effective_seed,
            "allowed_seeds": list(allowed_seeds),
        },
        "data_contract_version": data_contract_version,
        "duration_policy": {
            "mode": "observe_only",
            "automatic_termination": False,
            "elapsed_warning_seconds": 1800,
            "progress_warning_seconds": 1800,
            "legacy_requested_timeout_seconds": requested_timeout_seconds,
        },
        "limits": {
            "timeout_seconds": None,
            "cpu_count": allocation["cpu_count"],
            "memory_gb": allocation["memory_gb"],
            "compute_threads": allocation["compute_threads"],
            "blas_threads": allocation["blas_threads"],
            "torch_interop_threads": allocation["torch_interop_threads"],
            "dataloader_workers": allocation["dataloader_workers"],
            "qlib_kernels": MODEL_QLIB_KERNELS,
            "network": "none",
            "date_segments_modified": False,
            "universe_modified": False,
            "cpu_only": True,
            "cpu_only_timeseries_cap": model_type == "TimeSeries",
            "sequence_length": 20 if model_type == "TimeSeries" else None,
            "qlib_evaluation_concurrency_cap": QLIB_EVALUATION_CONCURRENCY_CAP,
            "exclusive_concurrency": (
                TRANSFORMER_CONCURRENCY_CAP
                if model_engine == "platform_transformer"
                else None
            ),
            "reserved_service_resource_fraction": RESERVED_SERVICE_RESOURCE_FRACTION,
        },
    }


def validate_model_code(source: str) -> None:
    """Restrict generated code to a pure PyTorch model definition.

    The Docker boundary protects the host.  This validator separately protects
    the independent evaluator from candidate code that tries to replace the
    runner, read files, execute commands, or monkeypatch its own score.
    """

    tree = ast.parse(source)
    defined_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    exported_model: str | None = None
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if not isinstance(
            statement,
            (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign),
        ):
            raise ValueError("model code may only define a pure model module")
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == "model_cls":
                    if not isinstance(statement.value, ast.Name):
                        raise ValueError("model_cls must reference a class defined in model.py")
                    exported_model = statement.value.id
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "model_cls"
        ):
            if not isinstance(statement.value, ast.Name):
                raise ValueError("model_cls must reference a class defined in model.py")
            exported_model = statement.value.id
    if exported_model is None or exported_model not in defined_classes:
        raise ValueError("model code must export model_cls as a locally defined class")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if not roots.issubset(_ALLOWED_IMPORT_ROOTS):
                raise ValueError(
                    f"model code imports forbidden modules: {sorted(roots - _ALLOWED_IMPORT_ROOTS)}"
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"model code imports forbidden module: {root or '<relative>'}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                raise ValueError(f"model code calls forbidden builtin: {node.func.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES:
            raise ValueError(f"model code accesses forbidden capability: {node.attr}")


def execute_model_candidate(
    *,
    code_path: Path,
    provider_path: Path,
    manifest: dict[str, Any],
    workspace: Path,
    runner_path: Path,
    additional_factors_path: Path | None = None,
    allow_final_oos: bool = False,
    allow_inference: bool = False,
    allow_live_retrain: bool = False,
    source_checkpoint_path: Path | None = None,
    source_checkpoint_sha256: str | None = None,
    source_checkpoint_format: str | None = None,
    timeout_seconds: int = 7200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_id = str(manifest.get("candidate_id") or "")
    if not _SAFE_ID.fullmatch(candidate_id):
        raise ValueError("model candidate id is invalid")
    if not code_path.is_file() or not provider_path.is_dir() or not runner_path.is_file():
        raise ValueError("model recomputation inputs are unavailable")
    validate_model_code(code_path.read_text(encoding="utf-8"))
    expected_code_sha256 = str(manifest.get("code_sha256") or "").lower()
    if not is_sha256(expected_code_sha256) or file_sha256(code_path) != expected_code_sha256:
        raise ValueError("model candidate code hash is invalid")
    label_contract = resolve_model_label_contract(
        research_window_contract=manifest.get("research_window_contract"),
        research_window_contract_sha256=manifest.get(
            "research_window_contract_sha256"
        ),
        label_horizon_sessions=manifest.get("label_horizon_sessions"),
    )
    label_contract_sha256 = canonical_sha256(label_contract)
    execution_cadence = None
    if not allow_inference and not allow_live_retrain:
        if label_contract["legacy"] is True:
            raise ValueError("legacy model research has no governed decision cadence")
        execution_cadence = validate_research_execution_cadence_contract(
            manifest.get("research_execution_cadence") or {},
            expected_horizon_profile=str(label_contract["horizon_profile"]),
        )
    data_contract_version = (
        MODEL_DATA_CONTRACT_VERSION
        if label_contract["legacy"] is True
        else HORIZON_MODEL_DATA_CONTRACT_VERSION
    )
    if sum(bool(value) for value in (allow_final_oos, allow_inference, allow_live_retrain)) > 1:
        raise ValueError("model execution authorizations are mutually exclusive")
    model_engine = str(manifest.get("model_engine") or "rdagent_pytorch")
    if allow_inference:
        if source_checkpoint_path is None:
            raise ValueError("live inference requires an immutable fitted checkpoint")
        verify_governed_checkpoint(
            source_checkpoint_path,
            model_engine=model_engine,
            checkpoint_format=str(source_checkpoint_format or ""),
            expected_sha256=str(source_checkpoint_sha256 or ""),
        )
    elif any(
        value is not None
        for value in (
            source_checkpoint_path,
            source_checkpoint_sha256,
            source_checkpoint_format,
        )
    ):
        raise ValueError("training execution cannot accept a source checkpoint")
    if allow_final_oos:
        if manifest.get("final_oos_opened") is not True:
            raise ValueError("formal model recomputation requires an opened final OOS ledger")
        if manifest.get("inference_only") is True:
            raise ValueError("formal model recomputation cannot be marked inference-only")
    elif allow_inference:
        if manifest.get("final_oos_opened") is not False:
            raise ValueError("research or inference model execution cannot open final OOS")
        if manifest.get("inference_only") is not True:
            raise ValueError("model inference authorization does not match the manifest")
        if manifest.get("live_retrain") is True:
            raise ValueError("live inference cannot also retrain")
    elif allow_live_retrain:
        if (
            manifest.get("final_oos_opened") is not False
            or manifest.get("inference_only") is True
            or manifest.get("live_retrain") is not True
        ):
            raise ValueError("model live-retrain authorization does not match the manifest")
    elif (
        manifest.get("final_oos_opened") is not False
        or manifest.get("inference_only") is True
        or manifest.get("live_retrain") is True
    ):
        raise ValueError("research model execution authorization is invalid")
    image = str(os.environ.get("MODEL_SANDBOX_IMAGE") or "").strip()
    if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError(
            "model recomputation requires a digest-pinned MODEL_SANDBOX_IMAGE"
        )
    if not shutil.which("docker"):
        raise ValueError("model recomputation requires the Docker CLI")
    image_result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    image_id = image_result.stdout.strip()
    if image_result.returncode != 0 or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", image_id
    ):
        raise ValueError("model sandbox image identity is invalid")
    runner_sha256 = file_sha256(runner_path)
    executor_source_sha256 = file_sha256(Path(__file__).resolve())
    template_source = Path(__file__).resolve().with_name("model_templates.py")
    portfolio_calendar_source = Path(__file__).resolve().with_name(
        "qlib_portfolio_calendar.py"
    )
    research_strategy_source = Path(__file__).resolve().with_name(
        "qlib_research_strategy.py"
    )
    research_cadence_source = Path(__file__).resolve().with_name(
        "research_execution_cadence.py"
    )
    research_horizon_source = Path(__file__).resolve().with_name(
        "research_horizon.py"
    )
    workflow_adapter_source = Path(__file__).resolve().with_name("qlib_workflow.py")
    upstream_versions_source = Path(__file__).resolve().with_name("upstream_versions.py")
    model_data_handler_source = Path(__file__).resolve().with_name("model_data_handler.py")
    prepared_sources = {
        name: Path(__file__).resolve().with_name(name)
        for name in ("model_prepared_data.py", "model_data_request.py")
    }
    if not all(
        path.is_file()
        for path in (
            template_source,
            portfolio_calendar_source,
            research_strategy_source,
            research_cadence_source,
            research_horizon_source,
            workflow_adapter_source,
            upstream_versions_source,
            model_data_handler_source,
        )
    ):
        raise ValueError("governed model runtime dependencies are unavailable")
    template_sha256 = file_sha256(template_source)
    portfolio_calendar_sha256 = file_sha256(portfolio_calendar_source)
    research_strategy_sha256 = file_sha256(research_strategy_source)
    research_cadence_sha256 = file_sha256(research_cadence_source)
    research_horizon_sha256 = file_sha256(research_horizon_source)
    workflow_adapter_sha256 = file_sha256(workflow_adapter_source)
    upstream_versions_sha256 = file_sha256(upstream_versions_source)
    model_data_handler_sha256 = file_sha256(model_data_handler_source)
    resource_policy = governed_model_resource_policy(
        model_type=str(manifest.get("model_type") or ""),
        model_engine=model_engine,
        requested_hyperparameters=dict(manifest.get("training_hyperparameters") or {}),
        stage=str(manifest.get("resource_stage") or "full_validation"),
        requested_timeout_seconds=timeout_seconds,
        seed=manifest.get("seed"),
        data_contract_version=data_contract_version,
        evaluation_profile_id=manifest.get("evaluation_profile_id"),
    )
    cell_allocation = governed_cell_resource_allocation(manifest)
    grid_policy = fixed_model_cell_grid_policy()
    compute_policy_source = Path(__file__).resolve().with_name("model_compute_policy.py")
    effective_timeout_seconds = resource_policy["limits"]["timeout_seconds"]
    execution_environment = {
        "contract_version": "model-execution-environment-v1",
        "executor_version": MODEL_RECOMPUTE_EXECUTOR_VERSION,
        "resource_policy_version": MODEL_RESOURCE_POLICY_VERSION,
        "model_cell_grid_policy": grid_policy,
        "model_compute_policy_sha256": file_sha256(compute_policy_source),
        "executor_source_sha256": executor_source_sha256,
        "runner_sha256": runner_sha256,
        "model_template_sha256": template_sha256,
        "qlib_portfolio_calendar_sha256": portfolio_calendar_sha256,
        "qlib_research_strategy_sha256": research_strategy_sha256,
        "research_execution_cadence_source_sha256": research_cadence_sha256,
        "research_horizon_source_sha256": research_horizon_sha256,
        "qlib_workflow_adapter_sha256": workflow_adapter_sha256,
        "upstream_versions_sha256": upstream_versions_sha256,
        "model_data_handler_sha256": model_data_handler_sha256,
        "prepared_data_producer": prepared_runtime_identity(
            runner_path, image=image, image_id=image_id,
        ),
        "prepared_data_controller_sha256": file_sha256(
            Path(__file__).resolve().with_name("model_prepared_execution.py")
        ),
        "prepared_data_cache_sha256": file_sha256(
            Path(__file__).resolve().with_name("model_prepared_cache.py")
        ),
        "execution_monitor_sha256": file_sha256(
            Path(__file__).resolve().with_name("model_execution_monitor.py")
        ),
        "sandbox_image": image,
        "sandbox_image_id": image_id,
        "mlflow_allow_file_store": MODEL_SANDBOX_MLFLOW_ALLOW_FILE_STORE,
    }
    execution_environment_sha256 = canonical_sha256(execution_environment)
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copy2(code_path, workspace / "model.py")
    shutil.copy2(runner_path, workspace / "runner.py")
    shutil.copy2(template_source, workspace / MODEL_TEMPLATE_FILENAME)
    sandbox_package = workspace / "quant_platform"
    sandbox_package.mkdir()
    (sandbox_package / "__init__.py").touch()
    shutil.copy2(
        portfolio_calendar_source,
        sandbox_package / "qlib_portfolio_calendar.py",
    )
    shutil.copy2(
        research_strategy_source,
        sandbox_package / "qlib_research_strategy.py",
    )
    shutil.copy2(
        research_cadence_source,
        sandbox_package / "research_execution_cadence.py",
    )
    shutil.copy2(
        research_horizon_source,
        sandbox_package / "research_horizon.py",
    )
    shutil.copy2(workflow_adapter_source, sandbox_package / "qlib_workflow.py")
    shutil.copy2(upstream_versions_source, sandbox_package / "upstream_versions.py")
    shutil.copy2(model_data_handler_source, sandbox_package / "model_data_handler.py")
    shutil.copy2(compute_policy_source, sandbox_package / "model_compute_policy.py")
    for name, source in prepared_sources.items():
        shutil.copy2(source, sandbox_package / name)
    runtime_checkpoint: Path | None = None
    if allow_inference:
        checkpoint_format = str(source_checkpoint_format)
        runtime_checkpoint = workspace / (
            "input-checkpoint" + MODEL_CHECKPOINT_SUFFIXES[checkpoint_format]
        )
        shutil.copy2(Path(source_checkpoint_path), runtime_checkpoint)
        runtime_checkpoint_additions = {
            "checkpoint_path": f"/work/{runtime_checkpoint.name}",
            "checkpoint_sha256": str(source_checkpoint_sha256).lower(),
            "checkpoint_format": checkpoint_format,
        }
    else:
        runtime_checkpoint_additions = {}
    if additional_factors_path is not None:
        if not additional_factors_path.is_file():
            raise ValueError("additional factor values are unavailable")
        runtime_factors = workspace / "additional_factors.parquet"
        shutil.copy2(additional_factors_path, runtime_factors)
        runtime_manifest_additions = {
            "additional_factors_path": "/work/additional_factors.parquet",
            "additional_factors_sha256": file_sha256(runtime_factors),
        }
    else:
        runtime_manifest_additions = {}
    runtime_manifest = {
        **manifest,
        **runtime_manifest_additions,
        **runtime_checkpoint_additions,
        "contract_version": "model-sandbox-input-v1",
        "provider_uri": "/qlib",
        "resource_policy": resource_policy,
        "model_cell_grid_policy": grid_policy,
        "model_cell_allocation": cell_allocation,
        "model_label_contract": label_contract,
        "model_label_contract_sha256": label_contract_sha256,
        **(
            {"research_execution_cadence": execution_cadence}
            if execution_cadence is not None
            else {}
        ),
        "execution_environment": execution_environment,
        "execution_environment_sha256": execution_environment_sha256,
    }
    if execution_cadence is None:
        runtime_manifest.pop("research_execution_cadence", None)
    workspace.chmod(0o777)
    readonly_names = [
        "model.py",
        "runner.py",
        MODEL_TEMPLATE_FILENAME,
        "quant_platform/__init__.py",
        "quant_platform/qlib_portfolio_calendar.py",
        "quant_platform/qlib_research_strategy.py",
        "quant_platform/research_execution_cadence.py",
        "quant_platform/research_horizon.py",
        "quant_platform/qlib_workflow.py",
        "quant_platform/upstream_versions.py",
        "quant_platform/model_data_handler.py",
        "quant_platform/model_compute_policy.py",
        *[f"quant_platform/{name}" for name in prepared_sources],
    ]
    if runtime_checkpoint is not None:
        readonly_names.append(runtime_checkpoint.name)
    for name in readonly_names:
        (workspace / name).chmod(0o444)
    cidfile = workspace / "container.cid"
    command = [
        "docker",
        "run",
        "--rm",
        "--cidfile",
        str(cidfile),
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--memory",
        f"{cell_allocation['memory_gb']}g",
        "--memory-swap",
        f"{cell_allocation['memory_gb']}g",
        "--cpus",
        str(cell_allocation["cpu_count"]),
        "--user",
        "65534:65534",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONPATH=/work",
        "--env",
        f"MLFLOW_ALLOW_FILE_STORE={MODEL_SANDBOX_MLFLOW_ALLOW_FILE_STORE}",
        *[
            argument
            for name, value in model_thread_environment(cell_allocation).items()
            for argument in ("--env", f"{name}={value}")
        ],
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=2g",
        "--mount",
        f"type=bind,src={workspace.resolve()},dst=/work",
        "--mount",
        f"type=bind,src={provider_path.resolve()},dst=/qlib,readonly",
        "--workdir",
        "/work",
        image,
        "python",
        "-I",
        "runner.py",
    ]
    try:
        completed = run_with_prepared_data(
            command=command, workspace=workspace, provider=provider_path,
            runner_path=runner_path, manifest=runtime_manifest,
            timeout_seconds=effective_timeout_seconds,
        )
    except PreparedDataBuildError as exc:
        if exc.returncode in {137, 143, -9, -15} or "out of memory" in str(exc).lower():
            raise ModelResourceLimitError(str(exc)) from exc
        raise
    except (subprocess.TimeoutExpired, TimeoutError) as exc:
        if cidfile.is_file():
            container_id = cidfile.read_text(encoding="utf-8").strip()
            if container_id:
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
        raise RuntimeError(
            "model execution control operation timed out; "
            "no computation deadline is configured"
        ) from exc
    memory_audit_path = workspace / "output" / "memory_stages.jsonl"
    memory_audit_records = _read_model_memory_audit(memory_audit_path)
    observed_memory_peak_bytes = _model_memory_peak_bytes(
        memory_audit_records, prepared_binding=runtime_manifest.get("prepared_data"),
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "model execution failed").strip()
        if completed.returncode in {137, 143, -9, -15} or "out of memory" in message.lower():
            governed_limit_bytes = (
                int(resource_policy["limits"]["memory_gb"]) * 1024**3
            )
            memory_failure = _memory_limit_failure_details(
                memory_audit_records,
                governed_limit_bytes=governed_limit_bytes,
            )
            raise ModelResourceLimitError(
                "model execution exceeded the governed CPU/memory budget "
                f"(exit={completed.returncode}, stage={resource_policy['stage']}"
                f", {memory_failure})"
            )
        raise ValueError(f"independent model recomputation failed: {message[-4000:]}")
    result_path = workspace / "output" / "result.json"
    if not result_path.is_file():
        raise ValueError("independent model recomputation did not create result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "passed" or result.get("candidate_id") != candidate_id:
        raise ValueError("independent model recomputation result is invalid")
    if result.get("resource_policy") != resource_policy:
        raise ValueError("independent model recomputation changed its resource policy")
    if result.get("model_cell_allocation") != cell_allocation:
        raise ValueError("independent model recomputation changed its cell allocation")
    if result.get("prepared_data") != runtime_manifest.get("prepared_data"):
        raise ValueError("independent model recomputation changed its prepared data binding")
    result_memory_audit = result.get("memory_audit") or {}
    if (
        result_memory_audit.get("contract_version")
        != MODEL_MEMORY_AUDIT_CONTRACT_VERSION
        or result_memory_audit.get("path") != "/work/output/memory_stages.jsonl"
        or not memory_audit_path.is_file()
        or result_memory_audit.get("sha256") != file_sha256(memory_audit_path)
        or result_memory_audit.get("stages") != memory_audit_records
    ):
        raise ValueError("independent model memory audit failed immutable verification")
    data_contract = result.get("model_data_contract") or {}
    if (
        data_contract.get("contract_version") != data_contract_version
        or result.get("model_data_contract_sha256") != canonical_sha256(data_contract)
    ):
        raise ValueError("independent model recomputation data contract is invalid")
    if (
        result.get("model_label_contract") != label_contract
        or result.get("model_label_contract_sha256") != label_contract_sha256
    ):
        raise ValueError("independent model recomputation changed its label contract")
    if execution_cadence is not None:
        if (
            result.get("research_execution_cadence") != execution_cadence
            or result.get("research_execution_cadence_sha256")
            != execution_cadence["evidence_sha256"]
        ):
            raise ValueError("independent model recomputation changed its decision cadence")
    elif (
        result.get("research_execution_cadence") is not None
        or result.get("research_execution_cadence_sha256") is not None
    ):
        raise ValueError("non-portfolio model execution claimed a decision cadence")
    model_spec = result.get("model_spec") or {}
    if result.get("model_spec_sha256") != canonical_sha256(model_spec):
        raise ValueError("independent model recomputation model specification is invalid")
    if result.get("final_oos_opened") is not bool(allow_final_oos):
        raise ValueError("independent model recomputation reported an invalid final OOS state")
    if (result.get("inference_only") is True) is not allow_inference:
        raise ValueError("independent model recomputation reported an invalid inference state")
    if (result.get("live_retrain") is True) is not allow_live_retrain:
        raise ValueError("independent model recomputation reported an invalid live-retrain state")
    predictions_path = workspace / "output" / "predictions.parquet"
    result_checkpoint_format = str(result.get("checkpoint_format") or "")
    if result_checkpoint_format != governed_checkpoint_format(model_engine):
        raise ValueError("independent model output checkpoint format is invalid")
    checkpoint_path = (
        runtime_checkpoint
        if allow_inference
        else workspace / "output" / governed_checkpoint_filename(model_engine)
    )
    if (
        not predictions_path.is_file()
        or result.get("predictions_sha256") != file_sha256(predictions_path)
        or checkpoint_path is None
        or not checkpoint_path.is_file()
        or result.get("checkpoint_sha256") != file_sha256(checkpoint_path)
    ):
        raise ValueError("independent model output files failed immutable verification")
    verify_governed_checkpoint(
        checkpoint_path,
        model_engine=model_engine,
        checkpoint_format=result_checkpoint_format,
        expected_sha256=str(result.get("checkpoint_sha256") or ""),
    )
    if (result.get("checkpoint_reused") is True) is not allow_inference:
        raise ValueError("independent model output reported an invalid checkpoint mode")
    portfolio_report_path = workspace / "output" / "portfolio_report.parquet"
    if not allow_inference and not allow_live_retrain and (
        not portfolio_report_path.is_file()
        or result.get("portfolio_report_sha256") != file_sha256(portfolio_report_path)
    ):
        raise ValueError("independent model portfolio report failed immutable verification")
    evidence = {
        "executor_version": MODEL_RECOMPUTE_EXECUTOR_VERSION,
        "candidate_id": candidate_id,
        "code_sha256": expected_code_sha256,
        "input_manifest_sha256": canonical_sha256(runtime_manifest),
        "result_sha256": file_sha256(result_path),
        "predictions_sha256": file_sha256(predictions_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_format": result_checkpoint_format,
        "checkpoint_reused": bool(allow_inference),
        "portfolio_report_sha256": (
            file_sha256(portfolio_report_path)
            if not allow_inference and not allow_live_retrain
            else None
        ),
        "sandbox_mode": "docker-isolated",
        "sandbox_image": image,
        "sandbox_image_id": image_id,
        "executor_source_sha256": executor_source_sha256,
        "runner_sha256": runner_sha256,
        "model_template_sha256": template_sha256,
        "model_data_contract_sha256": result["model_data_contract_sha256"],
        "model_label_contract_sha256": label_contract_sha256,
        "research_execution_cadence_sha256": (
            execution_cadence["evidence_sha256"]
            if execution_cadence is not None
            else None
        ),
        "research_window_contract_sha256": label_contract[
            "research_window_contract_sha256"
        ],
        "model_spec_sha256": result["model_spec_sha256"],
        "memory_audit_contract_version": MODEL_MEMORY_AUDIT_CONTRACT_VERSION,
        "memory_audit_sha256": file_sha256(memory_audit_path),
        "observed_memory_peak_bytes": observed_memory_peak_bytes,
        "execution_environment": execution_environment,
        "execution_environment_sha256": execution_environment_sha256,
        "network_mode": "none",
        "root_filesystem_read_only": True,
        "capabilities_dropped": "ALL",
        "no_new_privileges": True,
        "timeout_seconds": effective_timeout_seconds,
        "resource_policy": resource_policy,
        "model_cell_allocation": cell_allocation,
        "prepared_data": runtime_manifest["prepared_data"],
        "final_oos_opened": bool(allow_final_oos),
        "inference_only": bool(allow_inference),
        "live_retrain": bool(allow_live_retrain),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    result["execution_evidence_sha256"] = evidence["evidence_sha256"]
    result["execution_environment_sha256"] = execution_environment_sha256
    return result, evidence
