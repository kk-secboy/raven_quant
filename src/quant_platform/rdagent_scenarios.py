from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant_data.config import Settings
from quant_data.kaggle_assets import validate_kaggle_research_asset_manifest
from quant_data.research_assets import validate_finetune_asset_contract

RDAGENT_GENERIC_JOB_KIND = "rdagent_run"
RDAGENT_LEGACY_FACTOR_JOB_KIND = "rdagent_factor"
RDAGENT_MODEL_JOB_KIND = "rdagent_model"
RDAGENT_QUANT_JOB_KIND = "rdagent_quant"
RDAGENT_REPORT_JOB_KIND = "rdagent_factor_report"
RDAGENT_DATA_SCIENCE_JOB_KIND = "rdagent_data_science"
RDAGENT_LLM_FINETUNE_JOB_KIND = "rdagent_llm_finetune"
RDAGENT_JOB_KINDS = frozenset(
    {
        RDAGENT_GENERIC_JOB_KIND,
        RDAGENT_LEGACY_FACTOR_JOB_KIND,
        RDAGENT_MODEL_JOB_KIND,
        RDAGENT_QUANT_JOB_KIND,
        RDAGENT_REPORT_JOB_KIND,
        RDAGENT_DATA_SCIENCE_JOB_KIND,
        RDAGENT_LLM_FINETUNE_JOB_KIND,
    }
)

# Weight-reduction phase 1: these scenarios stay registered so historical
# runs, governed assets, and the public catalog remain readable, but no new
# run may be scheduled, created through the API, or executed by a worker.
# Weight-reduction phase C3 physically deleted their execution-side code
# (runner branches, orchestration, worker command kinds); do not remove the
# registry entries — historical runs and assets stay readable through them.
FROZEN_RDAGENT_SCENARIOS = frozenset(
    {"fin_factor", "fin_model", "general_model", "data_science", "llm_finetune"}
)

_ASSET_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
_FEATURE_SET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA256_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RDAgentScenario:
    id: str
    label: str
    category: str
    command: str
    requires_dataset: bool
    asset_kind: str | None = None
    min_assets: int = 0
    max_assets: int = 0
    capital_eligible: bool = False
    gpu_required: bool = False
    factor_output: bool = False
    requires_feature_set: bool = False
    auto_select_assets: bool = False

    @property
    def requires_assets(self) -> bool:
        return self.min_assets > 0

    @property
    def research_kind(self) -> str:
        # Preserve the established database/API meaning for existing factor runs.
        return "factor" if self.id == "fin_factor" else self.id

    @property
    def job_kind(self) -> str:
        # Existing automations may still enqueue rdagent_factor. All other scenarios
        # share the generic, registry-dispatched durable worker path.
        return {
            "fin_factor": RDAGENT_LEGACY_FACTOR_JOB_KIND,
            "fin_model": RDAGENT_MODEL_JOB_KIND,
            "fin_quant": RDAGENT_QUANT_JOB_KIND,
            "fin_factor_report": RDAGENT_REPORT_JOB_KIND,
            "data_science": RDAGENT_DATA_SCIENCE_JOB_KIND,
            "llm_finetune": RDAGENT_LLM_FINETUNE_JOB_KIND,
        }.get(self.id, RDAGENT_GENERIC_JOB_KIND)


_SCENARIOS = (
    RDAgentScenario(
        "fin_factor",
        "Factor research",
        "quant",
        "fin_factor",
        True,
        capital_eligible=True,
        factor_output=True,
        requires_feature_set=True,
    ),
    RDAgentScenario(
        "fin_model",
        "Prediction-model research",
        "quant",
        "fin_model",
        True,
        capital_eligible=True,
        requires_feature_set=True,
    ),
    RDAgentScenario(
        "fin_quant",
        "Joint factor and model research",
        "quant",
        "fin_quant",
        True,
        capital_eligible=True,
        requires_feature_set=True,
    ),
    RDAgentScenario(
        "fin_strategy",
        "Strategy proposal research",
        "quant",
        "fin_strategy",
        True,
        capital_eligible=False,
        requires_feature_set=True,
    ),
    RDAgentScenario(
        "fin_factor_report",
        "Report factor extraction",
        "quant",
        "fin_factor_report",
        True,
        asset_kind="pdf",
        min_assets=1,
        max_assets=20,
        capital_eligible=True,
        factor_output=True,
        auto_select_assets=True,
    ),
    RDAgentScenario(
        "general_model",
        "Paper model implementation",
        "lab",
        "general_model",
        False,
        asset_kind="pdf",
        min_assets=1,
        max_assets=1,
        auto_select_assets=True,
    ),
    RDAgentScenario(
        "data_science",
        "General data science",
        "lab",
        "data_science",
        False,
        asset_kind="dataset",
        min_assets=1,
        max_assets=1,
    ),
    RDAgentScenario(
        "llm_finetune",
        "LLM fine-tuning",
        "lab",
        "llm_finetune",
        False,
        asset_kind="finetune",
        min_assets=1,
        max_assets=1,
        gpu_required=True,
    ),
)
SCENARIOS = {item.id: item for item in _SCENARIOS}
_DESCRIPTIONS = {
    "fin_factor": "Propose and iterate factors before independent platform evaluation.",
    "fin_model": "Research models on a locked feature set before independent evaluation.",
    "fin_quant": "Jointly research factors and models before independent ablation evaluation.",
    "fin_strategy": (
        "Generate governed research-only StrategyProposal JSON and compile it through the "
        "deterministic strategy-rule allowlist; no proposal is capital or simulation eligible."
    ),
    "fin_factor_report": "Extract factors from verified report PDFs for independent evaluation.",
    "general_model": "Implement a model from one verified paper PDF for research only.",
    "data_science": "Run a general data-science experiment on an explicit governed dataset.",
    "llm_finetune": "Research LLM fine-tuning with governed assets and an NVIDIA GPU.",
}


def get_rdagent_scenario(value: str | None) -> RDAgentScenario:
    scenario_id = str(value or "fin_factor").strip()
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        raise ValueError(f"unsupported RD-Agent scenario: {scenario_id}") from exc


def scenario_from_research_run(run: dict[str, Any]) -> str:
    config = run.get("config")
    if isinstance(config, dict) and config.get("scenario") in SCENARIOS:
        return str(config["scenario"])
    kind = str(run.get("kind") or "")
    return "fin_factor" if kind == "factor" else kind


def is_rdagent_job(kind: str) -> bool:
    return kind in RDAGENT_JOB_KINDS


def validate_asset_id(value: str) -> str:
    candidate = str(value).strip()
    if not _ASSET_ID.fullmatch(candidate):
        raise ValueError(
            "asset_id must be a lowercase governed slug"
        )
    return candidate


def validate_feature_set_id(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    if not _FEATURE_SET_ID.fullmatch(candidate):
        raise ValueError("feature_set_id has an invalid governed identifier")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("research asset file path must be relative")
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("research asset file escapes its immutable asset root") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("research asset entries must be regular files")
    return resolved


def _read_asset(data_root: Path, asset_id: str) -> dict[str, Any]:
    asset_id = validate_asset_id(asset_id)
    root = (data_root / "artifacts" / "research-assets").resolve()
    path = (root / asset_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("research asset escapes the configured asset root") from exc
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"research asset is unavailable: {asset_id}")

    manifest_path = path / "manifest.json"
    sidecar_path = path / "manifest.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise ValueError(f"research asset {asset_id} has no sealed manifest")
    expected_manifest_sha256 = sidecar_path.read_text(encoding="ascii").strip().lower()
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if (
        not _SHA256.fullmatch(expected_manifest_sha256)
        or expected_manifest_sha256 != actual_manifest_sha256
    ):
        raise ValueError(f"research asset {asset_id} manifest digest is invalid")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"research asset {asset_id} manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"research asset {asset_id} manifest must be an object")
    if str(manifest.get("asset_id") or "") != asset_id:
        raise ValueError(f"research asset {asset_id} manifest identity disagrees")
    kind = str(manifest.get("kind") or "")
    if kind not in {"pdf", "dataset", "finetune"}:
        raise ValueError(f"research asset {asset_id} kind is unsupported")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"research asset {asset_id} manifest has no files")

    files: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"research asset {asset_id} has an invalid file entry")
        relative = str(entry.get("path") or "")
        digest = str(entry.get("sha256") or "").lower()
        media_type = str(entry.get("media_type") or "application/octet-stream").lower()
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"research asset {asset_id} file digest is invalid")
        file_path = _resolve_inside(path, relative)
        if _sha256_file(file_path) != digest:
            raise ValueError(f"research asset {asset_id} file digest disagrees: {relative}")
        if kind == "pdf":
            if media_type != "application/pdf" or file_path.suffix.lower() != ".pdf":
                raise ValueError(f"research asset {asset_id} is not a declared PDF")
            if not file_path.read_bytes()[:5].startswith(b"%PDF-"):
                raise ValueError(f"research asset {asset_id} has invalid PDF content")
        files.append(
            {
                "path": str(file_path),
                "relative_path": relative,
                "sha256": digest,
                "media_type": media_type,
            }
        )
    metadata = manifest.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"research asset {asset_id} metadata must be an object")
    asset_type = str(manifest.get("type") or metadata.get("type") or "")
    if kind == "dataset" and asset_type in {"kaggle_dataset", "kaggle_competition"}:
        validate_kaggle_research_asset_manifest(manifest, path)
    available_at_raw = str(manifest.get("available_at") or "").strip()
    try:
        available_at = datetime.fromisoformat(available_at_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"research asset {asset_id} available_at is invalid") from exc
    if available_at.tzinfo is None or available_at.utcoffset() is None:
        raise ValueError(f"research asset {asset_id} available_at must include a timezone")
    return {
        "asset_id": asset_id,
        "kind": kind,
        "manifest_sha256": actual_manifest_sha256,
        "path": str(path),
        "files": files,
        "metadata": metadata,
        "asset_type": asset_type,
        "status": str(manifest.get("status") or metadata.get("status") or ""),
        "selected_at": manifest.get("selected_at", metadata.get("selected_at")),
        "priority": manifest.get("priority", metadata.get("priority", 0)),
        "available_at": available_at.isoformat(),
    }


def _available_on(asset: dict[str, Any]) -> date:
    available_at = datetime.fromisoformat(str(asset["available_at"]))
    return available_at.astimezone(ZoneInfo("Asia/Shanghai")).date()


def _auto_select_asset_ids(
    settings: Settings,
    scenario: RDAgentScenario,
    *,
    excluded_asset_ids: set[str] | frozenset[str] | None = None,
    pre_final_end: date | None = None,
    selection_limit: int | None = None,
) -> list[str]:
    """Choose only sealed, explicitly-ready assets from the fixed asset root."""

    if not scenario.auto_select_assets:
        return []
    excluded = excluded_asset_ids or frozenset()
    expected_type = (
        "research_report" if scenario.id == "fin_factor_report" else "arxiv_paper"
    )
    root = settings.data_root / "artifacts" / "research-assets"
    if not root.is_dir():
        return []
    eligible: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if (
            not path.is_dir()
            or not _ASSET_ID.fullmatch(path.name)
            or path.name in excluded
        ):
            continue
        try:
            asset = _read_asset(settings.data_root, path.name)
            priority = int(asset.get("priority") or 0)
        except (OSError, TypeError, ValueError):
            continue
        if (
            asset["kind"] == scenario.asset_kind
            and asset["asset_type"] == expected_type
            and asset["status"] == "ready"
            and not asset.get("selected_at")
            and (pre_final_end is None or _available_on(asset) <= pre_final_end)
        ):
            asset["priority"] = priority
            eligible.append(asset)
    eligible.sort(
        key=lambda item: (-int(item["priority"]), str(item["asset_id"]))
    )
    limit = scenario.max_assets
    if selection_limit is not None:
        if selection_limit < scenario.min_assets:
            raise ValueError(
                f"{scenario.id} selection limit is below its minimum asset count"
            )
        limit = min(limit, selection_limit)
    return [str(item["asset_id"]) for item in eligible[:limit]]


def _validate_scenario_metadata(
    scenario: RDAgentScenario, assets: list[dict[str, Any]]
) -> dict[str, str]:
    if scenario.id == "data_science":
        return {
            # The immutable asset directory itself is the competition folder.
            # Do not let a manifest redirect the runner to a sibling path.
            "competition": assets[0]["asset_id"]
        }
    if scenario.id == "llm_finetune":
        metadata = assets[0]["metadata"]
        files = {
            str(item["relative_path"]).replace("\\", "/"): Path(str(item["path"]))
            for item in assets[0]["files"]
        }
        contract = validate_finetune_asset_contract(
            metadata,
            file_paths=files,
            file_reader=lambda relative: files[relative].read_bytes(),
        )
        governed = {
            "benchmark": str(contract["benchmark"]),
            "benchmark_description": str(contract["benchmark_description"]),
            "dataset": str(contract["dataset"]),
            "base_model": str(contract["base_model"]),
            "model_license_accepted": "true",
            "dataset_license_accepted": "true",
        }
        # The runner only needs model/dataset fields. Benchmark governance is
        # nevertheless validated above and remains bound to the sealed manifest.
        for subject in ("model", "dataset"):
            for suffix in (
                "revision",
                "license",
                "license_terms_sha256",
                "license_accepted_by",
                "license_accepted_at",
            ):
                field_name = f"{subject}_{suffix}"
                governed[field_name] = str(contract[field_name])
        return governed
    return {}


def resolve_rdagent_assets(
    settings: Settings,
    scenario: RDAgentScenario,
    asset_ids: list[str] | tuple[str, ...] | None,
    *,
    expected_manifest_sha256: dict[str, str] | None = None,
    excluded_auto_asset_ids: set[str] | frozenset[str] | None = None,
    pre_final_end: date | None = None,
    selection_limit: int | None = None,
) -> dict[str, Any]:
    if scenario.id == "fin_factor_report" and pre_final_end is None:
        raise ValueError(
            "fin_factor_report assets must be bound to the research pre-final end date"
        )
    normalized = [validate_asset_id(value) for value in (asset_ids or [])]
    if not normalized and scenario.auto_select_assets:
        normalized = _auto_select_asset_ids(
            settings,
            scenario,
            excluded_asset_ids=excluded_auto_asset_ids,
            pre_final_end=pre_final_end,
            selection_limit=selection_limit,
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("RD-Agent asset_ids contain duplicates")
    if selection_limit is not None and len(normalized) > selection_limit:
        raise ValueError(
            f"{scenario.id} accepts at most {selection_limit} governed assets for this run"
        )
    if not scenario.min_assets <= len(normalized) <= scenario.max_assets:
        if scenario.min_assets == scenario.max_assets:
            expected = str(scenario.min_assets)
        else:
            expected = f"{scenario.min_assets} to {scenario.max_assets}"
        raise ValueError(f"{scenario.id} requires {expected} governed research assets")
    assets = [_read_asset(settings.data_root, asset_id) for asset_id in normalized]
    for asset in assets:
        if pre_final_end is not None and _available_on(asset) > pre_final_end:
            raise ValueError(
                f"research asset {asset['asset_id']} became available after the "
                f"pre-final cutoff {pre_final_end.isoformat()}"
            )
        if asset["kind"] != scenario.asset_kind:
            raise ValueError(
                f"{scenario.id} requires {scenario.asset_kind} assets, got {asset['kind']}"
            )
        if scenario.id == "fin_factor_report" and asset["asset_type"] != "research_report":
            raise ValueError("fin_factor_report accepts only governed research_report PDFs")
        if scenario.id == "general_model" and asset["asset_type"] not in {
            "arxiv_paper",
            "manual_paper",
        }:
            raise ValueError("general_model accepts only governed paper PDFs")
        if expected_manifest_sha256 is not None:
            expected = str(expected_manifest_sha256.get(asset["asset_id"]) or "").lower()
            if expected != asset["manifest_sha256"]:
                raise ValueError(
                    f"research asset changed after enqueue: {asset['asset_id']}"
                )
    if scenario.id == "general_model":
        pdfs = [item for asset in assets for item in asset["files"]]
        if len(pdfs) != 1:
            raise ValueError("general_model requires exactly one governed PDF file")
    return {
        "assets": assets,
        "manifest_sha256": {
            asset["asset_id"]: asset["manifest_sha256"] for asset in assets
        },
        "scenario_options": _validate_scenario_metadata(scenario, assets),
    }


def rdagent_scenario_catalog(
    runtime: dict[str, Any], settings: Settings
) -> list[dict[str, Any]]:
    common_blockers: list[str] = []
    if not settings.rdagent_enabled:
        common_blockers.append("RD-Agent execution is disabled")
    if runtime.get("status") != "ok":
        common_blockers.append(str(runtime.get("error") or "RD-Agent runtime is unavailable"))
    if not runtime.get("llm_credentials_configured"):
        common_blockers.append(f"configure secret {settings.rdagent_llm_key_env}")
    if not runtime.get("docker_available"):
        common_blockers.append("Docker is required by the RD-Agent CoSTEER sandbox")
    if not runtime.get("source_tree_sha256"):
        common_blockers.append("RD-Agent source tree digest is unavailable")
    if not runtime.get("runtime_image_digest"):
        common_blockers.append("RDAGENT_RUNTIME_IMAGE_DIGEST is not configured")
    if runtime.get("repository_dirty") is True:
        common_blockers.append("RD-Agent source checkout is dirty and not reproducible")
    if runtime.get("production_reproducible") is not True:
        common_blockers.append("RD-Agent runtime identity is not production reproducible")

    catalog: list[dict[str, Any]] = []
    for scenario in _SCENARIOS:
        blockers = list(dict.fromkeys(common_blockers))
        evaluation_kind = {
            "fin_factor": "factor_evaluate",
            "fin_model": "model_evaluate",
            "fin_quant": "quant_bundle_evaluate",
            "fin_factor_report": "factor_evaluate",
            "general_model": "model_evaluate",
        }.get(scenario.id)
        if evaluation_kind is not None:
            evaluation_worker = runtime.get("evaluation_worker")
            if not isinstance(evaluation_worker, dict) or not evaluation_worker.get(
                "ready"
            ):
                blockers.append("the independent evaluation worker is unavailable")
            elif evaluation_kind not in set(evaluation_worker.get("job_kinds") or []):
                blockers.append(
                    f"the evaluation worker does not consume {evaluation_kind}"
                )
            if scenario.id in {"fin_model", "fin_quant", "general_model"}:
                if not _SHA256_IMAGE.fullmatch(settings.model_sandbox_image):
                    blockers.append(
                        "MODEL_SANDBOX_IMAGE must be an immutable image digest reference"
                    )
                expected_config_hash = hashlib.sha256(
                    settings.model_sandbox_image.encode()
                ).hexdigest()
                if not isinstance(evaluation_worker, dict) or not evaluation_worker.get(
                    "model_sandbox_ready"
                ):
                    blockers.append("the independent model sandbox is unavailable")
                elif evaluation_worker.get(
                    "model_sandbox_config_sha256"
                ) != expected_config_hash:
                    blockers.append(
                        "the evaluation worker model sandbox configuration disagrees"
                    )
        if scenario.id in {
            "fin_factor",
            "fin_model",
            "fin_quant",
            "fin_factor_report",
            "general_model",
        }:
            if not _SHA256_IMAGE.fullmatch(settings.rdagent_qlib_sandbox_image):
                blockers.append(
                    "RDAGENT_QLIB_SANDBOX_IMAGE must be an immutable image digest reference"
                )
            if not runtime.get("qlib_sandbox_preloaded"):
                blockers.append("the pinned RD-Agent Qlib sandbox image is not preloaded")
            if not runtime.get("qlib_smoke_passed"):
                blockers.append("the real Qlib data smoke test has not passed")
        if scenario.id == "data_science":
            data_science_worker = runtime.get("data_science_worker")
            if (
                not isinstance(data_science_worker, dict)
                or data_science_worker.get("status") != "ok"
            ):
                blockers.append("the dedicated Data Science worker is unavailable")
            elif data_science_worker.get("runtime_identity_matches") is not True:
                blockers.append(
                    "the Data Science worker runtime identity disagrees with the main worker"
                )
            if isinstance(data_science_worker, dict) and not data_science_worker.get(
                "llm_credentials_configured"
            ):
                blockers.append("the Data Science worker has no LLM credentials")
            if isinstance(data_science_worker, dict) and not data_science_worker.get(
                "docker_available"
            ):
                blockers.append("the Data Science worker has no Docker sandbox")
            if not _SHA256_IMAGE.fullmatch(settings.rdagent_data_science_image):
                blockers.append(
                    "RDAGENT_DATA_SCIENCE_IMAGE must be an immutable image digest reference"
                )
            if not isinstance(data_science_worker, dict) or not data_science_worker.get(
                "data_science_sandbox_preloaded"
            ):
                blockers.append("the pinned Data Science sandbox image is not preloaded")
            if not isinstance(data_science_worker, dict) or not data_science_worker.get(
                "data_science_smoke_passed"
            ):
                blockers.append("the offline Data Science sandbox smoke test has not passed")
        if scenario.gpu_required:
            gpu_worker = runtime.get("gpu_worker")
            if not isinstance(gpu_worker, dict) or gpu_worker.get("status") != "ok":
                blockers.append("the dedicated GPU worker is unavailable")
            elif gpu_worker.get("runtime_identity_matches") is not True:
                blockers.append(
                    "the GPU worker runtime identity disagrees with the main worker"
                )
            if isinstance(gpu_worker, dict) and not gpu_worker.get(
                "llm_credentials_configured"
            ):
                blockers.append("the GPU worker has no LLM credentials")
            if isinstance(gpu_worker, dict) and not gpu_worker.get("docker_available"):
                blockers.append("the GPU worker has no Docker sandbox")
            if not runtime.get("gpu_available"):
                blockers.append("a compatible NVIDIA GPU is required")
            if not runtime.get("gpu_driver_version"):
                blockers.append("NVIDIA driver evidence is unavailable")
            if not runtime.get("cuda_version"):
                blockers.append("CUDA runtime evidence is unavailable")
            if int(runtime.get("gpu_memory_free_mb") or 0) < int(
                settings.rdagent_finetune_min_gpu_memory_mb
            ):
                blockers.append(
                    "insufficient free GPU memory: require at least "
                    f"{settings.rdagent_finetune_min_gpu_memory_mb} MiB"
                )
            if not runtime.get("docker_gpu_runtime_available"):
                blockers.append("Docker NVIDIA runtime or CDI evidence is unavailable")
            if float(runtime.get("data_root_free_gb") or 0.0) < float(
                settings.rdagent_finetune_min_disk_gb
            ):
                blockers.append(
                    "insufficient DATA_ROOT free space: require at least "
                    f"{settings.rdagent_finetune_min_disk_gb:g} GiB"
                )
            for field, configured in (
                ("RDAGENT_FINETUNE_IMAGE", settings.rdagent_finetune_image),
                (
                    "RDAGENT_FINETUNE_BENCHMARK_IMAGE",
                    settings.rdagent_finetune_benchmark_image,
                ),
                (
                    "RDAGENT_FINETUNE_GPU_PROBE_IMAGE",
                    settings.rdagent_finetune_gpu_probe_image,
                ),
            ):
                if not _SHA256_IMAGE.fullmatch(configured):
                    blockers.append(f"{field} must be an immutable image digest reference")
            if not runtime.get("docker_gpu_smoke_passed"):
                blockers.append("the pinned Docker GPU smoke test has not passed")
            if not runtime.get("finetune_images_preloaded"):
                blockers.append("pinned fine-tune and benchmark images are not preloaded")
        catalog.append(
            {
                "id": scenario.id,
                "label": scenario.label,
                "description": _DESCRIPTIONS[scenario.id],
                "category": scenario.category,
                "frozen": scenario.id in FROZEN_RDAGENT_SCENARIOS,
                "ready": not blockers,
                "blockers": blockers,
                "requires_dataset": scenario.requires_dataset,
                "requires_assets": scenario.requires_assets,
                "requires_feature_set": scenario.requires_feature_set,
                "asset_kind": scenario.asset_kind,
                "asset_count": {
                    "min": scenario.min_assets,
                    "max": scenario.max_assets,
                },
                "auto_select_assets": scenario.auto_select_assets,
                "capital_eligible": scenario.capital_eligible,
                "gpu_required": scenario.gpu_required,
                "limits": {
                    "max_loops": settings.rdagent_max_loops,
                    "max_duration": settings.rdagent_max_duration,
                    "max_assets": scenario.max_assets,
                    **(
                        {
                            "reports_per_loop": 1,
                            "max_reports_per_run": settings.rdagent_max_loops,
                        }
                        if scenario.id == "fin_factor_report"
                        else {}
                    ),
                },
            }
        )
    return catalog


def require_ready_scenario(
    runtime: dict[str, Any], settings: Settings, scenario_id: str
) -> RDAgentScenario:
    scenario = get_rdagent_scenario(scenario_id)
    catalog = {
        item["id"]: item for item in rdagent_scenario_catalog(runtime, settings)
    }
    if not catalog[scenario.id]["ready"]:
        raise ValueError("; ".join(catalog[scenario.id]["blockers"]))
    return scenario
