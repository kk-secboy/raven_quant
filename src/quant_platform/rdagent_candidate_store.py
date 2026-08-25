from __future__ import annotations

import json
import math
import stat
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    candidate_asset_links,
    factor_candidates,
    factor_evaluations,
    model_candidates,
    model_evaluations,
    open_database,
    quant_bundle_candidates,
    quant_bundle_evaluations,
    research_assets,
    research_run_artifacts,
    research_runs,
    row_dict,
)
from quant_data.research_assets import load_research_asset_manifest

from .feature_set_registry import get_feature_set
from .model_research_governance import (
    MODEL_REFIT_POLICY,
    MODEL_REFIT_POLICY_SHA256,
    MODEL_RESEARCH_CONTRACT_VERSION,
    PRIMARY_MODEL_PROFILE,
    PRIMARY_MODEL_SEED,
    REQUIRED_MODEL_METRICS,
    REQUIRED_MODEL_SEEDS,
    REQUIRED_QUANT_ABLATIONS,
    REQUIRED_RESEARCH_PROFILES,
    canonical_sha256,
    file_sha256,
    is_sha256,
    validate_independent_model_evidence,
)
from .model_research_governance import (
    validate_quant_bundle_evidence as _validate_quant_bundle_evidence,
)

RESEARCH_ASSET_CONTRACT_VERSION = "research-asset-v1"
RUN_ARTIFACT_CONTRACT_VERSION = "research-run-artifact-v1"
MODEL_CANDIDATE_CONTRACT_VERSION = "model-candidate-v1"
QUANT_BUNDLE_CANDIDATE_CONTRACT_VERSION = "quant-bundle-candidate-v1"
ADMISSION_CONTRACT_VERSION = "rdagent-independent-admission-v1"

EvidenceRole = Literal["official_feedback", "independent_gate"]
CandidateKind = Literal["factor", "model", "quant_bundle"]


def _now() -> datetime:
    return datetime.now(UTC)


def _actor(value: str) -> str:
    actor = str(value or "").strip()
    if len(actor) < 2:
        raise ValueError("a responsible actor is required")
    return actor


def _sha(value: Any, label: str) -> str:
    normalized = str(value or "").lower()
    if not is_sha256(normalized):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return normalized


def _nonempty(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} is required")
    return normalized


def _path_evidence(path_value: str | Path) -> tuple[Path, str, int]:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ValueError(f"immutable evidence file does not exist: {path}")
    return path, file_sha256(path), path.stat().st_size


def _is_linkish(path: Path) -> bool:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"research asset path is unreadable: {path}") from exc
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _bundle_asset_evidence(manifest_path: Path) -> tuple[dict[str, Any], str, int]:
    """Rebuild exact evidence for a sealed multi-file dataset/finetune asset."""

    sealed_manifest = manifest_path.resolve(strict=True)
    manifest = load_research_asset_manifest(sealed_manifest)
    kind = str(manifest.get("kind") or "")
    if kind not in {"dataset", "finetune"}:
        raise ValueError("bundle evidence is only valid for dataset/finetune assets")
    root = sealed_manifest.parent.resolve(strict=True)
    sidecar = sealed_manifest.with_name("manifest.sha256")
    if _is_linkish(root) or _is_linkish(sealed_manifest) or _is_linkish(sidecar):
        raise ValueError("research asset bundle cannot contain links or junctions")
    declared: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    folded_paths: set[str] = set()
    total_bytes = 0
    for raw in manifest.get("files") or []:
        entry = dict(raw)
        relative = Path(str(entry.get("path") or ""))
        if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("research asset bundle contains an unsafe relative path")
        relative_posix = relative.as_posix()
        if relative_posix in declared_paths or relative_posix.casefold() in folded_paths:
            raise ValueError("research asset bundle paths collide")
        file_path = (root / relative).resolve(strict=True)
        try:
            file_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("research asset bundle file escapes its root") from exc
        if _is_linkish(file_path) or not stat.S_ISREG(
            file_path.stat(follow_symlinks=False).st_mode
        ):
            raise ValueError("research asset bundle entries must be regular files")
        actual_bytes = file_path.stat().st_size
        actual_sha256 = file_sha256(file_path)
        if (
            int(entry.get("bytes") or -1) != actual_bytes
            or str(entry.get("sha256") or "").lower() != actual_sha256
        ):
            raise ValueError("research asset bundle file changed after sealing")
        declared_paths.add(relative_posix)
        folded_paths.add(relative_posix.casefold())
        total_bytes += actual_bytes
        declared.append(
            {
                "path": relative_posix,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "media_type": str(entry.get("media_type") or "application/octet-stream"),
            }
        )
    actual_paths: set[str] = set()
    for item in root.rglob("*"):
        if _is_linkish(item):
            raise ValueError("research asset bundle cannot contain links or junctions")
        if item.is_dir():
            continue
        if not stat.S_ISREG(item.stat(follow_symlinks=False).st_mode):
            raise ValueError("research asset bundle contains a special file")
        actual_paths.add(item.relative_to(root).as_posix())
    expected_paths = {*declared_paths, "manifest.json", "manifest.sha256"}
    if actual_paths != expected_paths:
        raise ValueError("research asset bundle contains unsealed or missing files")
    inventory = {
        "contract_version": "research-asset-bundle-inventory-v1",
        "asset_id": str(manifest["asset_id"]),
        "kind": kind,
        "materialized_manifest_sha256": file_sha256(sealed_manifest),
        "files": sorted(declared, key=lambda item: item["path"]),
    }
    return inventory, canonical_sha256(inventory), total_bytes


def _periods(
    *,
    train_start: date,
    train_end: date,
    valid_start: date,
    valid_end: date,
    test_start: date,
    test_end: date,
) -> dict[str, str]:
    if not (train_start <= train_end < valid_start <= valid_end < test_start <= test_end):
        raise ValueError(
            "train, validation and pre-final test periods must be ordered and disjoint"
        )
    return {
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "valid_start": valid_start.isoformat(),
        "valid_end": valid_end.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
    }


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(metrics)
    missing = [name for name in REQUIRED_MODEL_METRICS if name not in normalized]
    if missing:
        raise ValueError(f"independent metrics are incomplete: {missing}")
    for name in REQUIRED_MODEL_METRICS:
        try:
            value = float(normalized[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"independent metric {name} is not numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"independent metric {name} is not finite")
    return normalized


def _verify_model_seed_artifacts(
    evidence: Mapping[str, Any], *, valid_start: date, valid_end: date
) -> None:
    predictions_sha256 = _sha(evidence.get("predictions_sha256"), "model predictions")
    checkpoint_sha256 = _sha(evidence.get("checkpoint_sha256"), "model checkpoint")
    report_sha256 = _sha(
        evidence.get("portfolio_report_sha256"), "model portfolio report"
    )
    _sha(evidence.get("execution_environment_sha256"), "model execution environment")
    for path_key, expected in (
        ("predictions_path", predictions_sha256),
        ("checkpoint_path", checkpoint_sha256),
        ("portfolio_report_path", report_sha256),
    ):
        path, observed, _ = _path_evidence(str(evidence.get(path_key) or ""))
        if observed != expected:
            raise ValueError(f"model evidence file hash changed: {path}")
    coverage = evidence.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("coverage_gate_passed") is not True:
        raise ValueError("model prediction coverage proof is missing or failed")
    if (
        str(coverage.get("artifact_sha256") or "").lower() != predictions_sha256
        or str(coverage.get("test_start") or "") != valid_start.isoformat()
        or str(coverage.get("test_end") or "") != valid_end.isoformat()
    ):
        raise ValueError("model prediction coverage proof does not match the validation window")


def _verify_quant_seed_artifacts(
    evidence: Mapping[str, Any], *, valid_start: date, valid_end: date
) -> None:
    predictions_sha256 = _sha(evidence.get("predictions_sha256"), "quant predictions")
    checkpoint_sha256 = _sha(evidence.get("checkpoint_sha256"), "quant checkpoint")
    report_sha256 = _sha(
        evidence.get("portfolio_report_sha256"), "quant portfolio report"
    )
    _sha(evidence.get("execution_evidence_sha256"), "quant execution evidence")
    _sha(evidence.get("execution_environment_sha256"), "quant execution environment")
    try:
        latest = date.fromisoformat(str(evidence.get("latest_prediction_date") or ""))
    except ValueError as exc:
        raise ValueError("quant evaluation has no latest prediction date") from exc
    if latest > valid_end:
        raise ValueError("quant evaluation predictions reach final OOS")
    coverage = evidence.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("coverage_gate_passed") is not True:
        raise ValueError("quant prediction coverage proof is missing or failed")
    if (
        str(coverage.get("artifact_sha256") or "").lower() != predictions_sha256
        or str(coverage.get("test_start") or "") != valid_start.isoformat()
        or str(coverage.get("test_end") or "") != valid_end.isoformat()
    ):
        raise ValueError("quant prediction coverage proof does not match validation window")
    path, observed, _ = _path_evidence(str(coverage.get("artifact_path") or ""))
    if observed != predictions_sha256:
        raise ValueError(f"quant prediction artifact changed after evaluation: {path}")
    checkpoint_path, observed_checkpoint, _ = _path_evidence(path.parent / "checkpoint.pt")
    if observed_checkpoint != checkpoint_sha256:
        raise ValueError(f"quant checkpoint changed after evaluation: {checkpoint_path}")
    report_path, observed_report, _ = _path_evidence(
        str(evidence.get("portfolio_report_path") or "")
    )
    if observed_report != report_sha256:
        raise ValueError(f"quant portfolio report changed after evaluation: {report_path}")


def _validate_factor_recompute_evidence(
    evidence: Mapping[str, Any], *, candidate_id: str, code_sha256: str
) -> None:
    if str(evidence.get("candidate_id") or "") != candidate_id:
        raise ValueError("factor recomputation evidence belongs to another candidate")
    if str(evidence.get("code_sha256") or "").lower() != code_sha256:
        raise ValueError("factor recomputation evidence uses another code artifact")
    expected = canonical_sha256(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    if evidence.get("evidence_sha256") != expected:
        raise ValueError("factor recomputation evidence SHA-256 is invalid")
    execution = evidence.get("execution")
    if (
        not isinstance(execution, Mapping)
        or str(execution.get("code_sha256") or "").lower() != code_sha256
        or not is_sha256(execution.get("input_sha256"))
        or not is_sha256(execution.get("output_sha256"))
    ):
        raise ValueError("factor independent execution evidence is incomplete")
    pit = evidence.get("pit_invariance")
    checks = pit.get("checks") if isinstance(pit, Mapping) else None
    if (
        not isinstance(pit, Mapping)
        or pit.get("contract_version") != "factor-pit-prefix-invariance-v1"
        or pit.get("status") != "passed"
        or not isinstance(checks, list)
        or len(checks) < 3
        or any(
            not isinstance(check, Mapping)
            or check.get("invariant") is not True
            or not is_sha256(check.get("input_sha256"))
            or not is_sha256(check.get("output_sha256"))
            for check in checks
        )
    ):
        raise ValueError("factor point-in-time invariance proof is incomplete")
    submitted = evidence.get("submitted_comparison")
    if (
        isinstance(submitted, Mapping)
        and submitted.get("available") is True
        and (
            submitted.get("exact_match") is not True
            or submitted.get("index_exact_match") is not True
        )
    ):
        raise ValueError("submitted factor values do not match independent recomputation")


def _validate_quant_factor_bundle_evidence(
    evidence: Mapping[str, Any], *, bundle_manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    frozen = {
        str(item["candidate_id"]): str(item["code_sha256"])
        for item in bundle_manifest.get("factors") or []
    }
    proofs = evidence.get("factor_recompute_evidence")
    if not isinstance(proofs, list) or len(proofs) != len(frozen):
        raise ValueError("quant factor recomputation evidence is incomplete")
    observed: set[str] = set()
    for proof in proofs:
        if not isinstance(proof, Mapping):
            raise ValueError("quant factor recomputation proof is malformed")
        factor_id = str(proof.get("candidate_id") or "")
        if factor_id in observed or factor_id not in frozen:
            raise ValueError("quant factor recomputation identity is invalid")
        observed.add(factor_id)
        _validate_factor_recompute_evidence(
            proof,
            candidate_id=factor_id,
            code_sha256=frozen[factor_id],
        )
    combined_sha256 = _sha(evidence.get("combined_factor_values_sha256"), "combined factor values")
    combined_path, observed_sha256, _ = _path_evidence(
        str(evidence.get("combined_factor_values_path") or "")
    )
    if observed_sha256 != combined_sha256:
        raise ValueError(f"combined factor values changed after evaluation: {combined_path}")
    return [dict(item) for item in proofs], combined_sha256


def validate_model_evaluation_evidence(
    evidence: Mapping[str, Any],
    *,
    candidate_id: str,
    dataset_identity_sha256: str,
    pre_final_end: str | date,
) -> dict[str, Any]:
    """Shared fail-closed validator for a complete 3-profile/3-seed model grid."""

    return validate_independent_model_evidence(
        evidence,
        candidate_id=candidate_id,
        dataset_identity_sha256=dataset_identity_sha256,
        pre_final_end=(
            pre_final_end.isoformat() if isinstance(pre_final_end, date) else str(pre_final_end)
        ),
    )


def validate_quant_bundle_evidence(
    evidence: Mapping[str, Any],
    *,
    dataset_identity_sha256: str,
) -> dict[str, Any]:
    """Shared validator for an immutable factor+model bundle and three ablations."""

    return _validate_quant_bundle_evidence(
        evidence,
        dataset_identity_sha256=dataset_identity_sha256,
    )


class RDAGentCandidateStore:
    """Research-only registry for RD-Agent assets, artifacts and candidates.

    RD-Agent's own decisions are retained as informational feedback. They never
    satisfy the independent evidence grid and nothing in this store is capital
    eligible. Final-OOS admission remains owned by the existing StrategyStore.
    """

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    @staticmethod
    def _one(connection: Any, table: Any, record_id: str, label: str) -> Any:
        row = connection.execute(select(table).where(table.c.id == record_id)).first()
        if row is None:
            raise KeyError(f"{label} {record_id!r} does not exist")
        return row

    @staticmethod
    def _verify_artifact_row(row: Any) -> None:
        if str(row.status) != "recorded":
            raise ValueError("research run artifact has been invalidated")
        path, digest, size = _path_evidence(str(row.storage_path))
        if digest != str(row.content_sha256) or size != int(row.size_bytes):
            raise ValueError(f"research run artifact changed after registration: {path}")
        expected = canonical_sha256(dict(row.manifest_json or {}))
        if expected != str(row.manifest_sha256):
            raise ValueError("research run artifact manifest changed after registration")
        if bool(row.capital_eligible):
            raise ValueError("research run artifact unexpectedly became capital eligible")

    def register_asset(
        self,
        *,
        asset_key: str,
        asset_type: str,
        media_type: str,
        storage_path: str | Path,
        license_metadata: Mapping[str, Any],
        actor: str,
        source_uri: str | None = None,
        publisher: str | None = None,
        published_at: datetime | None = None,
        retrieved_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        asset_id: str | None = None,
    ) -> dict[str, Any]:
        key = _nonempty(asset_key, "asset key")
        path, content_sha256, size_bytes = _path_evidence(storage_path)
        retrieved = retrieved_at or _now()
        if retrieved.tzinfo is None or retrieved.utcoffset() is None:
            raise ValueError("asset retrieval timestamp must include a timezone")
        if published_at is not None and (
            published_at.tzinfo is None or published_at.utcoffset() is None
        ):
            raise ValueError("asset publication timestamp must include a timezone")
        license_json = dict(license_metadata)
        if not license_json:
            raise ValueError("asset license/provenance metadata is required")
        asset_id = _nonempty(asset_id or uuid.uuid4().hex, "asset id")
        if len(asset_id) > 128 or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in asset_id
        ):
            raise ValueError("asset id is not a safe governed identifier")
        manifest = {
            "contract_version": RESEARCH_ASSET_CONTRACT_VERSION,
            "id": asset_id,
            "asset_key": key,
            "asset_type": _nonempty(asset_type, "asset type"),
            "media_type": _nonempty(media_type, "media type"),
            "source_uri": str(source_uri or "").strip() or None,
            "publisher": str(publisher or "").strip() or None,
            "published_at": published_at.isoformat() if published_at else None,
            "retrieved_at": retrieved.astimezone(UTC).isoformat(),
            "license": license_json,
            "content_sha256": content_sha256,
            "size_bytes": size_bytes,
            "metadata": dict(metadata or {}),
        }
        responsible_actor = _actor(actor)
        now = _now()
        try:
            with self.engine.begin() as connection:
                connection.execute(
                    insert(research_assets).values(
                        id=asset_id,
                        asset_key=key,
                        asset_type=manifest["asset_type"],
                        media_type=manifest["media_type"],
                        source_uri=manifest["source_uri"],
                        publisher=manifest["publisher"],
                        published_at=published_at.astimezone(UTC) if published_at else None,
                        retrieved_at=retrieved.astimezone(UTC),
                        license_json=license_json,
                        storage_path=str(path),
                        content_sha256=content_sha256,
                        size_bytes=size_bytes,
                        manifest_json=manifest,
                        manifest_sha256=canonical_sha256(manifest),
                        status="registered",
                        created_by=responsible_actor,
                        created_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"research asset {key!r} is already registered") from exc
        return self.get_asset(asset_id, verify=True)

    def import_manifest(
        self, manifest_path: Path, *, actor: str = "research-asset-importer"
    ) -> dict[str, Any]:
        """Import one already-verified fixed-path research asset idempotently."""

        sealed_manifest_path = Path(manifest_path).resolve()
        manifest = load_research_asset_manifest(sealed_manifest_path)
        source = dict(manifest.get("source") or {})
        kind = _nonempty(str(manifest.get("kind") or ""), "asset kind")
        files = list(manifest.get("files") or [])
        content = dict(manifest.get("content") or {})
        source_kind = _nonempty(str(source.get("kind") or ""), "asset source kind")
        source_id = _nonempty(str(source.get("source_id") or ""), "asset source id")
        asset_id = _nonempty(str(manifest.get("asset_id") or ""), "asset id")
        asset_key = f"{source_kind}:{source_id}"
        if kind == "pdf":
            storage_path = sealed_manifest_path.parent / str(content.get("path") or "content.pdf")
            expected_content_sha256 = str(content.get("sha256") or "")
            media_type = str(content.get("media_type") or "application/pdf")
            bundle_inventory = None
            bundle_size = None
        else:
            # Dataset and finetune inputs are multi-file bundles.  The sealed
            # manifest is their immutable root identity; every file hash was
            # already rechecked by load_research_asset_manifest above.
            bundle_inventory, expected_content_sha256, bundle_size = _bundle_asset_evidence(
                sealed_manifest_path
            )
            storage_path = sealed_manifest_path
            media_type = "application/vnd.quantlab.research-asset+json"
        with self.engine.connect() as connection:
            existing = connection.execute(
                select(research_assets).where(research_assets.c.asset_key == asset_key)
            ).first()
        if existing is not None:
            if str(existing.id) != asset_id or str(existing.content_sha256) != str(
                expected_content_sha256
            ):
                raise ValueError("research asset source conflicts with an immutable DB row")
            if str(existing.status) != "registered":
                raise ValueError(
                    f"research asset {asset_id} is {existing.status} and cannot be executed"
                )
            return self.get_asset(asset_id, verify=True)
        published_at = datetime.fromisoformat(str(manifest["published_at"]))
        acquired_at = datetime.fromisoformat(str(manifest["acquired_at"]))
        if bundle_inventory is not None:
            responsible_actor = _actor(actor)
            runtime_metadata = dict(manifest.get("metadata") or {})
            license_contract: dict[str, Any] = {
                "use_scope": "private_research_only",
                "redistribution_allowed": False,
                "source_terms_must_be_honored": True,
                "source_kind": source_kind,
            }
            if kind == "finetune":
                license_contract["model"] = {
                    field: runtime_metadata.get(f"model_{field}")
                    for field in (
                        "revision",
                        "license",
                        "license_terms_sha256",
                        "license_accepted_by",
                        "license_accepted_at",
                    )
                }
                license_contract["dataset"] = {
                    field: runtime_metadata.get(f"dataset_{field}")
                    for field in (
                        "revision",
                        "license",
                        "license_terms_sha256",
                        "license_accepted_by",
                        "license_accepted_at",
                    )
                }
            db_manifest = {
                "contract_version": RESEARCH_ASSET_CONTRACT_VERSION,
                "id": asset_id,
                "asset_key": asset_key,
                "asset_type": str(manifest.get("type") or "research_bundle"),
                "media_type": media_type,
                "source_uri": None,
                "publisher": source_kind,
                "published_at": published_at.isoformat(),
                "retrieved_at": acquired_at.astimezone(UTC).isoformat(),
                "license": license_contract,
                "content_sha256": expected_content_sha256,
                "size_bytes": int(bundle_size or 0),
                "metadata": {
                    "title": manifest.get("title"),
                    "asset_kind": kind,
                    "storage_contract": "research-asset-bundle-inventory-v1",
                    "bundle_inventory": bundle_inventory,
                    "bundle_inventory_sha256": expected_content_sha256,
                    "runtime_metadata": runtime_metadata,
                },
            }
            try:
                with self.engine.begin() as connection:
                    connection.execute(
                        insert(research_assets).values(
                            id=asset_id,
                            asset_key=asset_key,
                            asset_type=db_manifest["asset_type"],
                            media_type=media_type,
                            source_uri=None,
                            publisher=source_kind,
                            published_at=published_at.astimezone(UTC),
                            retrieved_at=acquired_at.astimezone(UTC),
                            license_json=db_manifest["license"],
                            storage_path=str(storage_path),
                            content_sha256=expected_content_sha256,
                            size_bytes=int(bundle_size or 0),
                            manifest_json=db_manifest,
                            manifest_sha256=canonical_sha256(db_manifest),
                            status="registered",
                            created_by=responsible_actor,
                            created_at=_now(),
                        )
                    )
            except IntegrityError as exc:
                raise ValueError(f"research asset {asset_key!r} is already registered") from exc
            return self.get_asset(asset_id, verify=True)
        return self.register_asset(
            asset_id=asset_id,
            asset_key=asset_key,
            asset_type=str(manifest.get("type") or "research_document"),
            media_type=media_type,
            storage_path=storage_path,
            source_uri=str(source.get("final_url") or source.get("original_url") or "") or None,
            publisher=source_kind,
            published_at=published_at,
            retrieved_at=acquired_at,
            license_metadata={
                "use_scope": "private_research_only",
                "redistribution_allowed": False,
                "source_terms_must_be_honored": True,
                "source_kind": source_kind,
            },
            metadata={
                "title": manifest.get("title"),
                "authors": manifest.get("authors") or [],
                "categories": manifest.get("categories") or [],
                "available_at": manifest.get("available_at"),
                "availability_rule": manifest.get("availability_rule"),
                "selection": manifest.get("selection") or {},
                "asset_kind": kind,
                "files": files,
                "materialized_manifest_sha256": sealed_manifest_path
                .with_name("manifest.sha256")
                .read_text(encoding="ascii")
                .strip(),
            },
            actor=actor,
        )

    def get_asset(self, asset_id: str, *, verify: bool = False) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = self._one(connection, research_assets, asset_id, "research asset")
        result = row_dict(row)
        if verify:
            if str(row.status) != "registered":
                raise ValueError(
                    f"research asset {asset_id} is {row.status} and cannot be executed"
                )
            if canonical_sha256(dict(row.manifest_json or {})) != str(row.manifest_sha256):
                raise ValueError("research asset manifest changed after registration")
            metadata = dict((row.manifest_json or {}).get("metadata") or {})
            if metadata.get("storage_contract") == "research-asset-bundle-inventory-v1":
                inventory, digest, size = _bundle_asset_evidence(Path(str(row.storage_path)))
                if (
                    inventory != metadata.get("bundle_inventory")
                    or digest != metadata.get("bundle_inventory_sha256")
                    or digest != str(row.content_sha256)
                    or size != int(row.size_bytes)
                ):
                    raise ValueError("research asset bundle changed after registration")
            else:
                path, digest, size = _path_evidence(str(row.storage_path))
                if digest != str(row.content_sha256) or size != int(row.size_bytes):
                    raise ValueError(f"research asset changed after registration: {path}")
        return result

    def quarantine_asset(self, asset_id: str, *, reason: str, actor: str) -> dict[str, Any]:
        why = _nonempty(reason, "quarantine reason")
        with self.engine.begin() as connection:
            row = self._one(connection, research_assets, asset_id, "research asset")
            if str(row.status) != "registered":
                raise ValueError("only a registered research asset may be quarantined")
            connection.execute(
                update(research_assets)
                .where(research_assets.c.id == asset_id)
                .values(
                    status="quarantined",
                    quarantined_at=_now(),
                    quarantine_reason=f"{why} (actor={_actor(actor)})",
                )
            )
        return self.get_asset(asset_id)

    def register_run_artifact(
        self,
        *,
        research_run_id: str,
        artifact_type: str,
        storage_path: str | Path,
        producer: str,
        actor: str,
        contract_version: str,
        source_iteration: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path, content_sha256, size_bytes = _path_evidence(storage_path)
        artifact_id = uuid.uuid4().hex
        now = _now()
        with self.engine.begin() as connection:
            self._one(connection, research_runs, research_run_id, "research run")
            manifest = {
                "contract_version": RUN_ARTIFACT_CONTRACT_VERSION,
                "artifact_contract_version": _nonempty(contract_version, "artifact contract"),
                "id": artifact_id,
                "research_run_id": research_run_id,
                "artifact_type": _nonempty(artifact_type, "artifact type"),
                "producer": _nonempty(producer, "artifact producer"),
                "source_iteration": source_iteration,
                "content_sha256": content_sha256,
                "size_bytes": size_bytes,
                "metadata": dict(metadata or {}),
            }
            try:
                connection.execute(
                    insert(research_run_artifacts).values(
                        id=artifact_id,
                        research_run_id=research_run_id,
                        artifact_type=manifest["artifact_type"],
                        contract_version=manifest["artifact_contract_version"],
                        status="recorded",
                        storage_path=str(path),
                        content_sha256=content_sha256,
                        size_bytes=size_bytes,
                        manifest_json=manifest,
                        manifest_sha256=canonical_sha256(manifest),
                        producer=manifest["producer"],
                        source_iteration=source_iteration,
                        capital_eligible=False,
                        created_by=_actor(actor),
                        created_at=now,
                    )
                )
            except IntegrityError as exc:
                raise ValueError("this immutable run artifact is already registered") from exc
        return self.get_run_artifact(artifact_id, verify=True)

    def get_run_artifact(self, artifact_id: str, *, verify: bool = False) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = self._one(connection, research_run_artifacts, artifact_id, "run artifact")
        if verify:
            self._verify_artifact_row(row)
        return row_dict(row)

    def invalidate_run_artifact(
        self, artifact_id: str, *, reason: str, actor: str
    ) -> dict[str, Any]:
        with self.engine.begin() as connection:
            row = self._one(connection, research_run_artifacts, artifact_id, "run artifact")
            if str(row.status) != "recorded":
                raise ValueError("run artifact is already invalidated")
            connection.execute(
                update(research_run_artifacts)
                .where(research_run_artifacts.c.id == artifact_id)
                .values(
                    status="invalidated",
                    invalidated_by=_actor(actor),
                    invalidated_at=_now(),
                    invalidation_reason=_nonempty(reason, "invalidation reason"),
                )
            )
        return self.get_run_artifact(artifact_id)

    def create_model_candidate(
        self,
        *,
        research_run_id: str,
        name: str,
        description: str,
        model_type: str,
        code_artifact_id: str,
        architecture: Mapping[str, Any],
        model_hyperparameters: Mapping[str, Any],
        training_hyperparameters: Mapping[str, Any],
        feature_set_id: str,
        dataset: str,
        dataset_identity_sha256: str,
        dataset_lineage_id: str | None = None,
        pre_final_end: date,
        final_oos_start: date,
        final_oos_end: date,
        source_iteration: int | None = None,
        rdagent_decision: bool | None = None,
        rdagent_feedback: str | None = None,
    ) -> dict[str, Any]:
        candidate_id = uuid.uuid4().hex
        dataset_identity = _sha(dataset_identity_sha256, "dataset identity")
        dataset_lineage = (
            _sha(dataset_lineage_id, "dataset lineage") if dataset_lineage_id is not None else None
        )
        features = get_feature_set(_nonempty(feature_set_id, "feature set id"))
        feature_sha = _sha(features["definition_sha256"], "feature-set definition")
        if not pre_final_end < final_oos_start <= final_oos_end:
            raise ValueError("model candidate final OOS boundary is invalid")
        now = _now()
        with self.engine.begin() as connection:
            self._one(connection, research_runs, research_run_id, "research run")
            artifact = self._one(
                connection, research_run_artifacts, code_artifact_id, "model code artifact"
            )
            self._verify_artifact_row(artifact)
            if str(artifact.research_run_id) != research_run_id:
                raise ValueError("model code artifact belongs to another research run")
            code_sha256 = _sha(artifact.content_sha256, "model code artifact")
            base_features = {
                "contract_version": features["contract_version"],
                "feature_set_id": features["id"],
                "feature_names": sorted(features["features"]),
                "feature_expressions": dict(features["features"]),
                "definition_sha256": feature_sha,
            }
            base_sha = canonical_sha256(base_features)
            recipe = {
                "model_type": _nonempty(model_type, "model type"),
                "architecture": dict(architecture),
                "model_hyperparameters": dict(model_hyperparameters),
                "training_hyperparameters": dict(training_hyperparameters),
            }
            manifest = {
                "contract_version": MODEL_CANDIDATE_CONTRACT_VERSION,
                "id": candidate_id,
                "research_run_id": research_run_id,
                "name": _nonempty(name, "candidate name"),
                "description": _nonempty(description, "candidate description"),
                "source_iteration": source_iteration,
                "code_artifact_id": code_artifact_id,
                "code_sha256": code_sha256,
                "recipe": recipe,
                "recipe_sha256": canonical_sha256(recipe),
                "base_features_manifest_sha256": base_sha,
                "feature_set_definition_sha256": feature_sha,
                "dataset": _nonempty(dataset, "dataset"),
                "dataset_identity_sha256": dataset_identity,
                "dataset_lineage_id": dataset_lineage,
                "pre_final_end": pre_final_end.isoformat(),
                "final_oos_start": final_oos_start.isoformat(),
                "final_oos_end": final_oos_end.isoformat(),
            }
            try:
                connection.execute(
                    insert(model_candidates).values(
                        id=candidate_id,
                        research_run_id=research_run_id,
                        name=manifest["name"],
                        description=manifest["description"],
                        status="awaiting_independent_evaluation",
                        source_iteration=source_iteration,
                        model_type=recipe["model_type"],
                        code_artifact_id=code_artifact_id,
                        code_sha256=code_sha256,
                        architecture_json=recipe["architecture"],
                        model_hyperparameters_json=recipe["model_hyperparameters"],
                        training_hyperparameters_json=recipe["training_hyperparameters"],
                        base_features_manifest_json=base_features,
                        base_features_manifest_sha256=base_sha,
                        feature_set_definition_sha256=feature_sha,
                        dataset=manifest["dataset"],
                        dataset_identity_sha256=dataset_identity,
                        pre_final_end=pre_final_end,
                        final_oos_start=final_oos_start,
                        final_oos_end=final_oos_end,
                        manifest_json=manifest,
                        manifest_sha256=canonical_sha256(manifest),
                        rdagent_decision=rdagent_decision,
                        rdagent_feedback=rdagent_feedback,
                        capital_eligible=False,
                        created_at=now,
                        updated_at=now,
                    )
                )
            except IntegrityError as exc:
                raise ValueError(f"model candidate {name!r} already exists in this run") from exc
        return self.get_model_candidate(candidate_id, verify=True)

    def get_model_candidate(self, candidate_id: str, *, verify: bool = False) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = self._one(connection, model_candidates, candidate_id, "model candidate")
            if verify:
                artifact = self._one(
                    connection,
                    research_run_artifacts,
                    str(row.code_artifact_id),
                    "model code artifact",
                )
                self._verify_artifact_row(artifact)
                if str(artifact.content_sha256) != str(row.code_sha256):
                    raise ValueError("model candidate no longer matches its code artifact")
                if canonical_sha256(dict(row.manifest_json or {})) != str(row.manifest_sha256):
                    raise ValueError("model candidate immutable manifest is invalid")
                if canonical_sha256(dict(row.base_features_manifest_json or {})) != str(
                    row.base_features_manifest_sha256
                ):
                    raise ValueError("model candidate base-feature manifest is invalid")
                feature_set = get_feature_set(
                    str((row.base_features_manifest_json or {}).get("feature_set_id") or "")
                )
                if feature_set["definition_sha256"] != str(row.feature_set_definition_sha256):
                    raise ValueError("model candidate feature-set definition drifted")
                manifest = dict(row.manifest_json or {})
                if (
                    manifest.get("code_sha256") != str(row.code_sha256)
                    or manifest.get("base_features_manifest_sha256")
                    != str(row.base_features_manifest_sha256)
                    or manifest.get("feature_set_definition_sha256")
                    != str(row.feature_set_definition_sha256)
                    or manifest.get("dataset_identity_sha256") != str(row.dataset_identity_sha256)
                    or manifest.get("pre_final_end") != row.pre_final_end.isoformat()
                    or manifest.get("final_oos_start") != row.final_oos_start.isoformat()
                    or manifest.get("final_oos_end") != row.final_oos_end.isoformat()
                ):
                    raise ValueError("model candidate columns do not match its frozen manifest")
                if str(row.status) == "research_admitted":
                    admission = dict(row.admission_evidence_json or {})
                    if admission.get("evidence_sha256") != str(row.admission_evidence_sha256):
                        raise ValueError("model admission evidence hash is invalid")
                    validate_model_evaluation_evidence(
                        admission,
                        candidate_id=candidate_id,
                        dataset_identity_sha256=str(row.dataset_identity_sha256),
                        pre_final_end=row.pre_final_end,
                    )
                    multiple = admission.get("multiple_testing")
                    if not isinstance(multiple, Mapping):
                        raise ValueError("model run-level multiple-testing evidence is missing")
                    returns_path, returns_sha256, _ = _path_evidence(
                        str(multiple.get("returns_path") or "")
                    )
                    if returns_sha256 != str(multiple.get("returns_sha256") or ""):
                        raise ValueError(
                            "model run-level return matrix changed: "
                            f"{returns_path}"
                        )
                if bool(row.capital_eligible):
                    raise ValueError("model candidate unexpectedly became capital eligible")
        return row_dict(row)

    def record_model_evaluation(
        self,
        *,
        model_candidate_id: str,
        run_artifact_id: str,
        evidence_role: EvidenceRole,
        profile_id: str,
        seed: int,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        test_start: date,
        test_end: date,
        metrics: Mapping[str, Any],
        gate_status: str,
        gate_reasons: Sequence[str],
        evaluator_version: str,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        role = str(evidence_role)
        if role not in {"official_feedback", "independent_gate"}:
            raise ValueError("unknown model evidence role")
        periods = _periods(
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
        )
        evidence_id = uuid.uuid4().hex
        now = _now()
        with self.engine.begin() as connection:
            candidate = self._one(
                connection, model_candidates, model_candidate_id, "model candidate"
            )
            artifact = self._one(
                connection, research_run_artifacts, run_artifact_id, "model evaluation artifact"
            )
            self._verify_artifact_row(artifact)
            if str(artifact.research_run_id) != str(candidate.research_run_id):
                raise ValueError("model evaluation artifact belongs to another run")
            if str(candidate.status) in {"research_admitted", "rejected", "invalidated"}:
                raise ValueError("model candidate evaluation grid is already closed")
            if role == "official_feedback":
                if gate_status != "informational":
                    raise ValueError("RD-Agent internal feedback must remain informational")
                normalized_metrics = dict(metrics)
            else:
                if profile_id not in REQUIRED_RESEARCH_PROFILES:
                    raise ValueError("independent model evidence uses an unknown research profile")
                if int(seed) not in REQUIRED_MODEL_SEEDS:
                    raise ValueError("independent model evidence uses an unregistered seed")
                if gate_status not in {"passed", "failed"}:
                    raise ValueError("independent model gate must explicitly pass or fail")
                if valid_end > candidate.pre_final_end:
                    raise ValueError("research-stage model evidence reaches the sealed final OOS")
                if test_start != candidate.final_oos_start or test_end != candidate.final_oos_end:
                    raise ValueError("model evidence uses another final OOS boundary")
                if str(evidence.get("source") or "") != "independent_qlib_recompute":
                    raise ValueError("RD-Agent internal scores cannot be recorded as independent")
                if evidence.get("final_oos_opened") is not False:
                    raise ValueError("independent research evidence must not open final OOS")
                if not is_sha256(evidence.get("predictions_sha256")):
                    raise ValueError("model evaluation requires immutable predictions")
                if not is_sha256(evidence.get("execution_evidence_sha256")):
                    raise ValueError("model evaluation requires immutable execution evidence")
                if not is_sha256(evidence.get("execution_environment_sha256")):
                    raise ValueError("model evaluation requires an immutable environment")
                try:
                    latest_prediction_date = date.fromisoformat(
                        str(evidence.get("latest_prediction_date") or "")
                    )
                except ValueError as exc:
                    raise ValueError("model evaluation has no latest prediction date") from exc
                if latest_prediction_date > valid_end:
                    raise ValueError("model evaluation predictions reach final OOS")
                _verify_model_seed_artifacts(
                    evidence,
                    valid_start=valid_start,
                    valid_end=valid_end,
                )
                normalized_metrics = _finite_metrics(metrics)
            evidence_json = {
                **dict(evidence),
                "contract_version": ADMISSION_CONTRACT_VERSION,
                "evidence_role": role,
                "candidate_id": model_candidate_id,
                "candidate_manifest_sha256": str(candidate.manifest_sha256),
                "run_artifact_id": run_artifact_id,
                "run_artifact_sha256": str(artifact.content_sha256),
                "dataset": str(candidate.dataset),
                "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                "feature_set_definition_sha256": str(candidate.feature_set_definition_sha256),
                "profile_id": profile_id,
                "seed": int(seed),
                "periods": periods,
                "gate_status": gate_status,
                "gate_reasons": [str(item) for item in gate_reasons],
                "metrics_sha256": canonical_sha256(normalized_metrics),
                "evaluator_version": _nonempty(evaluator_version, "evaluator version"),
            }
            try:
                connection.execute(
                    insert(model_evaluations).values(
                        id=evidence_id,
                        model_candidate_id=model_candidate_id,
                        run_artifact_id=run_artifact_id,
                        oos_vintage_id=None,
                        evidence_role=role,
                        profile_id=profile_id,
                        seed=int(seed),
                        dataset=str(candidate.dataset),
                        dataset_identity_sha256=str(candidate.dataset_identity_sha256),
                        train_start=train_start,
                        train_end=train_end,
                        valid_start=valid_start,
                        valid_end=valid_end,
                        final_oos_start=test_start,
                        final_oos_end=test_end,
                        metrics_json=normalized_metrics,
                        metrics_sha256=canonical_sha256(normalized_metrics),
                        gate_status=gate_status,
                        gate_reasons_json=[str(item) for item in gate_reasons],
                        evaluator_version=evidence_json["evaluator_version"],
                        candidate_manifest_sha256=str(candidate.manifest_sha256),
                        evidence_json=evidence_json,
                        evidence_sha256=canonical_sha256(evidence_json),
                        created_at=now,
                    )
                )
            except IntegrityError as exc:
                raise ValueError("this model profile/seed evidence is already recorded") from exc
            if role == "independent_gate":
                self._refresh_model_status(connection, model_candidate_id, now)
        return self.get_model_evaluation(evidence_id)

    def _refresh_model_status(self, connection: Any, candidate_id: str, now: datetime) -> None:
        rows = connection.execute(
            select(
                model_evaluations.c.profile_id,
                model_evaluations.c.seed,
                model_evaluations.c.gate_status,
            ).where(
                model_evaluations.c.model_candidate_id == candidate_id,
                model_evaluations.c.evidence_role == "independent_gate",
            )
        ).all()
        observed = {(str(row.profile_id), int(row.seed)) for row in rows}
        required = {
            (profile, seed)
            for profile in REQUIRED_RESEARCH_PROFILES
            for seed in REQUIRED_MODEL_SEEDS
        }
        status = "evaluating"
        rejection_reason = None
        if observed == required:
            failed = any(str(row.gate_status) != "passed" for row in rows)
            status = "rejected" if failed else "evaluated"
            rejection_reason = "one or more independent model gates failed" if failed else None
        connection.execute(
            update(model_candidates)
            .where(model_candidates.c.id == candidate_id)
            .values(
                status=status,
                rejection_reason=rejection_reason,
                updated_at=now,
            )
        )

    def get_model_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            return row_dict(
                self._one(connection, model_evaluations, evaluation_id, "model evaluation")
            )

    def admit_model_for_research(self, candidate_id: str, *, actor: str) -> dict[str, Any]:
        now = _now()
        with self.engine.begin() as connection:
            candidate = self._one(connection, model_candidates, candidate_id, "model candidate")
            if str(candidate.status) != "evaluated":
                raise ValueError("model requires a complete independent 3-profile/3-seed grid")
            rows = connection.execute(
                select(model_evaluations).where(
                    model_evaluations.c.model_candidate_id == candidate_id,
                    model_evaluations.c.evidence_role == "independent_gate",
                )
            ).all()
            profiles: dict[str, Any] = {}
            for profile in REQUIRED_RESEARCH_PROFILES:
                profile_rows = [row for row in rows if str(row.profile_id) == profile]
                if len(profile_rows) != len(REQUIRED_MODEL_SEEDS):
                    raise ValueError("model independent evidence grid is incomplete")
                period_values = {
                    (
                        row.train_start,
                        row.train_end,
                        row.valid_start,
                        row.valid_end,
                        row.final_oos_start,
                        row.final_oos_end,
                    )
                    for row in profile_rows
                }
                if len(period_values) != 1:
                    raise ValueError("model seeds within one profile use different periods")
                first = profile_rows[0]
                if (
                    first.final_oos_start != candidate.final_oos_start
                    or first.final_oos_end != candidate.final_oos_end
                ):
                    raise ValueError("model profile uses another final OOS boundary")
                seeds: dict[str, Any] = {}
                for row in profile_rows:
                    if str(row.dataset_identity_sha256) != str(candidate.dataset_identity_sha256):
                        raise ValueError("model evaluation dataset identity drifted")
                    if str(row.candidate_manifest_sha256) != str(candidate.manifest_sha256):
                        raise ValueError("model evaluation points to a changed candidate")
                    if canonical_sha256(dict(row.metrics_json or {})) != str(row.metrics_sha256):
                        raise ValueError("model evaluation metrics were changed")
                    if canonical_sha256(dict(row.evidence_json or {})) != str(row.evidence_sha256):
                        raise ValueError("model evaluation evidence was changed")
                    if row.oos_vintage_id is not None or row.valid_end > candidate.pre_final_end:
                        raise ValueError("model research evidence opened final OOS")
                    artifact = self._one(
                        connection,
                        research_run_artifacts,
                        str(row.run_artifact_id),
                        "model evaluation artifact",
                    )
                    self._verify_artifact_row(artifact)
                    evidence = dict(row.evidence_json or {})
                    _verify_model_seed_artifacts(
                        evidence,
                        valid_start=row.valid_start,
                        valid_end=row.valid_end,
                    )
                    seeds[str(row.seed)] = {
                        "status": str(row.gate_status),
                        "metrics": dict(row.metrics_json or {}),
                        "predictions_path": evidence.get("predictions_path"),
                        "predictions_sha256": evidence.get("predictions_sha256"),
                        "checkpoint_path": evidence.get("checkpoint_path"),
                        "checkpoint_sha256": evidence.get("checkpoint_sha256"),
                        "portfolio_report_path": evidence.get(
                            "portfolio_report_path"
                        ),
                        "portfolio_report_sha256": evidence.get(
                            "portfolio_report_sha256"
                        ),
                        "execution_evidence_sha256": evidence.get("execution_evidence_sha256"),
                        "execution_environment_sha256": evidence.get(
                            "execution_environment_sha256"
                        ),
                        "latest_prediction_date": evidence.get("latest_prediction_date"),
                        "coverage": evidence.get("coverage"),
                        "evidence_sha256": str(row.evidence_sha256),
                    }
                profiles[profile] = {
                    "periods": {
                        "train_start": first.train_start.isoformat(),
                        "train_end": first.train_end.isoformat(),
                        "valid_start": first.valid_start.isoformat(),
                        "valid_end": first.valid_end.isoformat(),
                        "test_start": first.final_oos_start.isoformat(),
                        "test_end": first.final_oos_end.isoformat(),
                    },
                    "seeds": seeds,
                }
            environment_hashes = {
                str(seed["execution_environment_sha256"])
                for profile_value in profiles.values()
                for seed in profile_value["seeds"].values()
            }
            multiple_evidence = {
                canonical_sha256(dict((row.evidence_json or {}).get("run_multiple_testing") or {})):
                dict((row.evidence_json or {}).get("run_multiple_testing") or {})
                for row in rows
            }
            multiple_trial_names = {
                str(
                    (row.evidence_json or {}).get("multiple_testing_trial_name")
                    or candidate_id
                )
                for row in rows
            }
            if (
                len(environment_hashes) != 1
                or len(multiple_evidence) != 1
                or len(multiple_trial_names) != 1
            ):
                raise ValueError("model grid environment or run-level statistics drifted")
            aggregate = {
                "contract_version": "model-research-independent-v1",
                "source": "independent_qlib_recompute",
                "candidate_id": candidate_id,
                "candidate_manifest_sha256": str(candidate.manifest_sha256),
                "feature_set_definition_sha256": str(candidate.feature_set_definition_sha256),
                "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                "final_oos_opened": False,
                "execution_environment_sha256": next(iter(environment_hashes)),
                "multiple_testing": next(iter(multiple_evidence.values())),
                "multiple_testing_trial_name": next(iter(multiple_trial_names)),
                "profiles": profiles,
            }
            validated = validate_model_evaluation_evidence(
                aggregate,
                candidate_id=candidate_id,
                dataset_identity_sha256=str(candidate.dataset_identity_sha256),
                pre_final_end=candidate.pre_final_end,
            )
            connection.execute(
                update(model_candidates)
                .where(model_candidates.c.id == candidate_id)
                .values(
                    status="research_admitted",
                    admission_evidence_json=validated,
                    admission_evidence_sha256=str(validated["evidence_sha256"]),
                    admitted_by=_actor(actor),
                    admitted_at=now,
                    updated_at=now,
                    capital_eligible=False,
                )
            )
        return self.get_model_candidate(candidate_id, verify=True)

    def ingest_model_evaluation_result(
        self,
        *,
        model_candidate_id: str,
        run_artifact_id: str,
        actor: str,
    ) -> dict[str, Any]:
        """Atomically ingest the governed ``evaluate_model_batch.py`` envelope.

        The outer ``status=ok`` only means the batch completed. Admission uses
        the matching candidate's independently validated evidence; an official
        RD-Agent trace or score is rejected even if it reports success.
        """

        responsible_actor = _actor(actor)
        now = _now()
        with self.engine.begin() as connection:
            candidate = self._one(
                connection, model_candidates, model_candidate_id, "model candidate"
            )
            if str(candidate.status) not in {
                "awaiting_independent_evaluation",
                "evaluating",
            }:
                raise ValueError("model candidate is not open for independent ingestion")
            artifact = self._one(
                connection, research_run_artifacts, run_artifact_id, "model result artifact"
            )
            self._verify_artifact_row(artifact)
            if str(artifact.research_run_id) != str(candidate.research_run_id):
                raise ValueError("model result artifact belongs to another research run")
            producer = str(artifact.producer or "").strip().lower()
            if producer in {"rdagent", "rd-agent", "official_rdagent", "official-rdagent"}:
                raise ValueError("RD-Agent internal output cannot be ingested as independent")
            try:
                envelope = json.loads(Path(str(artifact.storage_path)).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("model evaluation result artifact is unreadable") from exc
            if not isinstance(envelope, dict) or envelope.get("status") != "ok":
                raise ValueError("model evaluation batch did not complete successfully")
            matches = [
                item
                for item in (envelope.get("evaluations") or [])
                if isinstance(item, dict)
                and str(item.get("candidate_id") or "") == model_candidate_id
            ]
            if len(matches) != 1 or matches[0].get("status") != "passed":
                if len(matches) == 1 and matches[0].get("status") == "failed":
                    failure = {
                        "contract_version": ADMISSION_CONTRACT_VERSION,
                        "candidate_id": model_candidate_id,
                        "status": "failed",
                        "error": str(
                            matches[0].get("error") or "independent model evaluation failed"
                        ),
                        "run_artifact_id": run_artifact_id,
                        "run_artifact_sha256": str(artifact.content_sha256),
                        "recorded_by": responsible_actor,
                    }
                    connection.execute(
                        update(model_candidates)
                        .where(model_candidates.c.id == model_candidate_id)
                        .values(
                            status="rejected",
                            rejection_reason=failure["error"],
                            admission_evidence_json=failure,
                            admission_evidence_sha256=canonical_sha256(failure),
                            updated_at=now,
                            capital_eligible=False,
                        )
                    )
                    failed_row = connection.execute(
                        select(model_candidates).where(model_candidates.c.id == model_candidate_id)
                    ).one()
                    return row_dict(failed_row)
                raise ValueError("model result has no unique passed candidate evaluation")
            item = matches[0]
            evidence = item.get("evidence")
            if not isinstance(evidence, dict):
                raise ValueError("model result is missing independent evidence")
            evidence_without_hash = {
                key: value for key, value in evidence.items() if key != "evidence_sha256"
            }
            evidence_sha256 = canonical_sha256(evidence_without_hash)
            if (
                evidence.get("evidence_sha256") != evidence_sha256
                or item.get("evidence_sha256") != evidence_sha256
            ):
                raise ValueError("model result evidence SHA-256 is invalid")
            validated = validate_model_evaluation_evidence(
                evidence,
                candidate_id=model_candidate_id,
                dataset_identity_sha256=str(candidate.dataset_identity_sha256),
                pre_final_end=candidate.pre_final_end,
            )
            if str(validated.get("feature_set_definition_sha256") or "") != str(
                candidate.feature_set_definition_sha256
            ):
                raise ValueError("model result uses another governed feature set")
            existing = connection.scalar(
                select(model_evaluations.c.id)
                .where(
                    model_evaluations.c.model_candidate_id == model_candidate_id,
                    model_evaluations.c.evidence_role == "independent_gate",
                )
                .limit(1)
            )
            if existing is not None:
                raise ValueError("model independent evidence was already partially ingested")

            for profile_id in REQUIRED_RESEARCH_PROFILES:
                profile = validated["profiles"][profile_id]
                periods = dict(profile["periods"])
                try:
                    parsed = {
                        key: date.fromisoformat(str(periods[key]))
                        for key in (
                            "train_start",
                            "train_end",
                            "valid_start",
                            "valid_end",
                            "test_start",
                            "test_end",
                        )
                    }
                except (KeyError, ValueError) as exc:
                    raise ValueError("model result contains invalid research periods") from exc
                _periods(**parsed)
                if parsed["valid_end"] > candidate.pre_final_end:
                    raise ValueError("model result reaches beyond the pre-final boundary")
                if (
                    parsed["test_start"] != candidate.final_oos_start
                    or parsed["test_end"] != candidate.final_oos_end
                ):
                    raise ValueError("model result uses another final OOS boundary")
                for seed in REQUIRED_MODEL_SEEDS:
                    seed_result = profile["seeds"].get(str(seed), profile["seeds"].get(seed))
                    _verify_model_seed_artifacts(
                        seed_result,
                        valid_start=parsed["valid_start"],
                        valid_end=parsed["valid_end"],
                    )
                    metrics = _finite_metrics(seed_result["metrics"])
                    seed_evidence = {
                        "contract_version": ADMISSION_CONTRACT_VERSION,
                        "source": "independent_qlib_recompute",
                        "aggregate_evidence_sha256": evidence_sha256,
                        "candidate_id": model_candidate_id,
                        "candidate_manifest_sha256": str(candidate.manifest_sha256),
                        "run_artifact_id": run_artifact_id,
                        "run_artifact_sha256": str(artifact.content_sha256),
                        "dataset": str(candidate.dataset),
                        "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                        "feature_set_definition_sha256": str(
                            candidate.feature_set_definition_sha256
                        ),
                        "profile_id": profile_id,
                        "seed": seed,
                        "periods": periods,
                        "gate_status": "passed",
                        "gate_reasons": [],
                        "metrics_sha256": canonical_sha256(metrics),
                        "predictions_path": seed_result["predictions_path"],
                        "predictions_sha256": seed_result["predictions_sha256"],
                        "checkpoint_path": seed_result["checkpoint_path"],
                        "checkpoint_sha256": seed_result.get("checkpoint_sha256"),
                        "portfolio_report_path": seed_result["portfolio_report_path"],
                        "portfolio_report_sha256": seed_result[
                            "portfolio_report_sha256"
                        ],
                        "execution_evidence_sha256": seed_result["execution_evidence_sha256"],
                        "execution_environment_sha256": seed_result[
                            "execution_environment_sha256"
                        ],
                        "coverage": seed_result.get("coverage"),
                        "run_multiple_testing": validated["multiple_testing"],
                        "multiple_testing_trial_name": validated.get(
                            "multiple_testing_trial_name", model_candidate_id
                        ),
                        "final_oos_opened": False,
                    }
                    connection.execute(
                        insert(model_evaluations).values(
                            id=uuid.uuid4().hex,
                            model_candidate_id=model_candidate_id,
                            run_artifact_id=run_artifact_id,
                            oos_vintage_id=None,
                            evidence_role="independent_gate",
                            profile_id=profile_id,
                            seed=seed,
                            dataset=str(candidate.dataset),
                            dataset_identity_sha256=str(candidate.dataset_identity_sha256),
                            train_start=parsed["train_start"],
                            train_end=parsed["train_end"],
                            valid_start=parsed["valid_start"],
                            valid_end=parsed["valid_end"],
                            final_oos_start=parsed["test_start"],
                            final_oos_end=parsed["test_end"],
                            metrics_json=metrics,
                            metrics_sha256=canonical_sha256(metrics),
                            gate_status="passed",
                            gate_reasons_json=[],
                            evaluator_version="evaluate_model_batch-v1",
                            candidate_manifest_sha256=str(candidate.manifest_sha256),
                            evidence_json=seed_evidence,
                            evidence_sha256=canonical_sha256(seed_evidence),
                            created_at=now,
                        )
                    )
            connection.execute(
                update(model_candidates)
                .where(model_candidates.c.id == model_candidate_id)
                .values(
                    status="research_admitted",
                    admission_evidence_json=validated,
                    admission_evidence_sha256=evidence_sha256,
                    admitted_by=responsible_actor,
                    admitted_at=now,
                    updated_at=now,
                    capital_eligible=False,
                )
            )
        return self.get_model_candidate(model_candidate_id, verify=True)

    def create_quant_bundle_candidate(
        self,
        *,
        research_run_id: str,
        name: str,
        description: str,
        model_candidate_id: str,
        factor_candidate_ids: Sequence[str],
        bundle_artifact_id: str,
        experiment_family_id: str,
        feature_set_id: str,
        dataset: str,
        dataset_identity_sha256: str,
        pre_final_end: date,
        final_oos_start: date,
        final_oos_end: date,
        source_iteration: int | None = None,
        rdagent_decision: bool | None = None,
        rdagent_feedback: str | None = None,
    ) -> dict[str, Any]:
        factor_ids = [str(item) for item in factor_candidate_ids]
        if not factor_ids or len(set(factor_ids)) != len(factor_ids):
            raise ValueError("quant bundle requires a non-empty unique factor set")
        identity = _sha(dataset_identity_sha256, "dataset identity")
        dataset_name = _nonempty(dataset, "dataset")
        features = get_feature_set(_nonempty(feature_set_id, "feature set id"))
        feature_sha = _sha(features["definition_sha256"], "feature-set definition")
        if not pre_final_end < final_oos_start <= final_oos_end:
            raise ValueError("quant bundle final OOS boundary is invalid")
        bundle_id = uuid.uuid4().hex
        now = _now()
        with self.engine.begin() as connection:
            self._one(connection, research_runs, research_run_id, "research run")
            model = self._one(connection, model_candidates, model_candidate_id, "model candidate")
            if str(model.status) != "research_admitted":
                raise ValueError("quant bundle model has not passed independent research admission")
            if str(model.research_run_id) != research_run_id:
                raise ValueError("quant bundle model belongs to another research run")
            model_admission_sha256 = _sha(
                model.admission_evidence_sha256, "model admission evidence"
            )
            if str(model.dataset_identity_sha256) != identity:
                raise ValueError("quant bundle model uses another dataset identity")
            if str(model.dataset) != dataset_name:
                raise ValueError("quant bundle model uses another dataset")
            if str(model.feature_set_definition_sha256) != feature_sha:
                raise ValueError("quant bundle model uses another governed base feature set")
            if model.pre_final_end != pre_final_end:
                raise ValueError("quant bundle model uses another pre-final boundary")
            if model.final_oos_start != final_oos_start or model.final_oos_end != final_oos_end:
                raise ValueError("quant bundle model uses another final OOS boundary")
            artifact = self._one(
                connection, research_run_artifacts, bundle_artifact_id, "bundle artifact"
            )
            self._verify_artifact_row(artifact)
            if str(artifact.research_run_id) != research_run_id:
                raise ValueError("quant bundle artifact belongs to another research run")
            factors: list[dict[str, Any]] = []
            for factor_id in sorted(factor_ids):
                factor = self._one(connection, factor_candidates, factor_id, "factor candidate")
                if str(factor.status) != "promoted":
                    raise ValueError("quant bundle contains a factor without governed admission")
                if str(factor.research_run_id) != research_run_id:
                    raise ValueError("quant bundle factor belongs to another research run")
                code_sha = _sha(factor.code_sha256, "factor code")
                values_sha = _sha(factor.values_sha256, "factor values")
                evaluation = self._one(
                    connection,
                    factor_evaluations,
                    str(factor.promoted_evaluation_id),
                    "promoted factor evaluation",
                )
                if str(evaluation.dataset_identity_sha256) != identity:
                    raise ValueError("quant bundle factor uses another dataset identity")
                if str(evaluation.dataset) != dataset_name:
                    raise ValueError("quant bundle factor uses another dataset")
                if str(evaluation.evidence_sha256) != str(factor.promotion_evidence_sha256):
                    raise ValueError("quant bundle factor admission evidence is inconsistent")
                factors.append(
                    {
                        "candidate_id": factor_id,
                        "code_sha256": code_sha,
                        "values_sha256": values_sha,
                        "promotion_evidence_sha256": _sha(
                            factor.promotion_evidence_sha256, "factor promotion evidence"
                        ),
                    }
                )
            base_features = {
                "contract_version": features["contract_version"],
                "feature_set_id": features["id"],
                "feature_names": sorted(features["features"]),
                "feature_expressions": dict(features["features"]),
                "definition_sha256": feature_sha,
            }
            recipe = {
                "model_type": str(model.model_type),
                "architecture": dict(model.architecture_json or {}),
                "model_hyperparameters": dict(model.model_hyperparameters_json or {}),
                "training_hyperparameters": dict(model.training_hyperparameters_json or {}),
            }
            manifest = {
                "contract_version": QUANT_BUNDLE_CANDIDATE_CONTRACT_VERSION,
                "id": bundle_id,
                "research_run_id": research_run_id,
                "name": _nonempty(name, "bundle name"),
                "description": _nonempty(description, "bundle description"),
                "source_iteration": source_iteration,
                "experiment_family_id": _nonempty(experiment_family_id, "experiment family id"),
                "dataset": dataset_name,
                "dataset_identity_sha256": identity,
                "pre_final_end": pre_final_end.isoformat(),
                "final_oos_start": final_oos_start.isoformat(),
                "final_oos_end": final_oos_end.isoformat(),
                "feature_set_definition_sha256": feature_sha,
                "base_features_manifest_sha256": canonical_sha256(base_features),
                "bundle_artifact_id": bundle_artifact_id,
                "bundle_artifact_sha256": str(artifact.content_sha256),
                "factors": factors,
                "model": {
                    "candidate_id": model_candidate_id,
                    "code_sha256": str(model.code_sha256),
                    "recipe_sha256": canonical_sha256(recipe),
                    "admission_evidence_sha256": model_admission_sha256,
                },
            }
            try:
                connection.execute(
                    insert(quant_bundle_candidates).values(
                        id=bundle_id,
                        research_run_id=research_run_id,
                        name=manifest["name"],
                        description=manifest["description"],
                        status="awaiting_independent_evaluation",
                        source_iteration=source_iteration,
                        model_candidate_id=model_candidate_id,
                        factor_candidate_ids_json=sorted(factor_ids),
                        bundle_artifact_id=bundle_artifact_id,
                        bundle_artifact_sha256=str(artifact.content_sha256),
                        base_features_manifest_json=base_features,
                        base_features_manifest_sha256=manifest["base_features_manifest_sha256"],
                        feature_set_definition_sha256=feature_sha,
                        dataset=manifest["dataset"],
                        dataset_identity_sha256=identity,
                        pre_final_end=pre_final_end,
                        final_oos_start=final_oos_start,
                        final_oos_end=final_oos_end,
                        bundle_manifest_json=manifest,
                        bundle_manifest_sha256=canonical_sha256(manifest),
                        rdagent_decision=rdagent_decision,
                        rdagent_feedback=rdagent_feedback,
                        capital_eligible=False,
                        created_at=now,
                        updated_at=now,
                    )
                )
            except IntegrityError as exc:
                raise ValueError(f"quant bundle {name!r} already exists in this run") from exc
        return self.get_quant_bundle_candidate(bundle_id, verify=True)

    def create_joint_quant_bundle_candidate(
        self,
        *,
        research_run_id: str,
        name: str,
        description: str,
        model_candidate_id: str,
        factors: Sequence[Mapping[str, Any]],
        bundle_artifact_id: str,
        experiment_family_id: str,
        feature_set_id: str,
        dataset: str,
        dataset_identity_sha256: str,
        pre_final_end: date,
        final_oos_start: date,
        final_oos_end: date,
        source_iteration: int | None = None,
        rdagent_decision: bool | None = None,
        rdagent_feedback: str | None = None,
    ) -> dict[str, Any]:
        """Freeze an RD-Agent joint proposal before any component is admitted.

        Unlike ``create_quant_bundle_candidate``, this entry point never claims
        that the component model or factors passed their standalone gates. The
        proposed bundle is research-only and can change state only through the
        independent 27-cell joint evaluator.
        """

        if not factors:
            raise ValueError("joint quant proposal requires at least one factor")
        identity = _sha(dataset_identity_sha256, "dataset identity")
        dataset_name = _nonempty(dataset, "dataset")
        features = get_feature_set(_nonempty(feature_set_id, "feature set id"))
        feature_sha = _sha(features["definition_sha256"], "feature-set definition")
        if not pre_final_end < final_oos_start <= final_oos_end:
            raise ValueError("joint quant proposal final OOS boundary is invalid")
        family_id = _nonempty(experiment_family_id, "experiment family id")
        bundle_id = uuid.uuid4().hex
        now = _now()
        with self.engine.begin() as connection:
            self._one(connection, research_runs, research_run_id, "research run")
            model = self._one(connection, model_candidates, model_candidate_id, "model candidate")
            if (
                str(model.research_run_id) != research_run_id
                or str(model.status) != "awaiting_independent_evaluation"
                or str(model.dataset) != dataset_name
                or str(model.dataset_identity_sha256) != identity
                or str(model.feature_set_definition_sha256) != feature_sha
                or model.pre_final_end != pre_final_end
                or model.final_oos_start != final_oos_start
                or model.final_oos_end != final_oos_end
            ):
                raise ValueError("joint quant proposal model identity is inconsistent")
            artifact = self._one(
                connection, research_run_artifacts, bundle_artifact_id, "bundle artifact"
            )
            self._verify_artifact_row(artifact)
            if str(artifact.research_run_id) != research_run_id:
                raise ValueError("joint quant proposal artifact belongs to another run")

            code_digests: set[str] = set()
            frozen_factors: list[dict[str, Any]] = []
            factor_ids: list[str] = []
            for index, raw in enumerate(factors, start=1):
                code_path, code_sha256, _ = _path_evidence(str(raw.get("code_path") or ""))
                expected_code = _sha(raw.get("code_sha256"), "joint factor code")
                if code_sha256 != expected_code or code_sha256 in code_digests:
                    raise ValueError("joint quant factor code identity is invalid")
                code_digests.add(code_sha256)
                values_path_value = str(raw.get("submitted_values_path") or "").strip()
                values_path: Path | None = None
                values_sha256: str | None = None
                if values_path_value:
                    values_path, values_sha256, _ = _path_evidence(values_path_value)
                factor_id = uuid.uuid4().hex
                factor_ids.append(factor_id)
                factor_name = f"{name}-factor-{index:03d}"
                connection.execute(
                    insert(factor_candidates).values(
                        id=factor_id,
                        research_run_id=research_run_id,
                        name=factor_name,
                        description=str(raw.get("description") or factor_name),
                        formulation=raw.get("formulation"),
                        variables_json={
                            "source": "rdagent_fin_quant_joint_proposal",
                            "bundle_id": bundle_id,
                        },
                        status="awaiting_evaluation",
                        source_iteration=source_iteration,
                        experiment_family_id=family_id,
                        label_horizon_days=1,
                        experiment_count=len(factors),
                        code_path=str(code_path),
                        values_path=str(values_path) if values_path else None,
                        code_sha256=code_sha256,
                        values_sha256=values_sha256,
                        rdagent_decision=rdagent_decision,
                        rdagent_feedback=rdagent_feedback,
                        created_at=now,
                        updated_at=now,
                    )
                )
                frozen_factors.append(
                    {
                        "candidate_id": factor_id,
                        "code_sha256": code_sha256,
                        "values_sha256": values_sha256,
                    }
                )

            base_features = {
                "contract_version": features["contract_version"],
                "feature_set_id": features["id"],
                "feature_names": sorted(features["features"]),
                "feature_expressions": dict(features["features"]),
                "definition_sha256": feature_sha,
            }
            recipe = {
                "model_type": str(model.model_type),
                "architecture": dict(model.architecture_json or {}),
                "model_hyperparameters": dict(model.model_hyperparameters_json or {}),
                "training_hyperparameters": dict(model.training_hyperparameters_json or {}),
            }
            manifest = {
                "contract_version": "quant-bundle-joint-proposal-v1",
                "id": bundle_id,
                "research_run_id": research_run_id,
                "name": _nonempty(name, "bundle name"),
                "description": _nonempty(description, "bundle description"),
                "source_iteration": source_iteration,
                "experiment_family_id": family_id,
                "dataset": dataset_name,
                "dataset_identity_sha256": identity,
                "pre_final_end": pre_final_end.isoformat(),
                "final_oos_start": final_oos_start.isoformat(),
                "final_oos_end": final_oos_end.isoformat(),
                "feature_set_definition_sha256": feature_sha,
                "base_features_manifest_sha256": canonical_sha256(base_features),
                "bundle_artifact_id": bundle_artifact_id,
                "bundle_artifact_sha256": str(artifact.content_sha256),
                "factors": frozen_factors,
                "model": {
                    "candidate_id": model_candidate_id,
                    "code_sha256": str(model.code_sha256),
                    "recipe_sha256": canonical_sha256(recipe),
                },
            }
            connection.execute(
                insert(quant_bundle_candidates).values(
                    id=bundle_id,
                    research_run_id=research_run_id,
                    name=manifest["name"],
                    description=manifest["description"],
                    status="awaiting_independent_evaluation",
                    source_iteration=source_iteration,
                    model_candidate_id=model_candidate_id,
                    factor_candidate_ids_json=factor_ids,
                    bundle_artifact_id=bundle_artifact_id,
                    bundle_artifact_sha256=str(artifact.content_sha256),
                    base_features_manifest_json=base_features,
                    base_features_manifest_sha256=manifest["base_features_manifest_sha256"],
                    feature_set_definition_sha256=feature_sha,
                    dataset=dataset_name,
                    dataset_identity_sha256=identity,
                    pre_final_end=pre_final_end,
                    final_oos_start=final_oos_start,
                    final_oos_end=final_oos_end,
                    bundle_manifest_json=manifest,
                    bundle_manifest_sha256=canonical_sha256(manifest),
                    rdagent_decision=rdagent_decision,
                    rdagent_feedback=rdagent_feedback,
                    capital_eligible=False,
                    created_at=now,
                    updated_at=now,
                )
            )
        return self.get_quant_bundle_candidate(bundle_id, verify=True)

    def get_quant_bundle_candidate(
        self, candidate_id: str, *, verify: bool = False
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = self._one(
                connection, quant_bundle_candidates, candidate_id, "quant bundle candidate"
            )
            if verify:
                artifact = self._one(
                    connection,
                    research_run_artifacts,
                    str(row.bundle_artifact_id),
                    "quant bundle artifact",
                )
                self._verify_artifact_row(artifact)
                if str(artifact.content_sha256) != str(row.bundle_artifact_sha256):
                    raise ValueError("quant bundle no longer matches its registered artifact")
                if canonical_sha256(dict(row.bundle_manifest_json or {})) != str(
                    row.bundle_manifest_sha256
                ):
                    raise ValueError("quant bundle immutable manifest is invalid")
                if canonical_sha256(dict(row.base_features_manifest_json or {})) != str(
                    row.base_features_manifest_sha256
                ):
                    raise ValueError("quant bundle base-feature manifest is invalid")
                feature_set = get_feature_set(
                    str((row.base_features_manifest_json or {}).get("feature_set_id") or "")
                )
                if feature_set["definition_sha256"] != str(row.feature_set_definition_sha256):
                    raise ValueError("quant bundle feature-set definition drifted")
                manifest = dict(row.bundle_manifest_json or {})
                if (
                    manifest.get("bundle_artifact_sha256") != str(row.bundle_artifact_sha256)
                    or manifest.get("base_features_manifest_sha256")
                    != str(row.base_features_manifest_sha256)
                    or manifest.get("feature_set_definition_sha256")
                    != str(row.feature_set_definition_sha256)
                    or manifest.get("dataset_identity_sha256") != str(row.dataset_identity_sha256)
                    or manifest.get("pre_final_end") != row.pre_final_end.isoformat()
                    or manifest.get("final_oos_start") != row.final_oos_start.isoformat()
                    or manifest.get("final_oos_end") != row.final_oos_end.isoformat()
                ):
                    raise ValueError("quant bundle columns do not match its frozen manifest")
                model = self._one(
                    connection,
                    model_candidates,
                    str(row.model_candidate_id),
                    "quant bundle model",
                )
                frozen_model = dict(manifest.get("model") or {})
                joint_proposal = (
                    manifest.get("contract_version") == "quant-bundle-joint-proposal-v1"
                )
                if (
                    (not joint_proposal and str(model.status) != "research_admitted")
                    or (
                        joint_proposal
                        and str(model.status)
                        not in {
                            "awaiting_independent_evaluation",
                            "evaluating",
                            "research_admitted",
                        }
                    )
                    or frozen_model.get("candidate_id") != str(model.id)
                    or frozen_model.get("code_sha256") != str(model.code_sha256)
                    or (
                        not joint_proposal
                        and frozen_model.get("admission_evidence_sha256")
                        != str(model.admission_evidence_sha256)
                    )
                ):
                    raise ValueError("quant bundle model component is no longer valid")
                frozen_factors = {
                    str(item.get("candidate_id")): item
                    for item in (manifest.get("factors") or [])
                    if isinstance(item, dict)
                }
                if set(frozen_factors) != set(row.factor_candidate_ids_json or []):
                    raise ValueError("quant bundle factor membership changed")
                for factor_id, frozen in frozen_factors.items():
                    factor = self._one(
                        connection, factor_candidates, factor_id, "quant bundle factor"
                    )
                    if (
                        (not joint_proposal and str(factor.status) != "promoted")
                        or (joint_proposal and str(factor.status) != "awaiting_evaluation")
                        or frozen.get("code_sha256") != str(factor.code_sha256)
                        or frozen.get("values_sha256") != str(factor.values_sha256)
                        or (
                            not joint_proposal
                            and frozen.get("promotion_evidence_sha256")
                            != str(factor.promotion_evidence_sha256)
                        )
                    ):
                        raise ValueError("quant bundle factor component is no longer valid")
                if str(row.status) == "research_admitted":
                    admission = dict(row.admission_evidence_json or {})
                    if canonical_sha256(admission) != str(row.admission_evidence_sha256):
                        raise ValueError("quant bundle admission evidence hash is invalid")
                    ablations = dict(row.ablation_evidence_json or {})
                    if canonical_sha256(ablations) != str(row.ablation_evidence_sha256):
                        raise ValueError("quant bundle ablation evidence hash is invalid")
                    independent = admission.get("independent_bundle")
                    if isinstance(independent, Mapping):
                        validated = validate_quant_bundle_evidence(
                            independent,
                            dataset_identity_sha256=str(row.dataset_identity_sha256),
                        )
                        if (
                            admission.get("candidate_id") != candidate_id
                            or admission.get("bundle_manifest_sha256")
                            != str(row.bundle_manifest_sha256)
                            or admission.get("independent_bundle_sha256")
                            != validated.get("bundle_sha256")
                            or admission.get("final_oos_opened") is not False
                        ):
                            raise ValueError("quant bundle admission identity is invalid")
                        multiple = validated.get("multiple_testing")
                        if not isinstance(multiple, Mapping):
                            raise ValueError("quant bundle multiple-testing evidence is missing")
                        returns_path, returns_sha256, _ = _path_evidence(
                            str(multiple.get("returns_path") or "")
                        )
                        if returns_sha256 != str(multiple.get("returns_sha256") or ""):
                            raise ValueError(
                                "quant bundle multiple-testing return matrix changed: "
                                f"{returns_path}"
                            )
                if bool(row.capital_eligible):
                    raise ValueError("quant bundle unexpectedly became capital eligible")
        return row_dict(row)

    def record_quant_bundle_evaluation(
        self,
        *,
        quant_bundle_candidate_id: str,
        run_artifact_id: str,
        evidence_role: EvidenceRole,
        ablation: str,
        profile_id: str,
        seed: int,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        test_start: date,
        test_end: date,
        metrics: Mapping[str, Any],
        gate_status: str,
        gate_reasons: Sequence[str],
        evaluator_version: str,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        role = str(evidence_role)
        if role not in {"official_feedback", "independent_gate"}:
            raise ValueError("unknown quant bundle evidence role")
        if ablation not in REQUIRED_QUANT_ABLATIONS:
            raise ValueError("quant bundle evidence uses an unknown ablation")
        periods = _periods(
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
        )
        evaluation_id = uuid.uuid4().hex
        now = _now()
        with self.engine.begin() as connection:
            candidate = self._one(
                connection,
                quant_bundle_candidates,
                quant_bundle_candidate_id,
                "quant bundle candidate",
            )
            artifact = self._one(
                connection, research_run_artifacts, run_artifact_id, "quant evaluation artifact"
            )
            self._verify_artifact_row(artifact)
            if str(artifact.research_run_id) != str(candidate.research_run_id):
                raise ValueError("quant evaluation artifact belongs to another research run")
            if str(candidate.status) in {"research_admitted", "rejected", "invalidated"}:
                raise ValueError("quant bundle evaluation grid is already closed")
            if role == "official_feedback":
                if gate_status != "informational":
                    raise ValueError("RD-Agent internal feedback must remain informational")
                normalized_metrics = dict(metrics)
            else:
                if profile_id not in REQUIRED_RESEARCH_PROFILES:
                    raise ValueError("independent quant evidence uses an unknown profile")
                if int(seed) not in REQUIRED_MODEL_SEEDS:
                    raise ValueError("independent quant evidence uses an unregistered seed")
                if gate_status not in {"passed", "failed"}:
                    raise ValueError("independent quant gate must explicitly pass or fail")
                if valid_end > candidate.pre_final_end:
                    raise ValueError("research-stage quant evidence reaches sealed final OOS")
                if test_start != candidate.final_oos_start or test_end != candidate.final_oos_end:
                    raise ValueError("quant evaluation uses another final OOS boundary")
                if str(evidence.get("source") or "") != "independent_qlib_recompute":
                    raise ValueError("RD-Agent internal scores cannot be recorded as independent")
                if evidence.get("final_oos_opened") is not False:
                    raise ValueError("independent research evidence must not open final OOS")
                _verify_quant_seed_artifacts(
                    evidence,
                    valid_start=valid_start,
                    valid_end=valid_end,
                )
                if (
                    ablation == "joint"
                    and profile_id == REQUIRED_RESEARCH_PROFILES[0]
                    and int(seed) == REQUIRED_MODEL_SEEDS[0]
                ):
                    _validate_quant_factor_bundle_evidence(
                        evidence,
                        bundle_manifest=dict(candidate.bundle_manifest_json or {}),
                    )
                normalized_metrics = _finite_metrics(metrics)
            evidence_json = {
                **dict(evidence),
                "contract_version": ADMISSION_CONTRACT_VERSION,
                "evidence_role": role,
                "candidate_id": quant_bundle_candidate_id,
                "bundle_manifest_sha256": str(candidate.bundle_manifest_sha256),
                "run_artifact_id": run_artifact_id,
                "run_artifact_sha256": str(artifact.content_sha256),
                "dataset": str(candidate.dataset),
                "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                "feature_set_definition_sha256": str(candidate.feature_set_definition_sha256),
                "ablation": ablation,
                "profile_id": profile_id,
                "seed": int(seed),
                "periods": periods,
                "gate_status": gate_status,
                "gate_reasons": [str(item) for item in gate_reasons],
                "metrics_sha256": canonical_sha256(normalized_metrics),
                "evaluator_version": _nonempty(evaluator_version, "evaluator version"),
            }
            try:
                connection.execute(
                    insert(quant_bundle_evaluations).values(
                        id=evaluation_id,
                        quant_bundle_candidate_id=quant_bundle_candidate_id,
                        run_artifact_id=run_artifact_id,
                        oos_vintage_id=None,
                        evidence_role=role,
                        ablation=ablation,
                        profile_id=profile_id,
                        seed=int(seed),
                        dataset=str(candidate.dataset),
                        dataset_identity_sha256=str(candidate.dataset_identity_sha256),
                        train_start=train_start,
                        train_end=train_end,
                        valid_start=valid_start,
                        valid_end=valid_end,
                        final_oos_start=test_start,
                        final_oos_end=test_end,
                        metrics_json=normalized_metrics,
                        metrics_sha256=canonical_sha256(normalized_metrics),
                        gate_status=gate_status,
                        gate_reasons_json=[str(item) for item in gate_reasons],
                        evaluator_version=evidence_json["evaluator_version"],
                        bundle_manifest_sha256=str(candidate.bundle_manifest_sha256),
                        evidence_json=evidence_json,
                        evidence_sha256=canonical_sha256(evidence_json),
                        created_at=now,
                    )
                )
            except IntegrityError as exc:
                raise ValueError(
                    "this quant ablation/profile/seed evidence is already recorded"
                ) from exc
            if role == "independent_gate":
                self._refresh_quant_status(connection, quant_bundle_candidate_id, now)
        return self.get_quant_bundle_evaluation(evaluation_id)

    def _refresh_quant_status(self, connection: Any, candidate_id: str, now: datetime) -> None:
        rows = connection.execute(
            select(
                quant_bundle_evaluations.c.ablation,
                quant_bundle_evaluations.c.profile_id,
                quant_bundle_evaluations.c.seed,
                quant_bundle_evaluations.c.gate_status,
            ).where(
                quant_bundle_evaluations.c.quant_bundle_candidate_id == candidate_id,
                quant_bundle_evaluations.c.evidence_role == "independent_gate",
            )
        ).all()
        observed = {(str(row.ablation), str(row.profile_id), int(row.seed)) for row in rows}
        required = {
            (ablation, profile, seed)
            for ablation in REQUIRED_QUANT_ABLATIONS
            for profile in REQUIRED_RESEARCH_PROFILES
            for seed in REQUIRED_MODEL_SEEDS
        }
        status = "evaluating"
        rejection_reason = None
        if observed == required:
            failed = any(str(row.gate_status) != "passed" for row in rows)
            status = "rejected" if failed else "evaluated"
            rejection_reason = "one or more independent quant gates failed" if failed else None
        connection.execute(
            update(quant_bundle_candidates)
            .where(quant_bundle_candidates.c.id == candidate_id)
            .values(
                status=status,
                rejection_reason=rejection_reason,
                updated_at=now,
            )
        )

    def get_quant_bundle_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            return row_dict(
                self._one(
                    connection,
                    quant_bundle_evaluations,
                    evaluation_id,
                    "quant bundle evaluation",
                )
            )

    def ingest_quant_bundle_evaluation_result(
        self,
        *,
        quant_bundle_candidate_id: str,
        run_artifact_id: str,
        actor: str,
    ) -> dict[str, Any]:
        """Atomically ingest the governed 3-ablation/3-profile/3-seed result."""

        responsible_actor = _actor(actor)
        now = _now()
        with self.engine.begin() as connection:
            candidate = self._one(
                connection,
                quant_bundle_candidates,
                quant_bundle_candidate_id,
                "quant bundle candidate",
            )
            if str(candidate.status) not in {
                "awaiting_independent_evaluation",
                "evaluating",
            }:
                raise ValueError("quant bundle is not open for independent ingestion")
            artifact = self._one(
                connection, research_run_artifacts, run_artifact_id, "quant result artifact"
            )
            self._verify_artifact_row(artifact)
            if str(artifact.research_run_id) != str(candidate.research_run_id):
                raise ValueError("quant result artifact belongs to another research run")
            producer = str(artifact.producer or "").strip().lower()
            if producer in {"rdagent", "rd-agent", "official_rdagent", "official-rdagent"}:
                raise ValueError("RD-Agent internal output cannot be ingested as independent")
            try:
                envelope = json.loads(Path(str(artifact.storage_path)).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("quant evaluation result artifact is unreadable") from exc
            if not isinstance(envelope, dict) or envelope.get("status") != "ok":
                raise ValueError("quant evaluation batch did not complete successfully")
            matches = [
                item
                for item in (envelope.get("evaluations") or [])
                if isinstance(item, dict)
                and str(item.get("candidate_id") or "") == quant_bundle_candidate_id
            ]
            if len(matches) != 1:
                raise ValueError("quant result has no unique candidate evaluation")
            item = matches[0]
            if item.get("status") == "failed":
                failure = {
                    "contract_version": ADMISSION_CONTRACT_VERSION,
                    "candidate_id": quant_bundle_candidate_id,
                    "status": "failed",
                    "error": str(item.get("error") or "independent quant evaluation failed"),
                    "run_artifact_id": run_artifact_id,
                    "run_artifact_sha256": str(artifact.content_sha256),
                    "recorded_by": responsible_actor,
                }
                connection.execute(
                    update(quant_bundle_candidates)
                    .where(quant_bundle_candidates.c.id == quant_bundle_candidate_id)
                    .values(
                        status="rejected",
                        rejection_reason=failure["error"],
                        admission_evidence_json=failure,
                        admission_evidence_sha256=canonical_sha256(failure),
                        updated_at=now,
                        capital_eligible=False,
                    )
                )
            elif item.get("status") == "passed":
                evidence = item.get("evidence")
                if not isinstance(evidence, dict):
                    raise ValueError("quant result is missing independent evidence")
                validated = validate_quant_bundle_evidence(
                    evidence,
                    dataset_identity_sha256=str(candidate.dataset_identity_sha256),
                )
                if (
                    item.get("evidence_sha256") != validated.get("bundle_sha256")
                    or str(validated.get("id") or "") != quant_bundle_candidate_id
                    or validated.get("feature_set_definition_sha256")
                    != str(candidate.feature_set_definition_sha256)
                ):
                    raise ValueError("quant result identity or immutable hash is invalid")
                composition = dict(candidate.bundle_manifest_json or {})
                model_candidate = self._one(
                    connection,
                    model_candidates,
                    str(candidate.model_candidate_id),
                    "joint quant model candidate",
                )
                if str(model_candidate.status) not in {
                    "awaiting_independent_evaluation",
                    "evaluating",
                }:
                    raise ValueError("joint quant model is not open for independent admission")
                frozen_model = dict(composition.get("model") or {})
                result_model = dict(validated.get("model") or {})
                if (
                    validated.get("experiment_family_id") != composition.get("experiment_family_id")
                    or result_model.get("code_sha256") != frozen_model.get("code_sha256")
                    or result_model.get("recipe_sha256") != frozen_model.get("recipe_sha256")
                ):
                    raise ValueError("quant result model or experiment family changed")
                frozen_factor_codes = {
                    str(value["candidate_id"]): str(value["code_sha256"])
                    for value in composition.get("factors") or []
                }
                result_factor_codes = {
                    str(value["candidate_id"]): str(value["code_sha256"])
                    for value in validated.get("factors") or []
                }
                if result_factor_codes != frozen_factor_codes:
                    raise ValueError("quant result factor membership or code changed")
                factor_evidence = validated.get("factor_recompute_evidence")
                if not isinstance(factor_evidence, list) or len(factor_evidence) != len(
                    frozen_factor_codes
                ):
                    raise ValueError("quant result factor recomputation evidence is incomplete")
                observed_factor_evidence: set[str] = set()
                for proof in factor_evidence:
                    if not isinstance(proof, Mapping):
                        raise ValueError("quant factor recomputation proof is malformed")
                    factor_id = str(proof.get("candidate_id") or "")
                    if (
                        factor_id in observed_factor_evidence
                        or factor_id not in frozen_factor_codes
                    ):
                        raise ValueError("quant factor recomputation identity is invalid")
                    observed_factor_evidence.add(factor_id)
                    _validate_factor_recompute_evidence(
                        proof,
                        candidate_id=factor_id,
                        code_sha256=frozen_factor_codes[factor_id],
                    )
                combined_sha256 = _sha(
                    validated.get("combined_factor_values_sha256"),
                    "combined factor values",
                )
                combined_path = (
                    Path(str(artifact.storage_path)).resolve().parent
                    / "independent-quant-evaluations"
                    / quant_bundle_candidate_id
                    / "combined_factors.parquet"
                )
                _, observed_combined_sha256, _ = _path_evidence(combined_path)
                if observed_combined_sha256 != combined_sha256:
                    raise ValueError("combined factor values changed after quant evaluation")
                existing = connection.scalar(
                    select(quant_bundle_evaluations.c.id)
                    .where(
                        quant_bundle_evaluations.c.quant_bundle_candidate_id
                        == quant_bundle_candidate_id,
                        quant_bundle_evaluations.c.evidence_role == "independent_gate",
                    )
                    .limit(1)
                )
                if existing is not None:
                    raise ValueError("quant independent evidence was already partially ingested")

                profile_periods: dict[str, dict[str, date]] = {}
                for ablation in REQUIRED_QUANT_ABLATIONS:
                    ablation_evidence = validated["ablations"][ablation]
                    for profile_id in REQUIRED_RESEARCH_PROFILES:
                        profile = ablation_evidence["profiles"][profile_id]
                        raw_periods = dict(profile["periods"])
                        try:
                            periods = {
                                key: date.fromisoformat(str(raw_periods[key]))
                                for key in (
                                    "train_start",
                                    "train_end",
                                    "valid_start",
                                    "valid_end",
                                    "test_start",
                                    "test_end",
                                )
                            }
                        except (KeyError, ValueError) as exc:
                            raise ValueError("quant result contains invalid periods") from exc
                        _periods(**periods)
                        if periods["valid_end"] > candidate.pre_final_end:
                            raise ValueError("quant result reaches beyond the pre-final boundary")
                        if (
                            periods["test_start"] != candidate.final_oos_start
                            or periods["test_end"] != candidate.final_oos_end
                        ):
                            raise ValueError("quant result uses another final OOS boundary")
                        previous = profile_periods.setdefault(profile_id, periods)
                        if previous != periods:
                            raise ValueError("quant ablations use inconsistent profile periods")
                        for seed in REQUIRED_MODEL_SEEDS:
                            seed_result = profile["seeds"].get(
                                str(seed), profile["seeds"].get(seed)
                            )
                            _verify_quant_seed_artifacts(
                                seed_result,
                                valid_start=periods["valid_start"],
                                valid_end=periods["valid_end"],
                            )
                            metrics = _finite_metrics(seed_result["metrics"])
                            cell_evidence = {
                                "contract_version": ADMISSION_CONTRACT_VERSION,
                                "source": "independent_qlib_recompute",
                                "aggregate_bundle_sha256": validated["bundle_sha256"],
                                "ablation_evidence_sha256": ablation_evidence["evidence_sha256"],
                                "candidate_id": quant_bundle_candidate_id,
                                "bundle_manifest_sha256": str(candidate.bundle_manifest_sha256),
                                "run_artifact_id": run_artifact_id,
                                "run_artifact_sha256": str(artifact.content_sha256),
                                "dataset": str(candidate.dataset),
                                "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                                "feature_set_definition_sha256": str(
                                    candidate.feature_set_definition_sha256
                                ),
                                "ablation": ablation,
                                "profile_id": profile_id,
                                "seed": seed,
                                "periods": raw_periods,
                                "gate_status": "passed",
                                "gate_reasons": [],
                                "metrics_sha256": canonical_sha256(metrics),
                                "predictions_sha256": seed_result["predictions_sha256"],
                                "checkpoint_sha256": seed_result["checkpoint_sha256"],
                                "portfolio_report_path": seed_result[
                                    "portfolio_report_path"
                                ],
                                "portfolio_report_sha256": seed_result[
                                    "portfolio_report_sha256"
                                ],
                                "execution_evidence_sha256": seed_result[
                                    "execution_evidence_sha256"
                                ],
                                "execution_environment_sha256": seed_result[
                                    "execution_environment_sha256"
                                ],
                                "latest_prediction_date": seed_result["latest_prediction_date"],
                                "coverage": seed_result["coverage"],
                                "run_multiple_testing": validated[
                                    "multiple_testing"
                                ],
                                "multiple_testing_trial_name": (
                                    f"{quant_bundle_candidate_id}:{ablation}"
                                ),
                                "final_oos_opened": False,
                            }
                            connection.execute(
                                insert(quant_bundle_evaluations).values(
                                    id=uuid.uuid4().hex,
                                    quant_bundle_candidate_id=quant_bundle_candidate_id,
                                    run_artifact_id=run_artifact_id,
                                    oos_vintage_id=None,
                                    evidence_role="independent_gate",
                                    ablation=ablation,
                                    profile_id=profile_id,
                                    seed=seed,
                                    dataset=str(candidate.dataset),
                                    dataset_identity_sha256=str(candidate.dataset_identity_sha256),
                                    train_start=periods["train_start"],
                                    train_end=periods["train_end"],
                                    valid_start=periods["valid_start"],
                                    valid_end=periods["valid_end"],
                                    final_oos_start=periods["test_start"],
                                    final_oos_end=periods["test_end"],
                                    metrics_json=metrics,
                                    metrics_sha256=canonical_sha256(metrics),
                                    gate_status="passed",
                                    gate_reasons_json=[],
                                    evaluator_version="evaluate_quant_bundle-v1",
                                    bundle_manifest_sha256=str(candidate.bundle_manifest_sha256),
                                    evidence_json=cell_evidence,
                                    evidence_sha256=canonical_sha256(cell_evidence),
                                    created_at=now,
                                )
                            )
                model_only = validated["ablations"]["model_only"]
                model_profiles = dict(model_only["profiles"])
                model_aggregate = validate_model_evaluation_evidence(
                    {
                        "contract_version": MODEL_RESEARCH_CONTRACT_VERSION,
                        "source": "independent_qlib_recompute",
                        "candidate_id": str(model_candidate.id),
                        "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                        "feature_set_definition_sha256": str(
                            candidate.feature_set_definition_sha256
                        ),
                        "execution_environment_sha256": validated[
                            "execution_environment_sha256"
                        ],
                        "multiple_testing": validated["multiple_testing"],
                        "multiple_testing_trial_name": (
                            f"{quant_bundle_candidate_id}:model_only"
                        ),
                        "final_oos_opened": False,
                        "profiles": model_profiles,
                    },
                    candidate_id=str(model_candidate.id),
                    dataset_identity_sha256=str(candidate.dataset_identity_sha256),
                    pre_final_end=model_candidate.pre_final_end,
                )
                for profile_id in REQUIRED_RESEARCH_PROFILES:
                    raw_periods = dict(model_profiles[profile_id]["periods"])
                    parsed_periods = {
                        key: date.fromisoformat(str(raw_periods[key]))
                        for key in (
                            "train_start",
                            "train_end",
                            "valid_start",
                            "valid_end",
                            "test_start",
                            "test_end",
                        )
                    }
                    for seed in REQUIRED_MODEL_SEEDS:
                        seed_result = model_profiles[profile_id]["seeds"].get(
                            str(seed), model_profiles[profile_id]["seeds"].get(seed)
                        )
                        _verify_model_seed_artifacts(
                            seed_result,
                            valid_start=parsed_periods["valid_start"],
                            valid_end=parsed_periods["valid_end"],
                        )
                        metrics = _finite_metrics(seed_result["metrics"])
                        seed_evidence = {
                            "contract_version": ADMISSION_CONTRACT_VERSION,
                            "source": "independent_qlib_recompute",
                            "aggregate_evidence_sha256": model_aggregate["evidence_sha256"],
                            "aggregate_bundle_sha256": validated["bundle_sha256"],
                            "candidate_id": str(model_candidate.id),
                            "candidate_manifest_sha256": str(model_candidate.manifest_sha256),
                            "run_artifact_id": run_artifact_id,
                            "run_artifact_sha256": str(artifact.content_sha256),
                            "dataset": str(candidate.dataset),
                            "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                            "feature_set_definition_sha256": str(
                                candidate.feature_set_definition_sha256
                            ),
                            "profile_id": profile_id,
                            "seed": seed,
                            "periods": raw_periods,
                            "gate_status": "passed",
                            "gate_reasons": [],
                            "metrics_sha256": canonical_sha256(metrics),
                            "predictions_path": seed_result["predictions_path"],
                            "predictions_sha256": seed_result["predictions_sha256"],
                            "checkpoint_path": seed_result["checkpoint_path"],
                            "checkpoint_sha256": seed_result["checkpoint_sha256"],
                            "portfolio_report_path": seed_result[
                                "portfolio_report_path"
                            ],
                            "portfolio_report_sha256": seed_result[
                                "portfolio_report_sha256"
                            ],
                            "execution_evidence_sha256": seed_result["execution_evidence_sha256"],
                            "execution_environment_sha256": seed_result[
                                "execution_environment_sha256"
                            ],
                            "latest_prediction_date": seed_result["latest_prediction_date"],
                            "coverage": seed_result["coverage"],
                            "run_multiple_testing": validated[
                                "multiple_testing"
                            ],
                            "multiple_testing_trial_name": (
                                f"{quant_bundle_candidate_id}:model_only"
                            ),
                            "final_oos_opened": False,
                        }
                        connection.execute(
                            insert(model_evaluations).values(
                                id=uuid.uuid4().hex,
                                model_candidate_id=str(model_candidate.id),
                                run_artifact_id=run_artifact_id,
                                oos_vintage_id=None,
                                evidence_role="independent_gate",
                                profile_id=profile_id,
                                seed=seed,
                                dataset=str(candidate.dataset),
                                dataset_identity_sha256=str(candidate.dataset_identity_sha256),
                                train_start=parsed_periods["train_start"],
                                train_end=parsed_periods["train_end"],
                                valid_start=parsed_periods["valid_start"],
                                valid_end=parsed_periods["valid_end"],
                                final_oos_start=parsed_periods["test_start"],
                                final_oos_end=parsed_periods["test_end"],
                                metrics_json=metrics,
                                metrics_sha256=canonical_sha256(metrics),
                                gate_status="passed",
                                gate_reasons_json=[],
                                evaluator_version="evaluate_quant_bundle-model-only-v1",
                                candidate_manifest_sha256=str(model_candidate.manifest_sha256),
                                evidence_json=seed_evidence,
                                evidence_sha256=canonical_sha256(seed_evidence),
                                created_at=now,
                            )
                        )
                connection.execute(
                    update(model_candidates)
                    .where(model_candidates.c.id == str(model_candidate.id))
                    .values(
                        status="research_admitted",
                        admission_evidence_json=model_aggregate,
                        admission_evidence_sha256=model_aggregate["evidence_sha256"],
                        admitted_by=responsible_actor,
                        admitted_at=now,
                        updated_at=now,
                        capital_eligible=False,
                    )
                )
                admission = {
                    "contract_version": ADMISSION_CONTRACT_VERSION,
                    "candidate_id": quant_bundle_candidate_id,
                    "bundle_manifest_sha256": str(candidate.bundle_manifest_sha256),
                    "independent_bundle": validated,
                    "independent_bundle_sha256": validated["bundle_sha256"],
                    "final_oos_opened": False,
                }
                connection.execute(
                    update(quant_bundle_candidates)
                    .where(quant_bundle_candidates.c.id == quant_bundle_candidate_id)
                    .values(
                        status="research_admitted",
                        ablation_evidence_json=validated["ablations"],
                        ablation_evidence_sha256=canonical_sha256(validated["ablations"]),
                        admission_evidence_json=admission,
                        admission_evidence_sha256=canonical_sha256(admission),
                        admitted_by=responsible_actor,
                        admitted_at=now,
                        updated_at=now,
                        capital_eligible=False,
                    )
                )
            else:
                raise ValueError("quant candidate evaluation has an unknown status")
        return self.get_quant_bundle_candidate(quant_bundle_candidate_id, verify=True)

    def admit_quant_bundle_for_research(self, candidate_id: str, *, actor: str) -> dict[str, Any]:
        now = _now()
        with self.engine.begin() as connection:
            candidate = self._one(
                connection, quant_bundle_candidates, candidate_id, "quant bundle candidate"
            )
            if str(candidate.status) != "evaluated":
                raise ValueError("quant bundle requires a complete independent 27-cell grid")
            rows = connection.execute(
                select(quant_bundle_evaluations).where(
                    quant_bundle_evaluations.c.quant_bundle_candidate_id == candidate_id,
                    quant_bundle_evaluations.c.evidence_role == "independent_gate",
                )
            ).all()
            required = {
                (ablation, profile, seed)
                for ablation in REQUIRED_QUANT_ABLATIONS
                for profile in REQUIRED_RESEARCH_PROFILES
                for seed in REQUIRED_MODEL_SEEDS
            }
            observed = {(str(row.ablation), str(row.profile_id), int(row.seed)) for row in rows}
            if observed != required or len(rows) != len(required):
                raise ValueError("quant bundle independent evidence grid is incomplete")
            profile_periods: dict[str, tuple[date, ...]] = {}
            factor_proofs: list[dict[str, Any]] | None = None
            combined_factor_values_sha256: str | None = None
            composition = dict(candidate.bundle_manifest_json or {})
            run_multiple_testing: dict[str, Any] | None = None
            for row in rows:
                if str(row.gate_status) != "passed":
                    raise ValueError("every quant bundle ablation/profile/seed must pass")
                if str(row.dataset_identity_sha256) != str(candidate.dataset_identity_sha256):
                    raise ValueError("quant bundle evaluation dataset identity drifted")
                if str(row.bundle_manifest_sha256) != str(candidate.bundle_manifest_sha256):
                    raise ValueError("quant evaluation points to a changed bundle")
                if row.valid_end > candidate.pre_final_end or row.oos_vintage_id is not None:
                    raise ValueError("quant research evidence opened final OOS")
                if canonical_sha256(dict(row.metrics_json or {})) != str(row.metrics_sha256):
                    raise ValueError("quant evaluation metrics were changed")
                if canonical_sha256(dict(row.evidence_json or {})) != str(row.evidence_sha256):
                    raise ValueError("quant evaluation evidence was changed")
                evidence = dict(row.evidence_json or {})
                if (
                    evidence.get("source") != "independent_qlib_recompute"
                    or evidence.get("final_oos_opened") is not False
                    or not is_sha256(evidence.get("execution_evidence_sha256"))
                    or not is_sha256(evidence.get("execution_environment_sha256"))
                    or evidence.get("feature_set_definition_sha256")
                    != str(candidate.feature_set_definition_sha256)
                ):
                    raise ValueError("quant evaluation is not independent pre-final evidence")
                _verify_quant_seed_artifacts(
                    evidence,
                    valid_start=row.valid_start,
                    valid_end=row.valid_end,
                )
                observed_multiple = dict(evidence.get("run_multiple_testing") or {})
                if run_multiple_testing is None:
                    run_multiple_testing = observed_multiple
                elif canonical_sha256(observed_multiple) != canonical_sha256(
                    run_multiple_testing
                ):
                    raise ValueError("quant cells use different run-level statistics")
                periods = (
                    row.train_start,
                    row.train_end,
                    row.valid_start,
                    row.valid_end,
                    row.final_oos_start,
                    row.final_oos_end,
                )
                prior = profile_periods.setdefault(str(row.profile_id), periods)
                if prior != periods:
                    raise ValueError("quant ablations or seeds use inconsistent profile periods")
                artifact = self._one(
                    connection,
                    research_run_artifacts,
                    str(row.run_artifact_id),
                    "quant evaluation artifact",
                )
                self._verify_artifact_row(artifact)
                if evidence.get("run_artifact_sha256") != str(artifact.content_sha256):
                    raise ValueError("quant evaluation artifact hash is inconsistent")
                if (
                    str(row.ablation) == "joint"
                    and str(row.profile_id) == REQUIRED_RESEARCH_PROFILES[0]
                    and int(row.seed) == REQUIRED_MODEL_SEEDS[0]
                ):
                    factor_proofs, combined_factor_values_sha256 = (
                        _validate_quant_factor_bundle_evidence(
                            evidence,
                            bundle_manifest=composition,
                        )
                    )
            if factor_proofs is None or combined_factor_values_sha256 is None:
                raise ValueError("quant bundle has no independent factor/PIT anchor evidence")
            ablation_evidence: dict[str, Any] = {}
            for ablation in REQUIRED_QUANT_ABLATIONS:
                profiles: dict[str, Any] = {}
                for profile in REQUIRED_RESEARCH_PROFILES:
                    profile_rows = sorted(
                        (
                            row
                            for row in rows
                            if str(row.ablation) == ablation and str(row.profile_id) == profile
                        ),
                        key=lambda row: int(row.seed),
                    )
                    first = profile_rows[0]
                    seeds: dict[str, Any] = {}
                    for row in profile_rows:
                        evidence = dict(row.evidence_json or {})
                        seeds[str(row.seed)] = {
                            "status": "passed",
                            "metrics": dict(row.metrics_json or {}),
                            "latest_prediction_date": evidence["latest_prediction_date"],
                            "predictions_sha256": evidence["predictions_sha256"],
                            "checkpoint_sha256": evidence["checkpoint_sha256"],
                            "portfolio_report_path": evidence[
                                "portfolio_report_path"
                            ],
                            "portfolio_report_sha256": evidence[
                                "portfolio_report_sha256"
                            ],
                            "execution_evidence_sha256": evidence["execution_evidence_sha256"],
                            "execution_environment_sha256": evidence[
                                "execution_environment_sha256"
                            ],
                            "coverage": evidence["coverage"],
                            "cell_evidence_sha256": str(row.evidence_sha256),
                        }
                    profiles[profile] = {
                        "periods": {
                            "train_start": first.train_start.isoformat(),
                            "train_end": first.train_end.isoformat(),
                            "valid_start": first.valid_start.isoformat(),
                            "valid_end": first.valid_end.isoformat(),
                            "test_start": first.final_oos_start.isoformat(),
                            "test_end": first.final_oos_end.isoformat(),
                        },
                        "seeds": seeds,
                    }
                ablation_payload = {
                    "status": "passed",
                    "source": "independent_qlib_recompute",
                    "experiment_family_id": composition["experiment_family_id"],
                    "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                    "execution_environment_sha256": next(
                        iter(
                            {
                                seed["execution_environment_sha256"]
                                for profile_value in profiles.values()
                                for seed in profile_value["seeds"].values()
                            }
                        )
                    ),
                    "final_oos_opened": False,
                    "profiles": profiles,
                }
                ablation_payload["evidence_sha256"] = canonical_sha256(ablation_payload)
                ablation_evidence[ablation] = ablation_payload
            governed_bundle = {
                "contract_version": "quant-bundle-ablation-v1",
                "id": candidate_id,
                "dataset_identity_sha256": str(candidate.dataset_identity_sha256),
                "feature_set_definition_sha256": str(candidate.feature_set_definition_sha256),
                "experiment_family_id": composition["experiment_family_id"],
                "factors": composition["factors"],
                "factor_recompute_evidence": factor_proofs,
                "combined_factor_values_sha256": combined_factor_values_sha256,
                "model": composition["model"],
                "execution_environment_sha256": next(
                    iter(
                        {
                            item["execution_environment_sha256"]
                            for item in ablation_evidence.values()
                        }
                    )
                ),
                "multiple_testing": run_multiple_testing,
                "ablations": ablation_evidence,
                "final_oos_opened": False,
            }
            governed_bundle["bundle_sha256"] = canonical_sha256(governed_bundle)
            validated = validate_quant_bundle_evidence(
                governed_bundle,
                dataset_identity_sha256=str(candidate.dataset_identity_sha256),
            )
            admission = {
                "contract_version": ADMISSION_CONTRACT_VERSION,
                "candidate_id": candidate_id,
                "bundle_manifest_sha256": str(candidate.bundle_manifest_sha256),
                "independent_bundle": validated,
                "independent_bundle_sha256": str(validated["bundle_sha256"]),
                "final_oos_opened": False,
            }
            connection.execute(
                update(quant_bundle_candidates)
                .where(quant_bundle_candidates.c.id == candidate_id)
                .values(
                    status="research_admitted",
                    ablation_evidence_json=ablation_evidence,
                    ablation_evidence_sha256=canonical_sha256(ablation_evidence),
                    admission_evidence_json=admission,
                    admission_evidence_sha256=canonical_sha256(admission),
                    admitted_by=_actor(actor),
                    admitted_at=now,
                    updated_at=now,
                    capital_eligible=False,
                )
            )
        return self.get_quant_bundle_candidate(candidate_id, verify=True)

    def transition_candidate(
        self,
        candidate_kind: Literal["model", "quant_bundle"],
        candidate_id: str,
        *,
        status: Literal["rejected", "invalidated"],
        reason: str,
        actor: str,
    ) -> dict[str, Any]:
        table = model_candidates if candidate_kind == "model" else quant_bundle_candidates
        if candidate_kind not in {"model", "quant_bundle"}:
            raise ValueError("unknown governed candidate kind")
        why = _nonempty(reason, "candidate transition reason")
        with self.engine.begin() as connection:
            row = self._one(connection, table, candidate_id, "governed candidate")
            if str(row.status) in {"rejected", "invalidated"}:
                raise ValueError("governed candidate is already terminal")
            if str(row.status) == "research_admitted" and status == "rejected":
                raise ValueError("an admitted candidate may only be invalidated, not rejected")
            connection.execute(
                update(table)
                .where(table.c.id == candidate_id)
                .values(
                    status=status,
                    rejection_reason=f"{why} (actor={_actor(actor)})",
                    updated_at=_now(),
                    capital_eligible=False,
                )
            )
        return (
            self.get_model_candidate(candidate_id)
            if candidate_kind == "model"
            else self.get_quant_bundle_candidate(candidate_id)
        )

    def list_admitted_strategy_signals(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return DB-derived StrategySpec identities; callers never supply hashes."""

        bounded_limit = min(max(int(limit), 1), 500)
        with self.engine.connect() as connection:
            model_rows = connection.execute(
                select(model_candidates)
                .where(model_candidates.c.status == "research_admitted")
                .order_by(model_candidates.c.admitted_at.desc())
                .limit(bounded_limit)
            ).all()
            results: list[dict[str, Any]] = []
            for model in model_rows:
                model_evaluation = connection.execute(
                    select(model_evaluations)
                    .where(
                        model_evaluations.c.model_candidate_id == model.id,
                        model_evaluations.c.evidence_role == "independent_gate",
                        model_evaluations.c.gate_status == "passed",
                        model_evaluations.c.oos_vintage_id.is_(None),
                        model_evaluations.c.profile_id == PRIMARY_MODEL_PROFILE,
                        model_evaluations.c.seed == PRIMARY_MODEL_SEED,
                    )
                ).first()
                if model_evaluation is None:
                    continue
                manifest = dict(model.manifest_json or {})
                base_features = dict(model.base_features_manifest_json or {})
                model_config = {
                    "signal_source": "model_prediction",
                    "model_candidate_id": str(model.id),
                    "model_evaluation_id": str(model_evaluation.id),
                    "model_code_sha256": str(model.code_sha256),
                    "model_recipe_sha256": str(manifest.get("recipe_sha256") or ""),
                    "model_evidence_sha256": str(model.admission_evidence_sha256),
                    "feature_set_id": str(base_features.get("feature_set_id") or ""),
                    "feature_set_definition_sha256": str(model.feature_set_definition_sha256),
                    "model_primary_profile_id": PRIMARY_MODEL_PROFILE,
                    "model_primary_seed": PRIMARY_MODEL_SEED,
                    "model_refit_policy": dict(MODEL_REFIT_POLICY),
                    "model_refit_policy_sha256": MODEL_REFIT_POLICY_SHA256,
                }
                bundles = connection.execute(
                    select(quant_bundle_candidates).where(
                        quant_bundle_candidates.c.model_candidate_id == model.id,
                        quant_bundle_candidates.c.status == "research_admitted",
                    )
                ).all()
                if not bundles:
                    results.append(
                        {
                            "kind": "model",
                            "id": str(model.id),
                            "name": str(model.name),
                            "description": str(model.description),
                            "dataset": str(model.dataset),
                            "strategy_config": model_config,
                        }
                    )
                for bundle in bundles:
                    bundle_evaluation = connection.execute(
                        select(quant_bundle_evaluations)
                        .where(
                            quant_bundle_evaluations.c.quant_bundle_candidate_id == bundle.id,
                            quant_bundle_evaluations.c.evidence_role == "independent_gate",
                            quant_bundle_evaluations.c.ablation == "joint",
                            quant_bundle_evaluations.c.gate_status == "passed",
                            quant_bundle_evaluations.c.oos_vintage_id.is_(None),
                            quant_bundle_evaluations.c.profile_id == PRIMARY_MODEL_PROFILE,
                            quant_bundle_evaluations.c.seed == PRIMARY_MODEL_SEED,
                        )
                    ).first()
                    if bundle_evaluation is None:
                        continue
                    results.append(
                        {
                            "kind": "joint",
                            "id": str(bundle.id),
                            "name": str(bundle.name),
                            "description": str(bundle.description),
                            "dataset": str(bundle.dataset),
                            "strategy_config": {
                                **model_config,
                                "quant_bundle_candidate_id": str(bundle.id),
                                "quant_bundle_evaluation_id": str(bundle_evaluation.id),
                                "quant_bundle_sha256": str(bundle.bundle_manifest_sha256),
                            },
                        }
                    )
            return results[:bounded_limit]

    def run_audit_summary(self, research_run_id: str) -> dict[str, list[dict[str, Any]]]:
        """Return path-free model, bundle, artifact, asset, and provenance summaries."""

        with self.engine.connect() as connection:
            run = self._one(connection, research_runs, research_run_id, "research run")
            model_rows = connection.execute(
                select(model_candidates)
                .where(model_candidates.c.research_run_id == research_run_id)
                .order_by(model_candidates.c.created_at)
            ).all()
            bundle_rows = connection.execute(
                select(quant_bundle_candidates)
                .where(quant_bundle_candidates.c.research_run_id == research_run_id)
                .order_by(quant_bundle_candidates.c.created_at)
            ).all()
            artifact_rows = connection.execute(
                select(research_run_artifacts)
                .where(research_run_artifacts.c.research_run_id == research_run_id)
                .order_by(research_run_artifacts.c.created_at)
            ).all()
            factor_ids = connection.scalars(
                select(factor_candidates.c.id).where(
                    factor_candidates.c.research_run_id == research_run_id
                )
            ).all()
            model_ids = [str(row.id) for row in model_rows]
            bundle_ids = [str(row.id) for row in bundle_rows]
            model_evaluation_rows = (
                connection.execute(
                    select(model_evaluations)
                    .where(model_evaluations.c.model_candidate_id.in_(model_ids))
                    .order_by(model_evaluations.c.created_at)
                ).all()
                if model_ids
                else []
            )
            bundle_evaluation_rows = (
                connection.execute(
                    select(quant_bundle_evaluations)
                    .where(quant_bundle_evaluations.c.quant_bundle_candidate_id.in_(bundle_ids))
                    .order_by(quant_bundle_evaluations.c.created_at)
                ).all()
                if bundle_ids
                else []
            )
            candidate_filters = []
            if factor_ids:
                candidate_filters.append(
                    candidate_asset_links.c.factor_candidate_id.in_(factor_ids)
                )
            if model_ids:
                candidate_filters.append(candidate_asset_links.c.model_candidate_id.in_(model_ids))
            if bundle_ids:
                candidate_filters.append(
                    candidate_asset_links.c.quant_bundle_candidate_id.in_(bundle_ids)
                )
            link_rows = (
                connection.execute(
                    select(candidate_asset_links)
                    .where(or_(*candidate_filters))
                    .order_by(candidate_asset_links.c.created_at)
                ).all()
                if candidate_filters
                else []
            )
            config = dict(run.config_json or {})
            asset_ids = {
                str(value) for value in config.get("asset_ids") or [] if str(value).strip()
            }
            asset_ids.update(str(row.asset_id) for row in link_rows)
            asset_rows = (
                connection.execute(
                    select(research_assets)
                    .where(research_assets.c.id.in_(sorted(asset_ids)))
                    .order_by(research_assets.c.created_at)
                ).all()
                if asset_ids
                else []
            )

        models = [
            {
                "id": str(row.id),
                "name": str(row.name),
                "description": str(row.description),
                "status": str(row.status),
                "source_iteration": row.source_iteration,
                "model_type": str(row.model_type),
                "code_sha256": str(row.code_sha256),
                "feature_set_definition_sha256": str(row.feature_set_definition_sha256),
                "dataset": str(row.dataset),
                "dataset_identity_sha256": str(row.dataset_identity_sha256),
                "pre_final_end": row.pre_final_end,
                "final_oos_start": row.final_oos_start,
                "final_oos_end": row.final_oos_end,
                "manifest_sha256": str(row.manifest_sha256),
                "admission_evidence_sha256": row.admission_evidence_sha256,
                "rdagent_decision": row.rdagent_decision,
                "rdagent_feedback": row.rdagent_feedback,
                "capital_eligible": bool(row.capital_eligible),
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in model_rows
        ]
        bundles = [
            {
                "id": str(row.id),
                "name": str(row.name),
                "description": str(row.description),
                "status": str(row.status),
                "source_iteration": row.source_iteration,
                "model_candidate_id": str(row.model_candidate_id),
                "factor_candidate_ids": list(row.factor_candidate_ids_json or []),
                "bundle_artifact_sha256": str(row.bundle_artifact_sha256),
                "feature_set_definition_sha256": str(row.feature_set_definition_sha256),
                "dataset": str(row.dataset),
                "dataset_identity_sha256": str(row.dataset_identity_sha256),
                "pre_final_end": row.pre_final_end,
                "final_oos_start": row.final_oos_start,
                "final_oos_end": row.final_oos_end,
                "bundle_manifest_sha256": str(row.bundle_manifest_sha256),
                "admission_evidence_sha256": row.admission_evidence_sha256,
                "rdagent_decision": row.rdagent_decision,
                "rdagent_feedback": row.rdagent_feedback,
                "capital_eligible": bool(row.capital_eligible),
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in bundle_rows
        ]
        artifacts = [
            {
                "id": str(row.id),
                "artifact_type": str(row.artifact_type),
                "contract_version": str(row.contract_version),
                "status": str(row.status),
                "content_sha256": str(row.content_sha256),
                "size_bytes": int(row.size_bytes),
                "manifest_sha256": str(row.manifest_sha256),
                "producer": str(row.producer),
                "source_iteration": row.source_iteration,
                "capital_eligible": bool(row.capital_eligible),
                "created_at": row.created_at,
                "invalidated_at": row.invalidated_at,
                "invalidation_reason": row.invalidation_reason,
            }
            for row in artifact_rows
        ]
        model_evaluation_summaries = [
            {
                "id": str(row.id),
                "model_candidate_id": str(row.model_candidate_id),
                "evidence_role": str(row.evidence_role),
                "profile_id": str(row.profile_id),
                "seed": int(row.seed),
                "gate_status": str(row.gate_status),
                "gate_reasons": list(row.gate_reasons_json or []),
                "dataset": str(row.dataset),
                "dataset_identity_sha256": str(row.dataset_identity_sha256),
                "train_start": row.train_start,
                "train_end": row.train_end,
                "valid_start": row.valid_start,
                "valid_end": row.valid_end,
                "final_oos_start": row.final_oos_start,
                "final_oos_end": row.final_oos_end,
                "candidate_manifest_sha256": str(row.candidate_manifest_sha256),
                "created_at": row.created_at,
            }
            for row in model_evaluation_rows
        ]
        bundle_evaluation_summaries = [
            {
                "id": str(row.id),
                "quant_bundle_candidate_id": str(row.quant_bundle_candidate_id),
                "evidence_role": str(row.evidence_role),
                "ablation": str(row.ablation),
                "profile_id": str(row.profile_id),
                "seed": int(row.seed),
                "gate_status": str(row.gate_status),
                "gate_reasons": list(row.gate_reasons_json or []),
                "dataset": str(row.dataset),
                "dataset_identity_sha256": str(row.dataset_identity_sha256),
                "train_start": row.train_start,
                "train_end": row.train_end,
                "valid_start": row.valid_start,
                "valid_end": row.valid_end,
                "final_oos_start": row.final_oos_start,
                "final_oos_end": row.final_oos_end,
                "bundle_manifest_sha256": str(row.bundle_manifest_sha256),
                "created_at": row.created_at,
            }
            for row in bundle_evaluation_rows
        ]
        links = [
            {
                "id": str(row.id),
                "asset_id": str(row.asset_id),
                "factor_candidate_id": row.factor_candidate_id,
                "model_candidate_id": row.model_candidate_id,
                "quant_bundle_candidate_id": row.quant_bundle_candidate_id,
                "relationship": str(row.relationship),
                "created_at": row.created_at,
            }
            for row in link_rows
        ]
        assets = [
            {
                "id": str(row.id),
                "asset_key": str(row.asset_key),
                "asset_type": str(row.asset_type),
                "media_type": str(row.media_type),
                "publisher": row.publisher,
                "published_at": row.published_at,
                "retrieved_at": row.retrieved_at,
                "content_sha256": str(row.content_sha256),
                "size_bytes": int(row.size_bytes),
                "manifest_sha256": str(row.manifest_sha256),
                "status": str(row.status),
                "created_at": row.created_at,
            }
            for row in asset_rows
        ]
        return {
            "model_candidates": models,
            "model_evaluations": model_evaluation_summaries,
            "quant_bundle_candidates": bundles,
            "quant_bundle_evaluations": bundle_evaluation_summaries,
            "run_artifacts": artifacts,
            "asset_links": links,
            "assets": assets,
        }

    def link_asset(
        self,
        *,
        asset_id: str,
        candidate_kind: CandidateKind,
        candidate_id: str,
        relationship: str,
        actor: str,
    ) -> dict[str, Any]:
        columns = {
            "factor": (factor_candidates, "factor_candidate_id"),
            "model": (model_candidates, "model_candidate_id"),
            "quant_bundle": (quant_bundle_candidates, "quant_bundle_candidate_id"),
        }
        if candidate_kind not in columns:
            raise ValueError("unknown candidate kind")
        candidate_table, candidate_column = columns[candidate_kind]
        link_id = uuid.uuid4().hex
        with self.engine.begin() as connection:
            asset = self._one(connection, research_assets, asset_id, "research asset")
            if str(asset.status) != "registered":
                raise ValueError("quarantined or retired assets cannot be linked")
            self._one(connection, candidate_table, candidate_id, "candidate")
            values = {
                "id": link_id,
                "asset_id": asset_id,
                "factor_candidate_id": None,
                "model_candidate_id": None,
                "quant_bundle_candidate_id": None,
                "relationship": _nonempty(relationship, "asset relationship"),
                "created_by": _actor(actor),
                "created_at": _now(),
            }
            values[candidate_column] = candidate_id
            try:
                connection.execute(insert(candidate_asset_links).values(**values))
            except IntegrityError as exc:
                raise ValueError("candidate asset link already exists") from exc
            row = self._one(connection, candidate_asset_links, link_id, "candidate asset link")
        return row_dict(row)

    @staticmethod
    def require_capital_admission(candidate_kind: str, candidate_id: str) -> None:
        raise ValueError(
            f"{candidate_kind} candidate {candidate_id!r} is research-only; "
            "create a governed StrategySpec and consume its separately sealed final OOS "
            "before any simulation or capital workflow"
        )


__all__ = [
    "RDAGentCandidateStore",
    "REQUIRED_MODEL_SEEDS",
    "REQUIRED_RESEARCH_PROFILES",
    "REQUIRED_QUANT_ABLATIONS",
    "validate_model_evaluation_evidence",
    "validate_quant_bundle_evidence",
]
