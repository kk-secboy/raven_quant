"""Register externally produced announcement NLP factor artifacts into factor_candidates.

The announcement NLP layer (:mod:`quant_platform.announcement_nlp`) emits a
values artifact (``factors/<name>.parquet``) plus a sha256 manifest
(``factors/<name>.json``) without touching the database. This module is the
bridge into the governed factor library and deliberately reuses the existing
ResearchStore invariants instead of inventing new governance:

- ``factor_candidates.research_run_id`` is non-nullable, so the import is
  wrapped in a research run of kind ``announcement_nlp_factor_import``; the run
  is marked ``succeeded`` once the candidate lands (``failed`` otherwise) so it
  never holds the unique active-kind slot.
- Non-rejected candidates require a code artifact, so a deterministic
  provenance code file is generated next to the values parquet; both artifact
  hashes are computed and recorded by :meth:`ResearchStore.add_candidate`.
- The manifest sha256 is verified against the values parquet before anything
  is written (fail closed).

Idempotency: a candidate is keyed by (name, values_sha256). Re-registering the
same artifact returns the existing candidate without creating a new run; a
changed artifact (new sha256) creates a new candidate in a new run — candidate
rows are immutable once imported, so a new row is the existing versioning
mechanism.

:func:`register_external_factor` is the generic form of this channel, reused
by other external producers (e.g. the structured report_rc factors in
``quant_platform.report_rc_factors``) with their own source identity and
metadata instead of a parallel registry.

The standard RD-Agent recompute path (factor code executed against the
daily_pv provider input) does not apply to this factor family: the values
derive from announcement NLP fields, not market data. Wiring an evaluation
runner for external factors is a separate step; candidates land in
``awaiting_evaluation`` exactly like freshly imported RD-Agent candidates.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .announcement_nlp import (
    ANNOUNCEMENTS_DIR,
    FACTOR_NAME,
    LOGIC_FACTOR_NAME,
    NLP_SUBDIR,
)
from .research_store import ResearchStore

IMPORT_RUN_KIND = "announcement_nlp_factor_import"
IMPORT_ACTOR = "announcement-nlp-registrar"
SOURCE_DATASET = "announcement_nlp_fields"


def default_factors_dir(data_root: Path) -> Path:
    """Return the directory where announcement NLP factor artifacts land."""

    return data_root / ANNOUNCEMENTS_DIR / NLP_SUBDIR / "factors"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _verified_artifact(
    factors_dir: Path,
    factor_name: str,
    *,
    source_dataset: str = SOURCE_DATASET,
    required_source_keys: tuple[str, ...] = ("prompt_version", "model"),
) -> tuple[dict[str, Any], Path, str]:
    """Load the manifest, validate its schema and checksum; fail closed.

    Returns (manifest, artifact_path, values_sha256). Any missing file,
    malformed manifest, or checksum mismatch raises ValueError before the
    database is touched. The manifest ``source`` block must carry the
    expected dataset identity plus non-empty values for every
    ``required_source_keys`` entry.
    """

    if not factor_name or not factor_name.replace("_", "").isalnum():
        raise ValueError(f"invalid factor name: {factor_name!r}")
    manifest_path = factors_dir / f"{factor_name}.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"factor manifest is missing: {manifest_path}; "
            "run the announcement NLP pipeline first"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"factor manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"factor manifest must be a JSON object: {manifest_path}")
    if manifest.get("factor") != factor_name:
        raise ValueError("factor manifest name does not match the requested factor")
    if manifest.get("artifact") != f"{factor_name}.parquet":
        raise ValueError("factor manifest artifact name is unexpected")
    values_sha256 = manifest.get("sha256")
    if not _is_sha256(values_sha256):
        raise ValueError("factor manifest carries an invalid sha256")
    policy = manifest.get("availability_policy")
    if (
        not isinstance(policy, dict)
        or not isinstance(policy.get(factor_name), str)
        or not policy[factor_name].strip()
    ):
        raise ValueError("factor manifest misses the availability policy")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or source.get("dataset") != source_dataset
        or any(
            not str(source.get(key) or "").strip() for key in required_source_keys
        )
    ):
        raise ValueError("factor manifest carries an unexpected source identity")
    rows = manifest.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise ValueError("factor manifest carries an invalid row count")
    artifact_path = factors_dir / str(manifest["artifact"])
    if not artifact_path.is_file():
        raise ValueError(f"factor values artifact is missing: {artifact_path}")
    if _sha256_file(artifact_path) != values_sha256:
        raise ValueError("factor values artifact does not match the manifest sha256")
    return manifest, artifact_path, str(values_sha256)


def _code_artifact_source(
    *, factor_name: str, manifest: dict[str, Any], values_sha256: str
) -> str:
    """Deterministic provenance code bound to factor_candidates.code_sha256.

    The transformation mirrors announcement_nlp.build_tone_factor_series so the
    registered values can be rebuilt from the announcement NLP fields index.
    """

    if factor_name == FACTOR_NAME:
        builder = "build_tone_factor_series"
        explanation = "mean LLM tone score grouped by (available_at, ts_code)"
    elif factor_name == LOGIC_FACTOR_NAME:
        builder = "build_logic_factor_series"
        explanation = (
            "mean governed direction * horizon weight * confidence grouped by "
            "(available_at, ts_code)"
        )
    else:
        raise ValueError(f"unsupported announcement factor: {factor_name}")
    source = manifest["source"]
    policy = manifest["availability_policy"][factor_name]
    return f'''"""Provenance code artifact for the externally produced {factor_name} factor.

Generated at factor-registration time by
quant_platform.announcement_factor_registry. The registered factor values are
{explanation}, normalized with the factor_evaluator.normalize_series contract;
available_at is the first
trading day strictly after the announcement date.
Post-event returns and market-response labels are excluded from this feature.

source dataset: {source["dataset"]}
prompt_version: {source["prompt_version"]}
model: {source["model"]}
availability_policy: {policy}
values sha256: {values_sha256}
"""

from __future__ import annotations

import pandas as pd

from quant_platform.announcement_nlp import {builder}

FACTOR_NAME = {factor_name!r}


def compute_factor(fields: pd.DataFrame) -> pd.Series:
    """Rebuild the factor values from the announcement NLP fields index."""

    return {builder}(fields, FACTOR_NAME)
'''


def _write_code_artifact(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(source, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class ExternalFactorMetadata:
    """Per-source registration metadata consumed by register_external_factor.

    ``variables`` is stored in factor_candidates.variables_json; ``run_config``
    is merged into the research run config on top of the common identity keys.
    """

    description: str
    formulation: str
    variables: dict[str, Any]
    code_source: str
    run_config: dict[str, Any]
    rdagent_feedback: str


def register_external_factor(
    store: ResearchStore,
    factors_dir: Path,
    *,
    factor_name: str,
    run_kind: str,
    actor: str,
    build_metadata: Callable[[dict[str, Any], str], ExternalFactorMetadata],
    source_dataset: str = SOURCE_DATASET,
    required_source_keys: tuple[str, ...] = ("prompt_version", "model"),
) -> dict[str, Any]:
    """Verify and register an external factor artifact; idempotent.

    This is the single governed channel for externally produced factors
    (announcement NLP, report_rc structured, ...): manifest sha256 fail-closed
    verification, deterministic provenance code artifact, research-run
    lineage, ResearchStore.add_candidate, idempotency key
    (name, values_sha256). ``build_metadata(manifest, values_sha256)`` supplies
    the source-specific description/formulation/variables/code artifact.

    Returns a JSON-able summary. ``created`` is False when the same artifact
    (name + values sha256) was already registered; the existing candidate is
    then returned untouched and no new research run is created.
    """

    manifest, artifact_path, values_sha256 = _verified_artifact(
        factors_dir,
        factor_name,
        source_dataset=source_dataset,
        required_source_keys=required_source_keys,
    )

    existing = store.find_candidate(name=factor_name, values_sha256=values_sha256)
    if existing is not None:
        return {
            "created": False,
            "candidate_id": existing["id"],
            "run_id": existing["research_run_id"],
            "factor_name": factor_name,
            "status": existing["status"],
            "values_sha256": values_sha256,
            "code_sha256": existing["code_sha256"],
        }

    metadata = build_metadata(manifest, values_sha256)
    experiment_family_id = f"external:{source_dataset}:{factor_name}"
    experiment_count = store.count_candidates(name=factor_name) + 1
    code_path = factors_dir / f"{factor_name}_factor.py"
    _write_code_artifact(code_path, metadata.code_source)

    manifest_path = factors_dir / f"{factor_name}.json"
    run = store.create_run(
        kind=run_kind,
        objective=(
            f"Register the externally produced {factor_name} factor artifact "
            "into the governed factor library."
        ),
        dataset=source_dataset,
        requested_by=actor,
        budget={"loop_n": 0},
        config={
            "factor_name": factor_name,
            "values_sha256": values_sha256,
            "manifest": str(manifest_path),
            **metadata.run_config,
        },
        artifact_path=factors_dir,
    )
    try:
        candidate = store.add_candidate(
            run["id"],
            name=factor_name,
            description=metadata.description,
            formulation=metadata.formulation,
            variables=metadata.variables,
            source_iteration=None,
            code_path=str(code_path),
            values_path=str(artifact_path),
            code_sha256=None,
            rdagent_decision=None,
            rdagent_feedback=metadata.rdagent_feedback,
            experiment_family_id=experiment_family_id,
            experiment_count=experiment_count,
            actor=actor,
        )
    except Exception as exc:
        store.mark_run(
            run["id"], "failed", actor=actor, error=f"factor import failed: {exc}"
        )
        raise
    store.mark_run(
        run["id"],
        "succeeded",
        actor=actor,
        runtime={
            "imported_candidate_id": candidate["id"],
            "factor_name": factor_name,
            "values_sha256": values_sha256,
            "code_sha256": candidate["code_sha256"],
        },
    )
    return {
        "created": True,
        "candidate_id": candidate["id"],
        "run_id": run["id"],
        "factor_name": factor_name,
        "status": candidate["status"],
        "values_sha256": values_sha256,
        "code_sha256": candidate["code_sha256"],
    }


def _announcement_metadata(manifest: dict[str, Any], values_sha256: str) -> ExternalFactorMetadata:
    """Announcement NLP registration metadata (tone and governed logic factors)."""

    factor_name = str(manifest["factor"])
    source = manifest["source"]
    policy = manifest["availability_policy"]
    if factor_name == FACTOR_NAME:
        family = "tone"
        formulation = "mean(tone_score) over announcements"
    elif factor_name == LOGIC_FACTOR_NAME:
        family = "governed logic"
        formulation = (
            "mean(direction_enum_score * horizon_enum_weight * confidence) over announcements"
        )
    else:
        raise ValueError(f"unsupported announcement factor: {factor_name}")
    return ExternalFactorMetadata(
        description=(
            f"Announcement NLP {family} factor per (available_at, instrument). Availability: "
            f"{policy[factor_name]} — values become visible at available_at, the first "
            "trading day strictly after the announcement date; source fields carry "
            "available_at/ingested_at. Externally produced by announcement_nlp "
            f"(prompt_version={source['prompt_version']}, model={source['model']})."
        ),
        formulation=(
            f"{formulation} grouped by (available_at, ts_code); available_at = first "
            "trading day strictly after the announcement date; post-event returns are excluded"
        ),
        variables={
            "availability_policy": policy,
            "source": source,
            "values_sha256": values_sha256,
            "manifest": None,  # filled below with the manifest path
            "rows": manifest["rows"],
            "ingested_fields": ["available_at", "ingested_at"],
        },
        code_source=_code_artifact_source(
            factor_name=factor_name, manifest=manifest, values_sha256=values_sha256
        ),
        run_config={
            "prompt_version": source["prompt_version"],
            "model": source["model"],
            "availability_policy": policy,
        },
        rdagent_feedback=(
            "externally produced announcement NLP factor; "
            "manifest sha256 verified at registration"
        ),
    )


def register_announcement_factor(
    store: ResearchStore,
    factors_dir: Path,
    *,
    factor_name: str = FACTOR_NAME,
    actor: str = IMPORT_ACTOR,
) -> dict[str, Any]:
    """Verify and register the announcement NLP factor artifact; idempotent.

    Thin wrapper over :func:`register_external_factor` with the announcement
    NLP source identity and metadata.
    """

    def build_metadata(manifest: dict[str, Any], values_sha256: str) -> ExternalFactorMetadata:
        metadata = _announcement_metadata(manifest, values_sha256)
        metadata.variables["manifest"] = str(factors_dir / f"{factor_name}.json")
        return metadata

    return register_external_factor(
        store,
        factors_dir,
        factor_name=factor_name,
        run_kind=IMPORT_RUN_KIND,
        actor=actor,
        build_metadata=build_metadata,
    )
