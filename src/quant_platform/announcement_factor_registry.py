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
from pathlib import Path
from typing import Any

from .announcement_nlp import ANNOUNCEMENTS_DIR, FACTOR_NAME, NLP_SUBDIR
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
    factors_dir: Path, factor_name: str
) -> tuple[dict[str, Any], Path, str]:
    """Load the manifest, validate its schema and checksum; fail closed.

    Returns (manifest, artifact_path, values_sha256). Any missing file,
    malformed manifest, or checksum mismatch raises ValueError before the
    database is touched.
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
        or source.get("dataset") != SOURCE_DATASET
        or not str(source.get("prompt_version") or "").strip()
        or not str(source.get("model") or "").strip()
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

    source = manifest["source"]
    policy = manifest["availability_policy"][factor_name]
    return f'''"""Provenance code artifact for the externally produced {factor_name} factor.

Generated at factor-registration time by
quant_platform.announcement_factor_registry. The registered factor values are
the mean LLM tone score grouped by (available_at, ts_code), normalized with
the factor_evaluator.normalize_series contract; available_at is the first
trading day strictly after the announcement date.

source dataset: {source["dataset"]}
prompt_version: {source["prompt_version"]}
model: {source["model"]}
availability_policy: {policy}
values sha256: {values_sha256}
"""

from __future__ import annotations

import pandas as pd

from quant_platform.factor_evaluator import normalize_series

FACTOR_NAME = {factor_name!r}


def compute_factor(fields: pd.DataFrame) -> pd.Series:
    """Rebuild the factor values from the announcement NLP fields index."""

    frame = fields[["available_at", "ts_code", "tone_score"]].copy()
    frame["tone_score"] = pd.to_numeric(frame["tone_score"], errors="coerce")
    series = frame.groupby(["available_at", "ts_code"], sort=True)["tone_score"].mean()
    series = series.rename(FACTOR_NAME)
    series.index = series.index.set_names(["datetime", "instrument"])
    return normalize_series(series, FACTOR_NAME)
'''


def _write_code_artifact(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(source, encoding="utf-8")
    os.replace(temporary, path)


def register_announcement_factor(
    store: ResearchStore,
    factors_dir: Path,
    *,
    factor_name: str = FACTOR_NAME,
    actor: str = IMPORT_ACTOR,
) -> dict[str, Any]:
    """Verify and register the announcement NLP factor artifact; idempotent.

    Returns a JSON-able summary. ``created`` is False when the same artifact
    (name + values sha256) was already registered; the existing candidate is
    then returned untouched and no new research run is created.
    """

    manifest, artifact_path, values_sha256 = _verified_artifact(factors_dir, factor_name)

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

    code_path = factors_dir / f"{factor_name}_factor.py"
    _write_code_artifact(
        code_path,
        _code_artifact_source(
            factor_name=factor_name, manifest=manifest, values_sha256=values_sha256
        ),
    )

    source = manifest["source"]
    policy = manifest["availability_policy"]
    manifest_path = factors_dir / f"{factor_name}.json"
    description = (
        "Announcement NLP tone factor: mean LLM tone score per "
        "(available_at, instrument). Availability: "
        f"{policy[factor_name]} — values become visible at available_at, the first "
        "trading day strictly after the announcement date; source fields carry "
        "available_at/ingested_at. Externally produced by announcement_nlp "
        f"(prompt_version={source['prompt_version']}, model={source['model']})."
    )
    run = store.create_run(
        kind=IMPORT_RUN_KIND,
        objective=(
            f"Register the externally produced {factor_name} factor artifact "
            "into the governed factor library."
        ),
        dataset=SOURCE_DATASET,
        requested_by=actor,
        budget={"loop_n": 0},
        config={
            "factor_name": factor_name,
            "values_sha256": values_sha256,
            "manifest": str(manifest_path),
            "prompt_version": source["prompt_version"],
            "model": source["model"],
            "availability_policy": policy,
        },
        artifact_path=factors_dir,
    )
    try:
        candidate = store.add_candidate(
            run["id"],
            name=factor_name,
            description=description,
            formulation=(
                "mean(tone_score) over announcements grouped by (available_at, ts_code); "
                "available_at = first trading day strictly after the announcement date"
            ),
            variables={
                "availability_policy": policy,
                "source": source,
                "values_sha256": values_sha256,
                "manifest": str(manifest_path),
                "rows": manifest["rows"],
                "ingested_fields": ["available_at", "ingested_at"],
            },
            source_iteration=None,
            code_path=str(code_path),
            values_path=str(artifact_path),
            code_sha256=None,
            rdagent_decision=None,
            rdagent_feedback=(
                "externally produced announcement NLP factor; "
                "manifest sha256 verified at registration"
            ),
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
