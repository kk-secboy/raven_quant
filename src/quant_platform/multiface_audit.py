"""Fail-closed readiness audit for the governed multi-face research dataset.

The audit deliberately separates four claims that the UI used to blur:

* an immutable, lineage-verified Qlib dataset exists;
* each declared research feature is actually admitted by the Qlib contract;
* every information factor is checksum-verified, registered and evaluated by
  the strict rolling walk-forward path against the same dataset identity; and
* post-event market response remains a training label and never becomes a
  live/inference factor.

An evaluation is evidence even when its economic gate fails or the available
history is insufficient.  Operational failures and missing rolling-evaluation
configuration are not evidence.  This distinction prevents readiness from
being confused with a claim that a factor has alpha.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from quant_platform.announcement_nlp import FACTOR_NAME, LOGIC_FACTOR_NAME
from quant_platform.corpus_nlp import CORPUS_FACTOR_NAMES
from quant_platform.corpus_nlp import default_factors_dir as corpus_dir
from quant_platform.event_market_response import DEFAULT_HORIZONS, LABEL_ROLE
from quant_platform.major_news_mentions import (
    FACTOR_NAMES as MAJOR_NEWS_FACTOR_NAMES,
)
from quant_platform.major_news_mentions import (
    default_factors_dir as major_news_dir,
)
from quant_platform.news_flash_factors import (
    FACTOR_NAMES as NEWS_FLASH_FACTOR_NAMES,
)
from quant_platform.news_flash_factors import (
    default_factors_dir as news_flash_dir,
)
from quant_platform.report_rc_factors import (
    FACTOR_NAMES as REPORT_RC_FACTOR_NAMES,
)
from quant_platform.report_rc_factors import (
    default_factors_dir as report_rc_dir,
)

SCHEMA_VERSION = "multiface-readiness.v1"
MIN_RESEARCH_FEATURE_CONTRACT_VERSION = 5

TECHNICAL_FIELDS = ("open", "high", "low", "close", "vwap", "volume", "change", "amount")
CAPITAL_FLOW_FIELDS = (
    "mf_net_inflow_amount",
    "mf_net_inflow_ratio",
    "mf_large_order_imbalance",
)
FORBIDDEN_LABEL_CONSUMERS = (
    "factor_candidates",
    "qlib_inference_features",
    "live_signal_generation",
)

FACTOR_FACES: dict[str, tuple[str, ...]] = {
    FACTOR_NAME: ("sentiment",),
    LOGIC_FACTOR_NAME: ("logic",),
    **{name: ("fundamental",) for name in REPORT_RC_FACTOR_NAMES},
    **{name: ("sentiment",) for name in CORPUS_FACTOR_NAMES},
    MAJOR_NEWS_FACTOR_NAMES[0]: ("sentiment", "news"),
    MAJOR_NEWS_FACTOR_NAMES[1]: ("news",),
    **{name: ("news",) for name in NEWS_FLASH_FACTOR_NAMES},
}

SOURCE_BOUNDARIES = {
    "market_and_fundamental": {
        "requested_start": "2008-01-01",
        "rule": "immutable Qlib provenance records the actual source range",
    },
    "sell_side_report_rc": {
        "earliest_trusted": "2010-01-01",
        "reason": "upstream interface availability",
    },
    "regulatory_announcements": {
        "earliest_trusted": "2016-01-01",
        "reason": "CNInfo historical interface availability",
    },
    "capital_flow": {
        "earliest_observed": "2016-01-04",
        "reason": "first non-empty trusted moneyflow observation",
    },
    "news_and_policy_text": {
        "earliest_trusted": "2018-11-20",
        "reason": "upstream corpus interface availability",
    },
}


class ResearchLedger(Protocol):
    def find_candidate(self, *, name: str, values_sha256: str) -> dict[str, Any] | None: ...

    def list_candidates(
        self, *, run_id: str | None = None, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]: ...


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


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable JSON: {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"JSON root is not an object: {path}"
    return payload, None


def _factor_directories(data_root: Path) -> dict[str, Path]:
    announcement = data_root / "announcements" / "nlp" / "factors"
    return {
        FACTOR_NAME: announcement,
        LOGIC_FACTOR_NAME: announcement,
        **{name: report_rc_dir(data_root) for name in REPORT_RC_FACTOR_NAMES},
        **{name: corpus_dir(data_root) for name in CORPUS_FACTOR_NAMES},
        **{name: major_news_dir(data_root) for name in MAJOR_NEWS_FACTOR_NAMES},
        **{name: news_flash_dir(data_root) for name in NEWS_FLASH_FACTOR_NAMES},
    }


def _audit_qlib(data_root: Path, dataset: str) -> dict[str, Any]:
    dataset_path = data_root / "qlib" / dataset
    provenance_path = dataset_path / "metadata" / "provenance.json"
    provenance, error = _read_json(provenance_path)
    errors: list[str] = []
    if error:
        errors.append(error)
        return {
            "ready": False,
            "dataset": dataset,
            "dataset_path": str(dataset_path),
            "provenance_path": str(provenance_path),
            "errors": errors,
        }
    assert provenance is not None
    identity = provenance.get("dataset_identity_sha256")
    lineage_id = provenance.get("dataset_lineage_id")
    if provenance.get("lineage_verified") is not True:
        errors.append("Qlib provenance lineage_verified is not true")
    if not _is_sha256(identity):
        errors.append("Qlib provenance has no immutable dataset identity")
    if not _is_sha256(lineage_id):
        errors.append("Qlib provenance has no immutable dataset lineage id")

    declared_fields = {str(value) for value in (provenance.get("fields") or [])}
    missing_technical = sorted(set(TECHNICAL_FIELDS) - declared_fields)
    if missing_technical:
        errors.append(f"Qlib technical fields missing: {', '.join(missing_technical)}")

    contract = provenance.get("research_features")
    if not isinstance(contract, dict):
        errors.append("Qlib research feature contract is missing")
        contract = {}
    version = int(contract.get("version") or 0)
    if version < MIN_RESEARCH_FEATURE_CONTRACT_VERSION:
        errors.append(
            "Qlib research feature contract is obsolete: "
            f"v{version}, require v{MIN_RESEARCH_FEATURE_CONTRACT_VERSION}+"
        )
    fundamental_groups = contract.get("fundamental_fields")
    fundamental_targets = sorted(
        {
            str(target)
            for mapping in (fundamental_groups or {}).values()
            if isinstance(mapping, dict)
            for target in mapping.values()
        }
    ) if isinstance(fundamental_groups, dict) else []
    if not fundamental_targets:
        errors.append("Qlib has no admitted fundamental research fields")
    missing_fundamental = contract.get("missing_fundamental_fields") or {}
    all_null_fundamental = contract.get("all_null_fundamental_fields") or {}
    if missing_fundamental:
        errors.append("Qlib fundamental source contract has missing columns")
    if all_null_fundamental:
        errors.append("Qlib fundamental research fields contain all-null sources")

    capital_fields = {str(value) for value in (contract.get("capital_flow_fields") or [])}
    missing_capital = sorted(set(CAPITAL_FLOW_FIELDS) - capital_fields)
    if missing_capital:
        errors.append(f"Qlib capital-flow fields missing: {', '.join(missing_capital)}")
    if contract.get("missing_capital_flow_fields"):
        errors.append("Qlib capital-flow source contract has missing columns")
    if contract.get("all_null_capital_flow_fields"):
        errors.append("Qlib capital-flow research fields contain all-null sources")

    identity_ready = (
        provenance.get("lineage_verified") is True
        and _is_sha256(identity)
        and _is_sha256(lineage_id)
    )
    technical_ready = identity_ready and not missing_technical
    fundamental_ready = (
        identity_ready
        and version >= MIN_RESEARCH_FEATURE_CONTRACT_VERSION
        and bool(fundamental_targets)
        and not missing_fundamental
        and not all_null_fundamental
    )
    capital_ready = (
        identity_ready
        and version >= MIN_RESEARCH_FEATURE_CONTRACT_VERSION
        and not missing_capital
        and not contract.get("missing_capital_flow_fields")
        and not contract.get("all_null_capital_flow_fields")
    )
    return {
        "ready": not errors,
        "dataset": dataset,
        "dataset_path": str(dataset_path),
        "provenance_path": str(provenance_path),
        "dataset_identity_sha256": identity,
        "dataset_lineage_id": lineage_id,
        "lineage_verified": provenance.get("lineage_verified") is True,
        "identity_ready": identity_ready,
        "source_start_date": provenance.get("source_start_date"),
        "source_end_date": provenance.get("source_end_date"),
        "snapshot_name": provenance.get("snapshot_name"),
        "research_feature_contract_version": version,
        "technical": {
            "ready": technical_ready,
            "required": list(TECHNICAL_FIELDS),
            "missing": missing_technical,
        },
        "fundamental": {
            "ready": fundamental_ready,
            "admitted_fields": fundamental_targets,
            "missing_sources": missing_fundamental,
            "all_null_sources": all_null_fundamental,
        },
        "capital_flow": {
            "ready": capital_ready,
            "required": list(CAPITAL_FLOW_FIELDS),
            "admitted": sorted(capital_fields),
            "missing": missing_capital,
        },
        "errors": errors,
    }


def _rolling_evidence(evaluation: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    recompute = evaluation.get("recompute_evidence")
    config = recompute.get("config") if isinstance(recompute, dict) else None
    strict_config = isinstance(config, dict) and config.get("require_rolling_walk_forward") is True
    metrics = evaluation.get("metrics")
    rolling = metrics.get("rolling_walk_forward") if isinstance(metrics, dict) else None
    if isinstance(rolling, dict):
        status = str(rolling.get("status") or "")
        purge_days = int(
            rolling.get("purge_days")
            or (config or {}).get("rolling_purge_days")
            or 0
        )
        embargo_days = int(
            rolling.get("embargo_days")
            or (config or {}).get("rolling_embargo_days")
            or 0
        )
        ready = (
            strict_config
            and status in {"completed", "insufficient_evidence"}
            and rolling.get("uses_final_test_data") is False
            and purge_days >= 1
            and embargo_days >= 1
        )
        return ready, {
            "mode": "rolling_walk_forward",
            "status": status,
            "passed": rolling.get("passed"),
            "fold_count": rolling.get("fold_count", 0),
            "purge_days": purge_days,
            "embargo_days": embargo_days,
            "uses_final_test_data": rolling.get("uses_final_test_data"),
            "strict_config": strict_config,
        }
    # A factor can fail the pre-test evidence minimum before any fold can be
    # formed.  It still passed through the production strict evaluator when
    # the immutable external evidence records that configuration explicitly.
    insufficient = evaluation.get("gate_status") == "insufficient_evidence"
    ready = strict_config and insufficient
    return ready, {
        "mode": "pretest_insufficient_evidence" if insufficient else "missing",
        "status": evaluation.get("gate_status"),
        "strict_config": strict_config,
        "uses_final_test_data": False if insufficient else None,
    }


def _audit_factor(
    *,
    name: str,
    factors_dir: Path,
    ledger: ResearchLedger,
    dataset_identity_sha256: str | None,
) -> dict[str, Any]:
    manifest_path = factors_dir / f"{name}.json"
    manifest, error = _read_json(manifest_path)
    errors: list[str] = []
    if error:
        errors.append(error)
        return {
            "name": name,
            "faces": list(FACTOR_FACES[name]),
            "ready": False,
            "manifest_path": str(manifest_path),
            "errors": errors,
        }
    assert manifest is not None
    if manifest.get("factor") != name:
        errors.append("manifest factor name mismatch")
    artifact_name = manifest.get("artifact")
    artifact_path = factors_dir / str(artifact_name or "")
    expected_sha256 = manifest.get("sha256")
    if not artifact_name or not artifact_path.is_file():
        errors.append(f"factor values artifact is missing: {artifact_path}")
    elif not _is_sha256(expected_sha256):
        errors.append("factor manifest sha256 is invalid")
    elif _sha256_file(artifact_path) != expected_sha256:
        errors.append("factor values artifact checksum mismatch")
    rows = int(manifest.get("rows") or 0)
    if rows <= 0:
        errors.append("factor artifact has no usable rows")
    availability = manifest.get("availability_policy")
    if not isinstance(availability, dict) or not str(availability.get(name) or "").strip():
        errors.append("factor has no point-in-time availability policy")
    source = manifest.get("source")
    if not isinstance(source, dict) or not source:
        errors.append("factor has no governed source evidence")

    candidate = None
    evaluation = None
    rolling_ready = False
    rolling: dict[str, Any] = {"mode": "missing"}
    if _is_sha256(expected_sha256):
        candidate = ledger.find_candidate(name=name, values_sha256=str(expected_sha256))
    if candidate is None:
        errors.append("exact factor artifact is not registered in factor_candidates")
    else:
        evaluation = candidate.get("latest_evaluation")
        if not isinstance(evaluation, dict):
            errors.append("registered factor has no evaluation outcome")
        else:
            if evaluation.get("dataset_identity_sha256") != dataset_identity_sha256:
                errors.append("factor evaluation is not bound to this Qlib dataset identity")
            if evaluation.get("gate_status") not in {
                "passed",
                "failed",
                "insufficient_evidence",
            }:
                errors.append("factor evaluation has no terminal economic gate outcome")
            rolling_ready, rolling = _rolling_evidence(evaluation)
            if not rolling_ready:
                errors.append("factor lacks strict rolling walk-forward evidence")

    return {
        "name": name,
        "faces": list(FACTOR_FACES[name]),
        "ready": not errors,
        "manifest_path": str(manifest_path),
        "artifact_path": str(artifact_path),
        "values_sha256": expected_sha256,
        "rows": rows,
        "availability_policy": availability,
        "source": source,
        "candidate": (
            {
                "id": candidate.get("id"),
                "status": candidate.get("status"),
                "values_sha256": candidate.get("values_sha256"),
            }
            if candidate
            else None
        ),
        "evaluation": (
            {
                "id": evaluation.get("id"),
                "dataset_identity_sha256": evaluation.get("dataset_identity_sha256"),
                "gate_status": evaluation.get("gate_status"),
                "gate_reasons": evaluation.get("gate_reasons"),
                "rolling": rolling,
            }
            if isinstance(evaluation, dict)
            else None
        ),
        "errors": errors,
    }


def _audit_market_recognition(
    data_root: Path,
    snapshot_name: str,
    ledger: ResearchLedger,
) -> dict[str, Any]:
    labels_dir = data_root / "announcements" / "nlp" / "labels" / snapshot_name
    manifest_path = labels_dir / "event_market_response_labels.json"
    manifest, error = _read_json(manifest_path)
    errors: list[str] = []
    if error:
        errors.append(error)
        manifest = {}
    artifact_path = labels_dir / str(manifest.get("artifact") or "")
    expected_sha256 = manifest.get("sha256")
    if manifest.get("role") != LABEL_ROLE:
        errors.append(f"market-response role must be {LABEL_ROLE}")
    if not artifact_path.is_file():
        errors.append(f"market-response labels artifact is missing: {artifact_path}")
    elif not _is_sha256(expected_sha256) or _sha256_file(artifact_path) != expected_sha256:
        errors.append("market-response labels checksum verification failed")
    if int(manifest.get("rows") or 0) <= 0:
        errors.append("market-response labels have no usable rows")
    horizons = sorted(int(value) for value in (manifest.get("horizons") or []))
    if horizons != list(DEFAULT_HORIZONS):
        errors.append(f"market-response horizons must be {list(DEFAULT_HORIZONS)}")
    forbidden = {str(value) for value in (manifest.get("forbidden_consumers") or [])}
    missing_forbidden = sorted(set(FORBIDDEN_LABEL_CONSUMERS) - forbidden)
    if missing_forbidden:
        errors.append(
            "market-response label contract misses forbidden consumers: "
            + ", ".join(missing_forbidden)
        )

    label_path = artifact_path.resolve() if artifact_path.is_file() else None
    leaks: list[dict[str, Any]] = []
    for candidate in ledger.list_candidates(limit=5_000):
        variables = candidate.get("variables")
        candidate_role = variables.get("role") if isinstance(variables, dict) else None
        values_path = candidate.get("values_path")
        same_artifact = False
        if label_path is not None and values_path:
            try:
                same_artifact = Path(str(values_path)).resolve() == label_path
            except OSError:
                same_artifact = False
        name = str(candidate.get("name") or "").lower()
        suspicious_name = "market_response" in name or "market_recognition" in name
        if candidate_role == LABEL_ROLE or same_artifact or suspicious_name:
            leaks.append({"id": candidate.get("id"), "name": candidate.get("name")})
    if leaks:
        errors.append("training-only market-response labels leaked into factor_candidates")

    return {
        "ready": not errors,
        "role": manifest.get("role"),
        "manifest_path": str(manifest_path),
        "artifact_path": str(artifact_path),
        "sha256": expected_sha256,
        "rows": int(manifest.get("rows") or 0),
        "horizons": horizons,
        "forbidden_consumers": sorted(forbidden),
        "candidate_leaks": leaks,
        "errors": errors,
    }


def audit_multiface_readiness(
    data_root: Path,
    *,
    dataset: str,
    ledger: ResearchLedger,
    snapshot_name: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a content-addressed, fail-closed multi-face readiness report."""

    qlib = _audit_qlib(data_root, dataset)
    effective_snapshot = str(snapshot_name or qlib.get("snapshot_name") or "").strip()
    identity = qlib.get("dataset_identity_sha256")
    factors = [
        _audit_factor(
            name=name,
            factors_dir=directory,
            ledger=ledger,
            dataset_identity_sha256=str(identity) if _is_sha256(identity) else None,
        )
        for name, directory in _factor_directories(data_root).items()
    ]
    market_recognition = (
        _audit_market_recognition(data_root, effective_snapshot, ledger)
        if effective_snapshot
        else {
            "ready": False,
            "errors": ["snapshot name is unavailable for market-response label audit"],
        }
    )

    faces: dict[str, dict[str, Any]] = {
        "technical": {"ready": bool(qlib.get("technical", {}).get("ready"))},
        "fundamental": {
            "ready": bool(qlib.get("fundamental", {}).get("ready")) and all(
                item["ready"] for item in factors if "fundamental" in item["faces"]
            )
        },
        "capital_flow": {
            "ready": bool(qlib.get("capital_flow", {}).get("ready"))
        },
        "sentiment": {
            "ready": all(item["ready"] for item in factors if "sentiment" in item["faces"])
        },
        "news": {"ready": all(item["ready"] for item in factors if "news" in item["faces"])},
        "logic": {"ready": all(item["ready"] for item in factors if "logic" in item["faces"])},
        "market_recognition": {
            "ready": market_recognition["ready"],
            "training_label_only": True,
        },
    }
    ok = qlib["ready"] and all(item["ready"] for item in factors) and market_recognition["ready"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": (now or datetime.now(UTC)).isoformat(),
        "ok": ok,
        "dataset": dataset,
        "snapshot_name": effective_snapshot or None,
        "qlib": qlib,
        "faces": faces,
        "information_factors": factors,
        "market_recognition": market_recognition,
        "source_boundaries": SOURCE_BOUNDARIES,
        "semantics": {
            "ready_does_not_mean_profitable": True,
            "failed_economic_gate_is_a_valid_evaluation_outcome": True,
            "post_event_performance_is_never_a_live_feature": True,
        },
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def write_multiface_report(data_root: Path, report: dict[str, Any]) -> Path:
    output_dir = data_root / "verification" / "multiface"
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = str(report.get("report_sha256") or "")
    if not _is_sha256(identity):
        raise ValueError("multiface report has no content identity")
    target = output_dir / f"multiface-{identity[:16]}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, target)
    latest = data_root / "verification" / "multiface-latest.json"
    temporary_latest = latest.with_name(f".{latest.name}.{os.getpid()}.tmp")
    temporary_latest.write_text(payload, encoding="utf-8")
    os.replace(temporary_latest, latest)
    return target
