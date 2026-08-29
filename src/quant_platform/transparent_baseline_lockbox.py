"""Joint sealed-OOS preregistration for the three public control strategies.

The short, swing and long final-OOS windows necessarily overlap.  Treating
each public control as an unrelated standalone experiment would let whichever
backtest happens to start first prevent the other two from ever running.  This
module instead freezes the complete three-member family before any final OOS
is opened.  Every member still has a one-shot vintage; the joint binding only
allows the three predeclared, different-horizon windows to coexist.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import insert, or_, select, text

from quant_data.database import (
    audit_events,
    backtest_runs,
    jobs,
    oos_vintages,
    open_database,
    row_dict,
    strategies,
    strategy_versions,
    transparent_baseline_pre_result_repairs,
)
from quant_data.history_bounds import GOVERNED_DAILY_STOCK_SCOPE_VERSION
from quant_platform.eligibility import ELIGIBILITY_CONTRACT_VERSION
from quant_platform.strategy_recipes import TRANSPARENT_RESEARCH_BASELINE_IDS
from quant_platform.transparent_baseline_runner import (
    OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION,
    OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256,
    TRANSPARENT_BASELINE_RUNNER_FIELD,
)

LOCKBOX_CONTRACT_VERSION = "transparent-baseline-joint-lockbox-v1"
LOCKBOX_LINK_VERSION = "transparent-baseline-joint-lockbox-link-v1"
LOCKBOX_CONFIG_KEY = "transparent_baseline_joint_lockbox"
BOOTSTRAP_CONFIG_KEY = "transparent_baseline_bootstrap"
PRE_RESULT_REPAIR_ACTION = "transparent_baseline_pre_result_repair_registered"
PRE_RESULT_REPAIR_CONTRACT_VERSION_V1 = "transparent-baseline-pre-result-repair-v1"
PRE_RESULT_REPAIR_CONTRACT_VERSION_V2 = "transparent-baseline-pre-result-repair-v2"
# Keep the historical public name pinned to v1.  Existing receipts and callers
# must not silently acquire the wider v2 shape.
PRE_RESULT_REPAIR_CONTRACT_VERSION = PRE_RESULT_REPAIR_CONTRACT_VERSION_V1
PRE_RESULT_REPAIR_REGISTRY_VERSION = "transparent-baseline-pre-result-repair-registry-v1"
OPTIMIZER_APPLICABILITY_REPAIR_GENERATION = "v7-to-v8-optimizer-applicability"
OPTIMIZER_APPLICABILITY_REASON = "topk_equal_weight_optimizer_applicability"
OPTIMIZER_APPLICABILITY_SOURCE_COMMIT = (
    "79a88b3d3e2ed03b1785aa6ff6d578ff1d516860"
)
OPTIMIZER_APPLICABILITY_SOURCE_RECIPE_VERSION = (
    "qlib-rdagent-single-mainline-2026-08-30-v7"
)
OPTIMIZER_APPLICABILITY_SOURCE_BACKTEST_IDS = frozenset(
    {
        "f51d7fa2f4fd463e97fd5f6990b3721c",
        "8090c21aa11546bd9d59f732975afc25",
        "9c8a75ac646f452e8a5666bacd708936",
    }
)

_PRE_RESULT_REPAIR_REASON_CODES = frozenset(
    {
        "empty_eligible_session_pandas_concat_failure",
        "pre_2023_bse_history_outside_governed_scope",
    }
)
_PRE_RESULT_FAILURES = {
    "empty_eligible_session_pandas_concat_failure": (
        "cannot concatenate unaligned mixed dimensional NDFrame objects"
    ),
    "pre_2023_bse_history_outside_governed_scope": (
        "formal execution starts before native price-limit controls are complete"
    ),
}
OPTIMIZER_APPLICABILITY_FAILURE = (
    "optimizer requires 60 complete point-in-time return observations"
)
OPTIMIZER_APPLICABILITY_ERROR = f"ValueError: {OPTIMIZER_APPLICABILITY_FAILURE}"

_RECIPE_HORIZONS = {
    "short_relative_strength": "short_1_5d",
    "swing_trend": "swing_1_6m",
    "long_quality_value": "long_1_3y",
}
_MEMBER_KEYS = {
    "recipe_id",
    "recipe_version",
    "recipe_sha256",
    "horizon_profile",
    "base_config_sha256",
    "baseline_definition_sha256",
    "research_window_contract_sha256",
    "historical_start",
    "historical_end",
    "test_start",
    "test_end",
}
_V8_MEMBER_KEYS = _MEMBER_KEYS | {TRANSPARENT_BASELINE_RUNNER_FIELD}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _require_identifier(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 32 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase 32-character identifier")
    return normalized


def validate_pre_result_repair_receipt(value: Any) -> dict[str, Any]:
    """Validate an explicitly allowlisted no-performance baseline repair.

    This is intentionally narrower than a general retry token. It seals the
    exact prior attempts while they have no result/metrics and names the
    corrected recipe/data contracts before a replacement lockbox is opened.
    V1 remains byte-for-byte compatible with its historical receipt shape;
    V2 authorizes only the v7-to-v8 optimizer-applicability repair.
    """

    if not isinstance(value, Mapping):
        raise ValueError("transparent baseline pre-result repair receipt is required")
    common_keys = {
        "contract_version",
        "source_release_commit",
        "target_recipe_version",
        "target_eligibility_contract",
        "target_stock_scope_contract",
        "reason_codes",
        "performance_information_used",
        "members",
        "receipt_sha256",
    }
    contract_version = value.get("contract_version")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V1:
        keys = common_keys
        expected_reasons = _PRE_RESULT_REPAIR_REASON_CODES
        failure_markers = _PRE_RESULT_FAILURES
        repair_generation: str | None = None
    elif contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2:
        keys = common_keys | {
            "repair_generation",
            TRANSPARENT_BASELINE_RUNNER_FIELD,
        }
        expected_reasons = frozenset({OPTIMIZER_APPLICABILITY_REASON})
        failure_markers = {
            OPTIMIZER_APPLICABILITY_REASON: OPTIMIZER_APPLICABILITY_FAILURE
        }
        repair_generation = str(value.get("repair_generation") or "").strip()
        if repair_generation != OPTIMIZER_APPLICABILITY_REPAIR_GENERATION:
            raise ValueError("transparent baseline repair generation is not allowlisted")
    else:
        raise ValueError("transparent baseline pre-result repair contract is invalid")
    if set(value) != keys:
        raise ValueError("transparent baseline pre-result repair contract is invalid")
    payload = dict(value)
    receipt_sha256 = _require_sha256(
        payload.pop("receipt_sha256", None), field="receipt_sha256"
    )
    if canonical_sha256(payload) != receipt_sha256:
        raise ValueError("transparent baseline pre-result repair digest changed")
    commit = str(value.get("source_release_commit") or "").strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("transparent baseline repair source commit is invalid")
    target_recipe_version = str(value.get("target_recipe_version") or "").strip()
    if not target_recipe_version:
        raise ValueError("transparent baseline repair target recipe is required")
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2 and (
        commit != OPTIMIZER_APPLICABILITY_SOURCE_COMMIT
        or target_recipe_version != OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION
        or _require_sha256(
            value.get(TRANSPARENT_BASELINE_RUNNER_FIELD),
            field=TRANSPARENT_BASELINE_RUNNER_FIELD,
        )
        != OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
    ):
        raise ValueError("optimizer applicability repair source or target is not allowlisted")
    if (
        value.get("target_eligibility_contract") != ELIGIBILITY_CONTRACT_VERSION
        or value.get("target_stock_scope_contract")
        != GOVERNED_DAILY_STOCK_SCOPE_VERSION
    ):
        raise ValueError("transparent baseline repair target data contract changed")
    reason_codes = value.get("reason_codes")
    if (
        not isinstance(reason_codes, list)
        or set(reason_codes) != expected_reasons
        or len(reason_codes) != len(expected_reasons)
        or value.get("performance_information_used") is not False
    ):
        raise ValueError("transparent baseline repair reason or performance boundary is invalid")
    raw_members = value.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != 3:
        raise ValueError("transparent baseline repair must seal exactly three attempts")
    members: list[dict[str, Any]] = []
    identifiers: dict[str, set[str]] = {
        "backtest_id": set(),
        "job_id": set(),
        "strategy_version_id": set(),
    }
    observed_failure_markers: set[str] = set()
    member_keys = {
        "backtest_id",
        "strategy_version_id",
        "job_id",
        "dataset",
        "periods",
        "status",
        "job_status",
        "error",
        "metrics_absent",
        "result_absent",
        "files",
    }
    for raw in raw_members:
        if not isinstance(raw, Mapping) or set(raw) != member_keys:
            raise ValueError("transparent baseline repair member fields are invalid")
        member = dict(raw)
        for field in identifiers:
            identifier = _require_identifier(member.get(field), field=field)
            if identifier in identifiers[field]:
                raise ValueError(f"transparent baseline repair repeats {field}")
            identifiers[field].add(identifier)
            member[field] = identifier
        dataset = str(member.get("dataset") or "").strip()
        if not dataset or Path(dataset).name != dataset:
            raise ValueError("transparent baseline repair dataset is invalid")
        member["dataset"] = dataset
        periods = member.get("periods")
        if not isinstance(periods, Mapping) or set(periods) != {
            "historical_start",
            "historical_end",
            "start",
            "end",
        }:
            raise ValueError("transparent baseline repair periods are invalid")
        try:
            dates = {key: date.fromisoformat(str(periods[key])) for key in periods}
        except (TypeError, ValueError) as exc:
            raise ValueError("transparent baseline repair periods are invalid") from exc
        if not (
            dates["historical_start"]
            <= dates["historical_end"]
            < dates["start"]
            <= dates["end"]
        ):
            raise ValueError("transparent baseline repair periods are not ordered")
        member["periods"] = {key: dates[key].isoformat() for key in periods}
        if (
            member.get("metrics_absent") is not True
            or member.get("result_absent") is not True
            or member.get("status") not in {"running", "failed"}
            or member.get("job_status") != member.get("status")
        ):
            raise ValueError("transparent baseline repair member had a result or invalid state")
        if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2 and (
            member.get("status") != "failed"
        ):
            raise ValueError("optimizer applicability repair requires three failed attempts")
        error = member.get("error")
        if error is not None:
            error_text = str(error)
            if (
                contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2
                and error_text != OPTIMIZER_APPLICABILITY_ERROR
            ):
                raise ValueError("optimizer applicability repair error is not exact")
            matches = {
                code
                for code, marker in failure_markers.items()
                if marker in error_text
            }
            if len(matches) != 1 or member.get("status") != "failed":
                raise ValueError("transparent baseline repair failure is not allowlisted")
            observed_failure_markers.update(matches)
            member["error"] = error_text
        elif member.get("status") != "running":
            raise ValueError("transparent baseline repair failed member has no error")
        files = member.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("transparent baseline repair artifact inventory is missing")
        normalized_files: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for item in files:
            if not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256"}:
                raise ValueError("transparent baseline repair artifact fields are invalid")
            path = str(item.get("path") or "").replace("\\", "/").strip("/")
            if (
                not path
                or path in seen_paths
                or (path != "manifest.json" and not path.startswith("baseline/"))
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ValueError("transparent baseline repair contains a result artifact")
            size = item.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError("transparent baseline repair artifact size is invalid")
            seen_paths.add(path)
            normalized_files.append(
                {
                    "path": path,
                    "bytes": size,
                    "sha256": _require_sha256(
                        item.get("sha256"), field="artifact sha256"
                    ),
                }
            )
        if "manifest.json" not in seen_paths:
            raise ValueError("transparent baseline repair execution manifest is missing")
        member["files"] = normalized_files
        members.append(member)
    if contract_version == PRE_RESULT_REPAIR_CONTRACT_VERSION_V2 and (
        identifiers["backtest_id"] != OPTIMIZER_APPLICABILITY_SOURCE_BACKTEST_IDS
    ):
        raise ValueError("optimizer applicability repair backtests are not allowlisted")
    if observed_failure_markers != expected_reasons:
        raise ValueError("transparent baseline repair does not cover its allowlisted defect")
    return {
        **payload,
        "source_release_commit": commit,
        "target_recipe_version": target_recipe_version,
        **({"repair_generation": repair_generation} if repair_generation else {}),
        "reason_codes": list(reason_codes),
        "members": members,
        "receipt_sha256": receipt_sha256,
    }


def _normalize_member(raw: Mapping[str, Any]) -> dict[str, str]:
    recipe_id = str(raw.get("recipe_id") or "").strip()
    if recipe_id not in _RECIPE_HORIZONS:
        raise ValueError("transparent baseline lockbox contains an unknown recipe")
    horizon = str(raw.get("horizon_profile") or "").strip()
    if horizon != _RECIPE_HORIZONS[recipe_id]:
        raise ValueError("transparent baseline lockbox recipe horizon changed")
    recipe_version = str(raw.get("recipe_version") or "").strip()
    if not recipe_version:
        raise ValueError("transparent baseline lockbox recipe version is required")
    expected_keys = (
        _V8_MEMBER_KEYS
        if recipe_version == OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION
        else _MEMBER_KEYS
    )
    if set(raw) != expected_keys:
        raise ValueError("transparent baseline lockbox member fields are invalid")
    try:
        historical_start = date.fromisoformat(str(raw["historical_start"]))
        historical_end = date.fromisoformat(str(raw["historical_end"]))
        test_start = date.fromisoformat(str(raw["test_start"]))
        test_end = date.fromisoformat(str(raw["test_end"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transparent baseline lockbox periods are invalid") from exc
    if not historical_start <= historical_end < test_start <= test_end:
        raise ValueError("transparent baseline lockbox periods are not ordered")
    member = {
        "recipe_id": recipe_id,
        "recipe_version": recipe_version,
        "recipe_sha256": _require_sha256(
            raw.get("recipe_sha256"), field="recipe_sha256"
        ),
        "horizon_profile": horizon,
        "base_config_sha256": _require_sha256(
            raw.get("base_config_sha256"), field="base_config_sha256"
        ),
        "baseline_definition_sha256": _require_sha256(
            raw.get("baseline_definition_sha256"),
            field="baseline_definition_sha256",
        ),
        "research_window_contract_sha256": _require_sha256(
            raw.get("research_window_contract_sha256"),
            field="research_window_contract_sha256",
        ),
        "historical_start": historical_start.isoformat(),
        "historical_end": historical_end.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
    }
    if recipe_version == OPTIMIZER_APPLICABILITY_TARGET_RECIPE_VERSION:
        member[TRANSPARENT_BASELINE_RUNNER_FIELD] = _require_sha256(
            raw.get(TRANSPARENT_BASELINE_RUNNER_FIELD),
            field=TRANSPARENT_BASELINE_RUNNER_FIELD,
        )
        if (
            member[TRANSPARENT_BASELINE_RUNNER_FIELD]
            != OPTIMIZER_APPLICABILITY_TARGET_RUNNER_SHA256
        ):
            raise ValueError("transparent v8 lockbox runner identity changed")
    return member


def build_joint_lockbox(
    *,
    dataset: str,
    dataset_identity_sha256: str,
    dataset_lineage_id: str,
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the only supported three-baseline preregistration contract."""

    normalized_members = sorted(
        (_normalize_member(item) for item in members),
        key=lambda item: item["recipe_id"],
    )
    if [item["recipe_id"] for item in normalized_members] != sorted(
        TRANSPARENT_RESEARCH_BASELINE_IDS
    ):
        raise ValueError("joint lockbox must declare exactly the three public baselines")
    if len({item["horizon_profile"] for item in normalized_members}) != 3:
        raise ValueError("joint lockbox must declare one member per horizon")
    dataset_name = str(dataset or "").strip()
    if not dataset_name:
        raise ValueError("joint lockbox dataset is required")
    contract = {
        "contract_version": LOCKBOX_CONTRACT_VERSION,
        "dataset": dataset_name,
        "dataset_identity_sha256": _require_sha256(
            dataset_identity_sha256,
            field="dataset_identity_sha256",
        ),
        "dataset_lineage_id": _require_sha256(
            dataset_lineage_id,
            field="dataset_lineage_id",
        ),
        "members": normalized_members,
    }
    return {**contract, "batch_sha256": canonical_sha256(contract)}


def validate_joint_lockbox(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("transparent baseline joint lockbox is required")
    allowed = {
        "contract_version",
        "dataset",
        "dataset_identity_sha256",
        "dataset_lineage_id",
        "members",
        "batch_sha256",
    }
    if set(value) != allowed or value.get("contract_version") != LOCKBOX_CONTRACT_VERSION:
        raise ValueError("transparent baseline joint lockbox contract is invalid")
    members = value.get("members")
    if not isinstance(members, Sequence) or isinstance(members, (str, bytes)):
        raise ValueError("transparent baseline joint lockbox members are invalid")
    normalized = build_joint_lockbox(
        dataset=str(value.get("dataset") or ""),
        dataset_identity_sha256=str(value.get("dataset_identity_sha256") or ""),
        dataset_lineage_id=str(value.get("dataset_lineage_id") or ""),
        members=[item for item in members if isinstance(item, Mapping)],
    )
    if len(members) != len(normalized["members"]) or dict(value) != normalized:
        raise ValueError("transparent baseline joint lockbox digest or members changed")
    return normalized


def build_lockbox_member(
    *,
    config: Mapping[str, Any],
    formal_periods: Mapping[str, Any],
) -> dict[str, str]:
    """Describe one normalized StrategySpec before the batch binding is added."""

    bootstrap = config.get(BOOTSTRAP_CONFIG_KEY)
    if not isinstance(bootstrap, Mapping):
        raise ValueError("transparent baseline config has no frozen bootstrap contract")
    try:
        periods = {
            key: date.fromisoformat(str(formal_periods[key])).isoformat()
            for key in ("historical_start", "historical_end", "start", "end")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transparent baseline formal periods are invalid") from exc
    if dict(bootstrap.get("formal_periods") or {}) != periods:
        raise ValueError("transparent baseline formal periods differ from its bootstrap")
    raw_member = {
            "recipe_id": config.get("recipe_id"),
            "recipe_version": config.get("recipe_version"),
            "recipe_sha256": bootstrap.get("recipe_sha256"),
            "horizon_profile": config.get("horizon_profile"),
            "base_config_sha256": canonical_sha256(dict(config)),
            "baseline_definition_sha256": config.get(
                "baseline_definition_sha256"
            ),
            "research_window_contract_sha256": bootstrap.get(
                "research_window_contract_sha256"
            ),
            "historical_start": periods["historical_start"],
            "historical_end": periods["historical_end"],
            "test_start": periods["start"],
            "test_end": periods["end"],
    }
    runner_sha256 = bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
    if runner_sha256 is not None:
        raw_member[TRANSPARENT_BASELINE_RUNNER_FIELD] = runner_sha256
    return _normalize_member(raw_member)


def lockbox_member_link(config: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = config.get(LOCKBOX_CONFIG_KEY)
    if raw is None:
        return None
    lockbox = validate_joint_lockbox(raw)
    recipe_id = str(config.get("recipe_id") or "")
    matches = [
        item for item in lockbox["members"] if item["recipe_id"] == recipe_id
    ]
    if len(matches) != 1:
        raise ValueError("strategy recipe is not a unique joint-lockbox member")
    member = matches[0]
    base_config = dict(config)
    base_config.pop(LOCKBOX_CONFIG_KEY, None)
    bootstrap = base_config.get(BOOTSTRAP_CONFIG_KEY)
    if not isinstance(bootstrap, Mapping):
        raise ValueError("strategy config has no frozen bootstrap contract")
    if (
        canonical_sha256(base_config) != member["base_config_sha256"]
        or str(config.get("recipe_version") or "") != member["recipe_version"]
        or str(bootstrap.get("recipe_sha256") or "") != member["recipe_sha256"]
        or str(config.get("horizon_profile") or "") != member["horizon_profile"]
        or str(config.get("baseline_definition_sha256") or "")
        != member["baseline_definition_sha256"]
        or str(bootstrap.get("research_window_contract_sha256") or "")
        != member["research_window_contract_sha256"]
        or dict(bootstrap.get("formal_periods") or {})
        != {
            "historical_start": member["historical_start"],
            "historical_end": member["historical_end"],
            "start": member["test_start"],
            "end": member["test_end"],
        }
        or (
            member.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
            != bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
        )
    ):
        raise ValueError("strategy config differs from its joint-lockbox member")
    member_hashes = sorted(canonical_sha256(item) for item in lockbox["members"])
    return {
        "contract_version": LOCKBOX_LINK_VERSION,
        "batch_sha256": lockbox["batch_sha256"],
        "member_sha256": canonical_sha256(member),
        "member_sha256s": member_hashes,
        "recipe_id": recipe_id,
        "horizon_profile": member["horizon_profile"],
    }


def validate_lockbox_link(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("transparent baseline lockbox link is invalid")
    keys = {
        "contract_version",
        "batch_sha256",
        "member_sha256",
        "member_sha256s",
        "recipe_id",
        "horizon_profile",
    }
    if set(value) != keys or value.get("contract_version") != LOCKBOX_LINK_VERSION:
        raise ValueError("transparent baseline lockbox link contract is invalid")
    recipe_id = str(value.get("recipe_id") or "")
    if (
        recipe_id not in _RECIPE_HORIZONS
        or str(value.get("horizon_profile") or "") != _RECIPE_HORIZONS[recipe_id]
    ):
        raise ValueError("transparent baseline lockbox link recipe is invalid")
    member_hashes = value.get("member_sha256s")
    if not isinstance(member_hashes, list) or len(member_hashes) != 3:
        raise ValueError("transparent baseline lockbox link must name three members")
    normalized_hashes = sorted(
        _require_sha256(item, field="member_sha256") for item in member_hashes
    )
    member_sha256 = _require_sha256(
        value.get("member_sha256"), field="member_sha256"
    )
    if len(set(normalized_hashes)) != 3 or member_sha256 not in normalized_hashes:
        raise ValueError("transparent baseline lockbox member identities are invalid")
    return {
        "contract_version": LOCKBOX_LINK_VERSION,
        "batch_sha256": _require_sha256(
            value.get("batch_sha256"), field="batch_sha256"
        ),
        "member_sha256": member_sha256,
        "member_sha256s": normalized_hashes,
        "recipe_id": recipe_id,
        "horizon_profile": _RECIPE_HORIZONS[recipe_id],
    }


def baseline_oos_sealed_member_set(version: Mapping[str, Any]) -> dict[str, Any]:
    """Recreate the exact pure-baseline seal used by preregistration/backtest."""

    config = version.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("baseline strategy config is invalid")
    baseline_sha256 = _require_sha256(
        config.get("baseline_definition_sha256"),
        field="baseline_definition_sha256",
    )
    strategy_spec = {
        "strategy_type": "multifactor",
        "benchmark": version.get("benchmark"),
        "universe": version.get("universe"),
        "config_sha256": canonical_sha256(config),
        "baseline_definition_sha256": baseline_sha256,
    }
    result: dict[str, Any] = {
        "candidate_ids": [],
        "baseline_definition_sha256": baseline_sha256,
        "strategy_spec_sha256": canonical_sha256(strategy_spec),
        "model_signal": None,
    }
    link = lockbox_member_link(config)
    if link is not None:
        version_id = str(version.get("id") or "").strip()
        if not version_id:
            raise ValueError("joint-lockbox member requires a strategy version id")
        result.update(
            {
                "strategy_version_id": version_id,
                "transparent_baseline_lockbox": link,
            }
        )
    return result


def _repair_economic_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only version/data bindings; all economic rules must stay equal."""

    normalized = json.loads(json.dumps(dict(value), ensure_ascii=False))
    for key in ("recipe_version", BOOTSTRAP_CONFIG_KEY, LOCKBOX_CONFIG_KEY):
        normalized.pop(key, None)
    return normalized


def _artifact_result_exists(artifact_path: Any) -> bool:
    try:
        root = Path(str(artifact_path)).resolve()
    except (OSError, RuntimeError, ValueError):
        return True
    try:
        return any(path.is_file() for path in root.rglob("result.json"))
    except OSError:
        return True


def _artifact_inventory(artifact_path: Any) -> list[dict[str, Any]]:
    try:
        root = Path(str(artifact_path)).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("transparent baseline repair artifact root is invalid") from exc
    if not root.is_dir():
        raise ValueError("transparent baseline repair artifact root is missing")
    files: list[dict[str, Any]] = []
    try:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    except OSError as exc:
        raise ValueError("transparent baseline repair artifacts cannot be verified") from exc
    return files


class TransparentBaselineLockboxStore:
    """Atomically reserve/recover the three public baseline OOS vintages."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    @staticmethod
    def _repair_source_batches(
        rows: Sequence[Any],
        *,
        target_batch_sha256: str,
        superseded_source_batches: set[str] | None = None,
    ) -> dict[str, list[Any]]:
        superseded = superseded_source_batches or set()
        batches: dict[str, list[Any]] = {}
        for row in rows:
            sealed = dict(row.sealed_candidate_set_json or {})
            raw_link = sealed.get("transparent_baseline_lockbox")
            if not isinstance(raw_link, Mapping):
                continue
            try:
                link = validate_lockbox_link(raw_link)
            except ValueError:
                continue
            batch = str(link["batch_sha256"])
            if batch != target_batch_sha256 and batch not in superseded:
                batches.setdefault(batch, []).append(row)
        return batches

    @staticmethod
    def _version_repair_rows(connection: Any, version_ids: set[str]) -> dict[str, Any]:
        rows = connection.execute(
            select(
                strategy_versions.c.id,
                strategy_versions.c.strategy_id,
                strategy_versions.c.horizon_profile,
                strategy_versions.c.benchmark,
                strategy_versions.c.universe,
                strategy_versions.c.config_json,
                strategies.c.economic_hypothesis_group,
            )
            .join(strategies, strategies.c.id == strategy_versions.c.strategy_id)
            .where(strategy_versions.c.id.in_(version_ids))
        ).all()
        if len(rows) != len(version_ids):
            raise ValueError("transparent baseline repair strategy version is missing")
        return {str(row.id): row for row in rows}

    @classmethod
    def _validate_repair_event_binding(
        cls,
        connection: Any,
        *,
        audit_row: Any,
        receipt: Mapping[str, Any],
        source_batch_sha256: str,
        source_rows: Sequence[Any],
        target_batch_sha256: str,
        target_versions: Sequence[Mapping[str, Any]],
        target_expected: Sequence[Mapping[str, Any]],
        target_dataset: str,
        target_dataset_identity_sha256: str,
        target_dataset_lineage_id: str,
    ) -> dict[str, Any]:
        if (
            str(audit_row.method) != "INTERNAL"
            or str(audit_row.path) != "transparent-baseline/pre-result-repair"
            or int(audit_row.status_code) != 201
        ):
            raise ValueError("transparent baseline repair audit envelope is invalid")
        members = {str(item["strategy_version_id"]): item for item in receipt["members"]}
        source_version_ids = {
            str(dict(row.sealed_candidate_set_json or {}).get("strategy_version_id") or "")
            for row in source_rows
        }
        if (
            len(source_rows) != 3
            or len(source_version_ids) != 3
            or set(members) != source_version_ids
            or any(row.consumed_at is None for row in source_rows)
        ):
            raise ValueError("transparent baseline repair source lockbox is incomplete")
        source_links = [
            validate_lockbox_link(
                dict(row.sealed_candidate_set_json or {}).get(
                    "transparent_baseline_lockbox"
                )
            )
            for row in source_rows
        ]
        if (
            {str(item["batch_sha256"]) for item in source_links}
            != {source_batch_sha256}
            or {str(item["recipe_id"]) for item in source_links}
            != set(TRANSPARENT_RESEARCH_BASELINE_IDS)
        ):
            raise ValueError("transparent baseline repair source batch binding changed")
        source_lineages = {str(row.dataset_lineage_id or "") for row in source_rows}
        if len(source_lineages) != 1 or "" in source_lineages:
            raise ValueError("transparent baseline repair source lineage is invalid")
        source_lineage = next(iter(source_lineages))
        source_identities = {str(row.dataset_identity or "") for row in source_rows}
        if len(source_identities) != 1 or "" in source_identities:
            raise ValueError("transparent baseline repair source identity is invalid")
        source_identity = next(iter(source_identities))
        is_optimizer_applicability_repair = receipt["contract_version"] == (
            PRE_RESULT_REPAIR_CONTRACT_VERSION_V2
        )
        if is_optimizer_applicability_repair:
            if (
                source_lineage != target_dataset_lineage_id
                or source_identity != target_dataset_identity_sha256
                or any(
                    str(item["dataset"]) != target_dataset
                    for item in receipt["members"]
                )
            ):
                raise ValueError(
                    "optimizer applicability repair changed the dataset or lineage"
                )
        elif source_lineage == target_dataset_lineage_id:
            raise ValueError("transparent baseline repair may not fabricate a fresh lineage")

        source_versions = cls._version_repair_rows(connection, source_version_ids)
        target_version_ids = {str(item["id"]) for item in target_versions}
        if len(target_version_ids) != 3:
            raise ValueError("transparent baseline repair target versions are incomplete")
        target_version_rows = cls._version_repair_rows(connection, target_version_ids)
        target_by_recipe: dict[str, Any] = {}
        for row in target_version_rows.values():
            config = dict(row.config_json or {})
            recipe_id = str(config.get("recipe_id") or "")
            if (
                recipe_id not in TRANSPARENT_RESEARCH_BASELINE_IDS
                or config.get("recipe_version") != receipt["target_recipe_version"]
            ):
                raise ValueError("transparent baseline repair target recipe changed")
            target_by_recipe[recipe_id] = row
        if set(target_by_recipe) != set(TRANSPARENT_RESEARCH_BASELINE_IDS):
            raise ValueError("transparent baseline repair target recipes are incomplete")

        source_backtests = connection.execute(
            select(
                backtest_runs,
                jobs.c.kind.label("job_kind"),
                jobs.c.status.label("job_status"),
                jobs.c.error.label("job_error"),
            )
            .join(jobs, jobs.c.id == backtest_runs.c.job_id)
            .where(backtest_runs.c.id.in_({str(item["backtest_id"]) for item in members.values()}))
        ).all()
        if len(source_backtests) != 3:
            raise ValueError("transparent baseline repair source backtests are incomplete")
        by_version = {str(row.strategy_version_id): row for row in source_backtests}
        if set(by_version) != source_version_ids:
            raise ValueError("transparent baseline repair source backtest versions changed")
        receipt_created_at = audit_row.created_at
        if receipt_created_at is None:
            raise ValueError("transparent baseline repair receipt timestamp is missing")
        later_results: list[str] = []
        failed_without_results: list[str] = []
        source_recipe_map = {
            str(link["recipe_id"]): str(
                dict(row.sealed_candidate_set_json or {})["strategy_version_id"]
            )
            for row, link in zip(source_rows, source_links, strict=True)
        }
        expected_by_recipe = {
            str(item["link"]["recipe_id"]): item for item in target_expected
        }
        if set(expected_by_recipe) != set(TRANSPARENT_RESEARCH_BASELINE_IDS):
            raise ValueError("transparent baseline repair target lockbox is incomplete")

        for recipe_id, source_version_id in source_recipe_map.items():
            member = members[source_version_id]
            backtest = by_version[source_version_id]
            source_version = source_versions[source_version_id]
            target_version = target_by_recipe[recipe_id]
            expected = expected_by_recipe[recipe_id]
            recorded_periods = dict(backtest.periods_json or {})
            source_config = dict(source_version.config_json or {})
            if (
                str(backtest.id) != member["backtest_id"]
                or str(backtest.job_id or "") != member["job_id"]
                or str(backtest.dataset) != member["dataset"]
                or recorded_periods != member["periods"]
                or str(backtest.job_kind) != "strategy_backtest"
                or backtest.created_at is None
                or backtest.created_at > receipt_created_at
                or str(source_version.horizon_profile) != str(expected["link"]["horizon_profile"])
                or str(target_version.horizon_profile) != str(source_version.horizon_profile)
                or str(target_version.strategy_id) != str(source_version.strategy_id)
                or str(target_version.benchmark) != str(source_version.benchmark)
                or str(target_version.universe) != str(source_version.universe)
                or str(target_version.economic_hypothesis_group)
                != str(source_version.economic_hypothesis_group)
                or _repair_economic_config(dict(target_version.config_json or {}))
                != _repair_economic_config(source_config)
                or str(expected["test_start"]) != member["periods"]["start"]
                or str(expected["test_end"]) != member["periods"]["end"]
            ):
                raise ValueError("transparent baseline repair changed an economic or OOS binding")
            if is_optimizer_applicability_repair and source_config.get(
                "recipe_version"
            ) != OPTIMIZER_APPLICABILITY_SOURCE_RECIPE_VERSION:
                raise ValueError("optimizer applicability repair source recipe changed")
            target_bootstrap = dict(
                dict(target_version.config_json or {}).get(BOOTSTRAP_CONFIG_KEY) or {}
            )
            if is_optimizer_applicability_repair and (
                target_bootstrap.get(TRANSPARENT_BASELINE_RUNNER_FIELD)
                != receipt[TRANSPARENT_BASELINE_RUNNER_FIELD]
            ):
                raise ValueError("optimizer applicability target runner changed")
            has_metrics = backtest.metrics_json is not None
            has_result = _artifact_result_exists(backtest.artifact_path)
            declared_files = sorted(member["files"], key=lambda item: item["path"])
            observed_files = _artifact_inventory(backtest.artifact_path)
            observed_by_path = {item["path"]: item for item in observed_files}
            if any(
                observed_by_path.get(item["path"]) != item for item in declared_files
            ):
                raise ValueError("transparent baseline repair artifact evidence changed")
            if member["status"] == "failed" and observed_files != declared_files:
                raise ValueError("transparent baseline repair failure artifacts changed")
            if has_metrics or has_result:
                if (
                    member["status"] != "running"
                    or backtest.finished_at is None
                    or not receipt_created_at < backtest.finished_at
                ):
                    raise ValueError(
                        "transparent baseline repair was registered after performance existed"
                    )
                later_results.append(str(backtest.id))
            elif member["status"] == "failed":
                marker = str(member.get("error") or "")
                if (
                    str(backtest.status) != "failed"
                    or str(backtest.job_status) != "failed"
                    or marker not in str(backtest.error or "")
                    or marker not in str(backtest.job_error or "")
                ):
                    raise ValueError("transparent baseline repair failure evidence changed")
                failed_without_results.append(str(backtest.id))
            else:
                # The running member was preregistered without performance. It
                # may still be running/cancelled/failed, but if it eventually
                # produced a result the branch above requires a later finish.
                if backtest.finished_at is not None and backtest.finished_at <= receipt_created_at:
                    raise ValueError(
                        "transparent baseline running member ended before preregistration"
                    )

        if not is_optimizer_applicability_repair and (
            str(target_dataset or "").strip()
            == str(receipt["members"][0]["dataset"])
        ):
            raise ValueError("transparent baseline repair target dataset was not rematerialized")
        return {
            "contract_version": PRE_RESULT_REPAIR_REGISTRY_VERSION,
            "receipt_sha256": str(receipt["receipt_sha256"]),
            "source_audit_event_id": int(audit_row.id),
            "source_batch_sha256": source_batch_sha256,
            "target_batch_sha256": target_batch_sha256,
            "source_dataset_lineage_id": source_lineage,
            "target_dataset": target_dataset,
            "target_dataset_identity_sha256": target_dataset_identity_sha256,
            "target_dataset_lineage_id": target_dataset_lineage_id,
            "target_recipe_version": str(receipt["target_recipe_version"]),
            "receipt_contract_version": str(receipt["contract_version"]),
            "repair_generation": receipt.get("repair_generation"),
            TRANSPARENT_BASELINE_RUNNER_FIELD: receipt.get(
                TRANSPARENT_BASELINE_RUNNER_FIELD
            ),
            "source_backtest_ids": sorted(item["backtest_id"] for item in members.values()),
            "target_strategy_version_ids": sorted(target_version_ids),
            "receipt_created_at": receipt_created_at.isoformat(),
            "failed_without_results": sorted(failed_without_results),
            "results_created_after_preregistration": sorted(later_results),
            "performance_information_used": False,
        }

    @classmethod
    def _register_pre_result_repair(
        cls,
        connection: Any,
        *,
        source_batch_sha256: str,
        source_rows: Sequence[Any],
        target_batch_sha256: str,
        target_versions: Sequence[Mapping[str, Any]],
        target_expected: Sequence[Mapping[str, Any]],
        target_dataset: str,
        target_dataset_identity_sha256: str,
        target_dataset_lineage_id: str,
    ) -> dict[str, Any]:
        existing = connection.execute(
            select(transparent_baseline_pre_result_repairs).where(
                transparent_baseline_pre_result_repairs.c.target_batch_sha256
                == target_batch_sha256
            )
        ).first()
        if existing is not None:
            if str(existing.source_batch_sha256) != source_batch_sha256:
                raise ValueError("transparent baseline repair registry target was rebound")
            return dict(existing.verification_json or {})
        matches: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
        for audit_row in connection.execute(
            select(audit_events)
            .where(audit_events.c.action == PRE_RESULT_REPAIR_ACTION)
            .order_by(audit_events.c.created_at)
            .with_for_update()
        ).all():
            try:
                receipt = validate_pre_result_repair_receipt(
                    dict(audit_row.details_json or {})
                )
                verification = cls._validate_repair_event_binding(
                    connection,
                    audit_row=audit_row,
                    receipt=receipt,
                    source_batch_sha256=source_batch_sha256,
                    source_rows=source_rows,
                    target_batch_sha256=target_batch_sha256,
                    target_versions=target_versions,
                    target_expected=target_expected,
                    target_dataset=target_dataset,
                    target_dataset_identity_sha256=target_dataset_identity_sha256,
                    target_dataset_lineage_id=target_dataset_lineage_id,
                )
            except ValueError:
                continue
            matches.append((audit_row, receipt, verification))
        if len(matches) != 1:
            raise ValueError(
                "transparent baseline final OOS has more than one prior batch or "
                "requires exactly one valid pre-result repair receipt"
            )
        audit_row, receipt, verification = matches[0]
        connection.execute(
            insert(transparent_baseline_pre_result_repairs).values(
                receipt_sha256=receipt["receipt_sha256"],
                source_audit_event_id=int(audit_row.id),
                source_batch_sha256=source_batch_sha256,
                target_batch_sha256=target_batch_sha256,
                source_dataset_lineage_id=verification[
                    "source_dataset_lineage_id"
                ],
                target_dataset_lineage_id=target_dataset_lineage_id,
                target_recipe_version=receipt["target_recipe_version"],
                source_backtest_ids_json=verification["source_backtest_ids"],
                target_strategy_version_ids_json=verification[
                    "target_strategy_version_ids"
                ],
                verification_json=verification,
                created_at=datetime.now(UTC),
            )
        )
        return verification

    def reserve(
        self,
        *,
        versions: Sequence[Mapping[str, Any]],
        dataset: str,
        dataset_identity_sha256: str,
        dataset_lineage_id: str,
    ) -> dict[str, Any]:
        if len(versions) != 3:
            raise ValueError("joint lockbox reservation requires exactly three versions")
        identity = _require_sha256(
            dataset_identity_sha256,
            field="dataset_identity_sha256",
        )
        lineage = _require_sha256(dataset_lineage_id, field="dataset_lineage_id")
        expected: list[dict[str, Any]] = []
        batch_ids: set[str] = set()
        for version in versions:
            config = version.get("config")
            if not isinstance(config, Mapping):
                raise ValueError("joint lockbox strategy config is invalid")
            lockbox = validate_joint_lockbox(config.get(LOCKBOX_CONFIG_KEY))
            if (
                lockbox["dataset"] != dataset
                or lockbox["dataset_identity_sha256"] != identity
                or lockbox["dataset_lineage_id"] != lineage
            ):
                raise ValueError("joint lockbox dataset binding changed")
            batch_ids.add(str(lockbox["batch_sha256"]))
            member_set = baseline_oos_sealed_member_set(version)
            link = validate_lockbox_link(member_set["transparent_baseline_lockbox"])
            bootstrap = config.get(BOOTSTRAP_CONFIG_KEY)
            if not isinstance(bootstrap, Mapping):
                raise ValueError("joint lockbox member has no bootstrap contract")
            periods = dict(bootstrap.get("formal_periods") or {})
            expected.append(
                {
                    "version_id": str(version["id"]),
                    "test_start": date.fromisoformat(str(periods["start"])),
                    "test_end": date.fromisoformat(str(periods["end"])),
                    "sealed_member_set": member_set,
                    "sealed_member_set_sha256": canonical_sha256(member_set),
                    "link": link,
                }
            )
        if len(batch_ids) != 1:
            raise ValueError("public baseline versions do not share one joint lockbox")
        observed_links = {item["link"]["member_sha256"] for item in expected}
        declared_links = set(expected[0]["link"]["member_sha256s"])
        if observed_links != declared_links:
            raise ValueError("joint lockbox versions do not cover all declared members")
        target_batch_sha256 = next(iter(batch_ids))
        base_scope = f"lineage:{lineage}"
        scope = base_scope
        earliest = min(item["test_start"] for item in expected)
        latest = max(item["test_end"] for item in expected)
        now = datetime.now(UTC)
        repair_registration: dict[str, Any] | None = None
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:oos_scope))"),
                {"oos_scope": base_scope},
            )
            # A changed builder/eligibility contract produces a real new
            # lineage, but lineage churn is not permission to look at the same
            # final-OOS window twice. Scan all prior transparent lockboxes.
            # The only exception is a hashed receipt recorded before any
            # performance result and consumed into the append-only registry in
            # this same transaction.
            all_overlapping_rows = connection.execute(
                select(oos_vintages)
                .where(
                    oos_vintages.c.test_start <= latest,
                    oos_vintages.c.test_end >= earliest,
                )
                .with_for_update()
            ).all()
            registered_repairs = connection.execute(
                select(transparent_baseline_pre_result_repairs)
            ).all()
            superseded_source_batches = {
                str(row.source_batch_sha256) for row in registered_repairs
            }
            existing_target_repairs = [
                row
                for row in registered_repairs
                if str(row.target_batch_sha256) == target_batch_sha256
            ]
            if len(existing_target_repairs) > 1:
                raise ValueError("transparent baseline repair target is duplicated")
            if existing_target_repairs:
                repair_registration = dict(
                    existing_target_repairs[0].verification_json or {}
                )
            source_batches = self._repair_source_batches(
                all_overlapping_rows,
                target_batch_sha256=target_batch_sha256,
                superseded_source_batches=superseded_source_batches,
            )
            if source_batches and repair_registration is None:
                if len(source_batches) != 1:
                    raise ValueError(
                        "transparent baseline final OOS already has more than one prior batch"
                    )
                source_batch_sha256, source_rows = next(iter(source_batches.items()))
                repair_registration = self._register_pre_result_repair(
                    connection,
                    source_batch_sha256=source_batch_sha256,
                    source_rows=source_rows,
                    target_batch_sha256=target_batch_sha256,
                    target_versions=versions,
                    target_expected=expected,
                    target_dataset=dataset,
                    target_dataset_identity_sha256=identity,
                    target_dataset_lineage_id=lineage,
                )
            if repair_registration is not None and repair_registration.get(
                "repair_generation"
            ) == OPTIMIZER_APPLICABILITY_REPAIR_GENERATION:
                # The v7 and v8 attempts intentionally share the exact dataset
                # lineage and OOS dates. Use a repair-specific scope so v8
                # receives new immutable one-shot rows instead of mutating or
                # reusing the consumed v7 rows.
                scope = f"{base_scope}:repair:{target_batch_sha256}"
            scope_filter = oos_vintages.c.scope == scope
            if scope == base_scope:
                scope_filter = or_(
                    scope_filter,
                    oos_vintages.c.scope.like("dataset:%"),
                )
            rows = connection.execute(
                select(oos_vintages)
                .where(
                    scope_filter,
                    oos_vintages.c.test_start <= latest,
                    oos_vintages.c.test_end >= earliest,
                )
                .with_for_update()
            ).all()
            if rows:
                expected_by_window = {
                    (item["test_start"], item["test_end"]): item for item in expected
                }
                if len(rows) != 3:
                    raise ValueError(
                        "joint lockbox overlaps another reserved or consumed OOS vintage"
                    )
                for row in rows:
                    item = expected_by_window.get((row.test_start, row.test_end))
                    if (
                        item is None
                        or str(row.scope) != scope
                        or str(row.dataset_identity) != identity
                        or str(row.dataset_lineage_id or "") != lineage
                        or dict(row.sealed_candidate_set_json or {})
                        != item["sealed_member_set"]
                        or str(row.sealed_candidate_set_sha256)
                        != item["sealed_member_set_sha256"]
                    ):
                        raise ValueError(
                            "existing OOS vintages differ from the joint lockbox"
                        )
            else:
                for item in expected:
                    connection.execute(
                        insert(oos_vintages).values(
                            id=uuid.uuid4().hex,
                            scope=scope,
                            dataset_identity=identity,
                            dataset_lineage_id=lineage,
                            test_start=item["test_start"],
                            test_end=item["test_end"],
                            sealed_at=now,
                            first_opened_at=now,
                            consumed_at=None,
                            capital_oos_alpha_batch_id=None,
                            sealed_candidate_set_json=item["sealed_member_set"],
                            sealed_candidate_set_sha256=item[
                                "sealed_member_set_sha256"
                            ],
                            created_at=now,
                        )
                    )
            recorded = connection.execute(
                select(oos_vintages).where(
                    oos_vintages.c.scope == scope,
                    oos_vintages.c.test_start <= latest,
                    oos_vintages.c.test_end >= earliest,
                )
            ).all()
        members = []
        for row in sorted(recorded, key=lambda item: item.test_start):
            sealed = dict(row.sealed_candidate_set_json or {})
            link = validate_lockbox_link(sealed.get("transparent_baseline_lockbox"))
            members.append(
                {
                    "oos_vintage_id": str(row.id),
                    "strategy_version_id": str(
                        sealed.get("strategy_version_id") or ""
                    ),
                    "recipe_id": link["recipe_id"],
                    "horizon_profile": link["horizon_profile"],
                    "test_start": row.test_start.isoformat(),
                    "test_end": row.test_end.isoformat(),
                    "status": "consumed" if row.consumed_at is not None else "reserved",
                    "consumed_at": row.consumed_at,
                }
            )
        return {
            "contract_version": LOCKBOX_CONTRACT_VERSION,
            "batch_sha256": target_batch_sha256,
            "scope": scope,
            "dataset": dataset,
            "dataset_identity_sha256": identity,
            "dataset_lineage_id": lineage,
            "pre_result_repair": repair_registration,
            "members": members,
        }

    def get(self, batch_sha256: str) -> dict[str, Any]:
        batch = _require_sha256(batch_sha256, field="batch_sha256")
        with self.engine.connect() as connection:
            rows = connection.execute(select(oos_vintages)).all()
        matches = []
        for row in rows:
            sealed = dict(row.sealed_candidate_set_json or {})
            raw_link = sealed.get("transparent_baseline_lockbox")
            if not isinstance(raw_link, Mapping) or raw_link.get("batch_sha256") != batch:
                continue
            link = validate_lockbox_link(raw_link)
            matches.append(
                {
                    **row_dict(row),
                    "recipe_id": link["recipe_id"],
                    "horizon_profile": link["horizon_profile"],
                }
            )
        if len(matches) != 3:
            raise KeyError(batch)
        return {"batch_sha256": batch, "members": matches}
