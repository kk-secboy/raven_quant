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
from typing import Any

from sqlalchemy import insert, or_, select, text

from quant_data.database import oos_vintages, open_database, row_dict
from quant_platform.strategy_recipes import TRANSPARENT_RESEARCH_BASELINE_IDS

LOCKBOX_CONTRACT_VERSION = "transparent-baseline-joint-lockbox-v1"
LOCKBOX_LINK_VERSION = "transparent-baseline-joint-lockbox-link-v1"
LOCKBOX_CONFIG_KEY = "transparent_baseline_joint_lockbox"
BOOTSTRAP_CONFIG_KEY = "transparent_baseline_bootstrap"

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


def _normalize_member(raw: Mapping[str, Any]) -> dict[str, str]:
    if set(raw) != _MEMBER_KEYS:
        raise ValueError("transparent baseline lockbox member fields are invalid")
    recipe_id = str(raw.get("recipe_id") or "").strip()
    if recipe_id not in _RECIPE_HORIZONS:
        raise ValueError("transparent baseline lockbox contains an unknown recipe")
    horizon = str(raw.get("horizon_profile") or "").strip()
    if horizon != _RECIPE_HORIZONS[recipe_id]:
        raise ValueError("transparent baseline lockbox recipe horizon changed")
    recipe_version = str(raw.get("recipe_version") or "").strip()
    if not recipe_version:
        raise ValueError("transparent baseline lockbox recipe version is required")
    try:
        historical_start = date.fromisoformat(str(raw["historical_start"]))
        historical_end = date.fromisoformat(str(raw["historical_end"]))
        test_start = date.fromisoformat(str(raw["test_start"]))
        test_end = date.fromisoformat(str(raw["test_end"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transparent baseline lockbox periods are invalid") from exc
    if not historical_start <= historical_end < test_start <= test_end:
        raise ValueError("transparent baseline lockbox periods are not ordered")
    return {
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
    return _normalize_member(
        {
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
    )


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


class TransparentBaselineLockboxStore:
    """Atomically reserve/recover the three public baseline OOS vintages."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

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
        scope = f"lineage:{lineage}"
        earliest = min(item["test_start"] for item in expected)
        latest = max(item["test_end"] for item in expected)
        now = datetime.now(UTC)
        with self.engine.begin() as connection:
            connection.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:oos_scope))"),
                {"oos_scope": scope},
            )
            scope_filter = or_(
                oos_vintages.c.scope == scope,
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
            "batch_sha256": next(iter(batch_ids)),
            "scope": scope,
            "dataset": dataset,
            "dataset_identity_sha256": identity,
            "dataset_lineage_id": lineage,
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
