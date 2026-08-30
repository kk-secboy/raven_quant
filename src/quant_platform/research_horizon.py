"""Canonical research, decision and holding-horizon contracts.

The amount of history used to estimate a model is not its prediction horizon,
and neither value is a portfolio's holding period.  This module keeps those
concepts explicit and supplies the only supported product horizon profiles.

``legacy_ambiguous`` is intentionally non-executable as a research contract:
it records that an older StrategyVersion did not preserve enough information
to reconstruct its true horizon.  We never guess that history during a schema
migration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

RESEARCH_HORIZON_CONTRACT_VERSION = "research-horizon-v1"

SHORT_1_5D = "short_1_5d"
SWING_1_6M = "swing_1_6m"
LONG_1_3Y = "long_1_3y"
LEGACY_AMBIGUOUS = "legacy_ambiguous"

# This policy is intentionally versioned independently from
# ``research-horizon-v1``.  Changing which modelling label is primary must not
# silently rewrite the older horizon contract digests embedded in historical
# evidence.
PRIMARY_LABEL_POLICY_CONTRACT_VERSION = "primary-label-policy-v1"
_PRIMARY_LABEL_HORIZON_SESSIONS = {
    SHORT_1_5D: 5,
    SWING_1_6M: 63,
    LONG_1_3Y: 252,
}

SUPPORTED_HORIZON_PROFILES = (
    SHORT_1_5D,
    SWING_1_6M,
    LONG_1_3Y,
    LEGACY_AMBIGUOUS,
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResearchHorizonContract:
    """One immutable strategy research and operating horizon.

    All durations are exchange trading sessions, never calendar days.  The
    sealed OOS duration is a minimum reservation.  ``purge`` removes labels
    whose forward-return window crosses an adjacent fold; ``embargo`` isolates
    the final sealed OOS from every research observation.
    """

    horizon_profile: str
    label_horizons_sessions: tuple[int, ...]
    decision_interval_sessions: int | None
    review_interval_sessions: int | None
    holding_min_sessions: int | None
    holding_target_sessions: int | None
    holding_max_sessions: int | None
    execution_lag_sessions: int | None
    purge_sessions: int | None
    embargo_sessions: int | None
    sealed_oos_required: bool
    sealed_oos_sessions: int | None
    contract_version: str = RESEARCH_HORIZON_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != RESEARCH_HORIZON_CONTRACT_VERSION:
            raise ValueError("research horizon contract version is unsupported")
        if self.horizon_profile not in SUPPORTED_HORIZON_PROFILES:
            raise ValueError(f"unsupported horizon profile: {self.horizon_profile}")
        if self.horizon_profile == LEGACY_AMBIGUOUS:
            unknown = (
                self.decision_interval_sessions,
                self.review_interval_sessions,
                self.holding_min_sessions,
                self.holding_target_sessions,
                self.holding_max_sessions,
                self.execution_lag_sessions,
                self.purge_sessions,
                self.embargo_sessions,
                self.sealed_oos_sessions,
            )
            if self.label_horizons_sessions or any(item is not None for item in unknown):
                raise ValueError("legacy_ambiguous cannot invent missing horizon values")
            if self.sealed_oos_required:
                raise ValueError("legacy_ambiguous cannot claim a sealed OOS contract")
            return

        labels = self.label_horizons_sessions
        if (
            not labels
            or labels != tuple(sorted(set(labels)))
            or any(isinstance(item, bool) or item <= 0 for item in labels)
        ):
            raise ValueError("label horizons must be unique increasing positive sessions")
        required = (
            self.decision_interval_sessions,
            self.review_interval_sessions,
            self.holding_min_sessions,
            self.holding_target_sessions,
            self.holding_max_sessions,
            self.execution_lag_sessions,
            self.purge_sessions,
            self.embargo_sessions,
            self.sealed_oos_sessions,
        )
        if any(
            item is None or isinstance(item, bool) or item <= 0 for item in required
        ):
            raise ValueError("active horizon durations must be positive trading sessions")
        assert self.review_interval_sessions is not None
        assert self.decision_interval_sessions is not None
        if self.review_interval_sessions > self.decision_interval_sessions:
            raise ValueError("risk review cannot be less frequent than portfolio decisions")
        assert self.holding_min_sessions is not None
        assert self.holding_target_sessions is not None
        assert self.holding_max_sessions is not None
        if not (
            self.holding_min_sessions
            <= self.holding_target_sessions
            <= self.holding_max_sessions
        ):
            raise ValueError("holding sessions must satisfy min <= target <= max")
        if max(labels) > self.holding_max_sessions:
            raise ValueError("a label horizon cannot exceed the maximum holding horizon")
        assert self.purge_sessions is not None
        assert self.embargo_sessions is not None
        if self.purge_sessions < max(labels):
            raise ValueError("purge must cover the longest forward label")
        if self.embargo_sessions < max(labels):
            raise ValueError("embargo must isolate the longest label from sealed OOS")
        assert self.sealed_oos_sessions is not None
        if not self.sealed_oos_required or self.sealed_oos_sessions < self.holding_max_sessions:
            raise ValueError("active horizons require sealed OOS covering a full holding cycle")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["label_horizons_sessions"] = list(self.label_horizons_sessions)
        return value

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def strategy_version_columns(self) -> dict[str, Any]:
        """Return the denormalized immutable StrategyVersion columns."""

        return {
            "horizon_profile": self.horizon_profile,
            "label_horizons_json": list(self.label_horizons_sessions),
            "decision_interval_sessions": self.decision_interval_sessions,
            "review_interval_sessions": self.review_interval_sessions,
            "holding_min_sessions": self.holding_min_sessions,
            "holding_target_sessions": self.holding_target_sessions,
            "holding_max_sessions": self.holding_max_sessions,
            "execution_lag_sessions": self.execution_lag_sessions,
            "purge_sessions": self.purge_sessions,
            "embargo_sessions": self.embargo_sessions,
            "sealed_oos_required": self.sealed_oos_required,
            "sealed_oos_sessions": self.sealed_oos_sessions,
            "horizon_contract_json": self.to_dict(),
            "horizon_contract_sha256": self.sha256,
        }


_PROFILE_CONTRACTS = {
    SHORT_1_5D: ResearchHorizonContract(
        horizon_profile=SHORT_1_5D,
        label_horizons_sessions=(1, 2, 3, 5),
        decision_interval_sessions=1,
        review_interval_sessions=1,
        holding_min_sessions=1,
        holding_target_sessions=3,
        holding_max_sessions=5,
        execution_lag_sessions=1,
        purge_sessions=6,
        embargo_sessions=6,
        sealed_oos_required=True,
        sealed_oos_sessions=252,
    ),
    SWING_1_6M: ResearchHorizonContract(
        horizon_profile=SWING_1_6M,
        label_horizons_sessions=(21, 63, 126),
        decision_interval_sessions=5,
        review_interval_sessions=5,
        holding_min_sessions=21,
        holding_target_sessions=63,
        holding_max_sessions=126,
        execution_lag_sessions=1,
        purge_sessions=127,
        embargo_sessions=127,
        sealed_oos_required=True,
        sealed_oos_sessions=504,
    ),
    LONG_1_3Y: ResearchHorizonContract(
        horizon_profile=LONG_1_3Y,
        label_horizons_sessions=(63, 126, 252),
        decision_interval_sessions=21,
        review_interval_sessions=21,
        holding_min_sessions=252,
        holding_target_sessions=504,
        holding_max_sessions=756,
        execution_lag_sessions=1,
        purge_sessions=253,
        embargo_sessions=253,
        sealed_oos_required=True,
        sealed_oos_sessions=756,
    ),
    LEGACY_AMBIGUOUS: ResearchHorizonContract(
        horizon_profile=LEGACY_AMBIGUOUS,
        label_horizons_sessions=(),
        decision_interval_sessions=None,
        review_interval_sessions=None,
        holding_min_sessions=None,
        holding_target_sessions=None,
        holding_max_sessions=None,
        execution_lag_sessions=None,
        purge_sessions=None,
        embargo_sessions=None,
        sealed_oos_required=False,
        sealed_oos_sessions=None,
    ),
}

def research_horizon_contract(profile: str) -> ResearchHorizonContract:
    try:
        return _PROFILE_CONTRACTS[str(profile)]
    except KeyError as exc:
        raise ValueError(f"unsupported horizon profile: {profile}") from exc


def primary_label_horizon_sessions(profile: str) -> int:
    """Return the single prediction target used to compare one horizon.

    A horizon may expose auxiliary labels for diagnostics, but its factor and
    model champion must have one stable comparison target.  This mapping is a
    derived product rule and intentionally does not change the immutable
    ``ResearchHorizonContract`` digest stored by existing StrategyVersions.
    """

    contract = research_horizon_contract(profile)
    if contract.horizon_profile == LEGACY_AMBIGUOUS:
        raise ValueError("legacy_ambiguous has no primary prediction label")
    selected = _PRIMARY_LABEL_HORIZON_SESSIONS[contract.horizon_profile]
    if selected not in contract.label_horizons_sessions:
        raise ValueError("primary prediction label differs from its horizon contract")
    return selected


def primary_label_policy_contract() -> dict[str, Any]:
    """Return the immutable policy that selects one comparison label per horizon."""

    body: dict[str, Any] = {
        "contract_version": PRIMARY_LABEL_POLICY_CONTRACT_VERSION,
        "horizon_primary_labels_sessions": dict(
            sorted(_PRIMARY_LABEL_HORIZON_SESSIONS.items())
        ),
        "legacy_ambiguous_executable": False,
    }
    return {**body, "policy_sha256": canonical_sha256(body)}


def primary_label_policy_sha256() -> str:
    return str(primary_label_policy_contract()["policy_sha256"])


def research_cadence_bucket(horizon_profile: str, dataset_end_date: str) -> str:
    """Return the governed research event owning one daily publication."""

    try:
        session = date.fromisoformat(str(dataset_end_date))
    except ValueError as exc:
        raise ValueError("research dataset end date is invalid") from exc
    if horizon_profile == SHORT_1_5D:
        iso_year, iso_week, _ = session.isocalendar()
        return f"week:{iso_year:04d}-{iso_week:02d}"
    if horizon_profile == SWING_1_6M:
        return f"month:{session.year:04d}-{session.month:02d}"
    if horizon_profile == LONG_1_3Y:
        return f"quarter:{session.year:04d}-Q{((session.month - 1) // 3) + 1}"
    raise ValueError(f"unsupported automatic research horizon: {horizon_profile}")


def normalize_horizon_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a config to one canonical horizon without guessing old semantics.

    Existing callers that predate this contract become ``legacy_ambiguous``.
    New production callers opt into a supported profile by setting only
    ``horizon_profile``; submitted contract copies/hashes are accepted solely
    when they exactly match the canonical profile.
    """

    normalized = dict(config)
    profile = str(normalized.get("horizon_profile") or LEGACY_AMBIGUOUS)
    contract = research_horizon_contract(profile)
    submitted = normalized.get("horizon_contract")
    if submitted is not None and submitted != contract.to_dict():
        raise ValueError("submitted horizon contract differs from the canonical profile")
    submitted_sha256 = normalized.get("horizon_contract_sha256")
    if submitted_sha256 is not None and submitted_sha256 != contract.sha256:
        raise ValueError("submitted horizon contract SHA-256 is invalid")
    normalized.update(
        {
            "horizon_profile": contract.horizon_profile,
            "horizon_contract": contract.to_dict(),
            "horizon_contract_sha256": contract.sha256,
        }
    )
    return normalized


def horizon_columns_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_horizon_config(config)
    contract = research_horizon_contract(str(normalized["horizon_profile"]))
    return contract.strategy_version_columns()


def require_horizon_row(row: Mapping[str, Any]) -> ResearchHorizonContract:
    """Fail closed when denormalized StrategyVersion horizon data drifts."""

    contract = research_horizon_contract(str(row.get("horizon_profile") or ""))
    expected = contract.strategy_version_columns()
    for key, value in expected.items():
        observed = row.get(key)
        if key == "label_horizons_json" and observed is not None:
            observed = list(observed)
        if observed != value:
            raise ValueError(f"strategy horizon column {key} differs from its sealed contract")
    return contract


def require_label_horizon(profile: str, label_horizon_sessions: int) -> None:
    contract = research_horizon_contract(profile)
    if contract.horizon_profile == LEGACY_AMBIGUOUS:
        raise ValueError("legacy_ambiguous has no admissible label horizon")
    if label_horizon_sessions not in contract.label_horizons_sessions:
        raise ValueError(
            f"label horizon {label_horizon_sessions} is not allowed for {profile}"
        )
