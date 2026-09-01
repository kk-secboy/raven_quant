"""Versioned investor choices for the governed simulation account."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from math import isfinite
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from quant_data.database import (
    investor_simulation_profiles,
    open_database,
    row_dict,
)
from quant_platform.market_rules import (
    BOARD_BSE,
    BOARD_CHINEXT,
    BOARD_FUND,
    BOARD_SH_MAIN,
    BOARD_STAR,
    BOARD_SZ_MAIN,
    order_unit_rules,
)
from quant_platform.research_horizon import canonical_sha256

INVESTOR_SIMULATION_PROFILE_VERSION = "investor-simulation-profile-v1"
INVESTOR_PROFILE_BINDING_VERSION = "paper-investor-profile-binding-v1"
RISK_PROFILES = frozenset({"conservative", "balanced", "aggressive", "custom"})
MARKET_PERMISSION_KEYS = frozenset(
    {"main_board", "star_market", "chi_next", "beijing_exchange", "etf"}
)
_BOARD_PERMISSION_KEY = {
    BOARD_SH_MAIN: "main_board",
    BOARD_SZ_MAIN: "main_board",
    BOARD_STAR: "star_market",
    BOARD_CHINEXT: "chi_next",
    BOARD_BSE: "beijing_exchange",
    BOARD_FUND: "etf",
}


def _binding_payload(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable investor fields carried by one paper account."""

    permissions = binding.get("market_permissions")
    if not isinstance(permissions, Mapping) or set(permissions) != MARKET_PERMISSION_KEYS:
        raise ValueError("paper investor-profile binding has incomplete market permissions")
    if any(not isinstance(value, bool) for value in permissions.values()):
        raise ValueError("paper investor-profile permissions must be explicit booleans")
    profile_id = str(binding.get("profile_id") or "").strip()
    profile_key = str(binding.get("profile_key") or "").strip()
    profile_sha256 = str(binding.get("profile_content_sha256") or "").lower()
    try:
        profile_version = int(binding.get("profile_version"))
    except (TypeError, ValueError) as exc:
        raise ValueError("paper investor-profile binding has no valid version") from exc
    if (
        not profile_id
        or not profile_key
        or profile_version < 1
        or len(profile_sha256) != 64
        or any(character not in "0123456789abcdef" for character in profile_sha256)
    ):
        raise ValueError("paper investor-profile binding identity is invalid")
    return {
        "contract_version": INVESTOR_PROFILE_BINDING_VERSION,
        "profile_id": profile_id,
        "profile_key": profile_key,
        "profile_version": profile_version,
        "profile_content_sha256": profile_sha256,
        "market_permissions": {
            key: bool(permissions[key]) for key in sorted(MARKET_PERMISSION_KEYS)
        },
    }


def bind_investor_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze one validated profile version into an isolated paper account."""

    payload = _binding_payload(
        {
            "profile_id": profile.get("id"),
            "profile_key": profile.get("profile_key"),
            "profile_version": profile.get("version"),
            "profile_content_sha256": profile.get("content_sha256"),
            "market_permissions": profile.get("market_permissions"),
        }
    )
    return {**payload, "binding_sha256": canonical_sha256(payload)}


def validate_investor_profile_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the profile binding stored in execution policy."""

    payload = _binding_payload(binding)
    if str(binding.get("contract_version") or "") != INVESTOR_PROFILE_BINDING_VERSION:
        raise ValueError("paper investor-profile binding version is unsupported")
    if str(binding.get("binding_sha256") or "").lower() != canonical_sha256(payload):
        raise ValueError("paper investor-profile binding failed immutable verification")
    return {**payload, "binding_sha256": canonical_sha256(payload)}


def require_matching_active_profile_binding(
    binding: Mapping[str, Any],
    active_profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed when a paper account's investor profile is no longer active."""

    frozen = validate_investor_profile_binding(binding)
    if active_profile is None:
        raise ValueError(
            "paper investor profile is no longer active; forward evidence is blocked"
        )
    current = bind_investor_profile(active_profile)
    if current != frozen:
        raise ValueError(
            "active investor profile changed; the existing paper account cannot "
            "continue forward evidence"
        )
    return frozen


def investor_profile_permission(
    profile: Mapping[str, Any], instrument: str, *, on_date: date
) -> dict[str, Any]:
    """Resolve the first-run permission that governs a prospective buy.

    The profile is the product-level source of truth for the five choices the
    novice explicitly makes.  Existing positions may always be reduced or
    exited; callers use this result only when a target would add exposure.
    Unknown instruments fail closed instead of being guessed into main board.
    """

    permissions = profile.get("market_permissions")
    if not isinstance(permissions, Mapping) or set(permissions) != MARKET_PERMISSION_KEYS:
        return {
            "allowed": False,
            "permission_key": None,
            "reason": "investor_profile_permissions_incomplete",
        }
    try:
        rules = order_unit_rules(instrument, on_date)
    except ValueError:
        return {
            "allowed": False,
            "permission_key": None,
            "reason": "instrument_permission_scope_unknown",
        }
    permission_key = _BOARD_PERMISSION_KEY.get(rules.board)
    if permission_key is None:
        return {
            "allowed": False,
            "permission_key": None,
            "reason": "instrument_permission_scope_unknown",
        }
    allowed = permissions.get(permission_key) is True
    return {
        "allowed": allowed,
        "permission_key": permission_key,
        "reason": None if allowed else f"investor_permission_disabled:{permission_key}",
    }


def investor_profile_permission_map(
    profile: Mapping[str, Any],
    instruments: list[str] | set[str] | tuple[str, ...],
    *,
    on_date: date,
) -> dict[str, dict[str, Any]]:
    """Resolve deterministic permission evidence for a set of instruments."""

    return {
        instrument: investor_profile_permission(profile, instrument, on_date=on_date)
        for instrument in sorted({str(item).upper() for item in instruments})
    }


def require_investor_profile_target_permissions(
    binding: Mapping[str, Any],
    target_weights: Mapping[str, float],
    previous_weights: Mapping[str, float],
    *,
    on_date: date,
) -> dict[str, dict[str, Any]]:
    """Reject only prohibited exposure increases; reductions and exits stay valid."""

    profile = validate_investor_profile_binding(binding)
    normalized_targets = {
        str(instrument).upper(): float(weight)
        for instrument, weight in target_weights.items()
    }
    normalized_previous = {
        str(instrument).upper(): float(weight)
        for instrument, weight in previous_weights.items()
    }
    evidence = investor_profile_permission_map(
        profile,
        set(normalized_targets),
        on_date=on_date,
    )
    for instrument, target_weight in normalized_targets.items():
        if target_weight <= normalized_previous.get(instrument, 0.0) + 1e-12:
            continue
        permission = evidence[instrument]
        if permission["allowed"] is not True:
            raise ValueError(
                "paper target attempted an exposure increase outside the frozen "
                f"investor permissions: {instrument}:{permission['reason']}"
            )
    return evidence


def _now() -> datetime:
    return datetime.now(UTC)


def _profile_content(
    *,
    profile_key: str,
    version: int,
    supersedes_id: str | None,
    initial_capital: Decimal,
    risk_profile: str,
    min_cash_weight: float,
    max_gross_exposure: float,
    market_permissions: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract_version": INVESTOR_SIMULATION_PROFILE_VERSION,
        "profile_key": profile_key,
        "version": version,
        "supersedes_id": supersedes_id,
        "initial_capital": format(initial_capital, "f"),
        "risk_profile": risk_profile,
        "min_cash_weight": min_cash_weight,
        "max_gross_exposure": max_gross_exposure,
        "market_permissions": dict(market_permissions),
    }


class InvestorSimulationProfileStore:
    """Own explicit, versioned personal-capital and permission choices."""

    def __init__(self, database_url: str) -> None:
        self.engine = open_database(database_url)

    def create_version(
        self,
        *,
        profile_key: str,
        initial_capital: Decimal | float | int | str,
        risk_profile: str,
        min_cash_weight: float,
        max_gross_exposure: float,
        market_permissions: Mapping[str, Any],
        actor: str,
        activate: bool = False,
    ) -> dict[str, Any]:
        """Create a new immutable-choice version; capital has no default."""

        key = profile_key.strip()
        owner = actor.strip()
        risk = risk_profile.strip().lower()
        if not key or len(key) > 100:
            raise ValueError("profile_key must contain 1 to 100 characters")
        if len(owner) < 2:
            raise ValueError("a responsible investor-profile actor is required")
        if risk not in RISK_PROFILES:
            raise ValueError(f"unsupported investor risk profile: {risk_profile}")
        try:
            capital = Decimal(str(initial_capital))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("initial capital must be a positive finite amount") from exc
        if (
            not capital.is_finite()
            or capital <= 0
            or capital > Decimal("99999999999999.999999")
        ):
            raise ValueError("initial capital must be a positive finite amount")
        try:
            capital = capital.quantize(Decimal("0.000001"))
        except InvalidOperation as exc:
            raise ValueError("initial capital must fit the governed account precision") from exc
        cash_weight = float(min_cash_weight)
        gross = float(max_gross_exposure)
        if not isfinite(cash_weight) or not 0 <= cash_weight < 1:
            raise ValueError("minimum cash weight must be in [0, 1)")
        if not isfinite(gross) or not 0 < gross <= 1:
            raise ValueError("maximum gross exposure must be in (0, 1]")
        if cash_weight + gross > 1.0 + 1e-12:
            raise ValueError("minimum cash plus maximum gross exposure cannot exceed one")
        permissions = dict(market_permissions)
        if set(permissions) != MARKET_PERMISSION_KEYS or any(
            not isinstance(value, bool) for value in permissions.values()
        ):
            raise ValueError(
                "explicit boolean permissions are required for every supported market"
            )

        profile_id = uuid.uuid4().hex
        now = _now()
        try:
            with self.engine.begin() as connection:
                rows = connection.execute(
                    select(investor_simulation_profiles)
                    .where(investor_simulation_profiles.c.profile_key == key)
                    .order_by(investor_simulation_profiles.c.version.desc())
                    .with_for_update()
                ).all()
                latest = rows[0] if rows else None
                version = int(latest.version) + 1 if latest is not None else 1
                supersedes_id = str(latest.id) if latest is not None else None
                content = _profile_content(
                    profile_key=key,
                    version=version,
                    supersedes_id=supersedes_id,
                    initial_capital=capital,
                    risk_profile=risk,
                    min_cash_weight=cash_weight,
                    max_gross_exposure=gross,
                    market_permissions=permissions,
                )
                if activate:
                    connection.execute(
                        update(investor_simulation_profiles)
                        .where(
                            investor_simulation_profiles.c.profile_key == key,
                            investor_simulation_profiles.c.status == "active",
                        )
                        .values(status="retired", updated_by=owner, updated_at=now)
                    )
                connection.execute(
                    insert(investor_simulation_profiles).values(
                        id=profile_id,
                        profile_key=key,
                        version=version,
                        status="active" if activate else "draft",
                        supersedes_id=supersedes_id,
                        initial_capital=capital,
                        risk_profile=risk,
                        min_cash_weight=cash_weight,
                        max_gross_exposure=gross,
                        market_permissions_json=permissions,
                        content_sha256=canonical_sha256(content),
                        created_by=owner,
                        created_at=now,
                        updated_by=owner,
                        updated_at=now,
                    )
                )
        except IntegrityError as exc:
            raise ValueError("investor profile version conflicted with another writer") from exc
        return self.get(profile_id)

    def activate(self, profile_id: str, *, actor: str) -> dict[str, Any]:
        owner = actor.strip()
        if len(owner) < 2:
            raise ValueError("a responsible investor-profile actor is required")
        with self.engine.begin() as connection:
            profile = connection.execute(
                select(investor_simulation_profiles)
                .where(investor_simulation_profiles.c.id == profile_id)
                .with_for_update()
            ).first()
            if profile is None:
                raise KeyError(profile_id)
            if str(profile.status) not in {"active", "draft"}:
                raise ValueError("only a draft investor profile can be activated")
            if str(profile.status) == "draft":
                now = _now()
                connection.execute(
                    update(investor_simulation_profiles)
                    .where(
                        investor_simulation_profiles.c.profile_key == profile.profile_key,
                        investor_simulation_profiles.c.status == "active",
                    )
                    .values(status="retired", updated_by=owner, updated_at=now)
                )
                connection.execute(
                    update(investor_simulation_profiles)
                    .where(investor_simulation_profiles.c.id == profile_id)
                    .values(status="active", updated_by=owner, updated_at=now)
                )
        return self.get(profile_id)

    def get(self, profile_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(investor_simulation_profiles).where(
                    investor_simulation_profiles.c.id == profile_id
                )
            ).first()
        if row is None:
            raise KeyError(profile_id)
        result = row_dict(row)
        content = _profile_content(
            profile_key=str(row.profile_key),
            version=int(row.version),
            supersedes_id=str(row.supersedes_id) if row.supersedes_id else None,
            initial_capital=Decimal(row.initial_capital),
            risk_profile=str(row.risk_profile),
            min_cash_weight=float(row.min_cash_weight),
            max_gross_exposure=float(row.max_gross_exposure),
            market_permissions=dict(row.market_permissions_json or {}),
        )
        if canonical_sha256(content) != str(row.content_sha256):
            raise ValueError("investor simulation profile content seal is invalid")
        result["market_permissions"] = dict(result.pop("market_permissions_json"))
        return result

    def get_active(self, profile_key: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            profile_id = connection.scalar(
                select(investor_simulation_profiles.c.id).where(
                    investor_simulation_profiles.c.profile_key == profile_key.strip(),
                    investor_simulation_profiles.c.status == "active",
                )
            )
        return self.get(str(profile_id)) if profile_id else None

    def list_versions(self, profile_key: str, *, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1000:
            raise ValueError("investor profile limit must be between 1 and 1000")
        with self.engine.connect() as connection:
            ids = connection.execute(
                select(investor_simulation_profiles.c.id)
                .where(investor_simulation_profiles.c.profile_key == profile_key.strip())
                .order_by(investor_simulation_profiles.c.version.desc())
                .limit(limit)
            ).scalars().all()
        return [self.get(str(profile_id)) for profile_id in ids]
