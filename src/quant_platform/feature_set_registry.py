from __future__ import annotations

from typing import Any

from .factor_library import (
    FACTOR_DEFINITIONS,
    canonical_sha256,
    feature_expression_map,
    library_release_definition,
)

# Compatibility name retained for callers and tests. Values now come from the
# same immutable definitions as the unified library.
RDAGENT_ALPHA20: dict[str, str] = feature_expression_map("alpha20")

TRANSPARENT_STRATEGY_RECIPE_IDS = (
    "short_relative_strength",
    "swing_trend",
    "long_quality_value",
)


def _record(
    feature_set_id: str,
    name: str,
    features: dict[str, str],
    *,
    source: str,
    contract_version: str = "governed-feature-set-v2-unified-library",
    source_in_identity: bool = True,
) -> dict[str, Any]:
    definition = {
        "contract_version": contract_version,
        "id": feature_set_id,
        "name": name,
        "features": dict(features),
    }
    identity = {**definition, **({"source": source} if source_in_identity else {})}
    return {
        **definition,
        "source": source,
        "definition_sha256": canonical_sha256(identity),
    }


def _unified_features() -> dict[str, str]:
    return {
        item.id: item.expression
        for item in sorted(FACTOR_DEFINITIONS, key=lambda definition: definition.id)
    }


FEATURE_SETS: dict[str, dict[str, Any]] = {
    "governed-baseline": _record(
        "governed-baseline",
        "Pinned RD-Agent Alpha20 baseline",
        RDAGENT_ALPHA20,
        source="rdagent-alpha20",
        contract_version="governed-feature-set-v1",
        source_in_identity=False,
    ),
    "qlib-alpha158": _record(
        "qlib-alpha158",
        "Pinned Qlib Alpha158",
        feature_expression_map("alpha158"),
        source="qlib-alpha158",
    ),
    "qlib-alpha360": _record(
        "qlib-alpha360",
        "Pinned Qlib Alpha360",
        feature_expression_map("alpha360"),
        source="qlib-alpha360",
    ),
    "platform-seed-v1": _record(
        "platform-seed-v1",
        "QuantLab governed 24-factor seed set",
        feature_expression_map("platform_seed"),
        source="platform-seed",
    ),
    "unified-research-v1": _record(
        "unified-research-v1",
        "Unified deduplicated Qlib and QuantLab research library",
        _unified_features(),
        source=library_release_definition()["id"],
    ),
}


def transparent_strategy_feature_set(recipe_id: str) -> dict[str, Any]:
    """Return the minimal, versioned feature grid owned by one public recipe.

    Importing the recipe lazily avoids making the general factor registry a
    second strategy registry.  The identifier includes the recipe version, and
    the immutable digest also includes the full recipe digest, so a material
    recipe change can never silently reuse an older research feature identity.
    """

    if recipe_id not in TRANSPARENT_STRATEGY_RECIPE_IDS:
        raise ValueError("unknown transparent strategy recipe")
    from .strategy_recipes import get_strategy_recipe

    recipe = get_strategy_recipe(recipe_id)
    factors = list(recipe.get("factor_baseline") or [])
    features = {
        str(item["id"]): str(item["qlib_expression"])
        for item in factors
        if str(item.get("id") or "").strip()
        and str(item.get("qlib_expression") or "").strip()
    }
    if len(features) != len(factors) or not features:
        raise ValueError("transparent strategy recipe has an incomplete feature grid")
    recipe_sha256 = canonical_sha256(recipe)
    feature_set_id = (
        f"transparent-fin-strategy:{recipe_id}:{recipe['version']}"
    )
    definition = _record(
        feature_set_id,
        f"Transparent fin_strategy grid for {recipe_id}",
        features,
        source=f"strategy-recipe:{recipe_id}:{recipe_sha256}",
        contract_version="transparent-fin-strategy-feature-set-v1",
    )
    existing = FEATURE_SETS.get(feature_set_id)
    if existing is not None and existing != definition:
        raise ValueError("transparent strategy feature identity changed in place")
    FEATURE_SETS[feature_set_id] = definition
    return get_feature_set(feature_set_id)


def transparent_strategy_feature_set_id(recipe_id: str) -> str:
    return str(transparent_strategy_feature_set(recipe_id)["id"])


def _load_transparent_feature_set(feature_set_id: str) -> None:
    for recipe_id in TRANSPARENT_STRATEGY_RECIPE_IDS:
        candidate = transparent_strategy_feature_set(recipe_id)
        if candidate["id"] == feature_set_id:
            return


def register_feature_set(definition: dict[str, Any]) -> dict[str, Any]:
    """Register an immutable feature set assembled from governed DB records."""

    candidate = dict(definition)
    supplied_sha256 = str(candidate.pop("definition_sha256", ""))
    feature_set_id = str(candidate.get("id") or "")
    features = candidate.get("features")
    if not feature_set_id or not isinstance(features, dict) or not features:
        raise ValueError("dynamic feature set is incomplete")
    actual_sha256 = canonical_sha256(candidate)
    if supplied_sha256 != actual_sha256:
        raise ValueError("dynamic feature set digest is invalid")
    normalized = {**candidate, "definition_sha256": actual_sha256}
    existing = FEATURE_SETS.get(feature_set_id)
    if existing is not None and existing != normalized:
        raise ValueError("feature set identity cannot be changed in place")
    FEATURE_SETS[feature_set_id] = normalized
    return get_feature_set(feature_set_id)


def resolve_feature_set(
    feature_set_id: str,
    embedded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if embedded is not None:
        if str(embedded.get("id") or "") != feature_set_id:
            raise ValueError("embedded feature set id disagrees with the request")
        existing = FEATURE_SETS.get(feature_set_id)
        if existing is not None:
            if existing != embedded:
                raise ValueError("embedded feature set disagrees with the registry")
        else:
            register_feature_set(embedded)
    return get_feature_set(feature_set_id)


def get_feature_set(feature_set_id: str) -> dict[str, Any]:
    if feature_set_id not in FEATURE_SETS and feature_set_id.startswith(
        "transparent-fin-strategy:"
    ):
        _load_transparent_feature_set(feature_set_id)
    try:
        item = FEATURE_SETS[feature_set_id]
    except KeyError as exc:
        raise ValueError(f"unknown governed feature set {feature_set_id!r}") from exc
    return {
        **item,
        "features": dict(item["features"]),
    }


def list_feature_sets() -> list[dict[str, Any]]:
    for recipe_id in TRANSPARENT_STRATEGY_RECIPE_IDS:
        transparent_strategy_feature_set(recipe_id)
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "source": item["source"],
            "feature_count": len(item["features"]),
            "definition_sha256": item["definition_sha256"],
            "contract_version": item["contract_version"],
        }
        for item in FEATURE_SETS.values()
    ]
