"""Shared, fail-closed identity contract for every stateless release service."""

from __future__ import annotations

import re
from collections.abc import Mapping

RELEASE_IDENTITY_ENV_TO_LABEL = {
    "QUANTLAB_RELEASE_ID": "quantlab.release",
    "QUANTLAB_CONFIG_DIGEST": "quantlab.config-digest",
    "QUANTLAB_RELEASE_KIND": "quantlab.release-kind",
    "QUANTLAB_RELEASE_ALIAS_OF": "quantlab.alias-of",
    "QUANTLAB_CANONICAL_BASELINE": "quantlab.canonical-baseline",
}
RELEASE_IDENTITY_ENV_KEYS = frozenset(RELEASE_IDENTITY_ENV_TO_LABEL)
STATEFUL_RELEASE_IDENTITY_EXEMPT = frozenset({"postgres"})

_RELEASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_CONFIG_DIGEST = re.compile(r"[0-9a-f]{64}")
_RELEASE_KINDS = frozenset({"canonical", "alias"})
_BOOLEAN_VALUES = frozenset({"true", "false"})


def release_identity_environment(
    release_id: str,
    config_digest: str,
    *,
    kind: str = "canonical",
    alias_of: str | None = None,
    canonical_baseline: bool = True,
) -> dict[str, str]:
    """Build one normalized environment identity and validate its alias semantics."""

    identity = {
        "QUANTLAB_RELEASE_ID": release_id.strip(),
        "QUANTLAB_CONFIG_DIGEST": config_digest.strip().lower(),
        "QUANTLAB_RELEASE_KIND": kind.strip().lower(),
        "QUANTLAB_RELEASE_ALIAS_OF": (alias_of or release_id).strip(),
        "QUANTLAB_CANONICAL_BASELINE": str(canonical_baseline).lower(),
    }
    error = release_identity_error(identity)
    if error is not None:
        raise ValueError(error)
    return identity


def release_identity_error(values: Mapping[str, object]) -> str | None:
    identity = {
        key: str(values.get(key) or "").strip()
        for key in RELEASE_IDENTITY_ENV_KEYS
    }
    release_id = identity["QUANTLAB_RELEASE_ID"]
    config_digest = identity["QUANTLAB_CONFIG_DIGEST"].lower()
    kind = identity["QUANTLAB_RELEASE_KIND"].lower()
    alias_of = identity["QUANTLAB_RELEASE_ALIAS_OF"]
    canonical_baseline = identity["QUANTLAB_CANONICAL_BASELINE"].lower()
    if not _RELEASE_ID.fullmatch(release_id):
        return "release id is absent or invalid"
    if not _CONFIG_DIGEST.fullmatch(config_digest):
        return "configuration digest is absent or invalid"
    if kind not in _RELEASE_KINDS:
        return "release kind must be canonical or alias"
    if not _RELEASE_ID.fullmatch(alias_of):
        return "release alias target is absent or invalid"
    if kind == "canonical" and alias_of != release_id:
        return "canonical release must alias itself"
    if kind == "alias" and alias_of == release_id:
        return "release alias must identify a different canonical release"
    if canonical_baseline not in _BOOLEAN_VALUES:
        return "canonical baseline marker must be true or false"
    return None


def normalized_release_identity(values: Mapping[str, object]) -> dict[str, str]:
    error = release_identity_error(values)
    if error is not None:
        raise ValueError(error)
    return {
        "QUANTLAB_RELEASE_ID": str(values["QUANTLAB_RELEASE_ID"]).strip(),
        "QUANTLAB_CONFIG_DIGEST": str(values["QUANTLAB_CONFIG_DIGEST"])
        .strip()
        .lower(),
        "QUANTLAB_RELEASE_KIND": str(values["QUANTLAB_RELEASE_KIND"])
        .strip()
        .lower(),
        "QUANTLAB_RELEASE_ALIAS_OF": str(
            values["QUANTLAB_RELEASE_ALIAS_OF"]
        ).strip(),
        "QUANTLAB_CANONICAL_BASELINE": str(
            values["QUANTLAB_CANONICAL_BASELINE"]
        )
        .strip()
        .lower(),
    }


def release_identity_labels(environment: Mapping[str, object]) -> dict[str, str]:
    identity = normalized_release_identity(environment)
    return {
        label: identity[variable]
        for variable, label in RELEASE_IDENTITY_ENV_TO_LABEL.items()
    }
