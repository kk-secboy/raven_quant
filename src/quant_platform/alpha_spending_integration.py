from __future__ import annotations

import re
from typing import Any

from quant_data.snapshot_lineage import canonical_sha256

CAPITAL_OOS_FAMILY_CONTRACT_VERSION = "capital-oos-stable-mandate-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXCLUDED_FIELDS = (
    "autopilot_cycle_id",
    "calendar_date",
    "config_revision",
    "dataset_identity_sha256",
    "dataset_lineage_id",
    "snapshot_name",
    "tournament_id",
    "incumbent_manifest_sha256",
    "champion_manifest_sha256",
    "sota_version_sha256",
    "frozen_bundle_manifest_sha256",
    "frozen_baseline_manifest_sha256",
    "research_data_end",
    "final_oos_window",
)
_FAMILY_FIELDS = (
    "contract_version",
    "universe",
    "benchmark",
    "label_horizon_days",
    "cost_contract_sha256",
    "execution_contract_sha256",
    "governance_contract_sha256",
)


def capital_oos_family_identity_policy() -> dict[str, Any]:
    """Describe, but do not alter, the stable family hash contract."""

    return {
        "contract_version": CAPITAL_OOS_FAMILY_CONTRACT_VERSION,
        "family_fields": list(_FAMILY_FIELDS),
        "excluded_from_family_identity": list(_EXCLUDED_FIELDS),
    }


def capital_oos_family_manifest(
    universe: str,
    benchmark: str,
    label_horizon_days: int,
    cost_contract_sha256: str,
    execution_contract_sha256: str,
    governance_contract_sha256: str,
) -> dict[str, Any]:
    """Build the stable mandate identity for all future final-OOS openings.

    Dataset lineage is batch evidence, not a family key. Incumbents, champions,
    SOTA members, bundles, dates and daily dataset identities deliberately
    cannot be passed here: changing any of them must not mint a fresh 0.05
    budget.
    """

    if not isinstance(universe, str) or not isinstance(benchmark, str):
        raise ValueError("universe and benchmark must be strings")
    universe_value = universe.strip().lower()
    benchmark_value = benchmark.strip().upper()
    if not universe_value or not benchmark_value:
        raise ValueError("universe and benchmark are required")
    if (
        isinstance(label_horizon_days, bool)
        or not isinstance(label_horizon_days, int)
        or label_horizon_days <= 0
    ):
        raise ValueError("label_horizon_days must be a positive integer")
    if not all(
        isinstance(value, str)
        for value in (
            cost_contract_sha256,
            execution_contract_sha256,
            governance_contract_sha256,
        )
    ):
        raise ValueError("cost, execution and governance contracts require SHA256")
    hashes = {
        "cost_contract_sha256": cost_contract_sha256.strip().lower(),
        "execution_contract_sha256": execution_contract_sha256.strip().lower(),
        "governance_contract_sha256": governance_contract_sha256.strip().lower(),
    }
    if any(not _SHA256.fullmatch(value) for value in hashes.values()):
        raise ValueError("cost, execution and governance contracts require SHA256")
    return {
        "contract_version": CAPITAL_OOS_FAMILY_CONTRACT_VERSION,
        "universe": universe_value,
        "benchmark": benchmark_value,
        "label_horizon_days": label_horizon_days,
        **hashes,
    }


def validate_capital_oos_family_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("capital OOS stable mandate must be an object")
    expected_keys = set(_FAMILY_FIELDS)
    if set(value) != expected_keys:
        raise ValueError("capital OOS stable mandate contains unstable or missing fields")
    if (
        not isinstance(value["universe"], str)
        or not isinstance(value["benchmark"], str)
        or isinstance(value["label_horizon_days"], bool)
        or not isinstance(value["label_horizon_days"], int)
        or not all(
            isinstance(value[key], str)
            for key in (
                "contract_version",
                "cost_contract_sha256",
                "execution_contract_sha256",
                "governance_contract_sha256",
            )
        )
    ):
        raise ValueError("capital OOS stable mandate has invalid field types")
    rebuilt = capital_oos_family_manifest(
        value["universe"],
        value["benchmark"],
        value["label_horizon_days"],
        value["cost_contract_sha256"],
        value["execution_contract_sha256"],
        value["governance_contract_sha256"],
    )
    if value != rebuilt:
        raise ValueError("capital OOS stable mandate is not canonical")
    return rebuilt


def capital_oos_family_manifest_sha256(value: Any) -> str:
    """Hash only a validated stable mandate, never a candidate identity."""

    return canonical_sha256(validate_capital_oos_family_manifest(value))


def capital_oos_family_sha256(
    universe: str,
    benchmark: str,
    label_horizon_days: int,
    cost_contract_sha256: str,
    execution_contract_sha256: str,
    governance_contract_sha256: str,
) -> str:
    return capital_oos_family_manifest_sha256(
        capital_oos_family_manifest(
            universe,
            benchmark,
            label_horizon_days,
            cost_contract_sha256,
            execution_contract_sha256,
            governance_contract_sha256,
        )
    )
