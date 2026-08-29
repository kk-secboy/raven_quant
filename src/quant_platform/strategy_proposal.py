from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import date
from typing import Any

from quant_platform.cost_model import KNOWN_COST_SCHEDULE_VERSIONS
from quant_platform.research_contracts import STRATEGY_SLOT_ORDER
from quant_platform.strategy_rule_ir import HORIZON_CONTRACTS, canonical_sha256

STRATEGY_PROPOSAL_VERSION = "strategy-proposal-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FEATURE_SET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PROPOSAL_FIELDS = frozenset(
    {
        "contract_version",
        "delivery_status",
        "name",
        "description",
        "horizon",
        "economic_hypothesis",
        "baseline_recipe_id",
        "baseline_recipe_version",
        "baseline_rules_sha256",
        "parent_strategy_version_id",
        "changed_slots",
        "data_contract",
        "evaluation_contract",
        "slots",
    }
)


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"strategy proposal contains duplicate key: {key}")
        result[key] = value
    return result


def parse_strategy_proposal_json(payload: str) -> dict[str, Any]:
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > 256 * 1024:
        raise ValueError("strategy proposal JSON is missing or too large")
    try:
        raw = json.loads(payload, object_pairs_hook=_no_duplicate_object)
    except json.JSONDecodeError as exc:
        raise ValueError("strategy proposal is not valid JSON") from exc
    return validate_strategy_proposal(raw)


def _bounded_text(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"strategy proposal {field} must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise ValueError(f"strategy proposal {field} is empty or too long")
    return normalized


def validate_strategy_proposal(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _PROPOSAL_FIELDS:
        actual = set(raw) if isinstance(raw, Mapping) else set()
        raise ValueError(
            "strategy proposal top-level contract drifted: "
            f"missing={sorted(_PROPOSAL_FIELDS - actual)}, "
            f"unexpected={sorted(actual - _PROPOSAL_FIELDS)}"
        )
    if raw["contract_version"] != STRATEGY_PROPOSAL_VERSION:
        raise ValueError("strategy proposal contract_version is unsupported")
    if raw["delivery_status"] != "research_only":
        raise ValueError("strategy proposals must remain research_only")
    horizon = str(raw["horizon"])
    if horizon not in HORIZON_CONTRACTS:
        raise ValueError("strategy proposal horizon is unsupported")
    parent = raw["parent_strategy_version_id"]
    if parent is not None:
        parent = _bounded_text(parent, field="parent_strategy_version_id", maximum=128)
    changed_slots = raw["changed_slots"]
    if (
        not isinstance(changed_slots, list)
        or not changed_slots
        or any(
            not isinstance(item, str) or item not in STRATEGY_SLOT_ORDER
            for item in changed_slots
        )
        or len(changed_slots) != len(set(changed_slots))
    ):
        raise ValueError("strategy proposal changed_slots is invalid")
    changed_slots = [slot for slot in STRATEGY_SLOT_ORDER if slot in changed_slots]

    data = raw["data_contract"]
    data_fields = {
        "dataset_snapshot_id",
        "feature_set_id",
        "feature_set_definition_sha256",
        "research_periods",
        "decision_frequency",
        "label_horizon_trading_days",
    }
    if not isinstance(data, Mapping) or set(data) != data_fields:
        raise ValueError("strategy proposal data_contract drifted")
    digest = str(data["feature_set_definition_sha256"])
    if not _SHA256.fullmatch(digest):
        raise ValueError("strategy proposal feature-set digest is invalid")
    dataset_snapshot_id = _bounded_text(
        data["dataset_snapshot_id"], field="dataset_snapshot_id", maximum=64
    )
    if not _SHA256.fullmatch(dataset_snapshot_id):
        raise ValueError("strategy proposal dataset snapshot identity is invalid")
    feature_set_id = _bounded_text(
        data["feature_set_id"], field="feature_set_id", maximum=128
    )
    if not _FEATURE_SET_ID.fullmatch(feature_set_id):
        raise ValueError("strategy proposal feature_set_id is invalid")
    research_periods = data["research_periods"]
    period_names = (
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "test_start",
        "test_end",
    )
    if not isinstance(research_periods, Mapping) or set(research_periods) != set(
        period_names
    ):
        raise ValueError("strategy proposal research periods drifted")
    normalized_periods: dict[str, str] = {}
    parsed_periods: dict[str, date] = {}
    for name in period_names:
        value = str(research_periods[name])
        try:
            parsed_periods[name] = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("strategy proposal research period is invalid") from exc
        normalized_periods[name] = value
    if not (
        parsed_periods["train_start"] <= parsed_periods["train_end"]
        < parsed_periods["valid_start"]
        <= parsed_periods["valid_end"]
        < parsed_periods["test_start"]
        <= parsed_periods["test_end"]
    ):
        raise ValueError("strategy proposal research periods must be ordered and non-overlapping")
    expected_horizon = HORIZON_CONTRACTS[horizon]
    if data["decision_frequency"] != expected_horizon["decision_frequency"]:
        raise ValueError("strategy proposal decision frequency disagrees with its horizon")
    label_horizon = data["label_horizon_trading_days"]
    if (
        isinstance(label_horizon, bool)
        or label_horizon != expected_horizon["label_horizon_trading_days"]
    ):
        raise ValueError("strategy proposal label horizon disagrees with its horizon")

    evaluation = raw["evaluation_contract"]
    evaluation_fields = {
        "benchmark",
        "primary_metric",
        "cost_schedule_version",
        "rolling_folds",
        "minimum_oos_observations",
        "final_oos_visible_during_selection",
    }
    if not isinstance(evaluation, Mapping) or set(evaluation) != evaluation_fields:
        raise ValueError("strategy proposal evaluation_contract drifted")
    if evaluation["final_oos_visible_during_selection"] is not False:
        raise ValueError("strategy proposal cannot expose final OOS during selection")
    if evaluation["benchmark"] != "SH000300":
        raise ValueError("strategy proposal benchmark must remain SH000300")
    if evaluation["primary_metric"] != "after_cost_information_ratio":
        raise ValueError("strategy proposal primary metric is unsupported")
    if evaluation["cost_schedule_version"] not in KNOWN_COST_SCHEDULE_VERSIONS:
        raise ValueError("strategy proposal cost schedule is unsupported")
    rolling_folds = evaluation["rolling_folds"]
    minimum_oos = evaluation["minimum_oos_observations"]
    if (
        isinstance(rolling_folds, bool)
        or not isinstance(rolling_folds, int)
        or not 3 <= rolling_folds <= 20
    ):
        raise ValueError("strategy proposal rolling_folds must be between 3 and 20")
    minimum_required_oos = {
        "short_1_5d": 252,
        "swing_1_6m": 504,
        "long_1_3y": 756,
    }[horizon]
    if (
        isinstance(minimum_oos, bool)
        or not isinstance(minimum_oos, int)
        or minimum_oos < minimum_required_oos
    ):
        raise ValueError(
            "strategy proposal minimum_oos_observations is too short for its horizon"
        )
    slots = raw["slots"]
    if not isinstance(slots, Mapping):
        raise ValueError("strategy proposal slots must be an object")

    normalized = {
        "contract_version": STRATEGY_PROPOSAL_VERSION,
        "delivery_status": "research_only",
        "name": _bounded_text(raw["name"], field="name", maximum=120),
        "description": _bounded_text(raw["description"], field="description", maximum=2000),
        "horizon": horizon,
        "economic_hypothesis": _bounded_text(
            raw["economic_hypothesis"], field="economic_hypothesis", maximum=4000
        ),
        "baseline_recipe_id": _bounded_text(
            raw["baseline_recipe_id"], field="baseline_recipe_id", maximum=128
        ),
        "baseline_recipe_version": _bounded_text(
            raw["baseline_recipe_version"], field="baseline_recipe_version", maximum=160
        ),
        "baseline_rules_sha256": str(raw["baseline_rules_sha256"]),
        "parent_strategy_version_id": parent,
        "changed_slots": changed_slots,
        "data_contract": {
            "dataset_snapshot_id": dataset_snapshot_id,
            "feature_set_id": feature_set_id,
            "feature_set_definition_sha256": digest,
            "research_periods": normalized_periods,
            "decision_frequency": data["decision_frequency"],
            "label_horizon_trading_days": label_horizon,
        },
        "evaluation_contract": {
            "benchmark": _bounded_text(evaluation["benchmark"], field="benchmark", maximum=32),
            "primary_metric": _bounded_text(
                evaluation["primary_metric"], field="primary_metric", maximum=80
            ),
            "cost_schedule_version": _bounded_text(
                evaluation["cost_schedule_version"], field="cost_schedule_version", maximum=128
            ),
            "rolling_folds": rolling_folds,
            "minimum_oos_observations": minimum_oos,
            "final_oos_visible_during_selection": False,
        },
        "slots": {str(key): value for key, value in slots.items()},
    }
    if not _SHA256.fullmatch(normalized["baseline_rules_sha256"]):
        raise ValueError("strategy proposal baseline rules digest is invalid")
    normalized["proposal_sha256"] = canonical_sha256(normalized)
    return normalized


def strategy_proposal_json_contract() -> dict[str, Any]:
    """Return the compact allowlist contract supplied to the proposal LLM."""

    return {
        "contract_version": STRATEGY_PROPOSAL_VERSION,
        "delivery_status": "research_only",
        "horizons": sorted(HORIZON_CONTRACTS),
        "slot_order": list(STRATEGY_SLOT_ORDER),
        "top_level_fields": sorted(_PROPOSAL_FIELDS),
        "rules": [
            "Return exactly one JSON object and no markdown.",
            "Use only allowlisted components and parameters supplied by the caller.",
            "Never emit Python, shell, SQL, URLs, broker instructions or executable expressions.",
            "Keep final_oos_visible_during_selection=false.",
        ],
    }
