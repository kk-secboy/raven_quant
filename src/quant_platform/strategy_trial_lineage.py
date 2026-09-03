"""Fail-closed statistical trial accounting for immutable strategy versions.

Physical StrategyVersion history is never deleted.  The only versions that may
share one statistical trial are source and target members of a validated,
append-only pre-result implementation-repair receipt which explicitly proves
that no performance information was used.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

AuditValidator = Callable[..., Mapping[str, Any]]


def _field(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name)


def _default_audit_validator(
    audit: Any, *, expected_receipt_sha256: str
) -> Mapping[str, Any]:
    # Lazy import avoids widening the strategy-store/lockbox import cycle.
    from quant_platform.transparent_baseline_governance import (
        validate_pre_result_repair_audit_event,
    )

    return validate_pre_result_repair_audit_event(
        audit,
        expected_receipt_sha256=expected_receipt_sha256,
    )


def build_strategy_trial_lineage(
    *,
    version_configs: Mapping[str, Mapping[str, Any]],
    backtests_by_id: Mapping[str, Any],
    repair_rows: Sequence[Any],
    repair_audits: Mapping[int, Any],
    audit_validator: AuditValidator | None = None,
) -> dict[str, Any]:
    """Return deterministic trial components and their accepted repair links.

    Missing, malformed, cross-family, or performance-informed evidence fails
    closed: the physical versions remain separate statistical trials.
    """

    validator = audit_validator or _default_audit_validator
    version_ids = set(version_configs)
    parents = {item: item for item in sorted(version_ids)}

    def find(item: str) -> str:
        parent = parents[item]
        while parent != parents[parent]:
            parents[parent] = parents[parents[parent]]
            parent = parents[parent]
        while item != parent:
            previous = parents[item]
            parents[item] = parent
            item = previous
        return parent

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parents[second] = first

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for repair in repair_rows:
        receipt_sha256 = str(_field(repair, "receipt_sha256"))
        raw_source_ids = {
            str(item)
            for item in list(_field(repair, "source_backtest_ids_json") or [])
        }
        raw_target_ids = {
            str(item)
            for item in list(
                _field(repair, "target_strategy_version_ids_json") or []
            )
        }
        all_source_rows = [
            backtests_by_id[item]
            for item in sorted(raw_source_ids)
            if item in backtests_by_id
        ]
        source_rows = [
            row
            for row in all_source_rows
            if str(_field(row, "strategy_version_id")) in version_ids
        ]
        source_version_ids = {
            str(_field(row, "strategy_version_id")) for row in source_rows
        }
        source_backtest_ids = {str(_field(row, "id")) for row in source_rows}
        target_version_ids = raw_target_ids & version_ids
        if not source_version_ids and not target_version_ids:
            continue
        try:
            verification = dict(_field(repair, "verification_json") or {})
            if (
                verification.get("performance_information_used") is not False
                or set(verification.get("source_backtest_ids") or [])
                != raw_source_ids
                or set(verification.get("target_strategy_version_ids") or [])
                != raw_target_ids
                or not source_version_ids
                or not target_version_ids
                or not raw_source_ids <= set(backtests_by_id)
            ):
                raise ValueError("repair registry scope is incomplete")
            audit = repair_audits.get(int(_field(repair, "source_audit_event_id")))
            if audit is None:
                raise ValueError("repair audit event is missing")
            receipt = dict(
                validator(
                    audit,
                    expected_receipt_sha256=receipt_sha256,
                )
            )
            if receipt.get("performance_information_used") is not False:
                raise ValueError("repair receipt used performance information")
            receipt_members = list(receipt.get("members") or [])
            receipt_backtest_ids = {
                str(item.get("backtest_id") or "")
                for item in receipt_members
                if isinstance(item, Mapping)
            }
            if receipt_backtest_ids != raw_source_ids:
                raise ValueError("repair receipt source backtests changed")
            receipt_source_version_ids = {
                str(item.get("strategy_version_id") or "")
                for item in receipt_members
                if isinstance(item, Mapping)
                and str(item.get("backtest_id") or "") in source_backtest_ids
            }
            if receipt_source_version_ids != source_version_ids:
                raise ValueError("repair receipt source versions changed")
            source_recipe_ids = {
                str(version_configs[item].get("recipe_id") or "")
                for item in source_version_ids
            }
            target_recipe_ids = {
                str(version_configs[item].get("recipe_id") or "")
                for item in target_version_ids
            }
            if (
                not source_recipe_ids
                or source_recipe_ids != target_recipe_ids
                or any(
                    str(version_configs[item].get("recipe_version") or "")
                    != str(receipt.get("target_recipe_version") or "")
                    for item in target_version_ids
                )
            ):
                raise ValueError("repair receipt economic family changed")
        except (TypeError, ValueError) as exc:
            rejected.append(
                {
                    "receipt_sha256": receipt_sha256,
                    "reason": str(exc),
                }
            )
            continue
        for source_version_id in source_version_ids:
            for target_version_id in target_version_ids:
                union(source_version_id, target_version_id)
        accepted.append(
            {
                "classification": "pre_result_implementation_repair",
                "receipt_sha256": receipt_sha256,
                "repair_source_backtest_ids": sorted(raw_source_ids),
                "source_backtest_ids": sorted(source_backtest_ids),
                "source_strategy_version_ids": sorted(source_version_ids),
                "target_strategy_version_ids": sorted(target_version_ids),
                "performance_information_used": False,
            }
        )

    components: dict[str, list[str]] = {}
    for item in sorted(version_ids):
        components.setdefault(find(item), []).append(item)
    return {
        "strategy_version_count": len(version_ids),
        "strategy_trial_count": len(components),
        "strategy_trial_components": [
            {
                "trial_root_strategy_version_id": root,
                "strategy_version_ids": members,
            }
            for root, members in sorted(components.items())
        ],
        "accepted_pre_result_repair_links": sorted(
            accepted,
            key=lambda item: str(item["receipt_sha256"]),
        ),
        "rejected_pre_result_repair_receipts": sorted(
            rejected,
            key=lambda item: str(item["receipt_sha256"]),
        ),
    }
