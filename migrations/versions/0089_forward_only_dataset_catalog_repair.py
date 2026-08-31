"""Repair the v18 runtime seal after normalizing the production Qlib catalog.

Revision ID: 0089_v18_catalog_repair
Revises: 0088_forward_only_rehab
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0089_v18_catalog_repair"
down_revision: str | None = "0088_forward_only_rehab"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
_RUNTIME_CONSTRAINT = "ck_strategy_versions_v18_runtime_identity"
_V18_RECIPE = "qlib-rdagent-single-mainline-2026-08-31-v18"
_V18_RUNNER_SHA256 = (
    "c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d"
)
_PREVIOUS_RUNTIME_BUNDLE_SHA256 = (
    "c495044915133b41bd7f3c13df3b82fd9624e3e892e51ac4deb892eddbf01dfa"
)
_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "3f0c60adbe3b50ff26771e11ce75f6a48d13772ac5bff549f716469748b92874"
)


def _runtime_identity_constraint(runtime_bundle_sha256: str) -> str:
    return (
        "(CASE WHEN COALESCE(config_json ->> 'recipe_version', '') = "
        f"'{_V18_RECIPE}' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') = "
        "'short_relative_strength' "
        "AND evidence_mode = 'consumed_historical_replay' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runner_sha256' = '{_V18_RUNNER_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runtime_bundle_sha256' = '{runtime_bundle_sha256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE"
    )


def _lock_and_require_empty_v18_evidence() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "SELECT pg_advisory_xact_lock("
            "hashtext('forward-only-rehabilitation:0089-runtime-repair'))"
        )
    )
    bind.execute(
        sa.text(
            "LOCK TABLE quantlab.strategy_versions, quantlab.backtest_runs, "
            "quantlab.strategy_incomplete_family_eligibilities, "
            "quantlab.strategy_forward_only_rehabilitations "
            "IN SHARE ROW EXCLUSIVE MODE"
        )
    )
    counts = {
        "strategy_versions": int(
            bind.scalar(
                sa.text(
                    "SELECT count(*) FROM quantlab.strategy_versions "
                    "WHERE config_json ->> 'recipe_version' = :recipe "
                    "OR evidence_mode = 'consumed_historical_replay'"
                ),
                {"recipe": _V18_RECIPE},
            )
            or 0
        ),
        "backtest_runs": int(
            bind.scalar(
                sa.text(
                    "SELECT count(*) FROM quantlab.backtest_runs AS backtest "
                    "LEFT JOIN quantlab.strategy_versions AS version "
                    "ON version.id = backtest.strategy_version_id "
                    "WHERE backtest.evidence_mode = 'consumed_historical_replay' "
                    "OR version.config_json ->> 'recipe_version' = :recipe"
                ),
                {"recipe": _V18_RECIPE},
            )
            or 0
        ),
        "incomplete_family_receipts": int(
            bind.scalar(
                sa.text(
                    "SELECT count(*) FROM "
                    "quantlab.strategy_incomplete_family_eligibilities"
                )
            )
            or 0
        ),
        "rehabilitation_receipts": int(
            bind.scalar(
                sa.text(
                    "SELECT count(*) FROM "
                    "quantlab.strategy_forward_only_rehabilitations"
                )
            )
            or 0
        ),
    }
    if any(counts.values()):
        raise RuntimeError(
            "cannot replace the v18 runtime seal after governed v18 evidence exists: "
            f"{counts}"
        )


def _require_current_runtime(expected: str, forbidden: str) -> None:
    definition = op.get_bind().scalar(
        sa.text(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = 'quantlab.strategy_versions'::regclass "
            "AND conname = :constraint AND contype = 'c' AND convalidated"
        ),
        {"constraint": _RUNTIME_CONSTRAINT},
    )
    required = (
        _V18_RECIPE,
        _V18_RUNNER_SHA256,
        "short_relative_strength",
        "consumed_historical_replay",
        expected,
    )
    if not isinstance(definition, str) or any(item not in definition for item in required):
        raise RuntimeError("the installed v18 runtime constraint is not the expected seal")
    if forbidden in definition:
        raise RuntimeError("the installed v18 runtime constraint is not the expected seal")


def _replace_runtime_constraint(runtime_bundle_sha256: str) -> None:
    op.drop_constraint(
        _RUNTIME_CONSTRAINT,
        "strategy_versions",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        _RUNTIME_CONSTRAINT,
        "strategy_versions",
        _runtime_identity_constraint(runtime_bundle_sha256),
        schema=SCHEMA,
    )


def upgrade() -> None:
    _lock_and_require_empty_v18_evidence()
    _require_current_runtime(
        _PREVIOUS_RUNTIME_BUNDLE_SHA256,
        _TARGET_RUNTIME_BUNDLE_SHA256,
    )
    _replace_runtime_constraint(_TARGET_RUNTIME_BUNDLE_SHA256)
    _require_current_runtime(
        _TARGET_RUNTIME_BUNDLE_SHA256,
        _PREVIOUS_RUNTIME_BUNDLE_SHA256,
    )


def downgrade() -> None:
    _lock_and_require_empty_v18_evidence()
    _require_current_runtime(
        _TARGET_RUNTIME_BUNDLE_SHA256,
        _PREVIOUS_RUNTIME_BUNDLE_SHA256,
    )
    _replace_runtime_constraint(_PREVIOUS_RUNTIME_BUNDLE_SHA256)
    _require_current_runtime(
        _PREVIOUS_RUNTIME_BUNDLE_SHA256,
        _TARGET_RUNTIME_BUNDLE_SHA256,
    )
