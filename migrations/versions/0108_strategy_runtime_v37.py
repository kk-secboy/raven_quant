"""Add the v37 capacity-aware whole-lot execution strategy runtime identity.

Revision ID: 0108_strategy_runtime_v37
Revises: 0107_strategy_runtime_v36
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0108_strategy_runtime_v37"
down_revision: str | None = "0107_strategy_runtime_v36"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v37_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-06-v37"
RUNNER_SHA256 = "372d8744947144be822aeda155fa2907b6aefda464422b759d1083b772148054"
RUNTIME_BUNDLE_SHA256 = (
    "77eccc2d3b7150027e9dde45ff261049272331cb1952b047c28367f552ffd4b1"
)


def _constraint() -> str:
    return (
        "(CASE WHEN COALESCE(config_json ->> 'recipe_version', '') = "
        f"'{RECIPE_VERSION}' THEN ("
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        "('short_relative_strength','swing_trend','long_quality_value') "
        "AND evidence_mode = 'sealed_final_oos' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runner_sha256' = '{RUNNER_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runtime_bundle_sha256' = '{RUNTIME_BUNDLE_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE"
    )


def upgrade() -> None:
    op.create_check_constraint(
        CONSTRAINT,
        "strategy_versions",
        _constraint(),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint(
        CONSTRAINT,
        "strategy_versions",
        schema=SCHEMA,
        type_="check",
    )
