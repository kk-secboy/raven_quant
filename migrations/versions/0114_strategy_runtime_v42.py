"""Add the v42 trusted prepared model data strategy runtime identity.

Revision ID: 0114_strategy_runtime_v42
Revises: 0113_strategy_runtime_v41
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0114_strategy_runtime_v42"
down_revision: str | None = "0113_strategy_runtime_v41"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v42_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-07-v42"
# Frozen unified execution and research cadence identity; prior identities stay immutable.
RUNNER_SHA256 = "bc0cfad1188eb3103295f33f959e3d4bfed5f8b17527610873796d0c4cacaf4c"
RUNTIME_BUNDLE_SHA256 = (
    "6375f33d9a3dff138be9815f0fefc15ec7f21ae64cd303e0b682d265e5835dbf"
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
