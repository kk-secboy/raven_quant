"""Add the v28 collision-free derived-market-value strategy runtime identity.

Revision ID: 0099_strategy_runtime_v28
Revises: 0098_strategy_runtime_v27
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0099_strategy_runtime_v28"
down_revision: str | None = "0098_strategy_runtime_v27"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v28_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-04-v28"
RUNNER_SHA256 = "79ff5cee1f046dea95bba13358005d975f0a83d7324c8f0fe18b07e5962b9615"
RUNTIME_BUNDLE_SHA256 = (
    "d8bb1e2ba21413e4cfb04ab819a270e4699f7d5e9964939ae727aa8055876b85"
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
