"""Add the v38 physical-order quantity strategy runtime identity.

Revision ID: 0109_strategy_runtime_v38
Revises: 0108_strategy_runtime_v37
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0109_strategy_runtime_v38"
down_revision: str | None = "0108_strategy_runtime_v37"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v38_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-06-v38"
RUNNER_SHA256 = "48f241a9d03f63a87a54413f77b37577443a49285a28774716a4567382545c45"
RUNTIME_BUNDLE_SHA256 = (
    "43951d567fa610b8deed4ba4fec31716b0c7a1c7b9537ad37e5a9087ff1a4ec7"
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
