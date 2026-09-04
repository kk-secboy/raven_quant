"""Add the v30 strategy-scoped style-gate strategy runtime identity.

Revision ID: 0101_strategy_runtime_v30
Revises: 0100_strategy_runtime_v29
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0101_strategy_runtime_v30"
down_revision: str | None = "0100_strategy_runtime_v29"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v30_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-04-v30"
RUNNER_SHA256 = "f1ff1ab72165465675d8e699215fd0082c8dedbab0e12d910dc45485a470a457"
RUNTIME_BUNDLE_SHA256 = (
    "25d544bf2527cf4ca8d890b3e3a15836c87f5e6f28c75573ed7027da403171a1"
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
