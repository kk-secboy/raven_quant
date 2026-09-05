"""Add the v35 daily convertible-bond rating refresh strategy runtime identity.

Revision ID: 0106_strategy_runtime_v35
Revises: 0105_strategy_runtime_v34
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0106_strategy_runtime_v35"
down_revision: str | None = "0105_strategy_runtime_v34"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v35_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-05-v35"
RUNNER_SHA256 = "6df09f5bac9d8e10292a850da64707a1509d4896c4d65febd70aa3287413adf8"
RUNTIME_BUNDLE_SHA256 = (
    "c55ef3e92a8ef1390983399c114c5bf571711f3f2583185e7ed805854e4a72e4"
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
