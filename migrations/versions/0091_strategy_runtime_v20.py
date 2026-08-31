"""Add the v20 sealed strategy-research runtime identity.

Revision ID: 0091_strategy_runtime_v20
Revises: 0090_strategy_research_runtime
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0091_strategy_runtime_v20"
down_revision: str | None = "0090_strategy_research_runtime"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v20_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-01-v20"
RUNNER_SHA256 = "c45bd901f25ba0cf289e3ec1a71015865d190afb4510c6bc207d53c9cc2ab24d"
RUNTIME_BUNDLE_SHA256 = (
    "a0fe9491b27836526af9d7644c2c396973a95933fcfa4bf9365046cf634be53d"
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
