"""Add the v40 registered research activity strategy runtime identity.

Revision ID: 0112_strategy_runtime_v40
Revises: 0111_autopilot_research_events
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0112_strategy_runtime_v40"
down_revision: str | None = "0111_autopilot_research_events"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v40_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-07-v40"
RUNNER_SHA256 = "4a94328bf92b822da69530ffccccf0067cc3d2727d512d8d54f51095ef422717"
RUNTIME_BUNDLE_SHA256 = (
    "082416817e984f1956186a13b8f76864d4142267d9e1b297e00a1ebdfc4c94a1"
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
