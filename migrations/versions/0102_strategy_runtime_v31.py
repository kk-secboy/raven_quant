"""Add the v31 governed Qlib-kernels strategy runtime identity.

Revision ID: 0102_strategy_runtime_v31
Revises: 0101_strategy_runtime_v30
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0102_strategy_runtime_v31"
down_revision: str | None = "0101_strategy_runtime_v30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
CONSTRAINT = "ck_strategy_versions_v31_runtime_identity"
RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-09-04-v31"
RUNNER_SHA256 = "036d7979581b3de719532303435f8fba81e43f4403b3c82bf371e5c18820b7f2"
RUNTIME_BUNDLE_SHA256 = (
    "dac80be45fcae33d2bcdafff76aa2b71feac6624deb25b542bfc02961584f1b8"
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
