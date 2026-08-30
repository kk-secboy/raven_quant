"""Seal the append-only v14 transparent-baseline runtime identity.

Revision ID: 0081_baseline_v14_seal
Revises: 0080_baseline_v13_seal

Version 14 binds fill-aware holding-age reconciliation to Qlib's actually
filled portfolio.  Version 13 rows, constraints, jobs and OOS evidence remain
immutable history.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0081_baseline_v14_seal"
down_revision: str | None = "0080_baseline_v13_seal"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
_V14_RECIPE = "qlib-rdagent-single-mainline-2026-08-30-v14"
_V14_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
_V14_RUNTIME_BUNDLE_SHA256 = (
    "04f9dce110aaf44d32972c68db3edefafbd08cfdbf1f7dfb98aac9a15409bfe5"
)
_TRANSPARENT_RECIPE_IDS = (
    "short_relative_strength",
    "swing_trend",
    "long_quality_value",
)
_CONSTRAINT = "ck_strategy_versions_v14_runtime_identity"


def _recipe_ids_sql() -> str:
    return "(" + ",".join(f"'{value}'" for value in _TRANSPARENT_RECIPE_IDS) + ")"


def _v14_runtime_identity_constraint() -> str:
    return (
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        f"'{_V14_RECIPE}' AND "
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        f"{_recipe_ids_sql()} THEN ("
        "config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runner_sha256' = '{_V14_RUNNER_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        f"'{_V14_RUNTIME_BUNDLE_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ "
        "'^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE"
    )


def upgrade() -> None:
    op.create_check_constraint(
        _CONSTRAINT,
        "strategy_versions",
        _v14_runtime_identity_constraint(),
        schema=SCHEMA,
    )


def downgrade() -> None:
    v14_rows = op.get_bind().scalar(
        sa.text(
            "SELECT count(*) FROM quantlab.strategy_versions "
            "WHERE config_json ->> 'recipe_version' = :recipe "
            "AND config_json ->> 'recipe_id' IN "
            "('short_relative_strength','swing_trend','long_quality_value')"
        ),
        {"recipe": _V14_RECIPE},
    )
    if int(v14_rows or 0) > 0:
        raise RuntimeError(
            "cannot downgrade 0081_baseline_v14_seal: immutable v14 transparent "
            "baseline versions exist"
        )
    op.drop_constraint(
        _CONSTRAINT,
        "strategy_versions",
        schema=SCHEMA,
        type_="check",
    )
