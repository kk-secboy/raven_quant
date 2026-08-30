"""Seal the material v12 transparent-baseline runtime identity.

Revision ID: 0078_baseline_v12_seal
Revises: 0077_baseline_input_scope

Version 12 changes holding, reduction and point-in-time instrument-risk
semantics.  It is therefore a new economic recipe, not a no-result repair of
v11.  This revision deliberately leaves every v2-v5 repair receipt and the
same-lineage repair constraint from 0077 untouched.  It only makes the v12
runner, imported runtime bundle, and release-built worker image identities
database-enforced before a v12 StrategyVersion can be reserved.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0078_baseline_v12_seal"
down_revision: str | None = "0077_baseline_input_scope"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
_V12_RECIPE = "qlib-rdagent-single-mainline-2026-08-30-v12"
_V12_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
_V12_RUNTIME_BUNDLE_SHA256 = (
    "6366bd2b77c1c60ea4069afde43d5335c1e362c18acf2955f13902e7ba9ccbc6"
)
_TRANSPARENT_RECIPE_IDS = (
    "short_relative_strength",
    "swing_trend",
    "long_quality_value",
)
_CONSTRAINT = "ck_strategy_versions_v12_runtime_identity"


def _recipe_ids_sql() -> str:
    return "(" + ",".join(f"'{value}'" for value in _TRANSPARENT_RECIPE_IDS) + ")"


def _v12_runtime_identity_constraint() -> str:
    return (
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        f"'{_V12_RECIPE}' AND "
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        f"{_recipe_ids_sql()} THEN ("
        "config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runner_sha256' = '{_V12_RUNNER_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        f"'{_V12_RUNTIME_BUNDLE_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ "
        "'^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE"
    )


def upgrade() -> None:
    op.create_check_constraint(
        _CONSTRAINT,
        "strategy_versions",
        _v12_runtime_identity_constraint(),
        schema=SCHEMA,
    )


def downgrade() -> None:
    v12_rows = op.get_bind().scalar(
        sa.text(
            "SELECT count(*) FROM quantlab.strategy_versions "
            "WHERE config_json ->> 'recipe_version' = :recipe "
            "AND config_json ->> 'recipe_id' IN "
            "('short_relative_strength','swing_trend','long_quality_value')"
        ),
        {"recipe": _V12_RECIPE},
    )
    if int(v12_rows or 0) > 0:
        raise RuntimeError(
            "cannot downgrade 0078_baseline_v12_seal: immutable v12 transparent "
            "baseline versions exist"
        )
    op.drop_constraint(
        _CONSTRAINT,
        "strategy_versions",
        schema=SCHEMA,
        type_="check",
    )
