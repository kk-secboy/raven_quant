"""Seal the complete v17 bootstrap binding during interruption recovery.

Revision ID: 0087_recovery_recipe_path
Revises: 0086_formal_bt_interrupt
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "0087_recovery_recipe_path"
down_revision: str | None = "0086_formal_bt_interrupt"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FUNCTION_SIGNATURE = "quantlab.validate_formal_backtest_interruption_recovery()"
_RECOVERY_LOCK_IDENTITY = (
    "formal-backtest-interruption:0a113fe28ca741b6be9c09ab046c9d02"
)
_BACKTEST_ID = "0a113fe28ca741b6be9c09ab046c9d02"
_JOB_ID = "858a75a6f1994c359fa9c3567ed09f57"
_STRATEGY_VERSION_ID = "4414d202dbb641608975e5305bc18da4"
_RECEIPT_SHA256 = "6345c455862d8bb587e12f8ce0be7c1da291f2dde82fbdbb31e954bb88b9d3df"
_VALIDATOR_BEGIN = "BEGIN\n            IF NOT ((NEW.verification_json"
_LOCKED_VALIDATOR_BEGIN = (
    "BEGIN\n"
    "            PERFORM pg_advisory_xact_lock(\n"
    "                hashtext('formal-backtest-interruption:"
    "0a113fe28ca741b6be9c09ab046c9d02'));\n"
    "            IF NOT ((NEW.verification_json"
)
_TOP_LEVEL_RECIPE_BINDING = (
    "source_version.config_json ->> 'recipe_sha256' =\n"
    "                    "
    "'dee3551a73f2ebb3fbbbddfafdf99f4618e3dfdd98981fdb4b4d8849723d5fd8'"
)
_BOOTSTRAP_RECIPE_BINDING = (
    "source_version.config_json -> 'transparent_baseline_bootstrap' ->> "
    "'recipe_sha256' =\n"
    "                    "
    "'dee3551a73f2ebb3fbbbddfafdf99f4618e3dfdd98981fdb4b4d8849723d5fd8'\n"
    "                AND source_version.config_json -> "
    "'transparent_baseline_bootstrap' ->> 'recipe_id' =\n"
    "                    'short_relative_strength'\n"
    "                AND source_version.config_json -> "
    "'transparent_baseline_bootstrap' ->> 'recipe_version' =\n"
    "                    'qlib-rdagent-single-mainline-2026-08-31-v17'\n"
    "                AND source_version.config_json -> "
    "'transparent_baseline_bootstrap' ->> 'target_runner_sha256' =\n"
    "                    "
    "'31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3'\n"
    "                AND source_version.config_json -> "
    "'transparent_baseline_bootstrap' ->> 'target_runtime_bundle_sha256' =\n"
    "                    "
    "'c3d6aeb00a8f5286ee117f885231b43ad49e461c5183f5cbf32c4601824bb9fc'\n"
    "                AND source_version.config_json -> "
    "'transparent_baseline_bootstrap' ->> "
    "'target_worker_runtime_image_digest' =\n"
    "                    "
    "'sha256:b41f78f9c99dd9853998d85a52593bcab247907ac60f9d721d863e20904e8bb7'\n"
    "                AND source_version.config_json -> "
    "'transparent_baseline_bootstrap' ->> 'dataset' =\n"
    "                    'cn-20080101-20260828-v7-failclosed-ed5c8b3'\n"
    "                AND source_version.config_json -> "
    "'transparent_baseline_bootstrap' ->> 'dataset_identity_sha256' =\n"
    "                    "
    "'eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2'\n"
    "                AND source_version.config_json -> "
    "'transparent_baseline_bootstrap' ->> 'dataset_lineage_id' =\n"
    "                    "
    "'1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e'\n"
    "                AND source_version.config_json -> "
    "'transparent_baseline_bootstrap' -> 'formal_periods' =\n"
    "                    '{\"end\":\"2019-11-20\","
    "\"historical_end\":\"2018-10-10\","
    "\"historical_start\":\"2008-01-02\","
    "\"start\":\"2018-11-08\"}'::jsonb"
)
_VALIDATOR_REPLACEMENTS = (
    (_VALIDATOR_BEGIN, _LOCKED_VALIDATOR_BEGIN),
    (_TOP_LEVEL_RECIPE_BINDING, _BOOTSTRAP_RECIPE_BINDING),
)


def _function_definition(bind: Any) -> str:
    definition = bind.scalar(
        sa.text(
            "SELECT pg_get_functiondef("
            "to_regprocedure(CAST(:signature AS text)))"
        ),
        {"signature": _FUNCTION_SIGNATURE},
    )
    if not isinstance(definition, str) or not definition.strip():
        raise RuntimeError(
            "formal-backtest interruption recovery validator function is missing"
        )
    return definition


def _replace_validator_fragments(
    bind: Any, *, replacements: tuple[tuple[str, str], ...]
) -> None:
    definition = _function_definition(bind)
    for old, new in replacements:
        if definition.count(old) != 1 or new in definition:
            raise RuntimeError(
                "formal-backtest interruption recovery validator source is not exact"
            )
    replacement = definition
    for old, new in replacements:
        replacement = replacement.replace(old, new, 1)
    if not replacement.lstrip().upper().startswith("CREATE OR REPLACE FUNCTION "):
        raise RuntimeError(
            "formal-backtest interruption recovery validator DDL is not replaceable"
        )

    # This is server-returned PL/pgSQL, not a SQLAlchemy statement template.
    # It contains both literal `%ROWTYPE` tokens and JSON text with `:` tokens.
    # Sending it as raw driver SQL avoids SQLAlchemy bind parsing, while
    # ``no_parameters`` makes the DBAPI call ``cursor.execute(statement)`` so
    # psycopg does not parse literal percent tokens as placeholders.
    bind.execution_options(no_parameters=True).exec_driver_sql(replacement)

    verified = _function_definition(bind)
    for old, new in replacements:
        if old in verified or verified.count(new) != 1:
            raise RuntimeError(
                "formal-backtest interruption recovery validator replacement was not exact"
            )


def _lock_recovery_registration(bind: Any) -> None:
    bind.execute(
        sa.text("SELECT pg_advisory_xact_lock(hashtext(:identity))"),
        {"identity": _RECOVERY_LOCK_IDENTITY},
    )
    bind.execute(
        sa.text(
            "LOCK TABLE quantlab.formal_backtest_interruption_recoveries "
            "IN SHARE ROW EXCLUSIVE MODE"
        )
    )


def _install_strategy_version_guard(bind: Any) -> None:
    bind.exec_driver_sql(
        f"""
        CREATE OR REPLACE FUNCTION
            quantlab.guard_v17_recovery_strategy_version()
        RETURNS trigger AS $$
        DECLARE
            recovery_exists boolean;
            recovery_job_status text;
            recovery_backtest_status text;
        BEGIN
            IF OLD.id <> '{_STRATEGY_VERSION_ID}' THEN
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END IF;
            PERFORM pg_advisory_xact_lock(hashtext('{_RECOVERY_LOCK_IDENTITY}'));
            SELECT EXISTS(
                SELECT 1
                FROM quantlab.formal_backtest_interruption_recoveries recovery
                WHERE recovery.strategy_version_id = OLD.id
                  AND recovery.job_id = '{_JOB_ID}'
                  AND recovery.backtest_id = '{_BACKTEST_ID}'
                  AND recovery.receipt_sha256 = '{_RECEIPT_SHA256}'
            ) INTO recovery_exists;
            IF recovery_exists IS NOT TRUE THEN
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END IF;
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION
                    'recovered v17 strategy version cannot be deleted';
            END IF;
            SELECT source_job.status, source_backtest.status
            INTO recovery_job_status, recovery_backtest_status
            FROM quantlab.jobs source_job
            JOIN quantlab.backtest_runs source_backtest
              ON source_backtest.id = '{_BACKTEST_ID}'
             AND source_backtest.job_id = source_job.id
             AND source_backtest.strategy_version_id = OLD.id
            WHERE source_job.id = '{_JOB_ID}';
            IF NOT FOUND THEN
                RAISE EXCEPTION
                    'recovered v17 strategy execution identity is missing';
            END IF;
            IF recovery_job_status IN ('queued', 'running')
               OR recovery_backtest_status IN ('queued', 'running') THEN
                RAISE EXCEPTION
                    'recovered v17 strategy version is frozen during execution';
            END IF;
            IF (
                to_jsonb(NEW) - ARRAY[
                    'status', 'promotion_stage', 'approved_by',
                    'approval_reason', 'approved_at'
                ]
            ) IS DISTINCT FROM (
                to_jsonb(OLD) - ARRAY[
                    'status', 'promotion_stage', 'approved_by',
                    'approval_reason', 'approved_at'
                ]
            ) THEN
                RAISE EXCEPTION
                    'recovered v17 strategy computational identity is immutable';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    bind.exec_driver_sql(
        "DROP TRIGGER IF EXISTS trg_guard_v17_recovery_strategy_version "
        "ON quantlab.strategy_versions"
    )
    bind.exec_driver_sql(
        "CREATE TRIGGER trg_guard_v17_recovery_strategy_version "
        "BEFORE UPDATE OR DELETE ON quantlab.strategy_versions "
        "FOR EACH ROW EXECUTE FUNCTION "
        "quantlab.guard_v17_recovery_strategy_version()"
    )


def upgrade() -> None:
    bind = op.get_bind()
    _lock_recovery_registration(bind)
    _replace_validator_fragments(bind, replacements=_VALIDATOR_REPLACEMENTS)
    _install_strategy_version_guard(bind)


def downgrade() -> None:
    bind = op.get_bind()
    _lock_recovery_registration(bind)
    recovery_count = int(
        bind.scalar(
            sa.text(
                "SELECT count(*) FROM "
                "quantlab.formal_backtest_interruption_recoveries"
            )
        )
        or 0
    )
    if recovery_count:
        raise RuntimeError(
            "cannot downgrade after immutable formal-backtest interruption evidence"
        )
    bind.exec_driver_sql(
        "DROP TRIGGER IF EXISTS trg_guard_v17_recovery_strategy_version "
        "ON quantlab.strategy_versions"
    )
    bind.exec_driver_sql(
        "DROP FUNCTION IF EXISTS "
        "quantlab.guard_v17_recovery_strategy_version()"
    )
    _replace_validator_fragments(
        bind,
        replacements=tuple((new, old) for old, new in _VALIDATOR_REPLACEMENTS),
    )
