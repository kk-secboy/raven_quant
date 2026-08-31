"""Add the consumed-history forward-only rehabilitation authority.

Revision ID: 0088_forward_only_rehab
Revises: 0087_recovery_recipe_path

The 2018-11-08..2019-11-20 short-baseline window has already been opened and
consumed.  It may be replayed for deterministic engineering verification, but
it is not sealed, final, or unseen evidence.  This migration adds an immutable
evidence-mode label and one append-only qualification receipt.  The receipt
does not create a second lifecycle: an admitted version uses the existing
paper stage, simulation ledger, and forward gate, with at least 365 calendar
days and 252 certified trading days of genuinely new evidence.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0088_forward_only_rehab"
down_revision: str | None = "0087_recovery_recipe_path"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
JSON = sa.JSON().with_variant(JSONB(), "postgresql")

SOURCE_VERSION_ID = "4414d202dbb641608975e5305bc18da4"
SOURCE_BACKTEST_ID = "0a113fe28ca741b6be9c09ab046c9d02"
SOURCE_JOB_ID = "858a75a6f1994c359fa9c3567ed09f57"
SOURCE_INTERRUPTION_RECEIPT_SHA256 = (
    "6345c455862d8bb587e12f8ce0be7c1da291f2dde82fbdbb31e954bb88b9d3df"
)
DATASET = "cn-20080101-20260828-v7-failclosed-ed5c8b3"
DATASET_IDENTITY = "eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2"
DATASET_LINEAGE = "1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e"
RULES_SHA256 = "644d9ee73ea4c167c7d8f58b2e8b1707289cb569c48ae73ce131cce139f7e756"
EXECUTION_SHA256 = "0ffa8939a70f0c46499e3ae472b877f68cce1b9fb95dfe81886ab8430ddcce61"
V18_RECIPE_VERSION = "qlib-rdagent-single-mainline-2026-08-31-v18"
V18_RUNNER_SHA256 = "0" * 64
V18_RUNTIME_BUNDLE_SHA256 = "0" * 64


def upgrade() -> None:
    for table in ("strategy_versions", "backtest_runs"):
        op.add_column(
            table,
            sa.Column(
                "evidence_mode",
                sa.String(),
                nullable=False,
                server_default="legacy_ambiguous",
            ),
            schema=SCHEMA,
        )
        op.create_check_constraint(
            f"ck_{table}_evidence_mode",
            table,
            "evidence_mode IN "
            "('legacy_ambiguous', 'sealed_final_oos', "
            "'consumed_historical_replay')",
            schema=SCHEMA,
        )

    op.create_check_constraint(
        "ck_strategy_versions_v18_runtime_identity",
        "strategy_versions",
        "(CASE WHEN COALESCE(config_json ->> 'recipe_version', '') = "
        f"'{V18_RECIPE_VERSION}' AND "
        "COALESCE(config_json ->> 'recipe_id', '') = "
        "'short_relative_strength' THEN ("
        "evidence_mode = 'consumed_historical_replay' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runner_sha256' = '{V18_RUNNER_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runtime_bundle_sha256' = '{V18_RUNTIME_BUNDLE_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ '^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE",
        schema=SCHEMA,
    )

    op.create_table(
        "strategy_forward_only_rehabilitations",
        sa.Column("receipt_sha256", sa.String(), primary_key=True),
        sa.Column(
            "source_audit_event_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{SCHEMA}.audit_events.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "source_strategy_version_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.strategy_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "source_backtest_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.backtest_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "source_job_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.jobs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "source_interruption_recovery_receipt_sha256",
            sa.String(),
            sa.ForeignKey(
                f"{SCHEMA}.formal_backtest_interruption_recoveries.receipt_sha256",
                ondelete="RESTRICT",
            ),
            nullable=False,
        ),
        sa.Column("source_interruption_receipt_authority", sa.String(), nullable=False),
        sa.Column("source_lockbox_contract_version", sa.String(), nullable=False),
        sa.Column("source_lockbox_batch_sha256", sa.String(), nullable=False),
        sa.Column("source_lockbox_member_sha256", sa.String(), nullable=False),
        sa.Column("source_history_selection_sha256", sa.String(), nullable=False),
        sa.Column("source_unavailable_horizons_sha256", sa.String(), nullable=False),
        sa.Column("source_unavailable_evidence_sha256s_json", JSON, nullable=False),
        sa.Column("source_cash_only_scope", sa.String(), nullable=False),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.strategy_versions.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "backtest_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.backtest_runs.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "consumed_oos_vintage_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.oos_vintages.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("contract_version", sa.String(), nullable=False),
        sa.Column("evidence_mode", sa.String(), nullable=False),
        sa.Column("authority", sa.String(), nullable=False),
        sa.Column("recipe_id", sa.String(), nullable=False),
        sa.Column("horizon_profile", sa.String(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("dataset_lineage_id", sa.String(), nullable=False),
        sa.Column("strategy_rules_sha256", sa.String(), nullable=False),
        sa.Column("execution_contract_hash", sa.String(), nullable=False),
        sa.Column("runner_sha256", sa.String(), nullable=False),
        sa.Column("runtime_bundle_sha256", sa.String(), nullable=False),
        sa.Column("worker_runtime_image_digest", sa.String(), nullable=False),
        sa.Column("replay_periods_json", JSON, nullable=False),
        sa.Column("replay_manifest_sha256", sa.String(), nullable=False),
        sa.Column("replay_result_sha256", sa.String(), nullable=False),
        sa.Column("replay_artifact_manifest_sha256", sa.String(), nullable=False),
        sa.Column("replay_daily_returns_sha256", sa.String(), nullable=False),
        sa.Column("strategy_trial_count", sa.Integer(), nullable=False),
        sa.Column("trial_count_audit_sha256", sa.String(), nullable=False),
        sa.Column("incomplete_family_eligibility_sha256", sa.String(), nullable=False),
        sa.Column("forward_criteria_json", JSON, nullable=False),
        sa.Column("forward_criteria_sha256", sa.String(), nullable=False),
        sa.Column("qualification_json", JSON, nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "contract_version = 'forward-only-rehabilitation-v1' "
            "AND evidence_mode = 'consumed_historical_replay' "
            "AND authority = 'historical_description_only' "
            "AND source_interruption_receipt_authority = "
            "'interruption_identity_only_not_pre_result' "
            "AND source_lockbox_contract_version = "
            "'transparent-baseline-available-horizons-lockbox-v3' "
            "AND source_cash_only_scope = 'cash_only_projection_only' "
            "AND strategy_version_id <> source_strategy_version_id "
            "AND backtest_id <> source_backtest_id "
            "AND strategy_trial_count > 1",
            name="ck_forward_only_rehabilitation_authority",
        ),
        sa.CheckConstraint(
            "receipt_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_interruption_recovery_receipt_sha256 = "
            f"'{SOURCE_INTERRUPTION_RECEIPT_SHA256}' "
            "AND source_lockbox_batch_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_lockbox_member_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_history_selection_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_unavailable_horizons_sha256 ~ '^[0-9a-f]{64}$' "
            "AND dataset_identity_sha256 ~ '^[0-9a-f]{64}$' "
            "AND dataset_lineage_id ~ '^[0-9a-f]{64}$' "
            "AND strategy_rules_sha256 ~ '^[0-9a-f]{64}$' "
            "AND execution_contract_hash ~ '^[0-9a-f]{64}$' "
            "AND runner_sha256 ~ '^[0-9a-f]{64}$' "
            "AND runtime_bundle_sha256 ~ '^[0-9a-f]{64}$' "
            "AND worker_runtime_image_digest ~ '^sha256:[0-9a-f]{64}$' "
            "AND replay_manifest_sha256 ~ '^[0-9a-f]{64}$' "
            "AND replay_result_sha256 ~ '^[0-9a-f]{64}$' "
            "AND replay_artifact_manifest_sha256 ~ '^[0-9a-f]{64}$' "
            "AND replay_daily_returns_sha256 ~ '^[0-9a-f]{64}$' "
            "AND trial_count_audit_sha256 ~ '^[0-9a-f]{64}$' "
            "AND incomplete_family_eligibility_sha256 ~ '^[0-9a-f]{64}$' "
            "AND forward_criteria_sha256 ~ '^[0-9a-f]{64}$'",
            name="ck_forward_only_rehabilitation_hashes",
        ),
        sa.CheckConstraint(
            "(jsonb_typeof(replay_periods_json) = 'object' "
            "AND replay_periods_json ?& "
            "ARRAY['start','end','historical_start','historical_end'] "
            "AND jsonb_typeof(forward_criteria_json) = 'object' "
            "AND (forward_criteria_json -> 'thresholds' ->> "
            "'min_forward_calendar_days')::integer >= 365 "
            "AND (forward_criteria_json -> 'thresholds' ->> "
            "'min_forward_trading_days')::integer >= 252 "
            "AND jsonb_typeof(qualification_json) = 'object' "
            "AND jsonb_typeof(source_unavailable_evidence_sha256s_json) = 'object' "
            "AND source_unavailable_evidence_sha256s_json ?& "
            "ARRAY['swing_1_6m','long_1_3y'] "
            "AND jsonb_object_length(source_unavailable_evidence_sha256s_json) = 2 "
            "AND qualification_json -> 'historical_replay_opened' = 'true'::jsonb "
            "AND qualification_json -> 'final_oos_opened' = 'true'::jsonb "
            "AND qualification_json -> 'capital_eligible' = 'false'::jsonb "
            "AND qualification_json -> 'consumed_oos_replayed' = 'true'::jsonb "
            "AND qualification_json -> 'sealed_final_oos' = 'false'::jsonb "
            "AND qualification_json -> 'unseen_oos' = 'false'::jsonb "
            "AND qualification_json ->> 'authority' = "
            "'historical_description_only' "
            "AND qualification_json ->> 'receipt_sha256' = receipt_sha256) IS TRUE",
            name="ck_forward_only_rehabilitation_evidence",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_forward_only_rehabilitations_created",
        "strategy_forward_only_rehabilitations",
        [sa.text("created_at DESC")],
        schema=SCHEMA,
    )
    op.create_table(
        "strategy_incomplete_family_eligibilities",
        sa.Column("receipt_sha256", sa.String(), primary_key=True),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.strategy_versions.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("contract_version", sa.String(), nullable=False),
        sa.Column("evidence_mode", sa.String(), nullable=False),
        sa.Column("authority", sa.String(), nullable=False),
        sa.Column(
            "source_strategy_version_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.strategy_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "source_backtest_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.backtest_runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "source_job_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.jobs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("economic_hypothesis_group", sa.String(), nullable=False),
        sa.Column("eligible_strategy_version_ids_json", JSON, nullable=False),
        sa.Column("strategy_trial_count", sa.Integer(), nullable=False),
        sa.Column("trial_count_audit_json", JSON, nullable=False),
        sa.Column("trial_count_audit_sha256", sa.String(), nullable=False),
        sa.Column("missing_artifacts_json", JSON, nullable=False),
        sa.Column("cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("qualification_json", JSON, nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(contract_version = 'incomplete-factor-family-eligibility-v1' "
            "AND evidence_mode = 'consumed_historical_replay' "
            "AND authority = 'conservative_bonferroni_only' "
            "AND strategy_trial_count > 1 "
            "AND receipt_sha256 ~ '^[0-9a-f]{64}$' "
            "AND trial_count_audit_sha256 ~ '^[0-9a-f]{64}$' "
            "AND jsonb_typeof(eligible_strategy_version_ids_json) = 'array' "
            "AND jsonb_array_length(eligible_strategy_version_ids_json) > 1 "
            "AND jsonb_typeof(trial_count_audit_json) = 'object' "
            "AND jsonb_typeof(missing_artifacts_json) = 'array' "
            "AND jsonb_array_length(missing_artifacts_json) >= 3 "
            "AND qualification_json ->> 'receipt_sha256' = receipt_sha256) IS TRUE",
            name="ck_incomplete_family_eligibility_authority",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_incomplete_family_eligibilities_created",
        "strategy_incomplete_family_eligibilities",
        [sa.text("created_at DESC")],
        schema=SCHEMA,
    )

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION quantlab.validate_forward_only_rehabilitation()
        RETURNS trigger AS $$
        DECLARE
            source_version quantlab.strategy_versions%ROWTYPE;
            target_version quantlab.strategy_versions%ROWTYPE;
            source_backtest quantlab.backtest_runs%ROWTYPE;
            target_backtest quantlab.backtest_runs%ROWTYPE;
            source_job quantlab.jobs%ROWTYPE;
            vintage quantlab.oos_vintages%ROWTYPE;
            source_audit quantlab.audit_events%ROWTYPE;
            family_eligibility quantlab.strategy_incomplete_family_eligibilities%ROWTYPE;
            interruption quantlab.formal_backtest_interruption_recoveries%ROWTYPE;
            factor_count bigint;
        BEGIN
            PERFORM pg_advisory_xact_lock(
                hashtext('forward-only-rehabilitation:' || NEW.consumed_oos_vintage_id)
            );
            SELECT * INTO source_version FROM quantlab.strategy_versions
            WHERE id = NEW.source_strategy_version_id FOR SHARE;
            SELECT * INTO target_version FROM quantlab.strategy_versions
            WHERE id = NEW.strategy_version_id FOR UPDATE;
            SELECT * INTO source_backtest FROM quantlab.backtest_runs
            WHERE id = NEW.source_backtest_id FOR SHARE;
            SELECT * INTO target_backtest FROM quantlab.backtest_runs
            WHERE id = NEW.backtest_id FOR SHARE;
            SELECT * INTO source_job FROM quantlab.jobs
            WHERE id = NEW.source_job_id FOR SHARE;
            SELECT * INTO vintage FROM quantlab.oos_vintages
            WHERE id = NEW.consumed_oos_vintage_id FOR SHARE;
            SELECT * INTO source_audit FROM quantlab.audit_events
            WHERE id = NEW.source_audit_event_id FOR SHARE;
            SELECT * INTO family_eligibility
            FROM quantlab.strategy_incomplete_family_eligibilities
            WHERE strategy_version_id = NEW.strategy_version_id FOR SHARE;
            SELECT * INTO interruption
            FROM quantlab.formal_backtest_interruption_recoveries
            WHERE receipt_sha256 = NEW.source_interruption_recovery_receipt_sha256
            FOR SHARE;
            SELECT count(*) INTO factor_count FROM quantlab.strategy_factors
            WHERE strategy_version_id = NEW.strategy_version_id;

            IF NOT ((
                NEW.source_strategy_version_id = '{SOURCE_VERSION_ID}'
                AND NEW.source_backtest_id = '{SOURCE_BACKTEST_ID}'
                AND NEW.source_job_id = '{SOURCE_JOB_ID}'
                AND NEW.source_interruption_recovery_receipt_sha256 =
                    '{SOURCE_INTERRUPTION_RECEIPT_SHA256}'
                AND NEW.source_interruption_receipt_authority =
                    'interruption_identity_only_not_pre_result'
                AND NEW.source_lockbox_contract_version =
                    'transparent-baseline-available-horizons-lockbox-v3'
                AND NEW.source_cash_only_scope = 'cash_only_projection_only'
                AND source_version.config_json ->
                    'transparent_baseline_joint_lockbox' ->> 'contract_version' =
                    NEW.source_lockbox_contract_version
                AND source_version.config_json ->
                    'transparent_baseline_joint_lockbox' ->> 'batch_sha256' =
                    NEW.source_lockbox_batch_sha256
                AND source_version.config_json ->
                    'transparent_baseline_joint_lockbox' ->
                    'unopened_history_selection' ->> 'selection_sha256' =
                    NEW.source_history_selection_sha256
                AND NEW.qualification_json ->> 'source_lockbox_batch_sha256' =
                    NEW.source_lockbox_batch_sha256
                AND NEW.qualification_json ->> 'source_lockbox_member_sha256' =
                    NEW.source_lockbox_member_sha256
                AND NEW.qualification_json ->> 'source_history_selection_sha256' =
                    NEW.source_history_selection_sha256
                AND NEW.qualification_json ->> 'source_unavailable_horizons_sha256' =
                    NEW.source_unavailable_horizons_sha256
                AND NEW.qualification_json -> 'source_unavailable_evidence_sha256s' =
                    NEW.source_unavailable_evidence_sha256s_json
                AND NEW.qualification_json ->> 'source_cash_only_scope' =
                    NEW.source_cash_only_scope
                AND interruption.strategy_version_id = NEW.source_strategy_version_id
                AND interruption.backtest_id = NEW.source_backtest_id
                AND interruption.job_id = NEW.source_job_id
                AND source_backtest.strategy_version_id = source_version.id
                AND source_backtest.job_id = source_job.id
                AND source_backtest.dataset = '{DATASET}'
                AND source_backtest.periods_json = NEW.replay_periods_json
                AND source_job.status = 'cancelled'
                AND source_backtest.status IN ('failed', 'cancelled')
                AND target_version.id IS NOT NULL
                AND target_version.id <> source_version.id
                AND target_version.is_legacy = false
                AND target_version.status = 'draft'
                AND target_version.promotion_stage IS NULL
                AND target_version.evidence_mode = 'consumed_historical_replay'
                AND target_version.horizon_profile = 'short_1_5d'
                AND target_version.strategy_rules_sha256 = '{RULES_SHA256}'
                AND target_version.execution_contract_hash = '{EXECUTION_SHA256}'
                AND target_version.config_json ->> 'recipe_id' =
                    'short_relative_strength'
                AND target_version.config_json ->> 'recipe_version' =
                    '{V18_RECIPE_VERSION}'
                AND target_version.config_json ->> 'factor_source_mode' =
                    'qlib_baseline'
                AND target_version.config_json ->> 'evidence_mode' =
                    NEW.evidence_mode
                AND target_version.config_json ->> 'horizon_profile' =
                    NEW.horizon_profile
                AND target_version.config_json ->> 'strategy_rules_sha256' =
                    NEW.strategy_rules_sha256
                AND target_version.config_json ->> 'execution_contract_hash' =
                    NEW.execution_contract_hash
                AND target_version.config_json ->
                    'forward_only_rehabilitation' -> 'replay_periods' =
                    NEW.replay_periods_json
                AND target_version.config_json ->
                    'forward_only_rehabilitation' ->> 'consumed_oos_vintage_id' =
                    NEW.consumed_oos_vintage_id
                AND target_version.config_json ->
                    'transparent_baseline_bootstrap' -> 'formal_periods' =
                    NEW.replay_periods_json
                AND target_version.config_json ->
                    'transparent_baseline_bootstrap' ->
                    'research_window_contract' -> 'periods' =
                    NEW.replay_periods_json
                AND target_version.config_json ->
                    'transparent_baseline_bootstrap' ->> 'target_runner_sha256' =
                    NEW.runner_sha256
                AND target_version.config_json ->
                    'transparent_baseline_bootstrap' ->>
                    'target_runtime_bundle_sha256' = NEW.runtime_bundle_sha256
                AND target_version.config_json ->
                    'transparent_baseline_bootstrap' ->>
                    'target_worker_runtime_image_digest' =
                    NEW.worker_runtime_image_digest
                AND factor_count = 0
                AND target_backtest.strategy_version_id = target_version.id
                AND target_backtest.status = 'succeeded'
                AND target_backtest.is_legacy = false
                AND target_backtest.evidence_mode = 'consumed_historical_replay'
                AND target_backtest.dataset = '{DATASET}'
                AND target_backtest.periods_json = NEW.replay_periods_json
                AND target_backtest.metrics_json ->> 'evidence_mode' =
                    'consumed_historical_replay'
                AND target_backtest.metrics_json ->
                    'historical_replay_opened' = 'true'::jsonb
                AND target_backtest.metrics_json ->
                    'final_oos_opened' = 'true'::jsonb
                AND target_backtest.metrics_json ->
                    'capital_eligible' = 'false'::jsonb
                AND target_backtest.metrics_json ->
                    'consumed_oos_replayed' = 'true'::jsonb
                AND target_backtest.metrics_json ->
                    'sealed_final_oos' = 'false'::jsonb
                AND target_backtest.metrics_json -> 'unseen_oos' = 'false'::jsonb
                AND target_backtest.metrics_json ->> 'authority' =
                    'historical_description_only'
                AND (target_backtest.metrics_json ->>
                    'strategy_trial_count')::integer = NEW.strategy_trial_count
                AND (target_backtest.metrics_json -> 'formal_validation' ->
                    'multiple_testing' ->> 'trial_count')::integer =
                    NEW.strategy_trial_count
                AND target_backtest.metrics_json -> 'formal_validation' ->
                    'multiple_testing' ->> 'trial_count_audit_sha256' =
                    NEW.trial_count_audit_sha256
                AND target_backtest.metrics_json -> 'formal_validation' ->
                    'multiple_testing' ->> 'eligibility_receipt_sha256' =
                    NEW.incomplete_family_eligibility_sha256
                AND target_backtest.metrics_json -> 'provenance' ->>
                    'worker_runtime_image_digest' = NEW.worker_runtime_image_digest
                AND target_backtest.metrics_json -> 'provenance' ->>
                    'execution_manifest_sha256' = NEW.replay_manifest_sha256
                AND target_backtest.metrics_json -> 'provenance' ->>
                    'artifact_manifest_sha256' =
                    NEW.replay_artifact_manifest_sha256
                AND vintage.consumed_at IS NOT NULL
                AND vintage.test_start = DATE '2018-11-08'
                AND vintage.test_end = DATE '2019-11-20'
                AND vintage.dataset_identity = '{DATASET_IDENTITY}'
                AND vintage.dataset_lineage_id = '{DATASET_LINEAGE}'
                AND NEW.recipe_id = 'short_relative_strength'
                AND NEW.horizon_profile = 'short_1_5d'
                AND NEW.dataset = '{DATASET}'
                AND NEW.dataset_identity_sha256 = '{DATASET_IDENTITY}'
                AND NEW.dataset_lineage_id = '{DATASET_LINEAGE}'
                AND NEW.strategy_rules_sha256 = '{RULES_SHA256}'
                AND NEW.execution_contract_hash = '{EXECUTION_SHA256}'
                AND family_eligibility.receipt_sha256 =
                    NEW.incomplete_family_eligibility_sha256
                AND family_eligibility.strategy_trial_count =
                    NEW.strategy_trial_count
                AND family_eligibility.trial_count_audit_sha256 =
                    NEW.trial_count_audit_sha256
                AND NEW.qualification_json ->> 'contract_version' =
                    NEW.contract_version
                AND NEW.qualification_json ->> 'evidence_mode' = NEW.evidence_mode
                AND NEW.qualification_json ->> 'authority' = NEW.authority
                AND NEW.qualification_json ->> 'source_strategy_version_id' =
                    NEW.source_strategy_version_id
                AND NEW.qualification_json ->> 'source_backtest_id' =
                    NEW.source_backtest_id
                AND NEW.qualification_json ->> 'source_job_id' = NEW.source_job_id
                AND NEW.qualification_json ->>
                    'source_interruption_recovery_receipt_sha256' =
                    NEW.source_interruption_recovery_receipt_sha256
                AND NEW.qualification_json ->>
                    'source_interruption_receipt_authority' =
                    NEW.source_interruption_receipt_authority
                AND NEW.qualification_json ->> 'strategy_version_id' =
                    NEW.strategy_version_id
                AND NEW.qualification_json ->> 'backtest_id' = NEW.backtest_id
                AND NEW.qualification_json ->> 'consumed_oos_vintage_id' =
                    NEW.consumed_oos_vintage_id
                AND NEW.qualification_json ->> 'recipe_id' = NEW.recipe_id
                AND NEW.qualification_json ->> 'horizon_profile' =
                    NEW.horizon_profile
                AND NEW.qualification_json ->> 'dataset' = NEW.dataset
                AND NEW.qualification_json ->> 'dataset_identity_sha256' =
                    NEW.dataset_identity_sha256
                AND NEW.qualification_json ->> 'dataset_lineage_id' =
                    NEW.dataset_lineage_id
                AND NEW.qualification_json ->> 'strategy_rules_sha256' =
                    NEW.strategy_rules_sha256
                AND NEW.qualification_json ->> 'execution_contract_hash' =
                    NEW.execution_contract_hash
                AND NEW.qualification_json ->> 'runner_sha256' = NEW.runner_sha256
                AND NEW.qualification_json ->> 'runtime_bundle_sha256' =
                    NEW.runtime_bundle_sha256
                AND NEW.qualification_json ->> 'worker_runtime_image_digest' =
                    NEW.worker_runtime_image_digest
                AND NEW.qualification_json -> 'replay_periods' =
                    NEW.replay_periods_json
                AND NEW.qualification_json ->> 'replay_manifest_sha256' =
                    NEW.replay_manifest_sha256
                AND NEW.qualification_json ->> 'replay_result_sha256' =
                    NEW.replay_result_sha256
                AND NEW.qualification_json ->> 'replay_artifact_manifest_sha256' =
                    NEW.replay_artifact_manifest_sha256
                AND NEW.qualification_json ->> 'replay_daily_returns_sha256' =
                    NEW.replay_daily_returns_sha256
                AND (NEW.qualification_json ->> 'strategy_trial_count')::integer =
                    NEW.strategy_trial_count
                AND NEW.qualification_json ->> 'trial_count_audit_sha256' =
                    NEW.trial_count_audit_sha256
                AND NEW.qualification_json ->>
                    'incomplete_family_eligibility_sha256' =
                    NEW.incomplete_family_eligibility_sha256
                AND NEW.qualification_json -> 'forward_criteria' =
                    NEW.forward_criteria_json
                AND NEW.qualification_json ->> 'forward_criteria_sha256' =
                    NEW.forward_criteria_sha256
                AND jsonb_object_length(NEW.replay_periods_json) = 4
                AND NEW.replay_periods_json ->> 'historical_start' = '2008-01-02'
                AND NEW.replay_periods_json ->> 'historical_end' = '2018-10-10'
                AND NEW.replay_periods_json ->> 'start' = '2018-11-08'
                AND NEW.replay_periods_json ->> 'end' = '2019-11-20'
                AND source_audit.action =
                    'strategy.forward_only_rehabilitation_admitted'
                AND source_audit.status_code = 201
                AND source_audit.details_json ->> 'receipt_sha256' =
                    NEW.receipt_sha256
                AND source_audit.details_json ->> 'strategy_version_id' =
                    NEW.strategy_version_id
                AND source_audit.details_json ->> 'backtest_id' = NEW.backtest_id
            ) IS TRUE) THEN
                RAISE EXCEPTION
                    'forward-only rehabilitation does not match the exact consumed replay';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_validate_forward_only_rehabilitation
        BEFORE INSERT ON quantlab.strategy_forward_only_rehabilitations
        FOR EACH ROW EXECUTE FUNCTION quantlab.validate_forward_only_rehabilitation();
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.guard_forward_only_rehabilitation()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'forward-only rehabilitation receipts are append-only';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_forward_only_rehabilitation_append_only
        BEFORE UPDATE OR DELETE ON quantlab.strategy_forward_only_rehabilitations
        FOR EACH ROW EXECUTE FUNCTION quantlab.guard_forward_only_rehabilitation();
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_incomplete_family_eligibility_append_only
        BEFORE UPDATE OR DELETE ON quantlab.strategy_incomplete_family_eligibilities
        FOR EACH ROW EXECUTE FUNCTION quantlab.guard_forward_only_rehabilitation();
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_forward_only_rehabilitation_no_truncate
        BEFORE TRUNCATE ON quantlab.strategy_forward_only_rehabilitations
        FOR EACH STATEMENT EXECUTE FUNCTION quantlab.guard_forward_only_rehabilitation();
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_incomplete_family_eligibility_no_truncate
        BEFORE TRUNCATE ON quantlab.strategy_incomplete_family_eligibilities
        FOR EACH STATEMENT EXECUTE FUNCTION quantlab.guard_forward_only_rehabilitation();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION quantlab.validate_incomplete_family_eligibility()
        RETURNS trigger AS $$
        DECLARE
            target_version quantlab.strategy_versions%ROWTYPE;
            target_strategy quantlab.strategies%ROWTYPE;
        BEGIN
            PERFORM pg_advisory_xact_lock(
                hashtext('strategy-trial-family:' || NEW.economic_hypothesis_group)
            );
            SELECT * INTO target_version FROM quantlab.strategy_versions
            WHERE id = NEW.strategy_version_id FOR UPDATE;
            SELECT * INTO target_strategy FROM quantlab.strategies
            WHERE id = target_version.strategy_id FOR SHARE;
            IF NOT ((
                target_version.id IS NOT NULL
                AND target_strategy.id IS NOT NULL
                AND target_version.status = 'draft'
                AND target_version.promotion_stage IS NULL
                AND target_version.is_legacy = false
                AND target_version.evidence_mode = 'consumed_historical_replay'
                AND target_version.strategy_rules_sha256 = '{RULES_SHA256}'
                AND target_version.execution_contract_hash = '{EXECUTION_SHA256}'
                AND NEW.economic_hypothesis_group =
                    target_strategy.economic_hypothesis_group
                AND NEW.source_strategy_version_id = '{SOURCE_VERSION_ID}'
                AND NEW.source_backtest_id = '{SOURCE_BACKTEST_ID}'
                AND NEW.source_job_id = '{SOURCE_JOB_ID}'
                AND NEW.eligible_strategy_version_ids_json ? target_version.id
                AND NEW.eligible_strategy_version_ids_json ? '{SOURCE_VERSION_ID}'
                AND NEW.qualification_json ->> 'contract_version' =
                    NEW.contract_version
                AND NEW.qualification_json ->> 'evidence_mode' = NEW.evidence_mode
                AND NEW.qualification_json ->> 'authority' = NEW.authority
                AND NEW.qualification_json ->> 'source_strategy_version_id' =
                    NEW.source_strategy_version_id
                AND NEW.qualification_json ->> 'source_backtest_id' =
                    NEW.source_backtest_id
                AND NEW.qualification_json ->> 'source_job_id' = NEW.source_job_id
                AND NEW.qualification_json ->> 'strategy_version_id' =
                    NEW.strategy_version_id
                AND NEW.qualification_json ->> 'economic_hypothesis_group' =
                    NEW.economic_hypothesis_group
                AND NEW.qualification_json -> 'eligible_strategy_version_ids' =
                    NEW.eligible_strategy_version_ids_json
                AND (NEW.qualification_json ->> 'strategy_trial_count')::integer =
                    NEW.strategy_trial_count
                AND NEW.qualification_json -> 'trial_count_audit' =
                    NEW.trial_count_audit_json
                AND NEW.qualification_json ->> 'trial_count_audit_sha256' =
                    NEW.trial_count_audit_sha256
                AND NEW.qualification_json -> 'missing_artifacts' =
                    NEW.missing_artifacts_json
                AND (NEW.qualification_json ->> 'cutoff_at')::timestamptz =
                    NEW.cutoff_at
                AND NEW.qualification_json ->> 'receipt_sha256' =
                    NEW.receipt_sha256
            ) IS TRUE) THEN
                RAISE EXCEPTION 'incomplete factor-family eligibility is invalid';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_validate_incomplete_family_eligibility
        BEFORE INSERT ON quantlab.strategy_incomplete_family_eligibilities
        FOR EACH ROW EXECUTE FUNCTION quantlab.validate_incomplete_family_eligibility();
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.guard_strategy_evidence_mode()
        RETURNS trigger AS $$
        BEGIN
            IF NEW.evidence_mode IS DISTINCT FROM OLD.evidence_mode THEN
                RAISE EXCEPTION 'strategy evidence mode is immutable';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for table in ("strategy_versions", "backtest_runs"):
        op.execute(
            f"""
            CREATE TRIGGER trg_{table}_evidence_mode_immutable
            BEFORE UPDATE ON quantlab.{table}
            FOR EACH ROW EXECUTE FUNCTION quantlab.guard_strategy_evidence_mode();
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    receipt_count = bind.scalar(
        sa.text(
            "SELECT (SELECT count(*) FROM "
            "quantlab.strategy_forward_only_rehabilitations) + "
            "(SELECT count(*) FROM "
            "quantlab.strategy_incomplete_family_eligibilities)"
        )
    )
    if int(receipt_count or 0) > 0:
        raise RuntimeError(
            "cannot downgrade after immutable forward-only rehabilitation evidence exists"
        )
    for table in ("backtest_runs", "strategy_versions"):
        op.execute(
            f"DROP TRIGGER IF EXISTS trg_{table}_evidence_mode_immutable "
            f"ON quantlab.{table}"
        )
    op.execute("DROP FUNCTION IF EXISTS quantlab.guard_strategy_evidence_mode()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_forward_only_rehabilitation_append_only "
        "ON quantlab.strategy_forward_only_rehabilitations"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_incomplete_family_eligibility_append_only "
        "ON quantlab.strategy_incomplete_family_eligibilities"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_forward_only_rehabilitation_no_truncate "
        "ON quantlab.strategy_forward_only_rehabilitations"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_incomplete_family_eligibility_no_truncate "
        "ON quantlab.strategy_incomplete_family_eligibilities"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_validate_incomplete_family_eligibility "
        "ON quantlab.strategy_incomplete_family_eligibilities"
    )
    op.execute("DROP FUNCTION IF EXISTS quantlab.validate_incomplete_family_eligibility()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_validate_forward_only_rehabilitation "
        "ON quantlab.strategy_forward_only_rehabilitations"
    )
    op.execute("DROP FUNCTION IF EXISTS quantlab.guard_forward_only_rehabilitation()")
    op.execute("DROP FUNCTION IF EXISTS quantlab.validate_forward_only_rehabilitation()")
    op.drop_index(
        "idx_forward_only_rehabilitations_created",
        table_name="strategy_forward_only_rehabilitations",
        schema=SCHEMA,
    )
    op.drop_table("strategy_forward_only_rehabilitations", schema=SCHEMA)
    op.drop_index(
        "idx_incomplete_family_eligibilities_created",
        table_name="strategy_incomplete_family_eligibilities",
        schema=SCHEMA,
    )
    op.drop_table("strategy_incomplete_family_eligibilities", schema=SCHEMA)
    op.drop_constraint(
        "ck_strategy_versions_v18_runtime_identity",
        "strategy_versions",
        schema=SCHEMA,
        type_="check",
    )
    for table in ("backtest_runs", "strategy_versions"):
        op.drop_constraint(
            f"ck_{table}_evidence_mode",
            table,
            schema=SCHEMA,
            type_="check",
        )
        op.drop_column(table, "evidence_mode", schema=SCHEMA)
