"""Add one append-only v17 formal-backtest interruption recovery.

Revision ID: 0086_formal_bt_interrupt
Revises: 0085_baseline_v17_industry
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0086_formal_bt_interrupt"
down_revision: str | None = "0085_baseline_v17_industry"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
JSON = sa.JSON().with_variant(JSONB(), "postgresql")

CONTRACT_VERSION = "transparent-baseline-service-interruption-recovery-v1"
RECOVERY_GENERATION = "v17-control-plane-backup-sigterm-20260831"
REASON_CODE = "external_service_sigterm_before_formal_result"
BACKTEST_ID = "0a113fe28ca741b6be9c09ab046c9d02"
JOB_ID = "858a75a6f1994c359fa9c3567ed09f57"
STRATEGY_VERSION_ID = "4414d202dbb641608975e5305bc18da4"
PAYLOAD_SHA256 = "f99fa6c8f2a1364d56a0d0bff9d7401b1f504f3b020c321d996f05a70a05df42"
LOG_PREFIX_SHA256 = "c287af99368bf95d16330513f155f795b5fe77bc1e595c47d2fed7a1a447addb"
LOG_PREFIX_BYTES = 25_665
ARTIFACT_INVENTORY_SHA256 = (
    "c0272f59ca8878d26e95db2b4328cfc9551c4b3dff4382a237c38cd87f00f63e"
)
SOURCE_JOB_ROW_SHA256 = (
    "8a99cddcafb2554414c897e7bf11f6ba822490b76d77c78cb9a7d425422105e9"
)
SOURCE_BACKTEST_ROW_SHA256 = (
    "cfd64d7ff3006e1d3c5cbf8ed9f80f1efe9067ae8f876f8c1c21732573f95bc4"
)
RECEIPT_SHA256 = "6345c455862d8bb587e12f8ce0be7c1da291f2dde82fbdbb31e954bb88b9d3df"
RECOVERY_APPLICATION_NAME = "quantlab-v17-recovery-b41f78f9"
RECOVERY_AUTHORIZER_APPLICATION_NAME = "quantlab-v17-recovery-authorizer"
RECOVERY_CONTROLLER_SHA256 = (
    "1329aaf46bc9db8a3172abf6ecbce2837a12f1b555aad7c33c4ff0a9e70a2c0d"
)
EXTERNAL_OBSERVED_AT = "2026-08-30T19:30:09.189629+00:00"
EXTERNAL_JOURNAL_SHA256 = (
    "c7a93f1fe86a57ed6fcd51f32fe74be55ef7115cb4a79454e12ccbee338efcc7"
)
EXTERNAL_JOURNAL_EXCERPT_JSON = (
    '{"contract_version":"quantlab-systemd-docker-journal-excerpt-v1","records":['
    '{"message":"Starting quantlab-backup.service - QuantLab bounded control-plane '
    'backup...","observed_at":"2026-08-30T19:29:57.618523+00:00","source":"systemd",'
    '"unit":"quantlab-backup.service"},'
    '{"message":"Container quantlab-platform-evaluation-worker-1 Stopping",'
    '"observed_at":"2026-08-30T19:30:09.189629+00:00","source":"systemd",'
    '"unit":"quantlab-backup.service"},'
    '{"container_id":"710730b1ce29bdd43e081f967860e7f49a791e9e62e62aeeaa5e2b98e3f03218",'
    '"daemon_shutting_down":false,"exit_status":0,"has_been_manually_stopped":true,'
    '"observed_at":"2026-08-30T19:30:14.800407712+00:00","source":"dockerd"},'
    '{"message":"Container quantlab-platform-evaluation-worker-1 Stopped",'
    '"observed_at":"2026-08-30T19:30:14.869740+00:00","source":"systemd",'
    '"unit":"quantlab-backup.service"},'
    '{"message":"Container quantlab-platform-evaluation-worker-1 Started",'
    '"observed_at":"2026-08-30T19:34:33.195752+00:00","source":"systemd",'
    '"unit":"quantlab-backup.service"},'
    '{"message":"Finished quantlab-backup.service - QuantLab bounded control-plane backup.",'
    '"observed_at":"2026-08-30T19:35:56.019550+00:00","source":"systemd",'
    '"unit":"quantlab-backup.service"}]}'
)
EXTERNAL_JOURNAL_EXCERPT_SQL = EXTERNAL_JOURNAL_EXCERPT_JSON.replace(":", "\\:")
EXPECTED_RECEIPT_JSON = r"""{"contract_version":"transparent-baseline-service-interruption-recovery-v1","execution_controller":{"application_name":"quantlab-v17-recovery-b41f78f9","authorization_application_name":"quantlab-v17-recovery-authorizer","canonical_output_path":"/data/artifacts/backtests/0a113fe28ca741b6be9c09ab046c9d02","contract_version":"quantlab-v17-sealed-one-shot-controller-v1","controller_sha256":"1329aaf46bc9db8a3172abf6ecbce2837a12f1b555aad7c33c4ff0a9e70a2c0d","data_mount_mode":"volumes-from-read-only-with-persistent-target-samefile-bind","sealed_worker_image_digest":"sha256:b41f78f9c99dd9853998d85a52593bcab247907ac60f9d721d863e20904e8bb7","target_artifact_path":"/data/artifacts/formal-backtest-recoveries/0a113fe28ca741b6be9c09ab046c9d02/attempt-2","target_execution_log_mount_mode":"single-file-read-write-bind","target_execution_log_path":"/data/artifacts/formal-backtest-recoveries/0a113fe28ca741b6be9c09ab046c9d02/attempt-2.log"},"external_interruption":{"backtest_id":"0a113fe28ca741b6be9c09ab046c9d02","contract_version":"quantlab-external-service-interruption-evidence-v1","has_been_manually_stopped":true,"job_id":"858a75a6f1994c359fa9c3567ed09f57","journal_excerpt":{"contract_version":"quantlab-systemd-docker-journal-excerpt-v1","records":[{"message":"Starting quantlab-backup.service - QuantLab bounded control-plane backup...","observed_at":"2026-08-30T19:29:57.618523+00:00","source":"systemd","unit":"quantlab-backup.service"},{"message":"Container quantlab-platform-evaluation-worker-1 Stopping","observed_at":"2026-08-30T19:30:09.189629+00:00","source":"systemd","unit":"quantlab-backup.service"},{"container_id":"710730b1ce29bdd43e081f967860e7f49a791e9e62e62aeeaa5e2b98e3f03218","daemon_shutting_down":false,"exit_status":0,"has_been_manually_stopped":true,"observed_at":"2026-08-30T19:30:14.800407712+00:00","source":"dockerd"},{"message":"Container quantlab-platform-evaluation-worker-1 Stopped","observed_at":"2026-08-30T19:30:14.869740+00:00","source":"systemd","unit":"quantlab-backup.service"},{"message":"Container quantlab-platform-evaluation-worker-1 Started","observed_at":"2026-08-30T19:34:33.195752+00:00","source":"systemd","unit":"quantlab-backup.service"},{"message":"Finished quantlab-backup.service - QuantLab bounded control-plane backup.","observed_at":"2026-08-30T19:35:56.019550+00:00","source":"systemd","unit":"quantlab-backup.service"}]},"journal_sha256":"c7a93f1fe86a57ed6fcd51f32fe74be55ef7115cb4a79454e12ccbee338efcc7","observed_at":"2026-08-30T19:30:09.189629+00:00","oom_killed":false,"service":"evaluation-worker","signal":"SIGTERM","source":"systemd-docker-journal","stop_owner":"quantlab-backup.service"},"immutable_binding":{"dataset":"cn-20080101-20260828-v7-failclosed-ed5c8b3","dataset_identity_sha256":"eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2","dataset_lineage_id":"1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e","execution_contract_hash":"0ffa8939a70f0c46499e3ae472b877f68cce1b9fb95dfe81886ab8430ddcce61","periods":{"end":"2019-11-20","historical_end":"2018-10-10","historical_start":"2008-01-02","start":"2018-11-08"},"recipe_id":"short_relative_strength","recipe_sha256":"dee3551a73f2ebb3fbbbddfafdf99f4618e3dfdd98981fdb4b4d8849723d5fd8","recipe_version":"qlib-rdagent-single-mainline-2026-08-31-v17","runner_sha256":"31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3","runtime_bundle_sha256":"c3d6aeb00a8f5286ee117f885231b43ad49e461c5183f5cbf32c4601824bb9fc","source_repair_receipt_sha256":"980ea643755d261cc7ee39ffec5e23af3f8e27ebdc1e2f2a647e0802b9dc636d","strategy_rules_sha256":"644d9ee73ea4c167c7d8f58b2e8b1707289cb569c48ae73ce131cce139f7e756","strategy_version_id":"4414d202dbb641608975e5305bc18da4","worker_runtime_image_digest":"sha256:b41f78f9c99dd9853998d85a52593bcab247907ac60f9d721d863e20904e8bb7"},"performance_information_used":false,"pre_result_evidence":{"artifact_inventory":[{"bytes":57532420,"path":"baseline/composite.parquet","sha256":"3e682431fe9844927d30ba039409b12d733a9adc39c6a1192c4a74d6a222028d"},{"bytes":57206461,"path":"baseline/normalized/amount_expansion_5d.parquet","sha256":"0c7a7fec840be9d26153c2725f40345022cb531a2b918ff3568133a4799c68f3"},{"bytes":53130406,"path":"baseline/normalized/close_location_5d.parquet","sha256":"12ecd49faa1e6a5fb3926f827472c9ea161634f35ac94f6ab1504bb3c42129af"},{"bytes":57217048,"path":"baseline/normalized/extension_penalty_5d.parquet","sha256":"f70f2a59471187e5e54506210828c893e0c5d6076d2cad69c583c376573534c1"},{"bytes":55912111,"path":"baseline/normalized/relative_strength_5d.parquet","sha256":"50c49cbc1223554ba3b0b1a3cd6667cded6407edb89b257b61cc81a26ad3ac12"},{"bytes":33356095,"path":"baseline/raw/amount_expansion_5d.parquet","sha256":"d166122759b5ea42ed582fe217867de2937f62d03142d6d8630b2bd10d24fe2d"},{"bytes":30035531,"path":"baseline/raw/close_location_5d.parquet","sha256":"3be7552ff64404260908b1cc196524fe06dd2f7c6dbc3b2478edd956521f48f2"},{"bytes":32735138,"path":"baseline/raw/extension_penalty_5d.parquet","sha256":"a7697b9a94e333881c2921071e12e28a8adf09a33a2411afe97fb352593e0d8c"},{"bytes":29600893,"path":"baseline/raw/relative_strength_5d.parquet","sha256":"55fddb6562cd756f82f18c268e82a2719fccfd46986996de83826e8f9ef11836"},{"bytes":69327,"path":"manifest.json","sha256":"a4b72701a88247a7bf783b3bfe2950131c936d6d3dcea7d42601ab5bb8d115c7"}],"artifact_inventory_sha256":"c0272f59ca8878d26e95db2b4328cfc9551c4b3dff4382a237c38cd87f00f63e","artifact_path":"/data/artifacts/backtests/0a113fe28ca741b6be9c09ab046c9d02","backtest_metrics_absent":true,"job_progress_absent":true,"log_prefix":{"bytes":25665,"path":"/data/platform/logs/strategy-backtest-0a113fe28ca741b6be9c09ab046c9d02.log","sha256":"c287af99368bf95d16330513f155f795b5fe77bc1e595c47d2fed7a1a447addb"},"manifest_sha256":"a4b72701a88247a7bf783b3bfe2950131c936d6d3dcea7d42601ab5bb8d115c7","result_absent":true,"terminal_artifacts_absent":["artifact_manifest.json","daily_returns.parquet","result.json"]},"reason_code":"external_service_sigterm_before_formal_result","receipt_sha256":"6345c455862d8bb587e12f8ce0be7c1da291f2dde82fbdbb31e954bb88b9d3df","recovery_generation":"v17-control-plane-backup-sigterm-20260831","source_backtest":{"artifact_path":"/data/artifacts/backtests/0a113fe28ca741b6be9c09ab046c9d02","created_at":"2026-08-30T18:56:25.223043+00:00","dataset":"cn-20080101-20260828-v7-failclosed-ed5c8b3","error_absent":true,"execution_contract_hash":"0ffa8939a70f0c46499e3ae472b877f68cce1b9fb95dfe81886ab8430ddcce61","execution_dataset":null,"finished_at_absent":true,"id":"0a113fe28ca741b6be9c09ab046c9d02","job_id":"858a75a6f1994c359fa9c3567ed09f57","metrics_absent":true,"periods":{"end":"2019-11-20","historical_end":"2018-10-10","historical_start":"2008-01-02","start":"2018-11-08"},"qlib_commit":"d5379c520f66a39953bad76234a7019a72796fd0","qlib_version":"0.0.dev0+gd5379c520f66a39953bad76234a7019a72796fd0","rdagent_commit":"4f9ecb005881cddc08df0124a2e894c018007679","rdagent_version":"0.0.dev0+g4f9ecb005881cddc08df0124a2e894c018007679","row_sha256":"cfd64d7ff3006e1d3c5cbf8ed9f80f1efe9067ae8f876f8c1c21732573f95bc4","started_at":"2026-08-30T18:56:26.196535+00:00","status":"running","strategy_version_id":"4414d202dbb641608975e5305bc18da4"},"source_job":{"attempts":1,"created_at":"2026-08-30T18:56:25.236023+00:00","error":"Worker restarted after the bounded attempt limit; operator review is required","exit_code":143,"finished_at":"2026-08-30T19:34:35.198471+00:00","id":"858a75a6f1994c359fa9c3567ed09f57","idempotency_key":"transparent-baseline:4414d202dbb641608975e5305bc18da4:0a113fe28ca741b6be9c09ab046c9d02","kind":"strategy_backtest","log_path":"/data/platform/logs/strategy-backtest-0a113fe28ca741b6be9c09ab046c9d02.log","max_attempts":1,"payload_sha256":"f99fa6c8f2a1364d56a0d0bff9d7401b1f504f3b020c321d996f05a70a05df42","progress_absent":true,"row_sha256":"8a99cddcafb2554414c897e7bf11f6ba822490b76d77c78cb9a7d425422105e9","started_at":"2026-08-30T18:56:26.193585+00:00","status":"failed"},"target":{"artifact_path":"/data/artifacts/formal-backtest-recoveries/0a113fe28ca741b6be9c09ab046c9d02/attempt-2","authorized_attempt":2,"backtest_id":"0a113fe28ca741b6be9c09ab046c9d02","job_id":"858a75a6f1994c359fa9c3567ed09f57","max_attempts":2,"same_formal_oos_identity":true,"strategy_version_id":"4414d202dbb641608975e5305bc18da4"}}"""
EXPECTED_RECEIPT_SQL = EXPECTED_RECEIPT_JSON.replace(":", "\\:")
EXPECTED_PAYLOAD_JSON = (
    '{"backtest_id":"0a113fe28ca741b6be9c09ab046c9d02",'
    '"dataset":"cn-20080101-20260828-v7-failclosed-ed5c8b3",'
    '"dataset_path":"/data/qlib/cn-20080101-20260828-v7-failclosed-ed5c8b3",'
    '"execution_dataset":null,"periods":{"end":"2019-11-20",'
    '"historical_end":"2018-10-10","historical_start":"2008-01-02",'
    '"start":"2018-11-08"},'
    '"strategy_version_id":"4414d202dbb641608975e5305bc18da4",'
    '"transparent_baseline_runner_sha256":'
    '"31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3",'
    '"transparent_baseline_runtime_bundle_sha256":'
    '"c3d6aeb00a8f5286ee117f885231b43ad49e461c5183f5cbf32c4601824bb9fc",'
    '"transparent_baseline_worker_runtime_image_digest":'
    '"sha256:b41f78f9c99dd9853998d85a52593bcab247907ac60f9d721d863e20904e8bb7"}'
)
EXPECTED_PAYLOAD_SQL = EXPECTED_PAYLOAD_JSON.replace(":", "\\:")
EXPECTED_PERIODS_JSON = (
    '{"end":"2019-11-20","historical_end":"2018-10-10",'
    '"historical_start":"2008-01-02","start":"2018-11-08"}'
)
EXPECTED_PERIODS_SQL = EXPECTED_PERIODS_JSON.replace(":", "\\:")
TARGET_ARTIFACT_PATH = (
    "/data/artifacts/formal-backtest-recoveries/"
    "0a113fe28ca741b6be9c09ab046c9d02/attempt-2"
)


def _production_source_constraint() -> str:
    return (
        f"(contract_version = '{CONTRACT_VERSION}' "
        f"AND recovery_generation = '{RECOVERY_GENERATION}' "
        f"AND reason_code = '{REASON_CODE}' "
        f"AND backtest_id = '{BACKTEST_ID}' "
        f"AND job_id = '{JOB_ID}' "
        f"AND strategy_version_id = '{STRATEGY_VERSION_ID}' "
        f"AND receipt_sha256 = '{RECEIPT_SHA256}' "
        f"AND source_job_row_sha256 = '{SOURCE_JOB_ROW_SHA256}' "
        f"AND source_backtest_row_sha256 = '{SOURCE_BACKTEST_ROW_SHA256}' "
        f"AND source_payload_sha256 = '{PAYLOAD_SHA256}' "
        f"AND source_log_prefix_sha256 = '{LOG_PREFIX_SHA256}' "
        f"AND source_log_prefix_bytes = {LOG_PREFIX_BYTES} "
        f"AND source_artifact_inventory_sha256 = '{ARTIFACT_INVENTORY_SHA256}' "
        f"AND target_artifact_path = '{TARGET_ARTIFACT_PATH}') IS TRUE"
    )


def _receipt_constraint() -> str:
    external = "verification_json -> 'external_interruption'"
    return (
        "(jsonb_typeof(verification_json) = 'object' "
        "AND verification_json ->> 'receipt_sha256' = receipt_sha256 "
        "AND verification_json ->> 'contract_version' = contract_version "
        "AND verification_json ->> 'recovery_generation' = recovery_generation "
        "AND verification_json ->> 'reason_code' = reason_code "
        "AND (verification_json ->> 'performance_information_used')::boolean = false "
        "AND verification_json -> 'source_job' ->> 'id' = job_id "
        "AND verification_json -> 'source_job' ->> 'row_sha256' = "
        "source_job_row_sha256 "
        "AND verification_json -> 'source_backtest' ->> 'id' = backtest_id "
        "AND verification_json -> 'source_backtest' ->> 'row_sha256' = "
        "source_backtest_row_sha256 "
        "AND verification_json -> 'target' ->> 'job_id' = job_id "
        "AND verification_json -> 'target' ->> 'backtest_id' = backtest_id "
        "AND verification_json -> 'target' ->> 'strategy_version_id' = "
        "strategy_version_id "
        "AND verification_json -> 'target' ->> 'artifact_path' = "
        "target_artifact_path "
        "AND (verification_json -> 'target' ->> 'authorized_attempt')::integer = 2 "
        "AND (verification_json -> 'target' ->> 'max_attempts')::integer = 2 "
        f"AND {external} ->> 'service' = 'evaluation-worker' "
        f"AND {external} ->> 'stop_owner' = 'quantlab-backup.service' "
        f"AND {external} ->> 'signal' = 'SIGTERM' "
        f"AND ({external} ->> 'has_been_manually_stopped')::boolean = true "
        f"AND ({external} ->> 'oom_killed')::boolean = false "
        f"AND {external} ->> 'observed_at' = '{EXTERNAL_OBSERVED_AT}' "
        f"AND {external} ->> 'journal_sha256' = '{EXTERNAL_JOURNAL_SHA256}' "
        f"AND {external} -> 'journal_excerpt' = "
        f"'{EXTERNAL_JOURNAL_EXCERPT_SQL}'::jsonb) IS TRUE"
    )


def _exact_receipt_constraint() -> str:
    return f"(verification_json = '{EXPECTED_RECEIPT_SQL}'::jsonb) IS TRUE"


def upgrade() -> None:
    op.create_table(
        "formal_backtest_interruption_recoveries",
        sa.Column("receipt_sha256", sa.String(), primary_key=True),
        sa.Column(
            "source_audit_event_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{SCHEMA}.audit_events.id", ondelete="RESTRICT"),
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
            "job_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.jobs.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "strategy_version_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.strategy_versions.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("contract_version", sa.String(), nullable=False),
        sa.Column("recovery_generation", sa.String(), nullable=False, unique=True),
        sa.Column("reason_code", sa.String(), nullable=False),
        sa.Column("source_job_row_sha256", sa.String(), nullable=False, unique=True),
        sa.Column("source_backtest_row_sha256", sa.String(), nullable=False, unique=True),
        sa.Column("source_payload_sha256", sa.String(), nullable=False),
        sa.Column("source_log_prefix_sha256", sa.String(), nullable=False),
        sa.Column("source_log_prefix_bytes", sa.BigInteger(), nullable=False),
        sa.Column("source_artifact_inventory_sha256", sa.String(), nullable=False),
        sa.Column("target_artifact_path", sa.Text(), nullable=False),
        sa.Column("verification_json", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "receipt_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_job_row_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_backtest_row_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_payload_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_log_prefix_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_artifact_inventory_sha256 ~ '^[0-9a-f]{64}$' "
            "AND source_log_prefix_bytes > 0",
            name="ck_formal_backtest_interruption_recovery_sha256",
        ),
        sa.CheckConstraint(
            _production_source_constraint(),
            name="ck_formal_backtest_interruption_recovery_v17_source",
        ),
        sa.CheckConstraint(
            _receipt_constraint(),
            name="ck_formal_backtest_interruption_recovery_receipt",
        ),
        sa.CheckConstraint(
            _exact_receipt_constraint(),
            name="ck_formal_backtest_interruption_recovery_exact_receipt",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_formal_backtest_interruption_recoveries_created",
        "formal_backtest_interruption_recoveries",
        [sa.text("created_at DESC")],
        schema=SCHEMA,
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION quantlab.validate_formal_backtest_interruption_recovery()
        RETURNS trigger AS $$
        DECLARE
            source_job quantlab.jobs%ROWTYPE;
            source_backtest quantlab.backtest_runs%ROWTYPE;
            source_version quantlab.strategy_versions%ROWTYPE;
            source_audit quantlab.audit_events%ROWTYPE;
        BEGIN
            IF NOT ((NEW.verification_json = '{EXPECTED_RECEIPT_SQL}'::jsonb) IS TRUE) THEN
                RAISE EXCEPTION 'formal backtest interruption receipt JSON is not exact';
            END IF;
            SELECT * INTO source_job FROM quantlab.jobs
            WHERE id = '{JOB_ID}' FOR UPDATE;
            IF NOT ((
                FOUND
                AND source_job.id = '{JOB_ID}'
                AND source_job.kind = 'strategy_backtest'
                AND source_job.idempotency_key =
                    'transparent-baseline:{STRATEGY_VERSION_ID}:{BACKTEST_ID}'
                AND source_job.status = 'failed'
                AND source_job.attempts = 1
                AND source_job.max_attempts = 1
                AND source_job.exit_code = 143
                AND source_job.error =
                    'Worker restarted after the bounded attempt limit; operator review is required'
                AND source_job.payload_json = '{EXPECTED_PAYLOAD_SQL}'::jsonb
                AND source_job.progress_json IS NULL
                AND source_job.next_attempt_at IS NULL
                AND source_job.cancel_requested_at IS NULL
                AND source_job.log_path =
                    '/data/platform/logs/strategy-backtest-{BACKTEST_ID}.log'
                AND source_job.created_at =
                    TIMESTAMPTZ '2026-08-30 18:56:25.236023+00:00'
                AND source_job.started_at =
                    TIMESTAMPTZ '2026-08-30 18:56:26.193585+00:00'
                AND source_job.finished_at =
                    TIMESTAMPTZ '2026-08-30 19:34:35.198471+00:00'
            ) IS TRUE) THEN
                RAISE EXCEPTION 'formal backtest interruption source job is not exact';
            END IF;
            SELECT * INTO source_backtest FROM quantlab.backtest_runs
            WHERE id = '{BACKTEST_ID}' FOR UPDATE;
            IF NOT ((
                FOUND
                AND source_backtest.job_id = '{JOB_ID}'
                AND source_backtest.strategy_version_id = '{STRATEGY_VERSION_ID}'
                AND source_backtest.status = 'running'
                AND source_backtest.dataset =
                    'cn-20080101-20260828-v7-failclosed-ed5c8b3'
                AND source_backtest.execution_dataset IS NULL
                AND source_backtest.periods_json = '{EXPECTED_PERIODS_SQL}'::jsonb
                AND source_backtest.artifact_path =
                    '/data/artifacts/backtests/{BACKTEST_ID}'
                AND source_backtest.metrics_json IS NULL
                AND source_backtest.error IS NULL
                AND source_backtest.finished_at IS NULL
                AND source_backtest.execution_contract_hash =
                    '0ffa8939a70f0c46499e3ae472b877f68cce1b9fb95dfe81886ab8430ddcce61'
                AND source_backtest.qlib_version =
                    '0.0.dev0+gd5379c520f66a39953bad76234a7019a72796fd0'
                AND source_backtest.qlib_commit =
                    'd5379c520f66a39953bad76234a7019a72796fd0'
                AND source_backtest.rdagent_version =
                    '0.0.dev0+g4f9ecb005881cddc08df0124a2e894c018007679'
                AND source_backtest.rdagent_commit =
                    '4f9ecb005881cddc08df0124a2e894c018007679'
                AND source_backtest.created_at =
                    TIMESTAMPTZ '2026-08-30 18:56:25.223043+00:00'
                AND source_backtest.started_at =
                    TIMESTAMPTZ '2026-08-30 18:56:26.196535+00:00'
            ) IS TRUE) THEN
                RAISE EXCEPTION 'formal backtest interruption source aggregate is not exact';
            END IF;
            SELECT * INTO source_version FROM quantlab.strategy_versions
            WHERE id = '{STRATEGY_VERSION_ID}' FOR UPDATE;
            IF NOT ((
                FOUND
                AND source_version.status = 'draft'
                AND source_version.strategy_type = 'multifactor'
                AND source_version.horizon_profile = 'short_1_5d'
                AND source_version.strategy_rules_sha256 =
                    '644d9ee73ea4c167c7d8f58b2e8b1707289cb569c48ae73ce131cce139f7e756'
                AND source_version.execution_contract_hash =
                    '0ffa8939a70f0c46499e3ae472b877f68cce1b9fb95dfe81886ab8430ddcce61'
                AND source_version.config_json ->> 'recipe_id' = 'short_relative_strength'
                AND source_version.config_json ->> 'recipe_version' =
                    'qlib-rdagent-single-mainline-2026-08-31-v17'
                AND source_version.config_json ->> 'recipe_sha256' =
                    'dee3551a73f2ebb3fbbbddfafdf99f4618e3dfdd98981fdb4b4d8849723d5fd8'
            ) IS TRUE) THEN
                RAISE EXCEPTION 'formal backtest interruption strategy version is not exact';
            END IF;
            SELECT * INTO source_audit FROM quantlab.audit_events
            WHERE id = NEW.source_audit_event_id FOR UPDATE;
            IF NOT ((
                FOUND
                AND source_audit.action = 'formal_backtest_interruption_recovery_registered'
                AND source_audit.method = 'INTERNAL'
                AND source_audit.path =
                    'transparent-baseline/formal-backtest-interruption-recovery'
                AND source_audit.status_code = 201
                AND source_audit.user_id IS NULL
                AND source_audit.ip_hash IS NULL
                AND source_audit.username IS NOT NULL
                AND btrim(source_audit.username) <> ''
                AND source_audit.user_agent =
                    'recover_transparent_baseline_v17_interruption.py'
                AND source_audit.details_json = NEW.verification_json
                AND source_audit.created_at = NEW.created_at
            ) IS TRUE) THEN
                RAISE EXCEPTION 'formal backtest interruption audit event is not exact';
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM quantlab.transparent_baseline_pre_result_repairs
                WHERE receipt_sha256 =
                    '980ea643755d261cc7ee39ffec5e23af3f8e27ebdc1e2f2a647e0802b9dc636d'
            ) THEN
                RAISE EXCEPTION 'formal backtest interruption source repair is missing';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_validate_formal_backtest_interruption_recovery
        BEFORE INSERT ON quantlab.formal_backtest_interruption_recoveries
        FOR EACH ROW EXECUTE FUNCTION
            quantlab.validate_formal_backtest_interruption_recovery();
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.guard_formal_backtest_interruption_recovery()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'formal backtest interruption recovery receipts are append-only';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_formal_backtest_interruption_recovery
        BEFORE UPDATE OR DELETE ON quantlab.formal_backtest_interruption_recoveries
        FOR EACH ROW EXECUTE FUNCTION quantlab.guard_formal_backtest_interruption_recovery();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION quantlab.guard_formal_backtest_job_attempts()
        RETURNS trigger AS $$
        DECLARE
            recovery_exists boolean;
            initial_authorization boolean;
            restart_finalization boolean;
            runtime_authorized boolean;
            authorizer_authorized boolean;
        BEGIN
            IF OLD.kind IS DISTINCT FROM NEW.kind
               AND (OLD.kind = 'strategy_backtest' OR NEW.kind = 'strategy_backtest') THEN
                RAISE EXCEPTION 'formal backtest job kind is immutable';
            END IF;
            IF OLD.kind = 'strategy_backtest'
               AND (
                   NEW.id IS DISTINCT FROM OLD.id
                   OR NEW.payload_json IS DISTINCT FROM OLD.payload_json
                   OR NEW.idempotency_key IS DISTINCT FROM OLD.idempotency_key
                   OR NEW.log_path IS DISTINCT FROM OLD.log_path
                   OR NEW.created_at IS DISTINCT FROM OLD.created_at
               ) THEN
                RAISE EXCEPTION 'formal backtest job immutable identity changed';
            END IF;
            IF OLD.kind <> 'strategy_backtest' AND NEW.kind <> 'strategy_backtest' THEN
                RETURN NEW;
            END IF;

            SELECT EXISTS(
                SELECT 1
                FROM quantlab.formal_backtest_interruption_recoveries recovery
                WHERE recovery.job_id = OLD.id
                  AND recovery.backtest_id = '{BACKTEST_ID}'
                  AND recovery.strategy_version_id = '{STRATEGY_VERSION_ID}'
                  AND recovery.receipt_sha256 = '{RECEIPT_SHA256}'
                  AND recovery.source_job_row_sha256 = '{SOURCE_JOB_ROW_SHA256}'
                  AND recovery.source_backtest_row_sha256 =
                      '{SOURCE_BACKTEST_ROW_SHA256}'
                  AND recovery.source_payload_sha256 = '{PAYLOAD_SHA256}'
                  AND recovery.source_log_prefix_sha256 = '{LOG_PREFIX_SHA256}'
                  AND recovery.source_log_prefix_bytes = {LOG_PREFIX_BYTES}
                  AND recovery.source_artifact_inventory_sha256 =
                      '{ARTIFACT_INVENTORY_SHA256}'
                  AND recovery.target_artifact_path = '{TARGET_ARTIFACT_PATH}'
                  AND recovery.verification_json = '{EXPECTED_RECEIPT_SQL}'::jsonb
            ) INTO recovery_exists;

            runtime_authorized := (
                current_setting('application_name', true) =
                    '{RECOVERY_APPLICATION_NAME}'
            );
            authorizer_authorized := (
                current_setting('application_name', true) =
                    '{RECOVERY_AUTHORIZER_APPLICATION_NAME}'
            );

            initial_authorization := (
                OLD.id = '{JOB_ID}'
                AND OLD.status = 'failed'
                AND OLD.attempts = 1
                AND OLD.max_attempts = 1
                AND NEW.status = 'queued'
                AND NEW.attempts = 1
                AND NEW.max_attempts = 2
                AND NEW.payload_json = OLD.payload_json
                AND NEW.progress_json IS NULL
                AND NEW.exit_code IS NULL
                AND NEW.error IS NULL
                AND NEW.started_at IS NULL
                AND NEW.finished_at IS NULL
                AND NEW.cancel_requested_at IS NULL
                AND NEW.next_attempt_at IS NULL
                AND recovery_exists
                AND authorizer_authorized
            );
            restart_finalization := (
                OLD.id = '{JOB_ID}'
                AND OLD.status = 'running'
                AND OLD.attempts = 2
                AND OLD.max_attempts = 2
                AND NEW.status = 'failed'
                AND NEW.attempts = 2
                AND NEW.max_attempts = 2
                AND NEW.exit_code = 143
                AND NEW.error =
                    'Worker restarted after the bounded attempt limit; operator review is required'
                AND NEW.finished_at IS NOT NULL
                AND NEW.payload_json = OLD.payload_json
                AND recovery_exists
            );

            IF recovery_exists
               AND OLD.id = '{JOB_ID}'
               AND OLD.status = 'failed'
               AND OLD.attempts = 1
               AND OLD.max_attempts = 1 THEN
                IF initial_authorization IS NOT TRUE THEN
                    RAISE EXCEPTION 'formal backtest recovery authorization is not exact';
                END IF;
                RETURN NEW;
            END IF;
            IF recovery_exists
               AND OLD.id = '{JOB_ID}'
               AND OLD.status = 'queued'
               AND OLD.attempts = 1
               AND OLD.max_attempts = 2 THEN
                IF NOT ((
                    runtime_authorized
                    AND NEW.status = 'running'
                    AND NEW.attempts = 2
                    AND NEW.max_attempts = 2
                    AND NEW.progress_json IS NOT DISTINCT FROM OLD.progress_json
                    AND NEW.exit_code IS NULL
                    AND NEW.error IS NULL
                    AND NEW.started_at IS NOT NULL
                    AND NEW.finished_at IS NULL
                    AND NEW.cancel_requested_at IS NULL
                    AND NEW.next_attempt_at IS NULL
                ) IS TRUE) THEN
                    RAISE EXCEPTION 'formal backtest recovery queued job is frozen';
                END IF;
                RETURN NEW;
            END IF;

            IF NEW.attempts < OLD.attempts THEN
                RAISE EXCEPTION 'formal backtest attempts are monotonic';
            END IF;
            IF NEW.attempts > OLD.attempts
               AND NOT (
                   OLD.status = 'queued'
                   AND NEW.status = 'running'
                   AND NEW.attempts = OLD.attempts + 1
               ) THEN
                RAISE EXCEPTION 'formal backtest attempts advance only on claim';
            END IF;
            IF OLD.status = 'queued'
               AND NEW.status = 'running'
               AND NEW.attempts <> OLD.attempts + 1 THEN
                RAISE EXCEPTION 'formal backtest running transition requires one claim';
            END IF;
            IF NEW.max_attempts IS DISTINCT FROM OLD.max_attempts
               AND initial_authorization IS NOT TRUE THEN
                RAISE EXCEPTION 'formal backtest attempt budget is immutable';
            END IF;
            IF OLD.attempts > 1
               AND OLD.status IN ('succeeded', 'failed', 'cancelled') THEN
                RAISE EXCEPTION 'formal backtest attempt 2 terminal row is immutable';
            END IF;
            IF OLD.status IN ('succeeded', 'failed', 'cancelled')
               AND NEW.status IS DISTINCT FROM OLD.status
               AND initial_authorization IS NOT TRUE THEN
                RAISE EXCEPTION 'formal backtest terminal jobs are immutable';
            END IF;
            IF OLD.attempts > 1 OR NEW.attempts > 1 THEN
                IF OLD.id <> '{JOB_ID}'
                   OR NOT recovery_exists
                   OR NEW.attempts > 2
                   OR NEW.max_attempts <> 2
                   OR NEW.payload_json IS DISTINCT FROM OLD.payload_json THEN
                    RAISE EXCEPTION 'formal backtest repeated execution is not authorized';
                END IF;
                IF runtime_authorized IS NOT TRUE
                   AND restart_finalization IS NOT TRUE THEN
                    RAISE EXCEPTION 'formal backtest attempt 2 runtime is not authorized';
                END IF;
                IF OLD.attempts = 1
                   AND NOT (
                       OLD.status = 'queued'
                       AND NEW.status = 'running'
                       AND NEW.attempts = 2
                   ) THEN
                    RAISE EXCEPTION 'formal backtest attempt 2 must begin with one exact claim';
                END IF;
                IF OLD.attempts = 2
                   AND OLD.status = 'running'
                   AND NEW.status NOT IN ('running', 'succeeded', 'failed', 'cancelled') THEN
                    RAISE EXCEPTION 'formal backtest attempt 2 transition is invalid';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_guard_formal_backtest_job_attempts
        BEFORE UPDATE ON quantlab.jobs
        FOR EACH ROW EXECUTE FUNCTION quantlab.guard_formal_backtest_job_attempts();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION quantlab.guard_formal_backtest_recovery_aggregate()
        RETURNS trigger AS $$
        DECLARE
            recovery_exists boolean;
            runtime_authorized boolean;
            authorizer_authorized boolean;
            initial_authorization boolean;
            restart_finalization boolean;
            controller_failure_finalization boolean;
            current_job quantlab.jobs%ROWTYPE;
        BEGIN
            IF OLD.id <> '{BACKTEST_ID}' AND NEW.id <> '{BACKTEST_ID}' THEN
                RETURN NEW;
            END IF;
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.job_id IS DISTINCT FROM OLD.job_id
               OR NEW.strategy_version_id IS DISTINCT FROM OLD.strategy_version_id
               OR NEW.dataset IS DISTINCT FROM OLD.dataset
               OR NEW.execution_dataset IS DISTINCT FROM OLD.execution_dataset
               OR NEW.periods_json IS DISTINCT FROM OLD.periods_json
               OR NEW.execution_contract_hash IS DISTINCT FROM OLD.execution_contract_hash
               OR NEW.qlib_version IS DISTINCT FROM OLD.qlib_version
               OR NEW.qlib_commit IS DISTINCT FROM OLD.qlib_commit
               OR NEW.rdagent_version IS DISTINCT FROM OLD.rdagent_version
               OR NEW.rdagent_commit IS DISTINCT FROM OLD.rdagent_commit
               OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
                RAISE EXCEPTION 'formal backtest recovery aggregate identity changed';
            END IF;
            SELECT EXISTS(
                SELECT 1 FROM quantlab.formal_backtest_interruption_recoveries recovery
                WHERE recovery.backtest_id = '{BACKTEST_ID}'
                  AND recovery.job_id = '{JOB_ID}'
                  AND recovery.receipt_sha256 = '{RECEIPT_SHA256}'
                  AND recovery.verification_json = '{EXPECTED_RECEIPT_SQL}'::jsonb
            ) INTO recovery_exists;
            SELECT * INTO current_job FROM quantlab.jobs WHERE id = '{JOB_ID}';
            runtime_authorized := (
                current_setting('application_name', true) =
                    '{RECOVERY_APPLICATION_NAME}'
            );
            authorizer_authorized := (
                current_setting('application_name', true) =
                    '{RECOVERY_AUTHORIZER_APPLICATION_NAME}'
            );
            initial_authorization := (
                recovery_exists
                AND OLD.status = 'running'
                AND OLD.artifact_path = '/data/artifacts/backtests/{BACKTEST_ID}'
                AND OLD.metrics_json IS NULL
                AND OLD.error IS NULL
                AND OLD.finished_at IS NULL
                AND NEW.status = 'queued'
                AND NEW.artifact_path = '{TARGET_ARTIFACT_PATH}'
                AND NEW.metrics_json IS NULL
                AND NEW.error IS NULL
                AND NEW.started_at IS NULL
                AND NEW.finished_at IS NULL
                AND current_job.status = 'failed'
                AND current_job.attempts = 1
                AND current_job.max_attempts = 1
                AND authorizer_authorized
            );
            restart_finalization := (
                recovery_exists
                AND OLD.status IN ('queued', 'running')
                AND NEW.status = 'failed'
                AND NEW.artifact_path = OLD.artifact_path
                AND NEW.metrics_json IS NULL
                AND NEW.error =
                    'Worker restarted after the bounded attempt limit; operator review is required'
                AND NEW.finished_at IS NOT NULL
                AND current_job.status = 'failed'
                AND current_job.attempts = 2
                AND current_job.max_attempts = 2
                AND current_job.exit_code = 143
                AND current_job.error = NEW.error
            );
            controller_failure_finalization := (
                recovery_exists
                AND runtime_authorized
                AND OLD.status IN (
                    'queued', 'running', 'succeeded', 'failed', 'cancelled'
                )
                AND OLD.artifact_path = '{TARGET_ARTIFACT_PATH}'
                AND NEW.status = 'failed'
                AND NEW.artifact_path = OLD.artifact_path
                AND NEW.metrics_json IS NOT DISTINCT FROM OLD.metrics_json
                AND NEW.error =
                    'Sealed v17 recovery controller terminated before worker '
                    'finalization; no further execution is authorized'
                AND NEW.started_at IS NOT DISTINCT FROM OLD.started_at
                AND (
                    (
                        OLD.status IN ('queued', 'running')
                        AND OLD.finished_at IS NULL
                        AND NEW.finished_at IS NOT NULL
                    )
                    OR (
                        OLD.status IN ('succeeded', 'failed', 'cancelled')
                        AND OLD.finished_at IS NOT NULL
                        AND NEW.finished_at IS NOT DISTINCT FROM OLD.finished_at
                    )
                )
                AND (
                    OLD.status <> 'succeeded'
                    OR (OLD.metrics_json IS NOT NULL AND OLD.error IS NULL)
                )
                AND current_job.status IN ('failed', 'cancelled')
                AND current_job.attempts = 2
                AND current_job.max_attempts = 2
                AND current_job.finished_at IS NOT NULL
            );
            IF initial_authorization IS TRUE THEN
                RETURN NEW;
            END IF;
            IF recovery_exists
               AND OLD.status = 'queued'
               AND OLD.artifact_path = '{TARGET_ARTIFACT_PATH}'
               AND current_job.status = 'running'
               AND current_job.attempts = 2
               AND current_job.max_attempts = 2 THEN
                IF NOT ((
                    runtime_authorized
                    AND NEW.status = 'running'
                    AND NEW.artifact_path = OLD.artifact_path
                    AND NEW.metrics_json IS NULL
                    AND NEW.error IS NULL
                    AND NEW.started_at IS NOT NULL
                    AND NEW.finished_at IS NULL
                ) IS TRUE) THEN
                    RAISE EXCEPTION 'formal backtest recovery queued aggregate is frozen';
                END IF;
                RETURN NEW;
            END IF;
            IF controller_failure_finalization IS TRUE THEN
                RETURN NEW;
            END IF;
            IF NOT ((
                recovery_exists
                AND OLD.artifact_path = '{TARGET_ARTIFACT_PATH}'
                AND NEW.artifact_path = '{TARGET_ARTIFACT_PATH}'
                AND current_job.attempts = 2
                AND current_job.max_attempts = 2
            ) IS TRUE) THEN
                RAISE EXCEPTION 'formal backtest recovery aggregate is not authorized';
            END IF;
            IF OLD.status IN ('succeeded', 'failed', 'cancelled') THEN
                RAISE EXCEPTION 'formal backtest recovery terminal aggregate is immutable';
            END IF;
            IF runtime_authorized IS NOT TRUE
               AND restart_finalization IS NOT TRUE THEN
                RAISE EXCEPTION 'formal backtest aggregate runtime is not authorized';
            END IF;
            IF OLD.status = 'queued' AND NEW.status NOT IN ('running', 'failed') THEN
                RAISE EXCEPTION 'formal backtest recovery aggregate claim is invalid';
            END IF;
            IF OLD.status = 'running'
               AND NEW.status NOT IN ('running', 'succeeded', 'failed', 'cancelled') THEN
                RAISE EXCEPTION 'formal backtest recovery aggregate transition is invalid';
            END IF;
            IF OLD.status = 'running'
               AND NEW.status = 'succeeded'
               AND NOT ((
                   NEW.metrics_json IS NOT NULL
                   AND NEW.error IS NULL
                   AND NEW.finished_at IS NOT NULL
               ) IS TRUE) THEN
                RAISE EXCEPTION 'formal backtest recovery success is incomplete';
            END IF;
            IF OLD.status = 'running'
               AND NEW.status IN ('failed', 'cancelled')
               AND NOT ((
                   NEW.metrics_json IS NULL
                   AND NEW.error IS NOT NULL
                   AND NEW.finished_at IS NOT NULL
               ) IS TRUE) THEN
                RAISE EXCEPTION 'formal backtest recovery failure is incomplete';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_guard_formal_backtest_recovery_aggregate
        BEFORE UPDATE ON quantlab.backtest_runs
        FOR EACH ROW EXECUTE FUNCTION
            quantlab.guard_formal_backtest_recovery_aggregate();
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "SELECT pg_advisory_xact_lock("
            "hashtext('formal-backtest-interruption:0086-downgrade'))"
        )
    )
    bind.execute(
        sa.text(
            "LOCK TABLE quantlab.formal_backtest_interruption_recoveries "
            "IN ACCESS EXCLUSIVE MODE"
        )
    )
    bind.execute(
        sa.text(
            "LOCK TABLE quantlab.jobs, quantlab.backtest_runs "
            "IN SHARE ROW EXCLUSIVE MODE"
        )
    )
    recovery_count = int(
        bind.scalar(
            sa.text(
                "SELECT count(*) FROM quantlab.formal_backtest_interruption_recoveries"
            )
        )
        or 0
    )
    repeated_job_count = int(
        bind.scalar(
            sa.text(
                "SELECT count(*) FROM quantlab.jobs "
                "WHERE id = :job_id AND (attempts > 1 OR max_attempts > 1)"
            ),
            {"job_id": JOB_ID},
        )
        or 0
    )
    if recovery_count or repeated_job_count:
        raise RuntimeError(
            "cannot downgrade after immutable formal-backtest interruption recovery evidence"
        )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_guard_formal_backtest_recovery_aggregate "
        "ON quantlab.backtest_runs"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS quantlab.guard_formal_backtest_recovery_aggregate()"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_guard_formal_backtest_job_attempts ON quantlab.jobs"
    )
    op.execute("DROP FUNCTION IF EXISTS quantlab.guard_formal_backtest_job_attempts()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_formal_backtest_interruption_recovery "
        "ON quantlab.formal_backtest_interruption_recoveries"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS quantlab.guard_formal_backtest_interruption_recovery()"
    )
    op.execute(
        "DROP TRIGGER IF EXISTS trg_validate_formal_backtest_interruption_recovery "
        "ON quantlab.formal_backtest_interruption_recoveries"
    )
    op.execute(
        "DROP FUNCTION IF EXISTS "
        "quantlab.validate_formal_backtest_interruption_recovery()"
    )
    op.drop_index(
        "idx_formal_backtest_interruption_recoveries_created",
        table_name="formal_backtest_interruption_recoveries",
        schema=SCHEMA,
    )
    op.drop_table("formal_backtest_interruption_recoveries", schema=SCHEMA)
