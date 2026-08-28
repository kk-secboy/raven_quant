"""Add the capital-only final-OOS alpha-spending ledger.

Revision ID: 0070_alpha_spending_ledger
Revises: 0069_model_artifact_checkpoints
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0070_alpha_spending_ledger"
down_revision: str | None = "0069_model_artifact_checkpoints"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
JSON = sa.JSON().with_variant(JSONB(), "postgresql")
ALPHA = sa.Numeric(38, 28)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def upgrade() -> None:
    op.create_table(
        "capital_oos_alpha_families",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("capital_oos_family_sha256", sa.String(), nullable=False),
        sa.Column("mandate_json", JSON, nullable=False),
        sa.Column("total_alpha", ALPHA, nullable=False),
        sa.Column("policy_json", JSON, nullable=False),
        sa.Column("policy_sha256", sa.String(), nullable=False),
        sa.Column("next_ordinal", sa.Integer(), nullable=False),
        sa.Column("reserved_alpha", ALPHA, nullable=False),
        sa.Column("settled_alpha", ALPHA, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "capital_oos_family_sha256", name="uq_capital_oos_alpha_family"
        ),
        sa.CheckConstraint("total_alpha = 0.05", name="ck_capital_oos_alpha_total"),
        sa.CheckConstraint(
            "capital_oos_family_sha256 ~ '^[0-9a-f]{64}$' "
            "AND policy_sha256 ~ '^[0-9a-f]{64}$' "
            "AND jsonb_typeof(mandate_json) = 'object' "
            "AND jsonb_typeof(policy_json) = 'object'",
            name="ck_capital_oos_alpha_family_identity",
        ),
        sa.CheckConstraint(
            "next_ordinal > 0 AND reserved_alpha >= 0 AND settled_alpha >= 0 "
            "AND settled_alpha <= reserved_alpha AND reserved_alpha <= total_alpha",
            name="ck_capital_oos_alpha_family_counters",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_capital_oos_alpha_families_created",
        "capital_oos_alpha_families",
        ["created_at"],
        schema=SCHEMA,
    )

    op.create_table(
        "capital_oos_legacy_attempts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "oos_vintage_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.oos_vintages.id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("dataset_identity", sa.String(), nullable=False),
        sa.Column("dataset_lineage_id", sa.String()),
        sa.Column("final_oos_start", sa.Date(), nullable=False),
        sa.Column("final_oos_end", sa.Date(), nullable=False),
        sa.Column("first_opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True)),
        sa.Column("sealed_candidate_set_sha256", sa.String(), nullable=False),
        sa.Column("raw_p_value", sa.Float(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("failure_recorded", sa.Boolean(), nullable=False),
        sa.Column("legacy_evidence_json", JSON, nullable=False),
        sa.Column("legacy_evidence_sha256", sa.String(), nullable=False),
        sa.Column(
            "reconciled_family_id",
            sa.String(),
            sa.ForeignKey(
                f"{SCHEMA}.capital_oos_alpha_families.id", ondelete="RESTRICT"
            ),
        ),
        sa.Column("ordinal", sa.Integer()),
        sa.Column("spent_alpha", ALPHA),
        sa.Column("reconciliation_json", JSON),
        sa.Column("reconciliation_sha256", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reconciled_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint(
            "reconciled_family_id",
            "ordinal",
            name="uq_capital_oos_legacy_family_ordinal",
        ),
        sa.CheckConstraint(
            "status IN ('unreconciled', 'reconciled')",
            name="ck_capital_oos_legacy_status",
        ),
        sa.CheckConstraint(
            "length(btrim(scope)) > 0 AND length(btrim(dataset_identity)) > 0 "
            "AND final_oos_end >= final_oos_start "
            "AND sealed_candidate_set_sha256 ~ '^[0-9a-f]{64}$' "
            "AND legacy_evidence_sha256 ~ '^[0-9a-f]{64}$' "
            "AND jsonb_typeof(legacy_evidence_json) = 'object' "
            "AND raw_p_value = 1 AND passed IS FALSE AND failure_recorded IS TRUE",
            name="ck_capital_oos_legacy_evidence",
        ),
        sa.CheckConstraint(
            "(status = 'unreconciled' AND reconciled_family_id IS NULL "
            "AND ordinal IS NULL AND spent_alpha IS NULL "
            "AND reconciliation_json IS NULL AND reconciliation_sha256 IS NULL "
            "AND reconciled_at IS NULL) OR "
            "(status = 'reconciled' AND reconciled_family_id IS NOT NULL "
            "AND ordinal > 0 AND spent_alpha > 0 "
            "AND jsonb_typeof(reconciliation_json) = 'object' "
            "AND reconciliation_sha256 ~ '^[0-9a-f]{64}$' "
            "AND reconciled_at IS NOT NULL)",
            name="ck_capital_oos_legacy_reconciliation",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_capital_oos_legacy_status",
        "capital_oos_legacy_attempts",
        ["status", "first_opened_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "idx_capital_oos_legacy_family_window",
        "capital_oos_legacy_attempts",
        ["reconciled_family_id", "final_oos_start", "final_oos_end"],
        schema=SCHEMA,
    )

    legacy_source_rows = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT id, scope, dataset_identity, dataset_lineage_id, test_start, "
                "test_end, first_opened_at, consumed_at, sealed_candidate_set_json, "
                "sealed_candidate_set_sha256 "
                "FROM quantlab.oos_vintages ORDER BY first_opened_at, test_start, id"
            )
        )
        .mappings()
        .all()
    )
    legacy_target = sa.table(
        "capital_oos_legacy_attempts",
        sa.column("id"),
        sa.column("oos_vintage_id"),
        sa.column("status"),
        sa.column("scope"),
        sa.column("dataset_identity"),
        sa.column("dataset_lineage_id"),
        sa.column("final_oos_start"),
        sa.column("final_oos_end"),
        sa.column("first_opened_at"),
        sa.column("consumed_at"),
        sa.column("sealed_candidate_set_sha256"),
        sa.column("raw_p_value"),
        sa.column("passed"),
        sa.column("failure_recorded"),
        sa.column("legacy_evidence_json"),
        sa.column("legacy_evidence_sha256"),
        sa.column("created_at"),
        schema=SCHEMA,
    )
    for row in legacy_source_rows:
        if _canonical_sha256(dict(row["sealed_candidate_set_json"] or {})) != str(
            row["sealed_candidate_set_sha256"]
        ):
            raise RuntimeError("legacy OOS sealed candidate set hash is invalid")
        evidence = {
            "contract_version": "capital-oos-legacy-attempt-v1",
            "oos_vintage_id": str(row["id"]),
            "scope": str(row["scope"]),
            "dataset_identity": str(row["dataset_identity"]),
            "dataset_lineage_id": row["dataset_lineage_id"],
            "final_oos_start": _iso(row["test_start"]),
            "final_oos_end": _iso(row["test_end"]),
            "first_opened_at": _iso(row["first_opened_at"]),
            "consumed_at": _iso(row["consumed_at"]),
            "sealed_candidate_set_sha256": str(row["sealed_candidate_set_sha256"]),
            "raw_p_value": 1.0,
            "passed": False,
            "failure_recorded": True,
            "attribution_status": "unreconciled",
        }
        op.get_bind().execute(
            legacy_target.insert().values(
                id=_canonical_sha256(
                    {
                        "kind": "capital-oos-legacy-attempt-v1",
                        "oos_vintage_id": str(row["id"]),
                    }
                ),
                oos_vintage_id=str(row["id"]),
                status="unreconciled",
                scope=str(row["scope"]),
                dataset_identity=str(row["dataset_identity"]),
                dataset_lineage_id=row["dataset_lineage_id"],
                final_oos_start=row["test_start"],
                final_oos_end=row["test_end"],
                first_opened_at=row["first_opened_at"],
                consumed_at=row["consumed_at"],
                sealed_candidate_set_sha256=str(row["sealed_candidate_set_sha256"]),
                raw_p_value=1.0,
                passed=False,
                failure_recorded=True,
                legacy_evidence_json=evidence,
                legacy_evidence_sha256=_canonical_sha256(evidence),
                created_at=sa.func.now(),
            )
        )

    op.create_table(
        "capital_oos_alpha_batches",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "family_id",
            sa.String(),
            sa.ForeignKey(f"{SCHEMA}.capital_oos_alpha_families.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("batch_key", sa.String(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("frozen_bundle_manifest_sha256", sa.String(), nullable=False),
        sa.Column("frozen_baseline_manifest_sha256", sa.String(), nullable=False),
        sa.Column("dataset_lineage_id", sa.String(), nullable=False),
        sa.Column("dataset_identity_sha256", sa.String(), nullable=False),
        sa.Column("hypothesis_count", sa.Integer(), nullable=False),
        sa.Column("research_data_end", sa.Date(), nullable=False),
        sa.Column("final_oos_start", sa.Date(), nullable=False),
        sa.Column("final_oos_end", sa.Date(), nullable=False),
        sa.Column("trading_day_count", sa.Integer(), nullable=False),
        sa.Column("trading_dates_json", JSON, nullable=False),
        sa.Column("trading_dates_sha256", sa.String(), nullable=False),
        sa.Column("embargo_trading_day_count", sa.Integer(), nullable=False),
        sa.Column("embargo_trading_dates_json", JSON, nullable=False),
        sa.Column("embargo_trading_dates_sha256", sa.String(), nullable=False),
        sa.Column("preregistration_json", JSON, nullable=False),
        sa.Column("preregistration_sha256", sa.String(), nullable=False),
        sa.Column("batch_alpha", ALPHA, nullable=False),
        sa.Column("raw_p_value", sa.Float()),
        sa.Column("passed", sa.Boolean()),
        sa.Column("failure_recorded", sa.Boolean()),
        sa.Column("settlement_evidence_json", JSON),
        sa.Column("settlement_evidence_sha256", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("family_id", "batch_key", name="uq_capital_oos_alpha_batch_key"),
        sa.UniqueConstraint("family_id", "ordinal", name="uq_capital_oos_alpha_batch_ordinal"),
        sa.CheckConstraint(
            "ordinal > 0 AND hypothesis_count = 1 AND batch_alpha > 0",
            name="ck_capital_oos_alpha_batch_values",
        ),
        sa.CheckConstraint(
            "status IN ('reserved', 'settled')",
            name="ck_capital_oos_alpha_batch_status",
        ),
        sa.CheckConstraint(
            "research_data_end < final_oos_start "
            "AND final_oos_end >= final_oos_start AND trading_day_count >= 252 "
            "AND embargo_trading_day_count >= 20 "
            "AND (embargo_trading_dates_json ->> 0)::date > research_data_end "
            "AND (embargo_trading_dates_json ->> "
            "(embargo_trading_day_count - 1))::date < final_oos_start",
            name="ck_capital_oos_alpha_batch_window",
        ),
        sa.CheckConstraint(
            "frozen_bundle_manifest_sha256 ~ '^[0-9a-f]{64}$' "
            "AND frozen_baseline_manifest_sha256 ~ '^[0-9a-f]{64}$' "
            "AND dataset_lineage_id ~ '^[0-9a-f]{64}$' "
            "AND dataset_identity_sha256 ~ '^[0-9a-f]{64}$' "
            "AND trading_dates_sha256 ~ '^[0-9a-f]{64}$' "
            "AND embargo_trading_dates_sha256 ~ '^[0-9a-f]{64}$' "
            "AND preregistration_sha256 ~ '^[0-9a-f]{64}$' "
            "AND jsonb_typeof(trading_dates_json) = 'array' "
            "AND jsonb_array_length(trading_dates_json) = trading_day_count "
            "AND jsonb_typeof(embargo_trading_dates_json) = 'array' "
            "AND jsonb_array_length(embargo_trading_dates_json) = embargo_trading_day_count "
            "AND jsonb_typeof(preregistration_json) = 'object' "
            "AND (settlement_evidence_sha256 IS NULL OR "
            "settlement_evidence_sha256 ~ '^[0-9a-f]{64}$')",
            name="ck_capital_oos_alpha_batch_evidence",
        ),
        sa.CheckConstraint(
            "(status = 'reserved' AND raw_p_value IS NULL AND passed IS NULL "
            "AND failure_recorded IS NULL AND settlement_evidence_json IS NULL "
            "AND settlement_evidence_sha256 IS NULL AND settled_at IS NULL) OR "
            "(status = 'settled' AND raw_p_value IS NOT NULL AND passed IS NOT NULL "
            "AND failure_recorded IS NOT NULL AND settlement_evidence_json IS NOT NULL "
            "AND settlement_evidence_sha256 IS NOT NULL AND settled_at IS NOT NULL)",
            name="ck_capital_oos_alpha_batch_settlement",
        ),
        sa.CheckConstraint(
            "status = 'reserved' OR (raw_p_value > 0 AND raw_p_value <= 1 "
            "AND (failure_recorded IS FALSE OR "
            "(failure_recorded IS TRUE AND raw_p_value = 1 AND passed IS FALSE)))",
            name="ck_capital_oos_alpha_batch_result",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "idx_capital_oos_alpha_batches_family_window",
        "capital_oos_alpha_batches",
        ["family_id", "final_oos_start", "final_oos_end"],
        schema=SCHEMA,
    )
    op.create_index(
        "uq_capital_oos_alpha_one_reserved_family",
        "capital_oos_alpha_batches",
        ["family_id"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("status = 'reserved'"),
    )
    op.add_column(
        "oos_vintages",
        sa.Column("capital_oos_alpha_batch_id", sa.String()),
        schema=SCHEMA,
    )
    op.create_foreign_key(
        "fk_oos_vintages_capital_oos_alpha_batch",
        "oos_vintages",
        "capital_oos_alpha_batches",
        ["capital_oos_alpha_batch_id"],
        ["id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete="RESTRICT",
    )
    op.create_unique_constraint(
        "uq_oos_vintages_capital_oos_alpha_batch",
        "oos_vintages",
        ["capital_oos_alpha_batch_id"],
        schema=SCHEMA,
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.guard_capital_oos_vintage_link()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' AND OLD.capital_oos_alpha_batch_id IS NOT NULL THEN
                RAISE EXCEPTION 'capital OOS linked vintage is immutable';
            END IF;
            IF TG_OP = 'UPDATE' THEN
                IF OLD.capital_oos_alpha_batch_id IS NOT NULL THEN
                    RAISE EXCEPTION 'capital OOS linked vintage is immutable';
                END IF;
                IF NEW.capital_oos_alpha_batch_id IS NOT NULL THEN
                    RAISE EXCEPTION 'capital OOS vintage link must be inserted atomically';
                END IF;
            END IF;
            IF TG_OP = 'DELETE' THEN
                RETURN OLD;
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_capital_oos_vintage_link
        BEFORE UPDATE OR DELETE ON quantlab.oos_vintages
        FOR EACH ROW EXECUTE FUNCTION quantlab.guard_capital_oos_vintage_link();
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.guard_capital_oos_alpha_family()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'capital OOS alpha families are durable';
            END IF;
            IF NEW.id IS DISTINCT FROM OLD.id
               OR NEW.capital_oos_family_sha256 IS DISTINCT FROM OLD.capital_oos_family_sha256
               OR NEW.mandate_json IS DISTINCT FROM OLD.mandate_json
               OR NEW.total_alpha IS DISTINCT FROM OLD.total_alpha
               OR NEW.policy_json IS DISTINCT FROM OLD.policy_json
               OR NEW.policy_sha256 IS DISTINCT FROM OLD.policy_sha256
               OR NEW.created_at IS DISTINCT FROM OLD.created_at
               OR NEW.next_ordinal < OLD.next_ordinal
               OR NEW.reserved_alpha < OLD.reserved_alpha
               OR NEW.settled_alpha < OLD.settled_alpha THEN
                RAISE EXCEPTION 'capital OOS alpha family identity and counters are immutable';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_capital_oos_alpha_family
        BEFORE UPDATE OR DELETE ON quantlab.capital_oos_alpha_families
        FOR EACH ROW EXECUTE FUNCTION quantlab.guard_capital_oos_alpha_family();
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.guard_capital_oos_legacy_attempt()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'legacy capital OOS attempts are durable';
            END IF;
            IF OLD.status = 'reconciled' THEN
                RAISE EXCEPTION 'reconciled legacy capital OOS attempt is immutable';
            END IF;
            IF NEW.status <> 'reconciled'
               OR NEW.id IS DISTINCT FROM OLD.id
               OR NEW.oos_vintage_id IS DISTINCT FROM OLD.oos_vintage_id
               OR NEW.scope IS DISTINCT FROM OLD.scope
               OR NEW.dataset_identity IS DISTINCT FROM OLD.dataset_identity
               OR NEW.dataset_lineage_id IS DISTINCT FROM OLD.dataset_lineage_id
               OR NEW.final_oos_start IS DISTINCT FROM OLD.final_oos_start
               OR NEW.final_oos_end IS DISTINCT FROM OLD.final_oos_end
               OR NEW.first_opened_at IS DISTINCT FROM OLD.first_opened_at
               OR NEW.consumed_at IS DISTINCT FROM OLD.consumed_at
               OR NEW.sealed_candidate_set_sha256
                  IS DISTINCT FROM OLD.sealed_candidate_set_sha256
               OR NEW.raw_p_value IS DISTINCT FROM OLD.raw_p_value
               OR NEW.passed IS DISTINCT FROM OLD.passed
               OR NEW.failure_recorded IS DISTINCT FROM OLD.failure_recorded
               OR NEW.legacy_evidence_json IS DISTINCT FROM OLD.legacy_evidence_json
               OR NEW.legacy_evidence_sha256 IS DISTINCT FROM OLD.legacy_evidence_sha256
               OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
                RAISE EXCEPTION 'legacy capital OOS evidence is immutable';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_capital_oos_legacy_attempt
        BEFORE UPDATE OR DELETE ON quantlab.capital_oos_legacy_attempts
        FOR EACH ROW EXECUTE FUNCTION quantlab.guard_capital_oos_legacy_attempt();
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION quantlab.guard_capital_oos_alpha_batch()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'capital OOS alpha batches are append-only';
            END IF;
            IF OLD.status = 'settled' THEN
                RAISE EXCEPTION 'settled capital OOS alpha batch is immutable';
            END IF;
            IF NEW.status <> 'settled'
               OR NEW.id IS DISTINCT FROM OLD.id
               OR NEW.family_id IS DISTINCT FROM OLD.family_id
               OR NEW.batch_key IS DISTINCT FROM OLD.batch_key
               OR NEW.ordinal IS DISTINCT FROM OLD.ordinal
               OR NEW.frozen_bundle_manifest_sha256
                  IS DISTINCT FROM OLD.frozen_bundle_manifest_sha256
               OR NEW.frozen_baseline_manifest_sha256
                  IS DISTINCT FROM OLD.frozen_baseline_manifest_sha256
               OR NEW.dataset_lineage_id IS DISTINCT FROM OLD.dataset_lineage_id
               OR NEW.dataset_identity_sha256 IS DISTINCT FROM OLD.dataset_identity_sha256
               OR NEW.hypothesis_count IS DISTINCT FROM OLD.hypothesis_count
               OR NEW.research_data_end IS DISTINCT FROM OLD.research_data_end
               OR NEW.final_oos_start IS DISTINCT FROM OLD.final_oos_start
               OR NEW.final_oos_end IS DISTINCT FROM OLD.final_oos_end
               OR NEW.trading_day_count IS DISTINCT FROM OLD.trading_day_count
               OR NEW.trading_dates_json IS DISTINCT FROM OLD.trading_dates_json
               OR NEW.trading_dates_sha256 IS DISTINCT FROM OLD.trading_dates_sha256
               OR NEW.embargo_trading_day_count IS DISTINCT FROM OLD.embargo_trading_day_count
               OR NEW.embargo_trading_dates_json IS DISTINCT FROM OLD.embargo_trading_dates_json
               OR NEW.embargo_trading_dates_sha256 IS DISTINCT FROM OLD.embargo_trading_dates_sha256
               OR NEW.preregistration_json IS DISTINCT FROM OLD.preregistration_json
               OR NEW.preregistration_sha256 IS DISTINCT FROM OLD.preregistration_sha256
               OR NEW.batch_alpha IS DISTINCT FROM OLD.batch_alpha
               OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
                RAISE EXCEPTION 'capital OOS alpha preregistration is immutable';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_capital_oos_alpha_batch
        BEFORE UPDATE OR DELETE ON quantlab.capital_oos_alpha_batches
        FOR EACH ROW EXECUTE FUNCTION quantlab.guard_capital_oos_alpha_batch();
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_capital_oos_vintage_link ON quantlab.oos_vintages"
    )
    op.execute("DROP FUNCTION IF EXISTS quantlab.guard_capital_oos_vintage_link()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_capital_oos_alpha_batch ON quantlab.capital_oos_alpha_batches"
    )
    op.execute("DROP FUNCTION IF EXISTS quantlab.guard_capital_oos_alpha_batch()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_capital_oos_legacy_attempt "
        "ON quantlab.capital_oos_legacy_attempts"
    )
    op.execute("DROP FUNCTION IF EXISTS quantlab.guard_capital_oos_legacy_attempt()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_capital_oos_alpha_family ON quantlab.capital_oos_alpha_families"
    )
    op.execute("DROP FUNCTION IF EXISTS quantlab.guard_capital_oos_alpha_family()")
    op.drop_constraint(
        "uq_oos_vintages_capital_oos_alpha_batch",
        "oos_vintages",
        schema=SCHEMA,
        type_="unique",
    )
    op.drop_constraint(
        "fk_oos_vintages_capital_oos_alpha_batch",
        "oos_vintages",
        schema=SCHEMA,
        type_="foreignkey",
    )
    op.drop_column("oos_vintages", "capital_oos_alpha_batch_id", schema=SCHEMA)
    op.drop_index(
        "uq_capital_oos_alpha_one_reserved_family",
        table_name="capital_oos_alpha_batches",
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_capital_oos_alpha_batches_family_window",
        table_name="capital_oos_alpha_batches",
        schema=SCHEMA,
    )
    op.drop_table("capital_oos_alpha_batches", schema=SCHEMA)
    op.drop_index(
        "idx_capital_oos_legacy_family_window",
        table_name="capital_oos_legacy_attempts",
        schema=SCHEMA,
    )
    op.drop_index(
        "idx_capital_oos_legacy_status",
        table_name="capital_oos_legacy_attempts",
        schema=SCHEMA,
    )
    op.drop_table("capital_oos_legacy_attempts", schema=SCHEMA)
    op.drop_index(
        "idx_capital_oos_alpha_families_created",
        table_name="capital_oos_alpha_families",
        schema=SCHEMA,
    )
    op.drop_table("capital_oos_alpha_families", schema=SCHEMA)
