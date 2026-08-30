"""Make automatic research cycles horizon-specific.

Revision ID: 0079_autopilot_horizon_cycles
Revises: 0078_baseline_v12_seal
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0079_autopilot_horizon_cycles"
down_revision: str | None = "0078_baseline_v12_seal"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
PRIMARY_LABEL_POLICY_SHA256 = (
    "f90f34e67b4721c0e7b82181007872cc099093e80ea92f8ed3d2d88e3e7adfdc"
)


def upgrade() -> None:
    op.add_column(
        "autopilot_cycles",
        sa.Column("horizon_profile", sa.String(), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "autopilot_cycles",
        sa.Column("primary_label_policy_sha256", sa.String(), nullable=True),
        schema=SCHEMA,
    )
    # Historical cycles did not freeze the new 5/63/252 comparison policy.
    # Preserve them as non-executable legacy evidence; never relabel their old
    # one-session outputs as a five-session short-horizon result.
    op.execute(
        sa.text(
            f"UPDATE {SCHEMA}.autopilot_cycles "
            "SET horizon_profile = 'legacy_ambiguous', "
            "primary_label_policy_sha256 = :policy_sha, "
            "state_json = state_json || CAST(:policy_state AS jsonb)"
        ).bindparams(
            policy_sha=PRIMARY_LABEL_POLICY_SHA256,
            policy_state=json.dumps(
                {
                    "horizon_profile": "legacy_ambiguous",
                    "label_horizon_sessions": None,
                    "historical_results_only": True,
                    "capital_eligible": False,
                    "final_oos_must_not_open": True,
                    "migrated_from_unbound_autopilot_cycle": True,
                    "primary_label_policy": {
                        "contract_version": "primary-label-policy-v1",
                        "horizon_primary_labels_sessions": {
                            "long_1_3y": 252,
                            "short_1_5d": 5,
                            "swing_1_6m": 63,
                        },
                        "legacy_ambiguous_executable": False,
                        "policy_sha256": PRIMARY_LABEL_POLICY_SHA256,
                    },
                },
                separators=(",", ":"),
            ),
        )
    )
    op.alter_column(
        "autopilot_cycles", "horizon_profile", nullable=False, schema=SCHEMA
    )
    op.alter_column(
        "autopilot_cycles",
        "primary_label_policy_sha256",
        nullable=False,
        schema=SCHEMA,
    )
    op.drop_constraint(
        "autopilot_cycles_dataset_identity_sha256_key",
        "autopilot_cycles",
        schema=SCHEMA,
        type_="unique",
    )
    op.create_check_constraint(
        "ck_autopilot_cycles_horizon",
        "autopilot_cycles",
        "horizon_profile IN "
        "('short_1_5d', 'swing_1_6m', 'long_1_3y', 'legacy_ambiguous')",
        schema=SCHEMA,
    )
    op.create_unique_constraint(
        "uq_autopilot_cycle_dataset_horizon",
        "autopilot_cycles",
        ["dataset_identity_sha256", "horizon_profile"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    # Older code can represent only the pre-0079 unbound rows.  Never erase or
    # reinterpret any newly created horizon-specific cycle to make it start.
    bind = op.get_bind()
    non_short_cycles = bind.scalar(
        sa.text(
            f"SELECT count(*) FROM {SCHEMA}.autopilot_cycles "
            "WHERE horizon_profile <> 'legacy_ambiguous'"
        )
    )
    if int(non_short_cycles or 0):
        raise RuntimeError(
            "cannot downgrade horizon-specific Autopilot cycles without "
            "deleting retained short/swing/long research history"
        )
    op.drop_constraint(
        "uq_autopilot_cycle_dataset_horizon",
        "autopilot_cycles",
        schema=SCHEMA,
        type_="unique",
    )
    op.drop_constraint(
        "ck_autopilot_cycles_horizon",
        "autopilot_cycles",
        schema=SCHEMA,
        type_="check",
    )
    op.create_unique_constraint(
        "autopilot_cycles_dataset_identity_sha256_key",
        "autopilot_cycles",
        ["dataset_identity_sha256"],
        schema=SCHEMA,
    )
    op.drop_column("autopilot_cycles", "primary_label_policy_sha256", schema=SCHEMA)
    op.drop_column("autopilot_cycles", "horizon_profile", schema=SCHEMA)
