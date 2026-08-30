"""Seal v17 and allow one exact v16-to-v17 pre-result repair.

Revision ID: 0085_baseline_v17_industry
Revises: 0084_baseline_v16_pos_cap

The v17 generation is append-only. It preserves every v2-v7 repair and
evidence rule byte-for-byte, adds one production-bound v8 single-member
exception, and seals the new runtime identity. No v16 row is updated.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import sqlalchemy as sa
from alembic import op

revision: str = "0085_baseline_v17_industry"
down_revision: str | None = "0084_baseline_v16_pos_cap"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
_RUNTIME_CONSTRAINT = "ck_strategy_versions_v17_runtime_identity"
_REPAIR_CONSTRAINT = "ck_transparent_baseline_repair_distinct_target"
_EVIDENCE_CONSTRAINT = "ck_transparent_baseline_repair_evidence"

_V17_RECIPE = "qlib-rdagent-single-mainline-2026-08-31-v17"
_V17_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
_V17_RUNTIME_BUNDLE_SHA256 = (
    "c3d6aeb00a8f5286ee117f885231b43ad49e461c5183f5cbf32c4601824bb9fc"
)
_TRANSPARENT_RECIPE_IDS = (
    "short_relative_strength",
    "swing_trend",
    "long_quality_value",
)

_V8_CONTRACT = "transparent-baseline-pre-result-repair-v8"
_V8_GENERATION = "v16-to-v17-topk-industry-capacity-partial-cash"
_V8_SOURCE_COMMIT = "92c722b89450d9b25549ac777ca171fd163c57ca"
_V8_SOURCE_BATCH_SHA256 = (
    "9721d472a343295294d6199de9c17bfcd932f1494b88e9dfa18928a3c682f5c9"
)
_V8_SOURCE_DATASET_IDENTITY_SHA256 = (
    "eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2"
)
_V8_SOURCE_DATASET_LINEAGE_ID = (
    "1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e"
)
_V8_SOURCE_BACKTEST_ID = "b1a0f92acf3145ba9a6d887d459d1129"
_V8_SOURCE_JOB_ID = "5587b686787f47dca72abad634d18ec3"
_V8_SOURCE_STRATEGY_VERSION_ID = "934a776a9d6c4c538411473165d3c93e"
_V8_SOURCE_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
_V8_SOURCE_RUNTIME_BUNDLE_SHA256 = (
    "ac1c2020996efa4e3c3e9609dd4c736d8c65893dccd7ba9c90e3464402d2a329"
)
_V8_TARGET_RECIPE = _V17_RECIPE
_V8_TARGET_RUNNER_SHA256 = _V17_RUNNER_SHA256
_V8_TARGET_RUNTIME_BUNDLE_SHA256 = _V17_RUNTIME_BUNDLE_SHA256
_V8_RUNTIME_CONTRACT = (
    "transparent-baseline-topk-industry-capacity-partial-cash-repair-v1"
)
_V8_SOURCE_ARTIFACT_INVENTORIES_SHA256 = (
    "fb4a4a2e457ae0e674ba638af7c0e11c604a9847375d1470bdf4e3d255630435"
)
_V8_SOURCE_SELECTION_SHA256 = (
    "b86ed4a518a6d58b049ad3aeb31e058daaff75de67619b316f39f8fb2f2b1d67"
)
_V8_SOURCE_UNAVAILABLE_HORIZONS_SHA256 = (
    "b9f36421f0047ab024eec0f2b20909a105c2a57e17dc94a50aa25e632f8c3644"
)
_V8_UNAVAILABLE_EVIDENCE_SHA256S = (
    "62f5642a4ad024cc24d23ec34b9ce760a552f560c783612b63dab459ba00d14d",
    "b57d3dba4ab3d2284b2a001cb30a7b2862a8659c3738d5fc23419e7e7a6b5729",
)
_V8_TARGET_CHANGE_CODES = ("topk_industry_capacity_partial_cash",)
_V8_RECEIPT_SHA256 = (
    "980ea643755d261cc7ee39ffec5e23af3f8e27ebdc1e2f2a647e0802b9dc636d"
)


@lru_cache(maxsize=1)
def _previous_module() -> ModuleType:
    path = Path(__file__).with_name(
        "0084_transparent_baseline_v16_position_cap_repair.py"
    )
    spec = importlib.util.spec_from_file_location("baseline_v16_position_cap_0084", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("0084 baseline migration cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_ids(values: Sequence[str]) -> str:
    return "[" + ",".join(f'"{value}"' for value in values) + "]"


def _v8_generation() -> str:
    source_ids = _source_ids((_V8_SOURCE_BACKTEST_ID,))
    source_bindings = (
        "[{"
        f'"backtest_id":"{_V8_SOURCE_BACKTEST_ID}",'
        f'"job_id":"{_V8_SOURCE_JOB_ID}",'
        f'"strategy_version_id":"{_V8_SOURCE_STRATEGY_VERSION_ID}"'
        "}]"
    )
    return (
        "("
        f"verification_json ->> 'receipt_contract_version' = '{_V8_CONTRACT}' AND "
        f"verification_json ->> 'repair_generation' = '{_V8_GENERATION}' AND "
        "verification_json ->> 'target_runner_sha256' = "
        f"'{_V8_TARGET_RUNNER_SHA256}' AND "
        f"target_recipe_version = '{_V8_TARGET_RECIPE}' AND "
        f"source_backtest_ids_json = '{source_ids}'::jsonb"
        f" AND receipt_sha256 = '{_V8_RECEIPT_SHA256}'"
        " AND verification_json ->> 'receipt_sha256' = "
        f"'{_V8_RECEIPT_SHA256}'"
        f" AND source_batch_sha256 = '{_V8_SOURCE_BATCH_SHA256}'"
        " AND verification_json ->> 'source_release_commit' = "
        f"'{_V8_SOURCE_COMMIT}'"
        " AND verification_json ->> 'source_batch_sha256' = "
        f"'{_V8_SOURCE_BATCH_SHA256}'"
        " AND verification_json ->> 'source_dataset_identity_sha256' = "
        f"'{_V8_SOURCE_DATASET_IDENTITY_SHA256}'"
        " AND verification_json ->> 'source_dataset_lineage_id' = "
        f"'{_V8_SOURCE_DATASET_LINEAGE_ID}'"
        f" AND source_dataset_lineage_id = '{_V8_SOURCE_DATASET_LINEAGE_ID}'"
        f" AND target_dataset_lineage_id = '{_V8_SOURCE_DATASET_LINEAGE_ID}'"
        " AND verification_json ->> 'source_runner_sha256' = "
        f"'{_V8_SOURCE_RUNNER_SHA256}'"
        " AND verification_json ->> 'source_runtime_bundle_sha256' = "
        f"'{_V8_SOURCE_RUNTIME_BUNDLE_SHA256}'"
        " AND verification_json ->> 'target_runtime_bundle_sha256' = "
        f"'{_V8_TARGET_RUNTIME_BUNDLE_SHA256}'"
        " AND verification_json ->> 'runtime_contract_version' = "
        f"'{_V8_RUNTIME_CONTRACT}'"
        " AND verification_json ->> 'source_artifact_inventories_sha256' = "
        f"'{_V8_SOURCE_ARTIFACT_INVENTORIES_SHA256}'"
        " AND verification_json ->> "
        "'source_unopened_history_selection_sha256' = "
        f"'{_V8_SOURCE_SELECTION_SHA256}'"
        " AND verification_json ->> 'source_unavailable_horizons_sha256' = "
        f"'{_V8_SOURCE_UNAVAILABLE_HORIZONS_SHA256}'"
        " AND verification_json -> 'source_unavailable_evidence_sha256s' = "
        f"'{_source_ids(_V8_UNAVAILABLE_EVIDENCE_SHA256S)}'::jsonb"
        " AND verification_json -> 'target_change_codes' = "
        f"'{_source_ids(_V8_TARGET_CHANGE_CODES)}'::jsonb"
        " AND verification_json -> 'source_backtest_ids' = "
        f"'{source_ids}'::jsonb"
        " AND verification_json -> 'source_bindings' = "
        f"'{source_bindings}'::jsonb"
        ") IS TRUE"
    )


def _previous_repair_constraint() -> str:
    return str(_previous_module()._repair_constraint())


def _repair_constraint() -> str:
    previous = _previous_repair_constraint()
    if not previous.endswith(")"):
        raise RuntimeError("0084 repair constraint is malformed")
    extended = previous[:-1] + " OR " + _v8_generation() + ")"
    # PostgreSQL accepts UNKNOWN for CHECK constraints.  Close the complete
    # inherited V2-V8 disjunction so missing JSON keys cannot turn an invalid
    # legacy or current branch into an accepted NULL result.
    return "(" + extended + ") IS TRUE"


def _v8_single_member_evidence() -> str:
    source_ids = f"'[\"{_V8_SOURCE_BACKTEST_ID}\"]'::jsonb"
    return (
        "(jsonb_array_length(source_backtest_ids_json) = 1 AND "
        "jsonb_array_length(target_strategy_version_ids_json) = 1 AND "
        f"receipt_sha256 = '{_V8_RECEIPT_SHA256}' AND "
        f"target_recipe_version = '{_V8_TARGET_RECIPE}' AND "
        f"source_backtest_ids_json = {source_ids} AND "
        "verification_json -> 'source_backtest_ids' = source_backtest_ids_json AND "
        "verification_json -> 'target_strategy_version_ids' = "
        "target_strategy_version_ids_json AND "
        "verification_json ->> 'receipt_sha256' = receipt_sha256 AND "
        "verification_json ->> 'receipt_contract_version' = "
        f"'{_V8_CONTRACT}' AND "
        "verification_json ->> 'repair_generation' = "
        f"'{_V8_GENERATION}') IS TRUE"
    )


def _previous_evidence_constraint() -> str:
    return str(_previous_module()._v16_evidence_constraint())


def _v17_evidence_constraint() -> str:
    previous = _previous_evidence_constraint()
    if not previous.endswith(")"):
        raise RuntimeError("0084 evidence constraint is malformed")
    extended = previous[:-1] + " OR (" + _v8_single_member_evidence() + "))"
    return "(" + extended + ") IS TRUE"


def _recipe_ids_sql() -> str:
    return "(" + ",".join(f"'{value}'" for value in _TRANSPARENT_RECIPE_IDS) + ")"


def _v17_runtime_identity_constraint() -> str:
    return (
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        f"'{_V17_RECIPE}' AND "
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        f"{_recipe_ids_sql()} THEN ("
        "config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runner_sha256' = '{_V17_RUNNER_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        f"'{_V17_RUNTIME_BUNDLE_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_worker_runtime_image_digest' ~ "
        "'^sha256:[0-9a-f]{64}$'"
        ") ELSE true END) IS TRUE"
    )


def upgrade() -> None:
    op.drop_constraint(
        _REPAIR_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
        type_="check",
    )
    op.drop_constraint(
        _EVIDENCE_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        _REPAIR_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        _repair_constraint(),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        _EVIDENCE_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        _v17_evidence_constraint(),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        _RUNTIME_CONSTRAINT,
        "strategy_versions",
        _v17_runtime_identity_constraint(),
        schema=SCHEMA,
    )


def downgrade() -> None:
    v17_rows = op.get_bind().scalar(
        sa.text(
            "SELECT count(*) FROM quantlab.strategy_versions "
            "WHERE config_json ->> 'recipe_version' = :recipe "
            "AND config_json ->> 'recipe_id' IN "
            "('short_relative_strength','swing_trend','long_quality_value')"
        ),
        {"recipe": _V17_RECIPE},
    )
    v8_registry_rows = op.get_bind().scalar(
        sa.text(
            "SELECT count(*) "
            "FROM quantlab.transparent_baseline_pre_result_repairs "
            "WHERE verification_json ->> 'receipt_contract_version' = :contract "
            "AND verification_json ->> 'repair_generation' = :generation"
        ),
        {"contract": _V8_CONTRACT, "generation": _V8_GENERATION},
    )
    v8_audit_rows = op.get_bind().scalar(
        sa.text(
            "SELECT count(*) FROM quantlab.audit_events "
            "WHERE action = 'transparent_baseline_pre_result_repair_registered' "
            "AND details_json ->> 'receipt_sha256' = :receipt "
            "AND details_json ->> 'contract_version' = :contract "
            "AND details_json ->> 'repair_generation' = :generation"
        ),
        {
            "receipt": _V8_RECEIPT_SHA256,
            "contract": _V8_CONTRACT,
            "generation": _V8_GENERATION,
        },
    )
    if any(
        int(value or 0) > 0
        for value in (v17_rows, v8_registry_rows, v8_audit_rows)
    ):
        raise RuntimeError(
            "cannot downgrade 0085_baseline_v17_industry: immutable v17 strategy, "
            "v8 repair registry, or v8 audit receipt evidence exists"
        )
    op.drop_constraint(
        _RUNTIME_CONSTRAINT,
        "strategy_versions",
        schema=SCHEMA,
        type_="check",
    )
    op.drop_constraint(
        _REPAIR_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
        type_="check",
    )
    op.drop_constraint(
        _EVIDENCE_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        _REPAIR_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        _previous_repair_constraint(),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        _EVIDENCE_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        _previous_evidence_constraint(),
        schema=SCHEMA,
    )
