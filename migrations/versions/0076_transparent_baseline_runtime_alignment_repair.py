"""Allow the exact v9-to-v10 runtime-contract pre-result repair.

Revision ID: 0076_baseline_runtime_repair
Revises: 0075_baseline_lf_repair
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0076_baseline_runtime_repair"
down_revision: str | None = "0075_baseline_lf_repair"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
_V2_CONTRACT = "transparent-baseline-pre-result-repair-v2"
_V2_GENERATION = "v7-to-v8-optimizer-applicability"
_V2_TARGET_RECIPE = "qlib-rdagent-single-mainline-2026-08-30-v8"
_V2_TARGET_RUNNER_SHA256 = (
    "fa1090deaa66ca77a045c1a872f7b6451043e116e717954533217908135c584e"
)
_V2_SOURCE_BACKTEST_IDS = (
    "8090c21aa11546bd9d59f732975afc25",
    "9c8a75ac646f452e8a5666bacd708936",
    "f51d7fa2f4fd463e97fd5f6990b3721c",
)
_V3_CONTRACT = "transparent-baseline-pre-result-repair-v3"
_V3_GENERATION = "v8-to-v9-canonical-lf-packaging"
_V3_SOURCE_COMMIT = "413024ff5971115d4eb6a33872ebcafe9619cc9e"
_V3_TARGET_RECIPE = "qlib-rdagent-single-mainline-2026-08-30-v9"
_V3_SOURCE_EXPECTED_RUNNER_SHA256 = (
    "fa1090deaa66ca77a045c1a872f7b6451043e116e717954533217908135c584e"
)
_V3_SOURCE_OBSERVED_RUNNER_SHA256 = (
    "1ce281eb2e0141922215b9966f1ba91e09d073019a2e6e4b6a4614049b769e44"
)
_V3_TARGET_RUNNER_SHA256 = (
    "256bbfd579865e7bc1442f1f64003d655241abecb320e5d27b440dd635224d57"
)
_V3_PACKAGING_CONTRACT = "git-archive-canonical-lf-v1"
_V3_SOURCE_BACKTEST_IDS = (
    "0c5f5a0b66284a66ba68225afd22b623",
    "1ca979f22a0d4e2981e6e5c5e478f583",
    "7b4386b52cf24d6d913dae3b232fbe7f",
)
_V4_CONTRACT = "transparent-baseline-pre-result-repair-v4"
_V4_GENERATION = "v9-to-v10-runtime-contract-alignment"
_V4_SOURCE_COMMIT = "3e3bc56a2daf7e35206e94c8d36a4c9ffb2a9fd6"
_V4_TARGET_RECIPE = "qlib-rdagent-single-mainline-2026-08-30-v10"
_V4_SOURCE_RUNNER_SHA256 = (
    "256bbfd579865e7bc1442f1f64003d655241abecb320e5d27b440dd635224d57"
)
_V4_TARGET_RUNNER_SHA256 = (
    "0f868f2f3beaf5cff5db461f11c411ab3ef960c620c3f2f902ac385fc79eff4d"
)
_V4_SOURCE_RUNTIME_BUNDLE_SHA256 = (
    "b2b501cf1b59201b8732bacfa787ab38e41867993234aa2f9f605d9b163e7c68"
)
_V4_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "eea7ca8854acbed0370ead8d97fdfdb128059f41d8a5995a5050b7ec9e9f3a34"
)
_V4_RUNTIME_CONTRACT = "transparent-baseline-runtime-contract-alignment-v1"
_V4_SOURCE_ARTIFACT_INVENTORIES_SHA256 = (
    "6a988b654c9cd5ac9caca183d0efc83ce2f337f88b4fe33c8ac32c355b8fb86c"
)
_V4_SOURCE_BACKTEST_IDS = (
    "e05b237e364c4811a97ff5e2b49fc68c",
    "ef927367c58143448ebeaab2eceab5f1",
    "f795119754ea41c2960a710e2626bd19",
)


def _source_ids(values: Sequence[str]) -> str:
    return "[" + ",".join(f'"{value}"' for value in values) + "]"


def _same_lineage_generation(
    *,
    contract: str,
    generation: str,
    target_recipe: str,
    target_runner_sha256: str,
    source_backtest_ids: Sequence[str],
    extra: str = "",
) -> str:
    return (
        "("
        f"verification_json ->> 'receipt_contract_version' = '{contract}' AND "
        f"verification_json ->> 'repair_generation' = '{generation}' AND "
        f"verification_json ->> 'target_runner_sha256' = '{target_runner_sha256}' AND "
        f"target_recipe_version = '{target_recipe}' AND "
        f"source_backtest_ids_json = '{_source_ids(source_backtest_ids)}'::jsonb"
        f"{extra})"
    )


def _v2_generation() -> str:
    return _same_lineage_generation(
        contract=_V2_CONTRACT,
        generation=_V2_GENERATION,
        target_recipe=_V2_TARGET_RECIPE,
        target_runner_sha256=_V2_TARGET_RUNNER_SHA256,
        source_backtest_ids=_V2_SOURCE_BACKTEST_IDS,
    )


def _v3_generation() -> str:
    return _same_lineage_generation(
        contract=_V3_CONTRACT,
        generation=_V3_GENERATION,
        target_recipe=_V3_TARGET_RECIPE,
        target_runner_sha256=_V3_TARGET_RUNNER_SHA256,
        source_backtest_ids=_V3_SOURCE_BACKTEST_IDS,
        extra=(
            " AND verification_json ->> 'source_release_commit' = "
            f"'{_V3_SOURCE_COMMIT}'"
            " AND verification_json ->> 'source_runner_expected_sha256' = "
            f"'{_V3_SOURCE_EXPECTED_RUNNER_SHA256}'"
            " AND verification_json ->> 'source_runner_observed_sha256' = "
            f"'{_V3_SOURCE_OBSERVED_RUNNER_SHA256}'"
            " AND verification_json ->> 'packaging_contract_version' = "
            f"'{_V3_PACKAGING_CONTRACT}'"
        ),
    )


def _v4_generation() -> str:
    return _same_lineage_generation(
        contract=_V4_CONTRACT,
        generation=_V4_GENERATION,
        target_recipe=_V4_TARGET_RECIPE,
        target_runner_sha256=_V4_TARGET_RUNNER_SHA256,
        source_backtest_ids=_V4_SOURCE_BACKTEST_IDS,
        extra=(
            " AND verification_json ->> 'source_release_commit' = "
            f"'{_V4_SOURCE_COMMIT}'"
            " AND verification_json ->> 'source_runner_sha256' = "
            f"'{_V4_SOURCE_RUNNER_SHA256}'"
            " AND verification_json ->> 'source_runtime_bundle_sha256' = "
            f"'{_V4_SOURCE_RUNTIME_BUNDLE_SHA256}'"
            " AND verification_json ->> 'target_runtime_bundle_sha256' = "
            f"'{_V4_TARGET_RUNTIME_BUNDLE_SHA256}'"
            " AND verification_json ->> 'runtime_contract_version' = "
            f"'{_V4_RUNTIME_CONTRACT}'"
            " AND verification_json ->> 'source_artifact_inventories_sha256' = "
            f"'{_V4_SOURCE_ARTIFACT_INVENTORIES_SHA256}'"
        ),
    )


def _constraint(*generations: str) -> str:
    return (
        "source_batch_sha256 <> target_batch_sha256 AND ("
        "source_dataset_lineage_id <> target_dataset_lineage_id OR "
        + " OR ".join(generations)
        + ")"
    )


def upgrade() -> None:
    op.drop_constraint(
        "ck_transparent_baseline_repair_distinct_target",
        "transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        "ck_transparent_baseline_repair_distinct_target",
        "transparent_baseline_pre_result_repairs",
        _constraint(_v2_generation(), _v3_generation(), _v4_generation()),
        schema=SCHEMA,
    )


def downgrade() -> None:
    v4_rows = op.get_bind().scalar(
        sa.text(
            "SELECT count(*) FROM quantlab.transparent_baseline_pre_result_repairs "
            "WHERE verification_json ->> 'receipt_contract_version' = :contract "
            "AND verification_json ->> 'repair_generation' = :generation"
        ),
        {"contract": _V4_CONTRACT, "generation": _V4_GENERATION},
    )
    if int(v4_rows or 0) > 0:
        raise RuntimeError(
            "cannot downgrade 0076_baseline_runtime_repair: append-only v4 "
            "runtime-alignment repair evidence exists"
        )
    op.drop_constraint(
        "ck_transparent_baseline_repair_distinct_target",
        "transparent_baseline_pre_result_repairs",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        "ck_transparent_baseline_repair_distinct_target",
        "transparent_baseline_pre_result_repairs",
        _constraint(_v2_generation(), _v3_generation()),
        schema=SCHEMA,
    )
