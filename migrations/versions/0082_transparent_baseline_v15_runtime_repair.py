"""Seal v15 and allow one exact v13-to-v15 pre-result repair.

Revision ID: 0082_baseline_v15_repair
Revises: 0081_baseline_v14_seal

The v15 generation is append-only.  It keeps every v2-v5 same-lineage repair
authorization unchanged and adds one production-bound v6 member whose failed
v13 run produced no performance result.  All v14 identities remain immutable.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0082_baseline_v15_repair"
down_revision: str | None = "0081_baseline_v14_seal"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "quantlab"
_RUNTIME_CONSTRAINT = "ck_strategy_versions_v15_runtime_identity"
_REPAIR_CONSTRAINT = "ck_transparent_baseline_repair_distinct_target"

_V15_RECIPE = "qlib-rdagent-single-mainline-2026-08-30-v15"
_V15_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
_V15_RUNTIME_BUNDLE_SHA256 = (
    "e361d85d69d77cb6e0de7072db6f8aaff5d83f1e7902fe16ef06c2e28fce1867"
)
_TRANSPARENT_RECIPE_IDS = (
    "short_relative_strength",
    "swing_trend",
    "long_quality_value",
)

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
_V5_CONTRACT = "transparent-baseline-pre-result-repair-v5"
_V5_GENERATION = "v10-to-v11-runtime-input-scope"
_V5_SOURCE_COMMIT = "bb4d1139f848f0e39b82d13f7c4d58ff7659d8d2"
_V5_SOURCE_BATCH_SHA256 = (
    "9bc6a3017c2c718497dd53395da43c6d7392ceefdd0a268587c48d77813bcd65"
)
_V5_SOURCE_DATASET_IDENTITY_SHA256 = (
    "4771fc24680dcdca18fbcf73c887f604aad70d5f586c25116102f75d51d6e042"
)
_V5_SOURCE_DATASET_LINEAGE_ID = (
    "6e169632f9f322db93856f0e7ade3c9436b310e3509806f68d0b47db837e1b6d"
)
_V5_TARGET_RECIPE = "qlib-rdagent-single-mainline-2026-08-30-v11"
_V5_SOURCE_RUNNER_SHA256 = (
    "0f868f2f3beaf5cff5db461f11c411ab3ef960c620c3f2f902ac385fc79eff4d"
)
_V5_TARGET_RUNNER_SHA256 = (
    "64fa634b4e774279356c5655e70c741890ed9b208ccf75df757a378c0f56a432"
)
_V5_SOURCE_RUNTIME_BUNDLE_SHA256 = (
    "eea7ca8854acbed0370ead8d97fdfdb128059f41d8a5995a5050b7ec9e9f3a34"
)
_V5_TARGET_RUNTIME_BUNDLE_SHA256 = (
    "687ad83efd9d734a238bd6b52b3f5e670cec1e5d165ec2b25cabcef724feb7cf"
)
_V5_RUNTIME_CONTRACT = "transparent-baseline-runtime-input-scope-v1"
_V5_SOURCE_ARTIFACT_INVENTORIES_SHA256 = (
    "c2727672b33fed580841721551c45df1a11e864dd899cb6d473c220821da0dc0"
)
_V5_TARGET_CHANGE_CODES = (
    "missing-5d-extension-evidence-per-instrument-new-entry-rejection",
    "missing-trend-evidence-holding-continuity-new-entry-rejection",
    "shared-governed-style-exposure-snapshot-backtest-recommendation",
    "topk-benchmark-weight-non-consumption",
)
_V5_SOURCE_BACKTEST_IDS = (
    "00f1b9171d3d4ec3a04ade9ddec7d061",
    "04e84f759929477a9cae3de4ba1749fe",
    "4aa9972029d34793848f2c5a3e4ddb43",
)

_V6_CONTRACT = "transparent-baseline-pre-result-repair-v6"
_V6_GENERATION = "v13-to-v15-fill-aware-holding-age"
_V6_SOURCE_COMMIT = "ed5c8b3d0ea118dddbec7b1b5cf3c82b8cd72c08"
_V6_SOURCE_BATCH_SHA256 = (
    "34e96967cdb021913489ec95eebfd82eda68df7bdbee92ead837e467c3e7af96"
)
_V6_SOURCE_DATASET_IDENTITY_SHA256 = (
    "eab69dff43abcc77e60f47ad51d2182c7bfd5d90f5d36c479e3de61cebf768f2"
)
_V6_SOURCE_DATASET_LINEAGE_ID = (
    "1b97efcc3956b4be2dfebff7567efd4707c73d806f9d0cb2b63540d9dedd852e"
)
_V6_SOURCE_BACKTEST_ID = "51959d99d199410fa13a718658d03773"
_V6_SOURCE_JOB_ID = "4f4c1f51e74e4ed18cb6cb9f6ed757c7"
_V6_SOURCE_STRATEGY_VERSION_ID = "520b4c76f75445a88039a6f02d0d7dd1"
_V6_SOURCE_RUNNER_SHA256 = (
    "31d4c7a294ae61c19edcdb8e014b521c1b544836d6891af36cfba3ecf4ab43a3"
)
_V6_SOURCE_RUNTIME_BUNDLE_SHA256 = (
    "3000d84d183fe589da13d01902f88b1da1d402f47e18e4aafe91831cc59bdd8f"
)
_V6_TARGET_RECIPE = _V15_RECIPE
_V6_TARGET_RUNNER_SHA256 = _V15_RUNNER_SHA256
_V6_TARGET_RUNTIME_BUNDLE_SHA256 = _V15_RUNTIME_BUNDLE_SHA256
_V6_RUNTIME_CONTRACT = "transparent-baseline-fill-aware-holding-age-repair-v1"
_V6_SOURCE_ARTIFACT_INVENTORIES_SHA256 = (
    "7a83e76cde3fbc8a62583da41aef00613d4e15479a71b6e83a03b069d6005e01"
)
_V6_SOURCE_SELECTION_SHA256 = (
    "145161dbb19e8448995dcaaf781a9d433cfb0ce99ff9812367d5b1fbf1860076"
)
_V6_SOURCE_UNAVAILABLE_HORIZONS_SHA256 = (
    "b9f36421f0047ab024eec0f2b20909a105c2a57e17dc94a50aa25e632f8c3644"
)
_V6_UNAVAILABLE_EVIDENCE_SHA256S = (
    "62f5642a4ad024cc24d23ec34b9ce760a552f560c783612b63dab459ba00d14d",
    "b57d3dba4ab3d2284b2a001cb30a7b2862a8659c3738d5fc23419e7e7a6b5729",
)
_V6_TARGET_CHANGE_CODES = (
    "preserve-and-reconcile-holding-age-from-actual-holdings-after-"
    "unfilled-or-partial-exit",
)
_V6_RECEIPT_SHA256 = (
    "4cbd548de4cc38df791e4abeed59423c333769ffbc55f055291d457d4d332aeb"
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


def _v5_generation() -> str:
    return _same_lineage_generation(
        contract=_V5_CONTRACT,
        generation=_V5_GENERATION,
        target_recipe=_V5_TARGET_RECIPE,
        target_runner_sha256=_V5_TARGET_RUNNER_SHA256,
        source_backtest_ids=_V5_SOURCE_BACKTEST_IDS,
        extra=(
            " AND verification_json ->> 'source_release_commit' = "
            f"'{_V5_SOURCE_COMMIT}'"
            " AND verification_json ->> 'source_batch_sha256' = "
            f"'{_V5_SOURCE_BATCH_SHA256}'"
            " AND verification_json ->> 'source_dataset_identity_sha256' = "
            f"'{_V5_SOURCE_DATASET_IDENTITY_SHA256}'"
            " AND verification_json ->> 'source_dataset_lineage_id' = "
            f"'{_V5_SOURCE_DATASET_LINEAGE_ID}'"
            " AND verification_json ->> 'source_runner_sha256' = "
            f"'{_V5_SOURCE_RUNNER_SHA256}'"
            " AND verification_json ->> 'source_runtime_bundle_sha256' = "
            f"'{_V5_SOURCE_RUNTIME_BUNDLE_SHA256}'"
            " AND verification_json ->> 'target_runtime_bundle_sha256' = "
            f"'{_V5_TARGET_RUNTIME_BUNDLE_SHA256}'"
            " AND verification_json ->> 'runtime_contract_version' = "
            f"'{_V5_RUNTIME_CONTRACT}'"
            " AND verification_json ->> 'source_artifact_inventories_sha256' = "
            f"'{_V5_SOURCE_ARTIFACT_INVENTORIES_SHA256}'"
            " AND verification_json -> 'target_change_codes' = "
            f"'{_source_ids(_V5_TARGET_CHANGE_CODES)}'::jsonb"
        ),
    )


def _v6_generation() -> str:
    source_ids = _source_ids((_V6_SOURCE_BACKTEST_ID,))
    source_bindings = (
        "[{"
        f'"backtest_id":"{_V6_SOURCE_BACKTEST_ID}",'
        f'"job_id":"{_V6_SOURCE_JOB_ID}",'
        f'"strategy_version_id":"{_V6_SOURCE_STRATEGY_VERSION_ID}"'
        "}]"
    )
    return _same_lineage_generation(
        contract=_V6_CONTRACT,
        generation=_V6_GENERATION,
        target_recipe=_V6_TARGET_RECIPE,
        target_runner_sha256=_V6_TARGET_RUNNER_SHA256,
        source_backtest_ids=(_V6_SOURCE_BACKTEST_ID,),
        extra=(
            f" AND receipt_sha256 = '{_V6_RECEIPT_SHA256}'"
            " AND verification_json ->> 'receipt_sha256' = "
            f"'{_V6_RECEIPT_SHA256}'"
            " AND verification_json ->> 'source_release_commit' = "
            f"'{_V6_SOURCE_COMMIT}'"
            " AND verification_json ->> 'source_batch_sha256' = "
            f"'{_V6_SOURCE_BATCH_SHA256}'"
            " AND verification_json ->> 'source_dataset_identity_sha256' = "
            f"'{_V6_SOURCE_DATASET_IDENTITY_SHA256}'"
            " AND verification_json ->> 'source_dataset_lineage_id' = "
            f"'{_V6_SOURCE_DATASET_LINEAGE_ID}'"
            " AND source_dataset_lineage_id = "
            f"'{_V6_SOURCE_DATASET_LINEAGE_ID}'"
            " AND target_dataset_lineage_id = "
            f"'{_V6_SOURCE_DATASET_LINEAGE_ID}'"
            " AND verification_json ->> 'source_runner_sha256' = "
            f"'{_V6_SOURCE_RUNNER_SHA256}'"
            " AND verification_json ->> 'source_runtime_bundle_sha256' = "
            f"'{_V6_SOURCE_RUNTIME_BUNDLE_SHA256}'"
            " AND verification_json ->> 'target_runtime_bundle_sha256' = "
            f"'{_V6_TARGET_RUNTIME_BUNDLE_SHA256}'"
            " AND verification_json ->> 'runtime_contract_version' = "
            f"'{_V6_RUNTIME_CONTRACT}'"
            " AND verification_json ->> 'source_artifact_inventories_sha256' = "
            f"'{_V6_SOURCE_ARTIFACT_INVENTORIES_SHA256}'"
            " AND verification_json ->> "
            "'source_unopened_history_selection_sha256' = "
            f"'{_V6_SOURCE_SELECTION_SHA256}'"
            " AND verification_json ->> 'source_unavailable_horizons_sha256' = "
            f"'{_V6_SOURCE_UNAVAILABLE_HORIZONS_SHA256}'"
            " AND verification_json -> 'source_unavailable_evidence_sha256s' = "
            f"'{_source_ids(_V6_UNAVAILABLE_EVIDENCE_SHA256S)}'::jsonb"
            " AND verification_json -> 'target_change_codes' = "
            f"'{_source_ids(_V6_TARGET_CHANGE_CODES)}'::jsonb"
            " AND verification_json -> 'source_backtest_ids' = "
            f"'{source_ids}'::jsonb"
            " AND verification_json -> 'source_bindings' = "
            f"'{source_bindings}'::jsonb"
        ),
    )


def _constraint(*generations: str) -> str:
    return (
        "source_batch_sha256 <> target_batch_sha256 AND ("
        "source_dataset_lineage_id <> target_dataset_lineage_id OR "
        + " OR ".join(generations)
        + ")"
    )


def _repair_constraint() -> str:
    return _constraint(
        _v2_generation(),
        _v3_generation(),
        _v4_generation(),
        _v5_generation(),
        _v6_generation(),
    )


def _recipe_ids_sql() -> str:
    return "(" + ",".join(f"'{value}'" for value in _TRANSPARENT_RECIPE_IDS) + ")"


def _v15_runtime_identity_constraint() -> str:
    return (
        "(CASE WHEN "
        "COALESCE(config_json ->> 'recipe_version', '') = "
        f"'{_V15_RECIPE}' AND "
        "COALESCE(config_json ->> 'recipe_id', '') IN "
        f"{_recipe_ids_sql()} THEN ("
        "config_json -> 'transparent_baseline_bootstrap' ->> "
        f"'target_runner_sha256' = '{_V15_RUNNER_SHA256}' "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = "
        f"'{_V15_RUNTIME_BUNDLE_SHA256}' "
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
    op.create_check_constraint(
        _REPAIR_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        _repair_constraint(),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        _RUNTIME_CONSTRAINT,
        "strategy_versions",
        _v15_runtime_identity_constraint(),
        schema=SCHEMA,
    )


def downgrade() -> None:
    v15_rows = op.get_bind().scalar(
        sa.text(
            "SELECT count(*) FROM quantlab.strategy_versions "
            "WHERE config_json ->> 'recipe_version' = :recipe "
            "AND config_json ->> 'recipe_id' IN "
            "('short_relative_strength','swing_trend','long_quality_value')"
        ),
        {"recipe": _V15_RECIPE},
    )
    v6_rows = op.get_bind().scalar(
        sa.text(
            "SELECT count(*) "
            "FROM quantlab.transparent_baseline_pre_result_repairs "
            "WHERE verification_json ->> 'receipt_contract_version' = :contract "
            "AND verification_json ->> 'repair_generation' = :generation"
        ),
        {"contract": _V6_CONTRACT, "generation": _V6_GENERATION},
    )
    if int(v15_rows or 0) > 0 or int(v6_rows or 0) > 0:
        raise RuntimeError(
            "cannot downgrade 0082_baseline_v15_repair: immutable v15 strategy "
            "or v6 pre-result repair evidence exists"
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
    op.create_check_constraint(
        _REPAIR_CONSTRAINT,
        "transparent_baseline_pre_result_repairs",
        _constraint(
            _v2_generation(),
            _v3_generation(),
            _v4_generation(),
            _v5_generation(),
        ),
        schema=SCHEMA,
    )
